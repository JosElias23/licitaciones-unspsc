"""Crawl Chilean public tenders and store them in DuckDB.

The API serves one tender per request at roughly 0.11 s, so a few thousand take
minutes rather than hours. Every month is cached as JSON on the way in, so this
runs once and every later script works offline.

DuckDB rather than a pile of CSVs: the corpus is small enough to fit in memory
but the questions asked of it are relational (label distribution per month, per
buyer, per procurement method), and those are far clearer in SQL than in pandas
chains. It is also a single file with no server, so cloning the repository and
running one command reproduces the whole warehouse.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --limit 300      # a quick sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from licitaciones.data import corpus_stats, fetch_month, month_total  # noqa: E402
from licitaciones.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

SCHEMA = """
CREATE OR REPLACE TABLE tenders (
    ocid              VARCHAR,
    tender_id         VARCHAR,
    title             VARCHAR,
    description       VARCHAR,
    buyer             VARCHAR,
    method            VARCHAR,
    published         VARCHAR,
    published_date    DATE,
    n_items           INTEGER,
    n_codes           INTEGER,
    dominant_segment  VARCHAR,
    is_single_category BOOLEAN,
    unspsc_codes      VARCHAR
);
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--limit", type=int, default=None, help="Tenders per month")
    p.add_argument("--force", action="store_true", help="Ignore the cache")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    data_cfg = config["data"]
    limit = args.limit or data_cfg["tenders_per_month"]

    all_tenders = []
    per_month = {}
    for year, month in data_cfg["months"]:
        available = month_total(year, month)
        log.info("%d-%02d: %d tenders published, fetching %d", year, month, available, limit)
        tenders = fetch_month(year, month, limit=limit,
                              cache_dir=data_cfg["raw_dir"], force=args.force)
        log.info("%d-%02d: %d carry a UNSPSC label (%.0f%%)",
                 year, month, len(tenders), 100 * len(tenders) / max(limit, 1))
        per_month[f"{year}-{month:02d}"] = {
            "available": available, "requested": limit, "labelled": len(tenders),
        }
        all_tenders.extend(tenders)

    log.info("Total labelled tenders: %d", len(all_tenders))

    db_path = PROJECT_ROOT / config["paths"]["duckdb"]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(SCHEMA)
    con.executemany(
        "INSERT INTO tenders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (t.ocid, t.tender_id, t.title, t.description, t.buyer, t.method,
             t.published, t.published[:10] or None, t.n_items, len(t.unspsc_codes),
             t.dominant_segment, t.is_single_category, ",".join(t.unspsc_codes))
            for t in all_tenders
        ],
    )

    # Everything below is a question about the corpus that reads better in SQL
    # than in pandas, which is the reason the warehouse exists.
    log.info("Rows in DuckDB: %d", con.execute("SELECT count(*) FROM tenders").fetchone()[0])

    by_segment = con.execute("""
        SELECT dominant_segment AS segment,
               count(*)                              AS tenders,
               round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct,
               round(avg(n_items), 1)                AS avg_items,
               count(DISTINCT buyer)                 AS buyers
        FROM tenders
        WHERE dominant_segment IS NOT NULL
        GROUP BY 1 ORDER BY tenders DESC
    """).fetchall()

    by_month = con.execute("""
        SELECT strftime(published_date, '%Y-%m')  AS month,
               count(*)                            AS tenders,
               count(DISTINCT dominant_segment)    AS segments,
               round(100.0 * avg(CASE WHEN is_single_category THEN 1 ELSE 0 END), 1)
                                                   AS pct_single_category
        FROM tenders
        WHERE published_date IS NOT NULL
        GROUP BY 1 ORDER BY 1
    """).fetchall()

    # A buyer that dominates one segment would let the model learn the
    # institution rather than the category, so it is worth knowing about.
    concentration = con.execute("""
        WITH pairs AS (
            SELECT dominant_segment AS segment, buyer, count(*) AS n
            FROM tenders WHERE dominant_segment IS NOT NULL
            GROUP BY 1, 2
        ), totals AS (
            SELECT segment, sum(n) AS total FROM pairs GROUP BY 1
        )
        SELECT p.segment, p.buyer, p.n, t.total,
               round(100.0 * p.n / t.total, 1) AS pct_of_segment
        FROM pairs p JOIN totals t USING (segment)
        WHERE t.total >= 20
        ORDER BY pct_of_segment DESC LIMIT 5
    """).fetchall()

    stats = corpus_stats(all_tenders)
    save_json(
        {
            "per_month": per_month,
            "corpus": stats,
            "sql_by_segment": [
                {"segment": r[0], "tenders": r[1], "pct": r[2],
                 "avg_items": r[3], "distinct_buyers": r[4]} for r in by_segment
            ],
            "sql_by_month": [
                {"month": r[0], "tenders": r[1], "segments": r[2],
                 "pct_single_category": r[3]} for r in by_month
            ],
            "most_buyer_concentrated_segments": [
                {"segment": r[0], "buyer": r[1], "tenders": r[2],
                 "segment_total": r[3], "pct_of_segment": r[4]} for r in concentration
            ],
        },
        "reports/metrics_corpus.json",
    )
    con.close()

    print()
    print(f"Tenders: {stats['n_tenders']:,} | line items: {stats['n_line_items']:,} | "
          f"segments: {stats['distinct_segments']}")
    print(f"Majority segment {stats['majority_segment']} covers "
          f"{stats['majority_share']:.1%} of tenders (the accuracy to beat)")
    print(f"Single-category tenders: {stats['single_category_share']:.1%}")
    print(f"Mean title length: {stats['mean_title_words']} words")
    print()
    print("| Segment | Tenders | Share | Avg items | Distinct buyers |")
    print("|---|---:|---:|---:|---:|")
    for r in by_segment[:12]:
        print(f"| {r[0]} | {r[1]} | {r[2]}% | {r[3]} | {r[4]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
