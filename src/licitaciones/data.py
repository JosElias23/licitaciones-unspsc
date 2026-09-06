"""Chilean public procurement data: fetching, and the leakage guard.

Source
------
ChileCompra / Mercado Publico publishes every public tender through an
OCDS-compliant API at api.mercadopublico.cl. No key, no registration, and the
publisher declares the data **CC0** (public domain), which is the most permissive
licence a public dataset can carry.

Each tender carries two things that matter here:

  tender.title         free text written by the buying institution, e.g.
                       "ADQ. SERV. REPARACION VEH-. FISCAL BT-278 5TA COP"
  items[].classification.id
                       the official UNSPSC code assigned to each line item

That pairing is a labelled dataset produced as a by-product of government
administration, at a scale no one could annotate by hand.

Two traps, both found by inspection rather than by assumption
------------------------------------------------------------
**1. The API returns URLs that do not work.** The listing endpoint hands back
`urlTender` values beginning with `http://`, and those connections are closed
without a response. The same path over `https://` succeeds. Every URL is
rewritten before use; without that, a naive crawler fails 100% of requests.

**2. The item description IS the label.** Item descriptions are not free text at
all, they are the UNSPSC taxonomy path rendered in Spanish:

    code 47131803 -> "Equipos y suministros de limpieza / Suministros de
                      limpieza / Soluciones de limpieza y desinfeccion /
                      Desinfectantes domesticos"

Measured over a 444-item sample, the first field of the item description is a
single constant string for **40 of 40** UNSPSC segments, and its token overlap
with the label path is 1.000. Training on it would score near-perfectly and
would be reading the answer, not predicting it.

The buyer-written title overlaps the label path by 0.093, and 23% of titles
share no content word with it at all. That is the real task, and
`FORBIDDEN_FEATURES` exists so the mistake cannot be made by accident.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from licitaciones.utils import PROJECT_ROOT

API_ROOT = "https://api.mercadopublico.cl/APISOCDS/OCDS"
USER_AGENT = "licitaciones-unspsc/0.1 (github.com/JosElias23; research use)"

# Any field derived from the item classification leaks the label. Named here so
# the feature builder can assert against it rather than relying on discipline.
FORBIDDEN_FEATURES: frozenset[str] = frozenset({
    "item_description",     # literally the UNSPSC path in Spanish
    "item_classification",
    "unspsc",
    "unspsc_segment",
    "unspsc_family",
    "classification_scheme",
})


class DataError(RuntimeError):
    """Raised when the API returns something the pipeline cannot trust."""


class LeakageError(RuntimeError):
    """Raised when a feature derived from the label reaches the model."""


@dataclass
class Tender:
    """One tender: the buyer's own words, plus the categories it was filed under.

    `title` and `description` are written by the procuring institution and are
    the only text a model may see. `segments` comes from the official
    classification of the line items and is the target.
    """

    ocid: str
    tender_id: str
    title: str
    description: str
    buyer: str
    method: str
    published: str
    unspsc_codes: list[str] = field(default_factory=list)
    n_items: int = 0

    @property
    def segments(self) -> list[str]:
        """Two-digit UNSPSC segment for each line item."""
        return [c[:2] for c in self.unspsc_codes if len(c) >= 2]

    @property
    def dominant_segment(self) -> str | None:
        """The segment covering most line items, with ties broken deterministically.

        One label per tender, not one per item. A single food-supply tender can
        carry hundreds of line items, and counting per item let that one tender
        supply 62% of a 444-item sample. Per-tender labelling keeps one
        institution's paperwork from becoming the dataset.

        The tie-break is not cosmetic. The obvious implementation,

            max(set(segs), key=segs.count)

        picks an arbitrary winner whenever two segments are equally common,
        because `max` returns the first maximum it meets and a set of strings
        iterates in an order that depends on Python's per-process randomised
        string hashing. Two runs of the identical pipeline over the identical
        warehouse produced 32 and 33 label classes, and majority-class baselines
        of 0.1408 and 0.1261.

        Seeding does not help: `PYTHONHASHSEED` is read at interpreter start-up,
        so setting it from inside a running process affects only child processes
        and gives a false sense of determinism.

        Sorting by (descending count, ascending segment code) makes the choice a
        property of the data rather than of the process.
        """
        segs = self.segments
        if not segs:
            return None
        counts = Counter(segs)
        return min(counts, key=lambda s: (-counts[s], s))

    @property
    def is_single_category(self) -> bool:
        """True when every line item falls in the same segment.

        Mixed tenders are genuinely ambiguous: a single title cannot be expected
        to predict several categories. They are kept in the corpus and reported
        separately rather than silently dropped.
        """
        return len(set(self.segments)) == 1


def _request(url: str, timeout: int = 60, retries: int = 3) -> dict:
    """GET and parse JSON, upgrading http to https and retrying transient errors."""
    # The API advertises its own endpoints over http, and those connections are
    # closed without a response. See the module docstring.
    url = url.replace("http://", "https://", 1)

    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            time.sleep(0.5 * (2**attempt))  # back off; this is a public service
    raise DataError(f"{url} failed after {retries} attempts: {last}")


def list_tender_urls(year: int, month: int, limit: int, offset: int = 0) -> list[str]:
    """Tender detail URLs for one calendar month."""
    url = f"{API_ROOT}/listaOCDSAgnoMes/{year}/{month:02d}/{offset}/{limit}"
    payload = _request(url)
    if "data" not in payload:
        raise DataError(f"Listing for {year}-{month:02d} has no 'data' key")
    return [row["urlTender"] for row in payload["data"] if row.get("urlTender")]


def month_total(year: int, month: int) -> int:
    """How many tenders exist in a month, from the API's own pagination block."""
    payload = _request(f"{API_ROOT}/listaOCDSAgnoMes/{year}/{month:02d}/0/1")
    return int(payload["pagination"]["total"])


def parse_tender(payload: dict) -> Tender | None:
    """Turn one API response into a Tender, or None when it carries no label."""
    releases = payload.get("releases") or []
    if not releases:
        return None
    release = releases[0]
    tender = release.get("tender") or {}

    codes = []
    for item in tender.get("items") or []:
        classification = item.get("classification") or {}
        if classification.get("scheme") == "UNSPSC" and classification.get("id"):
            codes.append(str(classification["id"]))

    if not codes:
        return None

    return Tender(
        ocid=release.get("ocid", ""),
        tender_id=tender.get("id", ""),
        title=(tender.get("title") or "").strip(),
        description=(tender.get("description") or "").strip(),
        buyer=((tender.get("procuringEntity") or {}).get("name") or "").strip(),
        method=(tender.get("procurementMethodDetails") or "").strip(),
        published=release.get("date", ""),
        unspsc_codes=codes,
        n_items=len(tender.get("items") or []),
    )


def fetch_month(
    year: int,
    month: int,
    limit: int,
    offset: int = 0,
    cache_dir: str | Path = "data/raw",
    force: bool = False,
    progress_every: int = 250,
) -> list[Tender]:
    """Fetch one month of tenders, caching the parsed result on disk.

    Roughly 0.11 s per tender against the live API, so a few thousand take
    minutes. The cache means every later script runs offline.
    """
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"tenders_{year}_{month:02d}_{offset}_{limit}.json"

    if cache_file.exists() and not force:
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        return [Tender(**row) for row in raw]

    urls = list_tender_urls(year, month, limit, offset)
    tenders: list[Tender] = []
    failures = 0

    for i, url in enumerate(urls, start=1):
        try:
            parsed = parse_tender(_request(url))
            if parsed is not None:
                tenders.append(parsed)
        except DataError:
            failures += 1
        if progress_every and i % progress_every == 0:
            print(f"[data] {year}-{month:02d}: {i}/{len(urls)} fetched, "
                  f"{len(tenders)} labelled, {failures} failed", flush=True)

    if failures > len(urls) * 0.1:
        raise DataError(
            f"{failures}/{len(urls)} requests failed for {year}-{month:02d}. "
            "Refusing to cache a partial month: a silently short dataset would "
            "change every metric downstream."
        )

    cache_file.write_text(
        json.dumps([t.__dict__ for t in tenders], ensure_ascii=False),
        encoding="utf-8",
    )
    return tenders


def assert_no_leakage(feature_names) -> None:
    """Fail loudly if any feature is derived from the label.

    The item description is the UNSPSC taxonomy path written out in Spanish, so
    a model given it scores near-perfectly while having learned nothing. This is
    the single mistake that would invalidate the whole project, which is why it
    is a hard check rather than a comment.
    """
    offending = sorted(set(feature_names) & FORBIDDEN_FEATURES)
    if offending:
        raise LeakageError(
            f"These features are derived from the label: {offending}. "
            "The item description is the UNSPSC path in Spanish; training on it "
            "reads the answer instead of predicting it."
        )


def temporal_split(
    tenders: list[Tender], cutoff: date
) -> tuple[list[Tender], list[Tender]]:
    """Split by publication date, not at random.

    Procurement vocabulary drifts: budget lines are renamed, new programmes
    appear, institutions merge. A random split lets the model see next month's
    phrasing while predicting this month's, which is not the situation it would
    face in use. Splitting on time is the only split that matches deployment.
    """
    train, test = [], []
    for t in tenders:
        published = t.published[:10]
        if not published:
            continue
        (train if date.fromisoformat(published) < cutoff else test).append(t)
    return train, test


def corpus_stats(tenders: list[Tender]) -> dict:
    """Summary used by the report and by the tests."""
    segments = Counter(t.dominant_segment for t in tenders if t.dominant_segment)
    total = sum(segments.values())
    single = sum(1 for t in tenders if t.is_single_category)
    title_words = [len(t.title.split()) for t in tenders if t.title]

    return {
        "n_tenders": len(tenders),
        "n_labelled": total,
        "n_line_items": sum(t.n_items for t in tenders),
        "distinct_segments": len(segments),
        "majority_segment": segments.most_common(1)[0][0] if segments else None,
        "majority_share": round(segments.most_common(1)[0][1] / total, 4) if total else 0.0,
        "single_category_share": round(single / len(tenders), 4) if tenders else 0.0,
        "mean_title_words": round(sum(title_words) / len(title_words), 1) if title_words else 0.0,
        "segments_by_size": dict(segments.most_common()),
    }
