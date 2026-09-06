"""Train every classifier, score them on a held-out future month, and compare.

Pipeline:
  1. Read the corpus from DuckDB.
  2. Split on publication date, never at random.
  3. Fold segments too rare to learn or evaluate into "other".
  4. Fit and score the ladder: majority, keywords, TF-IDF + linear SVM, local LLM.
  5. Paired bootstrap between consecutive rungs.

Usage:
    python scripts/run_experiment.py
    python scripts/run_experiment.py --skip-llm      # classical models only
    python scripts/run_experiment.py --llm-sample 300
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb  # noqa: E402

from licitaciones.data import Tender  # noqa: E402
from licitaciones.evaluate import (  # noqa: E402
    bootstrap_metric,
    confusion_pairs,
    format_table,
    paired_bootstrap,
    score,
)
from licitaciones.models import (  # noqa: E402
    KeywordClassifier,
    LlmClassifier,
    MajorityClassifier,
    TfidfLinearClassifier,
    build_label_set,
    tender_text,
)
from licitaciones.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--llm-sample", type=int, default=None,
                   help="Score the LLM on a random subset of the test set")
    return p.parse_args()


def load_tenders(db_path: Path) -> list[Tender]:
    con = duckdb.connect(str(db_path), read_only=True)
    rows = con.execute("""
        SELECT ocid, tender_id, title, description, buyer, method, published,
               n_items, dominant_segment, unspsc_codes
        FROM tenders
        WHERE dominant_segment IS NOT NULL
          AND length(trim(title)) > 0
        ORDER BY published
    """).fetchall()
    con.close()
    return [
        Tender(ocid=r[0], tender_id=r[1], title=r[2], description=r[3], buyer=r[4],
               method=r[5], published=r[6], n_items=r[7],
               unspsc_codes=(r[9] or "").split(",") if r[9] else [])
        for r in rows
    ]


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    db_path = PROJECT_ROOT / config["paths"]["duckdb"]
    if not db_path.exists():
        log.error("No warehouse at %s. Run scripts/download_data.py first.", db_path)
        return 1

    tenders = load_tenders(db_path)
    log.info("Loaded %d labelled tenders", len(tenders))

    cutoff = date.fromisoformat(config["data"]["split_cutoff"])
    train, test = [], []
    for t in tenders:
        day = t.published[:10]
        if not day:
            continue
        (train if date.fromisoformat(day) < cutoff else test).append(t)
    log.info("Temporal split at %s: %d train, %d test", cutoff, len(train), len(test))
    if not train or not test:
        log.error("One side of the split is empty; check data.split_cutoff.")
        return 1

    # Fold rare segments. Mapping is built on TRAIN only: deciding what counts as
    # rare using the test set would be a subtle leak of test information.
    raw_train_labels = [t.dominant_segment for t in train]
    mapping = build_label_set(raw_train_labels, config["data"]["min_segment_count"])
    y_train = [mapping.get(t.dominant_segment, "other") for t in train]
    y_test = [mapping.get(t.dominant_segment, "other") for t in test]

    folded = sum(1 for lbl in y_train if lbl == "other")
    log.info("Label set: %d classes after folding (%d/%d train tenders became 'other')",
             len(set(y_train)), folded, len(y_train))

    majority_share = max(y_test.count(c) for c in set(y_test)) / len(y_test)
    log.info("Majority-class accuracy on test: %.4f  <- the number to beat", majority_share)

    models = [
        MajorityClassifier(),
        KeywordClassifier(),
        TfidfLinearClassifier(config["baselines"]["tfidf"],
                              config["baselines"]["linear_svc"], config["seed"]),
    ]
    if not args.skip_llm:
        models.append(LlmClassifier(config["llm"]))

    results: dict[str, dict] = {}
    predictions: dict[str, list[str]] = {}
    llm_index: list[int] | None = None

    for model in models:
        log.info("Fitting %s ...", model.name)
        t0 = time.perf_counter()
        model.fit(train, y_train)
        fit_seconds = time.perf_counter() - t0

        eval_tenders, eval_truth = test, y_test
        if model.name.startswith("llm") and args.llm_sample:
            import numpy as np
            rng = np.random.default_rng(config["seed"])
            llm_index = sorted(rng.choice(len(test), size=min(args.llm_sample, len(test)),
                                          replace=False).tolist())
            eval_tenders = [test[i] for i in llm_index]
            eval_truth = [y_test[i] for i in llm_index]
            log.info("Scoring the LLM on a %d-tender random subset", len(eval_tenders))

        t0 = time.perf_counter()
        preds = model.predict(eval_tenders)
        predict_seconds = time.perf_counter() - t0

        result = score(eval_truth, preds)
        result["fit_seconds"] = round(fit_seconds, 2)
        result["predict_seconds"] = round(predict_seconds, 2)
        result["ms_per_tender"] = round(1000 * predict_seconds / len(eval_tenders), 2)
        result["n_evaluated"] = len(eval_tenders)
        if model.cost_note:
            result["cost_note"] = model.cost_note
        if hasattr(model, "parse_failures"):
            result["unusable_answers"] = model.parse_failures

        results[model.name] = result
        predictions[model.name] = preds
        log.info("%-14s accuracy %.4f | macro-F1 %.4f | %.1f ms/tender",
                 model.name, result["accuracy"], result["macro_f1"],
                 result["ms_per_tender"])

    # --- confidence intervals and pairwise tests --------------------------
    n_boot = config["evaluation"]["n_bootstrap"]
    intervals = {
        name: {
            "accuracy": bootstrap_metric(
                y_test if len(predictions[name]) == len(y_test)
                else [y_test[i] for i in (llm_index or [])],
                predictions[name], "accuracy", n_boot, config["seed"]),
            "macro_f1": bootstrap_metric(
                y_test if len(predictions[name]) == len(y_test)
                else [y_test[i] for i in (llm_index or [])],
                predictions[name], "macro_f1", n_boot, config["seed"]),
        }
        for name in predictions
    }

    comparisons = {}
    ladder = [m.name for m in models]
    for a, b in zip(ladder[1:], ladder[:-1], strict=True):
        # The LLM may have been scored on a subset, so compare on the rows both
        # models actually saw rather than silently comparing different test sets.
        if len(predictions[a]) != len(predictions[b]):
            idx = llm_index or []
            truth = [y_test[i] for i in idx]
            def align(preds, index=idx):
                return preds if len(preds) == len(index) else [preds[i] for i in index]

            pa, pb = align(predictions[a]), align(predictions[b])
        else:
            truth, pa, pb = y_test, predictions[a], predictions[b]
        for metric in ("accuracy", "macro_f1"):
            comparisons[f"{a}_vs_{b}__{metric}"] = paired_bootstrap(
                truth, pa, pb, metric, n_boot, config["seed"])

    best = max(results, key=lambda k: results[k]["macro_f1"])
    errors = confusion_pairs(
        y_test if len(predictions[best]) == len(y_test)
        else [y_test[i] for i in (llm_index or [])],
        predictions[best])

    save_json(
        {
            "split": {
                "cutoff": str(cutoff), "n_train": len(train), "n_test": len(test),
                "n_classes": len(set(y_train)),
                "folded_into_other": folded,
                "majority_class_accuracy": round(majority_share, 4),
            },
            "results": results,
            "confidence_intervals": intervals,
            "comparisons": comparisons,
            "top_confusions_best_model": errors,
            "llm_subset_indices": llm_index,
        },
        "reports/metrics_experiment.json",
    )

    print()
    print(format_table(results, ladder))
    print()
    print(f"Majority-class accuracy: {majority_share:.4f}")
    print()
    print("| Comparison | delta | 95% CI | p | Significant |")
    print("|---|---:|:--|---:|:--:|")
    for key, c in comparisons.items():
        print(f"| {key.replace('__', ' / ').replace('_vs_', ' vs ')} | "
              f"{c['difference']:+.4f} | [{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}] | "
              f"{c['p_value']:.4f} | {'yes' if c['significant'] else 'NO'} |")
    print()
    print(f"Most frequent confusions for {best}:")
    for e in errors[:6]:
        print(f"  true {e['true']} -> predicted {e['predicted']}: {e['count']} "
              f"({e['share_of_true_class']:.0%} of that class)")
    _ = tender_text
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
