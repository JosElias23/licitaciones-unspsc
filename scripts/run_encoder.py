"""Fine-tune BETO and compare it against the best model so far.

The README has called a fine-tuned Spanish encoder the honest favourite for this
task since the first commit, without running it. This runs it, on the same
split, the same text and the same metric, and reports what it actually costs.

It also produces the coverage/accuracy curve, because the deployable question is
not "how accurate is it" but "what share of tenders can be auto-categorised at
an error rate a procurement desk would accept".

Usage:
    python scripts/run_encoder.py
    python scripts/run_encoder.py --epochs 3
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_experiment import load_tenders  # noqa: E402

from licitaciones.encoder import (  # noqa: E402
    EncoderClassifier,
    confidence_discrimination,
    coverage_accuracy_curve,
)
from licitaciones.evaluate import (  # noqa: E402
    bootstrap_metric,
    confusion_pairs,
    paired_bootstrap,
    score,
)
from licitaciones.models import TfidfLinearClassifier, build_label_set  # noqa: E402
from licitaciones.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    log = setup_logging()
    config = load_config()
    set_seed(config["seed"])

    db_path = PROJECT_ROOT / config["paths"]["duckdb"]
    if not db_path.exists():
        log.error("No warehouse. Run scripts/download_data.py first.")
        return 1

    tenders = load_tenders(db_path)
    cutoff = date.fromisoformat(config["data"]["split_cutoff"])
    train = [t for t in tenders if t.published[:10]
             and date.fromisoformat(t.published[:10]) < cutoff]
    test = [t for t in tenders if t.published[:10]
            and date.fromisoformat(t.published[:10]) >= cutoff]

    mapping = build_label_set([t.dominant_segment for t in train],
                              config["data"]["min_segment_count"])
    y_train = [mapping.get(t.dominant_segment, "other") for t in train]
    y_test = [mapping.get(t.dominant_segment, "other") for t in test]
    log.info("Train %d | test %d | %d classes", len(train), len(test), len(set(y_train)))

    encoder_cfg = config.get("encoder", {})
    if args.epochs:
        encoder_cfg = {**encoder_cfg, "num_epochs": args.epochs}
    if args.batch_size:
        encoder_cfg = {**encoder_cfg, "batch_size": args.batch_size}

    results, predictions = {}, {}

    # The incumbent, refitted here so both models are scored in one process on
    # identical data.
    baseline = TfidfLinearClassifier(config["baselines"]["tfidf"],
                                     config["baselines"]["linear_svc"], config["seed"])
    baseline.fit(train, y_train)
    t0 = time.perf_counter()
    predictions["tfidf_svm"] = baseline.predict(test)
    results["tfidf_svm"] = score(y_test, predictions["tfidf_svm"])
    results["tfidf_svm"]["ms_per_tender"] = round(
        1000 * (time.perf_counter() - t0) / len(test), 3)
    log.info("tfidf_svm     accuracy %.4f | macro-F1 %.4f",
             results["tfidf_svm"]["accuracy"], results["tfidf_svm"]["macro_f1"])

    log.info("Fine-tuning BETO ...")
    encoder = EncoderClassifier(encoder_cfg, seed=config["seed"])
    encoder.fit(train, y_train)

    t0 = time.perf_counter()
    preds, confidence = encoder.predict_with_confidence(test)
    predict_seconds = time.perf_counter() - t0

    predictions[encoder.name] = preds
    results[encoder.name] = score(y_test, preds)
    results[encoder.name]["ms_per_tender"] = round(1000 * predict_seconds / len(test), 3)
    results[encoder.name]["train_seconds"] = round(encoder.train_seconds, 1)
    results[encoder.name]["cost_note"] = encoder.cost_note
    log.info("%-13s accuracy %.4f | macro-F1 %.4f | %s", encoder.name,
             results[encoder.name]["accuracy"], results[encoder.name]["macro_f1"],
             encoder.cost_note)

    n_boot = config["evaluation"]["n_bootstrap"]
    intervals = {
        name: {
            metric: bootstrap_metric(y_test, preds, metric, n_boot, config["seed"])
            for metric in ("accuracy", "macro_f1")
        }
        for name, preds in predictions.items()
    }
    comparison = {
        metric: paired_bootstrap(y_test, predictions[encoder.name],
                                 predictions["tfidf_svm"], metric, n_boot,
                                 config["seed"])
        for metric in ("accuracy", "macro_f1")
    }

    curve = coverage_accuracy_curve(y_test, preds, confidence)
    # The curve shows the operating points; this says whether the confidence
    # ranking them is real. A monotone curve is nearly automatic and was
    # previously offered as if it were evidence.
    discrimination = confidence_discrimination(y_test, preds, confidence,
                                               seed=config["seed"])
    errors = confusion_pairs(y_test, preds)

    save_json(
        {
            "n_train": len(train), "n_test": len(test), "n_classes": len(set(y_train)),
            "results": results,
            "confidence_intervals": intervals,
            "encoder_vs_tfidf": comparison,
            "coverage_accuracy_curve": curve,
            "confidence_discrimination": discrimination,
            "top_confusions": errors,
            "predictions": preds,
            "confidence": [round(float(c), 6) for c in confidence],
            "y_true": y_test,
        },
        "reports/metrics_encoder.json",
    )

    print()
    print("| Model | Accuracy | 95% CI | Macro-F1 | ms/tender |")
    print("|---|---:|:--|---:|---:|")
    for name, r in results.items():
        ci = intervals[name]["accuracy"]
        print(f"| {name} | {r['accuracy']:.4f} | [{ci['ci_lower']:.4f}, "
              f"{ci['ci_upper']:.4f}] | {r['macro_f1']:.4f} | {r['ms_per_tender']:.2f} |")
    print()
    for metric, c in comparison.items():
        print(f"{encoder.name} vs tfidf_svm / {metric}: {c['difference']:+.4f} "
              f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}], p={c['p_value']:.4f}, "
              f"{'significant' if c['significant'] else 'NOT significant'}")
    print()
    if discrimination.get("auroc") is not None:
        print(f"confidence vs correctness: AUROC {discrimination['auroc']:.4f}, "
              f"permutation p = {discrimination['permutation_p']:.4f} "
              f"({discrimination['n_permutations']} shuffles)")
        print()
    print("| Confidence >= | Coverage | Tenders kept | Accuracy on kept |")
    print("|---:|---:|---:|---:|")
    for row in curve:
        acc = f"{row['accuracy_on_kept']:.4f}" if row["accuracy_on_kept"] is not None else "-"
        print(f"| {row['threshold']:.2f} | {row['coverage']:.1%} | {row['n_kept']} | {acc} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
