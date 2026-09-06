"""Ask whether examples close the gap between the LLM and the linear model.

The zero-shot LLM lost by 32 accuracy points. The obvious objection is that it
was given no examples while the linear model had 2,874, so this script supplies
them three ways and measures each on the identical test subset:

  llm_zero_shot        no examples (the published baseline, re-run here)
  llm_fewshot_random   8 random training examples, fixed for every tender
  llm_fewshot_knn      the 8 most similar training titles, retrieved per tender
  llm_dspy_bootstrap   demonstrations chosen by DSPy from the model's own hits

The random arm is not filler. Without it, any improvement from retrieval could
equally be explained by the model finally seeing the output format, and those
are different findings.

Usage:
    python scripts/run_llm_variants.py --sample 200
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from licitaciones.evaluate import bootstrap_metric, paired_bootstrap, score  # noqa: E402
from licitaciones.llm_variants import (  # noqa: E402
    DspyClassifier,
    FewShotClassifier,
    RandomFewShotClassifier,
)
from licitaciones.models import (  # noqa: E402
    LlmClassifier,
    TfidfLinearClassifier,
    build_label_set,
)
from licitaciones.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_experiment import load_tenders  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=int, default=200,
                   help="Test tenders to score every variant on")
    p.add_argument("--k", type=int, default=8, help="Examples per prompt")
    p.add_argument("--skip-dspy", action="store_true")
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
    train = [t for t in tenders if t.published[:10] and
             date.fromisoformat(t.published[:10]) < cutoff]
    test = [t for t in tenders if t.published[:10] and
            date.fromisoformat(t.published[:10]) >= cutoff]

    mapping = build_label_set([t.dominant_segment for t in train],
                              config["data"]["min_segment_count"])
    y_train = [mapping.get(t.dominant_segment, "other") for t in train]
    y_test_all = [mapping.get(t.dominant_segment, "other") for t in test]

    # One subset, seeded, shared by every variant. Comparing arms scored on
    # different rows would make the paired test meaningless.
    rng = np.random.default_rng(config["seed"])
    idx = sorted(rng.choice(len(test), size=min(args.sample, len(test)),
                            replace=False).tolist())
    subset = [test[i] for i in idx]
    y_test = [y_test_all[i] for i in idx]
    log.info("Train %d | test subset %d | %d classes",
             len(train), len(subset), len(set(y_train)))

    llm_cfg = config["llm"]
    models = [
        TfidfLinearClassifier(config["baselines"]["tfidf"],
                              config["baselines"]["linear_svc"], config["seed"]),
        LlmClassifier(llm_cfg),
        RandomFewShotClassifier(llm_cfg, k=args.k, seed=config["seed"]),
        FewShotClassifier(llm_cfg, k=args.k),
    ]
    if not args.skip_dspy:
        models.append(DspyClassifier(llm_cfg, max_demos=args.k, seed=config["seed"]))

    results, predictions = {}, {}
    for model in models:
        log.info("Fitting %s ...", model.name)
        t0 = time.perf_counter()
        model.fit(train, y_train)
        fit_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        preds = model.predict(subset)
        predict_seconds = time.perf_counter() - t0

        result = score(y_test, preds)
        result["fit_seconds"] = round(fit_seconds, 1)
        result["ms_per_tender"] = round(1000 * predict_seconds / len(subset), 2)
        if model.cost_note:
            result["cost_note"] = model.cost_note
        if hasattr(model, "client"):
            result["llm_stats"] = model.client.stats()

        results[model.name] = result
        predictions[model.name] = preds
        log.info("%-20s accuracy %.4f | macro-F1 %.4f | %.0f ms/tender",
                 model.name, result["accuracy"], result["macro_f1"],
                 result["ms_per_tender"])

    n_boot = config["evaluation"]["n_bootstrap"]
    intervals = {
        name: bootstrap_metric(y_test, preds, "accuracy", n_boot, config["seed"])
        for name, preds in predictions.items()
    }

    # The three comparisons the experiment exists to make.
    pairs = [
        ("llm_fewshot_random", "llm_zero_shot"),      # do examples help at all?
        ("llm_fewshot_knn", "llm_fewshot_random"),    # does relevance help?
        ("llm_fewshot_knn", "tfidf_svm"),             # is the gap closed?
    ]
    if not args.skip_dspy:
        pairs.append(("llm_dspy_bootstrap", "llm_fewshot_knn"))

    comparisons = {}
    for a, b in pairs:
        if a in predictions and b in predictions:
            comparisons[f"{a}_vs_{b}"] = paired_bootstrap(
                y_test, predictions[a], predictions[b], "accuracy",
                n_boot, config["seed"])

    save_json(
        {
            "n_test_subset": len(subset), "k_examples": args.k,
            "test_indices": idx,
            "results": results,
            "confidence_intervals": intervals,
            "comparisons": comparisons,
        },
        "reports/metrics_llm_variants.json",
    )

    print()
    print("| Variant | Accuracy | 95% CI | Macro-F1 | ms/tender |")
    print("|---|---:|:--|---:|---:|")
    for name, r in results.items():
        ci = intervals[name]
        print(f"| {name} | {r['accuracy']:.4f} | "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] | "
              f"{r['macro_f1']:.4f} | {r['ms_per_tender']:.0f} |")
    print()
    print("| Comparison | delta | 95% CI | p | Significant |")
    print("|---|---:|:--|---:|:--:|")
    for key, c in comparisons.items():
        print(f"| {key.replace('_vs_', ' vs ')} | {c['difference']:+.4f} | "
              f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}] | {c['p_value']:.4f} | "
              f"{'yes' if c['significant'] else 'NO'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
