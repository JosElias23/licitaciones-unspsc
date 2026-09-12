"""Choose the encoder's epoch count on a validation month, not on the test month.

This script exists because of a defect in how section 6.1 of the write-up was
produced. The epoch table there -- 6, 15 and 30 epochs, with 15 winning -- was
built by training on January to March and reading accuracy off **April**, which
is the held-out month every headline number in this repository is reported on.
Nothing in the project ever held out a validation set.

That makes the published comparison optimistic in a specific, mechanical way.
Three candidates were scored on the test set and the best was kept, so the
reported accuracy is the maximum of three draws rather than one draw, and the
p-value of 0.186 against TF-IDF is not the p-value of the procedure that
produced it. The direction is not subtle either: at 6 epochs BETO loses
significantly and at 30 it loses significantly, so "draws level" is exactly the
conclusion selection would manufacture if the three points were noise.

The fix is the ordinary one. The training window has three months in it, so it
can be split the same way the outer split is:

    inner train   January + February 2026      1,922 tenders
    validation    March 2026                     952 tenders
    test          April 2026                     952 tenders   -- untouched here

Sweep the epoch grid on the inner split, pick the winner on **March**, then
refit on the full January-March training set at that epoch count and score
April exactly once. April is read at the end and never before.

Three seeds per grid point, because choosing a hyper-parameter off a single
noisy run is the same mistake in a smaller costume: if the seed-to-seed spread
at a fixed epoch count is comparable to the spread across epoch counts, then the
grid is not measuring what it claims to.

What it found, in one line: **the two months disagree about where the optimum
is.** March's accuracy rises monotonically across the whole grid and peaks at 30
epochs; April's peaks at 15 and falls on either side. The inverted U the
write-up described is a property of the month it was read from, not of the
model. Selecting on March gives 30 epochs, and at 30 epochs BETO loses to the
linear model by 3.57 accuracy points (p = 0.0196) rather than drawing level.

The robust statement, and the one that survives every choice in this script, is
simpler: BETO does not beat TF-IDF at any point in the grid. Its best showing is
-0.0200 and its worst is -0.1134. Selection decided whether the loss was
reported as significant, not whether there was one.

Usage:
    python scripts/select_epochs.py
    python scripts/select_epochs.py --grid 6,15,30 --seeds 42

Writes reports/metrics_epoch_selection.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_experiment import load_tenders  # noqa: E402

from licitaciones.encoder import EncoderClassifier  # noqa: E402
from licitaciones.evaluate import bootstrap_metric, paired_bootstrap, score  # noqa: E402
from licitaciones.models import TfidfLinearClassifier, build_label_set  # noqa: E402
from licitaciones.utils import (  # noqa: E402
    PROJECT_ROOT,
    load_config,
    save_json,
    set_seed,
    setup_logging,
)

# The inner cutoff. The outer one lives in configs/default.yaml; this one splits
# the training window and is only ever used to choose a hyper-parameter.
INNER_CUTOFF = date(2026, 3, 1)

DEFAULT_GRID = (3, 6, 10, 15, 20, 30)
DEFAULT_SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=str, default=",".join(str(e) for e in DEFAULT_GRID),
                   help="comma-separated epoch counts to try")
    p.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS),
                   help="comma-separated seeds, averaged at each grid point")
    p.add_argument("--reuse-selection", action="store_true",
                   help="read the selection sweep back from "
                        "reports/metrics_epoch_selection.json instead of "
                        "re-training it; the stored runs are the same runs")
    p.add_argument("--sensitivity", action="store_true",
                   help="also score the test month at every grid point, as a "
                        "disclosed sensitivity analysis rather than a selection "
                        "rule")
    return p.parse_args()


def split_on(tenders, cutoff: date):
    before, after = [], []
    for t in tenders:
        if not t.published[:10]:
            continue
        (before if date.fromisoformat(t.published[:10]) < cutoff else after).append(t)
    return before, after


def train_and_score(train, y_train, evaluate, y_eval, encoder_cfg, epochs, seed):
    """One fine-tune, one evaluation. Returns (metrics, seconds)."""
    cfg = {**encoder_cfg, "num_epochs": epochs}
    set_seed(seed)
    model = EncoderClassifier(cfg, seed=seed)
    started = time.perf_counter()
    model.fit(train, y_train)
    preds = model.predict(evaluate)
    elapsed = time.perf_counter() - started
    return score(y_eval, preds), elapsed, preds


def main() -> int:
    args = parse_args()
    grid = [int(x) for x in args.grid.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    log = setup_logging()
    config = load_config()
    encoder_cfg = config.get("encoder", {})
    min_count = config["data"]["min_segment_count"]

    db_path = PROJECT_ROOT / config["paths"]["duckdb"]
    if not db_path.exists():
        log.error("No warehouse. Run scripts/download_data.py first.")
        return 1

    tenders = load_tenders(db_path)
    outer_cutoff = date.fromisoformat(config["data"]["split_cutoff"])
    train_full, test = split_on(tenders, outer_cutoff)
    inner_train, validation = split_on(train_full, INNER_CUTOFF)

    log.info("inner train %d | validation %d | test %d (untouched until the end)",
             len(inner_train), len(validation), len(test))

    # Labels for the selection stage come from the inner training set only. Using
    # the full training set's label vocabulary here would let March influence
    # which classes exist while March is being used to score.
    inner_map = build_label_set([t.dominant_segment for t in inner_train], min_count)
    y_inner = [inner_map.get(t.dominant_segment, "other") for t in inner_train]
    y_val = [inner_map.get(t.dominant_segment, "other") for t in validation]

    # ---------------------------------------------------------------- selection
    selection = []
    stored = PROJECT_ROOT / "reports" / "metrics_epoch_selection.json"
    if args.reuse_selection:
        if not stored.exists():
            log.error("--reuse-selection needs %s; run the sweep first", stored)
            return 1
        previous = json.loads(stored.read_text(encoding="utf-8"))
        selection = previous["selection_on_validation"]
        grid = [row["epochs"] for row in selection]
        seeds = previous["protocol"]["seeds"]
        log.info("reusing the stored sweep: grid %s, seeds %s", grid, seeds)
    for epochs in (grid if not args.reuse_selection else []):
        runs = []
        for seed in seeds:
            metrics, elapsed, _ = train_and_score(
                inner_train, y_inner, validation, y_val, encoder_cfg, epochs, seed
            )
            runs.append({"seed": seed, **{k: metrics[k] for k in ("accuracy", "macro_f1")},
                         "train_predict_seconds": round(elapsed, 1)})
            log.info("epochs %2d seed %d -> validation accuracy %.4f (%.0fs)",
                     epochs, seed, metrics["accuracy"], elapsed)
        accs = [r["accuracy"] for r in runs]
        f1s = [r["macro_f1"] for r in runs]
        selection.append({
            "epochs": epochs,
            "runs": runs,
            "mean_val_accuracy": round(float(np.mean(accs)), 4),
            "sd_val_accuracy": round(float(np.std(accs, ddof=1)), 4) if len(accs) > 1 else None,
            "mean_val_macro_f1": round(float(np.mean(f1s)), 4),
        })

    best = max(selection, key=lambda r: r["mean_val_accuracy"])
    chosen = best["epochs"]
    log.info("validation picks %d epochs (mean accuracy %.4f)",
             chosen, best["mean_val_accuracy"])

    # How much of the spread across the grid is just seed noise? If the within-
    # grid-point sd is as large as the between-grid-point sd, the curve is not
    # measuring epoch count.
    between = float(np.std([r["mean_val_accuracy"] for r in selection], ddof=1))
    withins = [r["sd_val_accuracy"] for r in selection if r["sd_val_accuracy"] is not None]
    within = float(np.mean(withins)) if withins else None

    # -------------------------------------------------------------------- test
    # April is read from here on, once, at the epoch count March chose.
    full_map = build_label_set([t.dominant_segment for t in train_full], min_count)
    y_train_full = [full_map.get(t.dominant_segment, "other") for t in train_full]
    y_test = [full_map.get(t.dominant_segment, "other") for t in test]

    baseline = TfidfLinearClassifier(config["baselines"]["tfidf"],
                                     config["baselines"]["linear_svc"], config["seed"])
    baseline.fit(train_full, y_train_full)
    tfidf_preds = baseline.predict(test)
    tfidf_metrics = score(y_test, tfidf_preds)

    encoder_metrics, encoder_seconds, encoder_preds = train_and_score(
        train_full, y_train_full, test, y_test, encoder_cfg, chosen, config["seed"]
    )
    log.info("test at %d epochs: accuracy %.4f | macro-F1 %.4f",
             chosen, encoder_metrics["accuracy"], encoder_metrics["macro_f1"])

    n_boot = config["evaluation"]["n_bootstrap"]
    comparison = {
        metric: paired_bootstrap(y_test, encoder_preds, tfidf_preds, metric,
                                 n_boot, config["seed"])
        for metric in ("accuracy", "macro_f1")
    }
    intervals = {
        "beto_finetuned": {m: bootstrap_metric(y_test, encoder_preds, m, n_boot,
                                               config["seed"])
                           for m in ("accuracy", "macro_f1")},
        "tfidf_svm": {m: bootstrap_metric(y_test, tfidf_preds, m, n_boot,
                                          config["seed"])
                      for m in ("accuracy", "macro_f1")},
    }

    # ------------------------------------------------------------ sensitivity
    # Scoring April at every grid point, AFTER the choice is locked, is a
    # legitimate and useful thing to publish: it shows how much the conclusion
    # depends on a knob validation cannot turn. It is also exactly what the
    # first version of this write-up did -- with the crucial difference that it
    # then kept the best one. Selection happened above and is not revisited
    # here.
    sensitivity = None
    if args.sensitivity:
        sensitivity = []
        for epochs in grid:
            metrics, _, preds = train_and_score(
                train_full, y_train_full, test, y_test, encoder_cfg, epochs,
                config["seed"]
            )
            paired = paired_bootstrap(y_test, preds, tfidf_preds, "accuracy",
                                      n_boot, config["seed"])
            sensitivity.append({
                "epochs": epochs,
                "test_accuracy": metrics["accuracy"],
                "test_macro_f1": metrics["macro_f1"],
                "vs_tfidf": paired["difference"],
                "significant": paired["significant"],
                "p_value": paired["p_value"],
            })
            log.info("sensitivity: %2d epochs -> test %.4f (vs tfidf %+.4f, %s)",
                     epochs, metrics["accuracy"], paired["difference"],
                     "significant" if paired["significant"] else "not significant")

    save_json(
        {
            "why": ("The published epoch count was chosen on the test month. This "
                    "re-runs the choice on a validation month carved out of the "
                    "training window, and reports what the test set says "
                    "afterwards."),
            "protocol": {
                "inner_cutoff": INNER_CUTOFF.isoformat(),
                "outer_cutoff": config["data"]["split_cutoff"],
                "n_inner_train": len(inner_train),
                "n_validation": len(validation),
                "n_test": len(test),
                "n_train_full": len(train_full),
                "grid": grid,
                "seeds": seeds,
                "n_classes_inner": len(set(y_inner)),
                "n_classes_full": len(set(y_train_full)),
            },
            "selection_on_validation": selection,
            "chosen_epochs": chosen,
            "chosen_mean_val_accuracy": best["mean_val_accuracy"],
            "noise_check": {
                "sd_between_grid_points": round(between, 4),
                "mean_sd_within_grid_point_across_seeds": (
                    round(within, 4) if within is not None else None),
                "grid_separates_from_seed_noise": (
                    bool(within is not None and between > within)),
                "note": ("If the seed-to-seed spread at a fixed epoch count is as "
                         "large as the spread across epoch counts, the grid is "
                         "measuring noise and the choice of epoch count is "
                         "arbitrary."),
            },
            "test_at_chosen_epochs": {
                "beto_finetuned": {
                    **{k: encoder_metrics[k] for k in ("accuracy", "macro_f1")},
                    "seconds": round(encoder_seconds, 1),
                },
                "tfidf_svm": {k: tfidf_metrics[k] for k in ("accuracy", "macro_f1")},
                "confidence_intervals": intervals,
                "encoder_vs_tfidf": comparison,
            },
            "test_sensitivity_disclosed_not_used_for_selection": sensitivity,
        },
        "reports/metrics_epoch_selection.json",
    )

    # ------------------------------------------------------------------ console
    print()
    print(f"Selection on March {len(validation)} tenders "
          f"(trained on {len(inner_train)}), {len(seeds)} seeds per point")
    print("| Epochs | Mean val accuracy | sd across seeds | Mean val macro-F1 |")
    print("|---:|---:|---:|---:|")
    for row in selection:
        sd = f"{row['sd_val_accuracy']:.4f}" if row["sd_val_accuracy"] is not None else "-"
        mark = " **" if row["epochs"] == chosen else ""
        print(f"| {row['epochs']}{mark} | {row['mean_val_accuracy']:.4f} | {sd} | "
              f"{row['mean_val_macro_f1']:.4f} |")
    print()
    print(f"between-grid-point sd {between:.4f} vs within-point seed sd "
          f"{within:.4f}" if within is not None else "")
    print()
    print(f"Chosen: {chosen} epochs. Refit on all {len(train_full)} training "
          f"tenders, scored once on April.")
    print("| Model | Accuracy | 95% CI | Macro-F1 |")
    print("|---|---:|:--|---:|")
    for name, m, ci in (("tfidf_svm", tfidf_metrics, intervals["tfidf_svm"]),
                        ("beto_finetuned", encoder_metrics, intervals["beto_finetuned"])):
        c = ci["accuracy"]
        print(f"| {name} | {m['accuracy']:.4f} | [{c['ci_lower']:.4f}, "
              f"{c['ci_upper']:.4f}] | {m['macro_f1']:.4f} |")
    print()
    for metric, c in comparison.items():
        print(f"beto vs tfidf / {metric}: {c['difference']:+.4f} "
              f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}], p={c['p_value']:.4f}, "
              f"{'significant' if c['significant'] else 'NOT significant'}")

    if sensitivity:
        print()
        print("Sensitivity: what April would have said at each grid point. "
              "Disclosed, not used to choose.")
        print("| Epochs | Test accuracy | vs TF-IDF | p | Significant |")
        print("|---:|---:|---:|---:|:--:|")
        for row in sensitivity:
            print(f"| {row['epochs']} | {row['test_accuracy']:.4f} | "
                  f"{row['vs_tfidf']:+.4f} | {row['p_value']:.4f} | "
                  f"{'yes' if row['significant'] else 'NO'} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
