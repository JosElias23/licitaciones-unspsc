"""Scoring, and the statistics needed to compare two classifiers honestly.

Three choices shape everything reported:

**Macro-F1 alongside accuracy.** The label distribution is heavily skewed: the
largest segment covers roughly a fifth of tenders and the tail runs to forty-plus
classes. Accuracy rewards a model that serves the head and ignores the tail,
which is precisely the failure mode that matters to a buyer searching for an
unusual category. Both are reported, and they disagree.

**Paired bootstrap, not two independent intervals.** Comparing two models by
checking whether their separate confidence intervals overlap is a weaker test
than it looks: overlapping intervals do not imply an insignificant difference.
Resampling the same test tenders for both models removes the shared difficulty
of each tender and answers the question actually being asked, which is whether
model A beats model B on this data.

**Cost reported next to accuracy.** A model that is two points better and three
orders of magnitude slower has not obviously won. Every score is published
alongside the seconds it took to produce.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def score(y_true: list[str], y_pred: list[str]) -> dict:
    """Accuracy, macro-F1, weighted-F1 and the per-class breakdown."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} gold labels vs {len(y_pred)} predictions")

    correct = sum(a == b for a, b in zip(y_true, y_pred, strict=True))
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    per_class = {
        label: {
            "precision": round(v["precision"], 4),
            "recall": round(v["recall"], 4),
            "f1": round(v["f1-score"], 4),
            "support": int(v["support"]),
        }
        for label, v in report.items()
        if label not in {"accuracy", "macro avg", "weighted avg"}
    }

    return {
        "accuracy": round(correct / len(y_true), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
        "n": len(y_true),
        "n_classes_predicted": len(set(y_pred)),
        "n_classes_true": len(set(y_true)),
        "per_class": dict(sorted(per_class.items())),
    }


def bootstrap_metric(
    y_true: list[str],
    y_pred: list[str],
    metric: str = "accuracy",
    n_resamples: int = 5000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Confidence interval for one model's score, by resampling test rows."""
    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    n = len(truth)

    values = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        if metric == "accuracy":
            values[i] = (truth[idx] == pred[idx]).mean()
        else:
            values[i] = f1_score(truth[idx], pred[idx], average="macro", zero_division=0)

    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(values, [alpha, 1 - alpha])
    point = ((truth == pred).mean() if metric == "accuracy"
             else f1_score(truth, pred, average="macro", zero_division=0))
    return {
        "metric": metric,
        "point": round(float(point), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "n_resamples": n_resamples,
    }


def paired_bootstrap(
    y_true: list[str],
    pred_a: list[str],
    pred_b: list[str],
    metric: str = "accuracy",
    n_resamples: int = 5000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict:
    """Test whether model A beats model B, resampling the same rows for both.

    Both models are scored on the identical resampled tenders, which controls
    for the fact that some tenders are simply harder to classify than others.
    If the interval covers zero, the honest report is that this test set cannot
    separate the two models.
    """
    rng = np.random.default_rng(seed)
    truth = np.asarray(y_true)
    a = np.asarray(pred_a)
    b = np.asarray(pred_b)
    n = len(truth)

    def compute(t, p):
        if metric == "accuracy":
            return (t == p).mean()
        return f1_score(t, p, average="macro", zero_division=0)

    observed = compute(truth, a) - compute(truth, b)
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = compute(truth[idx], a[idx]) - compute(truth[idx], b[idx])

    alpha = (1 - confidence) / 2
    lower, upper = np.quantile(diffs, [alpha, 1 - alpha])
    tail = np.mean(diffs <= 0) if observed > 0 else np.mean(diffs >= 0)

    return {
        "metric": metric,
        "difference": round(float(observed), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "p_value": round(min(1.0, float(tail) * 2), 4),
        "significant": bool(lower > 0 or upper < 0),
        "n_resamples": n_resamples,
    }


def minimum_detectable_effect(
    comparison: dict, power: float = 0.80, alpha: float = 0.05
) -> dict:
    """How large a gap this comparison could have detected, given its size.

    An interval that covers zero means the test did not separate the two
    models. It does not mean the two models are equal, and the distance between
    those two statements is the whole content of this function.

    A comparison on 200 tenders has a standard error around four accuracy
    points, which makes anything under roughly eleven points invisible to it.
    Writing such a null up as "parity" or "it matches" claims a result the
    design could not have produced, in the one direction that flatters the
    conclusion.

    The standard error is recovered from the bootstrap interval's half-width
    rather than recomputed, so this describes the test that was actually run:

        se   = (upper - lower) / (2 * z_{1-alpha/2})
        mde  = (z_{1-alpha/2} + z_{power}) * se

    Two numbers come back and they answer different questions. The **MDE** is
    prospective: what this design could have caught. `largest_effect_not_
    excluded` is retrospective and is the one an equivalence claim has to clear
    -- the biggest true gap still consistent with the data, read straight off
    the far end of the interval. Equivalence within some margin is defensible
    only when that margin exceeds it.

    There is deliberately no boolean here saying a null is informative. Any
    non-significant difference has |delta| < z_{1-alpha/2} * se, which is
    always below the MDE, so such a flag could never be true and would only
    look like a check that had been performed.
    """
    half_width = (comparison["ci_upper"] - comparison["ci_lower"]) / 2
    # NormalDist rather than scipy: scipy is only present here as a
    # scikit-learn transitive dependency and is not declared in
    # pyproject.toml, so importing it directly would be a dependency this
    # project never took.
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    se = half_width / z_alpha
    mde = (z_alpha + z_power) * se
    bound = max(abs(comparison["ci_lower"]), abs(comparison["ci_upper"]))

    return {
        "standard_error": round(float(se), 4),
        "minimum_detectable_effect": round(float(mde), 4),
        "largest_effect_not_excluded": round(float(bound), 4),
        "power": power,
        "alpha": alpha,
        "observed_difference": comparison["difference"],
        "note": ("A null smaller than the MDE is a test that could not have "
                 "found anything. Equivalence is only defensible against a "
                 "margin wider than largest_effect_not_excluded."),
    }


def confusion_pairs(
    y_true: list[str], y_pred: list[str], top: int = 12
) -> list[dict]:
    """The most frequent confusions, which is where the error analysis starts.

    Aggregate accuracy says a model is wrong; this says what it is wrong about.
    Segments 72 (construction services) and 30 (construction materials) are the
    kind of pair a buyer would also hesitate over, and separating genuine model
    error from genuine annotation ambiguity needs the pairs, not the total.
    """
    labels = sorted(set(y_true) | set(y_pred))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    out = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and matrix[i][j] > 0:
                out.append({
                    "true": true_label,
                    "predicted": pred_label,
                    "count": int(matrix[i][j]),
                    "share_of_true_class": round(
                        float(matrix[i][j] / max(matrix[i].sum(), 1)), 4
                    ),
                })
    return sorted(out, key=lambda r: r["count"], reverse=True)[:top]


def format_table(results: dict[str, dict], order: list[str] | None = None) -> str:
    """Markdown table of every model, ready to paste into a report."""
    names = order or list(results)
    lines = [
        "| Model | Accuracy | Macro-F1 | Classes predicted | Seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        if name not in results:
            continue
        r = results[name]
        lines.append(
            f"| {name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['n_classes_predicted']} | {r.get('predict_seconds', 0):.1f} |"
        )
    return "\n".join(lines)
