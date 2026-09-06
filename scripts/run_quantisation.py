"""Quantise the fine-tuned encoder for CPU serving and report flips, not just accuracy.

Three arms, all scored on the identical held-out month in identical order,
because flips are a paired measurement:

  torch_fp32    the fine-tuned model as trained, on CPU
  onnx_fp32     exported to ONNX, same weights, so any difference is the runtime
  onnx_int8     dynamic INT8 quantisation of the ONNX graph

The middle arm exists to separate two things that get conflated: what the export
changes and what the quantisation changes. Without it, every difference gets
blamed on INT8.

Usage:
    python scripts/run_quantisation.py
    python scripts/run_quantisation.py --latency-sample 100
"""

from __future__ import annotations

import argparse
import sys

# The ONNX exporter logs a tick emoji on success, and the default Windows console
# encoding (cp1252) cannot represent it, which turns a successful export into a
# UnicodeEncodeError. Force UTF-8 before anything imports torch.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import numpy as np  # noqa: E402
from run_experiment import load_tenders  # noqa: E402

from licitaciones.encoder import EncoderClassifier  # noqa: E402
from licitaciones.evaluate import paired_bootstrap, score  # noqa: E402
from licitaciones.models import build_label_set, tender_text  # noqa: E402
from licitaciones.quantise import (  # noqa: E402
    ArmResult,
    OnnxRunner,
    compare_to_baseline,
    directory_size_mb,
    export_onnx,
    quantise_dynamic,
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
    p.add_argument("--latency-sample", type=int, default=60,
                   help="Tenders used for the batch-size-one latency measurement")
    p.add_argument("--epochs", type=int, default=None)
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

    encoder_cfg = dict(config.get("encoder", {}))
    if args.epochs:
        encoder_cfg["num_epochs"] = args.epochs

    log.info("Fine-tuning the encoder to quantise (%d train, %d test) ...",
             len(train), len(test))
    encoder = EncoderClassifier(encoder_cfg, seed=config["seed"])
    encoder.fit(train, y_train)

    texts = [tender_text(t) for t in test]
    latency_texts = texts[: args.latency_sample]
    work = PROJECT_ROOT / "models" / "quantisation"
    work.mkdir(parents=True, exist_ok=True)

    arms: list[ArmResult] = []

    # --- arm 1: PyTorch FP32 on CPU --------------------------------------
    import torch

    torch.set_num_threads(1)   # match the ONNX session, or the race is unfair
    encoder.model = encoder.model.cpu().eval()
    encoder.device = "cpu"

    probs_torch = []
    for start in range(0, len(texts), 32):
        batch = texts[start : start + 32]
        enc = encoder.tokenizer(batch, truncation=True,
                                max_length=encoder.max_length,
                                padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = encoder.model(**enc).logits
            probs_torch.append(torch.softmax(logits, dim=-1).numpy())
    probs_torch = np.vstack(probs_torch)
    labels_torch = [encoder.id2label[int(i)] for i in probs_torch.argmax(axis=-1)]

    for t in latency_texts[:10]:
        encoder.tokenizer(t, return_tensors="pt")
    torch_latencies = []
    for _ in range(3):
        for t in latency_texts:
            started = time.perf_counter()
            enc = encoder.tokenizer(t, truncation=True, max_length=encoder.max_length,
                                    padding=True, return_tensors="pt")
            with torch.no_grad():
                encoder.model(**enc)
            torch_latencies.append((time.perf_counter() - started) * 1000)

    torch_dir = work / "torch"
    torch_dir.mkdir(exist_ok=True)
    encoder.model.save_pretrained(torch_dir)

    s = score(y_test, labels_torch)
    arms.append(ArmResult(
        name="torch_fp32", accuracy=s["accuracy"], macro_f1=s["macro_f1"],
        size_mb=directory_size_mb(torch_dir),
        p50_ms=float(np.percentile(torch_latencies, 50)),
        p95_ms=float(np.percentile(torch_latencies, 95)),
        predictions=labels_torch, probabilities=probs_torch,
    ))
    log.info("torch_fp32  acc %.4f | %.0f MB | p50 %.1f ms",
             s["accuracy"], arms[-1].size_mb, arms[-1].p50_ms)

    # --- arm 2: ONNX FP32 -------------------------------------------------
    log.info("Exporting to ONNX ...")
    onnx_path = export_onnx(encoder.model, encoder.tokenizer, work / "onnx",
                            encoder.max_length)
    runner = OnnxRunner(onnx_path, encoder.tokenizer, encoder.id2label,
                        encoder.max_length)
    probs_onnx = runner.probabilities(texts)
    labels_onnx = runner.labels(probs_onnx)
    p50, p95 = runner.time_single(latency_texts)
    s = score(y_test, labels_onnx)
    arms.append(ArmResult(
        name="onnx_fp32", accuracy=s["accuracy"], macro_f1=s["macro_f1"],
        size_mb=directory_size_mb(onnx_path), p50_ms=p50, p95_ms=p95,
        predictions=labels_onnx, probabilities=probs_onnx,
    ))
    log.info("onnx_fp32   acc %.4f | %.0f MB | p50 %.1f ms", s["accuracy"],
             arms[-1].size_mb, p50)

    # --- arm 3: ONNX dynamic INT8 ----------------------------------------
    log.info("Quantising to INT8 ...")
    int8_path = quantise_dynamic(onnx_path, work / "onnx" / "model_int8.onnx")
    runner_int8 = OnnxRunner(int8_path, encoder.tokenizer, encoder.id2label,
                             encoder.max_length)
    probs_int8 = runner_int8.probabilities(texts)
    labels_int8 = runner_int8.labels(probs_int8)
    p50, p95 = runner_int8.time_single(latency_texts)
    s = score(y_test, labels_int8)
    arms.append(ArmResult(
        name="onnx_int8", accuracy=s["accuracy"], macro_f1=s["macro_f1"],
        size_mb=directory_size_mb(int8_path), p50_ms=p50, p95_ms=p95,
        predictions=labels_int8, probabilities=probs_int8,
    ))
    log.info("onnx_int8   acc %.4f | %.0f MB | p50 %.1f ms", s["accuracy"],
             arms[-1].size_mb, p50)

    # --- distance from the baseline --------------------------------------
    baseline = arms[0]
    for arm in arms[1:]:
        d = compare_to_baseline(baseline.probabilities, arm.probabilities,
                                baseline.predictions, arm.predictions, y_test)
        arm.flip_rate = d["flip_rate"]
        arm.mean_kl = d["mean_kl"]
        arm.flips_that_fix = d["flips_wrong_to_right"]
        arm.flips_that_break = d["flips_right_to_wrong"]
        arm.flips_wrong_to_wrong = d["flips_wrong_to_wrong"]

    n_boot = config["evaluation"]["n_bootstrap"]
    comparisons = {
        arm.name: paired_bootstrap(y_test, arm.predictions, baseline.predictions,
                                   "accuracy", n_boot, config["seed"])
        for arm in arms[1:]
    }

    save_json(
        {
            "n_test": len(test),
            "latency_sample": len(latency_texts),
            "threads": 1,
            "arms": [a.summary() for a in arms],
            "accuracy_vs_baseline": comparisons,
            "note": (
                "Latency is batch size one on a single CPU thread, after warm-up, "
                "three repeats. Flips are paired: every arm scored the identical "
                "test items in identical order."
            ),
        },
        "reports/metrics_quantisation.json",
    )

    print()
    print("| Arm | Accuracy | Macro-F1 | Size MB | p50 ms | p95 ms | Flip rate | Mean KL |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for a in arms:
        flip = f"{a.flip_rate:.2%}" if a.flip_rate is not None else "-"
        kl = f"{a.mean_kl:.5f}" if a.mean_kl is not None else "-"
        print(f"| {a.name} | {a.accuracy:.4f} | {a.macro_f1:.4f} | {a.size_mb:.0f} | "
              f"{a.p50_ms:.1f} | {a.p95_ms:.1f} | {flip} | {kl} |")
    print()
    print("Where the flips went (relative to torch_fp32):")
    for a in arms[1:]:
        print(f"  {a.name}: {a.flips_that_fix} wrong->right, "
              f"{a.flips_that_break} right->wrong, "
              f"{a.flips_wrong_to_wrong} wrong->wrong "
              f"(net accuracy change {a.accuracy - baseline.accuracy:+.4f})")
    print()
    for name, c in comparisons.items():
        print(f"{name} vs torch_fp32 accuracy: {c['difference']:+.4f} "
              f"[{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}], "
              f"{'significant' if c['significant'] else 'NOT significant'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
