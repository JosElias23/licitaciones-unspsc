"""Quantise the fine-tuned encoder for CPU serving, and measure what actually changed.

The usual way to report compression is a single line: "INT8, accuracy within
0.3% of the baseline, 3x faster". That line can be true and still hide most of
what happened.

Microsoft Research's *Accuracy Is Not All You Need* (arXiv 2407.09141) makes the
point directly: aggregate accuracy between a baseline and its compressed version
typically differs by under 2%, while the proportion of individual answers that
*change* is far larger. Two models with the same accuracy can disagree with each
other on a large minority of inputs, half the flips in each direction, and the
aggregate number stays flat. For anyone who has to explain to a user why the
answer changed after a deployment, the flip rate is the number that matters.

So every arm here reports both:

  aggregate    accuracy and macro-F1, the numbers usually published
  distance     flip rate against the FP32 baseline, mean KL divergence between
               the two probability distributions, and agreement on the argmax

plus CPU latency and on-disk size, because the entire reason to quantise is to
serve on hardware that has no GPU.

One rule the paper's argument depends on: the *same* test set, the *same*
inputs, in the *same* order, through every arm. Flips are a paired measurement
and are meaningless otherwise.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ArmResult:
    """One point on the quantisation ladder."""

    name: str
    accuracy: float
    macro_f1: float
    size_mb: float
    p50_ms: float
    p95_ms: float
    predictions: list[str] = field(default_factory=list)
    probabilities: np.ndarray | None = None
    flip_rate: float | None = None
    mean_kl: float | None = None
    flips_that_fix: int = 0
    flips_that_break: int = 0
    flips_wrong_to_wrong: int = 0

    def summary(self) -> dict:
        out = {
            "arm": self.name,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "size_mb": round(self.size_mb, 1),
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
        }
        if self.flip_rate is not None:
            out.update({
                "flip_rate": round(self.flip_rate, 4),
                "mean_kl_divergence": round(self.mean_kl, 6),
                "flips_wrong_to_right": self.flips_that_fix,
                "flips_right_to_wrong": self.flips_that_break,
                "flips_wrong_to_wrong": self.flips_wrong_to_wrong,
            })
        return out


def directory_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1e6
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def export_onnx(model, tokenizer, out_dir: Path, max_length: int = 64) -> Path:
    """Export the fine-tuned encoder to ONNX with dynamic axes.

    Batch and sequence length are both dynamic. Fixing them would force padding
    every request to the training length, which is exactly the waste CPU serving
    cannot afford on eight-word titles.
    """
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    model = model.cpu().eval()
    dummy = tokenizer(
        ["adquisicion de insumos de aseo"], return_tensors="pt",
        truncation=True, max_length=max_length, padding="max_length",
    )
    inputs = (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]) \
        if "token_type_ids" in dummy else (dummy["input_ids"], dummy["attention_mask"])
    names = ["input_ids", "attention_mask"] + (
        ["token_type_ids"] if "token_type_ids" in dummy else []
    )

    torch.onnx.export(
        model,
        inputs,
        str(onnx_path),
        input_names=names,
        output_names=["logits"],
        dynamic_axes={n: {0: "batch", 1: "sequence"} for n in names}
                     | {"logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        # The TorchScript exporter, not the dynamo one. torch>=2.9 defaults to
        # dynamo, which ignores `dynamic_axes` in favour of `dynamic_shapes` and
        # emits a graph whose shape inference onnxruntime's quantizer rejects
        # with "Inferred shape and existing shape differ in dimension 0:
        # (768) vs (32)". The legacy path produces a graph the quantizer
        # handles, which is what matters here.
        dynamo=False,
    )
    return onnx_path


def quantise_dynamic(onnx_path: Path, out_path: Path) -> Path:
    """Dynamic INT8: weights quantised ahead of time, activations at run time.

    Dynamic rather than static because static needs a calibration set, and a
    calibration set drawn from the wrong period would leak the test month into
    the quantisation parameters. Dynamic avoids that question entirely, at the
    cost of some speed.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )
    return out_path


class OnnxRunner:
    """Runs an ONNX classifier on CPU and returns probabilities."""

    def __init__(self, onnx_path: Path, tokenizer, id2label: dict, max_length: int = 64):
        import onnxruntime as ort

        options = ort.SessionOptions()
        # One thread. Serving containers are usually capped at a fraction of a
        # core, and a latency measured with every core of a workstation is not a
        # number anyone can plan capacity from.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.max_length = max_length

    def probabilities(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="np",
            )
            feed = {k: v.astype(np.int64) for k, v in encoded.items()
                    if k in self.input_names}
            logits = self.session.run(None, feed)[0]
            shifted = logits - logits.max(axis=-1, keepdims=True)
            exp = np.exp(shifted)
            out.append(exp / exp.sum(axis=-1, keepdims=True))
        return np.vstack(out)

    def labels(self, probabilities: np.ndarray) -> list[str]:
        return [self.id2label[int(i)] for i in probabilities.argmax(axis=-1)]

    def time_single(self, texts: list[str], repeats: int = 3) -> tuple[float, float]:
        """Per-item latency at batch size one, after a warm-up.

        Batch one is the interactive case, and the one a quantisation claim is
        usually made about. Warm-up matters: the first call pays for graph
        initialisation and would dominate a p50 over a few hundred items.
        """
        for t in texts[:10]:
            self.probabilities([t])

        latencies = []
        for _ in range(repeats):
            for t in texts:
                started = time.perf_counter()
                self.probabilities([t])
                latencies.append((time.perf_counter() - started) * 1000)
        return float(np.percentile(latencies, 50)), float(np.percentile(latencies, 95))


def compare_to_baseline(
    baseline_probs: np.ndarray,
    arm_probs: np.ndarray,
    baseline_labels: list[str],
    arm_labels: list[str],
    truth: list[str],
) -> dict:
    """Flip rate and KL divergence between a compressed model and its baseline.

    A flip is any input where the predicted label changes. It is split three
    ways, because "3% of answers changed" means something different depending on
    whether those changes were repairs or breakages, and the aggregate accuracy
    delta is the *net* of the first two while hiding the third entirely.
    """
    if baseline_probs.shape != arm_probs.shape:
        raise ValueError("Arms must be scored on identical inputs in identical order")

    flipped = [i for i, (a, b) in enumerate(zip(baseline_labels, arm_labels, strict=True))
               if a != b]

    fixes = breaks = wrong_to_wrong = 0
    for i in flipped:
        was_right = baseline_labels[i] == truth[i]
        now_right = arm_labels[i] == truth[i]
        if not was_right and now_right:
            fixes += 1
        elif was_right and not now_right:
            breaks += 1
        else:
            wrong_to_wrong += 1

    # KL(baseline || arm), the direction that asks how surprised the baseline
    # would be by the compressed model's distribution.
    eps = 1e-12
    p = np.clip(baseline_probs, eps, 1.0)
    q = np.clip(arm_probs, eps, 1.0)
    kl = float(np.mean(np.sum(p * np.log(p / q), axis=-1)))

    return {
        "flip_rate": len(flipped) / len(baseline_labels),
        "n_flipped": len(flipped),
        "mean_kl": kl,
        "flips_wrong_to_right": fixes,
        "flips_right_to_wrong": breaks,
        "flips_wrong_to_wrong": wrong_to_wrong,
    }


def cleanup(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
