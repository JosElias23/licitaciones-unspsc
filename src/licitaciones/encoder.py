"""Fine-tuning a Spanish encoder, the model the README called the honest favourite.

The ladder so far runs majority -> keywords -> TF-IDF -> LLM with retrieval, and
the linear model is still the one to beat. Every version of this project's
write-up has said that a Spanish BERT fine-tuned on the training split would
probably win, and until now that claim sat in the limitations section untested.
A favourite nobody ran is a hole, not a limitation.

BETO (`dccuchile/bert-base-spanish-wwm-cased`) is the natural choice: trained on
Spanish, cased, and small enough to fine-tune in minutes on a single consumer
GPU. The titles are eight words long, so `max_length` is tiny and batches are
large.

What makes this a fair comparison rather than a demonstration:

- it sees exactly the same text as every other model, through `tender_text`,
  which excludes the item description and the buyer;
- it trains on exactly the same 2,874 tenders, split on the same date;
- it is scored by the same `score()` and compared with the same paired
  bootstrap;
- its inference cost is measured the same way, so "better" and "more expensive"
  are both on the table.

A fine-tuned encoder ought to win. The point of running it is to find out by how
much, and at what cost, rather than to assume.
"""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset

from licitaciones.data import Tender, assert_no_leakage
from licitaciones.models import BaseClassifier, tender_text


class TenderDataset(Dataset):
    """Tokenised titles with integer labels."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length, padding=False
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


class EncoderClassifier(BaseClassifier):
    """BETO fine-tuned for sequence classification over UNSPSC segments."""

    name = "beto_finetuned"

    def __init__(self, cfg: dict, seed: int = 42):
        self.model_name = cfg.get("model_name", "dccuchile/bert-base-spanish-wwm-cased")
        self.max_length = cfg.get("max_length", 64)
        self.epochs = cfg.get("num_epochs", 6)
        self.batch_size = cfg.get("batch_size", 32)
        self.learning_rate = cfg.get("learning_rate", 3e-5)
        self.weight_decay = cfg.get("weight_decay", 0.01)
        self.warmup_ratio = cfg.get("warmup_ratio", 0.1)
        self.seed = seed
        self.train_seconds = 0.0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, tenders: list[Tender], labels: list[str]):
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )

        assert_no_leakage(["title", "description"])

        self.classes = sorted(set(labels))
        self.label2id = {c: i for i, c in enumerate(self.classes)}
        self.id2label = {i: c for c, i in self.label2id.items()}
        self.fallback = Counter(labels).most_common(1)[0][0]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.classes),
            id2label=self.id2label,
            label2id=self.label2id,
        )

        texts = [tender_text(t) for t in tenders]
        y = [self.label2id[lbl] for lbl in labels]
        dataset = TenderDataset(texts, y, self.tokenizer, self.max_length)

        args = TrainingArguments(
            output_dir="models/beto-unspsc",
            seed=self.seed,
            data_seed=self.seed,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            logging_steps=50,
            save_strategy="no",          # nothing here needs a checkpoint on disk
            report_to="none",
            bf16=(self.device == "cuda"),
            disable_tqdm=True,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset,
            data_collator=DataCollatorWithPadding(self.tokenizer),
        )

        started = time.perf_counter()
        trainer.train()
        self.train_seconds = time.perf_counter() - started

        self.model = trainer.model.to(self.device).eval()
        self.n_parameters = sum(p.numel() for p in self.model.parameters())
        return self

    def predict(self, tenders: list[Tender], batch_size: int = 64) -> list[str]:
        texts = [tender_text(t) for t in tenders]
        out: list[str] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits.cpu().numpy()
            out.extend(self.id2label[int(i)] for i in np.argmax(logits, axis=-1))

        return out

    def predict_with_confidence(
        self, tenders: list[Tender], batch_size: int = 64
    ) -> tuple[list[str], np.ndarray]:
        """Predictions plus the softmax probability of the chosen class.

        Kept because a category suggester is far more useful when it can say it
        does not know: a buyer would rather see "no confident suggestion" than a
        wrong code presented with the same authority as a right one.
        """
        texts = [tender_text(t) for t in tenders]
        labels: list[str] = []
        confidences: list[float] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch, truncation=True, max_length=self.max_length,
                padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                idx = int(np.argmax(row))
                labels.append(self.id2label[idx])
                confidences.append(float(row[idx]))

        return labels, np.asarray(confidences)

    @property
    def cost_note(self) -> str:
        return (f"fine-tuned in {self.train_seconds:.0f}s on {self.device}, "
                f"{getattr(self, 'n_parameters', 0) / 1e6:.0f}M parameters")


def coverage_accuracy_curve(
    y_true: list[str],
    y_pred: list[str],
    confidence: np.ndarray,
    thresholds: tuple[float, ...] = (0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> list[dict]:
    """Accuracy as a function of how much of the workload the model keeps.

    A classifier that must answer everything is judged on one number. A
    classifier allowed to abstain trades coverage for accuracy, and the useful
    question for a procurement desk is the operating point: what share of tenders
    can be auto-categorised at an error rate somebody will accept, leaving the
    rest for a human.
    """
    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    rows = []

    for threshold in thresholds:
        keep = confidence >= threshold
        n_kept = int(keep.sum())
        rows.append({
            "threshold": threshold,
            "coverage": round(n_kept / len(truth), 4),
            "n_kept": n_kept,
            "accuracy_on_kept": (
                round(float((truth[keep] == pred[keep]).mean()), 4) if n_kept else None
            ),
        })
    return rows
