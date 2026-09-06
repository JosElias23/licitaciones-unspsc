# Predicting purchasing categories from Chilean public tender titles

[![CI](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-35%20passing-brightgreen)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![data](https://img.shields.io/badge/data-CC0%20ChileCompra-lightgrey)](https://api.mercadopublico.cl)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Chile publishes every public tender through an open OCDS API, and each one
arrives with two things: the free-text title a civil servant typed, and the
official UNSPSC category its line items were filed under. That pairing is a
labelled dataset produced as a by-product of government administration.

This project asks whether eight words of abbreviated, capitalised Spanish are
enough to recover the category, and whether a local LLM does it better than a
linear model.

**It does not.** The LLM loses by 32 accuracy points and takes 9,345 times
longer.

---

## Results

Held out: April 2026 (952 tenders). Trained on January to March (2,874).
32 categories after folding the rare tail.

| Model | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| Majority class | 0.1429 | [0.1208, 0.1670] | 0.0078 | 0.00 |
| Keyword rules | 0.2384 | [0.2111, 0.2658] | 0.1295 | 0.08 |
| **TF-IDF + linear SVM** | **0.5578** | **[0.5263, 0.5893]** | **0.5077** | **0.24** |
| Local LLM (qwen2.5:7b) | 0.2767 | [0.2267, 0.3300] | 0.2248 | 2,242.89 |

Paired bootstrap over 5,000 resamples, on the rows both models saw:

| Comparison | Δ accuracy | 95% CI | Significant |
|---|---:|:--|:--:|
| TF-IDF vs keywords | +0.3193 | [+0.2805, +0.3571] | yes |
| **LLM vs TF-IDF** | **−0.3200** | **[−0.3900, −0.2500]** | **yes** |

The supervised model learns the vocabulary of Chilean procurement from 2,874
examples. The LLM has to reason about it from a prompt, and eight capitalised
abbreviated words are not much to reason from. That is a result about this task,
not about LLMs: few-shot prompting and fine-tuning are untried and are the
obvious next experiments.

---

## The finding that shaped the project

Every line item carries a `description` that looks like free text:

> `47131803` → "Equipos y suministros de limpieza / Suministros de limpieza /
> Soluciones de limpieza y desinfección / Desinfectantes domésticos"

It is not free text. It is the UNSPSC taxonomy path written out in Spanish. Over
a 444-item sample, the first field of that description is a single constant
string for **40 of 40** segments, and its token overlap with the label is 1.000.

Training on it would have produced a near-perfect score and learned nothing. The
buyer-written title overlaps the label by 0.093, and 23% of titles share no
content word with it at all. That is the real task.

`assert_no_leakage()` raises if any label-derived field reaches a model, and
seven tests cover it.

A second decision mattered nearly as much. Counting line items let one
food-supply tender with hundreds of lines supply 61.7% of a sample; labelling
one row per tender brought the majority class down to 10.1%.

Full reasoning, including a reproducibility bug that seeding did not catch, is in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

---

## Data

ChileCompra / Mercado Público OCDS API. **CC0**, no key, no registration.

| | |
|---|---:|
| Tenders with a UNSPSC label | 3,826 |
| Line items | 15,823 |
| Distinct segments | 54 |
| Single-category tenders | 91.6% |
| Mean title length | 8.3 words |
| Months covered | Jan–Apr 2026 |

One quirk worth knowing: the API returns its own detail URLs over `http://`, and
those connections are closed without a response. Every URL is rewritten to
`https://` before use. A crawler that trusts the payload fails every request.

---

## Running it

```bash
pip install -e ".[dev]"
python -m pytest                                     # 35 tests
python scripts/download_data.py                      # ~11 minutes
python scripts/run_experiment.py --skip-llm          # seconds
python scripts/run_experiment.py --llm-sample 300    # needs Ollama, ~11 minutes
```

The crawl is cached as JSON and loaded into one DuckDB file, so everything after
the first download runs offline. The corpus questions in `download_data.py`
(label distribution, buyer concentration, per-month drift) are SQL, because that
is what they are.

Every number in this README comes from `reports/metrics_corpus.json` and
`reports/metrics_experiment.json`, both produced by the scripts above.

---

## Limitations

**No few-shot or optimised prompting.** The zero-shot gap is 32 points. Closing
it with examples is plausible and untested.

**No fine-tuned encoder.** A Spanish BERT fine-tuned on 2,874 examples is the
honest favourite for this task and is not implemented.

**Segment only.** UNSPSC nests segment → family → class → commodity. Only the
two-digit segment is predicted; the four-digit family is harder and more useful.

**The `other` class is the largest single error source.** It is 22 rare segments
in a bag, so it has no coherent vocabulary. Reporting accuracy without saying so
would overstate how much of the error is genuine confusion.

**Four months.** No seasonality, and drift over a year is untested.

**The LLM ran on 300 of 952 test tenders**, for time. Its interval is wider
accordingly.

## License

MIT for the code; the data is CC0 from ChileCompra.
