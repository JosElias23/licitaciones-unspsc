# Predicting purchasing categories from Chilean public tender titles

[![CI](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml/badge.svg)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-83%20passing-brightgreen)](https://github.com/JosElias23/licitaciones-unspsc/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)](pyproject.toml)
[![data](https://img.shields.io/badge/data-CC0%20ChileCompra-lightgrey)](https://api.mercadopublico.cl)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**English** · [Español](README.es.md)

Chile publishes every public tender through an open OCDS API, and each one
arrives with two things: the free-text title a civil servant typed, and the
official UNSPSC category its line items were filed under. That pairing is a
labelled dataset produced as a by-product of government administration.

This project asks whether eight words of abbreviated, capitalised Spanish are
enough to recover the category, and whether a local LLM does it better than a
linear model.

**Zero-shot, it does not**: the LLM loses by 27.8 accuracy points and takes 624
times longer. Given eight *retrieved* examples it closes most of that gap — but
so does a plain majority vote over the same eight examples, with no language
model involved, at a fifteenth of the latency. Eight *random* examples change
nothing at all. Both control arms exist because without them the flattering
reading is the only one available.

---

## Results

Held out: April 2026 (952 tenders). Trained on January to March (2,874).
32 categories after folding the rare tail.

| Model | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| Majority class | 0.1429 | [0.1208, 0.1670] | 0.0078 | 0.00 |
| Keyword rules | 0.2384 | [0.2111, 0.2658] | 0.1295 | 0.08 |
| **TF-IDF + linear SVM** | **0.5578** | **[0.5263, 0.5893]** | **0.5077** | **0.26** |
| Local LLM (qwen2.5:7b) | 0.2794 | [0.2511, 0.3088] | 0.2491 | 162.19 |

Paired bootstrap over 5,000 resamples, every model on all 952 tenders:

| Comparison | Δ accuracy | 95% CI | Significant |
|---|---:|:--|:--:|
| TF-IDF vs keywords | +0.3193 | [+0.2805, +0.3571] | yes |
| **LLM vs TF-IDF** | **−0.2784** | **[−0.3172, −0.2395]** | **yes** |

The supervised model learns the vocabulary of Chilean procurement from 2,874
examples. Zero-shot, the LLM has to reason about it from a prompt, and eight
capitalised abbreviated words are not much to reason from.

> **Correction.** This table used to report the LLM at **2,242.89 ms** per
> tender and **9,345×** the cost of the linear model. About 2,054 ms of that was
> the client, not the model: `localhost` resolves to `::1` first, Ollama binds
> IPv4 only, and `urllib` waits for Windows to refuse the IPv6 connection before
> falling back — on every call, serially.
> [`scripts/measure_llm_overhead.py`](scripts/measure_llm_overhead.py) times an
> endpoint that computes nothing and finds 2,055.4 ms over the hostname against
> 1.3 ms over the IP.
>
> It also changed what was affordable. At 2.2 s per tender a full pass took 36
> minutes, so the LLM was scored on a 300-tender subset and the comparison came
> with a caveat about matched rows. At 162 ms it takes two and a half minutes,
> so every row above is the full test set and the caveat is gone — which is why
> the gap reads 27.8 points here rather than the 32 published before.

### Then it was given examples

Same held-out month, same seeded 200-tender subset, k = 8:

| Variant | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| TF-IDF + linear SVM | 0.5100 | [0.4400, 0.5800] | 0.4400 | 0.28 |
| LLM zero-shot | 0.3150 | [0.2500, 0.3800] | 0.2908 | 164.52 |
| LLM + 8 random examples | 0.2750 | [0.2150, 0.3400] | 0.2782 | 170.37 |
| **LLM + 8 retrieved examples** | **0.4650** | **[0.3950, 0.5350]** | 0.3839 | 277.56 |
| **kNN majority vote, no LLM** | **0.4700** | **[0.4000, 0.5400]** | **0.4152** | **18.58** |
| LLM + DSPy demonstrations | 0.4200 | [0.3549, 0.4900] | 0.3944 | 395.36 |

| Comparison | Δ accuracy | 95% CI | Significant | MDE @80% |
|---|---:|:--|:--:|---:|
| Random examples vs zero-shot | −0.0400 | [−0.0900, +0.0100] | **no** | 0.0715 |
| **Retrieved vs random** | **+0.1900** | **[+0.1150, +0.2650]** | **yes** | 0.1072 |
| **Retrieved vs kNN majority vote** | **−0.0050** | **[−0.0800, +0.0700]** | **no** | 0.1072 |
| Retrieved vs TF-IDF | −0.0450 | [−0.1150, +0.0250] | **no** | 0.1001 |
| DSPy vs retrieved | −0.0450 | [−0.1250, +0.0350] | **no** | 0.1144 |

Four things fall out of that table.

**Examples alone do nothing; relevant examples do everything.** Random
demonstrations moved accuracy by −0.040 (p = 0.134). Retrieved ones gained 19
points over the same count. Without the random arm, "few-shot helped" would have
been indistinguishable from "the model finally saw the output format".

**And the language model is not what earned those 19 points.** A majority vote
over the same eight retrieved neighbours — same retriever, same examples, no LLM
call — scores 0.4700 against the LLM's 0.4650 and beats it on macro-F1, at 18.58
ms against 277.56. This control did not exist before and it should have: without
it, "retrieval few-shot works" and "the retriever works" are the same number.

They are not, however, the same predictor. The two agree on only **44%** of
tenders, and each is right where the other is wrong about 30 times. The model is
not copying the majority label off its prompt; it does something genuinely
different that is worth nothing here and costs fifteen times as much.

**The gap to TF-IDF does not close to parity, and this README used to say it
did.** Retrieval sits 4.5 points below the linear model with an interval
covering zero — but on 200 tenders this comparison could not have detected
anything smaller than **10 points**, and the interval is still consistent with
the LLM being **11.5 points worse**. A null is not equivalence. Every null in
the table now carries the effect it could have found.

**DSPy is slower than retrieval, not six times faster.** This README reported
412 ms against 2,362 ms and explained it by compile-time demonstrations. Both
numbers were instruments rather than models: the retrieval arm was paying the
IPv6 timeout, and `dspy.LM` caches by default while no other arm does. Measured
the same way, DSPy costs **395 ms against 278 ms**. Its first run also scored
0.1050 for a reason that was a bug in this repository rather than in DSPy, and
that is written up in [`docs/DECISIONS.md`](docs/DECISIONS.md) section 5.3.

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

### And the favourite lost too

Every version of this README named a fine-tuned Spanish encoder as the likely
winner and left it untested. BETO (110 M parameters), fine-tuned on the same
2,874 tenders, at the epoch count chosen on a **validation** month:

| Model | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| TF-IDF + linear SVM | 0.5578 | [0.5263, 0.5893] | 0.5077 | 0.26 |
| BETO fine-tuned | 0.5221 | [0.4916, 0.5546] | 0.4646 | 0.44 |

−0.0357 accuracy, [−0.0651, −0.0063], p = 0.0196. **The prediction was wrong**:
the transformer does not win and does not draw level. It loses, significantly,
after 144 seconds of GPU training.

> **Correction.** An earlier version of this README reported 0.5378 and
> "draws level, p = 0.186", choosing 15 epochs from a table of 6/15/30 whose
> accuracies were all measured on **April — the held-out month**. There was no
> validation split in this project at all. That is selection on the test set,
> and of the three candidates exactly one gave a non-significant result, which
> is the one that got published.
>
> [`scripts/select_epochs.py`](scripts/select_epochs.py) splits the training
> window instead — train on January–February, choose on March, read April once
> — and picks **30** epochs, where BETO loses significantly. The two months do
> not even agree about the shape: March rises monotonically across the grid,
> April peaks at 15. The "inverted U" was a property of the month it was read
> from.
>
> The claim that survives every choice: **BETO does not beat TF-IDF at any
> epoch count in the grid.** Its best showing is −0.0200 and its worst −0.1134.
> Selection decided whether the loss was called significant, not whether there
> was one. Full protocol, the seed-noise check and the disclosed per-epoch test
> table are in [`docs/DECISIONS.md`](docs/DECISIONS.md) section 6.1.

Three reasons it does not win, in the order I would bet on them: about 90
examples per class, procurement titles that WordPiece shreds (`ADQ. SERV.
REPARACION VEH-.`) where character n-grams do not, and a macro-F1 gap wider than
the accuracy gap, which points at the rare classes.

### What would actually ship

Forced to answer everything, the encoder gets 52.2% right. Allowed to abstain:

| Confidence ≥ | Coverage | Accuracy on what it keeps |
|---:|---:|---:|
| 0.50 | 96.2% | 0.5349 |
| 0.70 | 90.4% | 0.5552 |
| 0.90 | 80.0% | 0.5958 |
| 0.95 | 74.5% | 0.6178 |

The confidence score does rank correct predictions above wrong ones: **AUROC
0.7401, permutation p = 0.0005** over 2,000 shuffles. An earlier version argued
this from the column rising monotonically, which is close to automatic — the
kept sets are nested, so any positive association at all produces a rising
column. The AUROC is the test; the monotone column was not one.

It is still not enough: 61.8% accuracy while abstaining on a quarter of tenders
means almost four in ten of the kept ones are wrong. **This is a suggestion
tool, not an auto-filing system**, and saying otherwise would be the easiest
overclaim in the whole project.

One cost of the correction above is worth naming. At the test-selected 15
epochs, this table was *better* — 39.7% coverage at 0.7275 accuracy. Thirty
epochs buys accuracy on paper and spends calibration, and the epoch count was
selected on accuracy alone. Re-selecting on a calibration-aware criterion now,
after seeing which one flatters this table, would be the original mistake in a
third costume.

### Quantised for CPU: the accuracy line hides most of the change

The encoder is the model a procurement desk would run, and it would run without
a GPU. Three arms, identical test items in identical order, one CPU thread,
batch size one:

| Arm | Accuracy | Macro-F1 | Size | p50 | **Flip rate** | Mean KL |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0.5231 | 0.4657 | 439.5 MB | 108.5 ms | — | — |
| ONNX FP32 | 0.5231 | 0.4657 | 439.7 MB | 82.2 ms | **0.00%** | −0.00000 |
| ONNX INT8 | 0.5179 | **0.4741** | **110.6 MB** | **27.7 ms** | **6.83%** | 0.09224 |

Exporting to ONNX changes nothing at all: zero flips, KL zero, 1.32× faster.
INT8 is 4.0× smaller, 3.9× faster, and its accuracy drop of 0.0053 has a
bootstrap interval of [−0.0168, +0.0063] that covers zero. Reported the usual
way, that reads as free.

It is not free. **65 of 952 predictions changed, 6.83%, while accuracy moved by
half a point.**

| Direction | Count |
|---|---:|
| Wrong → right | 15 |
| Right → wrong | 20 |
| Wrong → a *different* wrong answer | **30** |

Nearly half the flips are in that last row and no aggregate metric can see them:
accuracy nets the first two against each other and is blind to the third. The
framing follows Microsoft Research's [*Accuracy Is Not All You
Need*](https://arxiv.org/abs/2407.09141), which argues flips and KL divergence
are the right way to compare a compressed model to its baseline.

> **Correction.** An earlier version put a number on that contrast: "15.6 times
> more churn than the published number implies", dividing the flip rate by the
> net accuracy move. That ratio is not a quantity. Its denominator is a
> difference between two nearly equal numbers whose interval covers zero, so the
> ratio is unbounded — a denominator at either end of that interval sends it to
> infinity or flips its sign. The contrast is real and the arithmetic dressing
> it up was not.

INT8 is still what to ship here. It just is not the same model: it agrees with
the original 93% of the time and is equally good on average, and only the second
half of that sentence usually gets written down. It also has a *higher* macro-F1
than the model it compresses, which is a reminder that "equally good on average"
is doing a lot of work in that sentence.

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
python -m pytest                                # 83 tests
python scripts/download_data.py                 # ~11 minutes
python scripts/run_experiment.py --skip-llm     # seconds
python scripts/run_experiment.py                # needs Ollama, ~3 minutes
python scripts/measure_llm_overhead.py          # client vs model latency
python scripts/run_llm_variants.py --sample 200 # few-shot arms, ~8 minutes
python scripts/select_epochs.py --sensitivity   # epoch choice on validation, GPU
python scripts/run_encoder.py                   # BETO, GPU, ~3 minutes
python scripts/run_quantisation.py              # ONNX + INT8, CPU timings
```

The crawl is cached as JSON and loaded into one DuckDB file, so everything after
the first download runs offline. The corpus questions in `download_data.py`
(label distribution, buyer concentration, per-month drift) are SQL, because that
is what they are.

Two of these are here because of defects found after publication.
`measure_llm_overhead.py` separates the model's latency from the HTTP client's,
and `select_epochs.py` chooses the encoder's epoch count on a validation month
instead of on the held-out one.

Every number in this README comes from a file in `reports/`, all produced by the
scripts above: `metrics_corpus.json`, `metrics_experiment.json`,
`metrics_llm_variants.json`, `metrics_encoder.json`, `metrics_quantisation.json`,
`metrics_epoch_selection.json` and `metrics_llm_overhead.json`.
`tests/test_published_numbers.py` fails if a figure in the prose is not in one
of them.

---

## Limitations

**No GEPA.** DSPy 3.3 ships `dspy.GEPA`, reported to reach a given quality in
35× fewer rollouts. Only `BootstrapFewShot` was run, so nothing is claimed about
stronger optimisers.

**No calibration.** The encoder's softmax scores rank well but are not
calibrated, so the coverage table gives operating points rather than guaranteed
error rates. Temperature scaling or conformal prediction is the most useful
thing left undone.

**Segment only.** UNSPSC nests segment → family → class → commodity. Only the
two-digit segment is predicted; the four-digit family is harder and more useful.

**The `other` class is the largest single error source.** It is 22 rare segments
in a bag, so it has no coherent vocabulary. Reporting accuracy without saying so
would overstate how much of the error is genuine confusion.

**Four months.** No seasonality, and drift over a year is untested.

**The few-shot arms run on 200 of 952 test tenders**, for time. That subset is
small enough that the comparison against TF-IDF cannot detect a gap under about
10 accuracy points, which is why those nulls are reported with their minimum
detectable effect rather than as ties. The zero-shot arm now runs on all 952.

**The epoch count is chosen on a smaller corpus than it is applied to.**
Validation trains on 1,922 tenders and the final refit on 2,874, so 30 epochs is
the best setting for the smaller one. Matching optimisation *steps* instead of
epochs would remove that; it is not done. The bias runs against the model, not
for it.

**The LLM arm is not bit-reproducible.** Temperature is zero, but Ollama's
zero-shot accuracy moved from 0.2767 to 0.2794 between two runs of identical
code on identical data. Differences below about half a point in any LLM row here
should not be read as real.

## License

MIT for the code; the data is CC0 from ChileCompra.
