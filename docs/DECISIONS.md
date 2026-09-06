# Decision log

What was built, in what order, and why each choice was made. Every figure here
comes from a file in `reports/`, produced by running the code.

---

## 1. Why this task at all

**Question.** Given only the free-text title a Chilean public institution writes
when it publishes a tender, can the UNSPSC purchasing category be predicted, and
does a local LLM do it better than a linear model?

**Why it is worth asking.** ChileCompra publishes both halves of the pair: the
buyer's own words, and the official category assigned to each line item. That is
a labelled dataset produced as a by-product of government administration, at a
scale nobody would pay to annotate. The licence is **CC0**, the most permissive
a public dataset can carry, and the API needs no key.

**Why it might fail.** Titles average 8.3 words, are frequently ALL CAPS,
abbreviated (`ADQ.`, `SERV.`) and inconsistently accented. There are 54 distinct
segments. It is entirely possible that eight words are not enough.

---

## 2. Decisions that changed the result

### 2.1 The item description is the label, not a feature

This is the decision the project turns on.

Each line item carries a `description`. It looks like free text:

> `47131803` → "Equipos y suministros de limpieza / Suministros de limpieza /
> Soluciones de limpieza y desinfección / Desinfectantes domésticos"

It is not free text. It is the UNSPSC taxonomy path rendered in Spanish. Over a
444-item sample:

| | Value |
|---|---:|
| UNSPSC segments where the description's first field is a single constant string | **40 of 40** |
| Token overlap between item description and the label path | **1.000** |
| Token overlap between buyer-written title and the label path | **0.093** |
| Titles sharing zero content words with the label | **23%** |

Training on the item description would have scored near-perfectly and learned
nothing. A model reported that way is not wrong by a little, it is answering a
different question.

**Decision.** Only `title` and `description` (both buyer-written) may reach a
model. `FORBIDDEN_FEATURES` in `src/licitaciones/data.py` lists every field
derived from the label, and `assert_no_leakage()` raises if one appears. Seven
tests cover it, because a comment would not have stopped a future change.

**The buyer name is excluded too.** Some institutions buy almost exclusively in
one category, so a model keyed on the buyer would score well while learning
nothing about the words. The most concentrated case is CENABAST, which supplies
18% of segment 51 (pharmaceuticals). Not enough to dominate, but enough that
including it would blur what the model had learned.

### 2.2 One label per tender, not one per line item

The first sampling attempt counted line items. Over 444 items, segment 50 (food)
held 274 of them — **61.7%** — because a single food-supply tender carried
hundreds of lines.

That is one institution's paperwork becoming the dataset.

**Decision.** Each tender contributes exactly one row, labelled with the segment
covering most of its line items. The majority class fell from **61.7% to 10.1%**,
and the task went from "predict food" to a real 54-class problem.

91.6% of tenders are single-category, so the dominant-segment label is
unambiguous for nine in ten rows. The remaining 8.4% are genuinely mixed and are
kept rather than dropped, with the share reported.

### 2.3 The split is temporal, never random

Procurement vocabulary drifts: budget lines are renamed, programmes start and
end, institutions merge. A random split lets a model see April's phrasing while
predicting January's, which is not the situation it faces in use.

**Decision.** Everything published before 2026-04-01 trains (2,874 tenders),
everything after is held out (952). The four months are evenly sized (969 / 953 /
952 / 952), so the split is not an artefact of one unusual month.

### 2.4 Rare segments are folded, using the training set only

54 segments, with a long tail. A class seen twice in training cannot be learned,
and a class with one test example has recall of exactly 0 or 1.

**Decision.** Segments with fewer than 25 training tenders become `other`. That
leaves **32 classes**, and folds 309 of 2,874 training tenders (10.8%). The
threshold is applied to training counts only: deciding what counts as rare by
looking at the test set would leak test information into the label space.

### 2.5 A reproducibility bug that seeding did not catch

Two runs of the identical pipeline over the identical warehouse produced
different results:

| | Run A | Run B |
|---|---:|---:|
| Label classes after folding | 32 | 33 |
| Training tenders folded to `other` | 307 | 281 |
| Majority-class baseline | 0.1408 | 0.1261 |

The cause was one line:

```python
return max(set(segs), key=segs.count)       # before
```

When two segments tie on item count, `max` returns the first maximum it meets,
and a `set` of strings iterates in an order that depends on Python's per-process
randomised string hashing.

`set_seed()` was already being called and did not help: `PYTHONHASHSEED` is read
at interpreter start-up, so setting it from inside a running process affects only
child processes. **The seed gave a false sense of determinism.**

```python
counts = Counter(segs)
return min(counts, key=lambda s: (-counts[s], s))     # after
```

Three consecutive runs now give 32 classes, 309 folded, baseline 0.1429. Five
tests pin the tie-break.

### 2.6 Character n-grams alongside words

Titles are full of abbreviations and inconsistent accents: `MANTENCION` and
`MANTENCIÓN`, `ADQ.` and `ADQUISICION`. Whole-word features miss across those;
character n-grams degrade gracefully. Text is also lowercased and accent-folded,
which is a real gain here rather than a habit, because roughly half the titles
are written in capitals.

### 2.7 The LLM is given only the labels that exist in training

The prompt lists the 32 training segments, not all 57 UNSPSC segments. Offering
categories the evaluation cannot score would make the LLM's task strictly harder
than the one the classical models face, and the comparison would not be about
capability.

Unusable answers (a code outside the menu, or malformed JSON) fall back to the
majority class and are **counted**: 12 of 300. Dropping those rows would quietly
score the LLM on an easier test set than everything else.

---

## 3. Results

Held-out month (April 2026). Classical models on all 952 test tenders; the LLM on
a seeded random 300-tender subset, with every comparison computed on the rows
both models actually saw.

| Model | Accuracy | 95% CI | Macro-F1 | Classes predicted | ms per tender |
|---|---:|:--|---:|---:|---:|
| Majority class | 0.1429 | [0.1208, 0.1670] | 0.0078 | 1 | 0.00 |
| Keyword rules | 0.2384 | [0.2111, 0.2658] | 0.1295 | 21 | 0.08 |
| **TF-IDF + linear SVM** | **0.5578** | **[0.5263, 0.5893]** | **0.5077** | 32 | **0.24** |
| Local LLM (qwen2.5:7b) | 0.2767 | [0.2267, 0.3300] | 0.2248 | 28 | 2,242.89 |

Note that the first three rows are scored on all 952 test tenders and the LLM on
300 of them, so subtracting two rows of this table is not the comparison. On the
same 300 tenders the TF-IDF model scores 0.5967, and the paired test below uses
those matched rows.

Paired bootstrap, 5,000 resamples:

| Comparison | Δ accuracy | 95% CI | p | Significant |
|---|---:|:--|---:|:--:|
| Keywords vs majority | +0.0956 | [+0.0630, +0.1271] | <0.001 | yes |
| TF-IDF vs keywords | +0.3193 | [+0.2805, +0.3571] | <0.001 | yes |
| **LLM vs TF-IDF** | **−0.3200** | **[−0.3900, −0.2500]** | **<0.001** | **yes** |

### The headline is a negative result

**A 7-billion-parameter instruction-tuned model, prompted with the full label
menu, loses to a linear model on TF-IDF features by 32 accuracy points on matched
rows, and takes roughly 9,300 times longer per tender** (2,242.89 ms against
0.24 ms).

The interval on that difference is [−0.39, −0.25]. It does not come close to
covering zero.

This is worth publishing precisely because it is the opposite of the default
assumption. The LLM is not being used badly: it sees the same text, the same
label set, and a temperature of zero. It is simply the wrong tool for a
short-text classification task where 2,874 labelled examples exist. The
supervised model learns the vocabulary of Chilean procurement; the LLM has to
reason about it from a prompt, and eight capitalised abbreviated words are not
much to reason from.

**What this does not show.** That LLMs are bad at classification in general, that
a larger model would also lose, or that few-shot prompting or fine-tuning would
not close the gap. Those are separate experiments, listed in section 6.

---

## 4. Where the remaining errors are

Most frequent confusions for the TF-IDF model:

| True | Predicted | Count | Share of that class |
|---|---|---:|---:|
| other | 80 (management services) | 10 | 7% |
| 42 (medical equipment) | 85 (health services) | 8 | 9% |
| other | 42 | 8 | 6% |
| other | 43 (IT) | 8 | 6% |
| other | 72 (construction services) | 8 | 6% |
| 80 | 85 | 7 | 6% |

Two patterns, and they are different problems.

**Goods versus the service that delivers them.** 42 → 85 and 80 → 85 confuse
buying a medical device with buying a medical service. A human buyer would
hesitate over some of these too; the boundary is in the taxonomy, not in the
words.

**The `other` class is the largest single source of error.** That is expected and
is a consequence of decision 2.4: `other` is not a category, it is 22 rare
segments in a bag, so it has no coherent vocabulary to learn. Reporting accuracy
without saying this would overstate how much of the error is genuine confusion.

---

## 5. Do examples close the gap?

The zero-shot result invited an obvious objection: the LLM was asked to classify
Chilean procurement jargon with no examples of it, while the linear model had
2,874. An earlier version of this document listed that as untested.
It is now tested.

Four arms, all scored on the same seeded 200-tender subset of the held-out
month, k = 8 examples where applicable.

| Variant | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| TF-IDF + linear SVM | 0.5100 | [0.4400, 0.5800] | 0.4400 | 0.36 |
| LLM zero-shot | 0.3000 | [0.2350, 0.3650] | 0.2823 | 2,249 |
| LLM + 8 random examples | 0.2550 | [0.1950, 0.3200] | 0.2532 | 2,245 |
| **LLM + 8 retrieved examples** | **0.4700** | **[0.4000, 0.5400]** | 0.3852 | 2,362 |
| LLM + DSPy demonstrations | 0.4200 | [0.3549, 0.4900] | **0.3944** | **412** |

Paired bootstrap, 5,000 resamples:

| Comparison | Δ accuracy | 95% CI | p | Significant |
|---|---:|:--|---:|:--:|
| Random examples vs zero-shot | −0.0450 | [−0.0950, +0.0050] | 0.086 | **no** |
| **Retrieved vs random examples** | **+0.2150** | **[+0.1400, +0.2900]** | <0.001 | **yes** |
| Retrieved examples vs TF-IDF | −0.0400 | [−0.1150, +0.0350] | 0.314 | **no** |
| DSPy vs retrieved examples | −0.0500 | [−0.1300, +0.0300] | 0.237 | **no** |

### 5.1 Examples do not help. *Relevant* examples do.

This is why the random arm exists, and it is the finding that would have been
missed without it.

Eight random training examples changed nothing (−0.045, p = 0.086); if anything
they hurt slightly. Eight examples retrieved by similarity gained **21.5
accuracy points** over the same number of random ones, and that interval is
nowhere near zero.

A great many "few-shot improved our results" claims do not run this control, and
without it the two explanations, "the model learned the task from examples" and
"the model finally saw the output format", are indistinguishable. Here the
format explanation is ruled out: both arms showed the format, and only the
relevant one helped.

### 5.2 The gap closes, and the earlier headline needs qualifying

Retrieval few-shot sits 4 accuracy points below the linear model with an
interval of [−0.115, +0.035], which covers zero. **On this test set the two are
statistically indistinguishable.**

That does not overturn section 3, it qualifies it. The honest summary of the
whole project is now: a zero-shot LLM loses badly to a linear model on this
task, and a retrieval-augmented one matches it while remaining roughly 6,400
times slower per tender. Matching accuracy at four orders of magnitude more
compute is not a reason to deploy it.

### 5.3 A bug in my harness, not in the tool

The first DSPy run scored **0.1050**, below even the zero-shot prompt, and the
natural way to write that up would have been "the optimiser underperforms".

It was my mistake. The DSPy signature received `segmentos_posibles` as
`", ".join(allowed)` — that is, `"10, 15, 20, 25, ..."`, bare numbers with no
key — while every hand-written prompt received `"15: Combustibles y
lubricantes"` and so on. The optimiser was being asked to map Spanish tender
titles onto naked integers.

Supplying the same Spanish glosses the other arms already had:

| | Before | After |
|---|---:|---:|
| Accuracy | 0.1050 | **0.4200** |
| Macro-F1 | 0.0349 | **0.3944** |
| Unusable answers | 26 / 200 | **0 / 200** |

Four times better, from fixing the harness rather than the method. The
comparison had been measuring my plumbing.

The general lesson is the one worth keeping: when a tool underperforms a
baseline by a wide margin, the first hypothesis should be that it was handed a
worse problem, not that it is worse. Checking cost 15 minutes; publishing the
original number would have been a false claim about a real library.

### 5.4 DSPy matches retrieval at a sixth of the latency

DSPy's demonstrations are chosen once at compile time (20 s, 8 kept) and then
fixed, so its prompt is short and constant. Retrieval rebuilds a 910-token
prompt for every tender.

The accuracies are indistinguishable (−0.05, p = 0.237) and DSPy has the better
macro-F1 (0.3944 against 0.3852), but it answers in **412 ms against 2,362 ms**,
**5.7 times faster**. For a batch job that difference is irrelevant; for an
interactive endpoint it is the whole decision.

Neither is close to the 0.36 ms of the linear model.

---

## 6. The favourite loses too

Every version of this write-up named a fine-tuned Spanish encoder as the honest
favourite and left it in the limitations section. That is a comfortable place to
keep a prediction, because an untested favourite can never be wrong.

BETO (`dccuchile/bert-base-spanish-wwm-cased`, 110 M parameters) fine-tuned on
the same 2,874 training tenders, scored on the same 952 held-out ones:

| Model | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| TF-IDF + linear SVM | 0.5578 | [0.5263, 0.5893] | 0.5077 | 0.31 |
| BETO fine-tuned | 0.5378 | [0.5063, 0.5704] | 0.4886 | 0.50 |

Paired bootstrap: **−0.0200 accuracy [−0.0494, +0.0095], p = 0.186**, and
−0.0191 macro-F1 [−0.0536, +0.0145], p = 0.270. Both intervals cover zero.

**The prediction was wrong.** A fine-tuned transformer does not beat a linear
model on TF-IDF features here. It draws level, after 80 seconds of GPU training
against 2.5 seconds of CPU fitting.

### 6.1 The first answer was wrong in the other direction

Run at the configured 6 epochs, BETO scored 0.4989 accuracy and 0.3544 macro-F1,
losing significantly on both. Training loss was still 1.865, which is the tell.

| Epochs | Train loss | Accuracy | Macro-F1 | vs TF-IDF (accuracy) |
|---:|---:|---:|---:|:--|
| 6 | 1.865 | 0.4989 | 0.3544 | −0.0588, **significant** |
| **15** | **0.935** | **0.5378** | **0.4886** | −0.0200, **not significant** |
| 30 | 0.512 | 0.5221 | 0.4646 | −0.0357, **significant** |

An inverted U, and both ends are traps. At 6 epochs the model has not converged;
at 30 it has memorised 2,874 examples and generalises worse. Reporting either
end would have produced a confident, wrong headline, in opposite directions.

This is the same mistake as section 5.3 in a different costume. There I nearly
published "DSPy underperforms" when the harness was at fault; here I nearly
published "transformers lose to TF-IDF" when the training schedule was. The
habit worth keeping is to check whether the losing method was given a fair
chance before writing down that it lost.

### 6.2 Why the transformer does not win

Three plausible reasons, in the order I would bet on them:

**Roughly 90 examples per class.** 2,874 tenders across 32 categories. A 110 M
parameter model fine-tuned on that is working near the bottom of its data range,
while a linear model over sparse features is comfortable there.

**The text is not the Spanish BETO was trained on.** Procurement titles are
capitalised, abbreviated fragments: `ADQ. SERV. REPARACION VEH-. FISCAL BT-278`.
WordPiece splits `ADQ.` and `VEH-.` into rubble, whereas the character n-grams in
the TF-IDF model match across exactly those mutilations. Section 2.6 chose them
for that reason, and this is the evidence that the choice mattered.

**The macro-F1 gap is wider than the accuracy gap** at every epoch count, which
points at the tail. With 90 examples per class on average, the rare classes have
a few dozen, and that is where the transformer gives most of its ground away.

### 6.3 The result worth deploying: knowing when to abstain

A classifier forced to answer everything is judged on one number. Allowed to
abstain, it trades coverage for accuracy, and for a procurement desk that trade
is the actual product: auto-file what the model is sure about, route the rest to
a person.

Softmax confidence from the fine-tuned encoder, on the held-out month:

| Confidence >= | Coverage | Tenders kept | Accuracy on kept |
|---:|---:|---:|---:|
| 0.00 | 100.0% | 952 | 0.5378 |
| 0.50 | 85.8% | 817 | 0.5900 |
| 0.70 | 72.2% | 687 | 0.6390 |
| 0.80 | 63.5% | 605 | 0.6678 |
| 0.90 | 50.6% | 482 | 0.7054 |
| 0.95 | 39.7% | 378 | 0.7275 |

Accuracy climbs monotonically with the threshold, so the confidence score does
carry usable signal. But note what it costs: reaching 72.8% accuracy means
answering only 39.7% of tenders, and even then almost three in ten of those are
wrong. **This is not deployable as an auto-filing system.** It is a suggestion
tool that would save a categoriser some typing, and the honest framing is that,
not automation.

The softmax probabilities are also uncalibrated, so these thresholds are
rankings rather than error-rate guarantees. Temperature scaling or conformal
prediction would turn "confidence >= 0.9" into a statement with a coverage
guarantee attached, and neither is implemented.

---

## 7. Quantising for CPU: what the accuracy line does not show

The encoder is the model a procurement desk would actually run, and it would run
on a CPU. The standard way to report that step is one line: "INT8, accuracy
within half a point, four times faster". This section is about what that line
hides.

Microsoft Research's *Accuracy Is Not All You Need* (arXiv 2407.09141) argues
that aggregate accuracy is the wrong metric for compression, because a
compressed model can match its baseline on average while disagreeing with it on
a large minority of individual inputs, roughly half the disagreements in each
direction. Their proposal is to report **flips**, the share of predictions that
change, and the KL divergence between the two output distributions.

Three arms, identical test items in identical order, single CPU thread, batch
size one, measured after warm-up over three repeats:

| Arm | Accuracy | Macro-F1 | Size | p50 | p95 | Flip rate | Mean KL |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 0.5399 | 0.4890 | 440 MB | 116.7 ms | 125.9 ms | — | — |
| ONNX FP32 | 0.5399 | 0.4890 | 440 MB | 90.7 ms | 96.4 ms | **0.00%** | 0.00000 |
| ONNX INT8 | 0.5347 | 0.4838 | **111 MB** | **29.2 ms** | **31.0 ms** | **8.19%** | 0.06567 |

### 7.1 The export is free; the quantisation is not

The middle arm exists to separate two changes that get blamed on one. Exporting
to ONNX changed **nothing**: zero flips, KL of exactly zero, identical accuracy
to four decimals, and 1.29x faster. That is the same model, computed by a better
runtime.

INT8 is where behaviour changes. Four times smaller, four times faster than
PyTorch, and an accuracy drop of 0.0053 with a bootstrap interval of
[−0.0179, +0.0074] that covers zero. Reported conventionally, that reads as
"free".

### 7.2 It is not free: one prediction in twelve changed

**78 of 952 predictions flipped, 8.19%, against a net accuracy change of
−0.53%.** The churn is **15.6 times** the size of the number normally published.

Where the flips went:

| Direction | Count |
|---|---:|
| Wrong → right | 17 |
| Right → wrong | 22 |
| Wrong → a *different* wrong answer | **39** |

Half of all the flips are in that third row, and no aggregate metric can see
them. Accuracy nets the first two rows against each other and is blind to the
third. Macro-F1 barely moves. A user who reported a bad category last week and
sees a different bad category this week has experienced a change the monitoring
dashboard says did not happen.

For this system that is survivable, because it is a suggestion tool and its
suggestions were already wrong 46% of the time. For a system whose output is
cached, audited, or explained to a customer, an 8% silent change rate on a
deployment described as behaviour-preserving is a different matter.

### 7.3 What to actually ship

ONNX INT8: 111 MB instead of 440, 29.2 ms instead of 116.7 on one CPU thread,
and no statistically detectable accuracy cost. It fits in a small container with
no GPU, which is the whole point.

The honest deployment note is that INT8 is **not** the same model. It is a model
that agrees with the original 92% of the time and is equally good on average.
Those are both true and only the second one usually gets written down.

### 7.4 Two things that went wrong on the way

**The exporter.** torch 2.11 defaults to the dynamo ONNX exporter, which ignores
`dynamic_axes` in favour of `dynamic_shapes` and emits a graph that
onnxruntime's quantizer refuses with `Inferred shape and existing shape differ
in dimension 0: (768) vs (32)`. Passing `dynamo=False` selects the TorchScript
exporter, whose graph quantises cleanly.

**The console.** The exporter logs a tick emoji on success, and the default
Windows console encoding cannot represent it, so a successful export raised
`UnicodeEncodeError`. Both fixes are one line each and both cost more time to
diagnose than to write, which is the usual ratio for this kind of thing.

---

## 8. What was not done

- **No GEPA.** DSPy 3.3 ships `dspy.GEPA`, reported to reach a given quality in
  35x fewer rollouts than earlier optimisers. Only `BootstrapFewShot` was run
  here, so nothing is claimed about what a stronger optimiser would do.
- **No calibration.** The encoder's softmax scores rank well but are not
  calibrated, so the coverage table in section 6.3 gives operating points, not
  guaranteed error rates. Temperature scaling or conformal prediction would fix
  that and is the most useful thing left undone.
- **No hierarchical evaluation.** UNSPSC nests segment → family → class →
  commodity. Only the 2-digit segment is predicted. Predicting the 4-digit family
  is harder and more useful, and scoring partial credit down the hierarchy would
  be a better metric than flat accuracy.
- **One seed for the SVM.** LinearSVC is near-deterministic given fixed data, so
  the variance here is small, but it has not been measured.
- **Four months of data.** Drift over a year is not tested, and a single quarter
  cannot show seasonality.
- **The LLM ran on 300 of 952 test tenders**, for time. The confidence interval
  reflects that smaller sample and is correspondingly wider.

---

## 9. Reproducing

```bash
pip install -e ".[dev]"
python -m pytest                       # 35 tests
python scripts/download_data.py        # ~6,000 tenders, about 11 minutes
python scripts/run_experiment.py --skip-llm     # classical models, seconds
python scripts/run_experiment.py --llm-sample 300   # needs Ollama, ~11 minutes
```

The crawl is cached as JSON and loaded into a single DuckDB file, so everything
after the first download runs offline. Every number in this document is in
`reports/metrics_corpus.json` and `reports/metrics_experiment.json`.
