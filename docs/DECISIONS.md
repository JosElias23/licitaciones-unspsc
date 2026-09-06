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
not close the gap. Those are separate experiments, listed in
[Not done](#5-what-was-not-done).

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

## 5. What was not done

- **No few-shot prompting or DSPy optimisation of the LLM prompt.** The zero-shot
  gap is 32 points; closing it with examples is plausible and is the obvious next
  experiment, but it has not been run, so nothing is claimed about it.
- **No fine-tuned encoder.** A Spanish BERT fine-tuned on 2,874 examples would
  very likely beat both, and is the honest favourite for this task.
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

## 6. Reproducing

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
