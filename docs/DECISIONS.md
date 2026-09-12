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
| **TF-IDF + linear SVM** | **0.5578** | **[0.5263, 0.5893]** | **0.5077** | 32 | **0.26** |
| Local LLM (qwen2.5:7b) | 0.2794 | [0.2511, 0.3088] | 0.2491 | 30 | 162.19 |

Every row is scored on all 952 held-out tenders. An earlier version ran the LLM
on a seeded 300-tender subset and had to warn that subtracting two rows of the
table was not the comparison; that warning is gone because the subset is gone.
It existed only because a full pass cost 36 minutes, and section 4.4 explains
why it no longer does.

Paired bootstrap, 5,000 resamples:

| Comparison | delta accuracy | 95% CI | p | Significant |
|---|---:|:--|---:|:--:|
| Keywords vs majority | +0.0956 | [+0.0630, +0.1271] | <0.001 | yes |
| TF-IDF vs keywords | +0.3193 | [+0.2805, +0.3571] | <0.001 | yes |
| **LLM vs TF-IDF** | **-0.2784** | **[-0.3172, -0.2395]** | **<0.001** | **yes** |

### The headline is a negative result

**A 7-billion-parameter instruction-tuned model, prompted with the full label
menu, loses to a linear model on TF-IDF features by 27.8 accuracy points, and
takes roughly 620 times longer per tender** (162.19 ms against 0.26 ms).

The interval on that difference is [-0.32, -0.24]. It does not come close to
covering zero.

This is worth publishing precisely because it is the opposite of the default
assumption. The LLM is not being used badly: it sees the same text, the same
label set, and a temperature of zero. It is simply the wrong tool for a
short-text classification task where 2,874 labelled examples exist. The
supervised model learns the vocabulary of Chilean procurement; the LLM has to
reason about it from a prompt, and eight capitalised abbreviated words are not
much to reason from.

### 4.4 Correction: most of the published latency was the name resolver

An earlier version of this section reported **2,242.89 ms per tender** and a
ratio of **9,300 times**. Both were wrong, and not by a little.

The client is `urllib.request.urlopen` against `http://localhost:11434`. On this
machine:

```
getaddrinfo("localhost", 11434) -> ::1, 127.0.0.1
connect ::1        2049.5 ms   ConnectionRefusedError
connect 127.0.0.1    15.3 ms   OK
```

`localhost` resolves to the IPv6 loopback first, Ollama binds IPv4 only, and
Windows takes about two seconds to refuse the IPv6 connection. urllib does not
do happy-eyeballs, so that wait is paid in full, serially, on every single call.

`scripts/measure_llm_overhead.py` isolates it by timing an endpoint that does no
work at all:

| Host | `/api/tags` (no work) | `/api/generate` |
|---|---:|---:|
| `localhost` | 2,055.4 ms | 2,411.9 ms |
| `127.0.0.1` | **1.3 ms** | **272.6 ms** |

**2,054 ms of fixed client overhead, on a request that computes nothing** --
85% of what a generate call over the hostname costs. Against the 2,242.89 ms
this repository published per tender it is 92%, though those two denominators
come from different runs and only the first is measured by one script in one
pass. Either way the model was never the expensive part. The configured
`base_url` is now an IP literal and `tests/test_power_and_controls.py` fails if
it stops being one.

Two things about this are worth more than the corrected number.

**The evidence was already in my own results table.** The DSPy arm was reported
at 412 ms against 2,362 ms for retrieval few-shot, and I explained the gap with
a story about compile-time demonstrations. A longer prompt cannot be six times
faster. That table was telling me the two arms used different HTTP clients, and
I wrote an explanation instead of an experiment. Section 5.4 is the correction.

**It changed what experiments were affordable.** At 2.2 s per tender a full pass
over the test set was 36 minutes, which is why the LLM was scored on a 300-
tender subset and why the comparison carried a caveat about matched rows. At
162 ms it is two and a half minutes, so every LLM number in this document is now
on all 952.

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

Five arms, all scored on the same seeded 200-tender subset of the held-out
month, k = 8 examples where applicable.

| Variant | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| TF-IDF + linear SVM | 0.5100 | [0.4400, 0.5800] | 0.4400 | 0.28 |
| LLM zero-shot | 0.3150 | [0.2500, 0.3800] | 0.2908 | 164.52 |
| LLM + 8 random examples | 0.2750 | [0.2150, 0.3400] | 0.2782 | 170.37 |
| **LLM + 8 retrieved examples** | **0.4650** | **[0.3950, 0.5350]** | 0.3839 | 277.56 |
| **kNN majority vote, no LLM** | **0.4700** | **[0.4000, 0.5400]** | **0.4152** | **18.58** |
| LLM + DSPy demonstrations | 0.4200 | [0.3549, 0.4900] | 0.3944 | 395.36 |

Paired bootstrap, 5,000 resamples. Every null carries the effect the comparison
could have detected, because on 200 tenders that floor is around eleven accuracy
points and several of these differences are smaller than it:

| Comparison | delta accuracy | 95% CI | p | Significant | MDE @80% |
|---|---:|:--|---:|:--:|---:|
| Random examples vs zero-shot | -0.0400 | [-0.0900, +0.0100] | 0.134 | no | 0.0715 |
| **Retrieved vs random examples** | **+0.1900** | **[+0.1150, +0.2650]** | <0.001 | **yes** | 0.1072 |
| **Retrieved vs kNN majority vote** | **-0.0050** | **[-0.0800, +0.0700]** | 0.953 | no | 0.1072 |
| Retrieved examples vs TF-IDF | -0.0450 | [-0.1150, +0.0250] | 0.255 | no | 0.1001 |
| DSPy vs retrieved examples | -0.0450 | [-0.1250, +0.0350] | 0.306 | no | 0.1144 |

### 5.1 Examples do not help. *Relevant* examples do.

This is why the random arm exists, and it is the finding that would have been
missed without it.

Eight random training examples changed nothing (-0.040, p = 0.134); if anything
they hurt slightly. Eight examples retrieved by similarity gained **19.0
accuracy points** over the same number of random ones, and that interval is
nowhere near zero.

A great many "few-shot improved our results" claims do not run this control, and
without it the two explanations, "the model learned the task from examples" and
"the model finally saw the output format", are indistinguishable. Here the
format explanation is ruled out: both arms showed the format, and only the
relevant one helped.

### 5.1.1 The control that was missing: what is the language model adding?

Ruling out the format explanation still leaves two stories for the 19-point
gain, and section 5.1 quietly assumed the flattering one. Either the model
reasons over relevant evidence, or the eight retrieved titles are near
neighbours whose labels already concentrate on the answer, and any procedure
that reads them would score the same.

`KnnMajorityControl` distinguishes them. It inherits the retriever, so it sees
byte-identical examples in identical order, and then never calls the LLM at all
-- it returns the most common label among them.

**It scores 0.4700 against the LLM's 0.4650, with a better macro-F1 (0.4152
against 0.3839), at 18.58 ms against 277.56.** The paired difference is -0.005,
p = 0.953. Whatever the language model contributes over reading the same eight
labels, this test set cannot find it, and the interval rules out anything larger
than 8 accuracy points in either direction.

**But the two are not the same predictor, and the first draft of this section
said they were.** They agree on only 88 of 200 tenders, a 44% agreement rate.
The LLM is right where the control is wrong 29 times; the control is right where
the LLM is wrong 30 times. So the model is *not* copying the majority label off
the prompt -- it does something substantially different, which is worth exactly
nothing here and costs 15 times the latency.

That distinction matters for what this generalises to. "The LLM is a slow
lookup table" would be a clean story and it is false. The defensible claim is
narrower: **on this task the retrieval step carries the result, and the
generation step is a lateral move.**

The honest cost comparison is also not the one the write-up had been making. The
interesting ratio is not LLM-against-linear-model, it is LLM-against-the-cheapest
thing that does the same job. The retriever alone is 66 times the cost of the
linear model; adding the language model makes it 991 times, for nothing
measurable.

### 5.2 The gap does not close to parity, and saying so was an overclaim

Retrieval few-shot sits 4.5 accuracy points below the linear model with an
interval of [-0.115, +0.025], which covers zero. An earlier version of this
document read that as "on this test set the two are statistically
indistinguishable" and then as "it matches".

**A null is not equivalence, and this one carries very little information.** On
200 tenders the standard error of the paired difference is about 3.5 accuracy
points, so this comparison could not have detected a gap smaller than **10.0
points** at 80% power. The observed gap is 4.5. Worse, the interval is still
consistent with the LLM being **11.5 points worse** -- an enormous difference
that this design simply cannot rule out.

`minimum_detectable_effect` in `src/licitaciones/evaluate.py` now annotates every
comparison here with both numbers, and `tests/test_power_and_controls.py` pins
the property that makes the mistake tempting: a non-significant difference is
always smaller than its own MDE, so "not significant" can never, by itself, be
evidence of equality.

What survives is weaker and still worth stating: retrieval closes most of the
zero-shot gap, from 28 accuracy points down to something this test cannot
resolve, at 991 times the cost per tender of the linear model. Whether it truly
matches would need a test set several times larger, and that test has not been
run.

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

### 5.4 Correction: DSPy is slower than retrieval, not six times faster

An earlier version of this section was titled "DSPy matches retrieval at a sixth
of the latency" and explained the gap mechanistically: DSPy's demonstrations are
chosen once at compile time and then fixed, so its prompt is short and constant,
while retrieval rebuilds a 910-token prompt for every tender. **412 ms against
2,362 ms, 5.7 times faster.**

Both halves of that were wrong, for two independent reasons, and the table I
wrote it from contained the refutation.

**A longer prompt cannot be six times faster.** That should have been the end of
the explanation and the start of an experiment. The retrieval arm's 2,362 ms was
not the model — it was the IPv6 loopback timeout of section 4.4, paid by
`urllib` and not by DSPy's LiteLLM client. The two arms were being timed on
different HTTP stacks and only one of them was broken.

**`dspy.LM` caches by default.** Every other arm here uses a bespoke client with
no cache at all. Once the cache was warm, re-running this experiment reported
DSPy at **1 ms per tender** and a compile time of 3 s against the original 20 s,
which is what a dictionary lookup costs and what finally made the problem
obvious. `cache=False` is now passed explicitly, with the reason in the code.

With the loopback fixed and the cache off, both arms measured the same way:

| Variant | Accuracy | Macro-F1 | ms per tender |
|---|---:|---:|---:|
| LLM + 8 retrieved examples | 0.4650 | 0.3839 | **277.56** |
| LLM + DSPy demonstrations | 0.4200 | 0.3944 | **395.36** |

**DSPy is 1.42 times slower**, not 5.7 times faster. The accuracies remain
indistinguishable (-0.045, p = 0.306, and with an MDE of 11.4 points that null
says almost nothing), and DSPy keeps the slightly better macro-F1. Its 18-second
compile is a real one-off cost on top.

Neither is close to the 0.28 ms of the linear model, and both are beaten on
macro-F1 by a majority vote over the retrieved neighbours at 18.58 ms.

The general lesson is not the one section 5.3 drew. There, a losing tool turned
out to have been handed a worse problem. Here a *winning* number turned out to
have been measured on a different instrument, and I wrote a mechanism for it
instead of checking. **An explanation that fits a surprising number is not
evidence for it, and the more satisfying the mechanism, the less likely I was to
go back and measure.**

---

## 6. The favourite loses too

Every version of this write-up named a fine-tuned Spanish encoder as the honest
favourite and left it in the limitations section. That is a comfortable place to
keep a prediction, because an untested favourite can never be wrong.

BETO (`dccuchile/bert-base-spanish-wwm-cased`, 110 M parameters) fine-tuned on
the same 2,874 training tenders, scored on the same 952 held-out ones, at the
epoch count chosen on a validation month:

| Model | Accuracy | 95% CI | Macro-F1 | ms per tender |
|---|---:|:--|---:|---:|
| TF-IDF + linear SVM | 0.5578 | [0.5263, 0.5893] | 0.5077 | 0.26 |
| BETO fine-tuned | 0.5221 | [0.4916, 0.5546] | 0.4646 | 0.44 |

Paired bootstrap: **-0.0357 accuracy [-0.0651, -0.0063], p = 0.0196**, and
-0.0431 macro-F1 [-0.0775, -0.0074], p = 0.0168. Neither interval covers zero.

**The prediction was wrong, and by more than the first version of this document
admitted.** A fine-tuned transformer does not draw level with a linear model on
TF-IDF features here. It loses, significantly, after 144 seconds of GPU training.

### 6.1 Correction: the epoch count was chosen on the test month

An earlier version of this section reported BETO at 0.5378, drawing level with
TF-IDF at p = 0.186, and explained the choice of 15 epochs with a table of 6, 15
and 30 epochs shaped like an inverted U. Every accuracy in that table was
measured on April, the held-out month, and 15 was kept because it scored best
there. **There was no validation split anywhere in this project.**

That is selection on the test set, and it is the most serious methodological
defect this repository has had. The reported number was the maximum of three
draws rather than one draw, so its p-value is not the p-value of the procedure
that produced it. The direction was not neutral either: of the three candidates,
exactly one gave a non-significant result, and that is the one that was
published.

`scripts/select_epochs.py` redoes the choice. The training window holds three
months, so it splits the same way the outer split does:

| Split | Months | Tenders |
|---|---|---:|
| inner train | January + February 2026 | 1,922 |
| validation | March 2026 | 952 |
| test | April 2026 | 952 |

Six epoch counts, three seeds each, selected on March:

| Epochs | Mean validation accuracy | sd across seeds | Mean validation macro-F1 |
|---:|---:|---:|---:|
| 3 | 0.4366 | 0.0058 | 0.2251 |
| 6 | 0.4989 | 0.0095 | 0.3453 |
| 10 | 0.5252 | 0.0157 | 0.4446 |
| 15 | 0.5273 | 0.0074 | 0.4652 |
| 20 | 0.5315 | 0.0032 | 0.4690 |
| **30** | **0.5333** | 0.0034 | 0.4656 |

Three seeds per point is not decoration. The spread across grid points is 0.0375
and the mean spread between seeds at a fixed grid point is 0.0075, a factor of
five, so the curve is measuring epoch count rather than the random seed. Without
that check the whole table could have been noise.

**There is no inverted U on the validation month.** Accuracy rises across the
whole grid and flattens; 30 epochs, the setting the earlier write-up called
memorisation, is the best of the six. Refitting on all 2,874 training tenders at
30 epochs and scoring April once gives the table above: BETO loses by 3.57
accuracy points, significantly.

#### What April would have said, disclosed rather than used

| Epochs | Test accuracy | vs TF-IDF | p | Significant |
|---:|---:|---:|---:|:--|
| 3 | 0.4443 | -0.1134 | 0.0000 | yes |
| 6 | 0.4989 | -0.0588 | 0.0000 | yes |
| 10 | 0.5221 | -0.0357 | 0.0260 | yes |
| 15 | 0.5378 | -0.0200 | 0.1860 | no |
| 20 | 0.5336 | -0.0242 | 0.1080 | no |
| 30 | 0.5221 | -0.0357 | 0.0196 | yes |

This table is published for the same reason the defect is: it shows how much the
conclusion depended on a knob, and it is the evidence for the claim being made.
The selection happened on March and is not revisited here.

Two things fall out of it.

**The two months disagree about where the optimum is.** March rises to 30;
April peaks at 15 and falls on both sides. The inverted U was a property of the
month it was read from. A shape that moves when you change the month is not a
property of the model, and the earlier write-up's mechanism for it -- "at 30 it
has memorised 2,874 examples and generalises worse" -- was an explanation
constructed for an artefact.

**Only two of six settings produce a null, and one of them was published.** The
robust statement, and the one that survives every choice in this section, is
that BETO does not beat TF-IDF at any point in the grid. Its best showing is
-0.0200 and its worst -0.1134. Selection decided whether the loss was reported
as significant, not whether there was one.

#### What this protocol still cannot do

The inner split trains on 1,922 tenders and the final refit on 2,874, so the
epoch count chosen is the best for a smaller corpus than the one it is applied
to. Matching optimisation *steps* rather than epochs would remove that, and is
not done. The direction of the bias is worth stating: more data at a fixed epoch
count means more gradient steps, so if anything 30 epochs over-trains the final
model relative to what March endorsed -- which makes the reported loss the
conservative reading, not the flattering one.

March also cannot really separate 15, 20 and 30; they sit within 0.6 points of
each other, against a seed sd of 0.34 points. What it separates cleanly is that
3 and 6 epochs are worse. A more honest summary of the grid is that anything
from 10 epochs up is indistinguishable, and the test set was used to break that
tie before.

### 6.1.1 The habit this was supposed to be an example of

The earlier version of this section said it was the same mistake as section 5.3
in a different costume: there I nearly published "DSPy underperforms" when my
harness was at fault, here I nearly published "transformers lose to TF-IDF" when
the training schedule was, and the lesson was to check whether a losing method
was given a fair chance.

That lesson is right and it is also how the defect got in. Giving the losing
method a fair chance means tuning it -- and tuning it on the test set is how a
fair chance turns into a borrowed one. The missing half of the rule is that the
retry has to be scored somewhere the verdict is not.

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
| 0.00 | 100.0% | 952 | 0.5221 |
| 0.50 | 96.2% | 916 | 0.5349 |
| 0.70 | 90.4% | 861 | 0.5552 |
| 0.80 | 87.3% | 831 | 0.5692 |
| 0.90 | 80.0% | 762 | 0.5958 |
| 0.95 | 74.5% | 709 | 0.6178 |

#### The monotone column was not the evidence it was offered as

The earlier version of this section argued that because accuracy climbs
monotonically with the threshold, the confidence score carries usable signal.
That argument is much weaker than it looks. The kept sets are **nested** -- every
threshold's rows are a subset of the previous one's -- so successive accuracies
are strongly dependent, and any positive association at all, however slight,
tends to produce a rising column. Eight increasing numbers are close to free.

The quantity that actually answers the question is threshold-free: the
probability that a randomly chosen correct prediction carries more confidence
than a randomly chosen wrong one, which is the AUROC of confidence against
correctness. `confidence_discrimination` in `src/licitaciones/encoder.py`
computes it with a permutation test attached.

| | |
|---|---:|
| AUROC, confidence vs correctness | **0.7401** |
| Permutation p (2,000 shuffles) | **0.0005** |
| Mean confidence when right | 0.9677 |
| Mean confidence when wrong | 0.8844 |

**The claim survives, and now it has been tested rather than asserted.** 0.74 is
a real ranking ability, comfortably clear of the 0.5 null.

#### The validated model is worse at knowing when it is wrong

This is the uncomfortable consequence of section 6.1, and it is worth stating
plainly because it cuts against the correction being a clean win.

At the 15 epochs that were selected on the test month, the abstention table was
better shaped: 39.7% coverage at 0.7275 accuracy under a 0.95 threshold. At the
30 epochs March endorses, the same threshold keeps 74.5% of tenders at 0.6178.
Thirty epochs drives the mean confidence on wrong answers up to 0.8844, so the
thresholds no longer separate much.

More training buys accuracy on paper and costs calibration, and the epoch count
was selected on accuracy alone. If abstention is the product -- and section 6.3
is the argument that it is -- then accuracy was the wrong selection criterion,
and a calibration-aware one would likely have chosen differently.

Switching the criterion now, after seeing which one flatters the abstention
table, would be the same mistake as section 6.1 wearing a third costume. So the
grid stays selected on accuracy, and this sits here as a limitation instead.

#### What is deployable

Not much, and less than the earlier version claimed. Reaching 0.6178 accuracy
means abstaining on a quarter of tenders and still getting almost four in ten of
the kept ones wrong. **This is not an auto-filing system.** It is a suggestion
tool that would save a categoriser some typing, and the honest framing is that,
not automation.

The softmax probabilities are also uncalibrated, so these thresholds are
rankings rather than error-rate guarantees -- which is exactly what an AUROC of
0.74 describes and what the previous paragraph is about. Temperature scaling or
conformal prediction would turn "confidence >= 0.9" into a statement with a
coverage guarantee attached, and neither is implemented.

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
| PyTorch FP32 | 0.5231 | 0.4657 | 439.5 MB | 108.5 ms | 117.7 ms | - | - |
| ONNX FP32 | 0.5231 | 0.4657 | 439.7 MB | 82.2 ms | 93.6 ms | **0.00%** | -0.00000 |
| ONNX INT8 | 0.5179 | **0.4741** | **110.6 MB** | **27.7 ms** | **28.8 ms** | **6.83%** | 0.09224 |

These are not the numbers an earlier version of this section carried. The
encoder is retrained here at the epoch count section 6.1 chose on a validation
month rather than the one chosen on the test month, so every row moved. This
script trains its own encoder rather than reusing `run_encoder.py`'s, and the
two land within 0.001 accuracy of each other (0.5231 against 0.5221), which is
run-to-run variation in fine-tuning rather than disagreement.

### 7.1 The export is free; the quantisation is not

The middle arm exists to separate two changes that get blamed on one. Exporting
to ONNX changed **nothing**: zero flips, KL of zero, identical accuracy and
macro-F1 to four decimals, and 1.32x faster. That is the same model, computed by
a better runtime.

INT8 is where behaviour changes. Four times smaller, 3.9 times faster than
PyTorch, and an accuracy drop of 0.0053 with a bootstrap interval of
[-0.0168, +0.0063] that covers zero. Reported conventionally, that reads as
"free".

### 7.2 It is not free: one prediction in fifteen changed

**65 of 952 predictions flipped, 6.83%, while accuracy moved by half a point.**

Where the flips went:

| Direction | Count |
|---|---:|
| Wrong -> right | 15 |
| Right -> wrong | 20 |
| Wrong -> a *different* wrong answer | **30** |

Nearly half of all the flips are in that third row, and no aggregate metric can
see them. Accuracy nets the first two rows against each other and is blind to
the third. A user who reported a bad category last week and sees a different bad
category this week has experienced a change the monitoring dashboard says did
not happen.

**Correction: the ratio that used to be here was not a quantity.** An earlier
version wrote "the churn is 15.6 times the size of the number normally
published", dividing the flip rate by the net accuracy change. The denominator
is a difference between two nearly equal accuracies whose confidence interval
covers zero. A ratio whose denominator is consistent with zero is unbounded: at
one end of that interval it is about 4, at the other it changes sign, and near
the middle it goes to infinity. The three-significant-figure answer was an
artefact of dividing by a point estimate and pretending it was known.

The contrast survives without the arithmetic, and is stronger stated plainly:
**6.83% of predictions changed while every aggregate metric said nothing had
happened.** That is the finding. Quantifying "how much more" requires a
denominator this experiment does not have.

Macro-F1 makes the point a second way, and in the opposite direction from what
the compression story predicts: INT8's is **higher** than the model it
compresses, 0.4741 against 0.4657. Nothing should be concluded from that either
-- it is one number on one test month, and it moved in a direction no one would
have hypothesised. It is here because it is the kind of result that quietly
disappears from write-ups.

For this system the churn is survivable, because it is a suggestion tool and its
suggestions were already wrong 48% of the time. For a system whose output is
cached, audited, or explained to a customer, a 7% silent change rate on a
deployment described as behaviour-preserving is a different matter.

### 7.3 What to actually ship

ONNX INT8: 110.6 MB instead of 439.5, 27.7 ms instead of 108.5 on one CPU
thread, and no statistically detectable accuracy cost. It fits in a small
container with no GPU, which is the whole point.

The honest deployment note is that INT8 is **not** the same model. It is a model
that agrees with the original 93% of the time and is equally good on average.
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
