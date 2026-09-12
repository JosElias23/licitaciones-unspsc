"""Does giving the LLM examples close the 32-point gap to a linear model?

The zero-shot result was decisive and one-sided, which raises the obvious
objection: the model was asked to classify Chilean procurement jargon with no
examples of it, while the linear model had 2,874. That is not a fair fight, and
the README said so before this module existed.

Three ways of supplying examples, in increasing order of machinery:

`FewShotClassifier`
    Retrieve the k most similar training titles by TF-IDF cosine and put them in
    the prompt. Nearest-neighbour prompting: cheap, transparent, and the honest
    thing to try before reaching for a framework.

`RandomFewShotClassifier`
    The same k examples, chosen at random rather than by similarity. Its only
    job is to separate "examples help" from "*relevant* examples help", which
    are different claims and are usually conflated.

`DspyClassifier`
    DSPy chooses the demonstrations itself, by running the model over training
    data and keeping the ones it gets right. This is the automatic
    prompt-optimisation path, and the comparison against retrieval few-shot is
    the question worth answering: does the optimiser earn its extra calls?

Every variant sees only training examples. Retrieval is fitted on the training
split alone, so a test tender can never be its own neighbour.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from licitaciones.data import Tender
from licitaciones.models import SEGMENT_NAMES, BaseClassifier, normalise, tender_text


class OllamaChat:
    """Minimal Ollama client that records latency and token counts per call."""

    def __init__(self, cfg: dict):
        self.base_url = cfg.get("base_url", "http://127.0.0.1:11434")
        self.model = cfg.get("model", "qwen2.5:7b-instruct")
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 64)
        self.timeout = cfg.get("timeout_seconds", 120)
        self.latencies: list[float] = []
        self.prompt_tokens: list[int] = []
        self.eval_tokens: list[int] = []

    def generate(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "prompt": prompt, "format": "json", "stream": False,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
        }).encode()
        request = urllib.request.Request(
            f"{self.base_url}/api/generate", data=body,
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
        self.latencies.append(time.perf_counter() - started)
        self.prompt_tokens.append(int(payload.get("prompt_eval_count") or 0))
        self.eval_tokens.append(int(payload.get("eval_count") or 0))
        return payload.get("response", "")

    def stats(self) -> dict:
        if not self.latencies:
            return {}
        return {
            "calls": len(self.latencies),
            "median_latency_s": round(float(np.median(self.latencies)), 3),
            "p95_latency_s": round(float(np.percentile(self.latencies, 95)), 3),
            "mean_prompt_tokens": round(float(np.mean(self.prompt_tokens)), 1),
            "mean_output_tokens": round(float(np.mean(self.eval_tokens)), 1),
            "total_tokens": int(sum(self.prompt_tokens) + sum(self.eval_tokens)),
        }


class _PromptedClassifier(BaseClassifier):
    """Shared prompt assembly, parsing and failure accounting."""

    def __init__(self, cfg: dict, k: int = 8):
        self.client = OllamaChat(cfg)
        self.k = k
        self.parse_failures = 0
        self.out_of_menu = 0

    def _menu(self) -> str:
        return "\n".join(f"{s}: {SEGMENT_NAMES.get(s, 'otro')}" for s in self.allowed)

    def _examples_block(self, examples: list[tuple[str, str]]) -> str:
        if not examples:
            return ""
        rows = "\n".join(f'- "{t}" -> {lbl}' for t, lbl in examples)
        return f"\nEjemplos de licitaciones ya clasificadas:\n{rows}\n"

    def _prompt(self, text: str, examples: list[tuple[str, str]]) -> str:
        return (
            "Eres un clasificador de compras publicas chilenas. Debes asignar el "
            "segmento UNSPSC de dos digitos que corresponde a lo que se esta "
            "comprando.\n\n"
            f"Segmentos posibles:\n{self._menu()}\n"
            f"{self._examples_block(examples)}\n"
            f"Titulo de la licitacion:\n{text}\n\n"
            'Responde SOLO con JSON: {"segmento": "NN"}'
        )

    def _parse(self, raw: str) -> str:
        try:
            segment = str(json.loads(raw).get("segmento", "")).strip()[:2]
        except Exception:
            self.parse_failures += 1
            return self.fallback
        if segment not in self.allowed:
            self.out_of_menu += 1
            return self.fallback
        return segment

    def _select(self, tender: Tender) -> list[tuple[str, str]]:
        raise NotImplementedError

    def predict(self, tenders: list[Tender]) -> list[str]:
        out = []
        for i, t in enumerate(tenders, start=1):
            raw = self.client.generate(self._prompt(tender_text(t), self._select(t)))
            out.append(self._parse(raw))
            if i % 50 == 0:
                print(f"[{self.name}] {i}/{len(tenders)}", flush=True)
        return out

    @property
    def cost_note(self) -> str:
        s = self.client.stats()
        if not s:
            return ""
        return (f"{s['median_latency_s']}s median, {s['mean_prompt_tokens']:.0f} prompt "
                f"tokens, {self.parse_failures} unparseable, "
                f"{self.out_of_menu} out-of-menu")


class FewShotClassifier(_PromptedClassifier):
    """k nearest training titles by TF-IDF cosine, used as in-prompt examples."""

    name = "llm_fewshot_knn"

    def fit(self, tenders: list[Tender], labels: list[str]):
        self.allowed = sorted(set(labels))
        self.fallback = Counter(labels).most_common(1)[0][0]
        self.train_texts = [tender_text(t) for t in tenders]
        self.train_labels = labels
        # Fitted on training text only. A test tender must never be able to
        # retrieve itself, which is why the index is built here and not later.
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                          min_df=2, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(
            [normalise(t) for t in self.train_texts]
        )
        return self

    def _select(self, tender):
        query = self.vectorizer.transform([normalise(tender_text(tender))])
        sims = cosine_similarity(query, self.matrix)[0]
        top = np.argsort(sims)[-self.k:][::-1]
        return [(self.train_texts[i][:110], self.train_labels[i]) for i in top]


class KnnMajorityControl(FewShotClassifier):
    """The retrieval arm with the model taken out: majority vote over the same k.

    This is the control the few-shot result needed and did not have. When
    retrieval few-shot beat random few-shot by 21.5 points, the write-up read
    that as the model reasoning over relevant evidence. There is a duller
    explanation that fits the same number: the eight retrieved titles are near
    neighbours of the query, so their labels already concentrate on the right
    answer, and a model that simply copies the most common label it was shown
    would score the same.

    Those two stories are distinguishable, and the instrument is this class. It
    inherits the retriever, so it sees byte-identical examples in identical
    order, and then never calls the LLM at all -- it returns the most common
    label among them. Whatever it scores is the share of the retrieval arm's
    accuracy that is the *retriever's*, not the language model's.

    Ties break toward the nearer neighbour: `_select` returns examples in
    descending similarity and `Counter.most_common` is stable on insertion
    order, so the first-seen label wins a tie.
    """

    name = "knn_majority_control"

    def predict(self, tenders: list[Tender]) -> list[str]:
        out = []
        started = time.perf_counter()
        for tender in tenders:
            labels = [label for _, label in self._select(tender)]
            out.append(Counter(labels).most_common(1)[0][0] if labels else self.fallback)
        self.predict_seconds = time.perf_counter() - started
        self.n_predicted = len(tenders)
        return out

    @property
    def cost_note(self) -> str:
        per = 1000 * getattr(self, "predict_seconds", 0.0) / max(getattr(self, "n_predicted", 0), 1)
        return f"no LLM call; retrieval only, {per:.2f} ms per tender"


class RandomFewShotClassifier(_PromptedClassifier):
    """k training examples chosen at random, fixed across all test tenders.

    The control for FewShotClassifier. If random examples help as much as
    retrieved ones, the gain is about showing the output format, not about
    supplying relevant evidence, and those are very different conclusions.
    """

    name = "llm_fewshot_random"

    def __init__(self, cfg: dict, k: int = 8, seed: int = 42):
        super().__init__(cfg, k)
        self.seed = seed

    def fit(self, tenders: list[Tender], labels: list[str]):
        self.allowed = sorted(set(labels))
        self.fallback = Counter(labels).most_common(1)[0][0]
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(tenders), size=min(self.k, len(tenders)), replace=False)
        self.examples = [(tender_text(tenders[i])[:110], labels[i]) for i in idx]
        return self

    def _select(self, tender):
        return self.examples


class DspyClassifier(BaseClassifier):
    """DSPy picks the demonstrations, by keeping the ones the model gets right.

    BootstrapFewShot runs the model over training examples and retains those it
    answers correctly, so the demonstrations are selected by the model's own
    behaviour rather than by similarity or by hand. The comparison that matters
    is against retrieval few-shot: the optimiser makes extra calls during
    compilation, and this measures whether they buy anything.
    """

    name = "llm_dspy_bootstrap"

    def __init__(self, cfg: dict, max_demos: int = 8, train_subset: int = 200,
                 seed: int = 42):
        self.cfg = cfg
        self.max_demos = max_demos
        self.train_subset = train_subset
        self.seed = seed
        self.compile_seconds = 0.0
        self.parse_failures = 0

    def fit(self, tenders: list[Tender], labels: list[str]):
        import dspy

        self.allowed = sorted(set(labels))
        self.fallback = Counter(labels).most_common(1)[0][0]

        # The menu must carry the Spanish gloss, not bare codes.
        #
        # The first version of this passed ", ".join(self.allowed), i.e.
        # "10, 15, 20, 25, ...", asking the model to map Spanish procurement
        # titles onto numbers with no key. It scored 0.105 against 0.315 for the
        # hand-written zero-shot prompt, and the natural reading of that would
        # have been "DSPy underperforms". It was not DSPy's doing: the optimiser
        # was handed a strictly worse prompt than the baseline it was being
        # compared against, so the comparison measured the harness rather than
        # the method.
        menu = "\n".join(f"{s}: {SEGMENT_NAMES.get(s, 'otro')}" for s in self.allowed)

        lm = dspy.LM(
            f"ollama_chat/{self.cfg['model']}",
            api_base=self.cfg.get("base_url", "http://127.0.0.1:11434"),
            api_key="", temperature=self.cfg.get("temperature", 0.0),
            max_tokens=self.cfg.get("max_tokens", 64),
            # `dspy.LM` caches by default. Every other arm here uses a bespoke
            # client with no cache at all, so leaving it on times a dictionary
            # lookup against five arms timing a 7B model. It is the difference
            # between 1 ms and a real answer, and the earlier write-up reported
            # DSPy as "a sixth of the latency" on a number measured this way.
            cache=False,
        )
        dspy.configure(lm=lm)

        class ClasificarLicitacion(dspy.Signature):
            """Asigna el segmento UNSPSC de dos digitos a una licitacion chilena."""

            titulo: str = dspy.InputField(desc="Titulo escrito por el comprador")
            segmentos_posibles: str = dspy.InputField(
                desc="Codigos permitidos, uno por linea, con su descripcion"
            )
            segmento: str = dspy.OutputField(desc="Codigo de dos digitos")

        self.menu = menu
        self.program = dspy.Predict(ClasificarLicitacion)

        # A subset of training data. Compilation costs one model call per
        # candidate example, and the point of this comparison is what the
        # optimiser buys, not how long it can be left to run.
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(len(tenders), size=min(self.train_subset, len(tenders)),
                         replace=False)
        trainset = [
            dspy.Example(titulo=tender_text(tenders[i]), segmentos_posibles=menu,
                         segmento=labels[i]).with_inputs("titulo", "segmentos_posibles")
            for i in idx
        ]

        def exact(example, prediction, trace=None):
            return str(getattr(prediction, "segmento", "")).strip()[:2] == example.segmento

        started = time.perf_counter()
        optimiser = dspy.BootstrapFewShot(
            metric=exact, max_bootstrapped_demos=self.max_demos,
            max_labeled_demos=self.max_demos, max_rounds=1,
        )
        self.compiled = optimiser.compile(self.program, trainset=trainset)
        self.compile_seconds = time.perf_counter() - started
        self.n_demos = len(getattr(self.compiled, "demos", []) or [])
        return self

    def predict(self, tenders: list[Tender]) -> list[str]:
        out = []
        for i, t in enumerate(tenders, start=1):
            try:
                prediction = self.compiled(titulo=tender_text(t),
                                           segmentos_posibles=self.menu)
                segment = re.sub(r"\D", "", str(prediction.segmento))[:2]
                if segment not in self.allowed:
                    self.parse_failures += 1
                    segment = self.fallback
            except Exception:
                self.parse_failures += 1
                segment = self.fallback
            out.append(segment)
            if i % 50 == 0:
                print(f"[{self.name}] {i}/{len(tenders)}", flush=True)
        return out

    @property
    def cost_note(self) -> str:
        return (f"compiled in {self.compile_seconds:.0f}s, "
                f"{getattr(self, 'n_demos', 0)} demonstrations kept, "
                f"{self.parse_failures} unusable answers")
