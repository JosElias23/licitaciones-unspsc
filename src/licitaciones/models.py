"""Classifiers, from a majority-class stub to a local LLM, behind one interface.

Every model here takes a list of Tenders and returns one UNSPSC segment per
tender. They share an interface so the evaluation harness cannot accidentally
treat them differently, and so the cost comparison is like for like.

The ladder, in increasing order of what it costs to run:

  MajorityClassifier   predicts the most common segment. The number every other
                       model must beat to have done anything at all.
  KeywordClassifier    hand-written Spanish keyword rules. Cheap, interpretable,
                       and on short procurement titles a surprisingly hard bar.
  TfidfLinearClassifier  character and word n-grams into a linear SVM. The
                       workhorse of Spanish short-text classification, and the
                       model most likely to be the right answer in production.
  LlmClassifier        a local instruction-tuned model prompted with the label
                       set. Costs seconds per tender rather than microseconds.

The question the project asks is not "can an LLM do this" but "does it do it
enough better to be worth three orders of magnitude more compute", and that
only means something with the cheap models measured properly first.
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from licitaciones.data import FORBIDDEN_FEATURES, Tender, assert_no_leakage

# The two-digit UNSPSC segments, with the Spanish gloss the buyer would
# recognise. Used to build the LLM prompt and to label figures.
SEGMENT_NAMES: dict[str, str] = {
    "10": "Animales y plantas vivos",
    "11": "Material mineral, textil y vegetal",
    "12": "Productos quimicos",
    "13": "Materiales de caucho y plastico",
    "14": "Papel y productos de papel",
    "15": "Combustibles y lubricantes",
    "20": "Maquinaria de mineria y perforacion",
    "21": "Maquinaria agricola y forestal",
    "22": "Maquinaria de construccion",
    "23": "Maquinaria industrial",
    "24": "Manejo y almacenamiento de materiales",
    "25": "Vehiculos y equipamiento",
    "26": "Generacion y distribucion de energia",
    "27": "Herramientas y maquinaria general",
    "30": "Componentes y suministros de construccion",
    "31": "Componentes de fabricacion",
    "32": "Componentes electronicos",
    "39": "Iluminacion y componentes electricos",
    "40": "Distribucion, climatizacion y fontaneria",
    "41": "Equipos de laboratorio y medicion",
    "42": "Equipamiento medico y de laboratorio",
    "43": "Tecnologias de informacion y telecomunicaciones",
    "44": "Equipos y suministros de oficina",
    "45": "Equipos de imprenta y audiovisuales",
    "46": "Equipos de defensa y seguridad",
    "47": "Equipos y suministros de limpieza",
    "48": "Equipos de servicio y comercio",
    "49": "Equipos deportivos y recreativos",
    "50": "Alimentos y bebidas",
    "51": "Medicamentos y productos farmaceuticos",
    "52": "Articulos de uso domestico",
    "53": "Ropa, calzado y accesorios",
    "54": "Relojes y joyeria",
    "55": "Publicaciones y material impreso",
    "56": "Mobiliario",
    "60": "Instrumentos musicales, arte y educacion",
    "70": "Servicios agricolas y pesqueros",
    "71": "Servicios de mineria y petroleo",
    "72": "Servicios de construccion y mantenimiento",
    "73": "Servicios de produccion industrial",
    "76": "Servicios de limpieza y tratamiento de residuos",
    "77": "Servicios medioambientales",
    "78": "Servicios de transporte y logistica",
    "80": "Servicios de gestion y consultoria",
    "81": "Servicios de ingenieria e investigacion",
    "82": "Servicios editoriales y de diseno",
    "83": "Servicios publicos (agua, luz, telecomunicaciones)",
    "84": "Servicios financieros y de seguros",
    "85": "Servicios de salud",
    "86": "Servicios de educacion y capacitacion",
    "90": "Viajes, alimentacion y entretencion",
    "91": "Servicios personales y domesticos",
    "92": "Seguridad y orden publico",
    "93": "Servicios politicos y sociales",
    "94": "Organizaciones y clubes",
    "95": "Terrenos, edificios y estructuras",
}


def tender_text(tender: Tender) -> str:
    """The only text any model may see.

    Title and description only. The item description is the UNSPSC path written
    out in Spanish, so including it would be reading the label. The buyer name
    is also excluded: some institutions buy almost exclusively in one category,
    and a model keyed on the buyer would score well while having learned nothing
    about the words.
    """
    parts = [tender.title or "", tender.description or ""]
    return " ".join(p for p in parts if p).strip()


def normalise(text: str) -> str:
    """Lowercase, strip accents, collapse whitespace.

    Procurement titles are frequently ALL CAPS and inconsistently accented
    ("ADQUISICION" and "ADQUISICIÓN" are the same word to a buyer and different
    strings to a tokeniser), so folding both is a real gain rather than a habit.
    """
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text).strip()


class BaseClassifier:
    name = "base"

    def fit(self, tenders: list[Tender], labels: list[str]) -> BaseClassifier:
        return self

    def predict(self, tenders: list[Tender]) -> list[str]:
        raise NotImplementedError

    @property
    def cost_note(self) -> str:
        return ""


class MajorityClassifier(BaseClassifier):
    """Always predicts the most frequent training segment."""

    name = "majority"

    def fit(self, tenders, labels):
        self.majority = Counter(labels).most_common(1)[0][0]
        return self

    def predict(self, tenders):
        return [self.majority] * len(tenders)


class KeywordClassifier(BaseClassifier):
    """Hand-written Spanish keyword rules, in priority order.

    Written by reading a sample of titles, not by looking at the label
    distribution. It exists to answer a question the fancier models cannot: how
    much of this task is just spotting the word "combustible" or "capacitacion"?
    """

    name = "keywords"

    RULES: list[tuple[str, str]] = [
        ("15", r"\b(combustible|petroleo|diesel|gasolina|bencina|lubricante|gas licuado)\b"),
        ("50", r"\b(alimento|aliment|viveres|abarrote|racion|carne|frutas|verdura|lacteo|pan)\b"),
        ("51", r"\b(farmac|medicamento|insumo medico|remedio)\b"),
        ("42", r"\b(equipo medico|clinico|hospital|dental|insumos de laboratorio|jeringa)\b"),
        ("43", r"\b(software|licencia|computador|notebook|informatic|servidor|hosting|"
               r"telecomunicacion|internet|datacenter|microsoft|office 365)\b"),
        ("47", r"\b(aseo|limpieza|desinfect|sanitiz|higien)\b"),
        ("53", r"\b(vestuario|ropa|calzado|uniforme|zapato|buzo)\b"),
        ("56", r"\b(mobiliario|mueble|escritorio|silla|estante)\b"),
        ("44", r"\b(articulos de oficina|utiles de oficina|papeleria|toner|resma)\b"),
        ("25", r"\b(vehiculo|camion|automovil|camioneta|bus|neumatico|reparacion veh)\b"),
        ("72", r"\b(obra|construccion|edificacion|pavimento|reparacion de|mejoramiento|"
               r"conservacion|habilitacion)\b"),
        ("78", r"\b(transporte|traslado|flete|logistic|arriendo de bus)\b"),
        ("76", r"\b(residuo|basura|retiro de escombro|tratamiento de agua|limpiafosa)\b"),
        ("86", r"\b(capacitacion|curso|taller|formacion|educacion|docente)\b"),
        ("85", r"\b(atencion de salud|servicio medico|kinesiolog|psicolog|examen)\b"),
        ("80", r"\b(consultoria|asesoria|estudio|gestion|auditoria|evento|produccion de evento)\b"),
        ("81", r"\b(ingenieria|diseno de ingenieria|topograf|geolog|investigacion)\b"),
        ("90", r"\b(alojamiento|hoteler|banqueter|coffee break|alimentacion para)\b"),
        ("92", r"\b(seguridad|guardia|vigilancia|alarma)\b"),
        ("39", r"\b(iluminacion|luminaria|electric|alumbrado publico)\b"),
    ]

    def fit(self, tenders, labels):
        self.fallback = Counter(labels).most_common(1)[0][0]
        self.compiled = [(seg, re.compile(pat)) for seg, pat in self.RULES]
        return self

    def predict(self, tenders):
        out = []
        for t in tenders:
            text = normalise(tender_text(t))
            hit = next((seg for seg, pat in self.compiled if pat.search(text)), None)
            out.append(hit or self.fallback)
        return out


class TfidfLinearClassifier(BaseClassifier):
    """Word and character n-grams into a linear SVM.

    Character n-grams matter more here than they would in ordinary prose.
    Procurement titles are full of abbreviations and typos, "ADQ.", "SERV.",
    "MANTENCION" and "MANTENCIÓN", and character n-grams degrade gracefully
    across all of them where whole-word features simply miss.
    """

    name = "tfidf_svm"

    def __init__(self, tfidf_cfg: dict, svc_cfg: dict, seed: int = 42):
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=tuple(tfidf_cfg.get("ngram_range", (1, 2))),
            max_features=tfidf_cfg.get("max_features", 50000),
            min_df=tfidf_cfg.get("min_df", 2),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        )
        char = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5),
            max_features=tfidf_cfg.get("max_features", 50000),
            min_df=tfidf_cfg.get("min_df", 2), sublinear_tf=True,
        )
        self.pipeline = Pipeline([
            ("features", FeatureUnion([("word", word), ("char", char)])),
            ("svc", LinearSVC(C=svc_cfg.get("C", 1.0),
                              class_weight=svc_cfg.get("class_weight", "balanced"),
                              random_state=seed)),
        ])

    def fit(self, tenders, labels):
        assert_no_leakage(["title", "description"])
        self.pipeline.fit([normalise(tender_text(t)) for t in tenders], labels)
        return self

    def predict(self, tenders):
        return list(self.pipeline.predict([normalise(tender_text(t)) for t in tenders]))

    def top_features(self, segment: str, n: int = 10) -> list[str]:
        """Highest-weighted word features for one class, for the error analysis."""
        svc = self.pipeline.named_steps["svc"]
        union = self.pipeline.named_steps["features"]
        names = union.get_feature_names_out()
        if segment not in list(svc.classes_):
            return []
        idx = list(svc.classes_).index(segment)
        weights = svc.coef_[idx]
        return [str(names[i]).split("__", 1)[-1] for i in np.argsort(weights)[-n:][::-1]]


class LlmClassifier(BaseClassifier):
    """A local instruction-tuned model, prompted with the candidate label set.

    Runs against Ollama on the loopback, so the repository needs no API key and
    costs nothing to reproduce. Latency and token counts are recorded on every
    call, because the interesting comparison against a linear model is not
    accuracy alone but accuracy per unit of compute.

    The prompt lists only the segments present in training. Offering all 57
    UNSPSC segments would invite predictions the evaluation could never score,
    and would make the LLM's task harder than the one the classical models face.
    """

    name = "llm_zero_shot"

    def __init__(self, cfg: dict):
        self.base_url = cfg.get("base_url", "http://127.0.0.1:11434")
        self.model = cfg.get("model", "qwen2.5:7b-instruct")
        self.temperature = cfg.get("temperature", 0.0)
        self.max_tokens = cfg.get("max_tokens", 64)
        self.timeout = cfg.get("timeout_seconds", 120)
        self.latencies: list[float] = []
        self.eval_tokens: list[int] = []
        self.parse_failures = 0

    def fit(self, tenders, labels):
        self.allowed = sorted(set(labels))
        self.fallback = Counter(labels).most_common(1)[0][0]
        self.menu = "\n".join(
            f"{s}: {SEGMENT_NAMES.get(s, 'otro')}" for s in self.allowed
        )
        return self

    def _prompt(self, text: str) -> str:
        return (
            "Eres un clasificador de compras publicas chilenas. Debes asignar el "
            "segmento UNSPSC de dos digitos que corresponde a lo que se esta "
            "comprando.\n\n"
            f"Segmentos posibles:\n{self.menu}\n\n"
            f"Titulo de la licitacion:\n{text}\n\n"
            'Responde SOLO con JSON: {"segmento": "NN"}'
        )

    def _call(self, prompt: str) -> str:
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
        self.eval_tokens.append(int(payload.get("eval_count") or 0))
        return payload.get("response", "")

    def predict(self, tenders):
        out = []
        for i, t in enumerate(tenders, start=1):
            try:
                raw = self._call(self._prompt(tender_text(t)))
                segment = str(json.loads(raw).get("segmento", "")).strip()[:2]
                if segment not in self.allowed:
                    # An unusable answer is a failure of the model, not a reason
                    # to drop the row. Counting them and falling back keeps the
                    # comparison on the same test set for every model.
                    self.parse_failures += 1
                    segment = self.fallback
            except Exception:
                self.parse_failures += 1
                segment = self.fallback
            out.append(segment)
            if i % 50 == 0:
                print(f"[llm] {i}/{len(tenders)}", flush=True)
        return out

    @property
    def cost_note(self) -> str:
        if not self.latencies:
            return ""
        return (f"{np.median(self.latencies):.2f}s median per tender, "
                f"{np.mean(self.eval_tokens):.0f} output tokens, "
                f"{self.parse_failures} unusable answers")


def build_label_set(labels: list[str], min_count: int) -> dict[str, str]:
    """Fold rare segments into 'other'.

    With 40-plus segments and a long tail, a class seen twice in training cannot
    be learned and cannot be evaluated: a single test example makes its recall
    either 0 or 1. Folding them keeps the reported class count honest, and the
    share folded is reported rather than hidden.
    """
    counts = Counter(labels)
    return {s: (s if counts[s] >= min_count else "other") for s in counts}


_ = FORBIDDEN_FEATURES  # re-exported for tests
