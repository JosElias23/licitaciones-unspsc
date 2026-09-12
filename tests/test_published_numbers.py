"""Every number in the prose must exist in reports/, in both languages.

This repository published a churn ratio of "15.6 times" that was computed in a
sentence and stored in no file, a latency of 2,242.89 ms that was mostly the
name resolver, and an encoder accuracy chosen by reading the test month. Two of
those three are the kind of thing a test can catch, and this is that test.

It runs in both directions.

**Forwards**: every figure in `reports/*.json` that the write-up quotes must
appear, formatted the way each language writes numbers -- English 0.5578 and
6.83%, Spanish 0,5578 and 6,83 %. A figure corrected in one file and not the
other fails here.

**Backwards**: a retired claim may still appear, because the corrections quote
what they are correcting, but only inside a correction notice. If "9,345 times"
ever reappears as a plain assertion, this fails.

Skipped when reports/ has not been generated, so a fresh clone can run pytest
before crawling anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

ENGLISH_FILES = ("README.md", "docs/DECISIONS.md")
SPANISH_FILES = ("README.es.md",)
LANGS = ["en", "es"]


# --------------------------------------------------------------- formatting

def en_dec(x: float, places: int = 4) -> str:
    return f"{x:.{places}f}"


def es_dec(x: float, places: int = 4) -> str:
    return f"{x:.{places}f}".replace(".", ",")


def en_pct(x: float, places: int = 2) -> str:
    return f"{x:.{places}f}%"


def es_pct(x: float, places: int = 2) -> str:
    return f"{x:.{places}f}".replace(".", ",") + " %"


# ------------------------------------------------------------------ fixtures

def _load(name: str) -> dict | None:
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@pytest.fixture(scope="module")
def metrics():
    needed = {
        "experiment": "metrics_experiment.json",
        "variants": "metrics_llm_variants.json",
        "encoder": "metrics_encoder.json",
        "quantisation": "metrics_quantisation.json",
        "epochs": "metrics_epoch_selection.json",
    }
    loaded = {key: _load(name) for key, name in needed.items()}
    missing = [needed[k] for k, v in loaded.items() if v is None]
    if missing:
        pytest.skip(f"reports/ incomplete ({', '.join(missing)}); run scripts/ first")
    return loaded


def _read(names) -> str:
    return "\n".join((ROOT / n).read_text(encoding="utf-8") for n in names)


@pytest.fixture(scope="module")
def prose():
    return {
        "en": (_read(ENGLISH_FILES), en_dec, en_pct),
        "es": (_read(SPANISH_FILES), es_dec, es_pct),
    }


# --------------------------------------------------------------------- tests

class TestBothLanguagesExist:
    def test_the_spanish_readme_is_present(self):
        assert (ROOT / "README.es.md").exists()

    @pytest.mark.parametrize("a,b", [("README.md", "README.es.md"),
                                     ("README.es.md", "README.md")])
    def test_each_readme_links_to_the_other(self, a, b):
        assert b in (ROOT / a).read_text(encoding="utf-8"), f"{a} does not link to {b}"


@pytest.mark.parametrize("lang", LANGS)
class TestHeadlineComparison:
    def test_every_model_accuracy_is_quoted(self, metrics, prose, lang):
        text, dec, _ = prose[lang]
        for name, result in metrics["experiment"]["results"].items():
            printed = dec(result["accuracy"])
            assert printed in text, (
                f"[{lang}] {name} accuracy {printed} is in metrics_experiment.json "
                f"but not in the prose"
            )

    def test_the_llm_latency_is_quoted(self, metrics, prose, lang):
        text, dec, _ = prose[lang]
        ms = metrics["experiment"]["results"]["llm_zero_shot"]["ms_per_tender"]
        assert dec(ms, 2) in text, f"[{lang}] LLM latency {dec(ms, 2)} missing"

    def test_the_headline_gap_is_quoted(self, metrics, prose, lang):
        """The LLM-vs-TF-IDF difference, which is the project's headline."""
        text, dec, _ = prose[lang]
        comparisons = metrics["experiment"]["comparisons"]
        key = next(k for k in comparisons if "llm_zero_shot" in k and "accuracy" in k)
        gap = abs(comparisons[key]["difference"])
        assert dec(gap) in text, f"[{lang}] headline gap {dec(gap)} missing"


@pytest.mark.parametrize("lang", LANGS)
class TestFewShotArms:
    def test_every_variant_accuracy_is_quoted(self, metrics, prose, lang):
        text, dec, _ = prose[lang]
        for name, result in metrics["variants"]["results"].items():
            printed = dec(result["accuracy"])
            assert printed in text, f"[{lang}] variant {name} accuracy {printed} missing"

    def test_the_control_is_reported_as_cheaper_and_is(self, metrics, prose, lang):
        """The kNN control's latency must be quoted, and must beat the LLM's."""
        text, dec, _ = prose[lang]
        results = metrics["variants"]["results"]
        control = results["knn_majority_control"]["ms_per_tender"]
        llm = results["llm_fewshot_knn"]["ms_per_tender"]
        assert control < llm, "the control is supposed to be the cheap arm"
        assert dec(control, 2) in text, (
            f"[{lang}] control latency {dec(control, 2)} missing")

    def test_the_agreement_rate_is_quoted(self, metrics, prose, lang):
        """The number that stopped 'the LLM just copies the majority label'."""
        text, _, pct = prose[lang]
        rate = 100 * metrics["variants"]["llm_vs_retriever_agreement"]["agreement_rate"]
        assert pct(rate, 0) in text, f"[{lang}] agreement rate {pct(rate, 0)} missing"

    def test_dspy_is_quoted_as_slower_than_retrieval(self, metrics, prose, lang):
        """The correction: DSPy costs more per tender, not a sixth as much."""
        text, dec, _ = prose[lang]
        results = metrics["variants"]["results"]
        dspy = results["llm_dspy_bootstrap"]["ms_per_tender"]
        knn = results["llm_fewshot_knn"]["ms_per_tender"]
        assert dspy > knn, (
            "if DSPy ever becomes the faster arm, section 5.4 needs rewriting "
            "rather than this assertion relaxing"
        )
        assert dec(dspy, 2) in text, f"[{lang}] DSPy latency {dec(dspy, 2)} missing"


@pytest.mark.parametrize("lang", LANGS)
class TestEncoderAndEpochChoice:
    def test_the_encoder_accuracy_is_quoted(self, metrics, prose, lang):
        text, dec, _ = prose[lang]
        for name, result in metrics["encoder"]["results"].items():
            assert dec(result["accuracy"]) in text, (
                f"[{lang}] encoder arm {name} accuracy missing")

    def test_the_chosen_epoch_count_is_stated(self, metrics, prose, lang):
        text, _, _ = prose[lang]
        chosen = metrics["epochs"]["chosen_epochs"]
        assert str(chosen) in text, f"[{lang}] chosen epoch count {chosen} missing"

    def test_the_confidence_auroc_is_quoted(self, metrics, prose, lang):
        """Replaces the monotone-curve argument, so it must be visible."""
        text, dec, _ = prose[lang]
        auroc = metrics["encoder"]["confidence_discrimination"]["auroc"]
        assert dec(auroc) in text, f"[{lang}] AUROC {dec(auroc)} missing"


@pytest.mark.parametrize("lang", LANGS)
class TestQuantisation:
    def test_every_arm_accuracy_is_quoted(self, metrics, prose, lang):
        text, dec, _ = prose[lang]
        for arm in metrics["quantisation"]["arms"]:
            assert dec(arm["accuracy"]) in text, (
                f"[{lang}] {arm['arm']} accuracy {dec(arm['accuracy'])} missing")

    def test_the_flip_rate_is_quoted(self, metrics, prose, lang):
        text, _, pct = prose[lang]
        int8 = next(a for a in metrics["quantisation"]["arms"] if a["arm"] == "onnx_int8")
        assert pct(100 * int8["flip_rate"]) in text, (
            f"[{lang}] flip rate {pct(100 * int8['flip_rate'])} missing")


class TestInternalConsistency:
    """Properties of the data itself, independent of how it is written up."""

    def test_the_epoch_count_in_the_config_is_the_one_validation_chose(self, metrics):
        """The defect this whole protocol exists to prevent, pinned."""
        import yaml
        config = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(
            encoding="utf-8"))
        assert config["encoder"]["num_epochs"] == metrics["epochs"]["chosen_epochs"], (
            "configs/default.yaml disagrees with reports/metrics_epoch_selection.json; "
            "the epoch count must come from the validation month"
        )

    def test_the_flip_directions_add_up_to_the_flip_rate(self, metrics):
        """The three-row breakdown must be the same event as the headline rate."""
        int8 = next(a for a in metrics["quantisation"]["arms"] if a["arm"] == "onnx_int8")
        total = (int8["flips_right_to_wrong"] + int8["flips_wrong_to_right"]
                 + int8["flips_wrong_to_wrong"])
        n = metrics["quantisation"]["n_test"]
        assert total / n == pytest.approx(int8["flip_rate"], abs=1e-4)

    def test_validation_never_saw_the_test_month(self, metrics):
        """The inner cutoff must fall strictly before the outer one."""
        protocol = metrics["epochs"]["protocol"]
        assert protocol["inner_cutoff"] < protocol["outer_cutoff"]
        assert protocol["n_validation"] > 0 and protocol["n_test"] > 0


class TestRetiredClaims:
    """A corrected number may be quoted, but only by the correction.

    Each of these was published as fact and is now wrong. They still appear,
    because a correction that does not say what it is correcting is not much of
    a correction. What must never happen is one of them drifting back into a
    plain sentence.
    """

    RETIRED = ("9,345", "2,242.89", "15.6 times", "5.7 times faster",
               "statistically indistinguishable", "9.345", "15,6 veces")
    MARKERS = ("correction", "corrección", "earlier version", "versión anterior",
               "used to", "an earlier", "solía", "wrong", "not a quantity")

    @staticmethod
    def _enclosing_heading(lines, i: int) -> str:
        """The nearest heading above line i, which is the claim's context."""
        for j in range(i, -1, -1):
            if lines[j].lstrip().startswith("#"):
                return lines[j]
        return ""

    @pytest.mark.parametrize("claim", RETIRED)
    def test_a_retired_claim_appears_only_inside_a_correction(self, claim):
        for name in ENGLISH_FILES + SPANISH_FILES:
            lines = (ROOT / name).read_text(encoding="utf-8").splitlines()
            for i, line in enumerate(lines):
                if claim not in line:
                    continue
                # Either the sentences around it say it is being corrected, or
                # the section it sits in is itself a correction.
                context = " ".join(lines[max(0, i - 6):i + 4])
                context += " " + self._enclosing_heading(lines, i)
                assert any(m in context.lower() for m in self.MARKERS), (
                    f"{name}:{i + 1} states the retired claim {claim!r} without "
                    f"a correction anywhere near it: {line.strip()}"
                )
