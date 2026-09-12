"""Tests for the two instruments added after the conclusions were audited.

Both exist because a published claim in this repository could not be defended.

`minimum_detectable_effect` -- the write-up said the retrieval arm "closes the
gap to statistical parity" with the linear model, on the strength of an interval
covering zero over 200 tenders. An interval covering zero is not evidence of
equivalence, and on that sample size it could not have been.

`KnnMajorityControl` -- the write-up read a 21.5-point gain from retrieved
examples as the model reasoning over relevant evidence. Nothing in the design
separated that from the retriever having already found the answer.
"""

from __future__ import annotations

import urllib.request

import pytest

from licitaciones.data import Tender
from licitaciones.evaluate import minimum_detectable_effect
from licitaciones.llm_variants import FewShotClassifier, KnnMajorityControl

LLM_CFG = {"base_url": "http://127.0.0.1:11434", "model": "test"}


def tender(title: str, ocid: str = "x", published: str = "2026-01-05") -> Tender:
    return Tender(ocid=ocid, tender_id=ocid, title=title, description="",
                  buyer="B", method="LE", published=published)


class TestMinimumDetectableEffect:
    def test_a_wider_interval_means_a_larger_undetectable_effect(self):
        narrow = minimum_detectable_effect(
            {"difference": 0.0, "ci_lower": -0.01, "ci_upper": 0.01,
             "significant": False})
        wide = minimum_detectable_effect(
            {"difference": 0.0, "ci_lower": -0.20, "ci_upper": 0.20,
             "significant": False})
        assert wide["minimum_detectable_effect"] > narrow["minimum_detectable_effect"]

    def test_the_standard_error_is_read_off_the_interval(self):
        """Half-width / 1.96, so the MDE describes the test that was run."""
        result = minimum_detectable_effect(
            {"difference": 0.0, "ci_lower": -0.098, "ci_upper": 0.098,
             "significant": False})
        assert result["standard_error"] == pytest.approx(0.05, abs=1e-3)

    def test_no_null_result_can_ever_beat_its_own_mde(self):
        """The property that makes an 'is this null informative' flag dishonest.

        A non-significant difference satisfies |delta| < 1.96 * se, and the MDE
        is 2.80 * se, so the MDE always exceeds it. A flag claiming otherwise
        could never fire, and would read as a check that had been performed.
        """
        for half_width in (0.02, 0.05, 0.075, 0.15):
            for offset in (0.0, 0.3, 0.6, 0.9):
                delta = offset * half_width          # always inside the interval
                result = minimum_detectable_effect({
                    "difference": delta,
                    "ci_lower": delta - half_width,
                    "ci_upper": delta + half_width,
                    "significant": False,
                })
                assert result["minimum_detectable_effect"] > abs(delta)

    def test_the_equivalence_bound_is_the_far_end_of_the_interval(self):
        """What an equivalence claim has to clear, not the point estimate."""
        result = minimum_detectable_effect(
            {"difference": -0.04, "ci_lower": -0.115, "ci_upper": 0.035,
             "significant": False})
        assert result["largest_effect_not_excluded"] == pytest.approx(0.115)
        assert result["largest_effect_not_excluded"] > abs(result["observed_difference"])

    def test_the_published_parity_claim_does_not_clear_its_own_bound(self):
        """The comparison this function was written for.

        Retrieval few-shot sat 4.0 accuracy points below the linear model on 200
        tenders, interval [-0.115, +0.035]. The design could not have detected a
        gap under about 11 points, and the data remain consistent with the LLM
        being 11.5 points worse.
        """
        result = minimum_detectable_effect(
            {"difference": -0.04, "ci_lower": -0.115, "ci_upper": 0.035,
             "significant": False})
        assert result["minimum_detectable_effect"] > 0.10
        assert result["largest_effect_not_excluded"] > 0.10


class TestKnnMajorityControl:
    """The control must see exactly what the LLM sees, and call nothing."""

    def _fitted(self, k: int = 3):
        train = [
            tender("SERVICIO DE ASEO Y LIMPIEZA OFICINAS", "a"),
            tender("SERVICIO DE ASEO Y LIMPIEZA EDIFICIO", "b"),
            tender("SERVICIOS DE ASEO INTEGRAL DEPENDENCIAS", "c"),
            tender("ADQUISICION DE COMPUTADORES PORTATILES", "d"),
            tender("ADQUISICION DE COMPUTADORES DE ESCRITORIO", "e"),
            tender("COMPRA DE NOTEBOOK Y MONITORES", "f"),
        ]
        labels = ["76", "76", "76", "43", "43", "43"]
        return train, labels, KnnMajorityControl(LLM_CFG, k=k).fit(train, labels)

    def test_it_returns_the_majority_label_of_the_neighbours(self):
        _, _, control = self._fitted()
        query = tender("SERVICIO DE ASEO Y LIMPIEZA DE OFICINAS MUNICIPALES", "q")
        neighbours = [label for _, label in control._select(query)]
        assert control.predict([query]) == [max(set(neighbours), key=neighbours.count)]

    def test_it_never_opens_a_connection(self):
        """The claim the control exists to support: no language model involved.

        `urlopen` is replaced with something that raises, so a prediction that
        needed the LLM would fail loudly rather than pass quietly.
        """
        _, _, control = self._fitted()

        def explode(*args, **kwargs):
            raise AssertionError("the control called the LLM")

        original = urllib.request.urlopen
        urllib.request.urlopen = explode
        try:
            assert len(control.predict([tender("COMPRA DE NOTEBOOK", "q")])) == 1
        finally:
            urllib.request.urlopen = original

    def test_it_sees_byte_identical_examples_to_the_few_shot_arm(self):
        """Otherwise the two arms differ by more than the language model."""
        train, labels, control = self._fitted()
        few_shot = FewShotClassifier(LLM_CFG, k=3).fit(train, labels)
        query = tender("ADQUISICION DE COMPUTADORES", "q")
        assert control._select(query) == few_shot._select(query)

    def test_a_tie_breaks_toward_the_nearer_neighbour(self):
        """Documented behaviour, pinned so it cannot drift silently."""
        train = [
            tender("COMPRA DE COMPUTADORES PORTATILES PARA OFICINA", "a"),
            tender("SERVICIO DE TRANSPORTE ESCOLAR RURAL", "b"),
        ]
        labels = ["43", "78"]
        control = KnnMajorityControl(LLM_CFG, k=2).fit(train, labels)
        query = tender("COMPRA DE COMPUTADORES PORTATILES OFICINA", "q")
        nearest = control._select(query)[0][1]      # one vote each, so the tie
        assert control.predict([query]) == [nearest]


class TestLatencyIsNotMeasuringTheNameResolver:
    """Why the configured Ollama URL is an IP literal.

    Every LLM latency in the first version of this write-up was measured around
    `urlopen("http://localhost:11434/...")`. On this machine `localhost`
    resolves to `::1` first, Ollama binds IPv4 only, and urllib waits about two
    seconds for the IPv6 connection to be refused before falling back --
    per call, serially. That artefact was 85% of the published per-tender
    latency and none of it was the model.

    The defence is cheap and belongs in the suite rather than in a comment
    somebody will delete.
    """

    def test_the_configured_base_url_is_an_ip_literal(self):
        import ipaddress
        from urllib.parse import urlparse

        from licitaciones.utils import load_config

        host = urlparse(load_config()["llm"]["base_url"]).hostname
        ipaddress.ip_address(host)      # raises ValueError on a hostname

    def test_the_code_defaults_match_the_config(self):
        """A default of "localhost" would reintroduce it whenever a key is absent."""
        import inspect

        from licitaciones import llm_variants, models

        for module in (models, llm_variants):
            assert "localhost:11434" not in inspect.getsource(module), (
                f"{module.__name__} still defaults to the hostname")
