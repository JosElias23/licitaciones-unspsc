"""Integrity tests for the corpus and, above all, for the leakage guard.

The item description in this dataset is the UNSPSC taxonomy path written out in
Spanish. A model given it scores near-perfectly and has learned nothing. That is
the one mistake capable of invalidating every number in the project, so it gets
the most tests.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from licitaciones.data import (
    FORBIDDEN_FEATURES,
    LeakageError,
    Tender,
    assert_no_leakage,
    corpus_stats,
    parse_tender,
    temporal_split,
)
from licitaciones.models import normalise, tender_text

CACHE = Path(__file__).resolve().parents[1] / "data" / "raw"


def make_tender(**kwargs) -> Tender:
    base = dict(
        ocid="ocds-x-1", tender_id="1-1-LE26", title="ADQ. INSUMOS DE ASEO",
        description="Compra de insumos", buyer="MUNICIPALIDAD X",
        method="Licitacion Publica", published="2026-03-01T10:00:00Z",
        unspsc_codes=["47131803"], n_items=1,
    )
    base.update(kwargs)
    return Tender(**base)


class TestLeakageGuard:
    """The single most important guarantee in this repository."""

    def test_item_description_is_rejected(self):
        with pytest.raises(LeakageError):
            assert_no_leakage(["title", "item_description"])

    @pytest.mark.parametrize("name", sorted(FORBIDDEN_FEATURES))
    def test_every_forbidden_feature_is_rejected(self, name):
        with pytest.raises(LeakageError):
            assert_no_leakage(["title", name])

    def test_allowed_features_pass(self):
        # Must not raise. The guard has to permit legitimate features or it
        # would simply block all training.
        assert assert_no_leakage(["title", "description", "title_char_ngrams"]) is None

    def test_the_error_names_the_offending_feature(self):
        with pytest.raises(LeakageError, match="unspsc_segment"):
            assert_no_leakage(["unspsc_segment"])

    def test_model_input_never_includes_the_label_path(self):
        """tender_text is the only text any model sees; it must exclude the label.

        The UNSPSC path contains slash-separated taxonomy names. If one ever
        appeared in the model's input, the task would collapse into a lookup.
        """
        t = make_tender(
            title="ADQ. INSUMOS DE ASEO",
            description="Compra de insumos para el departamento",
        )
        text = tender_text(t)
        assert "Desinfectantes" not in text
        assert text.count("/") == 0

    def test_model_input_excludes_the_buyer(self):
        """Some institutions buy almost exclusively in one category.

        Keying on the buyer would score well while learning nothing about the
        words, so the buyer is deliberately not part of the model input.
        """
        t = make_tender(buyer="SERVICIO DE SALUD METROPOLITANO")
        assert "SERVICIO DE SALUD" not in tender_text(t)


class TestTender:
    def test_segment_is_the_first_two_digits(self):
        assert make_tender(unspsc_codes=["47131803"]).segments == ["47"]

    def test_dominant_segment_is_the_modal_one(self):
        t = make_tender(unspsc_codes=["50100000", "50200000", "43230000"])
        assert t.dominant_segment == "50"

    def test_single_category_detection(self):
        assert make_tender(unspsc_codes=["50100000", "50200000"]).is_single_category
        assert not make_tender(unspsc_codes=["50100000", "43230000"]).is_single_category

    def test_no_codes_yields_no_label(self):
        assert make_tender(unspsc_codes=[]).dominant_segment is None

    def test_one_label_per_tender_not_per_item(self):
        """A single tender must contribute exactly one training row.

        Counting per line item let one food-supply tender with hundreds of lines
        supply 62% of a 444-item sample. Per-tender labelling took the majority
        class from 61.7% down to 10.1%.
        """
        big = make_tender(unspsc_codes=["50100000"] * 300 + ["43230000"])
        assert big.dominant_segment == "50"
        assert len([big.dominant_segment]) == 1


class TestParsing:
    def test_payload_without_releases_returns_none(self):
        assert parse_tender({"releases": []}) is None

    def test_payload_without_unspsc_returns_none(self):
        payload = {"releases": [{"ocid": "x", "tender": {"items": [
            {"classification": {"scheme": "OTHER", "id": "1"}}]}}]}
        assert parse_tender(payload) is None

    def test_only_unspsc_codes_are_kept(self):
        payload = {"releases": [{"ocid": "x", "date": "2026-03-01", "tender": {
            "id": "1", "title": "t", "items": [
                {"classification": {"scheme": "UNSPSC", "id": "50100000"}},
                {"classification": {"scheme": "CPV", "id": "99999999"}},
            ]}}]}
        parsed = parse_tender(payload)
        assert parsed.unspsc_codes == ["50100000"]


class TestTemporalSplit:
    def test_split_is_by_date_not_at_random(self):
        early = [make_tender(published="2026-01-15T00:00:00Z") for _ in range(3)]
        late = [make_tender(published="2026-05-15T00:00:00Z") for _ in range(2)]
        train, test = temporal_split(early + late, date(2026, 4, 1))
        assert len(train) == 3
        assert len(test) == 2

    def test_no_tender_appears_on_both_sides(self):
        tenders = [make_tender(ocid=f"o{i}", published=f"2026-0{1 + i % 4}-10T00:00:00Z")
                   for i in range(20)]
        train, test = temporal_split(tenders, date(2026, 4, 1))
        assert not {t.ocid for t in train} & {t.ocid for t in test}

    def test_every_training_tender_predates_every_test_tender(self):
        """The property that makes the split meaningful.

        A random split would let the model see later phrasing while predicting
        earlier tenders, which is not the situation it faces in use.
        """
        tenders = [make_tender(ocid=f"o{i}", published=f"2026-0{1 + i % 5}-10T00:00:00Z")
                   for i in range(30)]
        train, test = temporal_split(tenders, date(2026, 4, 1))
        if train and test:
            assert max(t.published for t in train) < min(t.published for t in test)


class TestNormalisation:
    def test_accents_are_folded(self):
        assert normalise("MANTENCIÓN") == normalise("MANTENCION")

    def test_case_is_folded(self):
        assert normalise("ADQUISICION") == normalise("adquisicion")

    def test_whitespace_is_collapsed(self):
        assert normalise("  ADQ.   SERV.  ") == "adq. serv."


@pytest.mark.skipif(not list(CACHE.glob("tenders_*.json")),
                    reason="No cached corpus; run scripts/download_data.py")
class TestRealCorpus:
    @pytest.fixture(scope="class")
    def tenders(self):
        rows = []
        for f in CACHE.glob("tenders_*.json"):
            rows += [Tender(**r) for r in json.loads(f.read_text(encoding="utf-8"))]
        return rows

    def test_every_tender_carries_a_unspsc_code(self, tenders):
        assert all(t.unspsc_codes for t in tenders)

    def test_titles_are_short(self, tenders):
        """Roughly eight words. The task is short-text classification, and any
        method that assumes document-length context is mismatched to it."""
        stats = corpus_stats(tenders)
        assert 4 <= stats["mean_title_words"] <= 15

    def test_no_single_class_dominates(self, tenders):
        """Per-tender labelling is what keeps this true; per-item did not."""
        assert corpus_stats(tenders)["majority_share"] < 0.25

    def test_the_task_is_genuinely_multi_class(self, tenders):
        assert corpus_stats(tenders)["distinct_segments"] >= 30

    def test_item_descriptions_are_never_stored_on_the_tender(self, tenders):
        """The parser must not carry the label text into the corpus at all."""
        assert not any(hasattr(t, "item_description") for t in tenders[:50])


class TestDeterminism:
    """Identical inputs must produce identical labels, run after run.

    Two runs of this pipeline over the same warehouse once produced 32 and 33
    label classes, because ties in the dominant-segment vote were broken by set
    iteration order under Python's randomised string hashing.
    """

    def test_ties_are_broken_by_the_lowest_segment_code(self):
        t = make_tender(unspsc_codes=["80100000", "43230000"])   # one item each
        assert t.dominant_segment == "43"

    def test_tie_break_is_independent_of_code_order(self):
        forward = make_tender(unspsc_codes=["80100000", "43230000"]).dominant_segment
        reverse = make_tender(unspsc_codes=["43230000", "80100000"]).dominant_segment
        assert forward == reverse == "43"

    def test_a_clear_winner_still_wins(self):
        t = make_tender(unspsc_codes=["80100000", "80200000", "43230000"])
        assert t.dominant_segment == "80"

    def test_three_way_tie_is_deterministic(self):
        codes = ["93100000", "50100000", "72300000"]
        assert make_tender(unspsc_codes=codes).dominant_segment == "50"
        assert make_tender(unspsc_codes=list(reversed(codes))).dominant_segment == "50"

    def test_label_is_stable_across_repeated_evaluation(self):
        t = make_tender(unspsc_codes=["80100000", "43230000", "50100000"])
        assert len({t.dominant_segment for _ in range(200)}) == 1
