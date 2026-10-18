import csv
import json
import math

import pytest

from generate_requested_csv import generate
from generate_lecture_series import FIELDS
from matrix_engine import (
    load_matrix,
    recommend,
    sample_lessons,
)
from physics_engine import EngineCatalog


def test_matrix_covers_every_registered_engine_and_validates_links():
    concepts = load_matrix()
    assert concepts
    assert set(concepts[0].engine_weights) == set(EngineCatalog.available())
    assert all(
        math.isclose(sum(concept.engine_weights.values()), 1.0)
        or sum(concept.engine_weights.values()) == 0.0
        for concept in concepts
    )


def test_recommendation_links_eigenvectors_to_sources_engines_and_authored_lesson():
    recommendation = recommend("eigenvectors")

    assert recommendation.concept_matches[0].concept.concept_id == "coupled_modes"
    assert {"feynman_vol3", "strang_cse"} <= set(recommendation.source_refs)
    assert recommendation.engine_weights[0][0] == "linear_algebra"
    assert recommendation.lesson_weights == (("mix08", 1.0),)
    assert math.isclose(sum(weight for _, weight in recommendation.engine_weights), 1.0)


def test_requested_csv_uses_the_existing_render_schema_for_both_orientations(tmp_path):
    paths, recommendation, selected = generate(
        "eigenvectors",
        orientation="both",
        seed=14,
        output_dir=tmp_path,
    )

    assert selected == ("mix08",)
    assert recommendation.source_refs
    assert len(paths) == 2
    for path, orientation in zip(paths, ("horizontal", "vertical")):
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == FIELDS
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["lesson_id"] == "mix08"
        assert rows[0]["orientation"] == orientation
        assert json.loads(rows[0]["equations_json"])
        assert json.loads(rows[0]["visual_data_json"])


def test_weighted_lesson_sampling_is_reproducible():
    recommendation = recommend("matrix eigenvalues")
    assert sample_lessons(recommendation, seed=991) == sample_lessons(
        recommendation,
        seed=991,
    )


def test_computation_concepts_without_an_authored_lesson_fail_explicitly():
    recommendation = recommend("computer information bits")

    assert recommendation.engine_weights == ()
    assert recommendation.lesson_weights == ()
    with pytest.raises(ValueError, match="no authored lesson"):
        sample_lessons(recommendation)


def test_unknown_topic_is_not_silently_mapped_to_a_default_lesson():
    with pytest.raises(ValueError, match="no matrix concepts match"):
        recommend("unmapped astrophysical topic")
