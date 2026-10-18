import math

from matrix_engine import load_matrix, load_sources, match_concepts, recommend
from physics_engine import EngineCatalog


def test_knowledge_matrix_covers_every_engine_and_validates_source_links():
    concepts = load_matrix()
    sources = load_sources()

    assert concepts
    assert set(concepts[0].engine_weights) == set(EngineCatalog.available())
    assert all(
        math.isclose(math.fsum(concept.engine_weights.values()), 1.0)
        or math.fsum(concept.engine_weights.values()) == 0.0
        for concept in concepts
    )
    assert all(
        source in sources
        for concept in concepts
        for source in concept.source_refs
    )
    assert not hasattr(concepts[0], "lesson_ids")


def test_retrieval_links_calculus_and_feynman_sources_to_related_concepts():
    recommendation = recommend("magnetic flux, induction, and changing fields")

    assert recommendation.concept_matches[0].concept.concept_id == "electromagnetic_induction"
    assert "feynman_vol2" in recommendation.source_refs
    assert "strang_calculus" in recommendation.source_refs
    assert dict(recommendation.engine_weights)["electrodynamics"] > 0


def test_new_books_are_retrievable_with_unreviewed_limits():
    cases = (
        (
            "sorting algorithms",
            "algorithm_design",
            "mit_algorithms",
            "needs_review",
            "topic mapping only",
        ),
        (
            "information theory and channel coding",
            "information_theory",
            "shannon_information_theory",
            "needs_review",
            "topic mapping only",
        ),
        (
            "statistical mechanics and ensembles",
            "statistical_mechanics",
            "feynman_statistical_mechanics",
            "needs_review",
            "topic mapping only",
        ),
        (
            "parallel processors and hardware",
            "parallel_computing",
            "parallel_hardware",
            "needs_ocr",
            "none",
        ),
        (
            "machine intelligence and the imitation game",
            "machine_intelligence",
            "turing_mind",
            "needs_review",
            "title-level topic mapping only",
        ),
    )
    sources = load_sources()

    for query, concept_id, source_id, status, review_scope in cases:
        recommendation = recommend(query)

        assert recommendation.concept_matches[0].concept.concept_id == concept_id
        assert source_id in recommendation.source_refs
        assert sources[source_id]["status"] == status
        assert sources[source_id]["review_scope"] == review_scope


def test_matrix_retrieval_does_not_select_a_fixed_output_for_unmatched_queries():
    assert match_concepts("a topic not indexed here") == ()
    recommendation = recommend("a topic not indexed here")
    assert recommendation.concept_matches == ()
    assert recommendation.source_refs == ()
    assert recommendation.engine_weights == ()
