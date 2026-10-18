import csv
import json

import pytest

from generate_lecture_series import FIELDS
from generate_problem_csv import PROBLEM_FIELDS, generate
from matrix_engine import load_matrix
from problem_matrix import load_problem_matrix
from schema_validator import audio_word_bounds, validate_csv


def test_problem_matrix_links_verified_problems_to_sources_and_concepts():
    problems = load_problem_matrix()
    concepts = {concept.concept_id: concept for concept in load_matrix()}

    assert len(problems) == 4
    assert {problem.problem_id for problem in problems} == set(
        {
            "neet2024_physics_q01",
            "neet2024_physics_q10",
            "neet2022_physics_q02",
            "neet2022_physics_q06",
        }
    )
    assert all(problem.source_ref in problem.concept.source_refs for problem in problems)
    assert {"feynman_qed", "strang_calculus"} <= set(
        concepts["refraction"].source_refs
    )
    assert {"feynman_vol2", "strang_calculus"} <= set(
        concepts["electromagnetic_induction"].source_refs
    )
    assert problems[0].concept.concept_id == "uniform_circular_motion"


def test_problem_batches_use_render_schema_and_validate_in_both_orientations(
    tmp_path,
    capsys,
):
    outputs = generate(orientation="both", output_dir=tmp_path)

    assert len(outputs) == 2
    for path, orientation in zip(outputs, ("horizontal", "vertical")):
        assert validate_csv(str(path))
        assert "[WARN]" not in capsys.readouterr().out
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == list(PROBLEM_FIELDS)
            rows = list(reader)
        assert len(rows) == 4
        assert all(row["orientation"] == orientation for row in rows)
        assert all(row["concept_id"] and row["source_ref"] for row in rows)
        assert all(json.loads(row["equations_json"]) for row in rows)
        assert all(json.loads(row["visual_data_json"]) for row in rows)
        minimum, maximum = audio_word_bounds(orientation)
        assert all(minimum <= len(row["audio_script"].split()) <= maximum for row in rows)
    assert list(PROBLEM_FIELDS[:len(FIELDS)]) == FIELDS


def test_problem_batch_selection_is_explicit_and_unknown_ids_fail(tmp_path):
    output, = generate(
        orientation="vertical",
        output_dir=tmp_path,
        problem_ids=["neet2022_physics_q06"],
    )
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["feynman_sources"] == "feynman_vol2"
    assert rows[0]["strang_sources"] == "strang_calculus"
    assert rows[0]["related_lesson_ids"] == "mix07"

    with pytest.raises(ValueError, match="unknown problem IDs"):
        generate(
            orientation="vertical",
            output_dir=tmp_path,
            problem_ids=["missing_problem"],
        )
