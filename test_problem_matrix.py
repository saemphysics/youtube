from matrix_engine import load_matrix, load_sources
from problem_matrix import load_problem_matrix


def test_problem_records_are_reviewed_retrieval_evidence_not_authored_outputs():
    problems = load_problem_matrix()
    concepts = {concept.concept_id: concept for concept in load_matrix()}
    sources = load_sources()

    assert len(problems) == 4
    assert {problem.problem_id for problem in problems} == {
        "neet2024_physics_q01",
        "neet2024_physics_q10",
        "neet2022_physics_q02",
        "neet2022_physics_q06",
    }
    assert all(problem.source_ref in sources for problem in problems)
    assert all(problem.source_ref in problem.concept.source_refs for problem in problems)
    assert all(problem.concept.concept_id in concepts for problem in problems)
    assert all(problem.worked_solution for problem in problems)
