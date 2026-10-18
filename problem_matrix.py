"""Load the reviewed exam-problem matrix and validate its concept/source links."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re

from matrix_engine import MATRIX_PATH, SOURCES_PATH, MatrixConcept, load_matrix
from schema_validator import KNOWN_PRIMITIVES

ROOT = Path(__file__).resolve().parent
PROBLEM_MATRIX_PATH = ROOT / "problem_matrix.csv"
FIELDS = (
    "problem_id",
    "source_ref",
    "source_pages",
    "question_number",
    "problem_summary",
    "concept_id",
    "worked_solution",
    "equations_json",
    "visual_data_json",
    "review_status",
)
_PAGE_REFERENCE = re.compile(r"\d+(?:-\d+)?\Z")


@dataclass(frozen=True)
class Problem:
    problem_id: str
    source_ref: str
    source_pages: str
    question_number: int
    problem_summary: str
    concept: MatrixConcept
    worked_solution: str
    equations: tuple[str, ...]
    visual_data: tuple[dict[str, object], ...]


def _source_statuses() -> dict[str, str]:
    with SOURCES_PATH.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "source_id" not in reader.fieldnames:
            raise ValueError(f"{SOURCES_PATH.name} must contain a 'source_id' column")
        return {
            row["source_id"].strip(): row.get("status", "").strip()
            for row in reader
            if row.get("source_id", "").strip()
        }


def load_problem_matrix(
    path: Path = PROBLEM_MATRIX_PATH,
    concepts_path: Path = MATRIX_PATH,
) -> tuple[Problem, ...]:
    """Load verified problem entries and require valid source/concept links."""
    concepts = {concept.concept_id: concept for concept in load_matrix(concepts_path)}
    source_statuses = _source_statuses()
    problems = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"{path.name} columns must match the problem matrix schema")

        for row_number, row in enumerate(reader, start=2):
            problem_id = row["problem_id"].strip()
            if not problem_id or problem_id in seen_ids:
                raise ValueError(
                    f"{path.name}:{row_number} has a missing or duplicate problem_id"
                )
            seen_ids.add(problem_id)

            source_ref = row["source_ref"].strip()
            if source_ref not in source_statuses:
                raise ValueError(
                    f"{path.name}:{row_number} references unknown source {source_ref!r}"
                )
            if source_statuses[source_ref] != "readable":
                raise ValueError(
                    f"{path.name}:{row_number} references non-readable source {source_ref!r}"
                )

            concept_id = row["concept_id"].strip()
            if concept_id not in concepts:
                raise ValueError(
                    f"{path.name}:{row_number} references unknown concept {concept_id!r}"
                )
            concept = concepts[concept_id]
            if source_ref not in concept.source_refs:
                raise ValueError(
                    f"{path.name}:{row_number} source {source_ref!r} is not linked "
                    f"from concept {concept_id!r}"
                )

            pages = row["source_pages"].strip()
            if not _PAGE_REFERENCE.fullmatch(pages):
                raise ValueError(
                    f"{path.name}:{row_number} has an invalid PDF page reference"
                )
            try:
                question_number = int(row["question_number"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name}:{row_number} has an invalid question number"
                ) from exc
            if question_number < 1:
                raise ValueError(
                    f"{path.name}:{row_number} question number must be positive"
                )
            if row["review_status"].strip() != "verified":
                raise ValueError(
                    f"{path.name}:{row_number} is not marked as a verified problem"
                )
            if not row["problem_summary"].strip() or not row["worked_solution"].strip():
                raise ValueError(
                    f"{path.name}:{row_number} needs a prompt summary and worked solution"
                )

            try:
                equations = json.loads(row["equations_json"])
                visuals = json.loads(row["visual_data_json"])
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}:{row_number} contains invalid JSON: {exc.msg}"
                ) from exc
            if (
                not isinstance(equations, list)
                or not equations
                or not all(isinstance(equation, str) and equation for equation in equations)
            ):
                raise ValueError(
                    f"{path.name}:{row_number} equations_json must be a non-empty string array"
                )
            if not isinstance(visuals, list) or not visuals:
                raise ValueError(
                    f"{path.name}:{row_number} visual_data_json must be a non-empty array"
                )
            for visual in visuals:
                if not isinstance(visual, dict) or visual.get("type") not in KNOWN_PRIMITIVES:
                    raise ValueError(
                        f"{path.name}:{row_number} contains an unsupported visual primitive"
                    )

            problems.append(Problem(
                problem_id=problem_id,
                source_ref=source_ref,
                source_pages=pages,
                question_number=question_number,
                problem_summary=row["problem_summary"].strip(),
                concept=concept,
                worked_solution=row["worked_solution"].strip(),
                equations=tuple(equations),
                visual_data=tuple(visuals),
            ))

    if not problems:
        raise ValueError(f"{path.name} contains no verified problems")
    return tuple(problems)
