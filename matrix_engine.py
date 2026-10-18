"""Source-linked concept matrix and seeded weighted recommendations."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import random
import re

from physics_engine import EngineCatalog

ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / "knowledge_matrix.csv"
SOURCES_PATH = ROOT / "book_sources.csv"
PLAN_PATH = ROOT / "lecture_series_plan.csv"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "build", "by",
    "can", "create", "for", "from", "how", "i", "in", "into", "is", "it",
    "make", "me", "of", "on", "or", "please", "show", "that", "the", "this",
    "to", "use", "what", "when", "where", "which", "why", "with",
}


@dataclass(frozen=True)
class MatrixConcept:
    concept_id: str
    concept: str
    keywords: tuple[str, ...]
    source_refs: tuple[str, ...]
    lesson_ids: tuple[str, ...]
    engine_weights: dict[str, float]


@dataclass(frozen=True)
class ConceptMatch:
    concept: MatrixConcept
    relevance: float


@dataclass(frozen=True)
class Recommendation:
    topic: str
    concept_matches: tuple[ConceptMatch, ...]
    engine_weights: tuple[tuple[str, float], ...]
    lesson_weights: tuple[tuple[str, float], ...]
    source_refs: tuple[str, ...]


def _tokens(text: str) -> set[str]:
    words = _TOKEN.findall(str(text).lower())
    result = set()
    for word in words:
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        if word not in _STOP_WORDS:
            result.add(word)
    return result


def _read_column(path: Path, field: str) -> dict[str, str]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or field not in reader.fieldnames:
            raise ValueError(f"{path.name} must contain a {field!r} column")
        return {
            row[field].strip(): row.get("status", "")
            for row in reader
            if row.get(field, "").strip()
        }


def load_matrix(path: Path = MATRIX_PATH) -> tuple[MatrixConcept, ...]:
    """Load the reviewed matrix and reject stale engine/source/lesson links."""
    engine_names = EngineCatalog.available()
    metadata_fields = [
        "concept_id", "concept", "keywords", "source_refs", "lesson_ids",
    ]
    source_statuses = _read_column(SOURCES_PATH, "source_id")
    lesson_ids = set(_read_column(PLAN_PATH, "lesson_id"))
    concepts = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if (
            fields[:len(metadata_fields)] != metadata_fields
            or len(fields) != len(metadata_fields) + len(engine_names)
            or set(fields[len(metadata_fields):]) != set(engine_names)
        ):
            raise ValueError(
                f"{path.name} columns must match the registered engine catalog"
            )
        for row_number, row in enumerate(reader, start=2):
            concept_id = row["concept_id"].strip()
            if not concept_id or concept_id in seen_ids:
                raise ValueError(f"{path.name}:{row_number} has a missing or duplicate concept_id")
            seen_ids.add(concept_id)

            sources = tuple(filter(None, (item.strip() for item in row["source_refs"].split(";"))))
            missing_sources = set(sources) - set(source_statuses)
            if missing_sources:
                raise ValueError(
                    f"{path.name}:{row_number} references unknown sources: "
                    f"{', '.join(sorted(missing_sources))}"
                )
            unavailable = {
                item for item in sources
                if source_statuses[item] != "readable"
            }
            if unavailable:
                raise ValueError(
                    f"{path.name}:{row_number} references unreadable sources: "
                    f"{', '.join(sorted(unavailable))}"
                )

            lessons = tuple(filter(None, (item.strip() for item in row["lesson_ids"].split(";"))))
            missing_lessons = set(lessons) - lesson_ids
            if missing_lessons:
                raise ValueError(
                    f"{path.name}:{row_number} references unknown lessons: "
                    f"{', '.join(sorted(missing_lessons))}"
                )

            weights = {}
            for name in engine_names:
                try:
                    weight = float(row[name])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path.name}:{row_number} has an invalid {name} weight"
                    ) from exc
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(
                        f"{path.name}:{row_number} weights must be finite and non-negative"
                    )
                weights[name] = weight
            total = math.fsum(weights.values())
            if not (math.isclose(total, 1.0, abs_tol=1e-9) or total == 0.0):
                raise ValueError(
                    f"{path.name}:{row_number} engine weights must sum to 1 or all be 0"
                )
            if not row["concept"].strip() or not row["keywords"].strip():
                raise ValueError(f"{path.name}:{row_number} needs a concept and keywords")

            concepts.append(MatrixConcept(
                concept_id=concept_id,
                concept=row["concept"].strip(),
                keywords=tuple(
                    item.strip() for item in row["keywords"].split(";") if item.strip()
                ),
                source_refs=sources,
                lesson_ids=lessons,
                engine_weights=weights,
            ))

    if not concepts:
        raise ValueError(f"{path.name} contains no concepts")
    return tuple(concepts)


def match_concepts(
    topic: str,
    concepts: tuple[MatrixConcept, ...] | None = None,
) -> tuple[ConceptMatch, ...]:
    query = _tokens(topic)
    if not query:
        raise ValueError("topic must contain at least one searchable word")
    concepts = concepts if concepts is not None else load_matrix()
    matches = []
    for concept in concepts:
        vocabulary = _tokens(" ".join((
            concept.concept_id.replace("_", " "),
            concept.concept,
            *concept.keywords,
        )))
        overlap = query & vocabulary
        if overlap:
            matches.append(ConceptMatch(concept, len(overlap) / len(query)))
    return tuple(sorted(
        matches,
        key=lambda match: (-match.relevance, match.concept.concept_id),
    ))


def _normalize_weights(
    evidence: dict[str, float],
    categories: tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    evidence_total = math.fsum(evidence.values())
    if not categories or evidence_total <= 0:
        return ()
    probabilities = [
        (category, evidence.get(category, 0.0) / evidence_total)
        for category in categories
    ]
    return tuple(sorted(probabilities, key=lambda item: (-item[1], item[0])))


def recommend(topic: str, matrix_path: Path = MATRIX_PATH) -> Recommendation:
    """Return normalized keyword-evidence weights, not calibrated probabilities."""
    concepts = load_matrix(matrix_path)
    matches = match_concepts(topic, concepts)
    if not matches:
        raise ValueError(
            f"no matrix concepts match {topic!r}; add a reviewed concept mapping first"
        )

    engine_evidence = {name: 0.0 for name in EngineCatalog.available()}
    lesson_evidence: dict[str, float] = {}
    for match in matches:
        for name, weight in match.concept.engine_weights.items():
            engine_evidence[name] += match.relevance * weight
        if match.concept.lesson_ids:
            per_lesson = match.relevance / len(match.concept.lesson_ids)
            for lesson_id in match.concept.lesson_ids:
                lesson_evidence[lesson_id] = lesson_evidence.get(lesson_id, 0.0) + per_lesson

    engine_weights = _normalize_weights(
        engine_evidence,
        tuple(EngineCatalog.available()),
    )
    lesson_weights = _normalize_weights(
        lesson_evidence,
        tuple(sorted(lesson_evidence)),
    )
    sources = tuple(sorted({
        source for match in matches for source in match.concept.source_refs
    }))
    return Recommendation(
        topic=topic,
        concept_matches=matches,
        engine_weights=engine_weights,
        lesson_weights=lesson_weights,
        source_refs=sources,
    )


def sample_lessons(
    recommendation: Recommendation,
    count: int = 1,
    seed: int = 2026,
) -> tuple[str, ...]:
    """Sample distinct authored lessons from normalized matrix weights."""
    count = int(count)
    if count < 1:
        raise ValueError("count must be at least 1")
    if not recommendation.lesson_weights:
        raise ValueError(
            "matched concepts have no authored lesson; add and review a lesson first"
        )
    if count > len(recommendation.lesson_weights):
        raise ValueError(
            f"requested {count} lessons, but only "
            f"{len(recommendation.lesson_weights)} are mapped to this topic"
        )

    rng = random.Random(seed)
    available = list(recommendation.lesson_weights)
    chosen = []
    for _ in range(count):
        lesson_ids, probabilities = zip(*available)
        selected = rng.choices(lesson_ids, weights=probabilities, k=1)[0]
        chosen.append(selected)
        available = [item for item in available if item[0] != selected]
    return tuple(chosen)
