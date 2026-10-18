"""Knowledge-matrix retrieval and cross-domain evidence ranking."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re

from pypdf import PdfReader

from physics_engine import EngineCatalog

ROOT = Path(__file__).resolve().parent
MATRIX_PATH = ROOT / "knowledge_matrix.csv"
SOURCES_PATH = ROOT / "book_sources.csv"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "by", "can",
    "create", "for", "from", "how", "i", "in", "into", "is", "it", "make",
    "me", "of", "on", "or", "please", "show", "that", "the", "this", "to",
    "use", "what", "when", "where", "which", "why", "with",
}
_METADATA_FIELDS = ("concept_id", "concept", "keywords", "source_refs")
MAX_SOURCE_PASSAGES = 8
MAX_SOURCE_PASSAGE_CHARS = 1200
_MAX_SOURCE_PAGES = 1000


@dataclass(frozen=True)
class MatrixConcept:
    concept_id: str
    concept: str
    keywords: tuple[str, ...]
    source_refs: tuple[str, ...]
    engine_weights: dict[str, float]


@dataclass(frozen=True)
class ConceptMatch:
    concept: MatrixConcept
    relevance: float


@dataclass(frozen=True)
class Recommendation:
    query: str
    concept_matches: tuple[ConceptMatch, ...]
    engine_weights: tuple[tuple[str, float], ...]
    source_refs: tuple[str, ...]


def _tokens(text: str) -> set[str]:
    result = set()
    for word in _TOKEN.findall(str(text).lower()):
        if word.endswith("ies") and len(word) > 4:
            word = word[:-3] + "y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        if word not in _STOP_WORDS:
            result.add(word)
    return result


def load_sources(path: Path = SOURCES_PATH) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"source_id", "relative_path", "status", "review_scope", "notes"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name} is missing required source-ledger columns")
        sources = {
            row["source_id"].strip(): {
                field: row[field].strip()
                for field in required
            }
            for row in reader
            if row.get("source_id", "").strip()
        }
    if not sources:
        raise ValueError(f"{path.name} contains no sources")
    return sources


def load_matrix(
    path: Path = MATRIX_PATH,
    sources_path: Path = SOURCES_PATH,
) -> tuple[MatrixConcept, ...]:
    """Load concepts and verify that source and engine links remain current."""
    engine_names = tuple(EngineCatalog.available())
    sources = load_sources(sources_path)
    concepts: list[MatrixConcept] = []
    seen_ids: set[str] = set()

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if (
            fields[:len(_METADATA_FIELDS)] != list(_METADATA_FIELDS)
            or len(fields) != len(_METADATA_FIELDS) + len(engine_names)
            or set(fields[len(_METADATA_FIELDS):]) != set(engine_names)
        ):
            raise ValueError(
                f"{path.name} columns must contain concept metadata followed by "
                "the registered engine catalog"
            )

        for row_number, row in enumerate(reader, start=2):
            concept_id = row["concept_id"].strip()
            if not concept_id or concept_id in seen_ids:
                raise ValueError(
                    f"{path.name}:{row_number} has a missing or duplicate concept_id"
                )
            seen_ids.add(concept_id)
            if not row["concept"].strip() or not row["keywords"].strip():
                raise ValueError(
                    f"{path.name}:{row_number} needs a concept and searchable keywords"
                )

            source_refs = tuple(
                item.strip() for item in row["source_refs"].split(";") if item.strip()
            )
            missing_sources = set(source_refs) - set(sources)
            if missing_sources:
                raise ValueError(
                    f"{path.name}:{row_number} references unknown sources: "
                    f"{', '.join(sorted(missing_sources))}"
                )
            weights = {}
            for engine in engine_names:
                try:
                    weight = float(row[engine])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path.name}:{row_number} has an invalid {engine} weight"
                    ) from exc
                if not math.isfinite(weight) or weight < 0:
                    raise ValueError(
                        f"{path.name}:{row_number} weights must be finite and non-negative"
                    )
                weights[engine] = weight
            total = math.fsum(weights.values())
            if not (math.isclose(total, 1.0, abs_tol=1e-9) or total == 0.0):
                raise ValueError(
                    f"{path.name}:{row_number} engine weights must sum to 1 or all be 0"
                )

            concepts.append(MatrixConcept(
                concept_id=concept_id,
                concept=row["concept"].strip(),
                keywords=tuple(
                    item.strip() for item in row["keywords"].split(";") if item.strip()
                ),
                source_refs=source_refs,
                engine_weights=weights,
            ))

    if not concepts:
        raise ValueError(f"{path.name} contains no concepts")
    return tuple(concepts)


def match_concepts(
    query: str,
    concepts: tuple[MatrixConcept, ...] | None = None,
) -> tuple[ConceptMatch, ...]:
    query_tokens = _tokens(query)
    if not query_tokens:
        raise ValueError("query must contain at least one searchable word")
    available = concepts if concepts is not None else load_matrix()
    matches = []
    for concept in available:
        vocabulary = _tokens(" ".join((
            concept.concept_id.replace("_", " "),
            concept.concept,
            *concept.keywords,
        )))
        overlap = query_tokens & vocabulary
        if overlap:
            matches.append(ConceptMatch(concept, len(overlap) / len(query_tokens)))
    return tuple(sorted(
        matches,
        key=lambda match: (-match.relevance, match.concept.concept_id),
    ))


def _normalize_weights(
    evidence: dict[str, float],
) -> tuple[tuple[str, float], ...]:
    total = math.fsum(evidence.values())
    if total <= 0:
        return ()
    normalized = ((name, weight / total) for name, weight in evidence.items())
    return tuple(sorted(normalized, key=lambda pair: (-pair[1], pair[0])))


def recommend(query: str, matrix_path: Path = MATRIX_PATH) -> Recommendation:
    """Return related concepts, engines, and source IDs without choosing content."""
    concepts = load_matrix(matrix_path)
    matches = match_concepts(query, concepts)
    evidence = {engine: 0.0 for engine in EngineCatalog.available()}
    for match in matches:
        for engine, weight in match.concept.engine_weights.items():
            evidence[engine] += match.relevance * weight
    engine_weights = _normalize_weights(evidence)
    source_refs = tuple(sorted({
        source for match in matches for source in match.concept.source_refs
    }))
    return Recommendation(
        query=query,
        concept_matches=matches,
        engine_weights=engine_weights,
        source_refs=source_refs,
    )


@lru_cache(maxsize=8)
def _pdf_page_text(path: str) -> tuple[str, ...]:
    reader = PdfReader(path, strict=False)
    if reader.is_encrypted:
        raise ValueError(f"cannot retrieve text from encrypted source {Path(path).name}")
    if len(reader.pages) > _MAX_SOURCE_PAGES:
        raise ValueError(
            f"source {Path(path).name} exceeds the {_MAX_SOURCE_PAGES}-page retrieval limit"
        )
    return tuple(page.extract_text() or "" for page in reader.pages)


def retrieve_book_passages(
    query: str,
    source_refs: tuple[str, ...],
    *,
    sources: dict[str, dict[str, str]] | None = None,
    max_passages: int = MAX_SOURCE_PASSAGES,
) -> tuple[dict[str, str | int | float], ...]:
    """Return short, page-cited PDF excerpts from text-readable linked sources."""
    query_tokens = _tokens(query)
    if not query_tokens or max_passages < 1:
        return ()
    source_catalog = sources if sources is not None else load_sources()
    matches: list[tuple[float, str, int, dict[str, str | int | float]]] = []

    for source_id in sorted(set(source_refs)):
        source = source_catalog[source_id]
        if source["status"] not in {"readable", "needs_review"}:
            continue
        relative_path = Path(source["relative_path"])
        if relative_path.suffix.lower() != ".pdf":
            continue
        path = (ROOT / relative_path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"source {source_id!r} resolves outside the repository")

        for page_number, page_text in enumerate(_pdf_page_text(str(path)), start=1):
            page_tokens = _tokens(page_text)
            overlap = query_tokens & page_tokens
            if not overlap:
                continue

            token_match = next(
                (
                    token.start()
                    for token in _TOKEN.finditer(page_text.lower())
                    if _tokens(token.group()) & overlap
                ),
                0,
            )
            start = max(0, token_match - MAX_SOURCE_PASSAGE_CHARS // 3)
            excerpt = " ".join(
                page_text[start:start + MAX_SOURCE_PASSAGE_CHARS].split()
            )
            score = len(overlap) / len(query_tokens)
            matches.append((
                score,
                source_id,
                page_number,
                {
                    "source_ref": source_id,
                    "relative_path": source["relative_path"],
                    "page_number": page_number,
                    "relevance": score,
                    "review_status": source["status"],
                    "review_scope": source["review_scope"],
                    "excerpt": excerpt,
                },
            ))

    matches.sort(key=lambda match: (-match[0], match[1], match[2]))
    return tuple(match[3] for match in matches[:max_passages])
