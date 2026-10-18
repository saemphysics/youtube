"""Provider-neutral processing of free-form requests into verified render plans."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import json
import os
from pathlib import Path
from typing import Any, Protocol
import uuid

from engine_router import EngineRouter
from matrix_engine import (
    Recommendation,
    load_matrix,
    load_sources,
    recommend,
    retrieve_book_passages,
)
from physics_engine import EngineCatalog
from problem_matrix import load_problem_matrix
from schema_validator import KNOWN_PRIMITIVES, SUPPORTED_BEAT_TYPES

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 1
MAX_REQUEST_CHARS = 100_000


@dataclass(frozen=True)
class UserRequest:
    text: str
    preferences: dict[str, Any]


@dataclass(frozen=True)
class EngineCall:
    call_id: str
    domain: str
    operation: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class AnalysisPlan:
    intent: str
    concept_ids: tuple[str, ...]
    source_refs: tuple[str, ...]
    engine_calls: tuple[EngineCall, ...]


@dataclass(frozen=True)
class GeneratedArtifact:
    artifact_id: str
    title: str
    tagline: str
    question: str
    narration: str
    equations: tuple[str, ...]
    visuals: tuple[dict[str, Any], ...]
    beats: tuple[dict[str, Any], ...]
    bindings: tuple[dict[str, Any], ...]
    orientation: str
    quality: str
    platform: str
    language: str
    audio: dict[str, Any]
    visual_direction: str


@dataclass(frozen=True)
class GeneratedPlan:
    schema_version: int
    request_id: str
    request_text: str
    analysis: AnalysisPlan
    engine_results: dict[str, Any]
    artifacts: tuple[GeneratedArtifact, ...]


class GenerationProvider(Protocol):
    def analyze(
        self,
        request: UserRequest,
        knowledge: dict[str, Any],
        engine_operations: dict[str, dict[str, str]],
    ) -> AnalysisPlan | dict[str, Any]:
        """Interpret the request and propose verifiable operations."""

    def compose(
        self,
        request: UserRequest,
        knowledge: dict[str, Any],
        analysis: AnalysisPlan,
        engine_results: dict[str, Any],
    ) -> list[GeneratedArtifact] | list[dict[str, Any]]:
        """Create original narration, visuals, audio direction, and render choices."""


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only finite JSON values") from exc
    return value


def _analysis_from(value: AnalysisPlan | dict[str, Any]) -> AnalysisPlan:
    if isinstance(value, AnalysisPlan):
        analysis = value
    elif isinstance(value, dict):
        calls = tuple(
            call if isinstance(call, EngineCall) else EngineCall(
                call_id=str(call["call_id"]),
                domain=str(call["domain"]),
                operation=str(call["operation"]),
                parameters=_json_object(call["parameters"], "engine call parameters"),
            )
            for call in value.get("engine_calls", [])
        )
        analysis = AnalysisPlan(
            intent=str(value.get("intent", "")),
            concept_ids=tuple(map(str, value.get("concept_ids", []))),
            source_refs=tuple(map(str, value.get("source_refs", []))),
            engine_calls=calls,
        )
    else:
        raise TypeError("provider analysis must return AnalysisPlan or a JSON object")
    if not analysis.intent.strip():
        raise ValueError("provider analysis must describe the interpreted intent")
    call_ids = [call.call_id for call in analysis.engine_calls]
    if any(not call_id for call_id in call_ids) or len(set(call_ids)) != len(call_ids):
        raise ValueError("engine call IDs must be non-empty and unique")
    return analysis


def _artifact_from(value: GeneratedArtifact | dict[str, Any]) -> GeneratedArtifact:
    if isinstance(value, GeneratedArtifact):
        artifact = value
    elif isinstance(value, dict):
        artifact = GeneratedArtifact(
            artifact_id=str(value.get("artifact_id", "")),
            title=str(value.get("title", "")),
            tagline=str(value.get("tagline", "")),
            question=str(value.get("question", "")),
            narration=str(value.get("narration", "")),
            equations=tuple(map(str, value.get("equations", []))),
            visuals=tuple(value.get("visuals", [])),
            beats=tuple(value.get("beats", [])),
            bindings=tuple(value.get("bindings", [])),
            orientation=str(value.get("orientation", "")),
            quality=str(value.get("quality", "")),
            platform=str(value.get("platform", "")),
            language=str(value.get("language", "")),
            audio=_json_object(value.get("audio", {}), "audio plan"),
            visual_direction=str(value.get("visual_direction", "")),
        )
    else:
        raise TypeError("provider artifacts must be GeneratedArtifact or JSON objects")

    if not all((
        artifact.artifact_id.strip(),
        artifact.title.strip(),
        artifact.narration.strip(),
        artifact.language.strip(),
        artifact.quality.strip(),
        artifact.platform.strip(),
    )):
        raise ValueError("each artifact needs an ID, title, narration, language, quality, and platform")
    if artifact.orientation not in {"horizontal", "vertical"}:
        raise ValueError("artifact orientation must be horizontal or vertical")
    if not artifact.visuals or not all(
        isinstance(item, dict) and item.get("type") in KNOWN_PRIMITIVES
        for item in artifact.visuals
    ):
        raise ValueError("each artifact needs renderer-supported visual primitives")
    if not all(isinstance(item, str) and item.strip() for item in artifact.equations):
        raise ValueError("equations must be non-empty strings")
    if not artifact.beats or not all(isinstance(item, dict) for item in artifact.beats):
        raise ValueError("each artifact needs a non-empty list of storyboard beats")
    unsupported_beats = {
        str(item.get("type", "")).lower()
        for item in artifact.beats
        if item.get("type") not in SUPPORTED_BEAT_TYPES
    }
    if unsupported_beats:
        raise ValueError(
            f"unsupported storyboard beat types: {', '.join(sorted(unsupported_beats))}"
        )
    if not all(isinstance(item, dict) for item in artifact.bindings):
        raise ValueError("visual bindings must be JSON objects")
    _json_object(asdict(artifact), "generated artifact")
    return artifact


def build_knowledge_context(
    text: str,
    *,
    max_concepts: int = 8,
    max_problems: int = 8,
) -> tuple[dict[str, Any], Recommendation]:
    recommendation = recommend(text)
    concept_matches = recommendation.concept_matches[:max_concepts]
    concepts = [
        {
            "concept_id": match.concept.concept_id,
            "concept": match.concept.concept,
            "keywords": list(match.concept.keywords),
            "source_refs": list(match.concept.source_refs),
            "relevance": match.relevance,
            "engine_weights": match.concept.engine_weights,
        }
        for match in concept_matches
    ]

    tokens = set(text.lower().split())
    related_problems = []
    for problem in load_problem_matrix():
        vocabulary = set(
            f"{problem.problem_summary} {problem.concept.concept} "
            f"{' '.join(problem.concept.keywords)}".lower().split()
        )
        overlap = tokens & vocabulary
        if overlap:
            related_problems.append((
                len(overlap),
                {
                    "source_ref": problem.source_ref,
                    "source_pages": problem.source_pages,
                    "question_number": problem.question_number,
                    "problem_summary": problem.problem_summary,
                    "concept_id": problem.concept.concept_id,
                    "verified_solution_summary": problem.worked_solution,
                },
            ))
    related_problems.sort(key=lambda item: (-item[0], item[1]["source_ref"], item[1]["question_number"]))
    selected_problems = [item for _, item in related_problems[:max_problems]]

    sources = load_sources()
    referenced_ids = {
        source_id
        for concept in concepts
        for source_id in concept["source_refs"]
    }
    referenced_ids.update(problem["source_ref"] for problem in selected_problems)
    context = {
        "matched_concepts": concepts,
        "related_verified_problems": selected_problems,
        "source_notes": {
            source_id: sources[source_id]
            for source_id in sorted(referenced_ids)
        },
        "source_passages": retrieve_book_passages(
            text,
            recommendation.source_refs,
            sources=sources,
        ),
        "engine_evidence": list(recommendation.engine_weights),
        "available_engines": EngineCatalog.describe(),
    }
    return context, recommendation


def run_pipeline(
    request: UserRequest,
    provider: GenerationProvider,
) -> GeneratedPlan:
    if not request.text.strip():
        raise ValueError("request text cannot be empty")
    if len(request.text) > MAX_REQUEST_CHARS:
        raise ValueError(f"request text exceeds {MAX_REQUEST_CHARS} characters")
    _json_object(request.preferences, "request preferences")

    context, recommendation = build_knowledge_context(request.text)
    analysis = _analysis_from(provider.analyze(
        request,
        context,
        EngineRouter.operations(),
    ))
    concepts = {concept.concept_id for concept in load_matrix()}
    unknown_concepts = set(analysis.concept_ids) - concepts
    if unknown_concepts:
        raise ValueError(
            f"provider analysis references unknown concepts: "
            f"{', '.join(sorted(unknown_concepts))}"
        )
    source_ids = set(load_sources())
    unknown_sources = set(analysis.source_refs) - source_ids
    if unknown_sources:
        raise ValueError(
            f"provider analysis references unknown sources: "
            f"{', '.join(sorted(unknown_sources))}"
        )
    relevant_source_ids = set(recommendation.source_refs)
    relevant_source_ids.update(
        problem["source_ref"] for problem in context["related_verified_problems"]
    )
    unsupported_sources = set(analysis.source_refs) - relevant_source_ids
    if unsupported_sources:
        raise ValueError(
            f"provider analysis cites sources not present in request context: "
            f"{', '.join(sorted(unsupported_sources))}"
        )

    engine_results = {}
    for call in analysis.engine_calls:
        engine_results[call.call_id] = EngineRouter.execute(
            call.domain,
            call.operation,
            call.parameters,
        )

    raw_artifacts = provider.compose(request, context, analysis, engine_results)
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError("provider must return at least one generated artifact")
    artifacts = tuple(_artifact_from(item) for item in raw_artifacts)
    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("generated artifact IDs must be unique")

    return GeneratedPlan(
        schema_version=SCHEMA_VERSION,
        request_id=uuid.uuid4().hex,
        request_text=request.text,
        analysis=analysis,
        engine_results=engine_results,
        artifacts=artifacts,
    )


def load_provider(specification: str | None = None) -> GenerationProvider:
    spec = specification or os.environ.get("SAEM_GENERATION_PROVIDER", "")
    if not spec or ":" not in spec:
        raise RuntimeError(
            "No generation provider configured. Set "
            "SAEM_GENERATION_PROVIDER=module.path:create_provider; no preset-content "
            "fallback is used."
        )
    module_name, factory_name = spec.split(":", maxsplit=1)
    if not module_name or not factory_name:
        raise ValueError("provider must use the form 'module.path:factory'")
    factory = getattr(importlib.import_module(module_name), factory_name, None)
    if not callable(factory):
        raise TypeError(f"generation provider factory {spec!r} is not callable")
    provider = factory()
    if not callable(getattr(provider, "analyze", None)) or not callable(
        getattr(provider, "compose", None)
    ):
        raise TypeError("generation provider must implement analyze() and compose()")
    return provider


def save_plan(plan: GeneratedPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(plan), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="free-form request text")
    parser.add_argument("--preferences", default="{}", help="optional JSON object of output constraints")
    parser.add_argument("--provider", help="provider factory as module.path:create_provider")
    parser.add_argument("--output", type=Path, required=True, help="generated plan JSON path")
    args = parser.parse_args()
    try:
        preferences = json.loads(args.preferences)
        request = UserRequest(args.request, _json_object(preferences, "request preferences"))
        plan = run_pipeline(request, load_provider(args.provider))
        save_plan(plan, args.output)
    except (ValueError, TypeError, RuntimeError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    print(f"Plan: {args.output}")
    print(f"Artifacts: {len(plan.artifacts)}")


if __name__ == "__main__":
    main()
