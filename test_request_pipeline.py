import csv
import json

import numpy as np
import pytest

from engine_router import EngineRouter, SafeExpression
from matrix_engine import load_matrix, recommend
from output_pipeline import RENDER_FIELDS, rows_for_plan, write_renderer_csv
from request_pipeline import (
    AnalysisPlan,
    EngineCall,
    GeneratedArtifact,
    UserRequest,
    build_knowledge_context,
    load_provider,
    run_pipeline,
)


class RequestDrivenProvider:
    def analyze(self, request, knowledge, engine_operations):
        assert request.text == "Explain the slope of x squared as a moving point."
        assert "calculus" in engine_operations
        return AnalysisPlan(
            intent="Explain how the derivative changes with position.",
            concept_ids=("instantaneous_power",),
            source_refs=("strang_calculus",),
            engine_calls=(
                EngineCall(
                    call_id="slope_at_two",
                    domain="calculus",
                    operation="derivative",
                    parameters={"func": {"expression": "x**2"}, "x": 2.0},
                ),
            ),
        )

    def compose(self, request, knowledge, analysis, engine_results):
        assert engine_results["slope_at_two"] == pytest.approx(4.0)
        return [{
            "artifact_id": "slope_at_two",
            "title": "A changing slope",
            "tagline": "How fast is x squared changing here?",
            "question": "Find the instantaneous slope of x^2 at x=2.",
            "narration": "The curve rises four units for each local unit of horizontal change at this point.",
            "equations": ["f(x)=x^2", "f'(2)=4"],
            "visuals": [{
                "type": "graph",
                "expression": "x**2",
                "x_range": [-2, 2],
                "y_range": [0, 4],
                "color": "#3498DB",
            }],
            "beats": [{"type": "hook", "text": "Locate the point"}],
            "bindings": [],
            "orientation": "vertical",
            "quality": "1080p",
            "platform": "shorts",
            "language": "en",
            "audio": {
                "direction": "measured, curious, no music",
                "voice_reference": "assets/voice.wav",
                "ambient": {"kind": "none"},
            },
            "visual_direction": "Use a custom indigo and amber palette.",
        }]


def test_matrix_retrieval_does_not_select_a_fixed_lesson_or_script():
    recommendation = recommend("matrix eigenvalues and coupled motion")
    assert recommendation.concept_matches[0].concept.concept_id == "coupled_modes"
    assert "linear_algebra" in dict(recommendation.engine_weights)
    assert not hasattr(recommendation, "lesson_weights")
    assert not hasattr(load_matrix()[0], "lesson_ids")


def test_provider_context_includes_new_book_sources_and_review_limits():
    context, recommendation = build_knowledge_context(
        "Explain entropy and information theory."
    )

    assert "information_theory" in {
        match.concept.concept_id for match in recommendation.concept_matches
    }
    assert context["source_notes"]["shannon_information_theory"]["status"] == "needs_review"
    assert context["source_notes"]["shannon_information_theory"]["review_scope"] == (
        "topic mapping only"
    )
    assert any(
        passage["source_ref"] == "shannon_information_theory"
        and passage["page_number"] > 0
        and "information" in passage["excerpt"].lower()
        for passage in context["source_passages"]
    )
    assert all(len(passage["excerpt"]) <= 1200 for passage in context["source_passages"])


def test_provider_context_skips_pdfs_without_extractable_text():
    context, _ = build_knowledge_context("parallel processors and hardware")

    assert context["source_notes"]["parallel_hardware"]["status"] == "needs_ocr"
    assert all(
        passage["source_ref"] != "parallel_hardware"
        for passage in context["source_passages"]
    )


def test_safe_engine_router_evaluates_expression_and_returns_json_values():
    derivative = EngineRouter.execute(
        "calculus",
        "derivative",
        {"func": {"expression": "x**2"}, "x": 3.0},
    )
    assert derivative == pytest.approx(6.0)
    assert np.asarray(EngineRouter.execute(
        "linear_algebra",
        "eigenvalues",
        {"matrix": [[2.0, 0.0], [0.0, 3.0]]},
    )) == pytest.approx([3.0, 2.0])


@pytest.mark.parametrize("expression", [
    "__import__('os').system('id')",
    "x.real",
    "x[0]",
    "sqrt(x, 2)",
    "x+y",
])
def test_safe_expression_rejects_non_arithmetic_or_ambiguous_code(expression):
    with pytest.raises(ValueError):
        SafeExpression(expression)


def test_pipeline_uses_provider_analysis_and_verified_engine_results():
    request = UserRequest(
        "Explain the slope of x squared as a moving point.",
        {"orientation": "vertical", "visual_palette": "indigo and amber"},
    )
    plan = run_pipeline(request, RequestDrivenProvider())

    assert plan.analysis.intent.startswith("Explain")
    assert plan.engine_results == {"slope_at_two": pytest.approx(4.0)}
    artifact = plan.artifacts[0]
    assert artifact.audio["direction"] == "measured, curious, no music"
    assert artifact.visual_direction == "Use a custom indigo and amber palette."
    assert artifact.narration.startswith("The curve")


def test_output_adapter_preserves_generated_content_and_preferences(tmp_path):
    plan = run_pipeline(
        UserRequest("Explain the slope of x squared as a moving point.", {}),
        RequestDrivenProvider(),
    )
    output = write_renderer_csv(plan, tmp_path / "render.csv")
    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    assert reader.fieldnames == list(RENDER_FIELDS)
    assert len(rows) == 1
    row = rows[0]
    assert row["audio_script"] == plan.artifacts[0].narration
    assert json.loads(row["audio_json"])["direction"] == "measured, curious, no music"
    assert row["visual_direction"] == "Use a custom indigo and amber palette."
    assert json.loads(row["calculation_results_json"])["slope_at_two"] == pytest.approx(4.0)


def test_missing_provider_fails_explicitly_without_preset_fallback(monkeypatch):
    monkeypatch.delenv("SAEM_GENERATION_PROVIDER", raising=False)
    with pytest.raises(RuntimeError, match="no preset-content fallback"):
        load_provider()


def test_unknown_source_claim_from_provider_is_rejected():
    class InvalidProvider(RequestDrivenProvider):
        def analyze(self, request, knowledge, engine_operations):
            return AnalysisPlan(
                intent="Invent unsupported citation",
                concept_ids=(),
                source_refs=("not-a-source",),
                engine_calls=(),
            )

    with pytest.raises(ValueError, match="unknown sources"):
        run_pipeline(UserRequest("Explain something new.", {}), InvalidProvider())


def test_unknown_visual_primitive_is_rejected():
    class InvalidVisualProvider(RequestDrivenProvider):
        def compose(self, request, knowledge, analysis, engine_results):
            result = super().compose(request, knowledge, analysis, engine_results)
            result[0]["visuals"] = [{"type": "made_up_renderer"}]
            return result

    with pytest.raises(ValueError, match="renderer-supported"):
        run_pipeline(
            UserRequest("Explain the slope of x squared as a moving point.", {}),
            InvalidVisualProvider(),
        )
