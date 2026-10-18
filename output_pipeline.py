"""Validate generated plans and adapt them to the existing media backends."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any

from request_pipeline import GeneratedArtifact, GeneratedPlan, _artifact_from

RENDER_FIELDS = (
    "video_id",
    "concept_type",
    "header_title",
    "tagline",
    "question_text",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_answer",
    "equations_json",
    "visual_data_json",
    "audio_script",
    "orientation",
    "beats_json",
    "bindings_json",
    "language",
    "quality",
    "platform",
    "audio_direction",
    "audio_json",
    "visual_direction",
    "source_refs",
    "calculation_results_json",
)


def _video_id(plan: GeneratedPlan, artifact: GeneratedArtifact) -> str:
    base = re.sub(r"[^\w-]+", "_", artifact.artifact_id.strip()).strip("_")
    if not base:
        raise ValueError("artifact ID must contain a usable filename character")
    return f"{base[:80]}_{plan.request_id[:8]}"


def rows_for_plan(plan: GeneratedPlan) -> list[dict[str, str]]:
    if plan.schema_version != 1:
        raise ValueError(f"unsupported generated plan schema version: {plan.schema_version}")
    rows = []
    for artifact in plan.artifacts:
        artifact = _artifact_from(artifact)
        rows.append({
            "video_id": _video_id(plan, artifact),
            "concept_type": "GENERIC",
            "header_title": artifact.title,
            "tagline": artifact.tagline,
            "question_text": artifact.question,
            "option_a": "",
            "option_b": "",
            "option_c": "",
            "option_d": "",
            "correct_answer": "",
            "equations_json": json.dumps(artifact.equations, ensure_ascii=False),
            "visual_data_json": json.dumps(artifact.visuals, ensure_ascii=False),
            "audio_script": artifact.narration,
            "orientation": artifact.orientation,
            "beats_json": json.dumps(artifact.beats, ensure_ascii=False),
            "bindings_json": json.dumps(artifact.bindings, ensure_ascii=False),
            "language": artifact.language,
            "quality": artifact.quality,
            "platform": artifact.platform,
            "audio_direction": artifact.audio.get("direction", ""),
            "audio_json": json.dumps(artifact.audio, ensure_ascii=False),
            "visual_direction": artifact.visual_direction,
            "source_refs": ";".join(plan.analysis.source_refs),
            "calculation_results_json": json.dumps(
                plan.engine_results,
                ensure_ascii=False,
                allow_nan=False,
            ),
        })
    if not rows:
        raise ValueError("generated plan has no output artifacts")
    video_ids = [row["video_id"] for row in rows]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("generated plan produces duplicate video IDs")
    return rows


def write_renderer_csv(plan: GeneratedPlan, path: Path) -> Path:
    rows = rows_for_plan(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=RENDER_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_plan(path: Path) -> GeneratedPlan:
    from request_pipeline import (
        AnalysisPlan,
        EngineCall,
        GeneratedPlan,
        _analysis_from,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("generated plan JSON must contain an object")
    analysis_data = payload.get("analysis")
    if not isinstance(analysis_data, dict):
        raise ValueError("generated plan is missing analysis metadata")
    analysis = _analysis_from(analysis_data)
    artifacts = tuple(_artifact_from(item) for item in payload.get("artifacts", []))
    plan = GeneratedPlan(
        schema_version=int(payload.get("schema_version", 0)),
        request_id=str(payload.get("request_id", "")),
        request_text=str(payload.get("request_text", "")),
        analysis=AnalysisPlan(
            intent=analysis.intent,
            concept_ids=analysis.concept_ids,
            source_refs=analysis.source_refs,
            engine_calls=tuple(
                EngineCall(
                    call_id=call.call_id,
                    domain=call.domain,
                    operation=call.operation,
                    parameters=call.parameters,
                )
                for call in analysis.engine_calls
            ),
        ),
        engine_results=payload.get("engine_results", {}),
        artifacts=artifacts,
    )
    if not plan.request_id or not plan.request_text.strip():
        raise ValueError("generated plan needs a request ID and original request")
    json.dumps(plan.engine_results, allow_nan=False)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path, help="generated plan JSON")
    parser.add_argument("--csv-output", type=Path, help="write the renderer adapter CSV here")
    parser.add_argument("--dry-run", action="store_true", help="validate and list outputs without rendering")
    parser.add_argument("--only", help="comma-separated generated video IDs to render")
    args = parser.parse_args()

    try:
        plan = load_plan(args.plan)
        if args.csv_output:
            csv_path = write_renderer_csv(plan, args.csv_output)
            print(f"Renderer adapter: {csv_path}")
            if args.dry_run:
                subprocess.run(
                    [sys.executable, "batch_runner.py", "--csv", str(csv_path), "--dry-run"],
                    check=True,
                )
                return
            command = [sys.executable, "batch_runner.py", "--csv", str(csv_path)]
            if args.only:
                command.extend(["--only", args.only])
            subprocess.run(command, check=True)
            return

        with tempfile.TemporaryDirectory(prefix="six-render-plan-") as temp_dir:
            csv_path = write_renderer_csv(plan, Path(temp_dir) / "render.csv")
            command = [
                sys.executable,
                "batch_runner.py",
                "--csv",
                str(csv_path),
            ]
            if args.dry_run:
                command.append("--dry-run")
            if args.only:
                command.extend(["--only", args.only])
            subprocess.run(command, check=True)
    except (OSError, ValueError, TypeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
