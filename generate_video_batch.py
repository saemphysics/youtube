"""Generate ten user-authored physics video plans and a renderer batch CSV."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from batch_runner import CLOSING_CARD_SEC
from output_pipeline import RENDER_FIELDS, rows_for_plan
from request_pipeline import (
    GeneratedPlan,
    UserRequest,
    load_provider,
    run_pipeline,
    save_plan,
)
from schema_validator import SUPPORTED_QUALITY

VIDEO_COUNT = 10
FORMAT_STYLE = (
    "Create an original educational explanation with first-principles reasoning, "
    "intuitive physical analogies when useful, precise mathematics, and elegant, "
    "readable visual explanations. Do not imitate a named creator's distinctive "
    "style, voice, or signature scenes."
)


def read_requests(path: Path) -> list[str]:
    """Read exactly ten non-empty, one-line natural-language requests."""
    try:
        requests = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except OSError as exc:
        raise ValueError(f"cannot read request file {path}: {exc}") from exc
    if len(requests) != VIDEO_COUNT:
        raise ValueError(
            f"{path} must contain exactly {VIDEO_COUNT} non-empty request lines; "
            f"found {len(requests)}"
        )
    return requests


def preferences_for(index: int) -> dict:
    """Assign five landscape long-form and five portrait short-form outputs."""
    if not 0 <= index < VIDEO_COUNT:
        raise ValueError(f"video index must be between 0 and {VIDEO_COUNT - 1}")
    orientation = "horizontal" if index < VIDEO_COUNT // 2 else "vertical"
    target_duration = 180 if orientation == "horizontal" else 60
    narration_duration = target_duration - CLOSING_CARD_SEC
    return {
        "orientation": orientation,
        "platform": "youtube",
        "quality": "1080p",
        "target_duration_seconds": target_duration,
        "closing_card_seconds": CLOSING_CARD_SEC,
        "audio": {
            "duration_seconds": narration_duration,
            "direction": "Clear, natural educational narration; no music unless requested.",
            "ambient": {"kind": "none"},
        },
        "editorial_direction": FORMAT_STYLE,
    }


def _validate_generated_plan(plan: GeneratedPlan, index: int) -> None:
    expected = preferences_for(index)
    if len(plan.artifacts) != 1:
        raise ValueError(
            f"request {index + 1} must produce exactly one video artifact; "
            f"provider returned {len(plan.artifacts)}"
        )
    artifact = plan.artifacts[0]
    if artifact.orientation != expected["orientation"]:
        raise ValueError(
            f"request {index + 1} needs {expected['orientation']} orientation, "
            f"but provider returned {artifact.orientation!r}"
        )
    if artifact.platform.lower() != expected["platform"]:
        raise ValueError(
            f"request {index + 1} must target YouTube, "
            f"but provider returned {artifact.platform!r}"
        )
    if artifact.quality.lower() not in SUPPORTED_QUALITY:
        raise ValueError(
            f"request {index + 1} returned unsupported render quality "
            f"{artifact.quality!r}"
        )
    if artifact.quality.lower() != expected["quality"]:
        raise ValueError(
            f"request {index + 1} needs {expected['quality']} quality, "
            f"but provider returned {artifact.quality!r}"
        )
    duration = artifact.audio.get("duration_seconds")
    expected_duration = expected["audio"]["duration_seconds"]
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or not math.isclose(duration, expected_duration, abs_tol=0.01)
    ):
        raise ValueError(
            f"request {index + 1} audio plan must set "
            f"audio.duration_seconds={expected_duration}; received {duration!r}"
        )


def generate_plans(requests: Sequence[str], provider) -> list[GeneratedPlan]:
    if len(requests) != VIDEO_COUNT:
        raise ValueError(f"exactly {VIDEO_COUNT} requests are required")
    plans = []
    for index, text in enumerate(requests):
        request = UserRequest(text, preferences_for(index))
        plan = run_pipeline(request, provider)
        _validate_generated_plan(plan, index)
        plans.append(plan)
        layout = preferences_for(index)["orientation"]
        seconds = preferences_for(index)["target_duration_seconds"]
        print(f"Planned {index + 1}/{VIDEO_COUNT}: {layout}, {seconds}s — {plan.artifacts[0].title}")
    return plans


def write_batch(plans: Sequence[GeneratedPlan], output_dir: Path) -> Path:
    if len(plans) != VIDEO_COUNT:
        raise ValueError(f"exactly {VIDEO_COUNT} generated plans are required")
    plan_dir = output_dir / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, plan in enumerate(plans):
        save_plan(plan, plan_dir / f"video-{index + 1:02d}.json")
        rows.extend(rows_for_plan(plan))

    if len(rows) != VIDEO_COUNT:
        raise ValueError(f"expected {VIDEO_COUNT} renderer rows, got {len(rows)}")
    csv_path = output_dir / "youtube-physics-batch.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=RENDER_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="UTF-8 text file with exactly ten non-empty request lines",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/youtube-physics-batch"),
        help="directory for plans and the renderer CSV",
    )
    parser.add_argument("--provider", help="provider factory as module.path:create_provider")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--dry-run",
        action="store_true",
        help="generate plans and validate the renderer CSV without rendering video",
    )
    action.add_argument(
        "--render",
        action="store_true",
        help="generate plans and render all ten videos",
    )
    args = parser.parse_args()

    try:
        requests = read_requests(args.input)
        plans = generate_plans(requests, load_provider(args.provider))
        csv_path = write_batch(plans, args.output_dir)
        print(f"Saved 10 plan files under {args.output_dir / 'plans'}")
        print(f"Renderer batch: {csv_path}")
        if args.dry_run or args.render:
            command = [
                sys.executable,
                "batch_runner.py",
                "--csv",
                str(csv_path),
                "--voice-backend",
                "espeak",
            ]
            if args.dry_run:
                command.append("--dry-run")
            subprocess.run(command, check=True)
    except (OSError, ValueError, TypeError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
