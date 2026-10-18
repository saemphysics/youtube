#!/usr/bin/env python3
"""Generate render-ready CSV rows from the reviewed matrix and authored lessons."""
from __future__ import annotations

import argparse
from pathlib import Path

from generate_lecture_series import export, export_one
from matrix_engine import Recommendation, recommend, sample_lessons

ROOT = Path(__file__).resolve().parent


def generate(
    topic: str,
    *,
    count: int = 1,
    seed: int = 2026,
    orientation: str = "vertical",
    output_dir: Path = ROOT,
    output: Path | None = None,
) -> tuple[tuple[Path, ...], Recommendation, tuple[str, ...]]:
    recommendation = recommend(topic)
    lesson_ids = sample_lessons(recommendation, count=count, seed=seed)

    if output is not None:
        if orientation == "both":
            raise ValueError("choose one orientation when specifying an output file")
        paths = (export_one(output, orientation, lesson_ids),)
    elif orientation == "both":
        outputs = export(output_dir, lesson_ids=lesson_ids)
        paths = (outputs["horizontal"], outputs["vertical"])
    else:
        path = output_dir / f"content_batch_{orientation}.csv"
        paths = (export_one(path, orientation, lesson_ids),)

    return paths, recommendation, lesson_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", required=True, help="topic or keywords from the reviewed matrix")
    parser.add_argument("--count", type=int, default=1, help="number of distinct matched lessons")
    parser.add_argument("--seed", type=int, default=2026, help="reproducible sampling seed")
    parser.add_argument(
        "--orientation",
        choices=("vertical", "horizontal", "both"),
        default="vertical",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional exact CSV path; only valid with one orientation",
    )
    args = parser.parse_args()

    try:
        paths, recommendation, lesson_ids = generate(
            args.topic,
            count=args.count,
            seed=args.seed,
            orientation=args.orientation,
            output_dir=args.output_dir,
            output=args.output,
        )
    except ValueError as exc:
        parser.error(str(exc))

    print("Matched concepts:", ", ".join(
        match.concept.concept_id for match in recommendation.concept_matches
    ))
    print("Selected lessons:", ", ".join(lesson_ids))
    if recommendation.engine_weights:
        print("Engine recommendation weights (not calibrated probabilities):")
        for name, weight in recommendation.engine_weights[:5]:
            print(f"  {name}: {weight:.3f}")
    else:
        print("Engine recommendation: no registered solver is mapped to this concept.")
    print("Sources:", ", ".join(recommendation.source_refs) or "none mapped")
    for path in paths:
        print(f"CSV: {path}")


if __name__ == "__main__":
    main()
