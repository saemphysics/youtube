import csv
import json

import pytest

from generate_video_batch import (
    VIDEO_COUNT,
    generate_plans,
    preferences_for,
    read_requests,
    write_batch,
)


class BatchProvider:
    def analyze(self, request, knowledge, engine_operations):
        return {
            "intent": "Create an original physics explanation for the user request.",
            "concept_ids": [],
            "source_refs": [],
            "engine_calls": [],
        }

    def compose(self, request, knowledge, analysis, engine_results):
        return [{
            "artifact_id": f"video-{request.preferences['target_duration_seconds']}",
            "title": "A physics idea",
            "tagline": "See the idea clearly",
            "question": request.text,
            "narration": "A clear explanation with a checked example.",
            "equations": [],
            "visuals": [{"type": "graph"}],
            "beats": [{"type": "hook", "text": "Start with a question"}],
            "bindings": [],
            "orientation": request.preferences["orientation"],
            "quality": request.preferences["quality"],
            "platform": request.preferences["platform"],
            "language": "en",
            "audio": {
                "direction": "Clear narration",
                "duration_seconds": request.preferences["audio"]["duration_seconds"],
                "ambient": {"kind": "none"},
            },
            "visual_direction": "Use readable, original diagrams.",
        }]


def test_preferences_assign_five_of_each_requested_video_format():
    preferences = [preferences_for(index) for index in range(VIDEO_COUNT)]

    assert [item["orientation"] for item in preferences] == (
        ["horizontal"] * 5 + ["vertical"] * 5
    )
    assert [item["target_duration_seconds"] for item in preferences] == (
        [180] * 5 + [60] * 5
    )
    assert [item["audio"]["duration_seconds"] for item in preferences] == (
        [175] * 5 + [55] * 5
    )
    assert all(item["platform"] == "youtube" for item in preferences)
    with pytest.raises(ValueError):
        preferences_for(VIDEO_COUNT)


def test_request_file_requires_exactly_ten_nonempty_lines(tmp_path):
    path = tmp_path / "requests.txt"
    path.write_text("\n".join(f"Explain physics idea {i}" for i in range(10)))

    assert len(read_requests(path)) == VIDEO_COUNT

    path.write_text("Only one request")
    with pytest.raises(ValueError, match="exactly 10"):
        read_requests(path)


def test_batch_generates_ten_plans_and_renderer_rows(tmp_path):
    requests = [
        f"Explain a physics concept and one related problem about topic {index}."
        for index in range(VIDEO_COUNT)
    ]
    plans = generate_plans(requests, BatchProvider())
    csv_path = write_batch(plans, tmp_path)

    assert len(plans) == VIDEO_COUNT
    assert len(list((tmp_path / "plans").glob("video-*.json"))) == VIDEO_COUNT
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == VIDEO_COUNT
    assert [row["orientation"] for row in rows] == (
        ["horizontal"] * 5 + ["vertical"] * 5
    )
    assert [json.loads(row["audio_json"])["duration_seconds"] for row in rows] == (
        [175] * 5 + [55] * 5
    )


def test_batch_rejects_provider_that_ignores_required_orientation():
    class WrongFormatProvider(BatchProvider):
        def compose(self, request, knowledge, analysis, engine_results):
            artifacts = super().compose(request, knowledge, analysis, engine_results)
            artifacts[0]["orientation"] = "vertical"
            return artifacts

    requests = [
        f"Explain a physics concept and one related problem about topic {index}."
        for index in range(VIDEO_COUNT)
    ]
    with pytest.raises(ValueError, match="needs horizontal orientation"):
        generate_plans(requests, WrongFormatProvider())


def test_batch_rejects_provider_that_ignores_requested_duration():
    class WrongDurationProvider(BatchProvider):
        def compose(self, request, knowledge, analysis, engine_results):
            artifacts = super().compose(request, knowledge, analysis, engine_results)
            artifacts[0]["audio"]["duration_seconds"] = 45
            return artifacts

    requests = [
        f"Explain a physics concept and one related problem about topic {index}."
        for index in range(VIDEO_COUNT)
    ]
    with pytest.raises(ValueError, match="audio.duration_seconds=175"):
        generate_plans(requests, WrongDurationProvider())
