import csv

from output_pipeline import RENDER_FIELDS
from schema_validator import validate_csv


def test_generated_narration_is_not_limited_by_preset_word_counts(tmp_path, capsys):
    row = {field: "" for field in RENDER_FIELDS}
    row.update({
        "video_id": "free-form-request",
        "concept_type": "GENERIC",
        "header_title": "A request-defined explanation",
        "equations_json": "[]",
        "visual_data_json": "[]",
        "audio_script": "A generated word " * 200,
        "orientation": "vertical",
        "beats_json": "[]",
        "bindings_json": "[]",
        "language": "en",
        "quality": "1080p",
        "platform": "shorts",
        "audio_json": '{"ambient":{"kind":"none"}}',
        "calculation_results_json": "{}",
    })
    path = tmp_path / "generated.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RENDER_FIELDS)
        writer.writeheader()
        writer.writerow(row)

    assert validate_csv(str(path))
    assert "[WARN]" not in capsys.readouterr().out
