import re
import wave

import batch_runner


def test_espeak_audio_uses_generated_audio_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_runner, "TEMP_AUDIO", tmp_path)
    calls = []
    pad_settings = []

    def fake_run(command, check):
        calls.append(command)
        if command[0] == "espeak-ng":
            source_path = command[command.index("-w") + 1]
            with wave.open(source_path, "wb") as source:
                source.setnchannels(1)
                source.setsampwidth(2)
                source.setframerate(44100)
                source.writeframes(b"\x00\x00" * (54 * 44100))
        else:
            output_path = command[-1]
            target = float(re.search(r"atrim=duration=([0-9.]+)", command[command.index("-af") + 1]).group(1))
            with wave.open(output_path, "wb") as voice:
                voice.setnchannels(1)
                voice.setsampwidth(2)
                voice.setframerate(44100)
                voice.writeframes(b"\x00\x00" * int(target * 44100))

    def fake_create_pad(duration, path, sound_plan, sample_rate=44100):
        pad_settings.append((duration, sound_plan))
        with wave.open(path, "wb") as pad:
            pad.setnchannels(1)
            pad.setsampwidth(2)
            pad.setframerate(sample_rate)
            pad.writeframes(b"\x00\x00" * int(duration * sample_rate))

    monkeypatch.setattr(batch_runner.subprocess, "run", fake_run)
    monkeypatch.setattr("soundscape.create_pad_wav", fake_create_pad)
    script = "Explain this generated concept."
    voice, pad, duration = batch_runner._synthesize_espeak_for_row(
        {
            "orientation": "horizontal",
            "language": "en",
            "audio_script": script,
            "audio_json": (
                '{"espeak_voice":"en-us","speech_rate_wpm":180,'
                '"duration_seconds":48.5,"ambient":{"kind":"none"}}'
            ),
        },
        "test-short",
    )

    assert duration == 48.5
    assert len(calls) == 2
    assert calls[0][:5] == ["espeak-ng", "-v", "en-us", "-s", "180"]
    assert calls[1][0] == "ffmpeg"
    assert pad_settings == [(48.5, {"kind": "none"})]
    with wave.open(voice, "rb") as result:
        assert result.getnframes() / result.getframerate() == 48.5
    with wave.open(pad, "rb") as result:
        assert result.getnframes() / result.getframerate() == 48.5
    assert not list(tmp_path.glob("*_espeak_*"))
