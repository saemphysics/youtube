"""
batch_runner.py — Subprocess orchestrator & FFmpeg audio/video stitcher.

Reads a renderer-adapter CSV emitted from a generated plan, for each row:
  1. Validates the CSV via schema_validator (pre-flight gate)
  2. Synthesizes voice + ambient pad via generate_audio.py (lazy import)
  3. Invokes render_universal.py as a subprocess
  4. Mixes voice + pad onto the raw video with FFmpeg

Flags:
  --dry-run        Validate + list rows, no render
  --only a,b,c     Only render these video_ids
  --resume         Skip rows whose final MP4 exists and is >1MB
  --chunk N        Shard index for parallel CI workers
  --total-chunks M Split into M parallel workers
  --retries K      Retry failing rows K times with exponential backoff
  --draft          Render at 480p15 for fast preview
  --no-voiceover   Skip VoiceoverScene; use manifest durations
  --skip-audio     Generate silent placeholder audio (testing)
  --skip-stitch    Keep raw MP4 without FFmpeg mux
  --orientation    Global override; per-row CSV column wins
  --log-file       Path to write JSON run log
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


# ────────────────────────────────────────────────────────────────────────
# Paths and constants
# ────────────────────────────────────────────────────────────────────────
OUTPUT_DIR       = Path("output_shorts")
CSV_FILE         = None
MEDIA_DIR        = Path("media")
TEMP_AUDIO       = Path("temp_audio")

FFMPEG_PRESET    = "veryfast"
FFMPEG_CRF       = "20"
DEFAULT_FPS      = 30
CLOSING_CARD_SEC = 5.0     # must match render_universal.py outro_duration
MIN_VALID_MP4_MB = 1.0     # files smaller than this are treated as corrupt


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name.strip())


def str_val(row: dict, key: str) -> str:
    v = row.get(key)
    return str(v).strip() if v is not None else ""


def find_rendered_file(vid_id: str) -> Path | None:
    for p in MEDIA_DIR.rglob(f"{vid_id}.mp4"):
        return p
    return None


def probe_fps(video: Path) -> int:
    """Read the frame rate of a video via ffprobe. Falls back to 30."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video)],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        num, den = out.split("/")
        return max(1, int(round(float(num) / float(den))))
    except Exception:
        return DEFAULT_FPS


def is_valid_output(path: Path, min_mb: float = MIN_VALID_MP4_MB) -> bool:
    """True if the file exists and is at least min_mb megabytes."""
    if not path.exists():
        return False
    try:
        return path.stat().st_size >= int(min_mb * 1024 * 1024)
    except OSError:
        return False


# ────────────────────────────────────────────────────────────────────────
# FFmpeg mux
# ────────────────────────────────────────────────────────────────────────
def stitch_av(raw_video: Path, voice: Path, pad: Path, out: Path):
    """Mix voice + pad onto raw video, extend video tail by CLOSING_CARD_SEC,
    and pad audio so the closing card has ambient bed."""
    fps = probe_fps(raw_video)
    pad_samples = int(44100 * CLOSING_CARD_SEC)

    filter_complex = (
        # mix voice + pad at unity gain, no clipping
        "[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=0:"
        "normalize=0[amix];"
        # extend audio with silence so the closing card has bed
        f"[amix]apad=pad_len={pad_samples}[apadded];"
        # safety limiter to prevent inter-sample clipping
        "[apadded]alimiter=limit=0.97[aout];"
        # extend video with last frame
        f"[0:v]tpad=stop_mode=clone:stop_duration={CLOSING_CARD_SEC},"
        f"fps={fps}[vout]"
    )

    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(raw_video),
            "-i", str(voice),
            "-i", str(pad),
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-preset", FFMPEG_PRESET, "-crf", FFMPEG_CRF,
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100",
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(out),
        ],
        check=True,
    )


# ────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Batch production runner for Manim physics videos."
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Renderer-adapter CSV emitted from a generated plan.",
    )
    p.add_argument("--only", type=str, default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--total-chunks", type=int, default=1)
    p.add_argument("--log-file", type=str, default="")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--draft", action="store_true",
                   help="Render at 480p15 (fast preview).")
    p.add_argument("--no-voiceover", action="store_true",
                   help="Skip VoiceoverScene; use manifest durations.")
    p.add_argument("--skip-audio", action="store_true",
                   help="Generate silent placeholder audio (testing).")
    p.add_argument(
        "--voice-backend",
        choices=["coqui", "espeak"],
        default="",
        help="Override the generated audio plan's voice backend.",
    )
    p.add_argument("--skip-stitch", action="store_true",
                   help="Keep raw MP4 without FFmpeg mux.")
    p.add_argument("--orientation", default="",
                   choices=["", "vertical", "horizontal"],
                   help="Global override; per-row CSV column wins.")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────
# Pre-flight
# ────────────────────────────────────────────────────────────────────────
def preflight_csv() -> None:
    """Run schema_validator.validate_csv on the active CSV. Exit 1 on error."""
    from schema_validator import validate_csv

    print(f"[{_ts()}] [preflight] Validating {CSV_FILE}…")
    if not validate_csv(str(CSV_FILE)):
        print(f"[{_ts()}] [preflight] FATAL — schema validation failed. "
              f"Fix the CSV and retry.")
        sys.exit(1)
    print(f"[{_ts()}] [preflight] OK.")


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def process_batch():
    global CSV_FILE
    args = parse_args()
    CSV_FILE = args.csv
    OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_AUDIO.mkdir(exist_ok=True)

    if not CSV_FILE.exists():
        raise FileNotFoundError(f"CSV file '{CSV_FILE}' not found.")

    # Pre-flight (unless we're in a hurry and just want to see the row list)
    if not args.dry_run:
        preflight_csv()

    with CSV_FILE.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    only_set = {sanitize_filename(x) for x in args.only.split(",")
                if x.strip()}
    if only_set:
        rows = [
            r for r in rows
            if sanitize_filename(str_val(r, "video_id")) in only_set
        ]
        if not rows:
            print(f"[{_ts()}] [FATAL] --only matched no rows. "
                  f"Requested: {sorted(only_set)}")
            sys.exit(1)

    rows.sort(key=lambda r: str_val(r, "video_id"))

    if args.total_chunks > 1:
        rows = [
            r for i, r in enumerate(rows)
            if i % args.total_chunks == args.chunk
        ]

    total = len(rows)
    print(f"\n{'=' * 60}")
    print(f"  Production Engine — {total} video(s)")
    if args.total_chunks > 1:
        print(f"  Chunk {args.chunk + 1}/{args.total_chunks}")
    print(f"{'=' * 60}\n")

    runlog: list[dict] = []
    t_start = time.time()
    n_ok = n_fail = n_skip = 0

    for idx, row in enumerate(rows, 1):
        raw_vid_id = str_val(row, "video_id") or f"short_{idx}"
        vid_id = sanitize_filename(raw_vid_id) or f"row_{idx}"
        concept = str_val(row, "concept_type").upper() or "GENERIC"
        final_output = OUTPUT_DIR / f"{vid_id}_final.mp4"

        if args.resume and is_valid_output(final_output):
            print(f"[{_ts()}] [{idx}/{total}] SKIP (Resume)  {vid_id}")
            runlog.append({"video_id": vid_id, "status": "skipped"})
            n_skip += 1
            continue

        if args.dry_run:
            print(f"[{_ts()}] [{idx}/{total}] DRY RUN  {vid_id}  ({concept})")
            runlog.append({
                "video_id": vid_id, "status": "dry_run",
                "concept_type": concept,
            })
            continue

        print(f"[{_ts()}] [{idx}/{total}] BUILD {vid_id}  ({concept})")

        last_err: Exception | None = None
        for attempt in range(1, args.retries + 2):
            try:
                audio_config = _audio_config(row)
                voice_backend = (
                    args.voice_backend
                    or str(audio_config.get("voice_backend", ""))
                    or ("coqui" if audio_config.get("voice_reference") else "espeak")
                )
                _render_one(
                    row, vid_id, final_output,
                    draft=args.draft,
                    no_voiceover=args.no_voiceover,
                    skip_audio=args.skip_audio,
                    voice_backend=voice_backend,
                    skip_stitch=args.skip_stitch,
                    orientation_override=args.orientation,
                )
                runlog.append({
                    "video_id": vid_id, "status": "ok",
                    "attempt": attempt,
                })
                n_ok += 1
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f"[{_ts()}]   ! Attempt {attempt} failed: {e}")
                if attempt <= args.retries:
                    backoff = 2 ** attempt
                    print(f"[{_ts()}]   ... Retrying in {backoff}s")
                    time.sleep(backoff)

        if last_err is not None:
            runlog.append({
                "video_id": vid_id, "status": "error",
                "error": f"{type(last_err).__name__}: {last_err}",
            })
            n_fail += 1
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"  Done in {elapsed/60:.1f} min  —  "
          f"{n_ok} ok, {n_fail} failed, {n_skip} skipped")
    print(f"{'=' * 60}\n")

    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.log_file).write_text(
            json.dumps({
                "elapsed_sec": round(elapsed, 1),
                "summary": {
                    "ok": n_ok, "failed": n_fail, "skipped": n_skip,
                },
                "entries": runlog,
            }, indent=2),
            encoding="utf-8",
        )

    # Clean empty temp dir
    if TEMP_AUDIO.exists() and not any(TEMP_AUDIO.iterdir()):
        TEMP_AUDIO.rmdir()

    if n_fail > 0:
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────────
# Single-row pipeline
# ────────────────────────────────────────────────────────────────────────
def _render_one(row: dict, vid_id: str, final_output: Path,
                draft: bool = False, no_voiceover: bool = False,
                skip_audio: bool = False, voice_backend: str = "coqui",
                skip_stitch: bool = False,
                orientation_override: str = ""):
    voice = pad = None
    raw_output = OUTPUT_DIR / f"{vid_id}_raw.mp4"

    try:
        # ---- 1. Audio ------------------------------------------------
        if skip_audio:
            voice, pad, duration = _make_silent_audio(row, vid_id)
        elif voice_backend == "espeak":
            voice, pad, duration = _synthesize_espeak_for_row(row, vid_id)
        elif voice_backend == "coqui":
            from generate_audio import synthesize_audio_for_row
            voice, pad, duration = synthesize_audio_for_row(
                row,
                output_dir=str(TEMP_AUDIO),
            )
        else:
            raise ValueError(f"Unsupported voice backend: {voice_backend!r}")

        # ---- 2. Render -----------------------------------------------
        ori = (orientation_override
               or str_val(row, "orientation")
               or "vertical").lower()
        if ori not in ("vertical", "horizontal"):
            ori = "vertical"

        options = [str_val(row, f"option_{c}") for c in "abcd"]
        options = [o for o in options if o]

        cmd = [
            sys.executable, "render_universal.py",
            "--video_id",         vid_id,
            "--header_title",     str_val(row, "header_title"),
            "--tagline",          str_val(row, "tagline"),
            "--question_text",    str_val(row, "question_text"),
            "--correct_answer",   str_val(row, "correct_answer"),
            "--equations_json",   str_val(row, "equations_json") or "[]",
            "--visual_data_json", str_val(row, "visual_data_json") or "[]",
            "--beats_json",       str_val(row, "beats_json") or "[]",
            "--bindings_json",    str_val(row, "bindings_json") or "[]",
            "--concept_type",     str_val(row, "concept_type") or "GENERIC",
            "--quality",          str_val(row, "quality") or "1080p",
            "--orientation",      ori,
            "--duration",         f"{duration:.3f}",
            "--options_json",     json.dumps(options),
            "--output",           str(raw_output),
        ]

        fs = str_val(row, "formula_sheet_json")
        if fs:
            cmd += ["--formula_sheet_json", fs]

        mp = str_val(row, "manifest_path")
        if mp:
            cmd += ["--manifest_path", mp]

        if draft:
            cmd += ["--draft"]
        if no_voiceover:
            cmd += ["--no_voiceover"]

        # Belt-and-suspenders: also set SAEM_ORIENTATION in the subprocess env
        env = {**os.environ, "SAEM_ORIENTATION": ori}

        print(f"[{_ts()}]   [render_universal] invoking renderer ({ori})…")
        try:
            subprocess.run(cmd, check=True,
                           stdout=sys.stdout, stderr=sys.stderr,
                           env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"render_universal.py exited {e.returncode}. "
                f"Command: {' '.join(cmd)}"
            ) from e

        if not raw_output.exists():
            raise FileNotFoundError(
                f"Renderer did not produce {raw_output}"
            )

        # ---- 3. Stitch -----------------------------------------------
        if skip_stitch:
            raw_output.replace(final_output)
            print(f"[{_ts()}]   → SKIP-STITCH: {final_output}\n")
            return

        print(f"[{_ts()}]   [FFmpeg] Stitching audio/video…")
        stitch_av(raw_output, Path(voice), Path(pad), final_output)
        print(f"[{_ts()}]   → SUCCESS: {final_output}\n")

    finally:
        # Clean up all temp artifacts, including raw_output on failure
        for p in (voice, pad):
            if p and Path(p).exists():
                Path(p).unlink(missing_ok=True)
        if raw_output.exists():
            raw_output.unlink(missing_ok=True)


def _make_silent_audio(row: dict, vid_id: str) -> tuple[str, str, float]:
    """Generate a silent WAV of a plausible duration for --skip-audio mode."""
    import wave
    duration = 60.0  # seconds — enough for a render test
    sample_rate = 44100
    path = TEMP_AUDIO / f"{vid_id}_voice.wav"
    pad_path = TEMP_AUDIO / f"{vid_id}_pad.wav"
    n_frames = int(duration * sample_rate)

    for out in (path, pad_path):
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16-bit
            w.setframerate(sample_rate)
            w.writeframes(b"\x00\x00" * n_frames)

    return str(path), str(pad_path), duration


def _audio_config(row: dict) -> dict:
    raw = str_val(row, "audio_json")
    if not raw:
        return {}
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid audio_json for '{str_val(row, 'video_id')}': {exc.msg}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"audio_json for '{str_val(row, 'video_id')}' must be an object")
    return config


def _synthesize_espeak_for_row(row: dict, vid_id: str) -> tuple[str, str, float]:
    """Create offline narration and fit its duration to the selected format."""
    import wave

    audio_config = _audio_config(row)
    script = str_val(row, "audio_script")
    if not script:
        raise ValueError(f"Audio script is missing for video ID '{vid_id}'")
    requested_duration = audio_config.get("duration_seconds")
    if requested_duration is not None:
        target_duration = float(requested_duration)
        if not 0.0 < target_duration < 3600.0:
            raise ValueError("audio.duration_seconds must be greater than zero and under one hour")
    TEMP_AUDIO.mkdir(parents=True, exist_ok=True)
    voice_path = TEMP_AUDIO / f"{vid_id}_voice.wav"
    pad_path = TEMP_AUDIO / f"{vid_id}_pad.wav"
    source_path = TEMP_AUDIO / f"{vid_id}_espeak_source.wav"
    script_path = TEMP_AUDIO / f"{vid_id}_espeak_script.txt"
    script_path.write_text(script, encoding="utf-8")

    try:
        speech_command = ["espeak-ng"]
        espeak_voice = audio_config.get("espeak_voice") or str_val(row, "language")
        if espeak_voice:
            speech_command.extend(["-v", str(espeak_voice)])
        speech_rate = audio_config.get("speech_rate_wpm")
        if speech_rate is not None:
            speech_rate = int(speech_rate)
            if not 80 <= speech_rate <= 450:
                raise ValueError("audio.speech_rate_wpm must be between 80 and 450")
            speech_command.extend(["-s", str(speech_rate)])
        speech_command.extend(["-w", str(source_path), "-f", str(script_path)])
        subprocess.run(speech_command, check=True)
        with wave.open(str(source_path), "rb") as source:
            source_duration = source.getnframes() / source.getframerate()
        if source_duration <= 0:
            raise ValueError(f"espeak-ng produced empty narration for '{vid_id}'")

        if requested_duration is None:
            target_duration = source_duration
        else:
            tempo = source_duration / target_duration
            if not 0.5 <= tempo <= 2.0:
                raise ValueError(
                    f"espeak narration for '{vid_id}' needs unsupported tempo "
                    f"factor {tempo:.2f}; revise the request duration or speech rate"
                )
            audio_filter = f"atempo={tempo:.8f},apad,atrim=duration={target_duration}"

        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(source_path),
        ]
        if requested_duration is not None:
            command.extend(["-af", audio_filter])
        command.extend([
            "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(voice_path),
        ])
        subprocess.run(command, check=True)
        from soundscape import create_pad_wav

        create_pad_wav(
            target_duration,
            str(pad_path),
            audio_config.get("ambient", {"kind": "none"}),
        )
    finally:
        source_path.unlink(missing_ok=True)
        script_path.unlink(missing_ok=True)

    return str(voice_path), str(pad_path), target_duration


if __name__ == "__main__":
    process_batch()
