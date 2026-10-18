"""
generate_audio.py — Neural voice synthesis (XTTS-v2) + ambient pad.

Produces two WAVs per row:
    <id>_voice.wav   synthesized speech in the cloned voice
    <id>_pad.wav     low-pass filtered noise bed mixed under the voice

Optional CSV columns:
    language         — 'en', 'hi', 'es', ... (default 'en')
    tts_speed        — speed multiplier (default 1.0)
    tts_temperature  — sampling temperature (default 0.65)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from TTS.api import TTS

try:
    from text_normalizer import normalize as normalize_text
except ImportError:
    def normalize_text(s):
        return s

os.environ.setdefault("COQUI_TOS_AGREED", "1")

CANONICAL_SR   = 44100
_TTS_MODEL: TTS | None = None
_SPEAKER_LATENT: tuple | None = None


def get_tts_model(use_gpu: bool | None = None) -> TTS:
    global _TTS_MODEL
    if _TTS_MODEL is None:
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        print(f"[TTS] Loading XTTS-v2 (GPU={use_gpu})…")
        _TTS_MODEL = TTS(
            "tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=use_gpu,
        )
    return _TTS_MODEL


def _get_speaker_latent(tts: TTS, speaker_wav: str) -> tuple:
    global _SPEAKER_LATENT
    if _SPEAKER_LATENT is None:
        print(f"[TTS] Computing speaker latent from '{speaker_wav}'…")
        gpt_cond_latent, speaker_embedding = (
            tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=[speaker_wav]
            )
        )
        _SPEAKER_LATENT = (gpt_cond_latent, speaker_embedding)
    return _SPEAKER_LATENT


def clean_audio_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip().strip('"\'').strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _seed_from_id(vid_id: str) -> int:
    h = hashlib.sha256(vid_id.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def get_wav_duration(path: str | Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _resample_to_canonical(path: str | Path):
    data, sr = sf.read(str(path))
    if sr != CANONICAL_SR:
        gcd = np.gcd(int(sr), CANONICAL_SR)
        up = CANONICAL_SR // gcd
        down = int(sr) // gcd
        resampled = resample_poly(data, up, down, axis=0)
        resampled = np.clip(resampled, -1.0, 1.0)
        sf.write(str(path), resampled.astype(np.float32),
                 CANONICAL_SR, subtype="PCM_16")


def create_ambient_pad(
    voice_path: str | Path,
    output_pad_path: str | Path,
    sound_plan: dict,
    sample_rate: int = CANONICAL_SR,
):
    data, sr = sf.read(str(voice_path))
    if sr != sample_rate:
        raise ValueError(
            f"Voice sample rate {sr} does not match target {sample_rate}. "
            f"Call _resample_to_canonical first."
        )
    duration = len(data) / sr if data.ndim == 1 else data.shape[0] / sr
    from soundscape import create_pad_wav

    create_pad_wav(duration, str(output_pad_path), sound_plan, int(sr))


def synthesize_audio_for_row(
    row: dict,
    speaker_wav: str | None = None,
    output_dir: str = ".",
) -> tuple[str, str, float]:
    vid_id = (str(row.get("video_id") or "short_video")).strip() or "short_video"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    voice_path = out / f"{vid_id}_voice.wav"
    pad_path   = out / f"{vid_id}_pad.wav"

    script = clean_audio_text(row.get("audio_script", ""))
    script = normalize_text(script)
    if not script:
        raise ValueError(f"Audio script is missing for video ID '{vid_id}'")
    try:
        audio_config = json.loads(row.get("audio_json") or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid audio_json for video ID '{vid_id}': {exc.msg}") from exc
    if not isinstance(audio_config, dict):
        raise ValueError(f"audio_json for video ID '{vid_id}' must be an object")
    speaker_wav = (
        speaker_wav
        or audio_config.get("voice_reference")
        or os.environ.get("SAEM_VOICE_REFERENCE")
    )
    if not speaker_wav:
        raise ValueError(
            f"Audio plan for '{vid_id}' needs a voice_reference or "
            "SAEM_VOICE_REFERENCE for the Coqui backend."
        )
    if not Path(speaker_wav).exists():
        raise FileNotFoundError(
            f"Reference voice sample '{speaker_wav}' not found"
        )

    language    = (str(row.get("language") or "en")).strip() or "en"
    speed       = float(audio_config.get("speed", row.get("tts_speed") or 1.0))
    temperature = float(audio_config.get("temperature", row.get("tts_temperature") or 0.65))
    seed        = _seed_from_id(vid_id)

    print(f"[TTS] Synthesizing '{vid_id}' "
          f"(lang={language}, speed={speed}, temp={temperature}, "
          f"seed={seed})")
    tts = get_tts_model()

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    gpt_cond_latent, speaker_embedding = _get_speaker_latent(tts, speaker_wav)

    def _infer(tts_ref: TTS, gpt, emb) -> np.ndarray:
        base = dict(
            text=script,
            language=language,
            gpt_cond_latent=gpt,
            speaker_embedding=emb,
            temperature=temperature,
            speed=speed,
            repetition_penalty=3.0,
        )
        try:
            result = tts_ref.synthesizer.tts_model.inference(
                **base, enable_text_splitting=True)
        except TypeError:
            result = tts_ref.synthesizer.tts_model.inference(**base)
        wav_raw = result["wav"]
        if isinstance(wav_raw, torch.Tensor):
            wav_raw = wav_raw.cpu().detach().numpy()
        return np.array(wav_raw, dtype=np.float32)

    try:
        wav_data = _infer(tts, gpt_cond_latent, speaker_embedding)
    except Exception as e:
        if "cuda" in str(e).lower():
            print("[TTS] CUDA OOM detected — retrying on CPU")
            global _TTS_MODEL, _SPEAKER_LATENT
            _TTS_MODEL = None
            _SPEAKER_LATENT = None
            tts = get_tts_model(use_gpu=False)
            gpt_cond_latent, speaker_embedding = _get_speaker_latent(
                tts, speaker_wav
            )
            wav_data = _infer(tts, gpt_cond_latent, speaker_embedding)
        else:
            raise

    sf.write(str(voice_path), wav_data, 24000, subtype="PCM_16")
    raw_duration = len(wav_data) / 24000.0
    _resample_to_canonical(voice_path)
    duration = get_wav_duration(voice_path)
    create_ambient_pad(
        voice_path,
        pad_path,
        audio_config.get("ambient", {"kind": "none"}),
        sample_rate=CANONICAL_SR,
    )

    drift = duration - raw_duration
    print(f"[TTS] Completed '{vid_id}' | "
          f"raw {raw_duration:.2f}s → final {duration:.2f}s "
          f"(drift {drift:+.3f}s)")
    return str(voice_path), str(pad_path), duration
