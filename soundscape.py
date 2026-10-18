#!/usr/bin/env python3
"""Voice mastering and request-defined sound-bed synthesis.

Public functions used by the pipeline:
    master_voice_only(input_wav, output_wav) -> duration_sec
    create_pad_wav(duration_sec, output_wav, sound_plan)
"""
from __future__ import annotations

import logging
import wave

import numpy as np

try:
    from scipy.io import wavfile
    from scipy.signal import butter, sosfilt, resample_poly
    HAS_SCIPY = True
except ImportError:  # pragma: no cover - fallback path used in lightweight environments
    wavfile = None
    butter = None
    sosfilt = None
    resample_poly = None
    HAS_SCIPY = False

try:
    import pyloudnorm as pyln
    HAS_PYLN = True
except ImportError:
    HAS_PYLN = False


log = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────
CANONICAL_SR = 44100
PEAK_TARGET  = 0.891           # ≈ -1 dBFS sample-peak ceiling
LUFS_TARGET  = -14.0           # YouTube normalization target

# Platform-specific loudness targets
LOUDNESS_PRESETS = {
    "youtube":   -14.0,
    "tiktok":    -11.0,
    "instagram": -12.0,
    "podcast":   -16.0,
    "broadcast": -23.0,        # EBU R128
    "default":   LUFS_TARGET,
}


# ────────────────────────────────────────────────────────────────────────
# Filter primitives
# ────────────────────────────────────────────────────────────────────────
def _simple_lowpass(sig: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    if sig.size == 0:
        return sig.astype(np.float32)
    if cutoff_hz <= 0 or cutoff_hz >= sample_rate / 2:
        return sig.astype(np.float32)
    window = min(31, max(3, int(sample_rate / cutoff_hz)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(sig.astype(np.float64), (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _highpass(sig: np.ndarray, sr: int,
              cutoff: float = 80.0) -> np.ndarray:
    if not HAS_SCIPY or butter is None or sosfilt is None:
        return sig.astype(np.float32) - _simple_lowpass(sig, cutoff, sr)
    nyq = sr / 2.0
    cutoff = max(1.0, min(cutoff, nyq * 0.95))
    sos = butter(2, cutoff / nyq, btype="high", output="sos")
    return sosfilt(sos, sig).astype(np.float32)


def _lowpass(sig: np.ndarray, sr: int,
             cutoff: float = 12000.0) -> np.ndarray:
    if not HAS_SCIPY or butter is None or sosfilt is None:
        return _simple_lowpass(sig, cutoff, sr)
    nyq = sr / 2.0
    cutoff = min(cutoff, nyq * 0.95)
    sos = butter(2, cutoff / nyq, btype="low", output="sos")
    return sosfilt(sos, sig).astype(np.float32)


def _dc_remove(sig: np.ndarray) -> np.ndarray:
    return (sig - float(sig.mean())).astype(np.float32)


def _sanitize(sig: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with 0 so downstream math can't propagate them."""
    if not np.all(np.isfinite(sig)):
        n_bad = int((~np.isfinite(sig)).sum())
        log.warning("sanitize: %d non-finite samples replaced with 0", n_bad)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig.astype(np.float32)


def _compress(sig: np.ndarray, threshold: float = 0.25,
              ratio: float = 3.0) -> np.ndarray:
    """Hard-knee compressor (was misnamed _soft_compress)."""
    out = sig.copy()
    mask = np.abs(sig) > threshold
    frac = float(mask.mean())
    if frac > 0.05:
        log.debug("compress: %.1f%% over %.2f", frac * 100, threshold)
    over = np.abs(sig[mask]) - threshold
    out[mask] = np.sign(sig[mask]) * (threshold + over / ratio)
    return out


_LUFS_METER_CACHE = {}

def _get_meter(sr: int):
    if not HAS_PYLN:
        return None
    if sr not in _LUFS_METER_CACHE:
        try:
            _LUFS_METER_CACHE[sr] = pyln.Meter(sr)
        except Exception:
            return None
    return _LUFS_METER_CACHE[sr]


def _measure_lufs(sig: np.ndarray, sr: int) -> float:
    """Measure integrated loudness in LUFS. Uses pyloudnorm if available,
    otherwise falls back to an RMS-based approximation."""
    if HAS_PYLN:
        meter = _get_meter(sr)
        if meter is not None:
            try:
                return float(meter.integrated_loudness(sig))
            except Exception as e:
                log.warning("pyloudnorm failed (%s); falling back to RMS", e)
    rms = float(np.sqrt(np.mean(sig ** 2) + 1e-12))
    if rms < 1e-6:
        return -120.0
    return 20.0 * np.log10(rms) + 3.0     # +3 ≈ RMS→LUFS on speech


def _loudness_normalize(sig: np.ndarray, sr: int,
                        target_lufs: float = LUFS_TARGET) -> np.ndarray:
    """Scale signal so its measured loudness hits target_lufs.
    True LUFS if pyloudnorm is installed; RMS approximation otherwise."""
    current = _measure_lufs(sig, sr)
    if current < -100.0:
        return sig
    delta_db = target_lufs - current
    gain = float(10.0 ** (delta_db / 20.0))
    gain = float(np.clip(gain, 0.05, 20.0))
    log.debug("loudness: %.1f LUFS → %.1f LUFS (gain %.2f×)",
              current, target_lufs, gain)
    return (sig * gain).astype(np.float32)


def _peak_limit(sig: np.ndarray, ceiling: float = PEAK_TARGET) -> np.ndarray:
    peak = float(np.max(np.abs(sig)))
    if peak > ceiling:
        sig = sig * (ceiling / peak)
    return sig.astype(np.float32)


def _voice_fade(sig: np.ndarray, sr: int,
                fade_in_ms: float = 5.0,
                fade_out_ms: float = 25.0) -> np.ndarray:
    """Asymmetric fade — very short in, slightly longer out."""
    n = len(sig)
    in_s = min(int(sr * fade_in_ms / 1000.0), n // 2)
    out_s = min(int(sr * fade_out_ms / 1000.0), n - in_s)
    env = np.ones(n, dtype=np.float32)
    if in_s > 0:
        env[:in_s] = np.linspace(0, 1, in_s)
    if out_s > 0:
        env[-out_s:] = np.linspace(1, 0, out_s)
    return (sig * env).astype(np.float32)


def _trim_silence(sig: np.ndarray, sr: int, threshold: float = 0.005,
                  min_silence_ms: float = 80.0) -> np.ndarray:
    """Trim leading/trailing silence below `threshold` amplitude."""
    if sig.size == 0:
        return sig
    min_n = int(sr * min_silence_ms / 1000.0)
    above = np.where(np.abs(sig) > threshold)[0]
    if above.size == 0:
        return sig
    start = max(0, int(above[0]) - min_n // 2)
    end = min(len(sig), int(above[-1]) + min_n // 2)
    return sig[start:end].astype(np.float32)


def _resample(sig: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return sig.astype(np.float32)
    if not HAS_SCIPY or resample_poly is None:
        ratio = sr_out / sr_in
        indices = np.linspace(0, len(sig) - 1, int(round(len(sig) * ratio)), dtype=np.float64)
        out = np.interp(indices, np.arange(len(sig)), sig.astype(np.float64))
        return np.clip(out, -1.0, 1.0).astype(np.float32)
    g = int(np.gcd(sr_in, sr_out))
    out = resample_poly(sig, sr_out // g, sr_in // g)
    # resample_poly can overshoot ±1; clip before downstream processing
    return np.clip(out, -1.0, 1.0).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Request-defined sound bed
# ────────────────────────────────────────────────────────────────────────
def generate_ambient_pad(
    duration_sec: float,
    sound_plan: dict,
    sample_rate: int = CANONICAL_SR,
) -> np.ndarray:
    duration_sec = float(duration_sec)
    sample_rate = int(sample_rate)
    if not np.isfinite(duration_sec) or duration_sec < 0:
        raise ValueError("duration_sec must be finite and non-negative")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if not isinstance(sound_plan, dict):
        raise TypeError("sound_plan must be a JSON object")

    kind = str(sound_plan.get("kind", "none")).lower()
    n = int(sample_rate * duration_sec)
    if kind == "none":
        return np.zeros(n, dtype=np.float32)
    if kind not in {"noise", "tones"}:
        raise ValueError(f"unsupported sound-bed kind: {kind!r}")

    gain_db = float(sound_plan.get("gain_db"))
    if not np.isfinite(gain_db) or not -80.0 <= gain_db <= 0.0:
        raise ValueError("sound-bed gain_db must be between -80 and 0 dB")
    t = np.arange(n, dtype=np.float64) / sample_rate
    if kind == "noise":
        cutoff_hz = float(sound_plan.get("cutoff_hz"))
        if not np.isfinite(cutoff_hz) or not 20.0 <= cutoff_hz < sample_rate / 2:
            raise ValueError("noise cutoff_hz must be finite and below Nyquist")
        seed = sound_plan.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("noise sound beds require an integer seed")
        white_noise = np.random.default_rng(seed).normal(size=n).astype(np.float32)
        if HAS_SCIPY and butter is not None and sosfilt is not None:
            sos = butter(2, cutoff_hz / (sample_rate / 2), btype="low", output="sos")
            signal = sosfilt(sos, white_noise).astype(np.float64)
        else:
            signal = _simple_lowpass(white_noise, cutoff_hz, sample_rate).astype(np.float64)
    else:
        frequencies = sound_plan.get("frequencies_hz")
        levels = sound_plan.get("levels")
        if (
            not isinstance(frequencies, list)
            or not frequencies
            or not isinstance(levels, list)
            or len(levels) != len(frequencies)
        ):
            raise ValueError("tone sound beds require matching frequencies_hz and levels arrays")
        signal = np.zeros(n, dtype=np.float64)
        for frequency, level in zip(frequencies, levels):
            frequency = float(frequency)
            level = float(level)
            if not np.isfinite(frequency) or not 20.0 <= frequency < sample_rate / 2:
                raise ValueError("tone frequencies must be finite and below Nyquist")
            if not np.isfinite(level) or level < 0.0:
                raise ValueError("tone levels must be finite and non-negative")
            signal += level * np.sin(2.0 * np.pi * frequency * t)

    rms = float(np.sqrt(np.mean(signal ** 2) + 1e-12))
    if rms == 0.0:
        return np.zeros(n, dtype=np.float32)
    signal = signal / rms * (10.0 ** (gain_db / 20.0))
    return np.clip(signal, -1.0, 1.0).astype(np.float32)


def _write_int16_wav(output_wav: str, sample_rate: int, signal: np.ndarray) -> None:
    with wave.open(output_wav, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes((np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16).tobytes())


def master_voice_only(input_wav: str, output_wav: str) -> float:
    with wave.open(input_wav, "rb") as src:
        frames = src.readframes(src.getnframes())
        dtype = np.dtype(f"<i{src.getsampwidth()}")
        signal = np.frombuffer(frames, dtype=dtype).astype(np.float64) / float(1 << (8 * src.getsampwidth() - 1))
        sample_rate = src.getframerate()
    signal = _sanitize(signal)
    signal = _lowpass(signal, sample_rate, cutoff=12000.0)
    signal = _peak_limit(signal)
    signal = _voice_fade(signal, sample_rate)
    _write_int16_wav(output_wav, sample_rate, signal)
    return signal.size / float(sample_rate)


def create_pad_wav(
    duration_sec: float,
    output_wav: str,
    sound_plan: dict,
    sample_rate: int = CANONICAL_SR,
) -> None:
    signal = generate_ambient_pad(duration_sec, sound_plan, sample_rate)
    _write_int16_wav(output_wav, sample_rate, signal)
