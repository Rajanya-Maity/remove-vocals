"""Reading, writing and matching audio.

Design rule: the ORIGINAL song is read once, kept as float32, and never resampled,
down-mixed or normalised. Only the *lead-vocal estimate* (which comes from a model)
is adapted to match the original, never the other way round.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

# libsndfile subtypes that are lossless PCM/float. Anything else (MP3, Vorbis, ...)
# is already lossy, and we say so in the report.
_LOSSLESS_SUBTYPES = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}


@dataclass
class Audio:
    data: np.ndarray  # float32, shape (n_samples, n_channels)
    sr: int
    subtype: str | None = None  # libsndfile subtype if known (e.g. "PCM_16")

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def n_channels(self) -> int:
        return self.data.shape[1]

    @property
    def duration(self) -> float:
        return self.n_samples / self.sr

    @property
    def is_lossless_source(self) -> bool:
        return self.subtype in _LOSSLESS_SUBTYPES


def load_audio(path: str | Path) -> Audio:
    """Load any audio file as float32 (n, ch) at its native sample rate.

    Uses libsndfile (WAV/FLAC/OGG/MP3/...). For formats it cannot read (m4a, aac,
    opus-in-mp4, wma...) we fall back to ffmpeg, which must then be on PATH.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return Audio(data, sr, sf.info(str(path)).subtype)
    except (RuntimeError, OSError):
        return _load_via_ffmpeg(path)


def _load_via_ffmpeg(path: Path) -> Audio:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            f"Could not read {path.name} directly and ffmpeg was not found on PATH. "
            "Install ffmpeg (https://ffmpeg.org) or convert the file to WAV/FLAC first."
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decoded.wav"
        # 32-bit float keeps the decoder output untouched; no resampling (-ar not set).
        cmd = [ffmpeg, "-v", "error", "-y", "-i", str(path), "-vn", "-c:a", "pcm_f32le", str(out)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to decode {path.name}: {res.stderr.strip()}")
        data, sr = sf.read(str(out), dtype="float32", always_2d=True)
    return Audio(data, sr, subtype=None)  # decoded from a (probably) lossy source


def write_audio(path: str | Path, data: np.ndarray, sr: int, bit_depth: int = 24) -> Path:
    """Write lossless audio. Format is chosen from the file extension (.flac or .wav)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".flac":
        if bit_depth not in (16, 24):
            raise ValueError("FLAC supports 16 or 24 bit.")
        subtype = f"PCM_{bit_depth}"
    elif ext == ".wav":
        subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth)
        if subtype is None:
            raise ValueError("WAV supports 16, 24 or 32 (float) bit.")
    else:
        raise ValueError("Output must be .flac or .wav (lossless). Re-encode to MP3 yourself if needed.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def match_channels(stem: np.ndarray, n_channels: int) -> np.ndarray:
    """Make the model's stem have the same channel count as the original."""
    if stem.shape[1] == n_channels:
        return stem
    if n_channels == 1:
        return stem.mean(axis=1, keepdims=True)
    if stem.shape[1] == 1:
        return np.repeat(stem, n_channels, axis=1)
    # e.g. 2-ch stem vs 6-ch original: use the first n channels / pad with the mean
    out = np.zeros((stem.shape[0], n_channels), dtype=stem.dtype)
    k = min(n_channels, stem.shape[1])
    out[:, :k] = stem[:, :k]
    return out


def resample_to(stem: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resample (high quality, zero-phase). No-op when rates already match."""
    if sr_from == sr_to:
        return stem
    g = math.gcd(sr_from, sr_to)
    return signal.resample_poly(stem, sr_to // g, sr_from // g, axis=0).astype(np.float32)


def fit_length(stem: np.ndarray, n: int) -> np.ndarray:
    """Trim or zero-pad to exactly n samples (models can be off by a few samples)."""
    if stem.shape[0] == n:
        return stem
    if stem.shape[0] > n:
        return stem[:n]
    pad = np.zeros((n - stem.shape[0], stem.shape[1]), dtype=stem.dtype)
    return np.concatenate([stem, pad], axis=0)


def estimate_lag(reference: np.ndarray, other: np.ndarray, sr: int, max_lag_ms: float = 50.0) -> int:
    """Estimate the sample offset of `other` relative to `reference` (both (n, ch)).

    Positive result means `other` is DELAYED compared with `reference`.
    Uses up to 30 s taken from the loudest part of `other`. Returns 0 if unsure.
    A correctly working separator returns 0 here; anything else makes subtraction
    leave audible residue, so the CLI warns about it.
    """
    a = reference.mean(axis=1)
    b = other.mean(axis=1)
    n = min(len(a), len(b))
    win = min(n, 30 * sr)
    if win < sr:  # shorter than 1 s: not enough to estimate
        return 0
    # choose the loudest 30 s window of `other` (coarse, 1 s hop)
    hop = sr
    energy = np.array([np.sum(b[i : i + hop] ** 2) for i in range(0, n - hop + 1, hop)])
    cum = np.concatenate([[0.0], np.cumsum(energy)])
    w = win // hop
    best = int(np.argmax(cum[w:] - cum[: len(cum) - w])) * hop if len(cum) > w else 0
    a_seg, b_seg = a[best : best + win], b[best : best + win]
    if np.max(np.abs(b_seg)) < 1e-6:
        return 0
    max_lag = int(max_lag_ms / 1000 * sr)
    corr = signal.fftconvolve(b_seg, a_seg[::-1], mode="full")
    centre = len(a_seg) - 1
    lo, hi = centre - max_lag, centre + max_lag + 1
    return int(np.argmax(corr[lo:hi])) - max_lag
