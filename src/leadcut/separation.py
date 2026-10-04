"""Getting a *lead-vocal estimate* out of a song.

Everything model-specific is isolated here. The rest of leadcut only needs one thing:
a float32 array with the same length / rate / channels as the original that contains
(an estimate of) the LEAD vocal only.

Key decisions (each one protects audio quality):
* The separator is fed a 32-bit-float WAV of the original, so nothing is rounded to
  16-bit on the way in, and the stems come back as float too.
* The library's output normalisation is switched off (threshold 1.0). Otherwise it can
  rescale a stem whose peak is above the threshold, and `original - stem` would then
  subtract the wrong amount of signal.
* The lead stem is cached on disk (keyed by file content + model), so trying different
  segments / strength / fade values is instant after the first run.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .audio_io import Audio, fit_length, match_channels, resample_to

# A backend takes (input_wav, output_dir) and returns {stem_tag: path_to_wav}.
# Real one = audio-separator. Tests plug in a fake.
Backend = Callable[[Path, Path], "dict[str, Path]"]

DEFAULT_MODEL_DIR = Path.home() / ".cache" / "leadcut" / "models"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    filename: str  # the file name audio-separator downloads & loads
    description: str
    # Stem tags that mean "lead vocal" for this model, in priority order (case-insensitive).
    lead_keywords: tuple[str, ...] = ("vocals", "lead", "other")


# Curated karaoke (lead-vs-rest) models that audio-separator can download by file name.
# NOTE: these are community models with their own licences. Weights are downloaded on first
# use into ~/.cache/leadcut/models and are never stored in this repository.
MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in [
        ModelSpec(
            "roformer-karaoke",
            "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
            "Mel-Band Roformer karaoke (aufr33 & viperx). Good general default.",
        ),
        ModelSpec(
            "roformer-karaoke-becruily",
            "mel_band_roformer_karaoke_becruily.ckpt",
            "Mel-Band Roformer karaoke (becruily). Worth A/B testing on dense film mixes.",
        ),
        ModelSpec(
            "roformer-karaoke-gabox",
            "mel_band_roformer_karaoke_gabox_v2.ckpt",
            "Mel-Band Roformer karaoke v2 (Gabox). Another strong candidate.",
        ),
        ModelSpec(
            "bs-roformer-karaoke-anvuew",
            "bs_roformer_karaoke_anvuew.ckpt",
            "BS-Roformer karaoke (anvuew). Different architecture, different errors.",
        ),
    ]
}
DEFAULT_MODEL = "roformer-karaoke"


# ------------------------------------------------------------------ stem handling

_TAG_RE = re.compile(r"_\(([^)]+)\)_")


def stem_tag(path: Path) -> str:
    """audio-separator names files like 'input_(Vocals)_modelname.wav' -> 'Vocals'."""
    m = _TAG_RE.search(path.name)
    return m.group(1) if m else path.stem


def choose_lead(stems: dict[str, Path], keywords: tuple[str, ...], override: str | None = None) -> tuple[str, str | None]:
    """Pick the lead-vocal stem. Returns (lead_tag, other_tag_or_None). Raises if unclear."""
    tags = list(stems)
    low = {t: t.lower() for t in tags}

    def _others(lead: str) -> str | None:
        rest = [t for t in tags if t != lead]
        return rest[0] if rest else None

    if override:
        hits = [t for t in tags if override.lower() in low[t]]
        if len(hits) == 1:
            return hits[0], _others(hits[0])
        raise RuntimeError(f"--lead-stem {override!r} matched {len(hits)} of the model outputs {tags}.")

    for kw in keywords:  # exact tag match first
        for t in tags:
            if low[t] == kw.lower():
                return t, _others(t)
    for kw in keywords:  # then substring, but never 'No Vocals' / 'No Other' style tags
        hits = [t for t in tags if kw.lower() in low[t] and not low[t].startswith("no")]
        if len(hits) == 1:
            return hits[0], _others(hits[0])

    raise RuntimeError(
        f"Could not tell which output is the lead vocal. The model produced: {tags}. "
        "Listen to them (they are in the work folder) and re-run with --lead-stem <word>."
    )


def low_freq_fraction(x: np.ndarray, sr: int, cutoff_hz: float = 80.0) -> float:
    """Share of energy below `cutoff_hz` (measured on up to 30 s from the middle).

    A voice has almost nothing below ~80 Hz; bass and kick drum do. We use this only as a
    sanity check that the stem we picked as 'lead' is not actually the accompaniment.
    """
    mono = x.mean(axis=1) if x.ndim == 2 else x
    win = min(len(mono), 30 * sr)
    start = (len(mono) - win) // 2
    seg = mono[start : start + win].astype(np.float64)
    if seg.size < 1024:
        return 0.0
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    total = spec.sum()
    return float(spec[freqs < cutoff_hz].sum() / total) if total > 0 else 0.0


# ---------------------------------------------------------------- real backend


def audio_separator_backend(model_filename: str, model_dir: Path = DEFAULT_MODEL_DIR, log=print) -> Backend:
    """Backend that runs `python-audio-separator` (CPU / CUDA / Apple MPS auto-detected)."""

    def run(input_wav: Path, out_dir: Path) -> dict[str, Path]:
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:  # pragma: no cover - depends on user's environment
            raise RuntimeError(
                f"The separation library could not be loaded: {exc}\n\n"
                "To repair the installation, stop leadcut (Ctrl+C) and run:\n"
                "    python -m leadcut doctor --fix"
            ) from exc

        model_dir.mkdir(parents=True, exist_ok=True)
        wanted = {
            "model_file_dir": str(model_dir),
            "output_dir": str(out_dir),
            "output_format": "WAV",
            "normalization_threshold": 1.0,  # 1.0 = never rescale the stems (see module docstring)
            "amplification_threshold": 0.0,  # never boost
            "use_soundfile": True,  # write float/PCM via soundfile, keeping the input's bit depth
        }
        accepted = inspect.signature(Separator.__init__).parameters
        kwargs = {k: v for k, v in wanted.items() if k in accepted}
        if "use_soundfile" not in accepted:
            log("  ! Your audio-separator is old: stems may be written as 16-bit. Upgrade it for best quality.")

        sep = Separator(**kwargs)
        log(f"  loading model {model_filename} (downloads on first use) ...")
        sep.load_model(model_filename=model_filename)
        log(f"  running on: {getattr(sep, 'torch_device', 'unknown device')}")
        produced = sep.separate(str(input_wav))

        stems: dict[str, Path] = {}
        for item in produced:
            p = Path(item)
            if not p.exists():
                p = out_dir / p.name  # older versions return bare file names
            stems[stem_tag(p)] = p
        return stems

    return run


# ----------------------------------------------------------------- cache + glue


def file_fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:10]


@dataclass
class LeadStem:
    data: np.ndarray  # float32 (n, ch), matched to the original
    folder: Path
    lead_tag: str
    warnings: list[str]
    from_cache: bool


def get_lead_stem(
    source_path: Path,
    original: Audio,
    *,
    model: ModelSpec,
    backend: Backend,
    workdir: Path,
    lead_override: str | None = None,
    log=print,
) -> LeadStem:
    """Return the lead-vocal estimate for `original`, computing (and caching) it if needed."""
    folder = workdir / f"{source_path.stem}-{file_fingerprint(source_path)}" / model.key
    cached = folder / "lead.wav"
    meta_file = folder / "meta.json"

    if cached.exists() and meta_file.exists() and not lead_override:
        data, sr = sf.read(str(cached), dtype="float32", always_2d=True)
        meta = json.loads(meta_file.read_text())
        log(f"  using cached lead stem ({cached})")
        return LeadStem(_adapt(data, sr, original), folder, meta.get("lead_tag", "?"), meta.get("warnings", []), True)

    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "raw"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir()

    # Neutral file name so the song title can never confuse stem-tag parsing.
    input_wav = folder / "input.wav"
    sf.write(str(input_wav), original.data, original.sr, subtype="FLOAT")

    log("  separating (this is the slow step; the result is cached) ...")
    stems = backend(input_wav, raw)
    if not stems:
        raise RuntimeError("The separator produced no output files.")

    lead_tag, other_tag = choose_lead(stems, model.lead_keywords, lead_override)
    lead_raw, lead_sr = sf.read(str(stems[lead_tag]), dtype="float32", always_2d=True)
    warnings: list[str] = []

    if lead_raw.size and float(np.max(np.abs(lead_raw))) >= 0.9999:
        warnings.append("The lead stem peaks at full scale. If the library rescaled it, the subtraction will be slightly off.")

    if other_tag is not None:
        other_raw, other_sr = sf.read(str(stems[other_tag]), dtype="float32", always_2d=True)
        lf_lead, lf_other = low_freq_fraction(lead_raw, lead_sr), low_freq_fraction(other_raw, other_sr)
        if lf_lead > 0.05 and lf_lead > 3 * lf_other:
            warnings.append(
                f"The stem chosen as LEAD ('{lead_tag}') has much more sub-80 Hz energy than '{other_tag}' "
                f"({lf_lead:.0%} vs {lf_other:.0%}). That looks like the accompaniment, not a voice. "
                "If the result sounds wrong, re-run with --lead-stem pointing at the other output."
            )

    lead = _adapt(lead_raw, lead_sr, original)
    sf.write(str(cached), lead, original.sr, subtype="FLOAT")
    meta_file.write_text(json.dumps({"lead_tag": lead_tag, "stems": list(stems), "model": model.filename, "warnings": warnings}))
    return LeadStem(lead, folder, lead_tag, warnings, False)


def _adapt(stem: np.ndarray, stem_sr: int, original: Audio) -> np.ndarray:
    """Bring a model stem to the original's rate, length and channel count (never the reverse)."""
    stem = resample_to(stem, stem_sr, original.sr)
    stem = match_channels(stem, original.n_channels)
    return fit_length(stem, original.n_samples).astype(np.float32, copy=False)
