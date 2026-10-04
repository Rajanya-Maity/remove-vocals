"""Time segments: parsing user input, merging, and auto-detecting where the lead sings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, order=True)
class Segment:
    start: float  # seconds
    end: float  # seconds

    @property
    def length(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:
        return f"{format_time(self.start)}-{format_time(self.end)}"


# --------------------------------------------------------------------------- time


def parse_time(text: str, duration: float | None = None) -> float:
    """'83', '1:23', '1:23.5' or '1:02:03' -> seconds. 'end' -> duration."""
    t = text.strip().lower()
    if t in ("end", "e"):
        if duration is None:
            raise ValueError("'end' used but song duration is unknown")
        return duration
    parts = t.split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Bad time value: {text!r}")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Bad time value: {text!r}") from exc
    if any(n < 0 for n in nums):
        raise ValueError(f"Negative time: {text!r}")
    seconds = 0.0
    for n in nums:
        seconds = seconds * 60 + n
    return seconds


def format_time(seconds: float) -> str:
    """Seconds -> 'm:ss.mmm' (always with milliseconds so round-trips are lossless enough)."""
    seconds = max(0.0, seconds)
    m = int(seconds // 60)
    s = seconds - 60 * m
    return f"{m}:{s:06.3f}"


# ---------------------------------------------------------------------- parsing

_RANGE_SPLIT = re.compile(r"\s*(?:-|–|—|\bto\b)\s*")


def parse_segments(spec: str, duration: float) -> list[Segment]:
    """Parse '0:45-1:30, 2:10-end', 'all', or '@file.txt' (one range per line, # comments)."""
    spec = spec.strip()
    if spec.startswith("@"):
        spec = Path(spec[1:]).read_text(encoding="utf-8")
    if spec.lower() == "all":
        return [Segment(0.0, duration)]

    # strip '# comments' per line FIRST (a comment may itself contain commas), then split ranges
    cleaned = "\n".join(line.split("#", 1)[0] for line in spec.splitlines())
    segs: list[Segment] = []
    for raw in re.split(r"[,;\n]", cleaned):
        line = raw.strip()
        if not line:
            continue
        pieces = _RANGE_SPLIT.split(line)
        if len(pieces) != 2:
            raise ValueError(f"Could not read range {raw.strip()!r}. Use the form 0:45-1:30")
        a, b = parse_time(pieces[0], duration), parse_time(pieces[1], duration)
        if b <= a:
            raise ValueError(f"Range {raw.strip()!r} ends before it starts")
        segs.append(Segment(a, b))
    if not segs:
        raise ValueError("No segments found in the segment specification")
    return merge_segments(segs, duration)


def merge_segments(segs: list[Segment], duration: float, gap: float = 0.0) -> list[Segment]:
    """Sort, clip to [0, duration] and merge segments that overlap or sit within `gap` seconds."""
    clipped = [Segment(max(0.0, s.start), min(duration, s.end)) for s in segs]
    clipped = sorted(s for s in clipped if s.end > s.start)
    merged: list[Segment] = []
    for s in clipped:
        if merged and s.start <= merged[-1].end + gap:
            merged[-1] = Segment(merged[-1].start, max(merged[-1].end, s.end))
        else:
            merged.append(s)
    return merged


def write_segments_file(path: str | Path, segs: list[Segment]) -> None:
    lines = ["# leadcut segments: lead vocal is removed ONLY inside these ranges", "# Edit freely, then pass with --segments @this_file"]
    lines += [str(s) for s in segs]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------------------------------------------------- detection


def detect_lead_segments(
    lead: np.ndarray,
    sr: int,
    *,
    threshold_db: float = -28.0,
    hysteresis_db: float = 4.0,
    frame_s: float = 0.05,
    min_gap_s: float = 0.8,
    min_len_s: float = 0.5,
    pad_s: float = 0.2,
) -> list[Segment]:
    """Suggest where the lead vocal is active, from the model's lead-vocal estimate.

    How it works
    ------------
    1. Take the loudness (RMS, dB) of the lead stem in 50 ms frames.
    2. The reference level is the 90th percentile of the *non-silent* frames, i.e. "how loud
       the lead normally is in this song". The on-threshold sits `threshold_db` below it
       (-28 dB by default), so the detector adapts to each song instead of using a fixed dBFS.
    3. Hysteresis: a frame switches ON above the threshold and only switches OFF once it
       drops `hysteresis_db` below it, which stops flickering on note tails.
    4. Gaps shorter than `min_gap_s` are filled (breaths / short pauses inside a phrase),
       islands shorter than `min_len_s` are dropped (model blips / chorus bleed),
       and each segment is widened by `pad_s` (lead reverb tails, model onset error).

    This is a heuristic. It is meant to give you a good first draft that you review.
    """
    mono = lead.mean(axis=1) if lead.ndim == 2 else lead
    duration = len(mono) / sr
    hop = max(1, int(frame_s * sr))
    n_frames = len(mono) // hop
    if n_frames < 2:
        return []

    frames = mono[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    level = 20 * np.log10(rms + 1e-12)

    audible = level > -60.0  # ignore digital silence / noise floor when picking the reference
    if not audible.any():
        return []
    reference = np.percentile(level[audible], 90)
    on_thr = reference + threshold_db
    off_thr = on_thr - hysteresis_db

    active = np.zeros(n_frames, dtype=bool)
    state = False
    for i, lv in enumerate(level):
        if not state and lv >= on_thr:
            state = True
        elif state and lv < off_thr:
            state = False
        active[i] = state

    # runs of True -> (start_frame, end_frame)
    edges = np.diff(np.concatenate([[0], active.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = [Segment(s * frame_s, e * frame_s) for s, e in zip(starts, ends)]

    runs = merge_segments(runs, duration, gap=min_gap_s)  # fill short gaps
    runs = [r for r in runs if r.length >= min_len_s]  # drop blips
    runs = [Segment(r.start - pad_s, r.end + pad_s) for r in runs]  # widen
    return merge_segments(runs, duration)
