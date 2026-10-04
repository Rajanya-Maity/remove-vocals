"""Render + write + verify, shared by the command line and the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_io import Audio, write_audio
from .segments import Segment
from .splice import build_weight, peak, render, verify_untouched


@dataclass
class ExportReport:
    path: Path
    sr: int
    channels: int
    bit_depth: int
    covered_s: float
    peak: float
    clipped: bool
    in_memory_diff: float  # exact check on the array we computed
    on_disk_diff: float  # check after reading the written file back
    on_disk_tol: float

    @property
    def memory_ok(self) -> bool:
        return self.in_memory_diff == 0

    @property
    def disk_ok(self) -> bool:
        return self.on_disk_diff <= self.on_disk_tol


def export_audio(
    original: Audio,
    lead: np.ndarray,
    segments: list[Segment],
    *,
    strength: float,
    fade_s: float,
    out_path: Path,
    bit_depth: int = 24,
) -> ExportReport:
    """Apply the splice, write a lossless file, and verify the untouched regions twice."""
    weight = build_weight(original.n_samples, original.sr, segments, fade_s)
    result = render(original.data, lead, weight, strength)
    in_mem = verify_untouched(original.data, result, weight)
    pk = peak(result)

    write_audio(out_path, result, original.sr, bit_depth)

    back, _ = sf.read(str(out_path), dtype="float32", always_2d=True)
    # A lossless 16/24-bit source round-trips exactly; a decoded MP3/AAC float does not
    # fit a 24-bit grid exactly, so allow one 24-bit step there.
    tol = 0.0 if original.is_lossless_source else 2.0**-23
    on_disk = verify_untouched(original.data, back, weight, tol=tol)

    return ExportReport(
        path=Path(out_path),
        sr=original.sr,
        channels=original.n_channels,
        bit_depth=bit_depth,
        covered_s=sum(s.length for s in segments),
        peak=pk,
        clipped=pk > 1.0,
        in_memory_diff=in_mem,
        on_disk_diff=on_disk,
        on_disk_tol=tol,
    )
