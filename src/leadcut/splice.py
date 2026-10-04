"""The heart of the tool: remove the lead vocal only inside chosen segments.

The maths, in one line
----------------------
    output = original - w(t) * strength * lead_estimate

where w(t) is 0 outside the segments, 1 inside, and a smooth 0->1 / 1->0 ramp at the edges.

Why this is better than "cut to a processed copy and back"
-----------------------------------------------------------
* Outside the segments w == 0, so the output sample is the ORIGINAL sample, bit for bit.
* The crossfade is built into w(t), so it is *amplitude-complementary*: mixing
  (1-w)*original + w*(original - strength*lead) gives exactly the formula above.
  The two signals being blended differ only by the lead, so they are highly correlated;
  a linear-amplitude fade is therefore the right kind (an equal-power fade, meant for
  uncorrelated material, would bump the level by up to 3 dB in the middle of the fade).
* The ramps live INSIDE each segment, so the guarantee is strict:
  every sample outside the segments you asked for is untouched.
"""

from __future__ import annotations

import numpy as np

from .segments import Segment


def build_weight(n_samples: int, sr: int, segments: list[Segment], fade_s: float) -> np.ndarray:
    """Per-sample weight in [0, 1]. Ramps are raised-cosine and fully inside each segment."""
    w = np.zeros(n_samples, dtype=np.float32)
    fade = int(round(fade_s * sr))
    for seg in segments:
        s = max(0, int(round(seg.start * sr)))
        e = min(n_samples, int(round(seg.end * sr)))
        if e <= s:
            continue
        f = min(fade, (e - s) // 2)  # very short segment: shrink the fade instead of overlapping
        region = np.ones(e - s, dtype=np.float32)
        if f > 0:
            ramp = (0.5 - 0.5 * np.cos(np.pi * (np.arange(f) + 0.5) / f)).astype(np.float32)
            region[:f] = ramp
            region[-f:] = ramp[::-1]
        w[s:e] = np.maximum(w[s:e], region)
    return w


def render(original: np.ndarray, lead: np.ndarray, weight: np.ndarray, strength: float) -> np.ndarray:
    """Subtract the weighted lead estimate from the original. Untouched samples are copied as-is."""
    if original.shape != lead.shape:
        raise ValueError(f"Shape mismatch: original {original.shape} vs lead {lead.shape}")
    out = original.copy()
    active = weight > 0
    if active.any():
        out[active] = original[active] - (weight[active] * np.float32(strength))[:, None] * lead[active]
    return out


def verify_untouched(original: np.ndarray, result: np.ndarray, weight: np.ndarray, tol: float = 0.0) -> float:
    """Largest absolute difference between result and original OUTSIDE the segments.

    With tol=0 this is an exact, sample-for-sample check. Pass a tiny tolerance
    (e.g. 2**-23) when `result` was read back from a 24-bit file.
    """
    outside = weight == 0
    if not outside.any():
        return 0.0
    diff = np.max(np.abs(result[outside] - original[outside]))
    return float(diff)


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x))) if x.size else 0.0
