import numpy as np
import pytest
import soundfile as sf

from leadcut.audio_io import estimate_lag, fit_length, match_channels, resample_to
from leadcut.segments import Segment, detect_lead_segments, merge_segments, parse_segments, parse_time
from leadcut.separation import choose_lead
from leadcut.splice import build_weight, render, verify_untouched

SR = 44100


def tone(freq, seconds, amp=0.3, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ---------------------------------------------------------------- parsing
def test_parse_time_formats():
    assert parse_time("83") == 83
    assert parse_time("1:23") == 83
    assert parse_time("1:23.5") == 83.5
    assert parse_time("1:02:03") == 3723
    assert parse_time("end", 200) == 200
    with pytest.raises(ValueError):
        parse_time("abc")


def test_parse_segments_and_merge():
    segs = parse_segments("0:45-1:30, 2:10 to end ; 1:20-1:40", 180)
    assert segs == [Segment(45, 100), Segment(130, 180)]


def test_parse_segments_file(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("# comment\n0:10-0:20  # inline\n\n0:30-0:40\n")
    assert parse_segments(f"@{f}", 60) == [Segment(10, 20), Segment(30, 40)]


def test_bad_ranges():
    with pytest.raises(ValueError):
        parse_segments("2:00-1:00", 300)
    with pytest.raises(ValueError):
        parse_segments("", 300)


def test_merge_clips_to_duration():
    assert merge_segments([Segment(-5, 3), Segment(9, 50)], 10) == [Segment(0, 3), Segment(9, 10)]


# ---------------------------------------------------------------- splice
def test_weight_is_zero_outside_and_one_inside():
    n = 10 * SR
    w = build_weight(n, SR, [Segment(2, 6)], 0.1)
    assert np.all(w[: 2 * SR] == 0) and np.all(w[6 * SR :] == 0)
    assert np.all(w[int(2.2 * SR) : int(5.8 * SR)] == 1)
    assert 0 < w[2 * SR + 10] < 0.01  # ramp starts at the boundary, from ~0
    assert np.all(np.diff(w[2 * SR : 2 * SR + int(0.1 * SR)]) >= 0)  # monotonic rise


def test_short_segment_shrinks_fade():
    w = build_weight(SR, SR, [Segment(0.5, 0.52)], 0.5)
    assert w.max() <= 1.0 and w.max() > 0.9


def test_render_untouched_is_bit_identical_and_lead_removed():
    n = 8 * SR
    lead = np.zeros((n, 2), np.float32)
    lead[2 * SR : 5 * SR] = tone(440, 3)[:, None]
    acc = np.stack([tone(110, 8, 0.2)] * 2, axis=1)
    mix = acc + lead
    w = build_weight(n, SR, [Segment(1.5, 5.5)], 0.08)
    out = render(mix, lead, w, 1.0)
    assert verify_untouched(mix, out, w) == 0.0
    inside = slice(int(2.2 * SR), int(4.8 * SR))
    assert np.max(np.abs(out[inside] - acc[inside])) < 1e-5


def test_strength_scales_removal():
    n = 4 * SR
    lead = np.stack([tone(440, 4)] * 2, axis=1)
    mix = lead.copy()
    w = build_weight(n, SR, [Segment(0, 4)], 0.0)
    half = render(mix, lead, w, 0.5)
    assert np.allclose(half, 0.5 * mix, atol=1e-6)


def test_fade_is_linear_amplitude_no_level_bump():
    # original and processed differ only by the lead; blend must equal original - w*lead
    n = SR
    lead = np.full((n, 1), 0.2, np.float32)
    mix = np.full((n, 1), 0.5, np.float32)
    w = build_weight(n, SR, [Segment(0, 1)], 0.5)
    out = render(mix, lead, w, 1.0)
    mid = n // 2
    assert out[mid, 0] == pytest.approx(0.3, abs=1e-6)
    assert out[10, 0] > 0.49 and out[n - 10, 0] > 0.49  # starts/ends near the original level


# ---------------------------------------------------------------- audio helpers
def test_estimate_lag_detects_delay():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((20 * SR, 1)).astype(np.float32) * 0.1
    delayed = np.concatenate([np.zeros((50, 1), np.float32), base])[: len(base)]
    assert estimate_lag(base, delayed, SR) == 50
    assert estimate_lag(base, base, SR) == 0


def test_resample_channels_length():
    x = tone(440, 1, sr=48000)[:, None]
    y = resample_to(x, 48000, 44100)
    assert abs(len(y) - 44100) <= 1
    assert match_channels(y, 2).shape[1] == 2
    assert match_channels(np.zeros((10, 2), np.float32), 1).shape == (10, 1)
    assert fit_length(np.zeros((10, 1), np.float32), 15).shape[0] == 15
    assert fit_length(np.zeros((10, 1), np.float32), 5).shape[0] == 5


# ---------------------------------------------------------------- detection
def test_detect_finds_phrases_and_ignores_bleed():
    n = 14 * SR
    lead = np.zeros((n, 2), np.float32)
    for a, b in [(2, 4), (7, 9.5)]:
        lead[int(a * SR) : int(b * SR)] = tone(440, b - a)[:, None]
    lead[int(11 * SR) : int(11.3 * SR)] = tone(300, 0.3, 0.3)[:, None]  # short blip -> ignored
    lead += 0.0005 * np.random.default_rng(1).standard_normal(lead.shape).astype(np.float32)  # bleed
    segs = detect_lead_segments(lead, SR)
    assert len(segs) == 2
    assert 1.6 < segs[0].start < 2.0 and 4.0 < segs[0].end < 4.4
    assert 6.6 < segs[1].start < 7.0 and 9.5 < segs[1].end < 9.9


def test_detect_silence_gives_nothing():
    assert detect_lead_segments(np.zeros((5 * SR, 2), np.float32), SR) == []


# ---------------------------------------------------------------- stem choice
def test_choose_lead():
    p = {"Vocals": 1, "Instrumental": 2}
    assert choose_lead(p, ("vocals", "lead")) == ("Vocals", "Instrumental")
    assert choose_lead({"No Vocals": 1, "Vocals": 2}, ("vocals",))[0] == "Vocals"
    assert choose_lead(p, ("vocals",), override="instr")[0] == "Instrumental"
    with pytest.raises(RuntimeError):
        choose_lead({"a": 1, "b": 2}, ("vocals",))
