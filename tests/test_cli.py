"""End-to-end tests of the CLI with a FAKE separator (no models / torch needed)."""
import numpy as np
import soundfile as sf

from leadcut import cli
from leadcut.separation import stem_tag

SR = 44100


def make_song(tmp_path, subtype="PCM_16", fmt="WAV", name="song.wav"):
    n = 12 * SR
    t = np.arange(n) / SR
    lead = np.zeros((n, 2), np.float32)
    for a, b in [(2, 4), (7, 9)]:
        lead[int(a * SR) : int(b * SR)] = (0.3 * np.sin(2 * np.pi * 440 * t[int(a * SR) : int(b * SR)]))[:, None]
    acc = np.stack([0.1 * np.sin(2 * np.pi * 110 * t), 0.1 * np.sin(2 * np.pi * 165 * t)], axis=1).astype(np.float32)
    path = tmp_path / name
    sf.write(str(path), lead + acc, SR, subtype=subtype, format=fmt)
    return path, lead, acc


def fake_backend_factory(lead, acc):
    def factory(_model_filename):
        def run(input_wav, out_dir):
            sf.write(str(out_dir / "input_(Vocals)_fake.wav"), lead, SR, subtype="FLOAT")
            sf.write(str(out_dir / "input_(Instrumental)_fake.wav"), acc, SR, subtype="FLOAT")
            return {stem_tag(p): p for p in out_dir.glob("*.wav")}

        return run

    return factory


def test_run_end_to_end_16bit_source(tmp_path, monkeypatch, capsys):
    song, lead, acc = make_song(tmp_path)
    monkeypatch.setattr(cli, "make_backend", fake_backend_factory(lead, acc))
    out = tmp_path / "out.flac"
    rc = cli.main(["run", str(song), "-o", str(out), "--workdir", str(tmp_path / "w"), "--segments", "1:50-4:50".replace("1:50", "0:01.5").replace("4:50", "0:04.5")])
    assert rc == 0
    orig, _ = sf.read(str(song), dtype="float32", always_2d=True)
    res, sr = sf.read(str(out), dtype="float32", always_2d=True)
    assert sr == SR and res.shape == orig.shape
    # untouched part (7-9 s lead is still there!) is bit-identical
    assert np.array_equal(res[5 * SR :], orig[5 * SR :])
    assert np.array_equal(res[: int(1.5 * SR)], orig[: int(1.5 * SR)])
    # the 2-4 s lead is gone from the processed segment (only 16-bit rounding noise left)
    seg = slice(int(2.2 * SR), int(3.8 * SR))
    assert np.max(np.abs(res[seg] - acc[seg])) < 1e-3
    assert np.max(np.abs(orig[seg] - acc[seg])) > 0.2
    text = capsys.readouterr().out
    assert "OK" in text and "FAIL" not in text


def test_auto_detect_then_cache_reuse(tmp_path, monkeypatch, capsys):
    song, lead, acc = make_song(tmp_path)
    monkeypatch.setattr(cli, "make_backend", fake_backend_factory(lead, acc))
    out = tmp_path / "auto.flac"
    args = ["run", str(song), "-o", str(out), "--workdir", str(tmp_path / "w")]
    assert cli.main(args) == 0
    assert "2 segment(s)" in capsys.readouterr().out

    # second run must come from the cache: a backend that explodes proves it
    def boom(_):
        raise AssertionError("separator should not run again")

    monkeypatch.setattr(cli, "make_backend", boom)
    assert cli.main(args + ["--strength", "0.8"]) == 0
    assert "cached" in capsys.readouterr().out

    orig, _ = sf.read(str(song), dtype="float32", always_2d=True)
    res, _ = sf.read(str(out), dtype="float32", always_2d=True)
    assert np.array_equal(res[: int(1.5 * SR)], orig[: int(1.5 * SR)])
    assert np.array_equal(res[int(10 * SR) :], orig[int(10 * SR) :])  # silence region after last phrase


def test_detect_writes_editable_file(tmp_path, monkeypatch):
    song, lead, acc = make_song(tmp_path)
    monkeypatch.setattr(cli, "make_backend", fake_backend_factory(lead, acc))
    seg_file = tmp_path / "s.txt"
    assert cli.main(["detect", str(song), "-o", str(seg_file), "--workdir", str(tmp_path / "w")]) == 0
    lines = [l for l in seg_file.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) == 2
    # file can be fed straight back in
    assert cli.main(["run", str(song), "-o", str(tmp_path / "o.wav"), "--segments", f"@{seg_file}", "--workdir", str(tmp_path / "w"), "--dry-run"]) == 0


def test_lossy_source_still_verifies(tmp_path, monkeypatch, capsys):
    song, lead, acc = make_song(tmp_path, subtype="MPEG_LAYER_III", fmt="MP3", name="song.mp3")
    monkeypatch.setattr(cli, "make_backend", fake_backend_factory(lead, acc))
    out = tmp_path / "o.flac"
    assert cli.main(["run", str(song), "-o", str(out), "--workdir", str(tmp_path / "w"), "--segments", "all"]) in (0,)
    assert "lossy" in capsys.readouterr().out


def test_bad_input_gives_clean_error(tmp_path, capsys):
    assert cli.main(["run", str(tmp_path / "nope.mp3")]) == 2
    assert "error:" in capsys.readouterr().err


def test_models_command(capsys):
    assert cli.main(["models"]) == 0
    assert "roformer-karaoke" in capsys.readouterr().out
