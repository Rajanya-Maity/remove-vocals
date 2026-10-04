# leadcut

**Remove only the lead vocal, only in the parts you choose, and leave everything else untouched.**

[![tests](https://github.com/<your-username>/leadcut/actions/workflows/tests.yml/badge.svg)](https://github.com/<your-username>/leadcut/actions)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![license](https://img.shields.io/badge/license-MIT-green)

Most vocal removers give you one all-or-nothing result: the whole song, processed. leadcut is built for the
case where you want to keep the **chorus and backing voices**, remove just the **lead singer**, and do it only
**where you pick**, for example for karaoke.

Everything runs on your own computer. Nothing is uploaded anywhere.

## How it works

```
original ──► karaoke model ──► lead-vocal estimate
   │                                  │
   └──►  output = original − w(t) · strength · lead_estimate
                  w(t) = 0 outside your segments, 1 inside, smooth ramps at the edges
```

- A pretrained **karaoke model** separates *lead* from *everything else* (backing vocals and instruments stay together).
- leadcut **subtracts only the lead estimate from your original file**, instead of using the model's own "instrumental" output.
  Anything the model did not classify as lead, including high-frequency detail, stays as it was.
- The subtraction is applied **only inside the segments you choose**. Outside them the weight is exactly 0, so those samples are
  **bit-identical** to the input. leadcut checks this on every export and prints the result.
- The crossfade at each segment edge is **linear in amplitude** and sits **inside** the segment. The two signals being blended
  differ only by the lead, so an equal-power fade would bump the level mid-fade.
- Your original is **never resampled, down-mixed or normalised**. Only the model's output is adapted to match it.

## Features

- **Waveform editor in your browser** (`leadcut gui`): drag to create, resize and move segments, zoom, undo, play and
  switch **Original / Lead removed** at the same position.
- **Auto-detect** suggests where the lead sings (adaptive threshold, hysteresis, gap filling, padding). You review and fix it.
- **Strength and crossfade sliders** with a live preview.
- **Lossless export** (FLAC or WAV, 16/24-bit, or 32-bit float WAV) with a built-in untouched-region check.
- **Cached separation:** the slow step runs once per song and model. Changing segments, strength or fade afterwards is instant.
- **Switchable models** (see below), so you can compare which works best on your music.
- **Command-line interface** for scripting, plus `leadcut doctor --fix` to diagnose and repair the installation.
- Works on **CPU or NVIDIA GPU** (same quality, the GPU is faster).

## Install

Requires **Python 3.10 to 3.12** (3.12 recommended) and [ffmpeg](https://ffmpeg.org) on your PATH.

### Windows

```powershell
git clone https://github.com/<your-username>/leadcut.git
cd leadcut
.\setup_windows.bat        # creates .venv, installs, runs the doctor
.\start_windows.bat        # opens the dashboard
```

Install ffmpeg once with `winget install Gyan.FFmpeg`, then open a new terminal.

### macOS / Linux

```bash
git clone https://github.com/<your-username>/leadcut.git
cd leadcut
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[cpu]"        # or ".[gpu]" for NVIDIA CUDA
python -m leadcut doctor --fix
python -m leadcut gui
```

`doctor --fix` checks Python, ffmpeg and the separation library, and installs any missing packages the library needs.
The model (a few hundred MB) downloads automatically on first use and is stored in `~/.cache/leadcut/models`.

## Usage

### Dashboard

```bash
python -m leadcut gui          # opens http://localhost:8765
```

1. Choose a song (or drop a file on the page) and press **Analyse**.
2. Check the auto-detected segments against the waveform. Drag to adjust.
3. Tune **strength** and **crossfade**, and compare **Original / Lead removed**.
4. **Export** a full-quality file.

| Action | How |
|---|---|
| Create a segment | Drag on empty waveform |
| Resize / move | Drag its edge / its body |
| Delete / undo | `Delete` / `Ctrl+Z` |
| Zoom / scroll | Mouse wheel / `Shift` + wheel |
| Play / pause | `Space` |

Segments and settings are saved per song.

### Command line

```bash
python -m leadcut detect "song.flac"                         # writes song.segments.txt to review and edit
python -m leadcut run "song.flac" -o "song_nolead.flac"       # auto-detected segments
python -m leadcut run "song.flac" --segments "0:45-1:30, 2:10-end"
python -m leadcut run "song.flac" --segments @song.segments.txt
python -m leadcut run "song.flac" --segments all             # whole song
python -m leadcut run "song.flac" --dry-run                  # show segments, write nothing
python -m leadcut models                                     # list models
```

| Option | Default | Purpose |
|---|---|---|
| `--strength` | `1.0` | Share of the lead estimate subtracted. `0.85–0.95` keeps more quiet chorus at the cost of more lead residue. |
| `--fade-ms` | `80` | Crossfade length at each segment edge. |
| `--model` | `roformer-karaoke` | Which karaoke model to use. |
| `--detect-threshold` | `-28` | dB below the song's typical lead level that counts as singing. Lower is more sensitive. |
| `--min-gap` / `--min-len` / `--pad` | `0.8` / `0.5` / `0.2` s | Fill short pauses, drop short blips, widen segments. |
| `--bit-depth` | `24` | `16`, `24`, or `32` (float, `.wav` only). |
| `--lead-stem` | auto | Force which model output is the lead, if auto-pick is wrong. |

## Models

Downloaded on first use through [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator). Weights are **not** stored in this repository, and each community model has its own licence, so check it before redistributing anything.

| Key | Model |
|---|---|
| `roformer-karaoke` (default) | Mel-Band Roformer karaoke (aufr33 & viperx) |
| `roformer-karaoke-becruily` | Mel-Band Roformer karaoke (becruily) |
| `roformer-karaoke-gabox` | Mel-Band Roformer karaoke v2 (Gabox) |
| `bs-roformer-karaoke-anvuew` | BS-Roformer karaoke (anvuew) |

Models differ in how gently they treat backing vocals, so try more than one on the same song.

## Limitations

Please read these before relying on the results.

- **Overlapping lead and quiet chorus is the hard case for every current model.** Expect some thinning of the chorus or faint
  lead "ghosts" where the two are sung together. Lower `--strength` and compare models.
- **Segments mostly act as a safety mask.** Where the lead stem is silent, subtracting it changes nothing. Segments protect
  you from model mistakes (for example chorus wrongly assigned to the lead) and keep everything else bit-exact.
- **Auto-detect is a draft, not an oracle.** Always review the segments.
- **Lossy sources (MP3, AAC, MP4) stay lossy-sourced.** leadcut cannot restore what the encoder removed. The output is lossless,
  so nothing is degraded further.
- **Separation quality depends entirely on the pretrained model.** Most were trained mainly on Western music, so results on
  film, ghazal, qawwali or classical recordings (heavy reverb, layered voices, instruments that resemble voices) vary a lot.
  leadcut has not been benchmarked on any dataset.
- **No progress percentage during analysis.** The separation library does not report it; the page shows a timer.

## Troubleshooting

| You see | Do this |
|---|---|
| "separation library could not be loaded ... No module named ..." | `python -m leadcut doctor --fix` |
| "ffmpeg was not found" | Install ffmpeg, then open a **new** terminal |
| First analysis seems stuck | Normal on the first run (model download, then CPU separation). The timer keeps counting. |
| Result sounds like only the music | Open **Advanced** in the dashboard and set the lead stem to `vocals`, or `instrumental`, then re-analyse |
| Install fails on Python 3.13 | Use Python 3.12, which the audio libraries are tested against |
| Anything else | `python -m leadcut doctor` and include its output in an issue |

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The unit and server tests use a **fake separator**, so they need no GPU, models or internet. They cover the splicing maths,
segment parsing and detection, the CLI, the doctor, and the dashboard's HTTP server (upload, analyse, preview with range
requests, export, validation, path-traversal and host checks). The page itself has a real-browser test:

```bash
pip install playwright && playwright install chromium
python tests/browser_e2e.py
```

```
src/leadcut/
  audio_io.py     load/write audio, channel and rate matching, alignment check
  splice.py       the weight mask, render and untouched-region verification
  segments.py     time parsing, merging, lead auto-detection
  separation.py   model registry, audio-separator backend, stem choice, caching
  pipeline.py     render + write + verify (shared by CLI and dashboard)
  server.py       local web server (standard library only)
  static/         the single-file waveform editor
  doctor.py       install checks and auto-repair
  cli.py          command-line interface
tests/            unit, server and browser tests
```

Issues and pull requests are welcome. Useful areas: better lead detection, more models, and comparisons on non-Western music.

## Security

The dashboard binds to `127.0.0.1` only, rejects requests whose `Host` header is not localhost, and requires a custom header
on every POST so other websites cannot drive it. Do not expose it beyond your own computer.

## Legal

leadcut only works on audio files you provide and has **no downloader**. Use songs you have the right to use, such as files you
bought, your own CDs, or official karaoke releases. Do not commit copyrighted audio to this repository. You are responsible for
complying with copyright law and with the licences of any models you download.

## Acknowledgements

Built on [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator), PyTorch, and the community
karaoke models by aufr33 & viperx, becruily, Gabox and anvuew.

## License

MIT. See [LICENSE](LICENSE).
