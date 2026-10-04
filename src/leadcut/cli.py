"""Command line interface.

    leadcut run    song.flac  -o song_nolead.flac  --segments auto
    leadcut detect song.flac                       # writes song.segments.txt to review/edit
    leadcut models                                 # list built-in models
    leadcut gui                                    # waveform editor in your browser
    leadcut doctor --fix                           # check/repair the installation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


from . import __version__
from .audio_io import estimate_lag, load_audio
from .segments import detect_lead_segments, merge_segments, parse_segments, write_segments_file
from .pipeline import export_audio
from .separation import DEFAULT_MODEL, MODELS, audio_separator_backend, get_lead_stem


def make_backend(model_filename: str):
    """Indirection so tests can swap the real (heavy) separator for a fake one."""
    return audio_separator_backend(model_filename)


def _log(msg: str = "") -> None:
    print(msg, flush=True)


def _prepare(args):
    src = Path(args.input)
    _log(f"Reading {src.name} ...")
    original = load_audio(src)
    _log(
        f"  {original.sr} Hz, {original.n_channels} ch, {original.duration:.1f} s"
        + ("" if original.is_lossless_source else "  (source is lossy: leadcut can't restore what the encoder already removed)")
    )
    if args.model not in MODELS:
        raise SystemExit(f"Unknown model {args.model!r}. Run `leadcut models` to see the choices.")
    model = MODELS[args.model]
    lead = get_lead_stem(
        src,
        original,
        model=model,
        backend=lambda inp, out: make_backend(model.filename)(inp, out),  # built only if no cache
        workdir=Path(args.workdir),
        lead_override=args.lead_stem,
        log=_log,
    )
    for w in lead.warnings:
        _log(f"  WARNING: {w}")
    lag = estimate_lag(original.data, lead.data, original.sr)
    if abs(lag) > max(1, int(0.001 * original.sr)):
        _log(
            f"  WARNING: the lead stem looks {lag} samples ({lag / original.sr * 1000:.1f} ms) out of step with the "
            "original. Subtraction will leave residue. Try another model."
        )
    return src, original, lead


def _detect(args, original, lead):
    return detect_lead_segments(
        lead.data,
        original.sr,
        threshold_db=args.detect_threshold,
        min_gap_s=args.min_gap,
        min_len_s=args.min_len,
        pad_s=args.pad,
    )


def _print_segments(segs, duration):
    covered = sum(s.length for s in segs)
    for s in segs:
        _log(f"    {s}   ({s.length:.1f} s)")
    _log(f"  {len(segs)} segment(s), {covered:.1f} s of {duration:.1f} s will be processed ({covered / duration:.0%}).")


def cmd_models(_args) -> int:
    _log("Built-in models (downloaded on first use; each has its own licence):\n")
    for key, m in MODELS.items():
        star = "  (default)" if key == DEFAULT_MODEL else ""
        _log(f"  {key}{star}\n      {m.description}\n      file: {m.filename}")
    return 0


def cmd_detect(args) -> int:
    src, original, lead = _prepare(args)
    segs = _detect(args, original, lead)
    out = Path(args.output) if args.output else src.with_suffix(".segments.txt")
    write_segments_file(out, segs)
    _log("\nSuggested segments (where the lead vocal is singing):")
    _print_segments(segs, original.duration)
    _log(f"\nSaved to {out}. Edit it if needed, then:\n  leadcut run {src} --segments @{out}")
    return 0


def cmd_run(args) -> int:
    if not 0.0 <= args.strength <= 1.5:
        raise SystemExit("--strength must be between 0 and 1.5 (1.0 = remove the full lead estimate).")
    src, original, lead = _prepare(args)

    spec = args.segments.strip()
    if spec.lower() == "auto":
        segs = _detect(args, original, lead)
        _log("\nAuto-detected segments:")
    else:
        segs = parse_segments(spec, original.duration)
        _log("\nSegments to process:")
    segs = merge_segments(segs, original.duration)
    _print_segments(segs, original.duration)
    if not segs:
        raise SystemExit("No segments to process: nothing would change. Try --detect-threshold -35 or give --segments.")
    if args.dry_run:
        _log("\n--dry-run: nothing written.")
        return 0

    out_path = Path(args.output) if args.output else src.with_name(f"{src.stem}_nolead.flac")
    rep = export_audio(
        original,
        lead.data,
        segs,
        strength=args.strength,
        fade_s=args.fade_ms / 1000,
        out_path=out_path,
        bit_depth=args.bit_depth,
    )
    if rep.clipped:
        _log("  WARNING: result peaks above full scale; integer output will clip. Lower --strength or use --bit-depth 32 with .wav")

    _log(f"\nWrote {rep.path}  ({rep.sr} Hz, {rep.channels} ch, {rep.bit_depth}-bit)")
    _log(f"  untouched-region check, in memory : max difference {rep.in_memory_diff:.3g}  {'OK' if rep.memory_ok else 'FAIL'}")
    _log(f"  untouched-region check, from file : max difference {rep.on_disk_diff:.3g}  {'OK' if rep.disk_ok else 'differs (see README: 16-bit source / dithering)'}")
    return 0 if rep.memory_ok else 1


def cmd_doctor(args) -> int:
    from .doctor import main as doctor_main

    return doctor_main(fix=args.fix)


def cmd_gui(args) -> int:
    from .server import serve

    return serve(
        workdir=args.workdir,
        output_dir=args.output_dir,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="leadcut", description="Remove only the lead vocal, only where you want it.")
    p.add_argument("--version", action="version", version=f"leadcut {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, with_segments=False):
        sp.add_argument("input", help="song file on your computer (flac, wav, mp3, m4a, ...)")
        sp.add_argument("-o", "--output", help="output file (.flac or .wav)")
        sp.add_argument("--model", default=DEFAULT_MODEL, help=f"model key (default: {DEFAULT_MODEL}); see `leadcut models`")
        sp.add_argument("--workdir", default="leadcut_work", help="cache folder for stems (default: ./leadcut_work)")
        sp.add_argument("--lead-stem", help="force which model output is the lead vocal (word from its name)")
        g = sp.add_argument_group("auto-detection tuning")
        g.add_argument("--detect-threshold", type=float, default=-28.0, help="dB below the song's typical lead level that counts as 'singing' (default -28; lower = more sensitive)")
        g.add_argument("--min-gap", type=float, default=0.8, help="fill pauses shorter than this many seconds (default 0.8)")
        g.add_argument("--min-len", type=float, default=0.5, help="ignore detections shorter than this (default 0.5 s)")
        g.add_argument("--pad", type=float, default=0.2, help="widen every segment by this many seconds (default 0.2)")

    r = sub.add_parser("run", help="separate, splice and write the result")
    common(r)
    r.add_argument("--segments", default="auto", help="'auto', 'all', '0:45-1:30,2:10-end', or '@file.txt' (default: auto)")
    r.add_argument("--strength", type=float, default=1.0, help="how much of the lead estimate to subtract, 0-1.5 (default 1.0)")
    r.add_argument("--fade-ms", type=float, default=80.0, help="crossfade length at each segment edge (default 80 ms)")
    r.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32], help="output bit depth (default 24; 32 = float, .wav only)")
    r.add_argument("--dry-run", action="store_true", help="show the segments and stop without writing audio")
    r.set_defaults(func=cmd_run)

    d = sub.add_parser("detect", help="suggest segments and save them to a text file you can edit")
    common(d)
    d.set_defaults(func=cmd_detect)

    g = sub.add_parser("gui", help="open the waveform editor in your browser (local web page)")
    g.add_argument("--workdir", default="leadcut_work", help="cache folder (default: ./leadcut_work)")
    g.add_argument("--output-dir", default="leadcut_output", help="where exported files go (default: ./leadcut_output)")
    g.add_argument("--port", type=int, default=8765, help="port to listen on (default 8765; next free one is used if busy)")
    g.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1 = this computer only)")
    g.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    g.set_defaults(func=cmd_gui)

    dr = sub.add_parser("doctor", help="check the install; add --fix to install missing packages automatically")
    dr.add_argument("--fix", action="store_true", help="install any missing Python packages the separation library needs")
    dr.set_defaults(func=cmd_doctor)

    m = sub.add_parser("models", help="list built-in models")
    m.set_defaults(func=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles choke on some characters (e.g. Hindi file names)
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
