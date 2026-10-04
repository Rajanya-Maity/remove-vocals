"""Local web dashboard for leadcut (standard library only: no Flask / FastAPI needed).

Everything runs on YOUR computer. The server binds to 127.0.0.1 by default, so other
machines on your network cannot reach it, and it also refuses requests whose Host header
is not localhost (DNS-rebinding protection) and POSTs without the `X-Leadcut` header
(so a random website open in another tab cannot drive it).

Flow: upload a song -> analyse (slow, cached) -> edit segments on the waveform ->
preview (fast, in memory) -> export a full-quality file.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import soundfile as sf

from .audio_io import Audio, estimate_lag, load_audio
from .pipeline import export_audio
from .segments import Segment, detect_lead_segments, merge_segments
from .separation import DEFAULT_MODEL, MODELS, audio_separator_backend, get_lead_stem
from .splice import build_weight, peak, render, verify_untouched

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD = 1 << 30  # 1 GiB
MAX_JSON = 5 << 20
_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_PREVIEW_RE = re.compile(r"^preview_(\d+)\.wav$")


class HttpError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Job:
    state: str = "idle"  # idle | running | done | error
    song_id: str | None = None
    message: str = ""
    error: str | None = None
    started: float = 0.0
    finished: float = 0.0

    def as_dict(self) -> dict:
        end = self.finished if self.state in ("done", "error") else time.time()
        return {
            "state": self.state,
            "song_id": self.song_id,
            "message": self.message,
            "error": self.error,
            "elapsed": round(max(0.0, end - self.started), 1) if self.started else 0,
        }


@dataclass
class Session:
    song_id: str
    folder: Path
    path: Path
    name: str
    original: Audio | None = None
    lead: np.ndarray | None = None
    model: str = DEFAULT_MODEL
    warnings: list[str] = field(default_factory=list)
    peaks: dict | None = None
    version: int = 0
    lock: threading.RLock = field(default_factory=threading.RLock)


# ------------------------------------------------------------------- the app


class App:
    def __init__(self, workdir: str | Path, output_dir: str | Path, backend_factory=None, log=print):
        self.workdir = Path(workdir).resolve()
        self.library = self.workdir / "library"
        self.output_dir = Path(output_dir).resolve()
        self.library.mkdir(parents=True, exist_ok=True)
        self.log = log
        # (model_filename, log_fn) -> backend(input_wav, out_dir) -> {tag: path}
        self.backend_factory = backend_factory or (lambda fn, lg: audio_separator_backend(fn, log=lg))
        self.job = Job()
        self.sessions: dict[str, Session] = {}
        self._mutex = threading.Lock()

    # ----------------------------------------------------------- library
    def _folder(self, song_id: str) -> Path:
        if not _ID_RE.match(song_id or ""):
            raise HttpError(400, "Bad song id")
        return self.library / song_id

    def _meta(self, song_id: str) -> dict:
        f = self._folder(song_id) / "meta.json"
        if not f.exists():
            raise HttpError(404, "Unknown song. Upload it again.")
        return json.loads(f.read_text(encoding="utf-8"))

    def library_list(self) -> list[dict]:
        items = []
        for meta_file in self.library.glob("*/meta.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            folder = meta_file.parent
            analysed = sorted({p.parent.name for p in folder.glob("original-*/*/lead.wav")})
            items.append(
                {
                    "id": folder.name,
                    "name": meta.get("name", folder.name),
                    "size": meta.get("size", 0),
                    "analysed_models": analysed,
                    "has_state": (folder / "state.json").exists(),
                    "mtime": meta_file.stat().st_mtime,
                }
            )
        items.sort(key=lambda i: i["mtime"], reverse=True)
        return items

    def save_upload(self, name: str, stream, length: int) -> str:
        if length <= 0:
            raise HttpError(400, "Empty upload")
        if length > MAX_UPLOAD:
            raise HttpError(413, "File too large (limit 1 GiB)")
        safe = re.sub(r"[^\w.\- ()\[\]]", "_", Path(name or "song").name)[:120] or "song"
        ext = Path(safe).suffix.lower()
        if not re.fullmatch(r"\.[a-z0-9]{1,5}", ext or ""):
            ext = ".bin"
        tmp = self.library / f"_upload_{uuid.uuid4().hex}.part"
        h = hashlib.sha1()
        remaining = length
        try:
            with open(tmp, "wb") as f:
                while remaining > 0:
                    chunk = stream.read(min(1 << 20, remaining))
                    if not chunk:
                        raise HttpError(400, "Upload ended early")
                    h.update(chunk)
                    f.write(chunk)
                    remaining -= len(chunk)
            song_id = h.hexdigest()[:12]
            folder = self.library / song_id
            folder.mkdir(parents=True, exist_ok=True)
            final = folder / f"original{ext}"
            for old in folder.glob("original.*"):  # same content re-uploaded under another extension
                if old != final:
                    old.unlink()
            os.replace(str(tmp), str(final))  # atomic; also overwrites on Windows
        finally:
            if tmp.exists():
                tmp.unlink()
        (folder / "meta.json").write_text(json.dumps({"name": safe, "ext": ext, "size": length}), encoding="utf-8")
        return song_id

    def forget(self, song_id: str) -> None:
        folder = self._folder(song_id)
        with self._mutex:
            if self.job.state == "running" and self.job.song_id == song_id:
                raise HttpError(409, "That song is being analysed right now.")
            self.sessions.pop(song_id, None)
        if folder.exists():
            try:
                shutil.rmtree(folder)
            except OSError as exc:  # Windows: a file is still open (e.g. audio being played)
                raise HttpError(409, f"Could not delete the files ({exc}). Pause playback, wait a moment and try again.") from exc

    # ---------------------------------------------------------- analysing
    def config(self) -> dict:
        return {
            "models": [{"key": k, "description": m.description} for k, m in MODELS.items()],
            "default_model": DEFAULT_MODEL,
            "library": self.library_list(),
            "job": self.job.as_dict(),
            "output_dir": str(self.output_dir),
        }

    def _set_message(self, msg: str) -> None:
        self.job.message = msg
        self.log(f"  {msg}")

    def start_analyse(self, song_id: str, model_key: str, lead_stem: str = "") -> None:
        self._meta(song_id)  # validates id + existence
        if model_key not in MODELS:
            raise HttpError(400, f"Unknown model {model_key!r}")
        with self._mutex:
            if self.job.state == "running":
                raise HttpError(409, "Another analysis is still running. Wait for it to finish.")
            self.job = Job(state="running", song_id=song_id, message="Starting ...", started=time.time())
        threading.Thread(target=self._analyse, args=(song_id, model_key, lead_stem.strip() or None), daemon=True).start()

    def _analyse(self, song_id: str, model_key: str, lead_stem: str | None = None) -> None:
        try:
            meta = self._meta(song_id)
            folder = self._folder(song_id)
            path = folder / f"original{meta.get('ext', '.bin')}"
            with self._mutex:
                for other in [k for k in self.sessions if k != song_id]:  # keep RAM use to one song
                    self.sessions.pop(other)
                sess = self.sessions.get(song_id) or Session(song_id, folder, path, meta.get("name", "song"))
                self.sessions[song_id] = sess

            self._set_message("Reading the audio file ...")
            original = load_audio(path)
            model = MODELS[model_key]

            def backend(inp: Path, out: Path):
                return self.backend_factory(model.filename, self._set_message)(inp, out)

            lead = get_lead_stem(path, original, model=model, backend=backend, workdir=folder, lead_override=lead_stem, log=self._set_message)
            warnings = list(lead.warnings)
            lag = estimate_lag(original.data, lead.data, original.sr)
            if abs(lag) > max(1, int(0.001 * original.sr)):
                warnings.append(
                    f"The lead stem is {lag / original.sr * 1000:.1f} ms out of step with the original, so subtraction "
                    "will leave residue. Try another model."
                )
            if not original.is_lossless_source:
                warnings.append("The source file is lossy (MP3/AAC/...). leadcut cannot restore what the encoder removed; output is lossless so nothing is degraded further.")

            self._set_message("Drawing the waveform and preparing playback ...")
            peaks = self._peaks(original.data, lead.data, original.duration)
            prev = folder / "original_preview.wav"
            if not prev.exists():  # same content every time; never rewrite a file the browser may hold open
                part = folder / "original_preview.wav.part"
                sf.write(str(part), np.clip(original.data, -1, 1), original.sr, subtype="PCM_16", format="WAV")
                os.replace(str(part), str(prev))

            with sess.lock:
                sess.original, sess.lead, sess.model = original, lead.data, model_key
                sess.warnings, sess.peaks = warnings, peaks
            self.job.state, self.job.message, self.job.finished = "done", "Ready", time.time()
        except Exception as exc:  # shown to the user in the page
            traceback.print_exc()
            text = str(exc) or exc.__class__.__name__
            if isinstance(exc, ImportError) and "doctor" not in text:
                text += "\n\nTo repair the installation, stop leadcut (Ctrl+C) and run:  python -m leadcut doctor --fix"
            self.job.state, self.job.error, self.job.finished = "error", text, time.time()

    @staticmethod
    def _peaks(orig: np.ndarray, lead: np.ndarray, duration: float) -> dict:
        n = int(min(80000, max(2000, duration * 100)))
        n = max(1, min(n, orig.shape[0]))

        def lane(x: np.ndarray):
            mono = x.mean(axis=1)
            idx = np.linspace(0, len(mono), n + 1).astype(np.int64)[:-1]
            lo = np.minimum.reduceat(mono, idx)
            hi = np.maximum.reduceat(mono, idx)
            scale = float(max(np.max(np.abs(lo)), np.max(np.abs(hi)), 1e-6))
            return np.round(lo, 4).tolist(), np.round(hi, 4).tolist(), scale

        o_lo, o_hi, o_sc = lane(orig)
        l_lo, l_hi, l_sc = lane(lead)
        return {"n": n, "orig_min": o_lo, "orig_max": o_hi, "orig_scale": o_sc, "lead_min": l_lo, "lead_max": l_hi, "lead_scale": l_sc}

    # ------------------------------------------------------------ session
    def _ready(self, song_id: str) -> Session:
        self._folder(song_id)
        sess = self.sessions.get(song_id)
        if sess is None or sess.original is None or sess.lead is None:
            raise HttpError(409, "This song is not analysed yet. Press Analyse first.")
        return sess

    def _parse_segments(self, raw, duration: float) -> list[Segment]:
        if not isinstance(raw, list):
            raise HttpError(400, "segments must be a list")
        segs = []
        for item in raw:
            try:
                a, b = float(item[0]), float(item[1])
            except (TypeError, ValueError, IndexError):
                raise HttpError(400, "each segment must be [start, end] in seconds") from None
            if not (math.isfinite(a) and math.isfinite(b)):
                raise HttpError(400, "segment times must be finite")
            segs.append(Segment(a, b))
        return merge_segments(segs, duration)

    @staticmethod
    def _num(payload: dict, key: str, lo: float, hi: float, default: float) -> float:
        try:
            v = float(payload.get(key, default))
        except (TypeError, ValueError):
            raise HttpError(400, f"{key} must be a number") from None
        if not math.isfinite(v) or not lo <= v <= hi:
            raise HttpError(400, f"{key} must be between {lo} and {hi}")
        return v

    def _detect(self, sess: Session, p: dict) -> list[Segment]:
        return detect_lead_segments(
            sess.lead,
            sess.original.sr,
            threshold_db=self._num(p, "threshold_db", -60, 0, -28),
            min_gap_s=self._num(p, "min_gap", 0, 10, 0.8),
            min_len_s=self._num(p, "min_len", 0, 10, 0.5),
            pad_s=self._num(p, "pad", 0, 5, 0.2),
        )

    def song_info(self, song_id: str) -> dict:
        sess = self._ready(song_id)
        o = sess.original
        state_file = sess.folder / "state.json"
        state = None
        if state_file.exists():
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                state = None
        if state and isinstance(state.get("segments"), list):
            segs = [[s.start, s.end] for s in self._parse_segments(state["segments"], o.duration)]
            source = "saved"
        else:
            segs = [[s.start, s.end] for s in self._detect(sess, {})]
            source = "auto"
        return {
            "id": song_id,
            "name": sess.name,
            "duration": o.duration,
            "sr": o.sr,
            "channels": o.n_channels,
            "lossless_source": o.is_lossless_source,
            "model": sess.model,
            "warnings": sess.warnings,
            "peaks": sess.peaks,
            "segments": segs,
            "segments_source": source,
            "strength": (state or {}).get("strength", 1.0),
            "fade_ms": (state or {}).get("fade_ms", 80.0),
            "original_url": f"/audio/{song_id}/original_preview.wav",
        }

    def detect(self, p: dict) -> dict:
        sess = self._ready(p.get("id", ""))
        return {"segments": [[s.start, s.end] for s in self._detect(sess, p)]}

    def _save_state(self, sess: Session, segs: list[Segment], strength: float, fade_ms: float) -> None:
        data = {"segments": [[s.start, s.end] for s in segs], "strength": strength, "fade_ms": fade_ms}
        (sess.folder / "state.json").write_text(json.dumps(data), encoding="utf-8")

    def render_preview(self, p: dict) -> dict:
        sess = self._ready(p.get("id", ""))
        segs = self._parse_segments(p.get("segments", []), sess.original.duration)
        strength = self._num(p, "strength", 0, 1.5, 1.0)
        fade_ms = self._num(p, "fade_ms", 0, 2000, 80)
        with sess.lock:
            weight = build_weight(sess.original.n_samples, sess.original.sr, segs, fade_ms / 1000)
            out = render(sess.original.data, sess.lead, weight, strength)
            exact = verify_untouched(sess.original.data, out, weight) == 0
            pk = peak(out)
            sess.version += 1
            name = f"preview_{sess.version}.wav"
            tmp = sess.folder / (name + ".part")
            sf.write(str(tmp), np.clip(out, -1, 1), sess.original.sr, subtype="PCM_16", format="WAV")
            tmp.replace(sess.folder / name)
            for old in sess.folder.glob("preview_*.wav"):  # keep the last two only
                m = _PREVIEW_RE.match(old.name)
                if m and int(m.group(1)) < sess.version - 1:
                    try:
                        old.unlink(missing_ok=True)
                    except OSError:
                        pass  # Windows: still being streamed to the browser; it is removed on a later render
            self._save_state(sess, segs, strength, fade_ms)
        return {
            "url": f"/audio/{sess.song_id}/{name}",
            "version": sess.version,
            "n_segments": len(segs),
            "covered_s": sum(s.length for s in segs),
            "peak": pk,
            "clipped": pk > 1.0,
            "untouched_exact": exact,
        }

    def export(self, p: dict) -> dict:
        sess = self._ready(p.get("id", ""))
        segs = self._parse_segments(p.get("segments", []), sess.original.duration)
        if not segs:
            raise HttpError(400, "There are no segments, so nothing would change.")
        strength = self._num(p, "strength", 0, 1.5, 1.0)
        fade_ms = self._num(p, "fade_ms", 0, 2000, 80)
        fmt = str(p.get("format", "flac")).lower()
        if fmt not in ("flac", "wav"):
            raise HttpError(400, "format must be flac or wav")
        try:
            bit_depth = int(p.get("bit_depth", 24))
        except (TypeError, ValueError):
            raise HttpError(400, "bit_depth must be 16, 24 or 32") from None
        if bit_depth not in (16, 24, 32) or (fmt == "flac" and bit_depth == 32):
            raise HttpError(400, "Choose 16 or 24 bit (FLAC/WAV), or 32-bit float (WAV only).")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w\- ()\[\]]", "_", Path(sess.name).stem)[:100] or "song"
        out = self.output_dir / f"{stem}_nolead.{fmt}"
        n = 2
        while out.exists():
            out = self.output_dir / f"{stem}_nolead_{n}.{fmt}"
            n += 1
        with sess.lock:
            rep = export_audio(sess.original, sess.lead, segs, strength=strength, fade_s=fade_ms / 1000, out_path=out, bit_depth=bit_depth)
            self._save_state(sess, segs, strength, fade_ms)
        return {
            "path": str(rep.path),
            "filename": rep.path.name,
            "size": rep.path.stat().st_size,
            "download_url": f"/download/{rep.path.name}",
            "sr": rep.sr,
            "channels": rep.channels,
            "bit_depth": rep.bit_depth,
            "covered_s": rep.covered_s,
            "clipped": rep.clipped,
            "memory_ok": rep.memory_ok,
            "disk_ok": rep.disk_ok,
            "in_memory_diff": rep.in_memory_diff,
            "on_disk_diff": rep.on_disk_diff,
        }

    # ------------------------------------------------------------- files
    def audio_path(self, song_id: str, name: str) -> Path:
        folder = self._folder(song_id)
        if name != "original_preview.wav" and not _PREVIEW_RE.match(name):
            raise HttpError(404, "No such audio")
        p = folder / name
        if not p.exists():
            raise HttpError(404, "No such audio (re-run Analyse)")
        return p

    def download_path(self, filename: str) -> Path:
        name = Path(unquote(filename)).name
        p = self.output_dir / name
        if name != unquote(filename) or not p.is_file():
            raise HttpError(404, "No such file")
        return p


# ------------------------------------------------------------- HTTP plumbing


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "leadcut"
    app: App
    allowed_hosts: set[str]
    allow_any_host = False  # only True when the user deliberately binds beyond localhost
    verbose = False

    def log_message(self, fmt, *args):  # quiet by default
        if self.verbose:
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers
    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj).encode("utf-8"), "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_JSON:
            raise HttpError(413, "Request too large")
        raw = self.rfile.read(n) if n else b"{}"
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            raise HttpError(400, "Body is not valid JSON") from None
        if not isinstance(data, dict):
            raise HttpError(400, "Body must be a JSON object")
        return data

    def _guard(self, post: bool):
        host = (self.headers.get("Host") or "").lower()
        if not self.allow_any_host and host not in self.allowed_hosts:
            raise HttpError(403, "Bad Host header")
        if post and self.headers.get("X-Leadcut") != "1":
            raise HttpError(403, "Missing X-Leadcut header")

    def _send_file(self, path: Path, ctype: str, download_name: str | None = None):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range")
        if rng:
            m = re.fullmatch(r"bytes=(\d*)-(\d*)", rng.strip())
            if m and (m.group(1) or m.group(2)):
                a, b = m.groups()
                if a == "":
                    start = max(0, size - int(b))
                else:
                    start = int(a)
                    end = min(size - 1, int(b)) if b else size - 1
                if start > end or start >= size:
                    self._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
                    return
                status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition", f'attachment; filename="{download_name}"')
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser cancelled (normal when seeking)

    def _dispatch(self, fn):
        try:
            fn()
        except HttpError as e:
            self._json({"error": e.message}, e.status)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # last resort: tell the page, print for the developer
            traceback.print_exc()
            try:
                self._json({"error": f"{e.__class__.__name__}: {e}"}, 500)
            except Exception:
                pass

    # -- routes
    def do_GET(self):
        self._dispatch(self._get)

    def do_POST(self):
        self._dispatch(self._post)

    def _get(self):
        self._guard(post=False)
        url = urlparse(self.path)
        path = url.path
        if path in ("/", "/index.html"):
            self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/config":
            self._json(self.app.config())
        elif path == "/api/job":
            self._json(self.app.job.as_dict())
        elif path.startswith("/api/song/"):
            self._json(self.app.song_info(path.rsplit("/", 1)[1]))
        elif path.startswith("/audio/"):
            parts = path.split("/")
            if len(parts) != 4:
                raise HttpError(404, "Not found")
            self._send_file(self.app.audio_path(parts[2], parts[3]), "audio/wav")
        elif path.startswith("/download/"):
            p = self.app.download_path(path[len("/download/") :])
            ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
            self._send_file(p, ctype, download_name=p.name)
        else:
            raise HttpError(404, "Not found")

    def _post(self):
        self._guard(post=True)
        url = urlparse(self.path)
        path = url.path
        if path == "/api/upload":
            name = (parse_qs(url.query).get("name") or ["song"])[0]
            length = int(self.headers.get("Content-Length") or 0)
            self._json({"id": self.app.save_upload(name, self.rfile, length)})
            return
        body = self._read_json()
        if path == "/api/analyse":
            self.app.start_analyse(str(body.get("id", "")), str(body.get("model", DEFAULT_MODEL)), str(body.get("lead_stem", "")))
            self._json({"ok": True})
        elif path == "/api/detect":
            self._json(self.app.detect(body))
        elif path == "/api/render":
            self._json(self.app.render_preview(body))
        elif path == "/api/export":
            self._json(self.app.export(body))
        elif path == "/api/forget":
            self.app.forget(str(body.get("id", "")))
            self._json({"ok": True})
        elif path == "/api/shutdown":
            self._json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            raise HttpError(404, "Not found")


def make_server(app: App, host: str = "127.0.0.1", port: int = 8765, tries: int = 20) -> ThreadingHTTPServer:
    """Create the HTTP server, moving to the next port if one is busy."""
    last: Exception | None = None
    for p in range(port, port + tries):
        allowed = {f"{h}:{p}" for h in ("127.0.0.1", "localhost", "[::1]")} | {f"{host}:{p}"}
        local = host in ("127.0.0.1", "localhost", "::1")
        handler = type("BoundHandler", (Handler,), {"app": app, "allowed_hosts": allowed, "allow_any_host": not local})
        try:
            httpd = ThreadingHTTPServer((host, p), handler)
            httpd.daemon_threads = True
            return httpd
        except OSError as exc:
            last = exc
    raise RuntimeError(f"Could not open a port in {port}-{port + tries - 1}: {last}")


def serve(workdir="leadcut_work", output_dir="leadcut_output", host="127.0.0.1", port=8765, open_browser=True) -> int:
    app = App(workdir, output_dir)
    httpd = make_server(app, host, port)
    shown = "localhost" if host in ("127.0.0.1", "::1") else host
    url = f"http://{shown}:{httpd.server_address[1]}/"
    print(f"leadcut dashboard running at {url}")
    print(f"  working folder : {app.workdir}")
    print(f"  exports go to  : {app.output_dir}")
    print("  press Ctrl+C here (or use Quit in the page) to stop.")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: you exposed the dashboard beyond this computer. Anyone who can reach it can use it.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping ...")
    finally:
        httpd.server_close()
    return 0
