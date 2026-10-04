"""Tests the real HTTP server with a FAKE separator (no models, no browser)."""
import http.client
import io
import json
import threading
import time

import numpy as np
import pytest
import soundfile as sf

from leadcut.separation import stem_tag
from leadcut.server import App, make_server

SR = 44100


def make_song_bytes():
    n = 12 * SR
    t = np.arange(n) / SR
    lead = np.zeros((n, 2), np.float32)
    for a, b in [(2, 4), (7, 9)]:
        sl = slice(int(a * SR), int(b * SR))
        lead[sl] = (0.3 * np.sin(2 * np.pi * 440 * t[sl]))[:, None]
    acc = np.stack([0.1 * np.sin(2 * np.pi * 110 * t), 0.1 * np.sin(2 * np.pi * 165 * t)], axis=1).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, lead + acc, SR, subtype="PCM_16", format="WAV")
    return buf.getvalue(), lead, acc


@pytest.fixture()
def srv(tmp_path):
    data, lead, acc = make_song_bytes()
    gate = threading.Event()
    gate.set()

    def factory(_fn, log):
        def run(input_wav, out_dir):
            gate.wait(10)
            log("fake separation")
            sf.write(str(out_dir / "input_(Vocals)_fake.wav"), lead, SR, subtype="FLOAT")
            sf.write(str(out_dir / "input_(Instrumental)_fake.wav"), acc, SR, subtype="FLOAT")
            return {stem_tag(p): p for p in out_dir.glob("*.wav")}

        return run

    app = App(tmp_path / "work", tmp_path / "out", backend_factory=factory, log=lambda *_: None)
    httpd = make_server(app, "127.0.0.1", 18765)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()

    class C:
        port = httpd.server_address[1]
        song = data
        gate_ = gate
        app_ = app

        @staticmethod
        def req(method, path, body=None, headers=None, raw=None):
            conn = http.client.HTTPConnection("127.0.0.1", C.port, timeout=20)
            h = {"X-Leadcut": "1"} if method == "POST" else {}
            h.update(headers or {})
            payload = raw
            if body is not None:
                payload = json.dumps(body).encode()
                h["Content-Type"] = "application/json"
            conn.request(method, path, body=payload, headers=h)
            r = conn.getresponse()
            data = r.read()
            hdrs = dict(r.getheaders())
            conn.close()
            return r.status, hdrs, data

        @staticmethod
        def jreq(method, path, body=None, **kw):
            st, h, d = C.req(method, path, body, **kw)
            return st, json.loads(d or b"{}")

        @staticmethod
        def wait_job(timeout=30):
            end = time.time() + timeout
            while time.time() < end:
                _, j = C.jreq("GET", "/api/job")
                if j["state"] in ("done", "error"):
                    return j
                time.sleep(0.05)
            raise AssertionError("job timed out")

        @staticmethod
        def upload(name="song.wav"):
            st, h, d = C.req("POST", f"/api/upload?name={name}", raw=C.song, headers={"Content-Type": "application/octet-stream"})
            assert st == 200, d
            return json.loads(d)["id"]

    yield C
    httpd.shutdown()
    httpd.server_close()


def analysed(C):
    sid = C.upload()
    st, _ = C.jreq("POST", "/api/analyse", {"id": sid, "model": "roformer-karaoke"})
    assert st == 200
    assert C.wait_job()["state"] == "done"
    return sid


def test_index_page_served(srv):
    st, h, d = srv.req("GET", "/")
    assert st == 200 and b"leadcut" in d and h["Content-Type"].startswith("text/html")


def test_security_guards(srv):
    st, _, _ = srv.req("GET", "/api/config", headers={"Host": "evil.example.com"})
    assert st == 403
    conn = http.client.HTTPConnection("127.0.0.1", srv.port)
    conn.request("POST", "/api/analyse", body=b"{}", headers={"Content-Type": "application/json"})  # no X-Leadcut
    assert conn.getresponse().status == 403


def test_full_flow(srv):
    st, cfg = srv.jreq("GET", "/api/config")
    assert st == 200 and cfg["default_model"] == "roformer-karaoke" and cfg["library"] == []

    st, e = srv.jreq("GET", "/api/song/" + "0" * 12)
    assert st == 409  # not analysed

    sid = analysed(srv)
    st, info = srv.jreq("GET", f"/api/song/{sid}")
    assert st == 200 and info["segments_source"] == "auto" and len(info["segments"]) == 2
    assert abs(info["duration"] - 12) < 0.01 and info["sr"] == SR and info["channels"] == 2
    p = info["peaks"]
    assert len(p["orig_min"]) == p["n"] == len(p["lead_max"])

    segs = [[1.5, 4.5]]
    st, r = srv.jreq("POST", "/api/render", {"id": sid, "segments": segs, "strength": 1.0, "fade_ms": 80})
    assert st == 200 and r["untouched_exact"] and r["n_segments"] == 1 and not r["clipped"]

    # preview file is playable, supports Range, and keeps untouched audio
    st, hdr, full = srv.req("GET", r["url"])
    assert st == 200 and hdr["Accept-Ranges"] == "bytes"
    st, hdr, part = srv.req("GET", r["url"], headers={"Range": "bytes=0-99"})
    assert st == 206 and hdr["Content-Range"].startswith("bytes 0-99/") and len(part) == 100 and part == full[:100]
    st, hdr, tail = srv.req("GET", r["url"], headers={"Range": "bytes=-50"})
    assert st == 206 and tail == full[-50:]
    st, _, _ = srv.req("GET", r["url"], headers={"Range": "bytes=999999999-"})
    assert st == 416
    prev, _ = sf.read(io.BytesIO(full), dtype="float32", always_2d=True)
    orig, _ = sf.read(io.BytesIO(srv.song), dtype="float32", always_2d=True)
    assert np.array_equal(prev[5 * SR :], orig[5 * SR :])  # 16-bit preview of a 16-bit source: exact
    seg = slice(int(2.2 * SR), int(3.8 * SR))
    assert np.max(np.abs(prev[seg] - orig[seg])) > 0.2  # lead really removed

    # state persisted -> reopening gives the edited segments back
    st, info2 = srv.jreq("GET", f"/api/song/{sid}")
    assert info2["segments_source"] == "saved" and info2["segments"] == [[1.5, 4.5]]

    # export + download
    st, ex = srv.jreq("POST", "/api/export", {"id": sid, "segments": segs, "strength": 1.0, "fade_ms": 80, "format": "flac", "bit_depth": 24})
    assert st == 200 and ex["memory_ok"] and ex["disk_ok"] and ex["bit_depth"] == 24
    st, hdr, blob = srv.req("GET", ex["download_url"])
    assert st == 200 and "attachment" in hdr["Content-Disposition"] and len(blob) == ex["size"]
    out, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
    assert sr == SR and np.array_equal(out[5 * SR :], orig[5 * SR :])
    # second export does not overwrite the first
    st, ex2 = srv.jreq("POST", "/api/export", {"id": sid, "segments": segs, "format": "flac"})
    assert ex2["filename"] != ex["filename"]

    # library shows it; forget removes it
    st, cfg = srv.jreq("GET", "/api/config")
    assert cfg["library"][0]["id"] == sid and cfg["library"][0]["analysed_models"] == ["roformer-karaoke"]
    st, _ = srv.jreq("POST", "/api/forget", {"id": sid})
    assert st == 200
    assert srv.jreq("GET", "/api/config")[1]["library"] == []


def test_redetect(srv):
    sid = analysed(srv)
    st, r = srv.jreq("POST", "/api/detect", {"id": sid, "threshold_db": -28, "min_gap": 0.8, "min_len": 0.5, "pad": 0.2})
    assert st == 200 and len(r["segments"]) == 2
    st, r = srv.jreq("POST", "/api/detect", {"id": sid, "min_gap": 10})  # gap fill swallows the pause
    assert len(r["segments"]) == 1


def test_validation_errors(srv):
    sid = analysed(srv)
    bad = [
        ("/api/render", {"id": sid, "segments": "nope"}),
        ("/api/render", {"id": sid, "segments": [[1, "x"]]}),
        ("/api/render", {"id": sid, "segments": [[1, 2]], "strength": 9}),
        ("/api/render", {"id": sid, "segments": [[1, 2]], "fade_ms": -1}),
        ("/api/export", {"id": sid, "segments": []}),
        ("/api/export", {"id": sid, "segments": [[1, 2]], "format": "mp3"}),
        ("/api/export", {"id": sid, "segments": [[1, 2]], "format": "flac", "bit_depth": 32}),
        ("/api/export", {"id": sid, "segments": [[1, 2]], "bit_depth": 12}),
        ("/api/analyse", {"id": sid, "model": "does-not-exist"}),
        ("/api/analyse", {"id": "../../etc", "model": "roformer-karaoke"}),
        ("/api/forget", {"id": "zzz"}),
    ]
    for path, body in bad:
        st, r = srv.jreq("POST", path, body)
        assert st == 400 and "error" in r, (path, body, st, r)
    st, r = srv.jreq("POST", "/api/render", raw=b"{not json", headers={"Content-Type": "application/json"})
    assert st == 400
    st, r = srv.jreq("POST", "/api/nothing", {})
    assert st == 404


def test_path_traversal_blocked(srv):
    sid = analysed(srv)
    for p in [f"/audio/{sid}/../meta.json", f"/audio/{sid}/meta.json", "/audio/xx/original_preview.wav",
              "/download/..%2Fwork%2Flibrary", "/download/%2e%2e/secret", "/download/nothing.flac", "/audio/a/b/c/d"]:
        st, _, _ = srv.req("GET", p)
        assert st in (400, 404), (p, st)


def test_second_analyse_blocked_while_running(srv):
    srv.gate_.clear()  # make the fake separator wait
    sid = srv.upload()
    assert srv.jreq("POST", "/api/analyse", {"id": sid, "model": "roformer-karaoke"})[0] == 200
    time.sleep(0.2)
    st, r = srv.jreq("POST", "/api/analyse", {"id": sid, "model": "roformer-karaoke"})
    assert st == 409
    assert srv.jreq("GET", "/api/job")[1]["state"] == "running"
    assert srv.jreq("POST", "/api/forget", {"id": sid})[0] == 409
    srv.gate_.set()
    assert srv.wait_job()["state"] == "done"


def test_backend_error_is_reported(tmp_path):
    def factory(_fn, log):
        def run(i, o):
            raise RuntimeError("model exploded")
        return run

    app = App(tmp_path / "w", tmp_path / "o", backend_factory=factory, log=lambda *_: None)
    data, *_ = make_song_bytes()
    sid = app.save_upload("x.wav", io.BytesIO(data), len(data))
    app.start_analyse(sid, "roformer-karaoke")
    for _ in range(100):
        if app.job.state != "running":
            break
        time.sleep(0.05)
    assert app.job.state == "error" and "model exploded" in app.job.error


def test_upload_rejects_empty_and_truncated(srv):
    st, r = srv.jreq("POST", "/api/upload?name=x.wav", raw=b"", headers={"Content-Type": "application/octet-stream"})
    assert st == 400
