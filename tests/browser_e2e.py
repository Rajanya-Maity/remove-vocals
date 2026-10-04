"""Real-browser end-to-end test of the dashboard (not run by pytest; it needs Playwright).

    pip install playwright && playwright install chromium
    python tests/browser_e2e.py

It uses a FAKE separator, so no model or GPU is needed. Screenshots go to the temp folder.
"""
import io, sys, tempfile, threading, time, json
import numpy as np, soundfile as sf
from playwright.sync_api import sync_playwright
from leadcut.server import App, make_server
from leadcut.separation import stem_tag

SR = 44100
DUR = 24
n = DUR * SR
t = np.arange(n) / SR
rng = np.random.default_rng(3)
lead = np.zeros((n, 2), np.float32)
for a, b in [(2, 6), (9, 14), (17, 21)]:
    sl = slice(int(a*SR), int(b*SR))
    env = (0.5 + 0.5*np.sin(2*np.pi*1.7*t[sl]))**0.5
    lead[sl] = (0.30*env*np.sin(2*np.pi*(300+60*np.sin(2*np.pi*0.8*t[sl]))*t[sl]))[:, None]
acc = np.stack([0.12*np.sin(2*np.pi*110*t)+0.03*rng.standard_normal(n), 0.12*np.sin(2*np.pi*165*t)+0.03*rng.standard_normal(n)], axis=1).astype(np.float32)
for a, b in [(6.2, 8.8), (14.2, 16.8)]:   # loud "chorus" bursts between phrases
    sl = slice(int(a*SR), int(b*SR)); acc[sl] += (0.25*np.sin(2*np.pi*520*t[sl]))[:, None]
TMP = tempfile.mkdtemp(prefix="leadcut_e2e_")
song = TMP + "/test song.wav"
sf.write(song, lead + acc, SR, subtype="PCM_16")

def factory(_fn, log):
    def run(inp, out):
        log("fake separation (stand-in for the model)"); time.sleep(1.2)
        sf.write(str(out/"input_(Vocals)_fake.wav"), lead, SR, subtype="FLOAT")
        sf.write(str(out/"input_(Instrumental)_fake.wav"), acc, SR, subtype="FLOAT")
        return {stem_tag(p): p for p in out.glob("*.wav")}
    return run

app = App(TMP + "/work", TMP + "/out", backend_factory=factory, log=lambda *_: None)
httpd = make_server(app, "127.0.0.1", 18900)
threading.Thread(target=httpd.serve_forever, daemon=True).start()
URL = f"http://localhost:{httpd.server_address[1]}/"

errors, results = [], []
def check(name, cond, extra=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra else ""))

with sync_playwright() as p:
    b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"])
    page = b.new_page(viewport={"width": 1280, "height": 1000})
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append("PAGEERROR " + str(e)))
    page.on("dialog", lambda d: d.accept())
    page.goto(URL)
    page.wait_for_selector("#model option", state="attached", timeout=10000)
    check("page loads, models listed", page.locator("#model option").count() >= 3)

    page.set_input_files("#file", song)
    page.wait_for_selector("#status .spinner", timeout=5000)
    check("progress shown during analysis", True)
    page.wait_for_selector("#editor:not(.hidden)", timeout=30000)
    page.wait_for_function("document.querySelector('#pvstatus').textContent.includes('up to date')", timeout=20000)
    segs = page.evaluate("S.segs")
    check("auto-detected 3 segments", len(segs) == 3, str(segs))
    check("table has 3 rows", page.locator("#segbody tr.segrow").count() == 3)
    page.screenshot(path=TMP + "/1_loaded.png")

    box = page.locator("#overlay").bounding_box()
    dur = page.evaluate("S.info.duration")
    X = lambda tt: box["x"] + tt / dur * box["width"]
    Y = box["y"] + 100

    # drag-create a new segment in empty space (22-23.5 s)
    page.mouse.move(X(22), Y); page.mouse.down(); page.mouse.move(X(22.8), Y, steps=5); page.mouse.move(X(23.5), Y, steps=5); page.mouse.up()
    segs = page.evaluate("S.segs")
    check("drag on empty space creates a segment", len(segs) == 4 and abs(segs[3][0]-22) < 0.2 and abs(segs[3][1]-23.5) < 0.2, str(segs[3:]))

    # resize its right edge
    page.mouse.move(X(23.5), Y); page.mouse.down(); page.mouse.move(X(23.9), Y, steps=5); page.mouse.up()
    check("dragging an edge resizes", abs(page.evaluate("S.segs")[3][1] - 23.9) < 0.2)

    # move the body
    page.mouse.move(X(22.9), Y); page.mouse.down(); page.mouse.move(X(22.2), Y, steps=6); page.mouse.up()
    s3 = page.evaluate("S.segs")[3]
    check("dragging the body moves it (length kept)", abs((s3[1]-s3[0]) - 1.9) < 0.15 and s3[0] < 21.8, str(s3))

    # undo x3 returns to 3 segments
    for _ in range(3): page.keyboard.press("Control+z")
    check("Ctrl+Z undoes edits", len(page.evaluate("S.segs")) == 3)

    # delete selected
    page.mouse.move(X(11), Y); page.mouse.down(); page.mouse.up()   # click inside seg 2 -> select
    check("clicking a segment selects it", page.evaluate("S.sel") == 1)
    page.keyboard.press("Delete")
    check("Delete removes selected", len(page.evaluate("S.segs")) == 2)
    page.keyboard.press("Control+z")
    check("undo restores deleted", len(page.evaluate("S.segs")) == 3)

    # click on empty = seek
    page.mouse.move(X(7.5), Y); page.mouse.down(); page.mouse.up()
    check("click on empty space seeks", abs(page.evaluate("S.cursor") - 7.5) < 0.2 and page.evaluate("S.sel") == -1)

    # table editing
    inp = page.locator("#segbody tr").nth(0).locator("input").nth(0)
    inp.fill("0:01.000"); inp.press("Tab")
    check("editing a time in the table works", abs(page.evaluate("S.segs[0][0]") - 1.0) < 0.001)
    inp.fill("garbage"); inp.press("Tab")
    check("invalid time is rejected", abs(page.evaluate("S.segs[0][0]") - 1.0) < 0.001)
    page.locator("#segbody tr").nth(1).locator("input").nth(0).focus()
    check("focusing a time box selects its segment", page.evaluate("S.sel") == 1)

    # text apply
    page.locator("details:has-text('Edit as text')").evaluate("d => d.open = true")
    page.fill("#segtext", "0:02-0:06\n0:09-0:14\n0:17-0:21")
    page.click("#applytext")
    segs = page.evaluate("S.segs")
    check("apply-as-text works", len(segs) == 3 and abs(segs[0][0]-2) < 0.001)

    # playback + A/B
    page.wait_for_function("document.querySelector('#pvstatus').textContent.includes('up to date')", timeout=20000)
    page.click("#play"); time.sleep(0.9)
    playing = page.evaluate("!aP.paused")
    c1 = page.evaluate("S.cursor")
    check("Play starts the lead-removed preview", playing and c1 > 0.3, f"cursor={c1:.2f}")
    page.click("#ab button[data-m=orig]"); time.sleep(0.6)
    check("A/B switch keeps playing from same position", page.evaluate("!aO.paused && aP.paused") and page.evaluate("aO.currentTime") > c1 - 0.2)
    page.keyboard.press("Space"); time.sleep(0.3)
    check("Space pauses", page.evaluate("aO.paused && aP.paused"))
    check("Space does not double-toggle via a focused button", page.evaluate("document.querySelector('#play').textContent.includes('Play')"))
    page.click("#ab button[data-m=proc]")

    # loop region: clicking the numbered cell selects the row
    page.locator("#segbody tr.segrow").nth(0).locator("td").first.click()
    check("clicking a table row selects the segment", page.evaluate("S.sel") == 0)
    page.check("#loop"); page.click("#play"); time.sleep(2.2)
    cur = page.evaluate("S.cursor")
    check("loop keeps playhead inside the selected segment", 1.9 < cur < 6.2, f"cursor={cur:.2f}")
    page.click("#play")

    # zoom (page may have scrolled: re-measure the canvas)
    page.locator("#overlay").scroll_into_view_if_needed()
    box = page.locator("#overlay").bounding_box(); Y = box["y"] + 100
    page.mouse.move(X(10), Y); page.mouse.wheel(0, -500); time.sleep(0.2)
    v = page.evaluate("S.view")
    check("wheel zooms in", v["dur"] < 20, str(v))
    page.screenshot(path=TMP + "/2_zoomed.png")
    page.click("#zfit")
    check("Fit shows whole song", abs(page.evaluate("S.view.dur") - dur) < 0.01)

    # sliders -> preview refresh
    page.evaluate("const e=document.querySelector('#strength'); e.value=0.9; e.dispatchEvent(new Event('input'))")
    page.wait_for_function("document.querySelector('#pvstatus').textContent.includes('up to date')", timeout=20000)
    check("changing strength updates the preview", abs(page.evaluate("S.strength") - 0.9) < 1e-9)

    # redetect
    page.locator("details:has-text('Auto-detect settings')").evaluate("d => d.open = true")
    page.click("#redetect"); time.sleep(0.8)
    check("re-detect returns segments", len(page.evaluate("S.segs")) == 3)

    # export
    page.click("#export")
    page.wait_for_selector("#exportres .ok", timeout=30000)
    txt = page.inner_text("#exportres")
    check("export verifies untouched regions (memory + file)", txt.count("identical") == 2, txt.replace("\n", " | ")[:200])
    page.screenshot(path=TMP + "/3_exported.png", full_page=True)

    # reload restores
    page.reload()
    page.wait_for_selector("#editor:not(.hidden)", timeout=20000)
    check("reload restores the song and saved edits", len(page.evaluate("S.segs")) == 3 and abs(page.evaluate("S.strength") - 0.9) < 1e-9)

    # library reopen
    check("library lists the song", page.locator("#library option").count() == 2)

    # quit
    page.click("#quit"); time.sleep(0.5)
    check("Quit stops the server", "stopped" in page.inner_text("body").lower())
    b.close()

real_errors = [e for e in errors if "favicon" not in e.lower()]
check("no console / page errors", not real_errors, "; ".join(real_errors)[:300])
print(f"\n{sum(ok for _, ok in results)}/{len(results)} checks passed (screenshots in {TMP})")
sys.exit(0 if all(ok for _, ok in results) else 1)
