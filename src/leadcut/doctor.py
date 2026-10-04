"""`leadcut doctor`: check the install, and with --fix install whatever Python packages are missing.

Why this exists: audio-separator imports a few packages it forgets to declare as
dependencies (for example `audioread`), and which ones are missing depends on your Python
version. Instead of you discovering them one error at a time, `--fix` runs the real import
in a fresh process, reads which module is missing, installs it, and repeats.
"""

from __future__ import annotations

import importlib.util
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

IMPORT_CHECK = "from audio_separator.separator import Separator"

# import-name -> pip-name, only where they differ
PIP_NAMES = {
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "pillow",
    "skimage": "scikit-image",
    "pytorch_lightning": "pytorch-lightning",
    "attr": "attrs",
}

_MISSING = re.compile(r"No module named '([^']+)'")


def missing_module(error_text: str) -> str | None:
    """'ModuleNotFoundError: No module named 'a.b'' -> 'a.b' (None if it is some other error)."""
    m = _MISSING.search(error_text or "")
    return m.group(1) if m else None


def pip_name(module: str) -> str:
    top = module.split(".")[0]
    return PIP_NAMES.get(top, top)


def run_import_check(python: str = sys.executable) -> tuple[bool, str]:
    """Import the separation library in a FRESH interpreter (so cached failures can't mislead)."""
    res = subprocess.run([python, "-c", IMPORT_CHECK], capture_output=True, text=True)
    return res.returncode == 0, (res.stderr or res.stdout or "").strip()


def pip_install(package: str, python: str = sys.executable) -> tuple[bool, str]:
    res = subprocess.run([python, "-m", "pip", "install", package], capture_output=True, text=True)
    return res.returncode == 0, ((res.stdout or "") + (res.stderr or "")).strip()


def is_installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def fix_missing(check=run_import_check, install=pip_install, installed=is_installed, max_rounds: int = 15, log=print) -> tuple[bool, str]:
    """Repeat: try the import, install the one missing package it names, until it works.

    Returns (ok, last_error_text). Never loops forever and never re-downloads a big package
    (like torch) that is already present but broken: that case is reported instead.
    """
    tried: set[str] = set()
    err = ""
    for _ in range(max_rounds):
        ok, err = check()
        if ok:
            return True, ""
        module = missing_module(err)
        if module is None:
            return False, err  # a different kind of error: show it, don't guess
        if module.split(".")[0] == "audio_separator":
            return False, 'audio-separator itself is not installed. Run:  python -m pip install -e ".[cpu]"'
        package = pip_name(module)
        if installed(module) or package in tried:
            return False, f"'{module}' is present but cannot be used. Details:\n{err}"
        tried.add(package)
        log(f"  installing missing package: {package} ...")
        ok_install, out = install(package)
        if not ok_install:
            return False, f"pip could not install '{package}':\n{out[-1500:]}"
    return False, "Gave up after too many rounds. Last error:\n" + err


def _torch_report(python: str = sys.executable) -> str:
    code = (
        "import torch\n"
        "cuda = torch.cuda.is_available()\n"
        "name = torch.cuda.get_device_name(0) if cuda else ''\n"
        "print(f'{torch.__version__}|{cuda}|{name}')\n"
    )
    res = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else ""


def main(fix: bool = False, log=print) -> int:
    problems: list[str] = []
    ok = lambda s: log(f"  [ OK ] {s}")
    bad = lambda s, hint="": (log(f"  [FAIL] {s}"), hint and log(f"         -> {hint}"), problems.append(s))
    note = lambda s: log(f"  [note] {s}")

    log("leadcut doctor\n")
    v = sys.version_info
    ok(f"Python {v.major}.{v.minor}.{v.micro} on {platform.system()} ({sys.executable})")
    if (v.major, v.minor) < (3, 10):
        bad("Python is too old (need 3.10+)", "install Python 3.12 from python.org")
    elif (v.major, v.minor) >= (3, 13):
        note("Python 3.13+ is newer than most audio libraries are tested on. If anything keeps failing, use Python 3.12 (see README).")

    for mod in ("numpy", "scipy", "soundfile"):
        if is_installed(mod):
            ok(f"{mod} installed")
        else:
            bad(f"{mod} is missing", 'run: python -m pip install -e ".[cpu]"')

    if shutil.which("ffmpeg"):
        ok("ffmpeg found on PATH")
    else:
        bad("ffmpeg not found on PATH", "Windows: winget install Gyan.FFmpeg   (then open a NEW terminal)")

    log("\n  Checking the separation library (the first check can take up to a minute) ...")
    good, err = run_import_check()
    if not good and fix:
        good, err = fix_missing(log=log)
    if good:
        ok("separation library loads")
        info = _torch_report()
        if info:
            ver, cuda, name = (info.split("|") + ["", "", ""])[:3]
            if cuda == "True":
                ok(f"PyTorch {ver} sees your GPU: {name}")
            else:
                note(f"PyTorch {ver} is running on the CPU. That works; it is just slower. (README: 'Using your GPU')")
    else:
        short = err.strip().splitlines()[-1] if err.strip() else "unknown error"
        bad("separation library fails to load: " + short, "run again with --fix:  python -m leadcut doctor --fix" if not fix else "copy the text below to get help")
        if fix and err:
            log("\n----- details -----\n" + err[-2500:] + "\n-------------------")

    free_gb = shutil.disk_usage(Path.cwd()).free / 1e9
    if free_gb < 3:
        bad(f"only {free_gb:.1f} GB free disk space", "models and stems need a few GB")
    else:
        ok(f"{free_gb:.0f} GB free disk space")

    log("")
    if problems:
        log(f"RESULT: {len(problems)} problem(s) found (see [FAIL] lines above).")
        return 1
    log("RESULT: ALL GOOD. Start the dashboard with:  python -m leadcut gui")
    return 0
