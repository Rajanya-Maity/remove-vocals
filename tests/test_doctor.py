from leadcut.doctor import fix_missing, missing_module, pip_name

ERR = "Traceback...\nModuleNotFoundError: No module named 'audioread'"


def test_parse_and_names():
    assert missing_module(ERR) == "audioread"
    assert missing_module("ImportError: DLL load failed") is None
    assert pip_name("yaml") == "pyyaml" and pip_name("sklearn.utils") == "scikit-learn" and pip_name("audioread") == "audioread"


def test_installs_one_missing_package_then_succeeds():
    state = {"installed": []}
    errs = iter([(False, ERR), (False, "ModuleNotFoundError: No module named 'pytorch_lightning'"), (True, "")])
    ok, err = fix_missing(check=lambda: next(errs), install=lambda p: (state["installed"].append(p) or (True, "")), installed=lambda m: False, log=lambda *_: None)
    assert ok and state["installed"] == ["audioread", "pytorch-lightning"]


def test_other_errors_are_reported_not_guessed():
    ok, err = fix_missing(check=lambda: (False, "OSError: DLL load failed"), install=lambda p: (_ for _ in ()).throw(AssertionError()), installed=lambda m: False, log=lambda *_: None)
    assert not ok and "DLL" in err


def test_never_reinstalls_a_present_but_broken_package():
    calls = []
    ok, err = fix_missing(check=lambda: (False, "No module named 'torch._C'"), install=lambda p: calls.append(p) or (True, ""), installed=lambda m: True, log=lambda *_: None)
    assert not ok and calls == [] and "torch._C" in err


def test_no_infinite_loop_when_install_does_not_help():
    calls = []
    ok, err = fix_missing(check=lambda: (False, ERR), install=lambda p: calls.append(p) or (True, ""), installed=lambda m: False, log=lambda *_: None)
    assert not ok and calls == ["audioread"]


def test_pip_failure_is_reported():
    ok, err = fix_missing(check=lambda: (False, ERR), install=lambda p: (False, "no network"), installed=lambda m: False, log=lambda *_: None)
    assert not ok and "no network" in err


def test_missing_main_library_points_to_proper_install():
    ok, err = fix_missing(check=lambda: (False, "No module named 'audio_separator'"), install=lambda p: (_ for _ in ()).throw(AssertionError()), installed=lambda m: False, log=lambda *_: None)
    assert not ok and ".[cpu]" in err
