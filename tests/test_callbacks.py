# Headless test suite for python/callbacks.pyt (container Callbacks DAT).
#
# Run from the repo root:
#     python tests\test_callbacks.py

from pathlib import Path

import _stubs

_REPO_ROOT = Path(__file__).resolve().parents[1]
PYT = _REPO_ROOT / "python" / "callbacks.pyt"


class RecordingCore:
    """Stands in for the cached core module: records cache clears."""

    def __init__(self):
        self.cleared = 0

    def clear_model_cache(self):
        self.cleared += 1


def make_container(with_status=True, core=None):
    root = _stubs.FakeComp("rfdetr")
    if with_status:
        root.addChild(_stubs.FakeTextDat("status", text=""))
    root.par.add("State", "init")
    if core is not None:
        root.store("rfdetr_core", core)
    return root


def change(mod, root, par_name, prev=None):
    par = _stubs.FakePar(par_name)
    par.owner = root
    mod.onValueChange(par, prev)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clears_via_public_api():
    mod = _stubs.load_pyt(PYT)
    core = RecordingCore()
    root = make_container(core=core)

    change(mod, root, "Model", "rfdetr-nano")

    assert core.cleared == 1
    assert root.par.State.value == "loading"
    assert root.children["status"].text.startswith("[loading]")


def test_ignores_other_params():
    mod = _stubs.load_pyt(PYT)
    core = RecordingCore()
    root = make_container(core=core)

    change(mod, root, "confidence", 0.5)

    assert core.cleared == 0
    assert root.par.State.value == "init"
    assert root.children["status"].text == ""


def test_missing_status_dat_no_crash():
    mod = _stubs.load_pyt(PYT)
    core = RecordingCore()
    root = make_container(with_status=False, core=core)

    change(mod, root, "device", "auto")

    assert core.cleared == 1
    assert root.par.State.value == "loading"


def test_missing_core_no_crash():
    mod = _stubs.load_pyt(PYT)
    root = make_container(core=None)

    change(mod, root, "model", "rfdetr-nano")

    assert root.par.State.value == "loading"
    assert root.children["status"].text.startswith("[loading]")


def test_case_insensitive_names():
    mod = _stubs.load_pyt(PYT)
    core = RecordingCore()
    root = make_container(core=core)

    change(mod, root, "customCheckpoint", "")

    assert core.cleared == 1
    assert root.par.State.value == "loading"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    tests = [
        test_clears_via_public_api,
        test_ignores_other_params,
        test_missing_status_dat_no_crash,
        test_missing_core_no_crash,
        test_case_insensitive_names,
    ]

    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as e:
            failed += 1
            print("FAIL %s: %r" % (fn.__name__, e))
        else:
            print("PASS %s" % fn.__name__)

    print("%d/%d pass" % (len(tests) - failed, len(tests)))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
