# Headless test suite for python/add_params.pyt (parameter installer DAT).
#
# Run from the repo root:
#     python tests\test_add_params.py

from pathlib import Path

import _stubs

_REPO_ROOT = Path(__file__).resolve().parents[1]
PYT = _REPO_ROOT / "python" / "add_params.pyt"

# The frozen parameter set (name and order).
FROZEN = [
    "Model",
    "Device",
    "Confidence",
    "Chromasharp",
    "Chromainvert",
    "Chromacolor",
    "Classwhitelist",
    "Customcheckpoint",
    "Hidesource",
    "State",
    "Ncount",
    "Inferms",
    "Dets",
]


def make_dat(name="add_params", container_name="my_rfdetr"):
    """A fake rfdetr/python/add_params DAT (deliberately renamed container)."""
    container = _stubs.FakeComp(container_name)
    py = container.addChild(_stubs.FakeComp("python"))
    dat = py.addChild(_stubs.FakeComp(name))
    return container, dat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_resolves_owner_relatively():
    container, dat = make_dat()
    _stubs.load_pyt(PYT, me=dat)

    page = container.customPages["Custom Parameters"]
    assert [p.name for p in page.pars] == FROZEN

    # Spot-check defaults from the installer.
    assert container.par.Model.value == "rfdetr-seg-medium"
    assert container.par.Device.value == "auto"
    assert container.par.Confidence.value == 0.5
    assert container.par.State.value == "init"


def test_rerun_is_idempotent():
    container, dat = make_dat()
    _stubs.load_pyt(PYT, me=dat)

    page1 = container.customPages["Custom Parameters"]
    n1 = len(page1.pars)
    assert n1 == 13

    _stubs.load_pyt(PYT, me=dat)

    page2 = container.customPages["Custom Parameters"]
    assert page1 is page2
    assert len(page2.pars) == n1
    assert [p.name for p in page2.pars] == FROZEN


def test_error_when_owner_missing():
    dat = _stubs.FakeComp("add_params")  # no parent chain
    try:
        _stubs.load_pyt(PYT, me=dat)
    except RuntimeError as e:
        assert "/atltvhead" not in str(e)
        assert 'op("/' not in str(e)
    else:
        raise AssertionError("expected RuntimeError")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    tests = [
        test_resolves_owner_relatively,
        test_rerun_is_idempotent,
        test_error_when_owner_missing,
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
