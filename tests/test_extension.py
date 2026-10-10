# Headless test suite for python/extension.pyt (infer Script TOP glue).
#
# Run from the repo root:
#     python tests\test_extension.py

from pathlib import Path

import numpy as np

import _stubs

_REPO_ROOT = Path(__file__).resolve().parents[1]
PYT = _REPO_ROOT / "python" / "extension.pyt"
CORE_TEXT = (_REPO_ROOT / "python" / "core.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture: a fake rfdetr component with the TOX's internal operators
# ---------------------------------------------------------------------------


def make_root(name="rfdetr", with_status=True):
    root = _stubs.FakeComp(name)

    core_dat = _stubs.FakeTextDat("core", text=CORE_TEXT)
    py = root.addChild(_stubs.FakeComp("python"))
    py.addChild(core_dat)

    if with_status:
        root.addChild(_stubs.FakeTextDat("status", text=""))

    boxdata = root.addChild(_stubs.FakeTop("boxdata"))
    root.addChild(_stubs.FakeTableDat("detections"))

    for pname, pval in (
        ("Model", "rfdetr-seg-medium"),
        ("Device", "auto"),
        ("Confidence", 0.5),
        ("Classwhitelist", ""),
        ("Customcheckpoint", ""),
        ("Hidesource", 0),
        ("State", "init"),
        ("Ncount", 0),
        ("Inferms", 0.0),
        ("Dets", 0),
    ):
        root.par.add(pname, pval)

    return root


def make_script_top(root, name="infer"):
    st = _stubs.FakeScriptTop(name)
    root.addChild(st)
    return st


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_root_is_parent():
    # Deliberately NOT named "rfdetr" — resolution must be structural.
    root = make_root("rfdetr2")
    st = make_script_top(root)
    mod = _stubs.load_pyt(PYT, me=st)
    assert mod._root() is root


def test_root_none_without_parent():
    st = _stubs.FakeScriptTop("infer")
    mod = _stubs.load_pyt(PYT, me=st)
    assert mod._root() is None


def test_source_has_no_absolute_paths():
    text = PYT.read_text(encoding="utf-8")
    assert 'op("/' not in text
    assert "op('/" not in text
    assert "ROOT_PATH" not in text
    assert "/atltvhead" not in text


def test_core_module_resolves_relative():
    root = make_root()
    st = make_script_top(root)
    mod = _stubs.load_pyt(PYT, me=st)

    m1 = mod.core_module()
    assert hasattr(m1, "build_union")
    u = m1.build_union(boxes=np.array([[0.0, 0.0, 1.0, 1.0]]), h=2, w=2)
    assert u.sum() == 1.0

    m2 = mod.core_module()
    assert m2 is m1


def test_oncook_no_deps():
    root = make_root()
    st = make_script_top(root)
    st.addInput(np.zeros((4, 4, 4), dtype=np.float32))
    mod = _stubs.load_pyt(PYT, me=st)

    with _stubs.BlockImport(["torch"]):
        mod.onCook(st)

    assert root.par.State.value == "no-deps"
    assert root.children["status"].text.startswith("[no-deps]")
    assert st.copied and st.copied[0].shape == (4, 4, 4)
    assert st.copied[0].sum() == 0.0
    assert root.par.Dets.value == 0
    assert root.par.Ncount.value == 0


def test_oncook_missing_input():
    root = make_root()
    st = make_script_top(root)
    mod = _stubs.load_pyt(PYT, me=st)

    with _stubs.StubbedTorchRfdetr(cuda_available=False):
        mod.onCook(st)

    assert root.par.State.value == "error"
    assert root.children["status"].text.startswith(
        "[error] infer Script TOP has no input."
    )
    assert st.copied == []


def test_oncook_full_path_no_detections():
    root = make_root()
    st = make_script_top(root)
    st.addInput(np.zeros((4, 4, 3), dtype=np.float32))
    mod = _stubs.load_pyt(PYT, me=st)

    with _stubs.StubbedTorchRfdetr(cuda_available=False):
        mod.onCook(st)

    assert root.par.State.value == "ready"
    assert root.par.Ncount.value == 0
    assert root.par.Dets.value == 0

    bd = root.children["boxdata"]
    assert bd.par.width.value == 384
    assert bd.par.height.value == 1
    assert bd._pixel is not None
    assert bd._pixel.shape == (384,)
    assert bd._pixel.sum() == 0.0

    det = root.children["detections"]
    assert det.rows[0] == [
        "class_id", "class", "confidence",
        "x1", "y1", "x2", "y2", "area", "frame",
    ]
    assert len(det.rows) == 1

    assert root.children["status"].text.startswith("[ready]")

    mask = st.copied[-1]
    assert mask.shape == (4, 4, 4)
    assert mask.sum() == 0.0


def test_oncook_whitelist_filters():
    root = make_root()
    st = make_script_top(root)
    st.addInput(np.zeros((4, 4, 3), dtype=np.float32))
    root.par.Classwhitelist.value = "person"
    mod = _stubs.load_pyt(PYT, me=st)

    det = _stubs.FakeDet(
        xyxy=[[1, 1, 3, 3], [0, 0, 2, 2]],
        class_id=[0, 2],
        confidence=[0.9, 0.7],
        data={"class_name": ["person", "car"]},
    )

    with _stubs.StubbedTorchRfdetr(cuda_available=False, result=det):
        mod.onCook(st)

    assert root.par.State.value == "ready"
    assert root.par.Ncount.value == 1
    assert root.par.Dets.value == 1

    flat = root.children["boxdata"]._pixel
    assert abs(flat[0] - 0.25) < 1e-6   # x1 = 1/4
    assert abs(flat[1] - 0.25) < 1e-6   # y1 = 1/4
    assert abs(flat[2] - 0.75) < 1e-6   # x2 = 3/4
    assert abs(flat[3] - 0.75) < 1e-6   # y2 = 3/4
    assert flat[4] == 0.0               # class_id
    assert abs(flat[5] - 0.9) < 1e-6    # confidence
    assert flat[6:12].sum() == 0.0      # second slot untouched

    rows = root.children["detections"].rows
    assert len(rows) == 2
    assert rows[1][1] == "person"

    # Union rebuilt from kept boxes only: 2x2 region.
    mask = st.copied[-1]
    assert mask.sum() == 16.0


def test_oncook_missing_status_dat():
    root = make_root(with_status=False)
    st = make_script_top(root)
    st.addInput(np.zeros((4, 4, 3), dtype=np.float32))
    mod = _stubs.load_pyt(PYT, me=st)

    with _stubs.StubbedTorchRfdetr(cuda_available=False):
        mod.onCook(st)

    assert root.par.State.value == "ready"
    assert st.copied and st.copied[-1].shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    tests = [
        test_root_is_parent,
        test_root_none_without_parent,
        test_source_has_no_absolute_paths,
        test_core_module_resolves_relative,
        test_oncook_no_deps,
        test_oncook_missing_input,
        test_oncook_full_path_no_detections,
        test_oncook_whitelist_filters,
        test_oncook_missing_status_dat,
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
