# Headless test suite for python/core.py (the pure inference core).
#
# Run from the repo root:
#     python tests\test_core.py
#
# Conventions (used by all suites in this directory):
#   - plain asserts, no pytest
#   - main() prints "<N>/<N> pass" and exits non-zero on any failure
#   - no binary test assets; frames are synthesized in code

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Load core.py by path (it is not a package module).
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "rfdetr_core_under_test",
    _REPO_ROOT / "python" / "core.py",
)
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError("expected %r, nothing raised" % (exc_type,))


class StubbedTorchRfdetr:
    """Replace torch/rfdetr in sys.modules with minimal fakes.

    Saves and restores the original modules on exit.
    """

    def __init__(self, cuda_available=False):
        self.cuda_available = cuda_available
        self._saved = {}

    def _make_torch(self):
        mod = types.ModuleType("torch")
        mod.__version__ = "0.0.0-stub"

        cuda = types.SimpleNamespace(
            is_available=lambda: self.cuda_available,
            empty_cache=lambda: None,
            current_device=lambda: 0,
        )
        mod.cuda = cuda

        def device(spec):
            return spec

        mod.device = device
        return mod

    def _make_rfdetr(self):
        mod = types.ModuleType("rfdetr")
        mod.__version__ = "0.0.0-stub"

        class FakeVariant:
            pass

        for class_name in core.VARIANTS.values():
            setattr(mod, class_name, FakeVariant)

        class RFDETR:
            @staticmethod
            def from_checkpoint(path):
                return FakeVariant()

        mod.RFDETR = RFDETR
        return mod

    def __enter__(self):
        for name in ("torch", "rfdetr"):
            self._saved[name] = sys.modules.get(name)
        sys.modules["torch"] = self._make_torch()
        sys.modules["rfdetr"] = self._make_rfdetr()
        return self

    def __exit__(self, *exc):
        for name, orig in self._saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
        return False


class FakeModel:
    """Stands in for a loaded RF-DETR model wrapper."""

    def __init__(self, result):
        self._result = result
        self.last_frame = None
        self.last_threshold = None

    def predict(self, frame, threshold=None, include_source_image=False):
        self.last_frame = frame
        self.last_threshold = threshold
        return self._result


class FakeDet:
    """Stands in for one RF-DETR Detection result."""

    def __init__(self, xyxy, class_id, confidence, data, mask=None):
        self.xyxy = xyxy
        self.class_id = class_id
        self.confidence = confidence
        self.data = data
        self.mask = mask

    def __len__(self):
        if self.xyxy is None:
            return 0
        return len(np.asarray(self.xyxy))


# ---------------------------------------------------------------------------
# build_union
# ---------------------------------------------------------------------------


def test_build_union_masks_precedence():
    m1 = np.zeros((1, 4, 4), dtype=bool)
    m1[0, 0:2, 0:2] = True
    m2 = np.zeros((1, 4, 4), dtype=bool)
    m2[0, 1:3, 1:3] = True
    masks = np.concatenate([m1, m2], axis=0)

    u = core.build_union(masks=masks, h=4, w=4)

    expected = np.zeros((4, 4))
    expected[0:2, 0:2] = 1.0
    expected[1:3, 1:3] = 1.0

    assert u.dtype == np.float32
    assert np.array_equal(u, expected)
    assert set(np.unique(u)) <= {0.0, 1.0}


def test_build_union_boxes_only():
    boxes = np.array([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0]])
    u = core.build_union(boxes=boxes, h=4, w=4)

    expected = np.zeros((4, 4))
    expected[1:3, 1:3] = 1.0
    expected[0:2, 0:2] = 1.0
    assert np.array_equal(u, expected)

    # Out-of-range box clips to the full frame.
    u2 = core.build_union(boxes=np.array([[-1.0, -1.0, 5.0, 5.0]]), h=4, w=4)
    assert u2.shape == (4, 4)
    assert u2.sum() == 16.0

    # Non-finite box is skipped.
    u3 = core.build_union(boxes=np.array([[float("nan")] * 4]), h=4, w=4)
    assert u3.sum() == 0.0


def test_build_union_degenerate_dims():
    u = core.build_union(masks=np.ones((1, 4, 4), dtype=bool), h=0, w=4)
    assert u.shape == (0, 4)
    assert u.dtype == np.float32

    u = core.build_union(boxes=np.array([[0.0, 0.0, 2.0, 2.0]]), h=4, w=0)
    assert u.shape == (4, 0)

    u = core.build_union(h=-2, w=3)
    assert u.shape == (0, 3)
    assert u.sum() == 0.0


def test_build_union_masks_crop():
    # 5x5 masks on a 4x4 output crop to 4x4.
    big = np.ones((1, 5, 5), dtype=bool)
    u = core.build_union(masks=big, h=4, w=4)
    assert u.shape == (4, 4)
    assert u.sum() == 16.0

    # 2x2 masks on a 4x4 output pad the rest with zeros.
    small = np.ones((1, 2, 2), dtype=bool)
    u = core.build_union(masks=small, h=4, w=4)
    expected = np.zeros((4, 4))
    expected[0:2, 0:2] = 1.0
    assert np.array_equal(u, expected)


# ---------------------------------------------------------------------------
# chromakey
# ---------------------------------------------------------------------------


def test_chromakey_keeps_subject():
    src = np.zeros((4, 4, 3), dtype=np.float32)
    src[...] = (1.0, 0.0, 0.0)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1, 1] = 1.0

    out = core.chromakey(src, mask)

    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float32
    assert tuple(out[1, 1]) == (1.0, 0.0, 0.0)   # subject kept
    assert tuple(out[0, 0]) == (0.0, 1.0, 0.0)   # background -> green


def test_chromakey_invert():
    src = np.zeros((4, 4, 3), dtype=np.float32)
    src[...] = (1.0, 0.0, 0.0)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1, 1] = 1.0

    out = core.chromakey(src, mask, invert=True)

    assert tuple(out[1, 1]) == (0.0, 1.0, 0.0)   # subject -> green
    assert tuple(out[0, 0]) == (1.0, 0.0, 0.0)   # background kept


def test_chromakey_validation():
    src = np.zeros((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=np.float32)

    # Shape mismatch.
    _assert_raises(
        ValueError,
        lambda: core.chromakey(src, np.zeros((3, 4), dtype=np.float32)),
    )
    # 2-channel src.
    _assert_raises(
        ValueError,
        lambda: core.chromakey(np.zeros((4, 4, 2), dtype=np.float32), mask),
    )
    # 3-D mask.
    _assert_raises(
        ValueError,
        lambda: core.chromakey(src, np.zeros((4, 4, 1), dtype=np.float32)),
    )


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def test_load_model_unknown_variant():
    with StubbedTorchRfdetr(cuda_available=False):
        try:
            core.load_model("nope")
        except ValueError as e:
            assert "Unknown RF-DETR variant" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_load_model_variant_cache():
    with StubbedTorchRfdetr(cuda_available=False):
        m1 = core.load_model("rfdetr-nano")
        m2 = core.load_model("rfdetr-nano")
        assert m1 is m2

        core.clear_model_cache()

        m3 = core.load_model("rfdetr-nano")
        assert m3 is not m1


def test_load_model_cuda_unavailable():
    with StubbedTorchRfdetr(cuda_available=False):
        _assert_raises(
            RuntimeError,
            lambda: core.load_model("rfdetr-nano", device="cuda"),
        )


def test_load_model_bad_device_string():
    with StubbedTorchRfdetr(cuda_available=False):
        _assert_raises(
            ValueError,
            lambda: core.load_model("rfdetr-nano", device="gpu"),
        )


# ---------------------------------------------------------------------------
# predict_frame
# ---------------------------------------------------------------------------


def test_predict_frame_none_result():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    model = FakeModel(None)

    out = core.predict_frame(model, frame, threshold=0.3)

    assert out["boxes"].shape == (0, 4)
    assert out["boxes"].dtype == np.float32
    assert out["class_ids"].shape == (0,)
    assert out["class_ids"].dtype == np.int32
    assert out["class_names"].shape == (0,)
    assert out["confidences"].shape == (0,)
    assert out["confidences"].dtype == np.float32
    assert out["masks"] is None
    assert out["union"].shape == (4, 4)
    assert out["union"].dtype == np.float32
    assert out["union"].sum() == 0.0
    assert model.last_threshold == 0.3


def test_predict_frame_with_detections():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    det = FakeDet(
        xyxy=[[1, 1, 3, 3]],
        class_id=[0],
        confidence=[0.9],
        data={"class_name": ["person"]},
    )

    out = core.predict_frame(FakeModel(det), frame, threshold=0.5)

    assert np.array_equal(out["boxes"], np.array([[1, 1, 3, 3]], dtype=np.float32))
    assert out["class_ids"].dtype == np.int32
    assert out["class_ids"][0] == 0
    assert abs(float(out["confidences"][0]) - 0.9) < 1e-6
    assert out["class_names"][0] == "person"
    assert out["masks"] is None

    expected = np.zeros((4, 4), dtype=np.float32)
    expected[1:3, 1:3] = 1.0
    assert np.array_equal(out["union"], expected)


def test_predict_frame_mask_crop():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    det = FakeDet(
        xyxy=[[0, 0, 4, 4]],
        class_id=[1],
        confidence=[0.8],
        data={"class_name": ["car"]},
        mask=np.ones((5, 5), dtype=bool),
    )

    out = core.predict_frame(FakeModel(det), frame)

    assert out["masks"].shape == (1, 4, 4)
    assert out["masks"].dtype == bool
    assert out["union"].shape == (4, 4)
    assert out["union"].sum() == 16.0


def test_predict_frame_bad_frame():
    model = FakeModel(None)

    # float32 frame.
    _assert_raises(
        ValueError,
        lambda: core.predict_frame(model, np.zeros((4, 4, 3), dtype=np.float32)),
    )
    # 2-D frame.
    _assert_raises(
        ValueError,
        lambda: core.predict_frame(model, np.zeros((4, 4), dtype=np.uint8)),
    )
    # 2-channel frame.
    _assert_raises(
        ValueError,
        lambda: core.predict_frame(model, np.zeros((4, 4, 2), dtype=np.uint8)),
    )


# ---------------------------------------------------------------------------
# Dependencies / cache
# ---------------------------------------------------------------------------


def test_check_dependencies_shape():
    info = core.check_dependencies()
    assert set(info) == {
        "torch",
        "torchvision",
        "rfdetr",
        "supervision",
        "opencv",
        "cuda",
        "torch_version",
        "torchvision_version",
        "rfdetr_version",
        "supervision_version",
        "opencv_version",
        "error",
    }


def test_clear_model_cache_evicts():
    with StubbedTorchRfdetr(cuda_available=False):
        core.load_model("rfdetr-nano")
        assert len(core._MODEL_CACHE) == 1

        core.clear_model_cache()

        assert len(core._MODEL_CACHE) == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main():
    tests = [
        test_build_union_masks_precedence,
        test_build_union_boxes_only,
        test_build_union_degenerate_dims,
        test_build_union_masks_crop,
        test_chromakey_keeps_subject,
        test_chromakey_invert,
        test_chromakey_validation,
        test_load_model_unknown_variant,
        test_load_model_variant_cache,
        test_load_model_cuda_unavailable,
        test_load_model_bad_device_string,
        test_predict_frame_none_result,
        test_predict_frame_with_detections,
        test_predict_frame_mask_crop,
        test_predict_frame_bad_frame,
        test_check_dependencies_shape,
        test_clear_model_cache_evicts,
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
