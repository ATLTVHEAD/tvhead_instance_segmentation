"""Headless tests for the pure RF-DETR core (python/core.py).

Run with TouchDesigner's Python:

    "C:\\Program Files\\Derivative\\TouchDesigner\\bin\\python.exe" tests\\test_core.py

Unit tests need only numpy. The end-to-end tests download model weights
on first run (to %USERPROFILE%\\.roboflow\\models) and use CUDA when
available, else CPU.
"""
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "python"))

import numpy as np

from core import (
    VARIANTS,
    SEG_VARIANTS,
    build_union,
    chromakey,
    load_model,
    predict_frame,
)

RESULTS = []
SAMPLE = os.path.join(HERE, "sample.jpg")


def check(name, fn):
    t0 = time.time()
    try:
        fn()
        RESULTS.append((name, True, ""))
        print(f"PASS  {name}  ({time.time() - t0:.1f}s)")
    except Exception:
        tb = traceback.format_exc()
        RESULTS.append((name, False, tb))
        print(f"FAIL  {name}\n{tb}")


# ------------------------------------------------------------- unit: variants

def test_variant_table():
    assert len(VARIANTS) == 10, f"expected 10 variants, got {len(VARIANTS)}"
    assert len(SEG_VARIANTS) == 6
    assert SEG_VARIANTS <= set(VARIANTS)
    for name in VARIANTS:
        assert name.startswith("rfdetr-"), name


# ------------------------------------------------------------- unit: build_union

def test_union_from_masks():
    h, w = 10, 20
    m0 = np.zeros((h, w), bool)
    m0[0:5, 0:10] = True
    m1 = np.zeros((h, w), bool)
    m1[5:10, 10:20] = True
    m0[5, 5] = True  # overlap pixel
    u = build_union(masks=np.stack([m0, m1]), boxes=None, h=h, w=w)
    assert u.shape == (h, w)
    assert u.dtype == np.float32
    assert u[2, 2] == 1.0   # covered by m0 only
    assert u[7, 15] == 1.0  # covered by m1 only
    assert u[5, 5] == 1.0   # overlap
    assert u[0, 15] == 0.0  # covered by neither


def test_union_from_boxes():
    h, w = 10, 20
    boxes = np.array(
        [[0.0, 0.0, 9.5, 9.5], [10.0, 5.0, 19.9, 9.9]], dtype=np.float32)
    u = build_union(masks=None, boxes=boxes, h=h, w=w)
    assert u[0, 0] == 1.0
    assert u[9, 9] == 1.0    # 9.5 -> cols/rows 0..9 inclusive
    assert u[7, 15] == 1.0
    assert u[0, 15] == 0.0   # row 0 outside box 2 (rows 5..9)


def test_union_boxes_clamped():
    h, w = 5, 5
    boxes = np.array([[-3.0, -3.0, 3.0, 3.0]], dtype=np.float32)
    u = build_union(masks=None, boxes=boxes, h=h, w=w)
    assert u[0, 0] == 1.0
    assert u.sum() == 9.0    # rows 0..2, cols 0..2 (negative side clamped)


def test_union_empty():
    u = build_union(masks=None, boxes=None, h=4, w=6)
    assert u.shape == (4, 6)
    assert u.dtype == np.float32
    assert u.max() == 0.0


def test_union_masks_take_precedence():
    h, w = 4, 4
    masks = np.zeros((1, h, w), bool)
    masks[0, 0, 0] = True
    boxes = np.array([[0.0, 0.0, 3.0, 3.0]], dtype=np.float32)
    u = build_union(masks=masks, boxes=boxes, h=h, w=w)
    assert u.sum() == 1.0    # masks present -> boxes ignored


# ------------------------------------------------------------- unit: chromakey

def test_chromakey_forward():
    src = np.zeros((4, 4, 3), np.float32)
    src[...] = [1.0, 0.5, 0.25]
    m = np.zeros((4, 4), np.float32)
    m[1:3, 1:3] = 1.0
    out = chromakey(src, m, green=(0.0, 1.0, 0.0), invert=False)
    assert out.shape == src.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], [0.0, 1.0, 0.0], atol=1e-6)    # bg -> green
    np.testing.assert_allclose(out[2, 2], [1.0, 0.5, 0.25], atol=1e-6)   # subject kept


def test_chromakey_invert():
    src = np.zeros((4, 4, 3), np.float32)
    src[...] = [1.0, 0.5, 0.25]
    m = np.zeros((4, 4), np.float32)
    m[1:3, 1:3] = 1.0
    out = chromakey(src, m, green=(0.0, 1.0, 0.0), invert=True)
    np.testing.assert_allclose(out[0, 0], [1.0, 0.5, 0.25], atol=1e-6)   # bg kept
    np.testing.assert_allclose(out[2, 2], [0.0, 1.0, 0.0], atol=1e-6)    # subject -> green


def test_chromakey_soft_mask():
    src = np.zeros((2, 2, 3), np.float32)
    src[...] = [1.0, 0.0, 1.0]  # magenta, so forward/invert differ
    m = np.full((2, 2), 0.25, np.float32)
    out_f = chromakey(src, m, green=(0.0, 1.0, 0.0), invert=False)
    out_i = chromakey(src, m, green=(0.0, 1.0, 0.0), invert=True)
    np.testing.assert_allclose(out_f[0, 0], [0.25, 0.75, 0.25], atol=1e-6)
    np.testing.assert_allclose(out_i[0, 0], [0.75, 0.25, 0.75], atol=1e-6)


# ------------------------------------------------------------- end to end

def _load_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def test_e2e_seg_medium():
    assert os.path.exists(SAMPLE), f"missing {SAMPLE}"
    frame = _load_rgb(SAMPLE)
    h, w = frame.shape[:2]
    model = load_model("rfdetr-seg-medium")
    out = predict_frame(model, frame, threshold=0.5)
    assert out["boxes"].shape[1] == 4
    assert out["boxes"].shape[0] >= 1
    names = list(out["class_names"])
    assert "dog" in names, f"expected 'dog' in {names}"
    u = out["union"]
    assert u.shape == (h, w)
    assert u.dtype == np.float32
    assert 0.0 < u.mean() < 1.0
    assert u[0, 0] == 0.0  # top-left corner of dog.jpg is background
    # chromakey mirror: the background corner is painted flat green
    src = frame.astype(np.float32) / 255.0
    ck = chromakey(src, u, green=(0.0, 1.0, 0.0), invert=False)
    np.testing.assert_allclose(ck[0, 0], [0.0, 1.0, 0.0], atol=1e-3)


def test_e2e_det_nano_boxes_only():
    frame = _load_rgb(SAMPLE)
    model = load_model("rfdetr-nano")
    out = predict_frame(model, frame, threshold=0.5)
    assert out["masks"] is None, "detection model must not return masks"
    assert out["boxes"].shape[0] >= 1
    # Close-up sample: the filled boxes cover most of the frame, so only
    # assert the union is non-trivial (not empty, not the whole frame).
    assert 0.0 < out["union"].mean() < 1.0


def test_e2e_mask_resolution_matches_frame():
    # Masks must come back at INPUT frame resolution (not the model's
    # internal square resolution) so the union mask aligns with the frame.
    frame = _load_rgb(SAMPLE)
    h, w = frame.shape[:2]
    model = load_model("rfdetr-seg-medium")
    out = predict_frame(model, frame, threshold=0.3)
    if out["masks"] is not None:
        assert out["masks"].shape[1:] == (h, w)


def main():
    unit = [
        ("variant_table", test_variant_table),
        ("union_from_masks", test_union_from_masks),
        ("union_from_boxes", test_union_from_boxes),
        ("union_boxes_clamped", test_union_boxes_clamped),
        ("union_empty", test_union_empty),
        ("union_masks_take_precedence", test_union_masks_take_precedence),
        ("chromakey_forward", test_chromakey_forward),
        ("chromakey_invert", test_chromakey_invert),
        ("chromakey_soft_mask", test_chromakey_soft_mask),
    ]
    e2e = [
        ("e2e_seg_medium", test_e2e_seg_medium),
        ("e2e_det_nano_boxes_only", test_e2e_det_nano_boxes_only),
        ("e2e_mask_resolution", test_e2e_mask_resolution_matches_frame),
    ]
    skip_e2e = "--unit-only" in sys.argv
    for name, fn in unit:
        check(name, fn)
    if not skip_e2e:
        for name, fn in e2e:
            check(name, fn)
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    if failed:
        print("failed:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
