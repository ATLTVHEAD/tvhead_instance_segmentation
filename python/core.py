"""Pure RF-DETR inference core for the TouchDesigner TOX.

No TouchDesigner imports here.

This module is imported by the Script TOP glue (extension.py) inside
TouchDesigner, and can also be imported headlessly by tests.

The Python environment is handled by tdPyEnvManager. This module does not
modify sys.path or install packages.

RF-DETR / Torch are imported lazily so numpy-only helpers such as build_union()
and chromakey() work even before the model dependencies are installed.

Expected environment:
    Python 3.11
    torch
    torchvision
    rfdetr
    supervision
    opencv-python-headless
"""

import numpy as np


# ---------------------------------------------------------------------------
# RF-DETR model variants
# ---------------------------------------------------------------------------
#
# Key:
#     Model identifier used by the TouchDesigner Model parameter.
#
# Value:
#     Class exported at the top level of the rfdetr package.
#
# These are the COCO-pretrained variants supported by this component.
#

VARIANTS = {
    "rfdetr-nano": "RFDETRNano",
    "rfdetr-small": "RFDETRSmall",
    "rfdetr-medium": "RFDETRMedium",
    "rfdetr-large": "RFDETRLarge",
    "rfdetr-seg-nano": "RFDETRSegNano",
    "rfdetr-seg-small": "RFDETRSegSmall",
    "rfdetr-seg-medium": "RFDETRSegMedium",
    "rfdetr-seg-large": "RFDETRSegLarge",
    "rfdetr-seg-xlarge": "RFDETRSegXLarge",
    "rfdetr-seg-2xlarge": "RFDETRSeg2XLarge",
}


SEG_VARIANTS = frozenset({
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "rfdetr-seg-medium",
    "rfdetr-seg-large",
    "rfdetr-seg-xlarge",
    "rfdetr-seg-2xlarge",
})


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_MODEL_CACHE = {}


# ---------------------------------------------------------------------------
# Environment / dependency helpers
# ---------------------------------------------------------------------------


def check_dependencies():
    """Check that the required runtime packages are available.

    Returns:
        dict containing package availability and versions.

    This does not install anything. Installation is handled by
    tdPyEnvManager / requirements.txt.
    """

    result = {
        "torch": False,
        "torchvision": False,
        "rfdetr": False,
        "supervision": False,
        "opencv": False,
        "cuda": False,
        "torch_version": None,
        "torchvision_version": None,
        "rfdetr_version": None,
        "supervision_version": None,
        "opencv_version": None,
        "error": None,
    }

    try:
        import torch

        result["torch"] = True
        result["torch_version"] = getattr(torch, "__version__", None)
        result["cuda"] = bool(torch.cuda.is_available())

    except Exception as e:
        result["error"] = "torch: %s" % e
        return result

    try:
        import torchvision

        result["torchvision"] = True
        result["torchvision_version"] = getattr(
            torchvision,
            "__version__",
            None,
        )

    except Exception:
        pass

    try:
        import rfdetr

        result["rfdetr"] = True
        result["rfdetr_version"] = getattr(
            rfdetr,
            "__version__",
            None,
        )

    except Exception:
        pass

    try:
        import supervision

        result["supervision"] = True
        result["supervision_version"] = getattr(
            supervision,
            "__version__",
            None,
        )

    except Exception:
        pass

    try:
        import cv2

        result["opencv"] = True
        result["opencv_version"] = getattr(
            cv2,
            "__version__",
            None,
        )

    except Exception:
        pass

    return result


def cuda_device_info():
    """Return information about the active CUDA device.

    Returns None when CUDA is unavailable.
    """

    import torch

    if not torch.cuda.is_available():
        return None

    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)

    return {
        "index": device_index,
        "name": props.name,
        "total_memory": int(props.total_memory),
        "total_memory_gb": float(props.total_memory) / 1e9,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(variant, device=None, checkpoint=None):
    """Load or return a cached RF-DETR model.

    Args:
        variant:
            Key from VARIANTS.

        device:
            "cuda", "cpu", or None.

            None automatically selects CUDA when available.

        checkpoint:
            Optional custom checkpoint path.

            When provided, the RF-DETR checkpoint loader is used instead of
            the pretrained variant constructor.

    Returns:
        RF-DETR model wrapper.
    """

    import torch
    import rfdetr

    # ---------------------------------------------------------------
    # Select device
    # ---------------------------------------------------------------

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = str(device).lower()

    if device not in ("cuda", "cpu"):
        raise ValueError("Unsupported device %r. Expected 'cuda' or 'cpu'." % device)

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    target = torch.device(device)

    # ---------------------------------------------------------------
    # Custom checkpoint
    # ---------------------------------------------------------------

    if checkpoint:
        key = (
            "checkpoint",
            str(checkpoint),
            str(target),
        )

        if key not in _MODEL_CACHE:
            if not hasattr(rfdetr.RFDETR, "from_checkpoint"):
                raise RuntimeError(
                    "This RF-DETR version does not expose RFDETR.from_checkpoint()."
                )

            model = rfdetr.RFDETR.from_checkpoint(checkpoint)
            model = _place(model, target)

            _MODEL_CACHE[key] = model

        return _MODEL_CACHE[key]

    # ---------------------------------------------------------------
    # Pretrained variant
    # ---------------------------------------------------------------

    if variant not in VARIANTS:
        raise ValueError(
            "Unknown RF-DETR variant %r. Expected one of: %s"
            % (
                variant,
                ", ".join(sorted(VARIANTS)),
            )
        )

    key = (
        "variant",
        variant,
        str(target),
    )

    if key not in _MODEL_CACHE:
        class_name = VARIANTS[variant]

        if not hasattr(rfdetr, class_name):
            raise RuntimeError(
                "Installed rfdetr package does not expose %s. "
                "Installed RF-DETR version: %s"
                % (
                    class_name,
                    getattr(rfdetr, "__version__", "unknown"),
                )
            )

        cls = getattr(rfdetr, class_name)

        model = cls()

        model = _place(model, target)

        _MODEL_CACHE[key] = model

    return _MODEL_CACHE[key]


def _place(model, device):
    """Set the device on an RF-DETR model wrapper.

    RF-DETR's public variant objects are wrappers rather than ordinary
    torch.nn.Module objects, so .to()/.eval() should not be called here.

    RF-DETR stores its device on the underlying model context.
    """

    ctx = getattr(model, "model", None)

    if ctx is not None:
        ctx.device = device

    return model


def clear_model_cache():
    """Clear cached RF-DETR models.

    Useful when changing models/devices or releasing GPU memory.
    """

    global _MODEL_CACHE

    _MODEL_CACHE.clear()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception:
        pass


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_frame(model, frame_u8, threshold=0.5):
    """Run RF-DETR inference on one RGB uint8 frame.

    Args:
        model:
            Loaded RF-DETR model.

        frame_u8:
            NumPy array with shape (H, W, 3), dtype uint8.

        threshold:
            Detection confidence threshold.

    Returns:
        dict containing:

            boxes
                (N, 4) float32 xyxy coordinates.

            class_ids
                (N,) int32 class IDs.

            class_names
                (N,) object array of class names.

            confidences
                (N,) float32 confidence values.

            masks
                (N, H, W) bool masks, or None.

            union
                (H, W) float32 mask where detected pixels are 1.
    """

    frame_u8 = np.asarray(frame_u8)

    # ---------------------------------------------------------------
    # Validate frame
    # ---------------------------------------------------------------

    if frame_u8.ndim != 3:
        raise ValueError(
            "frame_u8 must have shape (H, W, 3), got %s" % (frame_u8.shape,)
        )

    if frame_u8.shape[2] != 3:
        raise ValueError(
            "frame_u8 must have 3 channels, got shape %s" % (frame_u8.shape,)
        )

    if frame_u8.dtype != np.uint8:
        raise ValueError("frame_u8 must have dtype uint8, got %s" % frame_u8.dtype)

    h, w = frame_u8.shape[:2]

    # RF-DETR may write through the supplied array.
    if not frame_u8.flags.writeable:
        frame_u8 = frame_u8.copy()

    # ---------------------------------------------------------------
    # Run inference
    # ---------------------------------------------------------------

    det = model.predict(
        frame_u8,
        threshold=float(threshold),
        include_source_image=False,
    )

    if det is None:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "class_ids": np.zeros(0, dtype=np.int32),
            "class_names": np.zeros(0, dtype=object),
            "confidences": np.zeros(0, dtype=np.float32),
            "masks": None,
            "union": np.zeros((h, w), dtype=np.float32),
        }

    n = len(det)

    # ---------------------------------------------------------------
    # No detections
    # ---------------------------------------------------------------

    if n == 0:
        return {
            "boxes": np.zeros((0, 4), dtype=np.float32),
            "class_ids": np.zeros(0, dtype=np.int32),
            "class_names": np.zeros(0, dtype=object),
            "confidences": np.zeros(0, dtype=np.float32),
            "masks": None,
            "union": np.zeros((h, w), dtype=np.float32),
        }

    # ---------------------------------------------------------------
    # Boxes
    # ---------------------------------------------------------------

    boxes = np.asarray(
        det.xyxy,
        dtype=np.float32,
    )

    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise RuntimeError("Unexpected RF-DETR boxes shape: %s" % (boxes.shape,))

    # ---------------------------------------------------------------
    # Class IDs
    # ---------------------------------------------------------------

    class_ids = np.asarray(
        det.class_id,
        dtype=np.int32,
    ).reshape(-1)

    # ---------------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------------

    confidences = np.asarray(
        det.confidence,
        dtype=np.float32,
    ).reshape(-1)

    # ---------------------------------------------------------------
    # Class names
    # ---------------------------------------------------------------

    names = np.asarray(
        det.data.get("class_name", []),
        dtype=object,
    ).reshape(-1)

    if names.size != n:
        names = np.full(
            n,
            "",
            dtype=object,
        )

    # ---------------------------------------------------------------
    # Masks
    # ---------------------------------------------------------------

    masks = getattr(det, "mask", None)

    if masks is not None:
        masks = np.asarray(
            masks,
            dtype=bool,
        )

        if masks.ndim == 2:
            masks = masks[None, ...]

        if masks.ndim != 3:
            raise RuntimeError("Unexpected RF-DETR mask shape: %s" % (masks.shape,))

        # RF-DETR normally returns input-resolution masks, but crop/pad
        # defensively so the union always matches the input frame.

        if masks.shape[1] != h or masks.shape[2] != w:
            normalized = np.zeros(
                (masks.shape[0], h, w),
                dtype=bool,
            )

            copy_h = min(h, masks.shape[1])
            copy_w = min(w, masks.shape[2])

            normalized[:, :copy_h, :copy_w] = masks[:, :copy_h, :copy_w]

            masks = normalized

    # ---------------------------------------------------------------
    # Union mask
    # ---------------------------------------------------------------

    union = build_union(
        masks=masks,
        boxes=boxes,
        h=h,
        w=w,
    )

    return {
        "boxes": boxes,
        "class_ids": class_ids,
        "class_names": names,
        "confidences": confidences,
        "masks": masks,
        "union": union,
    }


# ---------------------------------------------------------------------------
# Union mask
# ---------------------------------------------------------------------------


def build_union(masks=None, boxes=None, h=0, w=0):
    """Build an (H, W) float32 union mask.

    Args:
        masks:
            (N, H, W) bool masks.

        boxes:
            (N, 4) xyxy boxes.

        h, w:
            Output dimensions.

    Masks take precedence over boxes.

    Returns:
        float32 array containing only 0.0 and 1.0.
    """

    h = int(h)
    w = int(w)

    if h <= 0 or w <= 0:
        return np.zeros(
            (max(h, 0), max(w, 0)),
            dtype=np.float32,
        )

    union = np.zeros(
        (h, w),
        dtype=np.float32,
    )

    # ---------------------------------------------------------------
    # Segmentation masks
    # ---------------------------------------------------------------

    if masks is not None and len(masks) > 0:
        m = np.asarray(
            masks,
            dtype=bool,
        )

        if m.ndim == 2:
            m = m[None, ...]

        if m.ndim != 3:
            raise ValueError("masks must have shape (N,H,W), got %s" % (m.shape,))

        acc = np.zeros(
            (h, w),
            dtype=bool,
        )

        copy_h = min(h, m.shape[1])
        copy_w = min(w, m.shape[2])

        if copy_h > 0 and copy_w > 0:
            for i in range(m.shape[0]):
                np.logical_or(
                    m[i, :copy_h, :copy_w],
                    acc[:copy_h, :copy_w],
                    out=acc[:copy_h, :copy_w],
                )

        union[acc] = 1.0

        return union

    # ---------------------------------------------------------------
    # Detection boxes
    # ---------------------------------------------------------------

    if boxes is not None and len(boxes) > 0:
        b = np.asarray(
            boxes,
            dtype=np.float64,
        )

        if b.ndim != 2 or b.shape[1] != 4:
            raise ValueError("boxes must have shape (N,4), got %s" % (b.shape,))

        for box in b:
            x1, y1, x2, y2 = box

            # Handle malformed boxes.
            if not np.all(np.isfinite(box)):
                continue

            xa = int(np.floor(x1))
            ya = int(np.floor(y1))
            xb = int(np.ceil(x2))
            yb = int(np.ceil(y2))

            xa = max(0, min(xa, w))
            xb = max(0, min(xb, w))
            ya = max(0, min(ya, h))
            yb = max(0, min(yb, h))

            if xb > xa and yb > ya:
                union[ya:yb, xa:xb] = 1.0

        return union

    return union


# ---------------------------------------------------------------------------
# Chromakey
# ---------------------------------------------------------------------------


def chromakey(
    src,
    mask,
    green=(0.0, 1.0, 0.0),
    invert=False,
):
    """Apply the NumPy equivalent of the TOX chromakey operation.

    Args:
        src:
            (H, W, 3) float32 image in 0..1.

        mask:
            (H, W) float32 mask in 0..1.
            1 = subject.

        green:
            RGB chromakey color.

        invert:
            False:
                subject kept, background becomes green.

            True:
                subject becomes green, background kept.

    Returns:
        (H, W, 3) float32 image.
    """

    src = np.asarray(
        src,
        dtype=np.float32,
    )

    mask = np.asarray(
        mask,
        dtype=np.float32,
    )

    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError("src must have shape (H,W,3), got %s" % (src.shape,))

    if mask.ndim != 2:
        raise ValueError("mask must have shape (H,W), got %s" % (mask.shape,))

    if src.shape[:2] != mask.shape:
        raise ValueError(
            "src and mask dimensions do not match: %s vs %s"
            % (
                src.shape[:2],
                mask.shape,
            )
        )

    g = np.asarray(
        green,
        dtype=np.float32,
    )

    if g.shape != (3,):
        raise ValueError("green must contain exactly 3 values, got %s" % (g.shape,))

    m = mask[..., None]

    if invert:
        return src * (1.0 - m) + g * m

    return g * (1.0 - m) + src * m
