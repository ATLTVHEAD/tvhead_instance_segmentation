# rfdetr.python.extension
#
# Callbacks DAT for the `infer` Script TOP.
#
# This file is the TouchDesigner glue layer.
# RF-DETR inference itself lives in the `core` DAT.
#
# The owning component is resolved component-relatively (me.parent()),
# so the TOX works at any path and under any instance name.

import time
import types
import numpy as np


# ---------------------------------------------------------------------------
# Component configuration
# ---------------------------------------------------------------------------

MAX_DETS = 64
BOXDATA_W = MAX_DETS * 6


# Cached Python module containing the contents of the core DAT.
_core = None


# ---------------------------------------------------------------------------
# Root component
# ---------------------------------------------------------------------------


def _root():
    """Return the RF-DETR component that owns this Script TOP."""

    try:
        return me.parent()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core module
# ---------------------------------------------------------------------------


def core_module():
    """Load the Python code from the core DAT once.

    The core DAT is pure Python and does not import TouchDesigner.

    Raises:
        RuntimeError: if the component or core DAT cannot be found.
    """

    global _core

    if _core is not None:
        return _core

    root = _root()

    if root is None:
        raise RuntimeError(
            "RF-DETR root component not found (me.parent() is None)"
        )

    core_dat_path = "python/core"
    core_dat = root.op(core_dat_path)

    if core_dat is None:
        raise RuntimeError(
            "RF-DETR core DAT not found: rfdetr/%s" % core_dat_path
        )

    if not hasattr(core_dat, "text"):
        raise RuntimeError(
            "RF-DETR core operator has no .text property: %s" % core_dat_path
        )

    mod = types.ModuleType("rfdetr_core")

    try:
        exec(
            compile(
                core_dat.text,
                "core.py",
                "exec",
            ),
            mod.__dict__,
        )
    except Exception as e:
        raise RuntimeError("Failed to load RF-DETR core.py: %s" % e) from e

    _core = mod

    # Store it on the component as well so other TD scripts can access it.
    try:
        root.store("rfdetr_core", _core)
    except Exception:
        pass

    return _core


# ---------------------------------------------------------------------------
# Safe parameter helpers
# ---------------------------------------------------------------------------


def _has_par(root, name):
    """Return True when a parameter exists on the root component."""

    if root is None:
        return False

    try:
        return hasattr(root.par, name)
    except Exception:
        return False


def _get_par(root, name, default=None):
    """Safely read a parameter."""

    if root is None:
        return default

    try:
        par = getattr(root.par, name)
    except Exception:
        return default

    try:
        return par.eval()
    except Exception:
        return default


def _get_text_par(root, name, default=""):
    """Safely read a text parameter."""

    if root is None:
        return default

    try:
        par = getattr(root.par, name)
    except Exception:
        return default

    try:
        value = par.eval()
    except Exception:
        try:
            value = par.text
        except Exception:
            return default

    if value is None:
        return default

    return str(value)


def _set_par(root, name, value):
    """Safely write a parameter.

    Returns True if the parameter was found and written.
    """

    if root is None:
        return False

    try:
        par = getattr(root.par, name)
    except Exception:
        return False

    try:
        par.val = value
        return True
    except Exception:
        try:
            setattr(root.par, name, value)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Status helper
# ---------------------------------------------------------------------------


def _set_status(root, state=None, message=None):
    """Update component State and status DAT without throwing secondary errors."""

    if root is None:
        return

    if state is not None:
        _set_par(root, "State", state)

    if message is None:
        return

    try:
        status_op = root.op("status")
    except Exception:
        status_op = None

    if status_op is None:
        return

    try:
        status_op.text = str(message)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def _device(root=None):
    """Return requested device or None for automatic selection."""

    if root is None:
        root = _root()

    value = _get_par(
        root,
        "Device",
        None,
    )

    if value is None:
        return None

    value = str(value).lower().strip()

    if value in ("cuda", "cpu"):
        return value

    return None


# ---------------------------------------------------------------------------
# Class whitelist
# ---------------------------------------------------------------------------


def _whitelist(root=None):
    """Return a set of allowed class names.

    Empty whitelist means no filtering.
    """

    if root is None:
        root = _root()

    text = _get_text_par(
        root,
        "Classwhitelist",
        "",
    ).strip()

    if not text:
        return None

    return {item.strip() for item in text.split(",") if item.strip()}


# ---------------------------------------------------------------------------
# Main Script TOP callback
# ---------------------------------------------------------------------------


def onCook(scriptOp):
    """Run one RF-DETR inference pass."""

    root = _root()

    # ---------------------------------------------------------------
    # Make sure the component exists.
    # ---------------------------------------------------------------

    if root is None:
        _emit_empty(scriptOp, None)

        # Nothing else can safely be done without the root component.
        return

    # ---------------------------------------------------------------
    # Load dependencies / core / model.
    # ---------------------------------------------------------------

    try:
        import torch  # noqa: F401

        core = core_module()

        variant = _get_par(
            root,
            "Model",
            "rfdetr-seg-medium",
        )

        checkpoint = _get_text_par(
            root,
            "Customcheckpoint",
            "",
        ).strip()

        model = core.load_model(
            variant,
            device=_device(root),
            checkpoint=checkpoint or None,
        )

    except ImportError as e:
        message = "[no-deps] %s\nRun Setup > Install Dependencies." % e

        _set_status(
            root,
            "no-deps",
            message,
        )

        _emit_empty(
            scriptOp,
            root,
        )

        return

    except Exception as e:
        message = "[error] %s" % e

        _set_status(
            root,
            "error",
            message,
        )

        _emit_empty(
            scriptOp,
            root,
        )

        return

    # ---------------------------------------------------------------
    # Input image
    # ---------------------------------------------------------------

    try:
        src = scriptOp.inputs[0] if scriptOp.inputs else None

        if src is None:
            _set_status(
                root,
                "error",
                "[error] infer Script TOP has no input.",
            )

            _emit_empty(
                scriptOp,
                root,
            )

            return

        h = int(src.height)
        w = int(src.width)

        if h <= 0 or w <= 0:
            _set_status(
                root,
                "error",
                "[error] invalid input dimensions: %dx%d" % (w, h),
            )

            _emit_empty(
                scriptOp,
                root,
            )

            return

        frame = np.asarray(
            src.numpyArray(),
        )

        if frame.ndim != 3:
            raise RuntimeError(
                "Expected input image with 3 dimensions, got %s" % (frame.shape,)
            )

        if frame.shape[2] >= 3:
            rgb = frame[..., :3]
        elif frame.shape[2] == 1:
            rgb = np.repeat(
                frame[..., :1],
                3,
                axis=2,
            )
        else:
            raise RuntimeError(
                "Input image has invalid channel count: %s" % (frame.shape,)
            )

        frame_u8 = (
            np.clip(
                rgb,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

    except Exception as e:
        _set_status(
            root,
            "error",
            "[error] input conversion: %s" % e,
        )

        _emit_empty(
            scriptOp,
            root,
        )

        return

    # ---------------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------------

    try:
        conf_val = float(
            _get_par(
                root,
                "Confidence",
                0.5,
            )
        )

        conf_val = max(
            0.0,
            min(
                1.0,
                conf_val,
            ),
        )

    except Exception:
        conf_val = 0.5

    # ---------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------

    try:
        t0 = time.perf_counter()

        out = core.predict_frame(
            model,
            frame_u8,
            threshold=conf_val,
        )

        ms = (time.perf_counter() - t0) * 1000.0

    except Exception as e:
        _set_status(
            root,
            "error",
            "[error] inference: %s" % e,
        )

        _emit_empty(
            scriptOp,
            root,
        )

        return

    # ---------------------------------------------------------------
    # Extract results
    # ---------------------------------------------------------------

    boxes = out["boxes"]
    ids = out["class_ids"]
    names = out["class_names"]
    confs = out["confidences"]
    masks = out["masks"]
    union = out["union"]

    # ---------------------------------------------------------------
    # Class whitelist
    # ---------------------------------------------------------------

    wl = _whitelist(root)

    if wl is not None:
        keep = np.array(
            [str(name) in wl for name in names],
            dtype=bool,
        )

        boxes = boxes[keep]
        ids = ids[keep]
        confs = confs[keep]
        names = names[keep]

        if masks is not None:
            masks = masks[keep]

        union = core.build_union(
            masks=masks,
            boxes=boxes,
            h=h,
            w=w,
        )

    # ---------------------------------------------------------------
    # Output mask
    # ---------------------------------------------------------------

    try:
        _emit_mask(
            scriptOp,
            union,
        )
    except Exception as e:
        _set_status(
            root,
            "error",
            "[error] mask output: %s" % e,
        )

        return

    # ---------------------------------------------------------------
    # Detection outputs
    # ---------------------------------------------------------------

    _emit_boxdata(
        root,
        boxes,
        ids,
        confs,
        w,
        h,
    )

    _emit_chop(
        root,
        boxes,
        ids,
        names,
        confs,
    )

    # ---------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------

    _set_par(
        root,
        "State",
        "ready",
    )

    _set_par(
        root,
        "Ncount",
        len(ids),
    )

    _set_par(
        root,
        "Inferms",
        ms,
    )

    _set_par(
        root,
        "Dets",
        min(
            len(ids),
            MAX_DETS,
        ),
    )

    _set_status(
        root,
        None,
        "[ready] infer: %.1fms | detections: %d"
        % (
            ms,
            len(ids),
        ),
    )


# ---------------------------------------------------------------------------
# Mask output
# ---------------------------------------------------------------------------


def _emit_mask(scriptOp, union):
    """Write the detection union into the Script TOP."""

    union = np.asarray(
        union,
        dtype=np.float32,
    )

    if union.ndim != 2:
        raise ValueError("Union mask must be 2D, got %s" % (union.shape,))

    # Stack into 4 channels (RGBA) so TD gets tightly-packed 4-component pixel data
    # (H, W) -> (H, W, 4)
    output = np.stack([union] * 4, axis=-1)

    # Ensure C-contiguous memory layout
    output = np.ascontiguousarray(output, dtype=np.float32)

    scriptOp.copyNumpyArray(output)


# ---------------------------------------------------------------------------
# Empty output
# ---------------------------------------------------------------------------


def _emit_empty(scriptOp, root=None):
    """Produce an empty mask and reset detection counters."""

    try:
        if scriptOp.inputs:
            src = scriptOp.inputs[0]

            _emit_mask(
                scriptOp,
                np.zeros(
                    (
                        int(src.height),
                        int(src.width),
                    ),
                    dtype=np.float32,
                ),
            )

    except Exception:
        pass

    if root is None:
        root = _root()

    if root is None:
        return

    _set_par(
        root,
        "Dets",
        0,
    )

    _set_par(
        root,
        "Ncount",
        0,
    )


# ---------------------------------------------------------------------------
# Box data TOP
# ---------------------------------------------------------------------------


def _emit_boxdata(
    root,
    boxes,
    ids,
    confs,
    w,
    h,
):
    """Write normalized detection data to the boxdata TOP."""

    if root is None:
        return

    if w <= 0 or h <= 0:
        return

    flat = np.zeros(
        BOXDATA_W,
        dtype=np.float32,
    )

    n = min(
        len(ids),
        MAX_DETS,
    )

    for i in range(n):
        x1, y1, x2, y2 = boxes[i]

        b = i * 6

        flat[b + 0] = float(x1) / float(w)

        flat[b + 1] = float(y1) / float(h)

        flat[b + 2] = float(x2) / float(w)

        flat[b + 3] = float(y2) / float(h)

        flat[b + 4] = float(ids[i])

        flat[b + 5] = float(confs[i])

    try:
        bd = root.op("boxdata")
    except Exception:
        bd = None

    if bd is None:
        return

    try:
        bd.par.width = BOXDATA_W
        bd.par.height = 1
        bd.pixel = flat
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Detection table
# ---------------------------------------------------------------------------


def _emit_chop(
    root,
    boxes,
    ids,
    names,
    confs,
):
    """Write detections to the detections DAT."""

    if root is None:
        return

    rows = [
        [
            "class_id",
            "class",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "area",
            "frame",
        ]
    ]

    try:
        fr = int(app.frame)
    except Exception:
        fr = 0

    for i in range(len(ids)):
        x1, y1, x2, y2 = boxes[i]

        area = max(
            0.0,
            float(x2) - float(x1),
        ) * max(
            0.0,
            float(y2) - float(y1),
        )

        rows.append([
            int(ids[i]),
            str(names[i]),
            round(
                float(confs[i]),
                4,
            ),
            round(
                float(x1),
                1,
            ),
            round(
                float(y1),
                1,
            ),
            round(
                float(x2),
                1,
            ),
            round(
                float(y2),
                1,
            ),
            round(
                area,
                1,
            ),
            fr,
        ])

    try:
        det_op = root.op("detections")
    except Exception:
        det_op = None

    if det_op is None:
        return

    try:
        det_op.clear()
        det_op.appendRows(rows)
    except Exception:
        try:
            det_op.tableData = rows
        except Exception:
            pass
