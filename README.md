# TV Head Instance Segmentation (RF-DETR TouchDesigner TOX)

A self-contained TouchDesigner TOX that runs [RF-DETR](https://github.com/robo-flow/rfdetr) real-time instance detection and segmentation inside TouchDesigner. Feed it a source image on the `infer` Script TOP and it outputs a union mask, normalized box data, and a detection table, ready for chromakey compositing. The TOX is component-relative: drop it into any network, rename it if you like, and it works without path edits.

## Requirements

- TouchDesigner with bundled **Python 3.11**.
- A CUDA GPU is **optional**: `Device = auto` selects CUDA when available and falls back to CPU.
- Dependencies install through **tdPyEnvManager** into the project's `.venv` from `requirements.txt` (see `TDPyEnvManagerContext.yaml`).

## Quick start

1. Drop `rfdetr.tox` into any network.
2. Run the tdPyEnvManager setup so the `.venv` is created and populated.
3. Wire a source image into the `infer` Script TOP.
4. Set `Model` and `Device` on the rfdetr container, then let it cook.

Outputs:

- **mask TOP** — RGBA 0/1 union of all detected instances.
- **`boxdata` TOP** — 384 floats (64 detections × 6): normalized `x1, y1, x2, y2, class_id, confidence` per slot.
- **`detections` DAT** — one row per detection: `class_id, class, confidence, x1, y1, x2, y2, area, frame`.
- **Params** — `State`, `Ncount`, `Inferms`, `Dets` for programmatic use.

## Parameters

| Name | Type | Default | Meaning |
|---|---|---|---|
| `Model` | menu | `rfdetr-seg-medium` | RF-DETR variant to load (10 variants). |
| `Device` | menu | `auto` | `auto` (CUDA if available, else CPU), `cuda`, or `cpu`. |
| `Confidence` | float | `0.5` | Minimum detection confidence threshold. |
| `Chromasharp` | float | `0.0` | Mask edge sharpening (0 = none, 1 = max). |
| `Chromainvert` | int | `0` | Invert the mask (1 = inverted). |
| `Chromacolor` | RGB | `(0, 1, 0)` | Chromakey fill color (green). |
| `Classwhitelist` | str | `""` | Comma-separated class names to keep; empty = no filter. |
| `Customcheckpoint` | str | `""` | Path to a custom `.ckpt`; empty = built-in variant. |
| `Hidesource` | int | `0` | 1 = output mask only, 0 = composite source over the key. |
| `State` | str | `init` | Pipeline state: `init`, `loading`, `ready`, `no-deps`, `error`. |
| `Ncount` | int | `0` | Detections after whitelist filtering. |
| `Inferms` | float | `0.0` | Last inference time in milliseconds. |
| `Dets` | int | `0` | Raw detection count before filtering. |

## Repository layout

- `python/extension.pyt` — code for the `infer` Script TOP (onCook glue: model load, inference, output emission).
- `python/callbacks.pyt` — container Callbacks DAT (model/device/checkpoint changes clear the model cache and flip `State` to `loading`).
- `python/add_params.pyt` — `python/add_params` DAT (idempotent installer for the 13 frozen parameters).
- `python/core.py` — pure inference core (model loading, prediction, mask/box utilities); loaded from the `python/core` DAT.
- `TDPyEnvManagerContext.yaml` — tdPyEnvManager environment (`.venv`, Python 3.11).
- `tests/` — headless test suites (no TouchDesigner required).
- `docs/` — architecture, implementation guide, and the new-implementation spec.

## Development

Edit a `.pyt` file, sync it into the matching DAT in `rfdetr.tox` (see [STATUS.html](STATUS.html) §4), then run the headless suites:

```bash
python tests\test_core.py
python tests\test_extension.py
python tests\test_add_params.py
python tests\test_callbacks.py
```

Changes to `python/core.py` must keep its top-level imports **numpy-only** — torch/rfdetr are imported lazily inside functions so the TOX can load without dependencies installed.

## Troubleshooting

- `[no-deps]` in the status DAT — run the tdPyEnvManager setup so the `.venv` is created and populated.
- First use of a model downloads its weights to `%USERPROFILE%\.roboflow\models`; the first cook is slow.
- `[error]` lines in the status DAT name the failing stage (e.g. missing input, unsupported device).

## Status

See [STATUS.html](STATUS.html) for the implementation checklist and current state.
