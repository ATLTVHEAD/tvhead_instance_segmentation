# RF-DETR Chroma TOX — Core Architecture

A self-contained TouchDesigner TOX that wraps Roboflow's RF-DETR transformer model for
real-time instance detection + segmentation, with a **zero-Python-thread** cook path and a
**GPU (GLSL) compositor**. It produces three outputs — a composite debug view, a transparent
overlay, and a **chromakey-green frame** — so the detected subject can be separated from the
background and composited over anything downstream.

---

## 1. Goals

| # | Goal |
|---|------|
| G1 | Run RF-DETR (detection + segmentation, COCO-pretrained) on a source video, live. |
| G2 | **Zero background Python threads touching TD objects.** Every TD object read happens on the cook thread, so the *"TouchDesigner objects cannot be referenced from separate threads"* warning is impossible. |
| G3 | **GPU compositor.** Mask/box/label painting happens in a GLSL TOP, not numpy alpha-blends in a Script TOP `onCook`. This is what keeps full-res compositing at sub-millisecond. |
| G4 | **Chromakey output (the new feature).** A frame where the detected subject(s) keep their original pixels and the rest is painted a flat, keyable chroma green — so foreground/background can be split in TD. |
| G5 | **No hang on model change.** Switching the Model dropdown never freezes TD. |
| G6 | **Self-installing.** An in-TOX Setup page installs dependencies as a real OS subprocess (TD stays responsive) and can pre-download model weights. |

### Non-goals (v1)
- Sub-10 ms latency (that needs ONNX + TensorRT — roadmap, not v1).
- Server-side batch processing (use `rfdetr` directly).
- ONNX export, ByteTrack tracking, pose variants (roadmap).

---

## 2. Environment (this machine)

| Item | Value | Consequence |
|---|---|---|
| TouchDesigner | installed, **bundles Python 3.11** (`python311.dll`) | All TOX Python runs in 3.11; deps must install into *TD's* Python, not the system 3.14. |
| GPU | **NVIDIA RTX 3090, 24 GB**, driver 596.49 / CUDA 13.2 | Target a **CUDA** torch build. Driver supports cu12x wheels. |
| Bundled in TD | CUDA 12, cuDNN 9, TensorRT 10, ONNX Runtime CUDA | CUDA runtime present; torch CUDA wheel will run. |
| `rfdetr` | **v1.11.2, `requires_python >= 3.10`** | Compatible with TD 3.11. Pulls `torch>=2.2`, `transformers 5.x`, `supervision>=0.29`, `numpy`, `pydantic>=2,<3`, `requests`. |
| Existing TOX | none | Build from scratch. |

**Device strategy:** auto-detect at load — `cuda` if available, else `cpu`. The TOX is written
device-agnostic so the same code runs on the RTX 3090 (CUDA) or any CPU fallback.

---

## 3. Design principles

1. **One inference point, many cheap views.** A single Script TOP runs RF-DETR once per frame
   and emits a single **union mask** image. Every visual output (overlay, chromakey, composite)
   is then a *pure GLSL function of `(source, mask)`*. Adding a fourth output later is a
   copy-paste of a GLSL TOP — no Python changes.

2. **Masks are the only heavy payload.** Per-frame per-instance pixel data is collapsed to a
   1-channel float union mask (0–1, full-res). Boxes, class ids, and confidences are small
   (N×4, N×1, N×1) and travel to GLSL as a tiny data texture / to a CHOP.

3. **Python does thinking, GPU does pixels.** Python: run the model, build the union mask,
   write the box data texture, publish CHOP data, manage model lifecycle. GLSL: every pixel of
   every output. No numpy alpha-blend of full-res masks on CPU.

4. **Model is cached, loaded lazily, never blocks indefinitely.** Each variant is loaded once
   and kept in a Python dict. First-ever load of an uncached variant downloads; the Setup
   pre-download step makes the dropdown a no-network operation afterwards.

---

## 4. System architecture (object graph)

```
   [user wires their source here]
            |
            v
      +-----------+     +--------------+
      |  input    |---->|  preprocess  |   (resize/format TOP -> model input res, e.g. 432)
      | (Input TOP)|     +--------------+
      +-----------+             |
                                v
                     +----------------------+
                     |       infer          |   Script TOP (onCook, COOK THREAD)
                     |  (rfdetr inference)  |   - reads preprocessed frame
                     |                      |   - runs cached model (CUDA)
                     |  emits:              |   - builds union mask (HxW float 0..1)
                     |   * union mask TOP   |   - writes box data texture
                     |   * box data (N x 4) |   - publishes CHOP + status params
                     +----------------------+
                          |            \
                          v             \
                  +-------------+        \
                  |    mask     |         v
                  | (union, 1ch)|   +------------------+
                  +-------------+   |  detections      |  (Table/Data CHOP)
                          |         |  class,conf,x1.. |
            +--------------+-------+ +------------------+
            |              |
            v              v
   +---------------+  +----------------+  +-------------------+
   |    overlay    |  |   chromakey    |  |    composite      |
   |  (GLSL TOP)   |  |   (GLSL TOP)   |  |    (GLSL TOP)     |
   | src + mask -> |  | src + mask ->  |  | src + mask + box  |
   | transparent   |  | GREEN SCREEN   |  | tint + bboxes     |
   +---------------+  +----------------+  +-------------------+
        [OUT]              [OUT]                [OUT]

   +---------------------------------------------------------------+
   |  setup (container/page)                                       |
   |   - device status        - Install Dependencies (subprocess)  |
   |   - Model dropdown       - Predownload (enabled variants)     |
   |   - variant checkboxes   - install log (Text DAT)             |
   +---------------------------------------------------------------+
```

Root TOX parameters (the user-facing knobs):

| Param | Type | Default | Purpose |
|---|---|---|---|
| `Model` | menu | `seg-medium` | Which variant (4 detection + 6 segmentation). |
| `Confidence` | float | `0.5` | NMS/detection threshold. |
| `ChromaInvert` | int 0/1 | `0` | `0` = subject on green; `1` = subject greened, bg kept. |
| `ChromaColor` | color | `(0, 1, 0)` | The green to paint. Tune to match your TD key. |
| `ClassWhitelist` | text | *(empty = all)* | Comma list of class names that count as foreground. |
| `CustomCheckpoint` | text | *(empty)* | Path to a `.pt` from `rfdetr.train(...)` to load instead. |
| `HideSource` | int 0/1 | `0` | Composite on black instead of source. |
| `Device` | menu | `auto` | `auto` / `cuda` / `cpu`. |

---

## 5. Per-frame data flow

1. `preprocess` resizes/formats the user's source to the model's input resolution
   (e.g. 432 px on the long edge) and normalizes to a tensor-friendly layout.
2. `infer` (Script TOP, **cook thread**) reads the preprocessed frame as a numpy array,
   converts to RGB uint8, and calls the cached model.
3. The model returns `boxes (N,4)`, `class_id (N,)`, `confidence (N,)`, and — for
   segmentation variants — `masks (N,H,W)` float 0–1 at input resolution.
4. `infer` computes the **union mask** `U = max(masks, axis=0)` → `(H,W)` float 0–1
   (all-zeros when no detections). For detection variants (no masks), the union mask is
   synthesized from the boxes (filled rectangles) so the same downstream GLSL works for
   both detection and segmentation.
5. `infer` writes:
   - the **union mask** as its TOP output (single channel, float 0–1),
   - a **box data texture** (a 1-D float array: per detection → x1,y1,x2,y2, classId, conf),
   - **CHOP** rows (class, confidence, x1,y1,x2,y2, area, frame),
   - **status params** (detection count, inference ms, state, error message).
6. The three GLSL TOPs each read `(source, mask[, box texture])` and emit their output.
   None of them re-run inference — they are pure pixel functions and cook independently.

---

## 6. The three outputs

All three share the same two inputs — the **source** frame and the **union mask** — and differ
only in the GLSL they apply. This is the "one inference point, many cheap views" principle in
action.

### 6.1 `overlay` — transparent cutout
The subject cut out on a transparent background. Directly usable for compositing over any
background with no keying step.

```glsl
// inputs: [0] = source (RGBA),  [1] = union mask (1ch, 0..1)
float m   = tex(1, uv).r;
vec3  rgb = tex(0, uv).rgb;
output.rgb = rgb;          // straight alpha: keep original pixels
output.a   = m;            // subject = opaque, background = transparent
```

### 6.2 `chromakey` — green screen  *(the new feature)*
The subject keeps its original pixels; everything else is painted a flat chroma green. Downstream
you key out the green (TD chroma-key / a 3-line GLSL key) to get the subject on transparent, or
feed the literal green-screen frame to any hardware/software keyer.

```glsl
// inputs: [0] = source (RGBA),  [1] = union mask (1ch, 0..1)
// params: g (color), invert (0/1), sharp (0..1 edge sharpness)
float m   = tex(1, uv).r;
if (invert > 0.5) m = 1.0 - m;                       // invert toggle
if (sharp > 0.0)  m = smoothstep(0.5 - sharp*0.5,    // optional hard edge
                                 0.5 + sharp*0.5, m);
vec3  src = tex(0, uv).rgb;
output.rgb = mix(g, src, m);   // m=1 -> subject, m=0 -> green
output.a   = 1.0;              // opaque green-screen frame
```

- **`invert = 0` (default):** `m=1` → subject (original), `m=0` → green. **Subject on green.**
- **`invert = 1`:** subject is greened, background kept → key out green to isolate the *background*.
- **Green:** a color param, defaulting to pure `(0, 255, 0)`. Because *we* paint it, it is a
  perfectly flat synthetic key that removes cleanly. Tune it to match your TD key if you prefer
  a "real" chroma green (e.g. `0, 177, 64`).
- **Edges:** the RF-DETR mask is a float (0–1), so `mix` yields a soft, anti-aliased boundary for
  free. `sharp` thresholds the mask for a harder edge when spill is a concern.
- **Which pixels are "foreground":** the union of all detected instances by default; the
  `ClassWhitelist` param (e.g. `person`) restricts the union to specific classes before the mask
  is built (Python side, section 5 step 4).

### 6.3 `composite` — debug / annotated view
The source with the mask tinted, boxes drawn, and labels rendered — the "what am I seeing" view.

```glsl
// inputs: [0] = source,  [1] = union mask,  [2] = box data texture (1D float)
float m   = tex(1, uv).r;
vec3  src = (hideSource > 0.5) ? vec3(0.0) : tex(0, uv).rgb;
vec3  out = mix(src, src + tint*0.35, m);   // subtle tint inside mask
// draw each box rectangle + label from the box data texture (loop over N)
//   (box coords are in normalized 0..1; label = class name + confidence)
output.rgb = out;
output.a   = 1.0;
```

Box rectangles and label text are the only non-trivial GLSL part. Box geometry comes from the
1-D data texture; labels are rendered either (a) by a tiny bitmap-font atlas in GLSL, or
(b) by a lightweight per-frame **Text/Composite** layer driven by the same CHOP data. Option (b)
keeps the GLSL simple and is the v1 default; (a) is a later optimization if label cost matters.

---

## 7. Inference & model management

### 7.1 `rfdetr` API (expected surface — confirm exact names at build time)

```python
from rfdetr import RFDETR

model = RFDETR.from_pretrained("rf-detr-seg-medium", device="cuda")
# model.class_names            # 80 COCO class names
# model.device                 # 'cuda' | 'cpu'

predictions = model.predict(rgb_uint8, threshold=0.5)   # list[RFDETRPredictions]
p = predictions[0]
p.boxes        # (N, 4)  [x1, y1, x2, y2] in pixels
p.class_id     # (N,)
p.confidence   # (N,)
p.masks        # (N, H, W) float 0..1 at input resolution  (seg variants only)
```

> Variant identifiers (`rf-detr-nano`, `rf-detr-seg-medium`, …) are taken from `rfdetr`'s model
> registry. A verification command lists them at build time (see Implementation Guide, step 0).

### 7.2 Model variants (10, matching the post)

| Family | Variants | Notes |
|---|---|---|
| Detection (4) | nano, small, medium, large | boxes only → union mask synthesized from boxes |
| Segmentation (6) | nano, small, medium, large, **xl, 2xl** | real masks; **xl / 2xl need a Roboflow commercial license** for commercial use |

v1 enables all 10 in the dropdown but **predownloads only the checked ones** (xl/2xl default OFF).

### 7.3 No-hang on model change (G5)
- Models are cached in a Python dict keyed by variant; loading is **once per variant**.
- On `Model` param change the TOX does **not** block the cook thread on a long load: it marks
  state = `loading`, keeps rendering the last valid mask (or black), and loads the new model.
  A cached load is sub-second, so the switch is effectively instant.
- First-ever load of an uncached variant downloads weights; the Setup **predownload** step
  (section 8) makes the dropdown a no-network operation afterwards.
- Model loading touches **no TD objects**, so it is the one piece that *could* run off-thread if
  ever needed — but v1 keeps it on the cook thread and simply non-blocking, to stay within G2.

### 7.4 Custom checkpoint (G: custom model deployment)
`CustomCheckpoint` (a file path) overrides the dropdown: the TOX loads that `.pt` (produced by
`rfdetr.train(...)`) instead of a pretrained variant. Class names come from the checkpoint's
metadata. This is the path for a model fine-tuned on the "tvhead" subject.

---

## 8. Setup / install (self-installing, G6)

A `setup` container with a Setup page. Everything runs as a **real OS subprocess** so TD stays
responsive (no long `pip` block on the UI thread).

### 8.1 Install Dependencies
From inside TD's Python, `sys.executable` **is** TD's Python 3.11 — so the subprocess targets the
correct interpreter automatically.

```
<sys.executable> -m pip install --upgrade pip
<sys.executable> -m pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121      # CUDA build (Windows)
<sys.executable> -m pip install rfdetr supervision opencv-python-headless
```
- Order matters: install the **CUDA torch wheel first** (from the PyTorch index), then `rfdetr`
  (which sees torch already satisfied and pulls `transformers 5.x`, `pydantic`, `numpy`,
  `requests`, `supervision` from PyPI).
- Fallback: if no NVIDIA device is detected, install the default (CPU) torch instead.
- Live output is streamed into a **Text DAT** ("install log") the user can watch; a status param
  flips to `installed` / `error` on completion.

### 8.2 Predownload Models
For each **checked** variant, run a subprocess:
```
<sys.executable> -c "from rfdetr import RFDETR; RFDETR.from_pretrained('<variant>', device='cuda')"
```
This triggers the HuggingFace download and caches weights in `~/.cache/huggingface`. Afterwards,
switching the Model dropdown is a cache hit (no network).

### 8.3 Device detection
At Setup (and on `Device` change), probe `torch.cuda.is_available()` and report the detected
device + free VRAM into the Setup page so the user knows what they're running.

### 8.4 `requirements.txt`
Pinned/expected dependency list shipped in the TOX, used by the installer and as documentation.

---

## 9. CHOP data output

A **Table CHOP** (`detections`) published per frame by the `infer` Script TOP (same cook thread).
Columns:

| column | meaning |
|---|---|
| `class_id` | COCO class index |
| `class` | class name (string) |
| `confidence` | 0–1 |
| `x1, y1, x2, y2` | box in pixels (input resolution) |
| `area` | `(x2-x1)*(y2-y1)` |
| `frame` | running frame counter |

This feeds TD's usual data-driven workflow — e.g. a Select CHOP filtered by `class_id` → sliders,
LFOs, particle counts, param expressions.

---

## 10. Error handling & states

A `state` param on the root: `ready` / `loading` / `error` / `no-deps`.

| Condition | Behavior |
|---|---|
| `rfdetr`/`torch` not installed | `infer.onEnable` catches `ImportError`; `state=no-deps`; a Text DAT + status param say *"run Setup → Install Dependencies"*; output is a flat placeholder. **No crash.** |
| Model not cached & not predownloaded | `state=loading`; first cook downloads; renders last-valid or black meanwhile. |
| Inference throws (e.g. OOM on a big variant) | caught; `state=error` + message; falls back to last valid mask (or empty). User can drop to a smaller variant. |
| No / malformed input TOP | `infer` outputs empty mask + a warning; downstream stays valid. |
| Custom checkpoint missing | clear error pointing at the `CustomCheckpoint` param. |

The GLSL TOPs never fail: with an empty/zero mask they simply pass through (overlay = transparent,
chromakey = full green, composite = source).

---

## 11. Key design decisions & rationale

| Decision | Choice | Why |
|---|---|---|
| Mask flow | **Union-mask intermediate** (1ch float) over per-instance multi-channel | Chromakey + overlay only need the union; a variable channel count (N per frame) is awkward for a fixed GLSL input. Keeps the graph minimal and each output trivially testable. Colored per-instance masks are a later add that doesn't touch the inference path. |
| Compositing | **GLSL TOPs**, not numpy in `onCook` | Full-res mask alpha-blend on CPU is the 30–80 ms/frame cost the original post hit. GLSL keeps it ~sub-ms and off the CPU. |
| Threads | **None** for TD access; model load non-blocking on cook thread | Eliminates the cross-thread TD warning class entirely while still not hanging on model change. |
| Torch build | **CUDA (cu121)** via the in-TOX installer | RTX 3090 + driver 596.49; roughly 2× faster than a CPU/MPS build for the same model. |
| Install | **Subprocess** pip into TD's own Python | TD stays responsive; `sys.executable` inside TD is already the right 3.11 interpreter. |
| Chromakey | **Painted by us**, flat green, float-mask soft edge | A synthetic key is perfectly keyable; the float mask gives anti-aliased boundaries without extra work. |

---

## 12. Roadmap (post-v1)
- ONNX export in the Setup page (downstream CoreML / TensorRT).
- Multi-source batching (one model, many inputs, batched inference).
- ByteTrack tracking (stable per-instance IDs across frames).
- Pose variant (new visualiser).
- Per-instance colored masks in the composite (GLSL bitmap-font labels).
