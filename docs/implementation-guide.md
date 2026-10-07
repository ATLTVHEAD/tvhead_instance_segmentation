# RF-DETR Chroma TOX — Implementation Guide

Step-by-step build for the TOX described in [`architecture.md`](./architecture.md). Target:
**TouchDesigner (Python 3.11) + RTX 3090 (CUDA)**, built from scratch.

> Conventions: TOX root op is named `rfdetr`. Python lives in Text DATs inside the TOX
> (`python/extension.pyt`, `python/callbacks.pyt`, `python/installer.pyt`, `python/shaders.glslt`),
> plus `requirements.txt`. The `.tox` is the serialized container; all Python is embedded in the DATs.

---

## 0. Step 0 — verify the `rfdetr` API on this machine

Before writing inference code, confirm the exact model identifiers and the prediction object
fields against the installed package. From **TD's** Python (so it matches the runtime):

```
"C:\Program Files\Derivative\TouchDesigner\bin\python.exe" -c "import sys; print(sys.version)"
```

After deps are installed (step 5), run:

```python
import rfdetr, torch
print("rfdetr", rfdetr.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available())
from rfdetr import RFDETR
# 1) list available pretrained names (exact strings to put in the dropdown)
#    - try RFDETR's registry / docs; at minimum confirm these load:
for name in ["rf-detr-nano", "rf-detr-seg-nano", "rf-detr-seg-medium", "rf-detr-seg-large"]:
    m = RFDETR.from_pretrained(name, device="cuda")
    print(name, "-> classes:", len(m.class_names), "device:", m.device)
# 2) confirm the prediction fields on a dummy image
import numpy as np
m = RFDETR.from_pretrained("rf-detr-seg-medium", device="cuda")
p = m.predict((np.full((432, 768, 3), 127, dtype=np.uint8)), threshold=0.5)[0]
print("boxes", p.boxes.shape, "class_id", p.class_id.shape,
      "conf", p.confidence.shape, "masks", getattr(p, "masks", None))
```

**Record the confirmed model-name strings and field names** — they go into the dropdown param and
`extension.py`. If `masks` is named differently or returned at a different resolution, adapt the
union-mask code in step 2 accordingly.

---

## 1. Prerequisites

- TouchDesigner installed (present), RTX 3090 + CUDA driver (present).
- Internet access for the one-time dependency install + weight downloads.
- Decide the model **input resolution** (default: 432 px long edge). Keep it a multiple of 32
  (transformer patch alignment) — e.g. 432×768 for 9:16, or match the source aspect.

---

## 2. Project layout

```
rfdetr.tox
└─ rfdetr (container, TOX root)
   ├─ [params] Model, Confidence, ChromaInvert, ChromaColor, ClassWhitelist,
   │           CustomCheckpoint, HideSource, Device, state
   ├─ input        (Input TOP, placeholder — user wires source)
   ├─ preprocess   (Resize TOP + Format TOP → model input res, RGB)
   ├─ infer        (Script TOP)  ← python/extension.pyt
   ├─ boxdata      (Data TOP, 1D float, fed by infer)   [optional; may be built inline]
   ├─ overlay      (GLSL TOP)   ← python/shaders.glslt (overlay block)
   ├─ chromakey    (GLSL TOP)   ← python/shaders.glslt (chromakey block)
   ├─ composite    (GLSL TOP)   ← python/shaders.glslt (composite block)
   ├─ detections   (Table CHOP) ← filled by infer (extension.pyt)
   ├─ setup        (container + page)  ← python/installer.pyt
   │   ├─ device_status    (Text DAT)
   │   ├─ install_log      (Text DAT)
   │   ├─ variant_check    (Checkbox DAT: which variants to predownload)
   │   ├─ btn_install      (Button → installer.install_deps())
   │   └─ btn_predl        (Button → installer.predownload())
   ├─ status       (Text DAT: state + last error, user-visible)
   └─ python/
       ├─ extension.pyt      (inference, union mask, box data, CHOP, model cache)
       ├─ callbacks.pyt      (onParamChanged, model reload, no-hang logic)
       ├─ installer.pyt      (subprocess pip + predownload + device detect)
       ├─ shaders.glslt      (overlay / chromakey / composite GLSL)
       └─ requirements.txt
```

---

## 3. Build order

Work in this order; each step is independently testable.

1. **Skeleton** — create the container + object graph above; wire `input → preprocess → infer → {overlay, chromakey, composite}` and `infer → detections`.
2. **Inference core** (`extension.pyt`) — get `infer` emitting a correct union mask + CHOP on a still image (no model switching yet).
3. **Shaders** (`shaders.glslt`) — wire the three GLSL outputs to `(source, mask)`; verify overlay + chromakey pixel values.
4. **CHOP** — confirm `detections` columns update per frame.
5. **Installer** (`installer.pyt`) — get Install + Predownload + device status working.
6. **Model switching + no-hang** (`callbacks.pyt`) — dropdown change reloads without freezing.
7. **Params & polish** — whitelist, invert, hide-source, custom checkpoint, status/error display.
8. **Packaging** — `requirements.txt`, README, INSTALL, save `.tox`.

---

## 4. Step 1 — skeleton (object graph)

Create the container `rfdetr` and the ops listed in section 2. Set:

- `preprocess`: Resize TOP (long edge → 432, aspect-fit) → Format TOP (RGB, no alpha).
  Keep the **aspect ratio** of the source so box/mask coords map back cleanly (store the
  source W×H as params on `infer` for de-normalizing).
- `infer`: Script TOP, output **1 channel, float32**, width/height = preprocessed size.
  Attach `python/extension.pyt` as its script.
- `overlay` / `chromakey` / `composite`: GLSL TOPs, inputs = `[preprocess (or input), infer]`
  (`composite` also takes `boxdata`). Attach `python/shaders.glslt`.

> **Script TOP output mechanism:** a Script TOP's `onCook(self)` fills the op's own output
> image. Set the output Width/Height/PixelFormat/Alpha params to match, then assign the numpy
> array to the output inside `onCook`. Confirm the exact assignment call against your TD
> version's Script TOP docs when you first run it (it differs subtly between versions).

**Resolution strategy (keep coords 1:1):** `preprocess` targets a size that matches the *source*
aspect ratio with long edge = 432 (rounded to a multiple of 32). Because preprocessed aspect ==
source aspect, a box or mask normalized to 0–1 in preprocessed space maps 1:1 to the source, and
the mask (sampled by UV in GLSL) auto-resizes to the output resolution. *If the model requires a
fixed input size*, use that fixed size instead and apply the letterbox scale/offset when mapping
boxes back to source space (confirm at step 0 whether RF-DETR accepts variable resolution).

---

## 5. Step 2 — inference core (`python/extension.pyt`)

Attached to `infer`. Runs RF-DETR on the cook thread; emits the union mask (output image), box
data texture, CHOP rows, and status params. **No background threads.** Two confirm-points are
marked `# CONFIRM`: the Script TOP output-assignment call, and the `rfdetr` field names.

```python
# python/extension.pyt  — attached to the `infer` Script TOP
import numpy as np
import time

_MODELS = {}      # variant key -> rfdetr model  (module-level, persists across cooks)
MAX_DETS = 64     # fixed capacity for the box data texture

def _root():
    return op("rfdetr")

def _device():
    d = _root().par.device.eval()
    if d in ("cuda", "cpu"):
        return d
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

def _get_model():
    root = _root()
    if root.par.customcheckpoint.file:
        key = ("custom", root.par.customcheckpoint.file)
    else:
        key = root.par.model.eval()
    if key not in _MODELS:
        from rfdetr import RFDETR
        if key[0] == "custom":
            _MODELS[key] = RFDETR.from_pretrained(key[1], device=_device())
        else:
            _MODELS[key] = RFDETR.from_pretrained(key, device=_device())
    return _MODELS[key]

def _whitelist(root):
    t = root.par.classwhitelist.text.strip()
    return ({s.strip() for s in t.split(",") if s.strip()} if t else None)

# ---- callbacks -------------------------------------------------------------
def onEnable(self):
    root = _root()
    try:
        import rfdetr, torch  # noqa
        root.par.state.value = "ready"
    except Exception as e:
        root.par.state.value = "no-deps"
        _status(root, "no-deps", f"import failed: {e}")

def onCook(self):
    root = _root()
    if root.par.state.value == "no-deps":
        _emit_empty(self); return
    try:
        model = _get_model()
    except Exception as e:                       # load/first-download error
        root.par.state.value = "error"
        _status(root, "error", str(e)); _emit_empty(self); return

    h, w = self.input.top.height, self.input.top.width
    frame = np.asarray(self.input.top.pixel)      # (h, w, c) float32 0..1
    if frame.shape[2] >= 3:
        rgb_u8 = (np.clip(frame[..., :3], 0, 1) * 255).astype(np.uint8)
    else:
        g = (np.clip(frame[..., 0:1], 0, 1) * 255).astype(np.uint8)
        rgb_u8 = np.repeat(g, 3, axis=2)

    t0 = time.perf_counter()
    p = model.predict(rgb_u8, threshold=float(root.par.confidence.value))[0]
    dt = (time.perf_counter() - t0) * 1000.0

    boxes = np.asarray(p.boxes, dtype=np.float32)      # (N,4) x1,y1,x2,y2 px   # CONFIRM field
    cls   = np.asarray(p.class_id, dtype=np.int64)     # (N,)                    # CONFIRM field
    conf  = np.asarray(p.confidence, dtype=np.float32) # (N,)                    # CONFIRM field
    masks = getattr(p, "masks", None)                  # (N,H,W) float 0..1      # CONFIRM field
    if masks is not None:
        masks = np.asarray(masks, dtype=np.float32)

    # class whitelist
    wl = _whitelist(root)
    if wl:
        names = model.class_names
        keep = np.array([names[int(c)] in wl for c in cls])
        boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
        if masks is not None:
            masks = masks[keep]

    n = len(cls)

    # union mask (H, W) float 0..1
    if masks is not None and n:
        union = np.max(masks, axis=0)
    elif n:                                       # detection variant -> boxes
        union = np.zeros((h, w), dtype=np.float32)
        for x1, y1, x2, y2 in boxes:
            union[max(0,int(y1)):min(h,int(y2)), max(0,int(x1)):min(w,int(x2))] = 1.0
    else:
        union = np.zeros((h, w), dtype=np.float32)

    _emit_image(self, union)                       # 1) union mask output
    _emit_boxdata(root, boxes, cls, conf, w, h)    # 2) box data texture
    _emit_chop(root, boxes, cls, conf, model)      # 3) CHOP

    root.par.state.value = "ready"
    root.par.ncount.value = n
    root.par.inferms.value = dt

# ---- emitters --------------------------------------------------------------
def _emit_image(self, arr2d):
    out = arr2d[:, :, None].astype(np.float32)     # (H, W, 1)
    # CONFIRM: Script TOP output assignment for your TD version, e.g.:
    self.par.output.pixel = out                    # or the version-specific call

def _emit_empty(self):
    _emit_image(self, np.zeros((self.input.top.height, self.input.top.width), dtype=np.float32))

def _emit_boxdata(root, boxes, cls, conf, w, h):
    flat = np.zeros(MAX_DETS * 6, dtype=np.float32)
    for i in range(min(len(cls), MAX_DETS)):
        x1, y1, x2, y2 = boxes[i]
        flat[i*6+0] = x1 / w;  flat[i*6+1] = y1 / h
        flat[i*6+2] = x2 / w;  flat[i*6+3] = y2 / h
        flat[i*6+4] = float(cls[i]); flat[i*6+5] = float(conf[i])
    root.boxdata.pixel = flat                       # Data TOP, 1D float (384 wide)
    root.boxdata.par.width.value = MAX_DETS * 6
    root.par.dets.value = min(len(cls), MAX_DETS)   # uniform: active count

def _emit_chop(root, boxes, cls, conf, model):
    names = model.class_names
    rows = [["class_id", "class", "confidence", "x1", "y1", "x2", "y2", "area", "frame"]]
    fr = int(op.getFrameNumber())
    for i in range(len(cls)):
        x1, y1, x2, y2 = boxes[i]
        rows.append([int(cls[i]), names[int(cls[i])], round(float(conf[i]), 4),
                     round(float(x1),1), round(float(y1),1), round(float(x2),1), round(float(y2),1),
                     round(max(0.0, x2-x1) * max(0.0, y2-y1),1), fr])
    root.detections.tableData = rows

def _status(root, state, msg):
    root.status.text = f"[{state}] {msg}"
```

**Test (step 2 done):** wire a still image to `input`, cook once, and confirm:
- `infer` outputs a 1-ch float mask (subject region ≈ 1, rest ≈ 0).
- `detections` CHOP has a row with a sensible class/conf/box.
- `state` = `ready`, `inferms` reasonable (~20 ms for seg-medium on the 3090).

---

## 6. Step 3 — shaders (`python/shaders.glslt`)

One DAT, three GLSL blocks (or three GLSL TOPs each pointing at one). All take
`[0] = source`, `[1] = union mask`; `composite` also takes `[2] = boxdata`.

```glsl
// ---------------- overlay ----------------
float m = tex(1, uv).r;
output.rgb = tex(0, uv).rgb;
output.a   = m;

// ---------------- chromakey ----------------
float m = tex(1, uv).r;
if (invert > 0.5) m = 1.0 - m;
if (sharp > 0.0)  m = smoothstep(0.5 - sharp*0.5, 0.5 + sharp*0.5, m);
vec3  src = tex(0, uv).rgb;
output.rgb = mix(chromaColor, src, m);
output.a   = 1.0;

// ---------------- composite ----------------
float m = tex(1, uv).r;
vec3  base = (hideSource > 0.5) ? vec3(0.0) : tex(0, uv).rgb;
vec3  out  = mix(base, base + vec3(0.2,0.6,1.0)*0.30, m);
int dets = int(detsCount);
for (int i = 0; i < MAXD; i++) {
    if (i >= dets) break;
    float x1 = tex(2, vec2((i*6+0)/(float(MAXD*6), 0.0)).r);
    float y1 = tex(2, vec2((i*6+1)/(float(MAXD*6), 0.0)).r);
    float x2 = tex(2, vec2((i*6+2)/(float(MAXD*6), 0.0)).r);
    float y2 = tex(2, vec2((i*6+3)/(float(MAXD*6), 0.0)).r);
    // draw a 2px rectangle border in (x1,y1)-(x2,y2) into `out` (color by class)
}
output.rgb = out;
output.a   = 1.0;
```
- `MAXD` = `MAX_DETS` (64); `detsCount`, `invert`, `sharp`, `hideSource`, `chromaColor` are GLSL
  params wired to the root params.
- v1 draws box borders in GLSL; **labels** (class name + conf) default to a lightweight per-frame
  **Text/Composite** layer driven by the `detections` CHOP (keeps GLSL simple). A bitmap-font
  GLSL label pass is a later optimization.

**Test (step 3 done):** with a subject in frame —
- `overlay` shows the subject on transparency (checkerboard).
- `chromakey` shows the subject on flat green; toggle `ChromaInvert` → subject greened, bg kept.
- Pixel check: a background pixel in `chromakey` == `ChromaColor` exactly.

---

## 7. Step 4 — CHOP verification

Confirm `detections` columns match section 9 of the architecture doc and update every frame.
Smoke-test the data-driven path: `detections` → Select CHOP (filter `class = person`) → a Value
CHOP / param expression, and watch it track the detection.

---

## 8. Steps 5–6 — installer + predownload (`python/installer.pyt`)

**Principle:** the heavy work (pip / HF download) runs in a **child OS process**. TD's Python
never busy-waits on it — a **Timer** polls the child a few lines per frame, so the UI stays
responsive and **no background Python thread touches TD objects** (G2 holds).

```python
# python/installer.pyt
import sys, subprocess

_state = {"proc": None, "phase": 0, "predl_idx": -1, "names": []}

def _py():
    return sys.executable          # TD's own Python 3.11

def _log(text):
    op("rfdetr").install_log.appendText(text)

def _cuda_index():
    # CONFIRM: pick the index for your torch version. On Windows the plain
    # PyPI torch wheel is normally CUDA-enabled, so this can be "" (none).
    return "--index-url https://download.pytorch.org/whl/cu121"

def start_install():
    idx = _cuda_index()
    cmds = [
        [_py(), "-m", "pip", "install", "--upgrade", "pip"],
        [_py(), "-m", "pip", "install", "torch", "torchvision"] + (idx.split() if idx else []),
        [_py(), "-m", "pip", "install", "rfdetr", "supervision", "opencv-python-headless"],
    ]
    _state.update(proc=_run(cmds[0]), phase=1, cmds=cmds)   # phase = next cmd to launch
    _log("install started\n")

def start_predownload():
    names = [c for c, v in op("rfdetr.variant_check").items() if v.value]  # checked variants
    _state.update(names=names, predl_idx=0, proc=None)
    _next_predl()

def _next_predl():
    i = _state["predl_idx"]
    if i >= len(_state["names"]):
        _log("\npredownload complete\n"); _state["proc"] = None; return
    name = _state["names"][i]
    code = f"from rfdetr import RFDETR; RFDETR.from_pretrained('{name}', device='cuda')"
    _state["proc"] = subprocess.Popen([_py(), "-c", code],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True)
    _log(f"predownload [{i+1}/{len(_state['names'])}] {name}\n")

def _run(cmd):
    _log("$ " + " ".join(cmd) + "\n")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

# ---- called by a Timer (e.g. 30 Hz) each frame -----------------------------
def tick():
    proc = _state.get("proc")
    if proc is None:
        return
    # drain available output (non-blocking line reads)
    import select  # Windows: use proc.stdout.readline() with a guard instead
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        _log(line.rstrip() + "\n")
    if proc.poll() is not None:                 # finished
        if _state.get("cmds"):                  # install: advance to next phase
            _state["phase"] += 1
            cmds = _state["cmds"]
            if _state["phase"] < len(cmds):
                _state["proc"] = _run(cmds[_state["phase"]])
            else:
                _state["proc"] = None; _state["cmds"] = None
                op("rfdetr").par.state.value = "ready"; _log("\ninstall complete\n")
                onEnable(op("rfdetr.infer"))    # re-probe deps
        else:                                   # predownload: next variant
            _state["predl_idx"] += 1; _next_predl()
```

> **Windows note:** `select.select()` is unreliable on pipes on Windows. Read with
> `proc.stdout.readline()` and rely on the per-frame cadence (a Timer) to avoid blocking; or set
> the pipe to non-blocking. Keep each `tick()` short (read a few lines, return).

Wire: `btn_install` pulse → `start_install()`; `btn_predl` pulse → `start_predownload()`; a
**Timer DAT** (or the op's `onTimer`) at ~30 Hz → `tick()`. `device_status` is filled by:

```python
def detect_device():
    import torch
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        return f"cuda — {p.name}, {p.total_memory/1e9:.1f} GB"
    return "cpu"
```

**Test:** run Install on the target machine, watch `install_log`, then `detect_device()` reports
the 3090; run Predownload on 2–3 checked variants, then the dropdown switches with no network.

---

## 9. Step 7 — model switching without a hang (`python/callbacks.pyt`)

Attached to the root (or `infer`). Handles `Model` / `Device` / `CustomCheckpoint` changes.

```python
# python/callbacks.pyt
def onParamChanged(self, paramInfo):
    if paramInfo.name in ("model", "device", "customcheckpoint"):
        # Drop the cached model so the next cook reloads. Mark loading; keep last mask.
        import extension
        key = (paramInfo.name, paramInfo.op.par.model.eval())
        extension._MODELS.clear()          # simple: reload on any change (cached per variant)
        self.par.state.value = "loading"
        _status(self, "loading", "loading model…")
```

- Because models are **cached per variant** (`extension._MODELS`), returning to a previously-used
  variant is a dict hit (instant). A brand-new variant loads on the next cook; the GLSL outputs
  keep showing the last valid mask (or black) during that brief window — no freeze.
- For the *first-ever* load of an uncached variant, the cook will block on the HF download. The
  Predownload step is what makes this a non-event. (Optional hardening: move the single
  `from_pretrained` call onto a non-TD background thread *only for the initial download*, since it
  touches no TD objects — keep the rest on the cook thread. v1 keeps it simple: block + show state.)

---

## 10. Step 8 — packaging

- **`python/requirements.txt`** — the expected pins (documentation + installer reference):
  ```
  torch>=2.2
  torchvision>=0.17
  rfdetr>=1.11
  supervision>=0.29
  transformers>=5.1,<6
  pydantic>=2,<3
  numpy
  requests
  opencv-python-headless
  ```
- **`README.md`** — what it is, the 3 outputs, the Model dropdown, the chromakey invert/color,
  the CHOP, custom checkpoint, and a 10-second "wire it up" (source → `input`; use `chromakey`).
- **`INSTALL.md`** — per-platform notes (Windows/CUDA primary here; macOS/MPS; CPU fallback),
  the Setup → Install Dependencies + Predownload flow, and troubleshooting (point to section 12).
- **Save the `.tox`** from the `rfdetr` container. Confirm it is self-contained (all Python in the
  embedded DATs) and small (the original post was ~12 KB).

---

## 11. Testing plan

Split into two layers because TouchDesigner's GUI can't be driven headlessly from here.

### 11.1 Headless core test (runnable now, exact runtime match)
Validate the **inference → union mask → chromakey math** using **TD's own Python** (so it is the
identical runtime the TOX will use). After the Setup install, run:

```
"C:\Program Files\Derivative\TouchDesigner\bin\python.exe" tests\test_core.py
```

```python
# tests/test_core.py — validates the CORE logic without the TD GUI
import numpy as np
from PIL import Image
from rfdetr import RFDETR

def build_union(masks, boxes, h, w):
    if masks is not None and len(masks):
        return np.max(np.asarray(masks, np.float32), axis=0)
    u = np.zeros((h, w), np.float32)
    for x1, y1, x2, y2 in boxes:
        u[int(y1):int(y2), int(x1):int(x2)] = 1.0
    return u

def chromakey(src, m, green=(0.0, 1.0, 0.0), invert=False):   # numpy mirror of the GLSL
    m = (1.0 - m) if invert else m
    g = np.array(green, np.float32)
    return g * (1 - m[..., None]) + src * m[..., None]

def main():
    model = RFDETR.from_pretrained("rf-detr-seg-medium", device="cuda")
    img = np.asarray(Image.open("tests/sample.jpg").convert("RGB").resize((768, 432)))
    p = model.predict(img, threshold=0.5)[0]
    boxes = np.asarray(p.boxes); cls = np.asarray(p.class_id)
    masks = getattr(p, "masks", None)
    assert len(cls) >= 1, "expected at least one detection"
    u = build_union(masks, boxes, 432, 768)
    assert 0.0 < u.mean() < 1.0, f"union mask looks wrong (mean={u.mean():.3f})"
    out = chromakey(img / 255.0, u)
    assert np.allclose(out[0, 0], [0, 1, 0], atol=1e-3), f"background not green: {out[0,0]}"
    out_inv = chromakey(img / 255.0, u, invert=True)
    assert np.allclose(out_inv[0, 0], img[0,0] / 255.0, atol=1e-2), "invert: bg should keep source"
    print("OK  n=%d  union_mean=%.3f  top_class=%s"
          % (len(cls), u.mean(), model.class_names[int(cls[0])]))

if __name__ == "__main__":
    main()
```

`tests/sample.jpg` = a COCO-style still with a clear person/subject. **Passes** = the core logic,
mask extraction, and chromakey math (both directions) are correct.

### 11.2 In-TD visual test (you run this)
Open TD, load the `.tox`, wire a source into `input`, run Setup → Install + Predownload, then
check the three outputs visually (overlay cutout, chromakey green screen, composite boxes) and
exercise the dropdown, invert toggle, and CHOP.

---

## 12. Verification checklist

| # | Check | How |
|---|-------|-----|
| 1 | Deps installed into **TD's** Python, not system | `install_log` clean; `test_core.py` imports succeed with TD's `python.exe` |
| 2 | CUDA active | `detect_device()` reports the 3090 + GB; `torch.cuda.is_available()` True |
| 3 | Model loads | dropdown → state `ready`, no error |
| 4 | Union mask correct | `infer` output: subject ≈1, rest ≈0; `test_core.py` mean in (0,1) |
| 5 | Overlay correct | subject on transparency (checkerboard) |
| 6 | **Chromakey correct** | subject on flat green; a bg pixel == `ChromaColor` exactly (pixel probe) |
| 7 | Invert toggle | `ChromaInvert=1` → subject greened, bg keeps source |
| 8 | Class whitelist | `person` only → union covers only people |
| 9 | CHOP | `detections` columns present + update per frame |
| 10 | No-hang on switch | change Model rapidly — TD stays responsive, state shows `loading` briefly |
| 11 | Predownload | after Predownload, switching variants needs no network |
| 12 | No cross-thread warning | TD **Console** stays clean (no "cannot be referenced from separate threads") |
| 13 | Performance | seg-medium @ 720p ≈ 30 fps display (matches the post's ballpark on a 3090) |
| 14 | Self-contained `.tox` | Python embedded in DATs; small file; opens clean in a fresh project |

---

## 13. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `state=no-deps`, import error | Deps not in TD's Python (installed to system 3.14 instead) | Run Setup → Install (uses TD's `sys.executable`); re-check `install_log` |
| `torch.cuda.is_available()` False | CPU torch wheel installed | Reinstall torch with the CUDA index for your torch version (`_cuda_index()`) |
| `from_pretrained` name error | Variant string wrong | Confirm the exact names at step 0; update the dropdown param |
| First cook of a variant stalls | Weights not cached | Run Setup → Predownload; or wait once (one-time download) |
| OOM / very slow | Variant too big (e.g. seg-large/xl) | Use seg-nano/small/medium; 3090 has 24 GB so most fit |
| Masks at wrong resolution/coords | `masks` field differs from assumed | Adjust `build_union` / box mapping to the confirmed field (step 0) |
| Boxes offset vs. subject | Aspect mismatch (fixed model input) | Use the aspect-matched input, or apply the letterbox scale/offset |
| Install freezes the UI | `tick()` blocking on the pipe | Keep `tick()` short (few lines/frame); Windows pipe non-blocking guard |
| Cross-thread TD warning | A TD object read outside the cook thread | Ensure only `infer.onCook` reads TD objects; installer only appends to the log DAT on the timer (cook) thread |
| Green spill on subject edges | Subject contains green / soft edge | Tune `ChromaColor` away from subject greens; raise `sharp` for a harder edge |
| Dropdown "hangs" on first switch | Single uncached download on the cook thread | Predownload first (recommended); optional: off-thread the one-time download only |
