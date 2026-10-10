# Self-Contained RF-DETR TOX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every Python file in `python/` resolve operators relative to the owning `rfdetr` component (no absolute paths anywhere), restore the headless test suite for the pure core, and ship `README.md` plus a new `STATUS.html`.

**Architecture:** `python/*.pyt` are the canonical sources for the DATs inside `rfdetr.tox`; `python/core.py` is the pure inference core (numpy-only top-level imports; torch/rfdetr imported lazily) which the TOX loads by exec'ing the text of its `python/core` DAT. Each `.pyt` file is standalone (no cross-DAT imports) and resolves its owner through exactly one named helper. Dependency management is tdPyEnvManager (`TDPyEnvManagerContext.yaml` → `.venv`, Python 3.11), replacing the old `python/installer.pyt` subprocess flow.

**Tech Stack:** TouchDesigner (bundles Python 3.11), numpy, torch 2.11.0+cu128 / rfdetr 1.11.2 (lazy), tdPyEnvManager 1.4.5, plain-Python test runner (no pytest).

**Spec:** `docs/new_implementation`

## Global Constraints

- No absolute operator paths in any file under `python/` — no `op("/...")`, no `root`, no `/atltvhead`. Every code task ends with a grep gate: `rg "op\(\"/|op\('/|/atltvhead|ROOT_PATH" python/` → zero matches.
- Exactly one named owner helper per `.pyt` file; never scatter `me.parent().parent()` calls (spec §2).
- `core.py`: numpy-only top-level imports; `torch`/`rfdetr` imported lazily inside functions.
- Parameter names are frozen: `Model, Device, Confidence, Chromasharp, Chromainvert, Chromacolor, Classwhitelist, Customcheckpoint, Hidesource, State, Ncount, Inferms, Dets`.
- `MAX_DETS = 64`, `BOXDATA_W = 384`; detections DAT header row is exactly `class_id, class, confidence, x1, y1, x2, y2, area, frame`.
- Tests run headless: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\<file>.py` (any Python 3.11 + numpy). Plain `assert`, a `main()` that prints `<N>/<N> pass` and exits non-zero on first failure.
- Test frames are synthesized in code — no binary test assets (`tests/sample.jpg` stays gitignored and unused by the suite).
- `python/*.pyt` stay executable standalone by TD: only TD builtins (`me`, `op`, `app`, `par`, `top`) plus stdlib + numpy at top level.

## Review Focus

1. **Renamed TOX instance** — user renames the component to e.g. `rfdetr2`: scripts must still work (resolution is structural, never name-based). Pinned in Tasks 2, 3.
2. **Deleted internal operator** — user removes the `status`/`boxdata`/`detections` DAT: scripts degrade gracefully (empty output, no secondary exception). Pinned in Task 2.
3. **Dependencies not installed** — first use without torch: status shows `[no-deps]` + hint, empty mask emitted, no traceback. Pinned in Task 2.
4. **Malformed infer input** — no input, 1-channel, zero-dimension frames: clean error status + empty output. Pinned in Tasks 1 (core validation) and 2 (cook path).
5. **Missing/invalid checkpoint** — `Customcheckpoint` points at a file that fails to load: status names the error, empty frame emitted, no hang. Pinned in Task 1.

## File Structure

| File | Responsibility |
|---|---|
| `python/core.py` | Pure inference core: variants, dependency check, model cache + load, `predict_frame`, `build_union`, `chromakey`. No TD imports. |
| `python/extension.pyt` | Glue for the `infer` Script TOP (owner = `me.parent()`): loads core, runs inference, emits mask/boxdata/detections, writes status params. |
| `python/callbacks.pyt` | Container Callbacks DAT (owner = `par.owner`): invalidates the model cache on model/device/checkpoint change. |
| `python/add_params.pyt` | One-shot, re-runnable parameter installer DAT (owner = `me.parent().parent()`). |
| `tests/_stubs.py` | Shared TD fakes (fake ops/params/pages, import blocker, `.pyt` loader). |
| `tests/test_core.py` … `tests/test_callbacks.py` | Headless suites, one per source file. |
| `README.md` | Repo README: quick start, parameters, layout, dev workflow. |
| `STATUS.html` | New status page: current state, data flow, TOX sync steps, in-TD checklist, troubleshooting. |

---

### Task 1: Track `core.py`; restore its headless test suite

**Files:**
- Modify: `.gitignore` (delete the `python/core.py` line; keep the `tests/sample.jpg` line)
- Create: `tests/test_core.py`
- `python/core.py` is NOT modified in this task.

**Interfaces:**
- Consumes: `python/core.py` public API — `VARIANTS: dict`, `SEG_VARIANTS: frozenset`, `check_dependencies() -> dict`, `load_model(variant, device=None, checkpoint=None)`, `predict_frame(model, frame_u8, threshold=0.5) -> dict` with keys `boxes (N,4) f32 / class_ids (N,) i32 / class_names (N,) object / confidences (N,) f32 / masks (N,H,W) bool|None / union (H,W) f32`, `build_union(masks=None, boxes=None, h=0, w=0) -> (H,W) f32`, `chromakey(src, mask, green=(0.0,1.0,0.0), invert=False) -> (H,W,3) f32`, `clear_model_cache()`, `cuda_device_info() -> dict|None`.
- Produces: the test-file convention (plain asserts, `main()` printing `<N>/<N> pass`, non-zero exit on failure) used by Tasks 2–4.

- [ ] **Step 1: Un-ignore the core source**

In `.gitignore`, delete the line `python/core.py`.

- [ ] **Step 2: Commit the in-flight refactor baseline**

```bash
git add -A
git commit -m "refactor: self-contained TOX baseline (tdPyEnvManager, core.py, rebuilt .tox)"
```

Expected: commit created; `git status` clean (except ignored entries). This commit is BASE for Task 1's review range.

- [ ] **Step 3: Write the test suite `tests/test_core.py`**

Tests (exact names; stub `torch`/`rfdetr` via `sys.modules` in `try/finally`):

- `test_build_union_masks_precedence` — two overlapping 4x4 bool masks → union == elementwise OR; dtype float32; values only in {0.0, 1.0}.
- `test_build_union_boxes_only` — boxes `[[1,1,3,3],[0,0,2,2]]` on 4x4 → exact covered pixels; out-of-range box `[-1,-1,5,5]` clips to the full 4x4; non-finite box `[[float('nan')]*4]` is skipped.
- `test_build_union_degenerate_dims` — `h=0` or `w=0` → all-zeros array of shape `(max(h,0), max(w,0))`.
- `test_build_union_masks_crop` — 5x5 masks on a 4x4 output crop to 4x4; 2x2 masks pad the rest.
- `test_chromakey_keeps_subject` — src all red (1,0,0), mask with one 1 → output equals src where mask=1, equals green where mask=0.
- `test_chromakey_invert` — `invert=True` → subject becomes green, background kept.
- `test_chromakey_validation` — shape mismatch, 2-channel src, 3-D mask each raise `ValueError`.
- `test_load_model_unknown_variant` — `load_model("nope")` raises `ValueError` containing `Unknown RF-DETR variant`.
- `test_load_model_variant_cache` — stubbed `rfdetr` exposing `RFDETRNano`; two `load_model("rfdetr-nano")` calls return the same object; after `clear_model_cache()` a new object.
- `test_load_model_cuda_unavailable` — stub torch with `cuda.is_available() -> False`; `load_model("rfdetr-nano", device="cuda")` raises `RuntimeError`.
- `test_load_model_bad_device_string` — `device="gpu"` raises `ValueError`.
- `test_predict_frame_none_result` — fake model whose `.predict(...)` returns `None` → empty `(0,4)`/`(0,)` arrays and zero `(H,W)` union.
- `test_predict_frame_with_detections` — fake det: `xyxy=[[1,1,3,3]]`, `class_id=[0]`, `confidence=[0.9]`, `data={"class_name":["person"]}`, `mask=None` → boxes/confidences/names correct; union has the box region set.
- `test_predict_frame_mask_crop` — det `mask` is 5x5 on a 4x4 frame → returned `masks` shape `(1,4,4)`, `union` shape `(4,4)`.
- `test_predict_frame_bad_frame` — float32 frame raises `ValueError`; 2-D frame raises `ValueError`.
- `test_check_dependencies_shape` — result dict contains exactly the keys `torch, torchvision, rfdetr, supervision, opencv, cuda, torch_version, torchvision_version, rfdetr_version, supervision_version, opencv_version, error`.
- `test_clear_model_cache_evicts` — after a load, `clear_model_cache()` leaves the internal cache empty.

Fake model: plain object with `.predict(frame_u8, threshold=..., include_source_image=False)`. Fake det: object with attributes `xyxy`, `class_id`, `confidence`, `data` (dict), `mask` (None or array).

- [ ] **Step 4: Run the suite**

Run: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\test_core.py`
Expected: `17/17 pass`. Ruling recorded at setup: `core.py` pre-dates this plan, so this is a pinning suite (no RED phase); Tasks 2–4 restore RED→GREEN.

- [ ] **Step 5: Commit**

```bash
git add tests/test_core.py
git commit -m "test: headless suite for pure inference core"
```

---

### Task 2: `extension.pyt` — component-relative owner resolution

**Files:**
- Modify: `python/extension.pyt` (remove `ROOT_PATH`; rewrite `_root()`; `core_module()` resolves the core DAT via `root.op("python/core")`)
- Create: `tests/_stubs.py`
- Create: `tests/test_extension.py`

**Interfaces:**
- Consumes: `tests/_stubs.py` fakes; `python/core.py` (exec'd from the fake `python/core` DAT's `.text`).
- Produces: `python/extension.pyt` module exposing `_root() -> comp|None` (returns `me.parent()`), `core_module() -> module` (reads `root.op("python/core")`), `onCook(scriptOp)`; module-level `MAX_DETS = 64`, `BOXDATA_W = 384`; no `ROOT_PATH` symbol anywhere in the file.

`tests/_stubs.py` must provide (used by Tasks 2–4):
- `FakePar(name, value=None)` — `.name`, `.val`/`.value` settable, `.eval()`.
- `FakeComp(name, children=None)` — `.name`, `.op(rel)` walks `children` by `a/b` path (returns None when missing), `.par` bag (AttributeError on missing name; supports `par[name]`), `.store(k, v)` / `.fetch(k, default)`.
- `FakeScriptTop(comp)` — a FakeComp whose `.inputs` list holds a fake pixel op (`.height`, `.width`, `.numpyArray()`) and records `copyNumpyArray(arr)`.
- `BlockImport(names)` — context manager adding a meta-path finder that raises `ImportError` for the given module names (and removes them from `sys.modules`).
- `load_pyt(path, me=None, extra=None)` — reads the file, `exec`s it in a fresh module namespace with TD builtins `me`, `op` (global `op` returns None), `app` (fake with `.frame = 0`) injected; returns the module.

- [ ] **Step 1: Write the failing tests `tests/test_extension.py`**

- `test_root_is_parent` — load with a FakeScriptTop whose parent is a FakeComp named `rfdetr2` (deliberately NOT `rfdetr`) → `_root() is that comp`.
- `test_root_none_without_parent` — `me.parent()` returns None → `_root() is None`.
- `test_source_has_no_absolute_paths` — file text contains neither `op("/` nor `ROOT_PATH` nor `/atltvhead`.
- `test_core_module_resolves_relative` — parent has child `python` → child `core` DAT whose `.text` is the real `python/core.py` content → `core_module().build_union` works; second call returns the cached module.
- `test_oncook_no_deps` — `BlockImport(["torch"])`; 4x4 input → `onCook` returns without raising; `par.State == "no-deps"`; status DAT text starts with `[no-deps]`; recorded mask is all-zero `(4,4,4)`; `Dets == 0`, `Ncount == 0`.
- `test_oncook_missing_input` — script top with no inputs → status starts with `[error] infer Script TOP has no input.`; no crash.
- `test_oncook_full_path_no_detections` — stub torch + rfdetr (fake model predict → None); 4x4 input → `State == "ready"`, `Ncount == 0`, boxdata received a 384-length float array of zeros, detections table rows == header only, status starts with `[ready]`.
- `test_oncook_whitelist_filters` — fake det returns `["person","car"]`; `Classwhitelist = "person"` → `Ncount == 1`; boxdata slot 0 filled, slot 6 (second det) zeroed; detections has exactly one data row named `person`.
- `test_oncook_missing_status_dat` — parent without a `status` child → `onCook` completes and still writes the mask.

- [ ] **Step 2: Run tests — verify they fail**

Run: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\test_extension.py`
Expected: FAIL — `test_root_is_parent` and `test_source_has_no_absolute_paths` fail (current `_root()` calls global `op(ROOT_PATH)` → None; `ROOT_PATH` present in source).

- [ ] **Step 3: Implement in `python/extension.pyt`**

Replace the `ROOT_PATH` constant and `_root()`:

```python
def _root():
    """Return the RF-DETR component that owns this Script TOP."""
    try:
        return me.parent()
    except Exception:
        return None
```

In `core_module()`: `core_dat = root.op("python/core")` (drop the `ROOT_PATH + "/python/core"` string). Update the file header comment (the component is addressed component-relatively, not by absolute path). No other behavioral changes.

- [ ] **Step 4: Run tests — verify they pass**

Run: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\test_extension.py`
Expected: `9/9 pass`.

- [ ] **Step 5: Grep gate**

Run: `rg "op\(\"/|op\('/|/atltvhead|ROOT_PATH" python/`
Expected: no matches in `extension.pyt` (other files handled by their own tasks).

- [ ] **Step 6: Commit**

```bash
git add python/extension.pyt tests/_stubs.py tests/test_extension.py
git commit -m "refactor: resolve rfdetr component from me.parent() in infer glue"
```

---

### Task 3: `add_params.pyt` — component-relative owner

**Files:**
- Modify: `python/add_params.pyt` (line 5: `root = op("/atltvhead_greenscreen/rfdetr")` → resolve from the DAT's own hierarchy; error message must not name an absolute path)
- Create: `tests/test_add_params.py`

**Interfaces:**
- Consumes: `tests/_stubs.py` (`load_pyt`, `FakeComp`, a `FakeDat` whose `.parent()` is the `python` sub-comp whose `.parent()` is the container).
- Produces: `python/add_params.pyt` resolving the container via `me.parent().parent()`; idempotent re-run (file header: "Safe to re-run"); `RuntimeError` when the container cannot be resolved.

- [ ] **Step 1: Write the failing tests `tests/test_add_params.py`**

- `test_resolves_owner_relatively` — fake DAT under `python` under a comp named `my_rfdetr`; exec → no exception; container has a `Custom Parameters` page containing all 13 frozen parameters.
- `test_rerun_is_idempotent` — exec twice → same page, parameter count unchanged (no duplicates).
- `test_error_when_owner_missing` — `me.parent().parent()` is None → `RuntimeError`; message contains no `/atltvhead` and no `op("/`.

- [ ] **Step 2: Run tests — verify they fail**

Run: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\test_add_params.py`
Expected: FAIL — current script raises `RuntimeError("rfdetr container not found at /atltvhead_greenscreen/rfdetr")` because stub `op()` returns None.

- [ ] **Step 3: Implement in `python/add_params.pyt`**

Replace lines 5–11 with a named helper:

```python
def _container():
    """The rfdetr container that owns this DAT (rfdetr/python/add_params)."""
    try:
        return me.parent().parent()
    except Exception:
        return None

root = _container()
if root is None:
    raise RuntimeError("rfdetr container not found above this DAT (expected rfdetr/python/add_params)")
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\test_add_params.py`
Expected: `3/3 pass`.

- [ ] **Step 5: Grep gate**

Run: `rg "op\(\"/|op\('/|/atltvhead" python/add_params.pyt`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add python/add_params.pyt tests/test_add_params.py
git commit -m "refactor: add_params resolves container from DAT hierarchy"
```

---

### Task 4: `callbacks.pyt` — public cache API + status guard

**Files:**
- Modify: `python/callbacks.pyt`
- Create: `tests/test_callbacks.py`

**Interfaces:**
- Consumes: `python/core.py` `clear_model_cache()`; `tests/_stubs.py`.
- Produces: `python/callbacks.pyt` with `onValueChange(par, prev)` that: ignores parameters outside `{"model","device","customcheckpoint"}` (case-insensitive); calls `core.clear_model_cache()` when `root.fetch("rfdetr_core", None)` is not None; sets `State = "loading"`; writes `[loading] reloading model (<par> changed)` to `root.op("status")` only when that operator exists.

- [ ] **Step 1: Write the failing tests `tests/test_callbacks.py`**

- `test_clears_via_public_api` — fake core with a recording `clear_model_cache()`; `par.name = "Model"` → recorder called exactly once; `State == "loading"`; status text starts with `[loading]`.
- `test_ignores_other_params` — `par.name = "confidence"` → state unchanged, no status write.
- `test_missing_status_dat_no_crash` — container without `status` child → no exception, state still set to `loading`.
- `test_missing_core_no_crash` — `fetch` returns None → no exception.
- `test_case_insensitive_names` — `par.name = "customCheckpoint"` → handled (cache cleared).

- [ ] **Step 2: Run tests — verify they fail**

Run: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\test_callbacks.py`
Expected: FAIL — `test_clears_via_public_api` fails: current code calls `core._MODEL_CACHE.clear()` (AttributeError on the fake) and `test_missing_status_dat_no_crash` fails on `root.op("status").text`.

- [ ] **Step 3: Implement in `python/callbacks.pyt`**

Replace `core._MODEL_CACHE.clear()` with `core.clear_model_cache()`; wrap the status write:

```python
status = root.op("status")
if status is not None:
    status.text = f"[loading] reloading model ({par.name} changed)"
```

- [ ] **Step 4: Run tests — verify they pass**

Run: `& "D:\projects\tvhead_instance_segmentation\.venv\Scripts\python.exe" tests\test_callbacks.py`
Expected: `5/5 pass`.

- [ ] **Step 5: Grep gate**

Run: `rg "op\(\"/|op\('/|/atltvhead|_MODEL_CACHE" python/callbacks.pyt`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add python/callbacks.pyt tests/test_callbacks.py
git commit -m "refactor: callbacks use clear_model_cache() and guard status DAT"
```

---

### Task 5: `README.md`

**Files:**
- Create (overwrite the empty file): `README.md`

**Interfaces:**
- Consumes: parameter table from `python/add_params.pyt`; layout facts from the repo.
- Produces: `README.md` with exactly these H2 sections, in order: `Requirements`, `Quick start`, `Parameters`, `Repository layout`, `Development`, `Troubleshooting`, `Status`.

Content requirements:
- Title: `TV Head Instance Segmentation (RF-DETR TouchDesigner TOX)`; one-paragraph description (RF-DETR real-time instance detection + segmentation in TouchDesigner; outputs union mask, box data, detection table; chromakey-ready).
- `Requirements`: TouchDesigner with bundled Python 3.11; GPU optional (CUDA auto-selects, CPU fallback); dependencies install via tdPyEnvManager into `.venv` from `requirements.txt`.
- `Quick start`: 1) drop `rfdetr.tox` into any network; 2) run the tdPyEnvManager setup so `.venv` is created and populated; 3) wire a source image into the `infer` Script TOP; 4) set `Model`/`Device`; outputs listed (mask TOP RGBA 0/1; `boxdata` TOP 384 floats, 6 per detection — normalized x1,y1,x2,y2, class_id, confidence; `detections` DAT; `State`/`Ncount`/`Inferms`/`Dets` params).
- `Parameters`: table of all 13 frozen parameters with type, default, and meaning (from `add_params.pyt`: Model menu default `rfdetr-seg-medium`; Device menu `auto|cuda|cpu` default `auto`; Confidence 0.5; Chromasharp 0.0; Chromainvert 0; Chromacolor (0,1,0); Classwhitelist "" (empty = no filter, comma-separated names); Customcheckpoint "" (path to a custom .ckpt); Hidesource 0; State init; Ncount 0; Inferms 0; Dets 0).
- `Repository layout`: `python/extension.pyt` → `infer` Script TOP callback DAT; `python/callbacks.pyt` → container Callbacks DAT; `python/add_params.pyt` → `python/add_params` DAT; `python/core.py` → `python/core` DAT; `TDPyEnvManagerContext.yaml` (tdPyEnvManager env: `.venv`, Python 3.11); `tests/` (headless suites); `docs/` (architecture, implementation guide, new-implementation spec).
- `Development`: edit a `.pyt` → sync it into the matching TOX DAT (see STATUS.html §4) → run the headless suites; `python/core.py` changes must keep numpy-only top-level imports.
- `Troubleshooting`: `[no-deps]` → run tdPyEnvManager setup; first model use downloads weights to `%USERPROFILE%\.roboflow\models`; `[error]` lines in the status DAT name the failing stage.
- `Status`: link to `STATUS.html`.
- No absolute operator paths anywhere in the README.

- [ ] **Step 1: Write `README.md`**

Per the content requirements above.

- [ ] **Step 2: Verify**

```bash
rg -c "^## " README.md            # Expected: 7
rg "atltvhead|ROOT_PATH" README.md # Expected: no matches
rg "TDPyEnvManagerContext.yaml" README.md  # Expected: present
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: repository README"
```

---

### Task 6: New `STATUS.html`

**Files:**
- Create (replaces the deleted file): `STATUS.html`
- Reference (read-only): the old page via `git show dev:STATUS.html` — reuse its `<style>` block verbatim (same Catppuccin dark theme, `.badge`, `.ok/.todo`, `ol.steps`, `.note`).

**Interfaces:**
- Consumes: test count from Task 1's final run; TOX facts from `docs/architecture.md` and `docs/flow-diagram.svg`.
- Produces: `STATUS.html` with section ids, in order: `status`, `diagram`, `prereq`, `build`, `verify`, `trouble`, `next`.

Content requirements:
- `<title>` + `<h1>`: `TV Head Instance Segmentation — Status`; header badges: `self-contained .tox`, `tdPyEnvManager (no installer)`, `core <N>/<N> tested`, `updated 2026-10-09`.
- `#status` Current status — table rows: Design docs (done); component-relative scripts (done — extension/callbacks/add_params, all tested); pure core `python/core.py` (done, tested `<N>/<N>`); dependency management (done — tdPyEnvManager `.venv` py3.11, `installer.pyt` removed); README (done); TOX rebuild (done — 46 KB, in TD); In-TD verification of the refactored DATs (todo — §5 checklist); model weights pre-download (partial — seg-medium + nano cached).
- `#diagram` Data flow — `<img class="fig" src="docs/flow-diagram.svg">` + one-line summary (source → infer Script TOP → union mask / boxdata TOP / detections DAT → GLSL outputs).
- `#prereq` Prerequisites — TD with Python 3.11; RTX-class GPU optional (CUDA cu128 wheels); internet for first weight download.
- `#build` Sync the TOX DATs — numbered `ol.steps`: open `rfdetr.tox`; paste `python/extension.pyt` into the `infer` Script TOP's callback DAT; `python/callbacks.pyt` into the container's Callbacks DAT; `python/add_params.pyt` into `python/add_params`; `python/core.py` into `python/core`; let tdPyEnvManager run setup (`.venv` + `requirements.txt`); save.
- `#verify` In-TD verification checklist — `State` cycles init → loading → ready on first cook; mask TOP is full-res 0/1; `boxdata` is 384×1; `detections` DAT has header + one row per detection; switching `Model` does not hang the UI; renaming the component to `rfdetr2` in a scratch project still works (portability proof); dropping the `.tox` into a brand-new empty project works.
- `#trouble` Troubleshooting — `[no-deps]` → tdPyEnvManager setup; missing weights → first cook downloads; CUDA unavailable → `Device: cpu`; `operator not found` messages name the missing internal DAT.
- `#next` Next steps — pre-download all variants; ONNX/TensorRT for sub-10 ms; per-class colors in the overlay; packaging.
- No `atltvhead` strings anywhere in the file.

- [ ] **Step 1: Extract the old CSS**

Run: `git show dev:STATUS.html` (read the `<style>` block; reuse verbatim).

- [ ] **Step 2: Write `STATUS.html`**

Per the content requirements above, with `<N>` replaced by the actual Task 1 test count.

- [ ] **Step 3: Verify**

```bash
rg -o 'id="(status|diagram|prereq|build|verify|trouble|next)"' STATUS.html  # Expected: 7 matches, in order
rg "atltvhead" STATUS.html          # Expected: no matches
rg "flow-diagram.svg" STATUS.html   # Expected: present
```

- [ ] **Step 4: Commit**

```bash
git add STATUS.html
git commit -m "docs: new status page for the self-contained TOX"
```

---

### Task 7: TOX sync + in-TD verification (manual gate)

**Files:**
- Modify (in TouchDesigner, then commit the binary): `rfdetr.tox`, optionally `tvhead-seg-tool.toe`

**Interfaces:**
- Consumes: the refactored `python/*.pyt` files and `python/core.py` (canonical sources).
- Produces: a `.tox` whose DAT contents match the repo files, verified by the `#verify` checklist.

The `.tox` binary is not safely rewritable from this machine; this task is GUI work in TouchDesigner. It cannot complete without the user.

- [ ] **Step 1 (user, in TD): sync DATs** — follow STATUS.html §4: paste the four repo files into their matching DATs.
- [ ] **Step 2 (user, in TD): run tdPyEnvManager setup** — confirm `.venv` installs `requirements.txt` cleanly.
- [ ] **Step 3 (user, in TD): run the STATUS.html §5 checklist**, including the `rfdetr2` rename portability check in a scratch project.
- [ ] **Step 4: commit the binaries**

```bash
git add rfdetr.tox tvhead-seg-tool.toe
git commit -m "chore: sync TOX DATs with refactored component-relative scripts"
```
