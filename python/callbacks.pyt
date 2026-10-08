# rfdetr.python.callbacks — Callbacks DAT on the `rfdetr` container.
# Wire: set the container's "Callbacks DAT" parameter to this DAT.
#
# When the model selection changes, drop the cached model so the next
# cook of `infer` loads the new variant (load_model caches per
# variant/device/checkpoint). The cook thread does the loading; we only
# flip the state text so the UI can show it.

# rfdetr.python.callbacks

_INVALIDATORS = {"model", "device", "customcheckpoint"}


def onValueChange(par, prev):
    if par.name.lower() not in _INVALIDATORS:
        return
    
    root = par.owner
    core = root.fetch("rfdetr_core", None)
    if core is not None:
        core._MODEL_CACHE.clear()
        
    root.par.State = "loading"
    root.op("status").text = f"[loading] reloading model ({par.name} changed)"