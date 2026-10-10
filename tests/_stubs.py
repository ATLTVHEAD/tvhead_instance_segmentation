# Shared TouchDesigner fakes for the headless test suites in tests/.
#
# These stand in for the small slice of the TD operator API that the
# python/*.pyt scripts use. They are NOT a general TD emulator.

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


class FakePar:
    """One parameter: .name, .value/.val (settable), .eval()."""

    def __init__(self, name, value=None):
        self.name = name
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, v):
        self._value = v

    @property
    def val(self):
        return self._value

    @val.setter
    def val(self, v):
        self._value = v

    def eval(self):
        return self._value

    def __repr__(self):
        return "FakePar(%s=%r)" % (self.name, self._value)


class FakeParBag:
    """Stand-in for <op>.par: attribute + index access, settable."""

    def __init__(self):
        object.__setattr__(self, "_pars", {})

    def add(self, name, value=None):
        if name not in self._pars:
            self._pars[name] = FakePar(name, value)
        return self._pars[name]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._pars[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if isinstance(value, FakePar):
            self._pars[name] = value
        elif name in self._pars:
            self._pars[name].value = value
        else:
            self.add(name, value)

    def __getitem__(self, name):
        return self._pars.get(name)

    def __contains__(self, name):
        return name in self._pars


# ---------------------------------------------------------------------------
# Operators / components
# ---------------------------------------------------------------------------


class FakeComp:
    """A component: .name, .op("a/b"), .par, .store/.fetch, .parent()."""

    def __init__(self, name, children=None):
        self.name = name
        self.children = {}
        self.par = FakeParBag()
        self._store = {}
        self._parent = None
        if children:
            for child in children:
                self.addChild(child)

    def addChild(self, node):
        self.children[node.name] = node
        node._parent = self
        return node

    def op(self, path):
        node = self
        for part in str(path).split("/"):
            if part == "":
                continue
            node = node.children.get(part)
            if node is None:
                return None
        return node

    def parent(self):
        return self._parent

    def store(self, key, value):
        self._store[key] = value

    def fetch(self, key, default=None):
        return self._store.get(key, default)

    def appendCustomPage(self, name):
        page = FakePage(name, self)
        self._pages[name] = page
        return page

    @property
    def customPages(self):
        if not hasattr(self, "_pages"):
            self._pages = {}
        return self._pages


class FakePage:
    """A custom parameter page; appended params register on the owner."""

    def __init__(self, name, owner):
        self.name = name
        self.owner = owner
        self.pars = []

    def _append(self, name, kind):
        par = self.owner.par.add(name, None)
        par.kind = kind
        self.pars.append(par)
        return par

    def appendMenu(self, name):
        return self._append(name, "menu")

    def appendFloat(self, name):
        return self._append(name, "float")

    def appendInt(self, name):
        return self._append(name, "int")

    def appendRGB(self, name):
        return self._append(name, "rgb")

    def appendStr(self, name):
        return self._append(name, "str")


class FakeTextDat:
    """A DAT with a .text property (read and write)."""

    def __init__(self, name, text=""):
        self.name = name
        self.text = text


class FakeTop:
    """A TOP with par.width/par.height and a .pixel setter."""

    def __init__(self, name):
        self.name = name
        self.par = FakeParBag()
        self.par.add("width", 0)
        self.par.add("height", 0)
        self._pixel = None

    @property
    def pixel(self):
        return self._pixel

    @pixel.setter
    def pixel(self, arr):
        self._pixel = np.asarray(arr)


class FakeTableDat:
    """A table DAT with clear()/appendRows()."""

    def __init__(self, name):
        self.name = name
        self.rows = []

    def clear(self):
        self.rows = []

    def appendRows(self, rows):
        self.rows.extend(rows)


class FakePixelOp:
    """A pixel operator exposing .height/.width/.numpyArray()."""

    def __init__(self, array):
        self._array = np.asarray(array)

    @property
    def height(self):
        return self._array.shape[0]

    @property
    def width(self):
        return self._array.shape[1]

    def numpyArray(self):
        return self._array


class FakeScriptTop(FakeComp):
    """A Script TOP: .inputs list, copyNumpyArray() recorded in .copied."""

    def __init__(self, name="infer"):
        super().__init__(name)
        self.inputs = []
        self.copied = []

    def addInput(self, array):
        inp = FakePixelOp(array)
        self.inputs.append(inp)
        return inp

    def copyNumpyArray(self, arr):
        self.copied.append(np.asarray(arr))


class FakeApp:
    """Stand-in for the global TD app object."""

    frame = 0


# ---------------------------------------------------------------------------
# Fakes for torch / rfdetr (module-level, for load_model and friends)
# ---------------------------------------------------------------------------


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


class StubbedTorchRfdetr:
    """Replace torch/rfdetr in sys.modules with minimal fakes.

    ``result`` is what the fake model's .predict() returns; set it
    before (or while) calling the code under test to inject detections.
    """

    def __init__(self, cuda_available=False, result=None):
        self.cuda_available = cuda_available
        self.result = result
        self._saved = {}

    def __enter__(self):
        for name in ("torch", "rfdetr"):
            self._saved[name] = sys.modules.get(name)

        torch_mod = types.ModuleType("torch")
        torch_mod.__version__ = "0.0.0-stub"
        torch_mod.cuda = types.SimpleNamespace(
            is_available=lambda: self.cuda_available,
            empty_cache=lambda: None,
            current_device=lambda: 0,
        )
        torch_mod.device = lambda spec: spec

        rfdetr_mod = types.ModuleType("rfdetr")
        rfdetr_mod.__version__ = "0.0.0-stub"

        mgr = self

        class FakeVariant:
            def predict(self, frame, threshold=None, include_source_image=False):
                return mgr.result

        # Variant class names come from the real core so the stubs track it.
        core_path = Path(__file__).resolve().parents[1] / "python" / "core.py"
        spec = importlib.util.spec_from_file_location("rfdetr_core_stubref", core_path)
        core_ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_ref)
        for class_name in core_ref.VARIANTS.values():
            setattr(rfdetr_mod, class_name, FakeVariant)

        class RFDETR:
            @staticmethod
            def from_checkpoint(path):
                return FakeVariant()

        rfdetr_mod.RFDETR = RFDETR

        sys.modules["torch"] = torch_mod
        sys.modules["rfdetr"] = rfdetr_mod
        return self

    def __exit__(self, *exc):
        for name, orig in self._saved.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig
        return False


# ---------------------------------------------------------------------------
# Import blocker
# ---------------------------------------------------------------------------


class _RaisingLoader:
    def __init__(self, name):
        self.name = name

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError("No module named %r (blocked by test)" % self.name)


class _BlockFinder:
    def __init__(self, names):
        self.names = names

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self.names or any(
            fullname.startswith(n + ".") for n in self.names
        ):
            return importlib.util.spec_from_loader(
                fullname, _RaisingLoader(fullname), is_package=True
            )
        return None


class BlockImport:
    """Context manager: ``import <name>`` raises ImportError inside."""

    def __init__(self, names):
        self.names = set(names)
        self._finder = _BlockFinder(self.names)
        self._saved = {n: sys.modules.get(n) for n in self.names}

    def __enter__(self):
        for n in self.names:
            sys.modules.pop(n, None)
        sys.meta_path.insert(0, self._finder)
        return self

    def __exit__(self, *exc):
        try:
            sys.meta_path.remove(self._finder)
        except ValueError:
            pass
        for n, orig in self._saved.items():
            if orig is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = orig
        return False


# ---------------------------------------------------------------------------
# .pyt loader
# ---------------------------------------------------------------------------


def load_pyt(path, me=None, extra=None):
    """Exec a .pyt file the way TD would, with TD builtins injected.

    Returns a namespace object (attribute access for the module's globals).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    namespace = {
        "__name__": "pyt_under_test",
        "__file__": str(path),
        "me": me,
        "op": lambda *a, **k: None,
        "app": FakeApp(),
    }
    if extra:
        namespace.update(extra)
    exec(compile(text, str(path), "exec"), namespace)
    return types.SimpleNamespace(
        **{k: v for k, v in namespace.items() if not k.startswith("__")}
    )
