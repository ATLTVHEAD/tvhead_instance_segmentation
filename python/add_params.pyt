# rfdetr.add_params
# TouchDesigner Textport
# Safe to re-run

root = op("/atltvhead_greenscreen/rfdetr")

if root is None:
    raise RuntimeError(
        "rfdetr container not found at "
        "/atltvhead_greenscreen/rfdetr"
    )


# ---------------------------------------------------------
# Get or create Custom Parameters page
# ---------------------------------------------------------

try:
    page = root.customPages["Custom Parameters"]
except Exception:
    page = root.appendCustomPage("Custom Parameters")


# ---------------------------------------------------------
# Helper
# ---------------------------------------------------------

def has_par(name):
    try:
        return root.par[name] is not None
    except Exception:
        return False


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

VARIANTS = [
    "rfdetr-nano",
    "rfdetr-small",
    "rfdetr-medium",
    "rfdetr-large",
    "rfdetr-seg-nano",
    "rfdetr-seg-small",
    "rfdetr-seg-medium",
    "rfdetr-seg-large",
    "rfdetr-seg-xlarge",
    "rfdetr-seg-2xlarge",
]

if not has_par("Model"):
    page.appendMenu("Model")

m = root.par.Model
m.menuNames = VARIANTS
m.menuLabels = VARIANTS
m.value = "rfdetr-seg-medium"


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

if not has_par("Device"):
    page.appendMenu("Device")

d = root.par.Device
d.menuNames = ["auto", "cuda", "cpu"]
d.menuLabels = ["auto", "cuda", "cpu"]
d.value = "auto"


# ---------------------------------------------------------
# Detection / image parameters
# ---------------------------------------------------------

if not has_par("Confidence"):
    page.appendFloat("Confidence")

root.par.Confidence.value = 0.5


if not has_par("Chromasharp"):
    page.appendFloat("Chromasharp")

root.par.Chromasharp.value = 0.0


if not has_par("Chromainvert"):
    page.appendInt("Chromainvert")

root.par.Chromainvert.value = 0


if not has_par("Chromacolor"):
    page.appendRGB("Chromacolor")

root.par.Chromacolor.value = (0, 1, 0)


# ---------------------------------------------------------
# String parameters
# ---------------------------------------------------------

if not has_par("Classwhitelist"):
    page.appendStr("Classwhitelist")


if not has_par("Customcheckpoint"):
    page.appendStr("Customcheckpoint")


# ---------------------------------------------------------
# Miscellaneous parameters
# ---------------------------------------------------------

if not has_par("Hidesource"):
    page.appendInt("Hidesource")

root.par.Hidesource.value = 0


if not has_par("State"):
    page.appendStr("State")

root.par.State.value = "init"


if not has_par("Ncount"):
    page.appendInt("Ncount")


if not has_par("Inferms"):
    page.appendFloat("Inferms")


if not has_par("Dets"):
    page.appendInt("Dets")


# ---------------------------------------------------------
# Set ranges
# ---------------------------------------------------------

for name, lo, hi in (
    ("Confidence", 0, 1),
    ("Chromasharp", 0, 1),
    ("Chromainvert", 0, 1),
    ("Hidesource", 0, 1),
):
    try:
        par = root.par[name]
        par.minValue = lo
        par.maxValue = hi
    except Exception:
        pass


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

print(
    "rfdetr params ready:",
    [p.name for p in page.pars]
)