# rfdetr.python.installer
# TouchDesigner Callbacks DAT

import os
import subprocess
import sys


# =========================================================
# Component path
# =========================================================

ROOT = "/atltvhead_greenscreen/rfdetr"


# =========================================================
# State
# =========================================================

_st = {
    "proc": None,
    "cmds": None,
    "phase": 0,
    "names": None,
    "idx": 0,
    "log_file": None,
}


# =========================================================
# Logging
# =========================================================


def _log(line):

    log_op = op(ROOT + "/setup/install_log")

    if log_op:
        log_op.text += line

    else:
        print("ERROR: install_log not found")


# =========================================================
# Process output file
# =========================================================


def _output_path():

    return os.path.join(
        os.environ.get(
            "TEMP",
            os.getcwd(),
        ),
        "rfdetr_install_output.txt",
    )


# =========================================================
# Launch subprocess
# =========================================================


def _launch(cmd):

    if _st["proc"] is not None:
        _log("ERROR: process already running\n")

        return False

    output_path = _output_path()

    _st["log_file"] = output_path

    # -----------------------------------------------------
    # Clear previous output
    # -----------------------------------------------------

    try:
        with open(
            output_path,
            "w",
            encoding="utf-8",
        ):
            pass

    except Exception as e:
        _log("ERROR creating process log: %s\n" % e)

        return False

    # -----------------------------------------------------
    # Log command
    # -----------------------------------------------------

    _log("\n$ " + " ".join(cmd) + "\n")

    _log("Launching subprocess...\n")

    # -----------------------------------------------------
    # Open output file
    # -----------------------------------------------------

    try:
        output_file = open(
            output_path,
            "w",
            encoding="utf-8",
        )

        proc = subprocess.Popen(
            cmd,
            stdout=output_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            shell=False,
        )

    except Exception as e:
        try:
            output_file.close()
        except:
            pass

        _log("ERROR launching subprocess: %s\n" % e)

        return False

    _st["proc"] = proc

    _log("PROCESS STARTED\n")

    return True


# =========================================================
# Read process output
# =========================================================


def _read_process_output():

    path = _st["log_file"]

    if not path:
        return

    if not os.path.exists(path):
        return

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            text = f.read()

        # -------------------------------------------------
        # Store how much we have already displayed
        # -------------------------------------------------

        previous = getattr(
            _read_process_output,
            "_length",
            0,
        )

        if len(text) > previous:
            new_text = text[previous:]

            _log(new_text)

            _read_process_output._length = len(text)

    except Exception:
        pass


# =========================================================
# Finish process
# =========================================================


def _finish_process():

    proc = _st["proc"]

    if proc is None:
        return

    # -----------------------------------------------------
    # Still running
    # -----------------------------------------------------

    return_code = proc.poll()

    if return_code is None:
        return

    # -----------------------------------------------------
    # Read final output
    # -----------------------------------------------------

    _read_process_output()

    # -----------------------------------------------------
    # Close process handles
    # -----------------------------------------------------

    try:
        proc.stdout.close()

    except Exception:
        pass

    _st["proc"] = None

    _log("\nPROCESS FINISHED — exit code %s\n" % return_code)

    # -----------------------------------------------------
    # Advance installer
    # -----------------------------------------------------

    _advance(return_code)


# =========================================================
# Device detection
# =========================================================


def detect_device():

    dev_op = op(ROOT + "/setup/device_status")

    if not dev_op:
        print("ERROR: device_status not found")

        return

    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)

            dev_op.text = "cuda — %s, %.1f GB" % (
                p.name,
                p.total_memory / 1e9,
            )

        else:
            dev_op.text = "cpu (no CUDA detected)"

    except Exception as e:
        dev_op.text = "error: %s" % e


# =========================================================
# Start installation
# =========================================================


def start_install():

    if _st["proc"] is not None:
        _log("install already running\n")

        return

    py = sys.executable

    _st["cmds"] = [
        [
            py,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "--disable-pip-version-check",
        ],
        [
            py,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ],
        [
            py,
            "-m",
            "pip",
            "install",
            "rfdetr",
            "supervision",
            "opencv-python-headless",
        ],
    ]

    _st["phase"] = 0

    _read_process_output._length = 0

    _log("\n========================================\n")

    _log("Starting RF-DETR installation\n")

    _log("Python: %s\n" % py)

    _log("========================================\n")

    _launch(_st["cmds"][0])


# =========================================================
# Predownload
# =========================================================


def start_predl():

    if _st["proc"] is not None:
        _log("predownload already running\n")

        return

    try:
        import rfdetr

    except ImportError as e:
        _log("rfdetr not importable: %s\n" % e)

        _log("Run Install first.\n")

        return

    variant_op = op(ROOT + "/setup/variant_check")

    if not variant_op:
        _log("variant_check not found\n")

        return

    names = []

    for row in variant_op.rows():
        if len(row) >= 2 and row[1].val in (
            "1",
            "true",
            "True",
        ):
            names.append(row[0].val)

    if not names:
        _log("no variants checked\n")

        return

    _st["names"] = names
    _st["idx"] = 0
    _st["cmds"] = None

    _read_process_output._length = 0

    _log("\nStarting model predownload...\n")

    _next_predl()


# =========================================================
# Load core
# =========================================================


def _core_mod():

    import types

    core_dat = op(ROOT + "/python/core")

    if not core_dat:
        raise RuntimeError("Could not find " + ROOT + "/python/core")

    mod = types.ModuleType("rfdetr_core_predl")

    exec(
        compile(
            core_dat.text,
            "core.py",
            "exec",
        ),
        mod.__dict__,
    )

    return mod


# =========================================================
# Next predownload
# =========================================================


def _next_predl():

    if _st["names"] is None:
        return

    i = _st["idx"]

    if i >= len(_st["names"]):
        _st["names"] = None

        _log("\npredownload complete\n")

        return

    key = _st["names"][i]

    try:
        cls = _core_mod().VARIANTS[key]

    except Exception as e:
        _log(
            "ERROR loading variant %s: %s\n"
            % (
                key,
                e,
            )
        )

        _st["idx"] += 1

        _next_predl()

        return

    code = "import rfdetr; getattr(rfdetr, '%s')()" % cls

    _log(
        "predownload [%d/%d] %s\n"
        % (
            i + 1,
            len(_st["names"]),
            key,
        )
    )

    _read_process_output._length = 0

    _launch([
        sys.executable,
        "-c",
        code,
    ])


# =========================================================
# Advance
# =========================================================


def _advance(return_code):

    # =====================================================
    # Installation
    # =====================================================

    if _st["cmds"] is not None:
        phase = _st["phase"]

        if return_code != 0:
            _log("\n========================================\n")

            _log("INSTALL FAILED\n")

            _log("Exit code: %s\n" % return_code)

            _log("========================================\n")

            _st["cmds"] = None
            _st["phase"] = 0

            return

        _log(
            "\nSTEP %d/%d FINISHED SUCCESSFULLY\n"
            % (
                phase + 1,
                len(_st["cmds"]),
            )
        )

        phase += 1

        _st["phase"] = phase

        if phase < len(_st["cmds"]):
            _log(
                "\nStarting step %d/%d...\n"
                % (
                    phase + 1,
                    len(_st["cmds"]),
                )
            )

            _read_process_output._length = 0

            _launch(_st["cmds"][phase])

        else:
            _st["cmds"] = None
            _st["phase"] = 0

            _log("\n========================================\n")

            _log("RF-DETR INSTALLATION COMPLETE\n")

            _log("========================================\n")

        return

    # =====================================================
    # Predownload
    # =====================================================

    if _st["names"] is not None:
        key = _st["names"][_st["idx"]]

        if return_code == 0:
            _log("predownload finished: %s\n" % key)

        else:
            _log(
                "predownload FAILED: %s "
                "(exit code %s)\n"
                % (
                    key,
                    return_code,
                )
            )

        _st["idx"] += 1

        _next_predl()


# =========================================================
# Buttons
# =========================================================


def on_button_click(btn_name):

    if btn_name == "btn_install":
        start_install()

    elif btn_name == "btn_predl":
        start_predl()

    elif btn_name == "btn_dev":
        detect_device()


def onPulse(par):

    on_button_click(par.op.name)


# =========================================================
# Cook
# =========================================================


def onCook(scriptOp):

    # Read whatever pip has written so far.

    _read_process_output()

    # Check whether the process has exited.

    _finish_process()
