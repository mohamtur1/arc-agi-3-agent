# CELL 4B-2 of 2 — install Lever-3 at runtime.
#
# No upstream file is edited. An earlier revision shipped a byte-pinned
# `git apply` patch; it failed on the live kernel with
# `patch does not apply ... tool_agent.py:19` because the kernel's harness is a
# different revision than the dataset copy the patch was cut against.
#
# This installs Lever-3 by NAME instead:
#   * lever3_runtime.py is written beside this cell by the 4b-1b cell
#   * install() either patches the already-imported harness, or arms a
#     meta-path watcher that patches it the moment it is first imported
#   * if a name it needs is missing, it says so and patches nothing
#   * memory is keyed by the frame's ASCII text, so no `.grid` is required
#
# VARIANT
#   lever3 : memory + hard prune gate + prompt directives  (default)
#   kernel : memory view present, gate off, prompt untouched (A/B rollback)
import importlib
import os
import sys
import traceback
from pathlib import Path

VARIANT = "lever3"  # k5-variant-line

# ------------------------------------------------------------------ locate
def _first_existing(paths):
    for p in paths:
        try:
            if p and Path(p).is_dir():
                return Path(p)
        except OSError:
            continue
    return None


WORKING_DIR = _first_existing([
    globals().get("WORKING_DIR"),
    os.environ.get("WORKING_DIR"),
    "/kaggle/working",
    str(Path.cwd()),
]) or Path.cwd()

# lever3_runtime.py is written next to the driver by the cell above
CANDIDATES = [
    globals().get("K5_DIR"),
    os.environ.get("K5_DIR"),
    WORKING_DIR / "kaggle5",
    Path.cwd(),
    Path(__file__).resolve().parent if "__file__" in globals() else None,
]
K5_DIR = None
for cand in CANDIDATES:
    if cand and (Path(cand) / "lever3_runtime.py").is_file():
        K5_DIR = Path(cand)
        break
if K5_DIR is None:
    raise FileNotFoundError(
        "lever 3 FAILED: lever3_runtime.py not found. Looked in: %s. "
        "Run the cell above it first." % ", ".join(str(c) for c in CANDIDATES))

if str(K5_DIR) not in sys.path:
    sys.path.insert(0, str(K5_DIR))

# ------------------------------------------------------------------ install
try:
    import lever3_runtime
    importlib.reload(lever3_runtime)          # pick up a re-pushed shim
except Exception:
    print("lever 3 FAILED to import lever3_runtime:")
    traceback.print_exc()
    raise

runtime = lever3_runtime.install(gate=(VARIANT == "lever3"))
for event in runtime.events:
    print("lever 3: %s" % event)

# ------------------------------------------------------------- verification
# Importing the harness now is not a side effect to avoid: if it imports, the
# watcher patches it immediately and we can prove Lever-3 is live before the
# games start. If it cannot import yet, the watcher stays armed.
try:
    ta = importlib.import_module("inference.agent.tool_agent")
    runtime = lever3_runtime._RUNTIME
    for event in runtime.events[len(runtime.events) - 1:]:
        print("lever 3: %s" % event)
except Exception as exc:
    print("lever 3: harness not importable yet (%s: %s)" % (type(exc).__name__, exc))
    print("lever 3: armed — it will be patched when first imported")

if not runtime.installed:
    print("lever 3: " + "!" * 60)
    print("lever 3: NOT ACTIVE. The watcher is armed but the harness has not")
    print("lever 3: been patched yet. If the games still run, Lever-3 is not")
    print("lever 3: in effect and this run should not be trusted.")
    print("lever 3: " + "!" * 60)
else:
    missing = []
    for hook, check in (
        ("_run_python_tool", lambda m: hasattr(m.ToolAgent, "_run_python_tool")),
        ("run_sandboxed_python",
         lambda m: "run_sandboxed_python" in getattr(m, "__dict__", {})),
    ):
        try:
            if not check(ta if "ta" in dir() else sys.modules["inference.agent.tool_agent"]):
                missing.append(hook)
        except Exception:
            missing.append(hook)
    print("lever 3: variant=%s gate=%s" % (VARIANT, runtime.gate))
    if missing:
        print("lever 3: WARNING hooks not confirmed: %s" % ", ".join(missing))
    else:
        print("lever 3: LIVE (memory + gate installed on %s)" % runtime.patched_module)
print("lever 3: done")
