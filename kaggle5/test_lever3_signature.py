#!/usr/bin/env python3
"""The shim must survive ANY harness signature, not just the local one.

Regression for the live failure:

  analyzer failed at action 1: _patch_module.<locals>.run_sandboxed_python()
  got an unexpected keyword argument 'animation_handler'

The wrapper had mirrored the local bundle's signature. The live harness passes
an extra kwarg, so every analyzer call died before the model was reached.
"""
import os
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "k5")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lever3_runtime  # noqa: E402

PASS, FAIL = [], []


def check(ok, label):
    (PASS if ok else FAIL).append(label)
    print(("  PASS  " if ok else "  FAIL  ") + label)


def fresh_harness(sandbox_fn, name="inference.agent.tool_agent"):
    """Register a stand-in tool_agent module backed by sandbox_fn."""
    mod = types.ModuleType(name)

    class ToolAgent:
        def _run_python_tool(self, state_path, arguments):
            return None

    mod.ToolAgent = ToolAgent
    mod.run_sandboxed_python = sandbox_fn
    sys.modules[name] = mod
    if "inference" not in sys.modules:
        sys.modules["inference"] = types.ModuleType("inference")
    if "inference.agent" not in sys.modules:
        sys.modules["inference.agent"] = types.ModuleType("inference.agent")
    return mod


FRAME = {"ascii": "AAA\nABA\nAAA", "level": 1, "step": 0,
         "shape": [3, 3], "grid": [[0, 0, 0], [0, 1, 0], [0, 0, 0]]}
INITIAL = {"current_frame": FRAME, "history": [], "valid_actions": ["UP", "DOWN"],
           "last_action_result": [{"action_num": 1, "executed": True}]}


print("\n[1] a harness that passes animation_handler (the live shape)")
seen = {}


def live_shaped(*, code, timeout_seconds, initial_state, action_handler,
                animation_handler=None):
    seen["animation_handler"] = animation_handler
    seen["code"] = code
    seen["memory"] = (initial_state or {}).get("memory")
    out = action_handler([{"action": "UP"}])
    seen["stop_reason"] = (out.get("action_result") or {}).get("stop_reason")
    return {"stdout": "", "result": None, "error": "", "action_results": []}


fresh_harness(live_shaped)
runtime = lever3_runtime.install(gate=True)
check(runtime.installed, "patched the stand-in harness")

state_path = Path(tempfile.mkdtemp()) / "game" / "tool_runtime_state.json"
state_path.parent.mkdir(parents=True)
runtime.state_path = state_path

sys.modules["inference.agent.tool_agent"].run_sandboxed_python(
    code="print('hi')",
    timeout_seconds=30,
    initial_state=dict(INITIAL),
    action_handler=lambda actions: {"action_result": {"executed": True},
                                    "state": dict(INITIAL)},
    animation_handler="SENTINEL",
)
check(seen.get("animation_handler") == "SENTINEL",
      "animation_handler forwarded verbatim (previously TypeError)")
check(seen.get("memory") is not None, "memory payload injected into initial_state")
check("_K5_MEMORY" in seen.get("code", ""), "sandbox preamble prepended")
check("print('hi')" in seen.get("code", ""), "original code preserved")

print("\n[2] the gate still fires through that signature")
runtime.memory_for(state_path)[0].observe(
    FRAME["ascii"], 1, {"action": "UP"},
    {"executed": True, "board_changed": False}, FRAME["ascii"])
sys.modules["inference.agent.tool_agent"].run_sandboxed_python(
    code="pass", timeout_seconds=30, initial_state=dict(INITIAL),
    action_handler=lambda actions: {"action_result": {"executed": True},
                                    "state": dict(INITIAL)},
    animation_handler=None,
)
check(seen.get("stop_reason") == "k5_pruned", "gate refuses the recorded no-op")

print("\n[3] a harness with the older, narrower signature still works")
called = {}


def narrow(*, code, timeout_seconds, initial_state, action_handler):
    called["yes"] = True
    action_handler([{"action": "DOWN"}])
    return {"stdout": "", "result": None, "error": "", "action_results": []}


fresh_harness(narrow)
runtime2 = lever3_runtime.install(gate=True)
runtime2.state_path = state_path
sys.modules["inference.agent.tool_agent"].run_sandboxed_python(
    code="pass", timeout_seconds=30, initial_state=dict(INITIAL),
    action_handler=lambda actions: {"action_result": {"executed": True},
                                    "state": dict(INITIAL)},
)
check(called.get("yes") is True, "narrow signature still works")

print("\n[4] defensive handling when last_action_result is a list (live regression)")
seen.clear()
fresh_harness(live_shaped)
runtime4 = lever3_runtime.install(gate=True)
state_path4 = Path(tempfile.mkdtemp()) / "game" / "tool_runtime_state.json"
state_path4.parent.mkdir(parents=True)
runtime4.state_path = state_path4
runtime4.memory_for(state_path4)[0].observe(
    FRAME["ascii"], 1, {"action": "UP"},
    {"executed": True, "board_changed": False}, FRAME["ascii"])

list_state = dict(INITIAL)
list_state["last_action_result"] = [{"action_num": 7, "executed": True}]
sys.modules["inference.agent.tool_agent"].run_sandboxed_python(
    code="pass", timeout_seconds=30, initial_state=list_state,
    action_handler=lambda actions: {"action_result": {"executed": True},
                                    "state": list_state},
    animation_handler=None,
)
check(seen.get("stop_reason") == "k5_pruned", "gate pruned successfully with list last_action_result")

print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
for f in FAIL:
    print("  - " + f)
sys.exit(1 if FAIL else 0)
