"""Lever-3 at runtime: deterministic episodic memory + a hard prune gate.

Why this shape
--------------
An earlier version shipped a byte-pinned `git apply` patch. It broke twice:
once because the bundle path was wrong, and once because the kernel's harness
is a different revision than the dataset copy the patch was cut against
(`patch does not apply` at tool_agent.py:19). Byte-pinning assumes you can see
the file you are patching; on this kernel we cannot.

This module installs Lever-3 by *name*, at import time:

  * no upstream file is edited, so there is nothing to drift
  * it works whether or not the harness was already imported
  * if a name it needs is missing, it says so loudly and does nothing else

Three hooks, all verified before use:

  1. ToolAgent._run_python_tool  -- tells us which game directory is live, so
     memory is per-game.
  2. tool_agent.run_sandboxed_python -- wraps the action handler, which is the
     only place that sees every real action and both frames around it.
     This is also where the gate lives: the refusal happens host-side, so a
     small model cannot talk its way past it.
  3. tool_agent._build_system_prompt -- appends the memory instructions.

State identity is the frame's ASCII rendering, not `.grid`: ASCII is present in
every revision of the payload, and hashing it makes memory work even where
`.grid` does not exist.
"""
from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import json
import sys
import traceback
from pathlib import Path

TARGET_MODULE = "inference.agent.tool_agent"
MEMORY_FILENAME = "tool_episodic_memory.json"

MAX_STATES = 512
MAX_EDGES_PER_STATE = 12
MAX_DEAD_ENDS = 512
SNAPSHOT_STATES = 256
SNAPSHOT_EDGES = 8
PRUNED = ("no_op", "dead_end", "loop")

_MASK64 = 0xFFFFFFFFFFFFFFFF
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_LEVEL_SALT = 0x9E3779B97F4A7C15

ADDENDUM = (
    "\n\nEpisodic memory (deterministic, host-side, survives every tool call):\n"
    "- `current_frame.memory.hash_grid(current_frame.ascii)` returns the 64-bit "
    "state hash for this board. Hash the ASCII, not a numeric grid.\n"
    "- `current_frame.memory.tried(h)` lists actions already executed from that "
    "state: dicts with `action`, `next`, `outcome`, `count`. Outcomes are "
    "`active`, `progress`, `no_op`, `loop`, `dead_end`.\n"
    "- `current_frame.memory.is_pruned(h, action)` is True when that action is a "
    "known no-op, loop, or dead end from that state.\n"
    "- `current_frame.memory.safe_actions(h, valid_actions)` filters a candidate "
    "list down to actions that are not pruned.\n"
    "- This is enforced in code, not prose: `action(...)` refuses a pruned action "
    "and returns `stop_reason='k5_pruned'` with the blocked actions, the "
    "transitions already recorded from this state, and the safe alternatives. "
    "Pick from `safe_actions` and continue in the same tool call.\n"
    "- If every valid action is pruned the gate stands down and lets the action "
    "through, so a level can never brick.\n"
    "- Read it before choosing: `h = current_frame.memory.hash_grid("
    "current_frame.ascii); print(current_frame.memory.tried(h))`.\n"
)


# --------------------------------------------------------------- state hash
def _ascii_text(frame_payload) -> str:
    if isinstance(frame_payload, dict):
        return str(frame_payload.get("ascii") or "")
    return ""


def _level_of(frame_payload) -> int:
    if isinstance(frame_payload, dict):
        try:
            return max(1, int(frame_payload.get("level") or 1))
        except (TypeError, ValueError):
            return 1
    return 1


def state_hash(ascii_text: str, level: int = 1) -> str:
    h = (_FNV_OFFSET ^ ((int(level) & 0xFFFF) * _LEVEL_SALT)) & _MASK64
    for byte in str(ascii_text).encode("utf-8", "replace"):
        h ^= byte
        h = (h * _FNV_PRIME) & _MASK64
    return "%016x" % h


def _num(value) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value).strip()


def action_key(action) -> str:
    if isinstance(action, dict):
        name = str(action.get("action") or "").strip().upper()
        parts = [name or "?"]
        if action.get("row") is not None:
            parts.append("R%s" % _num(action.get("row")))
        if action.get("col") is not None:
            parts.append("C%s" % _num(action.get("col")))
        return "|".join(parts)
    return str(action or "").strip().upper()


def action_name(action) -> str:
    return action_key(action).split("|", 1)[0]


# ------------------------------------------------------------------- memory
class CompactStateGraph:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, dict]] = {}
        self.dead_ends: list[str] = []
        self.stats = {k: 0 for k in
                      ("active", "progress", "no_op", "loop", "dead_end",
                       "overrides", "pruned_calls")}

    def observe(self, before: str, level: int, action, result: dict,
                after: str) -> str | None:
        if not before or not after or not isinstance(result, dict):
            return None
        if result.get("executed") is False:
            return None
        name = action_key(action)
        if not name:
            return None
        b = state_hash(before, level)
        a = state_hash(after, level)
        edge = self.states.setdefault(b, {})
        prior = edge.get(name)
        if result.get("game_over"):
            outcome = "dead_end"
        elif b == a or result.get("board_changed") is False:
            outcome = "no_op"
        elif result.get("level_completed") or result.get("run_complete"):
            outcome = "progress"
        elif prior is not None and prior.get("next") == a:
            outcome = "loop"
        else:
            outcome = "active"
        edge[name] = {"next": a, "outcome": outcome,
                      "count": int((prior or {}).get("count", 0)) + 1}
        while len(edge) > MAX_EDGES_PER_STATE:
            edge.pop(next(iter(edge)))
        if outcome == "dead_end" and a not in self.dead_ends:
            self.dead_ends.append(a)
            del self.dead_ends[:-MAX_DEAD_ENDS]
        self.stats[outcome] = self.stats.get(outcome, 0) + 1
        while len(self.states) > MAX_STATES:
            self.states.pop(next(iter(self.states)))
        return outcome

    def tried(self, h) -> list[dict]:
        return [{"action": k, "next": v.get("next"),
                 "outcome": v.get("outcome"), "count": v.get("count")}
                for k, v in self.states.get(str(h), {}).items()]

    def is_pruned(self, h, action) -> bool:
        key = str(h)
        rec = self.states.get(key, {}).get(action_key(action))
        if rec is None:
            return False
        return (rec.get("outcome") in PRUNED
                or rec.get("next") in self.dead_ends
                or rec.get("next") == key)

    def safe_actions(self, h, valid) -> list:
        return [a for a in (valid or []) if not self.is_pruned(h, a)]

    def gate(self, before: str, level: int, requested, valid):
        if not valid or not requested:
            return None
        h = state_hash(before, level)
        valid_names = {action_name(a) for a in valid}
        considered = [action_key(a) for a in requested
                      if action_name(a) in valid_names]
        if not considered:
            return None
        blocked = [a for a in considered if self.is_pruned(h, a)]
        if not blocked:
            return None
        safe = self.safe_actions(h, valid)
        if not safe:
            self.stats["overrides"] += 1
            return None
        self.stats["pruned_calls"] += 1
        return {"blocked": blocked, "safe_actions": safe, "tried": self.tried(h)}

    def to_payload(self) -> dict:
        keys = list(self.states)[-SNAPSHOT_STATES:]
        return {
            "version": 1,
            "states": {k: [[n, v.get("next"), v.get("outcome"),
                            int(v.get("count", 1))]
                           for n, v in list(self.states[k].items())[-SNAPSHOT_EDGES:]]
                       for k in keys},
            "dead_ends": list(self.dead_ends[-MAX_DEAD_ENDS:]),
            "stats": dict(self.stats),
        }

    @classmethod
    def from_payload(cls, payload) -> "CompactStateGraph":
        mem = cls()
        if not isinstance(payload, dict):
            return mem
        for key, rows in (payload.get("states") or {}).items():
            edges = {}
            for row in rows or []:
                if not isinstance(row, (list, tuple)) or len(row) < 3:
                    continue
                try:
                    count = int(row[3])
                except (IndexError, TypeError, ValueError):
                    count = 1
                edges[str(row[0])] = {"next": str(row[1]),
                                      "outcome": str(row[2]), "count": count}
            if edges:
                mem.states[str(key)] = edges
        mem.dead_ends = [str(x) for x in (payload.get("dead_ends") or [])]
        for k, v in (payload.get("stats") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mem.stats[str(k)] = int(v)
        return mem

    def save(self, path: Path | None) -> None:
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(self.to_payload(), indent=1),
                           encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path | None) -> "CompactStateGraph":
        if path is None or not path.is_file():
            return cls()
        try:
            return cls.from_payload(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return cls()

    def __len__(self) -> int:
        return sum(len(e) for e in self.states.values())


# --------------------------------------------- sandbox-side view (a preamble)
_PREAMBLE = r"""
# The sandbox strips __build_class__ from its builtins, so `class` would raise
# NameError here. Functions are fine; the view is assembled with type().
def _k5_fnv(self, text, level=None):
    lv = self._level if level is None else int(level)
    h = (0xcbf29ce484222325 ^ ((lv & 0xFFFF) * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
    for b in str(text).encode("utf-8", "replace"):
        h ^= b
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def _k5_key(self, action):
    if isinstance(action, dict):
        parts = [str(action.get("action") or "").strip().upper() or "?"]
        if action.get("row") is not None:
            parts.append("R" + str(action.get("row")).strip())
        if action.get("col") is not None:
            parts.append("C" + str(action.get("col")).strip())
        return "|".join(parts)
    return str(action or "").strip().upper()


def _k5_init(self, payload, level):
    payload = payload if isinstance(payload, dict) else {}
    self._level = int(level or 1)
    self._states = {}
    for key, rows in (payload.get("states") or {}).items():
        for row in rows or []:
            if isinstance(row, (list, tuple)) and len(row) >= 3:
                self._states.setdefault(str(key), {})[str(row[0])] = (
                    str(row[1]), str(row[2]),
                    int(row[3]) if len(row) > 3 else 1)
    self._dead = set(str(x) for x in (payload.get("dead_ends") or []))


def _k5_tried(self, h):
    out = []
    for k, v in self._states.get(str(h), {}).items():
        out.append({"action": k, "next": v[0], "outcome": v[1], "count": v[2]})
    return out


def _k5_is_pruned(self, h, action):
    key = str(h)
    rec = self._states.get(key, {}).get(self._key(action))
    if rec is None:
        return False
    return (rec[1] in ("no_op", "dead_end", "loop")
            or rec[0] in self._dead or rec[0] == key)


def _k5_safe(self, h, valid_actions):
    return [a for a in (valid_actions or []) if not self.is_pruned(h, a)]


def _k5_len(self):
    n = 0
    for v in self._states.values():
        n += len(v)
    return n


def _k5_repr(self):
    return "MemoryView(states=%d, edges=%d)" % (len(self._states), len(self))


_K5View = type("_K5MemoryView", (), {
    "__init__": _k5_init,
    "_key": _k5_key,
    "hash_grid": _k5_fnv,
    "tried": _k5_tried,
    "is_pruned": _k5_is_pruned,
    "safe_actions": _k5_safe,
    "__len__": _k5_len,
    "__repr__": _k5_repr,
})


def _k5_attach():
    global memory
    _view = _K5View(_K5_MEMORY, current_frame.level if current_frame is not None else 1)
    if current_frame is not None:
        current_frame.memory = _view
    memory = _view


_k5_attach()

_k5_original_action = action


def _k5_action(actions):
    _r = _k5_original_action(actions)
    _k5_attach()
    return _r


action = _k5_action
"""


# ------------------------------------------------------------ runtime patch
class _Runtime:
    def __init__(self, gate: bool) -> None:
        self.gate = bool(gate)
        self.memories: dict[str, CompactStateGraph] = {}
        self.state_path = None
        self.installed = False
        self.patched_module = None
        self.events: list[str] = []

    # memory is per game directory
    def memory_for(self, state_path) -> tuple[CompactStateGraph, Path | None]:
        if state_path is None:
            return CompactStateGraph(), None
        d = Path(state_path).parent
        key = str(d)
        sidecar = d / MEMORY_FILENAME
        if key not in self.memories:
            self.memories[key] = CompactStateGraph.load(sidecar)
        return self.memories[key], sidecar


_RUNTIME: _Runtime | None = None


def _patch_module(ta, runtime: _Runtime) -> None:
    missing = [n for n in ("ToolAgent", "run_sandboxed_python")
               if not hasattr(ta, n)]
    if missing:
        runtime.events.append("NOT PATCHED: %s is missing %s"
                              % (ta.__name__, ", ".join(missing)))
        return

    # 1) which game directory is live
    original_run = ta.ToolAgent._run_python_tool

    def _run_python_tool(self, state_path, arguments):
        previous = runtime.state_path
        runtime.state_path = state_path
        try:
            return original_run(self, state_path, arguments)
        finally:
            runtime.state_path = previous

    ta.ToolAgent._run_python_tool = _run_python_tool

    # 2) the gate + the recorder, around the sandbox call
    original_sandbox = ta.run_sandboxed_python

    def run_sandboxed_python(*, code, timeout_seconds, initial_state,
                             action_handler):
        memory, sidecar = runtime.memory_for(runtime.state_path)
        payload = memory.to_payload()
        state = dict(initial_state or {})
        state["memory"] = payload
        level = _level_of(state.get("current_frame"))
        before = _ascii_text(state.get("current_frame"))
        valid = list(state.get("valid_actions") or [])
        preamble = ("_K5_MEMORY = %s\n" % json.dumps(payload)) + _PREAMBLE

        def handler(actions):
            nonlocal before, valid, level, state
            if runtime.gate and before and valid:
                refusal = memory.gate(before, level, actions, valid)
                if refusal is not None:
                    blocked = ", ".join(str(b) for b in refusal["blocked"])
                    detail = (
                        "Episodic memory pruned %s: from this exact board those "
                        "actions are known no-ops, dead ends, or loops. Safe "
                        "actions now: %s. Already recorded here: %s. Pick from "
                        "safe_actions and continue in this same tool call."
                        % (blocked,
                           ", ".join(str(a) for a in refusal["safe_actions"]) or "none",
                           "; ".join("%s->%s" % (t["action"], t["outcome"])
                                     for t in refusal["tried"]) or "none"))
                    result = {
                        "executed": False,
                        "action_num": (state.get("last_action_result") or {}).get("action_num"),
                        "level": level, "score": None, "reward": 0.0,
                        "state": None, "valid_actions": list(valid),
                        "board_changed": False, "done": False,
                        "level_completed": False, "game_over": False,
                        "run_complete": False,
                        "requested_count": len(actions or []),
                        "executed_count": 0, "stopped_early": True,
                        "stop_reason": "k5_pruned", "stop_detail": detail,
                        "k5_pruned": {"blocked": list(refusal["blocked"]),
                                      "safe_actions": list(refusal["safe_actions"]),
                                      "tried": list(refusal["tried"])},
                    }
                    state["last_action_result"] = result
                    state["memory"] = memory.to_payload()
                    return {"action_result": result, "state": state}

            out = action_handler(actions)
            if not isinstance(out, dict):
                return out
            result = out.get("action_result") or {}
            new_state = out.get("state")
            after = _ascii_text((new_state or {}).get("current_frame"))
            if (len(actions or []) == 1 and before and after
                    and isinstance(result, dict)
                    and result.get("executed", True) is not False):
                memory.observe(before, level, actions[0], result, after)
                memory.save(sidecar)
            if after:
                before = after
                level = _level_of((new_state or {}).get("current_frame"))
            if isinstance(new_state, dict):
                if isinstance(new_state.get("valid_actions"), list):
                    valid = list(new_state["valid_actions"])
                new_state["memory"] = memory.to_payload()
                state = new_state
            return out

        return original_sandbox(code=preamble + code,
                                timeout_seconds=timeout_seconds,
                                initial_state=state,
                                action_handler=handler)

    ta.run_sandboxed_python = run_sandboxed_python

    # 3) the prompt
    if hasattr(ta, "_build_system_prompt"):
        original_build = ta._build_system_prompt

        def _build_system_prompt(*args, **kwargs):
            return original_build(*args, **kwargs) + ADDENDUM

        ta._build_system_prompt = _build_system_prompt
    elif hasattr(ta, "STRUCTURED_RUNTIME_STATE_ADDENDUM"):
        ta.STRUCTURED_RUNTIME_STATE_ADDENDUM = (
            ta.STRUCTURED_RUNTIME_STATE_ADDENDUM + ADDENDUM)
    else:
        runtime.events.append("prompt addendum skipped: no known prompt hook")

    runtime.patched_module = ta.__name__
    runtime.installed = True
    runtime.events.append("patched %s (gate=%s)" % (ta.__name__, runtime.gate))
    # Visible in the notebook stream at the moment the harness loads, and
    # written to disk so the outcome shows up in the kernel's output files.
    print("[k5] lever 3: PATCHED %s (gate=%s) -- episodic memory is live"
          % (ta.__name__, runtime.gate))
    try:
        status = Path(__file__).resolve().parent / "lever3_status.json"
        status.write_text(json.dumps(
            {"installed": True, "module": ta.__name__, "gate": runtime.gate,
             "events": runtime.events}, indent=1), encoding="utf-8")
    except Exception:
        pass


class _WatchingLoader(importlib.abc.Loader):
    def __init__(self, inner, runtime):
        self._inner = inner
        self._runtime = runtime

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            _patch_module(module, self._runtime)
        except Exception:
            self._runtime.events.append(
                "PATCH FAILED:\n" + traceback.format_exc())


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, runtime):
        self._runtime = runtime

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET_MODULE:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _WatchingLoader(spec.loader, self._runtime)
        return spec


def install(gate: bool = True) -> _Runtime:
    """Install Lever-3. Safe to call before or after the harness is imported."""
    global _RUNTIME
    runtime = _Runtime(gate)
    _RUNTIME = runtime
    if TARGET_MODULE in sys.modules:
        _patch_module(sys.modules[TARGET_MODULE], runtime)
        runtime.events.append("harness was already imported; patched in place")
    else:
        for i, finder in enumerate(sys.meta_path):
            if type(finder).__name__ == "_Finder":
                sys.meta_path.pop(i)
                break
        sys.meta_path.insert(0, _Finder(runtime))
        runtime.events.append("watching for %s" % TARGET_MODULE)
    return runtime
