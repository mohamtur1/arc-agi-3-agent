#!/usr/bin/env python3
"""k5_gym.py — the k5 local filter: a tiny deterministic gym for the agent.

Five hand-built micro-games drive MyAgent through the exact frame/action
contract the harness uses (duck-typed frames; no arcengine needed — the
agent only ever getattr()s the fields it reads). Each game ends on WIN or
after the agent's own MAX_ACTIONS decisions, so the whole suite runs in
seconds on CPU.

THIS IS A FILTER, NEVER A SCORE. Pass/fail here means "not obviously
broken". It cannot and does not predict the hidden-game leaderboard —
the numbers it prints are smoke-test counters from five toy grids, and
they are labelled as such everywhere they are stored.

Determinism: no randomness anywhere (boards are formulas). One run is one
data point; the suite is never repeated-and-averaged.

Usage
  python3 k5_gym.py --agent program_synthesis_agent_v4.py [--knobs file.json]
                    [--json] [--require-levels N]
Exit code: 0 = PASS, 1 = FAIL, 2 = usage/IO error.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import types

import numpy as np

GAMES = ("rarest_click", "largest_click", "move_chase", "frozen_void",
         "transient_anim")
FILTER_VERSION = 1


# --------------------------------------------------------------------------
# frame plumbing (duck-typed; matches what the agent actually reads)
# --------------------------------------------------------------------------

def _ns(**kw):
    return types.SimpleNamespace(**kw)


def frame_of(game, board, state):
    # COPY the board: the agent keeps frames by reference (astype copy=False),
    # so mutating one array in place would alias history[-1] == current frame.
    # The real harness always ships fresh arrays; the gym must too.
    return _ns(game_id=game.game_id,
               frame=[np.array(board, dtype=np.int64, copy=True)],
               state=state,                       # plain strings; the agent
               levels_completed=game.levels,      #   reads .value via getattr
               win_levels=game.win_levels,        #   with a str() fallback
               available_actions=list(game.available),
               action_input=None, guid=None, full_reset=False)


def extract_action(act):
    """(aid, x, y) from whatever action object the agent returned."""
    aid = 0
    try:
        aid = int(getattr(act, "value", getattr(act, "id", 0)) or 0)
    except Exception:
        aid = 0
    data = getattr(act, "action_data", None)
    x = int(getattr(data, "x", 0) or 0) if data is not None else 0
    y = int(getattr(data, "y", 0) or 0) if data is not None else 0
    return aid, x, y


def _cell_of(x, y):
    return (max(0, min(7, y // 8)), max(0, min(7, x // 8)))


def _put(board, r, c, h, w, color):
    r0, c0 = r * 8, c * 8
    board[r0:r0 + h, c0:c0 + w] = color


# --------------------------------------------------------------------------
# the five micro-games
# --------------------------------------------------------------------------

class Base:
    available = (6, 1, 2, 3, 4, 5)
    win_levels = 2

    def __init__(self, game_id):
        self.game_id = game_id
        self.state = "NOT_PLAYED"
        self.levels = 0
        self.board = np.zeros((64, 64), dtype=np.int64)
        self._build_level()

    def _build_level(self):
        raise NotImplementedError

    def _advance(self):
        self.levels += 1
        if self.levels >= self.win_levels:
            self.state = "WIN"
        else:
            self.board[:] = 0
            self._build_level()

    def observe(self):
        return frame_of(self, self.board, self.state)

    def apply(self, aid, x, y):
        if self.state == "NOT_PLAYED":
            if aid == 0:
                self.state = "NOT_FINISHED"
            return
        if self.state != "NOT_FINISHED":
            return
        self._act(aid, x, y)

    def _act(self, aid, x, y):
        raise NotImplementedError


class RarestClick(Base):
    """Click the uniquely rare blue block; it vanishes and the level clears.
    Correct click == the agent's recolor-to-zero prediction exactly."""

    game_kind = "rarest_click"

    def _build_level(self):
        _put(self.board, 2, 2, 8, 8, 3)                     # green  blocks
        _put(self.board, 2, 5, 8, 8, 3)
        _put(self.board, 5, 2, 8, 8, 3)
        _put(self.board, 5, 5, 8, 8, 1 if self.levels == 0 else 2)  # the rare one

    def _act(self, aid, x, y):
        if aid != 6:
            return
        if _cell_of(x, y) == (5, 5):
            self.board[40:48, 40:48] = 0
            self._advance()


class LargestClick(Base):
    """Only clicking the 16x16 block works; the three 8x8 blocks are decoys.
    Exercises selector demotion (rarest/smallest fail, largest verified)."""

    game_kind = "largest_click"

    def _build_level(self):
        _put(self.board, 1, 1, 8, 8, 3)
        _put(self.board, 1, 4, 8, 8, 2)
        _put(self.board, 4, 1, 8, 8, 5)
        _put(self.board, 4, 4, 16, 16, 4 if self.levels == 0 else 6)

    def _act(self, aid, x, y):
        if aid != 6:
            return
        if 32 <= y < 48 and 32 <= x < 48:
            self.board[32:48, 32:48] = 0
            self._advance()


class MoveChase(Base):
    """Move the 8x8 player onto the 8x8 goal. ACTION1..4 = up/down/left/right.
    Exercises control probing, direction learning, move predictions, and
    cross-level replay (level 2 is the same rule at a new location)."""

    game_kind = "move_chase"
    available = (1, 2, 3, 4)
    win_levels = 2
    DELTAS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    def _build_level(self):
        self.p = (2, 2) if self.levels == 0 else (5, 1)   # player cell
        self.g = (6, 5) if self.levels == 0 else (1, 6)   # goal cell
        _put(self.board, self.g[0], self.g[1], 8, 8, 3)
        self._draw_player()

    def _draw_player(self):
        _put(self.board, self.p[0], self.p[1], 8, 8, 5)

    def _act(self, aid, x, y):
        d = self.DELTAS.get(aid)
        if d is None:
            return
        nr = max(0, min(7, self.p[0] + d[0]))
        nc = max(0, min(7, self.p[1] + d[1]))
        if (nr, nc) == self.p:
            return                                      # wall: NO_EFFECT
        self.board[:] = 0
        self.p = (nr, nc)
        if self.p == self.g:
            self._advance()
            return
        _put(self.board, self.g[0], self.g[1], 8, 8, 3)
        self._draw_player()


class FrozenVoid(Base):
    """Nothing ever changes. The never-hang regression detector: v1 burned
    71/80 actions on ACTION1 here; v2+ must rotate/blacklist and terminate."""

    game_kind = "frozen_void"
    available = (1, 2, 3, 4, 5)
    win_levels = 1

    def _build_level(self):
        _put(self.board, 3, 3, 4, 4, 7)

    def _act(self, aid, x, y):
        return


class TransientAnim(Base):
    """Wrong clicks flash a stripe for exactly one frame, then revert.
    Right click (the blue block) deletes it and clears the level. Teaches
    the filter that typed-transient evidence must not corrupt the rules."""

    game_kind = "transient_anim"

    def _build_level(self):
        self._flash = False
        self._flash_count = 0              # flashes are finite: after three
        _put(self.board, 2, 2, 8, 8, 3)    # the animation settles for good,
        _put(self.board, 2, 5, 8, 8, 3)    # so wrong clicks can never hide
        _put(self.board, 5, 3, 8, 8, 1)    # the settled world permanently

    def _act(self, aid, x, y):
        if self._flash:                                  # animation frame over:
            self.board[0, :16] = 0                       # revert to settled
            self._flash = False
        if aid != 6:
            return
        r, c = _cell_of(x, y)
        if (r, c) == (5, 3):
            self.board[40:48, 24:32] = 0
            self._advance()
        elif self._flash_count < 3:
            self._flash_count += 1
            self.board[0, :16] = 9                       # one-frame flash
            self._flash = True


SUITE = (RarestClick, LargestClick, MoveChase, FrozenVoid, TransientAnim)


# --------------------------------------------------------------------------
# driver + agent loading
# --------------------------------------------------------------------------

def load_agent(path, knobs_path=None):
    """Import the agent module by path, with the knob channel pointed at
    knobs_path (the same contract the Kaggle run uses). Fresh module object
    per call so variant runs never share GAME_MEMORY."""
    if knobs_path:
        os.environ["K5_KNOBS_JSON"] = knobs_path
    elif "K5_KNOBS_JSON" in os.environ:
        del os.environ["K5_KNOBS_JSON"]
    import tempfile
    os.environ["K5_TELEMETRY_DIR"] = tempfile.mkdtemp(prefix="k5gym_telem_")
    name = "k5_agent_" + hashlib.sha256(
        (os.path.abspath(path) + str(os.path.getmtime(path))
         + str(knobs_path)).encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def play(agent_mod, game, trace=None):
    """One game: observe -> is_done -> choose_action -> apply, capped at
    the agent's own MAX_ACTIONS decisions (mirrors the harness)."""
    MyAgent = agent_mod.MyAgent
    agent = MyAgent(game_id=game.game_id)
    decisions = 0
    slow = 0
    t_cap = time.perf_counter() + 60.0                   # per-game wall guard
    while True:
        frame = game.observe()
        if agent.is_done([frame], frame):
            break
        if decisions >= MyAgent.MAX_ACTIONS or time.perf_counter() > t_cap:
            break
        t0 = time.perf_counter()
        act = agent.choose_action([frame], frame)
        dt = time.perf_counter() - t0
        if dt > 5.0:
            return {"game": game.game_kind, "error": "decision exceeded 5s",
                    "levels": game.levels}
        if dt > MyAgent.TIME_BUDGET_S:
            slow += 1
        decisions += 1
        aid, x, y = extract_action(act)
        if trace is not None:
            trace.append(aid)
        game.apply(aid, x, y)
    return {"game": game.game_kind,
            "levels": int(game.levels),
            "win": 1 if game.state == "WIN" else 0,
            "decisions": decisions,
            "slow_decisions": slow}


def run_suite(agent_path, knobs_path=None):
    out = {"filter_version": FILTER_VERSION,
           "agent_path": os.path.basename(agent_path),
           "agent_sha256": None,
           "games": [], "levels": 0, "wins": 0, "errors": []}
    try:
        with open(agent_path, "rb") as fh:
            out["agent_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        t0 = time.perf_counter()
        mod = load_agent(agent_path, knobs_path)
        for cls in SUITE:
            trace = []
            res = play(mod, cls(cls.game_kind), trace)
            res["trace_sha256_12"] = hashlib.sha256(
                json.dumps(trace).encode()).hexdigest()[:12]
            out["games"].append(res)
            if "error" in res:
                out["errors"].append(res["error"])
            out["levels"] += int(res.get("levels", 0))
            out["wins"] += int(res.get("win", 0))
        out["wall_seconds"] = round(time.perf_counter() - t0, 3)
    except Exception as exc:                              # a crash IS a result
        out["errors"].append("suite crashed: %r" % (exc,))
    out["ok"] = not out["errors"]
    return out


def verdict(base, cand, require_levels=0):
    """FAIL only on 'obviously broken': crash/hang, any win lost, or a drop
    of 2+ levels across the five toys. Anything else PASSES the filter —
    small shuffles between toys are exactly what a filter must not read
    into. Never stated or stored as a leaderboard prediction."""
    reasons = []
    if not cand.get("ok"):
        reasons.append("candidate crashed or hung: %s" % cand.get("errors"))
    if cand.get("wins", 0) < base.get("wins", 0):
        reasons.append("candidate lost a game win (%d -> %d)"
                       % (base.get("wins"), cand.get("wins")))
    if cand.get("levels", 0) <= base.get("levels", 0) - 2:
        reasons.append("candidate lost 2+ toy levels (%d -> %d)"
                       % (base.get("levels"), cand.get("levels")))
    if cand.get("levels", 0) < require_levels:
        reasons.append("below --require-levels floor (%d < %d)"
                       % (cand.get("levels"), require_levels))
    return ("FAIL" if reasons else "PASS"), reasons


def _print(result):
    print("  games:")
    for g in result.get("games", []):
        print("    %-15s levels=%d win=%d decisions=%d slow=%d trace=%s %s"
              % (g["game"], g.get("levels", 0), g.get("win", 0),
                 g.get("decisions", 0), g.get("slow_decisions", 0),
                 g.get("trace_sha256_12", "-"),
                 ("ERROR: " + g["error"]) if "error" in g else ""))
    print("  totals: levels=%d wins=%d wall=%ss errors=%s"
          % (result.get("levels", 0), result.get("wins", 0),
             result.get("wall_seconds", "?"), result.get("errors") or "none"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--agent", required=True, help="path to the agent .py")
    ap.add_argument("--knobs", default=None,
                    help="k5_knobs.json for the variant; omit for defaults")
    ap.add_argument("--base-knobs", default=None,
                    help="also run the incumbent from this JSON and compare")
    ap.add_argument("--require-levels", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    a = ap.parse_args()

    if not os.path.isfile(a.agent):
        print("agent not found: %s" % a.agent, file=sys.stderr)
        return 2

    cand = run_suite(a.agent, a.knobs)
    base = run_suite(a.agent, a.base_knobs) if a.base_knobs else None
    result = {"mode": "compare" if base else "single",
              "candidate": cand, "incumbent": base}
    if base:
        v, reasons = verdict(base, cand, a.require_levels)
        result["verdict"] = v
        result["verdict_reasons"] = reasons
    else:
        v, reasons = ("PASS" if cand["ok"]
                      and cand["levels"] >= a.require_levels else "FAIL"), \
                     ([] if cand["ok"] else cand["errors"])
        result["verdict"] = v
        result["verdict_reasons"] = reasons

    if a.json:
        print(json.dumps(result, indent=1, sort_keys=True))
    else:
        print("k5 gym (filter v%d) — a smoke test, NEVER a leaderboard score"
              % FILTER_VERSION)
        if base:
            print("incumbent:"); _print(base)
            print("candidate:"); _print(cand)
        else:
            _print(cand)
        print("verdict: %s %s"
              % (result["verdict"],
                 ("; ".join(result["verdict_reasons"])) or ""))
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
