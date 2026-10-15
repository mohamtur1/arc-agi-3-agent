"""program_synthesis_agent_v4.py — self-contained ARC-AGI-3 agent (v2.2)

v2.2 (k5 between-game learning loop) adds exactly two things over v2.1:
  K1 knob channel: module constants below can be overridden from a JSON file
     the daily driver writes before this module is imported
     (K5_KNOBS_JSON env var, else /kaggle/working/k5_knobs.json).
     No file = the v2.1 defaults apply = behaviour identical to v2.1.
  K2 telemetry: per-game outcome rows appended to k5_per_game.csv next to
     the run's output (K5_TELEMETRY_DIR env var, else /kaggle/working),
     plus a one-off k5_run_info.json self-report. Telemetry is wrapped so
     it can never break gameplay; it changes no decisions.

Carried over from v2.1:
  F4 state-action exploration graph (novelty-seeking when no hypothesis)
  F5 typed evidence: transient/revert detection (settled vs animation)
Carried over from v2.0:
  F1 frame diffing + no-op blacklist    F2 cross-level verified replay
  F3 level-1-first exploration
Carried over from v1:
  Protocol layer, perception, micro-DSL, hypothesis bank, never-hang wrapper.

Contract: class MyAgent with choose_action(frames, latest_frame) and
is_done(frames, latest_frame). numpy + stdlib only; no network.
"""
import copy
import time
from collections import deque

import numpy as np

# ============================================================================
# SECTION 1 — PROTOCOL LAYER (unchanged from v2)
# ============================================================================
GRID = 64
CELL = 8

try:
    from arcengine import FrameData as _RealFrameData  # noqa: F401
    from arcengine import GameAction, GameState
    from arcengine.enums import (SimpleAction, ComplexAction, FrameData,
                                 FrameDataRaw, ActionInput)  # noqa: F401
    ARCENGINE_AVAILABLE = True
except Exception:
    ARCENGINE_AVAILABLE = False

    from enum import Enum

    class GameState(str, Enum):
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class SimpleAction:
        def __init__(self, game_id: str = ""):
            self.game_id = game_id

        def model_dump(self):
            return {"game_id": self.game_id}

    class ComplexAction:
        def __init__(self, game_id: str = "", x: int = 0, y: int = 0):
            if not (0 <= x <= 63 and 0 <= y <= 63):
                raise ValueError("x and y must be within [0, 63]")
            self.game_id = game_id
            self.x = x
            self.y = y

        def model_dump(self):
            return {"game_id": self.game_id, "x": self.x, "y": self.y}

    class GameAction(Enum):
        RESET = (0, SimpleAction)
        ACTION1 = (1, SimpleAction)
        ACTION2 = (2, SimpleAction)
        ACTION3 = (3, SimpleAction)
        ACTION4 = (4, SimpleAction)
        ACTION5 = (5, SimpleAction)
        ACTION6 = (6, ComplexAction)
        ACTION7 = (7, SimpleAction)

        def __init__(self, action_id, action_type):
            self._value_ = action_id
            self._atype = action_type
            self.action_data = action_type()

        def is_simple(self):
            return self._atype is SimpleAction

        def is_complex(self):
            return self._atype is ComplexAction

        def set_data(self, data):
            self.action_data = self._atype(**data)
            return self.action_data

        @classmethod
        def from_id(cls, action_id):
            for a in cls:
                if a.value == action_id:
                    return a
            raise ValueError(f"No GameAction with id {action_id}")

    class ActionInput:
        def __init__(self, id=GameAction.RESET, data=None, reasoning=None):
            self.id = id
            self.data = data or {}
            self.reasoning = reasoning

    class FrameData:
        def __init__(self, game_id="", frame=None, state=GameState.NOT_PLAYED,
                     levels_completed=0, win_levels=0, action_input=None,
                     guid=None, full_reset=False, available_actions=None):
            self.game_id = game_id
            self.frame = frame or []
            self.state = state
            self.levels_completed = levels_completed
            self.win_levels = win_levels
            self.action_input = action_input or ActionInput()
            self.guid = guid
            self.full_reset = full_reset
            self.available_actions = list(available_actions or [])

        def is_empty(self):
            return len(self.frame) == 0


def _pick_frame(frames, latest_frame):
    """Prefer the freshest frame that actually carries grid data.

    The kit's main() seeds `frames` with a dummy FrameData and passes the
    converted current observation as latest_frame, so latest_frame wins
    whenever it has pixels; the history covers callers that pass a stale
    or empty second argument but real frames in `frames`.
    """
    try:
        if getattr(latest_frame, "frame", None):
            return latest_frame
    except Exception:
        pass
    try:
        for cand in reversed(list(frames or [])):
            if getattr(cand, "frame", None):
                return cand
    except Exception:
        pass
    return latest_frame


def _state_name(state):
    try:
        return str(getattr(state, "value", state)).split(".")[-1]
    except Exception:
        return ""


def _grid_of(frame):
    try:
        layers = getattr(frame, "frame", None)
        if not layers:
            layers = getattr(frame, "_frame", None)
        if not layers:
            return None
        arr = np.asarray(layers[0])
        if arr.ndim != 2:
            return None
        return arr.astype(np.int64, copy=False)
    except Exception:
        return None


def merged_view(frame):
    try:
        layers = getattr(frame, "frame", None) or []
        if not layers:
            return None
        base = np.asarray(layers[0]).astype(np.int64, copy=False)
        for extra in layers[1:]:
            arr = np.asarray(extra)
            if arr.shape == base.shape:
                base = np.where(arr != 0, arr, base)
        return base
    except Exception:
        return None


# ============================================================================
# SECTION 2 — PERCEPTION (unchanged from v2)
# ============================================================================
def find_objects(grid):
    objects = []
    visited = set()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0 or (r, c) in visited:
                continue
            color = int(grid[r, c])
            obj_id = len(objects) + 1
            q = deque([(r, c)])
            visited.add((r, c))
            pixels = []
            min_r, max_r, min_c, max_c = r, r, c, c
            while q:
                cr, cc = q.popleft()
                pixels.append((cr, cc))
                min_r, max_r = min(min_r, cr), max(max_r, cr)
                min_c, max_c = min(min_c, cc), max(max_c, cc)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if (0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited
                            and grid[nr, nc] == color):
                        visited.add((nr, nc))
                        q.append((nr, nc))
            cy = sum(p[0] for p in pixels) / len(pixels)
            cx = sum(p[1] for p in pixels) / len(pixels)
            objects.append({"id": obj_id, "color": color,
                            "pixels": frozenset(pixels),
                            "bbox": (min_r, min_c, max_r, max_c),
                            "size": len(pixels), "centroid": (cy, cx)})
    return objects


def frame_diff(grid_a, grid_b):
    if grid_a is None or grid_b is None or grid_a.shape != grid_b.shape:
        return np.ones_like(np.asarray(grid_b), dtype=bool)
    return grid_a != grid_b


def salience_rank(objects):
    color_count = {}
    for o in objects:
        color_count[o["color"]] = color_count.get(o["color"], 0) + 1
    ranked = sorted(objects,
                    key=lambda o: (color_count[o["color"]], o["size"], o["bbox"]))
    return ranked, color_count


# ============================================================================
# SECTION 3 — MICRO-DSL (unchanged from v2)
# ============================================================================
def prim_recolor_region(grid, mask, new_color):
    g = copy.deepcopy(grid)
    g[mask] = new_color
    return g


def prim_move_region(grid, mask, dr, dc):
    g = copy.deepcopy(grid)
    color = int(g[mask][0]) if mask.any() else 0
    g[mask] = 0
    rows = np.nonzero(mask)[0] + dr
    cols = np.nonzero(mask)[1] + dc
    keep = ((rows >= 0) & (rows < g.shape[0])
            & (cols >= 0) & (cols < g.shape[1]))
    g[rows[keep], cols[keep]] = color
    return g


# ============================================================================
# SECTION 4 — GOVERNANCE (UL-derived, unchanged from v2)
# ============================================================================
EFFECT_NO_EFFECT = "NO_EFFECT"
EFFECT_LOCAL = "LOCAL_CHANGE"
EFFECT_GLOBAL = "GLOBAL_CHANGE"
EFFECT_LEVEL = "LEVEL_ADVANCED"
EFFECT_TRANSIENT = "TRANSIENT_REVERT"   # v2.1 new

EVIDENCE_VERIFIED = "VERIFIED"
EVIDENCE_CORROBORATED = "CORROBORATED"

STATUS_ACTIVE = "ACTIVE"
STATUS_STALE = "STALE"
STATUS_QUARANTINED = "QUARANTINED"

BLACKLIST_CAP = 64
SEQ_STORE_CAP = 60
GLOBAL_CHANGE_FRACTION = 0.25

# v2.1 state-graph constants
GRAPH_HASH_STRIDE = 4               # downsample grid for hashing
GRAPH_RECENT_CAP = 12               # history of recent distinct states
GRAPH_TRANSIENT_MAX_DIFF = 32       # pixel diff threshold for transient
EXPLORATION_CAP = 12                # consecutive no-progress explore actions
                                    # (design §4.3: worst case = v2.0 minus
                                    #  at most EXPLORATION_CAP actions)


class EpisodeLedger:
    def __init__(self):
        self.events = []

    def append(self, kind, **payload):
        eid = len(self.events)
        self.events.append(dict(id=eid, kind=kind, **payload))
        return eid

    def __len__(self):
        return len(self.events)


class RuleRecord:
    def __init__(self, rule_id, kind):
        self.rule_id = rule_id
        self.kind = kind
        self.evidence_class = None
        self.status = STATUS_ACTIVE
        self.success = 0
        self.fail = 0
        self.rotations = 0
        self.evidence = []

    def cite(self, eid):
        self.evidence.append(eid)


# ============================================================================
# SECTION 4.5 — STATE-ACTION EXPLORATION GRAPH (v2.1 new)
# ============================================================================
class StateGraph:
    """Novelty-seeking state-action graph over settled frames.

    - States are hashed via a downsampled grid to filter single-pixel noise.
    - `recent` keeps distinct consecutive state hashes (auto-capped).
    - `visits` counts how many times each state was observed.
    - `transitions` records (from_state -> action_key -> to_state).
    - `action_uses` counts how many transitions each action participated in.

    Transient detection: a new state is flagged as transient (revert) iff
    it matches the state two distinct-changes ago AND the frame diff is
    small (< GRAPH_TRANSIENT_MAX_DIFF pixels). Small reverts are animation.
    """

    def __init__(self):
        self.recent = deque(maxlen=GRAPH_RECENT_CAP)
        self.visits = {}
        self.transitions = {}
        self.action_uses = {}

    @staticmethod
    def hash_grid(grid):
        if grid is None:
            return None
        try:
            arr = np.asarray(grid)
            if arr.ndim != 2:
                return None
            small = arr[::GRAPH_HASH_STRIDE, ::GRAPH_HASH_STRIDE]
            return hash(small.tobytes())
        except Exception:
            return None

    def is_revert(self, h, diff_pixels):
        """True if h matches the state two changes ago AND diff is small."""
        if h is None or diff_pixels > GRAPH_TRANSIENT_MAX_DIFF:
            return False
        if len(self.recent) < 2:
            return False
        return self.recent[-2] == h

    def observe(self, h):
        """Record this frame's hash. Idempotent for consecutive repeats."""
        if h is None:
            return
        if self.recent and self.recent[-1] == h:
            return
        self.recent.append(h)
        self.visits[h] = self.visits.get(h, 0) + 1

    def record_transition(self, from_h, key, to_h):
        if from_h is None or to_h is None or key is None:
            return
        self.transitions.setdefault(from_h, {})[key] = to_h
        aid = key if isinstance(key, int) else key[0]
        self.action_uses[aid] = self.action_uses.get(aid, 0) + 1

    def novelty_action(self, current_h, avail):
        """Pick a novelty-seeking action at the current state.
        Priority: untried simple action (least-used globally), then untried
        click cell (if 6 in avail), then action leading to least-visited
        state, else None."""
        if not avail:
            return None
        tried = self.transitions.get(current_h, {}) if current_h is not None else {}
        # 1) untried simple actions
        simple_untried = [a for a in avail if a != 6 and a not in tried]
        if simple_untried:
            return min(simple_untried,
                       key=lambda a: self.action_uses.get(a, 0))
        # 2) untried click cells (only if 6 is available)
        if 6 in avail:
            tried_cells = {k[1:] for k in tried
                           if isinstance(k, tuple) and k[0] == 6}
            for r in range(8):
                for c in range(8):
                    if (r, c) not in tried_cells:
                        return (6, r, c)
        # 3) least-visited successor (FIX audit B2: self-loops are dead
        # ends — a no-effect pair must never win the novelty pick)
        candidates = [(k, v) for k, v in tried.items() if v != current_h]
        if candidates:
            return min(candidates,
                       key=lambda p: self.visits.get(p[1], 0))[0]
        return None

    def clear_recent(self):
        self.recent.clear()


class GameKnowledge:
    def __init__(self, game_id):
        self.game_id = game_id
        self.rules = {}
        self.move = None
        self.sequences = {}
        self.graph = StateGraph()   # v2.1 persistent graph

    def rule(self, name):
        if name not in self.rules:
            self.rules[name] = RuleRecord(name, "click")
        return self.rules[name]

    def best_selector(self, order):
        def key(sel):
            rec = self.rules.get(sel)
            if rec is None:
                return (2, 0, 0)
            if rec.status == STATUS_QUARANTINED:
                return (9, 0, 0)
            if rec.status == STATUS_STALE:
                return (1, 0, 0)
            rank = {EVIDENCE_VERIFIED: 0, EVIDENCE_CORROBORATED: 1, None: 2}
            return (0, rank.get(rec.evidence_class, 2), -rec.success)
        return sorted(order, key=key)


def classify_effect(prev, cur, levels_before, levels_after, transient=False):
    if levels_after > levels_before:
        return EFFECT_LEVEL, 0
    d = frame_diff(prev, cur)
    n = int(d.sum())
    if n == 0:
        return EFFECT_NO_EFFECT, 0
    if transient:
        return EFFECT_TRANSIENT, n
    if n > GLOBAL_CHANGE_FRACTION * prev.size or n > 1024:
        return EFFECT_GLOBAL, n
    return EFFECT_LOCAL, n


# ============================================================================
# SECTION 5 — PROGRAM SYNTHESIS CORE v2.1
# ============================================================================
CLICK_SELECTORS = ["rarest_color", "smallest_object",
                   "corner_sample_color", "largest_object"]
PROBE_ATTEMPT_CAP = 3
REPLAY_ABORT_STALE = 2

# --------------------------------------------------------------------------
# K1 (v2.2 new) — k5 between-game learning loop: external knob overrides.
#
# The daily driver writes k5_knobs.json before this module is imported:
#   {"variant": "exp-001", "knobs": {"EXPLORATION_CAP": 24, ...}, ...}
# Only the names listed in K5_KNOB_NAMES may be overridden, and only with
# values of the matching type. GRID and CELL are structural and can never be
# overridden. With no file present (or any read failure) the module is
# byte-for-byte equivalent to v2.1 behaviour. K5_RUN_INFO records what was
# actually applied so a run can always be traced back to a ledger row.
# --------------------------------------------------------------------------
K5_KNOB_NAMES = (
    "BLACKLIST_CAP", "SEQ_STORE_CAP", "GLOBAL_CHANGE_FRACTION",
    "GRAPH_HASH_STRIDE", "GRAPH_RECENT_CAP", "GRAPH_TRANSIENT_MAX_DIFF",
    "EXPLORATION_CAP", "CLICK_SELECTORS", "PROBE_ATTEMPT_CAP",
    "REPLAY_ABORT_STALE",
)

K5_RUN_INFO = {"variant": None, "experiment": None, "reason": None,
               "applied_overrides": {}, "agent_sha256": None,
               "knobs": {}}


def _k5_self_sha256():
    import hashlib
    try:
        with open(__file__, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return None


def _k5_apply_knob_overrides():
    """Read the driver's JSON (if any) and rebind the knob constants.

    Runs at import time, before any class is built or game starts, so every
    consumer — definition-time (deque maxlen) and runtime (global lookups) —
    sees the same values. Never raises; returns the applied dict.
    """
    import json as _json
    import os as _os
    path = _os.environ.get("K5_KNOBS_JSON") or "/kaggle/working/k5_knobs.json"
    try:
        with open(path) as fh:
            payload = _json.load(fh)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    overrides = payload.get("knobs") if "knobs" in payload else payload
    if not isinstance(overrides, dict):
        overrides = {}
    g = globals()
    applied = {}
    for name in K5_KNOB_NAMES:
        if name not in overrides or overrides[name] is None:
            continue
        cur = g[name]
        val = overrides[name]
        try:
            if isinstance(cur, list):
                if not (isinstance(val, list) and val
                        and all(isinstance(v, str) for v in val)):
                    continue
                g[name] = list(val)
            elif isinstance(cur, bool):
                continue                      # no bool knobs; refuse coercion
            elif isinstance(cur, int):
                g[name] = int(val)
            elif isinstance(cur, float):
                g[name] = float(val)
            else:
                continue
        except (TypeError, ValueError):
            continue
        applied[name] = g[name]
    K5_RUN_INFO["variant"] = payload.get("variant")
    K5_RUN_INFO["experiment"] = payload.get("experiment")
    K5_RUN_INFO["reason"] = payload.get("reason")
    K5_RUN_INFO["agent_sha256"] = _k5_self_sha256()
    return applied


_K5_KNOBS_APPLIED = _k5_apply_knob_overrides()
K5_RUN_INFO["applied_overrides"] = dict(_K5_KNOBS_APPLIED)
K5_RUN_INFO["knobs"] = {n: globals()[n] for n in K5_KNOB_NAMES}
# Record this file's hash even when no knob file exists (baseline runs are
# the runs where provenance matters most).
if not K5_RUN_INFO.get("agent_sha256"):
    K5_RUN_INFO["agent_sha256"] = _k5_self_sha256()
if _K5_KNOBS_APPLIED:
    print("[k5] knob overrides applied at import: %s" % _K5_KNOBS_APPLIED)


def _action_key(pend):
    """Hashable key identifying the pending action for the state graph."""
    if not pend:
        return None
    aid = pend.get("aid")
    if aid is None:
        return None
    if aid == 6:
        data = pend.get("data") or {}
        x = int(data.get("x", 0))
        y = int(data.get("y", 0))
        return (6, y // CELL, x // CELL)
    return aid


class ProgramSynthesisV2Core:

    GAME_MEMORY = {}

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._reset_episode()

    def _reset_episode(self):
        self.game_id = ""
        self.ledger = EpisodeLedger()
        self.history = []
        self.pending = None
        self.level = 0
        self.mode = None
        self.sel_order = list(CLICK_SELECTORS)
        self.queue = []
        self.queue_ptr = 0
        self.clicked_sig = set()
        self.blacklist = set()
        self.dirs = {}
        self.probe_attempts = {}
        self.player_color = None
        self.stuck = 0
        self.replay = None
        self.level_actions = []
        self.level_verified = False
        # v2.1 exploration budget (design §4.3)
        self.explore_streak = 0
        self.explore_blocked = False
        self._explore_rr = 0
        # v2.1: transient graph, replaced in new_episode with persistent one
        self.graph = StateGraph()

    def new_episode(self, game_id):
        know = ProgramSynthesisV2Core.GAME_MEMORY.get(game_id)
        self._reset_episode()
        self.game_id = game_id or ""
        if know is None:
            know = GameKnowledge(game_id)
            ProgramSynthesisV2Core.GAME_MEMORY[game_id] = know
        self.knowledge = know
        self.graph = know.graph                     # v2.1 persistent
        if know.move:
            self.dirs = dict(know.move.get("dirs") or {})
            self.player_color = know.move.get("player_color")
        self.sel_order = know.best_selector(list(CLICK_SELECTORS))
        self.ledger.append("EPISODE_START", game_id=self.game_id)

    # ------------------------------------------------------------- main entry
    def next_action(self, frame):
        gid = ""
        try:
            gid = getattr(frame, "game_id", "") or ""
        except Exception:
            pass
        if gid and gid != self.game_id:
            self.new_episode(gid)

        grid = self._grid(frame)
        if grid is None:
            return 0, {"game_id": self.game_id}, {"note": "no frame data"}

        if self.history:
            try:
                self._process_feedback(self.history[-1], grid, frame)
            except Exception:
                self.pending = None
        self.history.append(grid)

        state = _state_name(getattr(frame, "state", None))
        if state in ("NOT_PLAYED", "GAME_OVER"):
            self._clear_level_state()
            self.ledger.append("RESET_ISSUED", why=state)
            return 0, {"game_id": self.game_id}, {
                "strategy": "reset", "why": state,
                "hypothesis": self.sel_order[0] if self.sel_order else None}

        avail = self._available(frame)
        if not avail:
            return 0, {"game_id": self.game_id}, {"note": "no available actions"}
        self.mode = "click" if 6 in avail else "move"

        if self.replay is None and self.mode:
            seqrec = self.knowledge.sequences.get(self.level)
            if seqrec and not seqrec.get("stale"):
                self.replay = {"actions": seqrec["actions"], "idx": 0}
                self.ledger.append("REPLAY_STARTED", level=self.level,
                                   evidence_class=seqrec["evidence_class"],
                                   n_actions=len(seqrec["actions"]))
                if self.verbose:
                    print(f"    [v3] replay level {self.level}: "
                          f"{len(seqrec['actions'])} actions")
        if self.replay is not None:
            return self._replay_step(grid, frame, avail)

        if self.mode == "click":
            return self._click_policy(grid, frame, avail)
        return self._move_policy(grid, frame, avail)

    # ------------------------------------------------------------- feedback
    def _process_feedback(self, prev, cur, frame):
        levels_now = int(getattr(frame, "levels_completed", 0) or 0)
        pend = self.pending or {}
        self.pending = None
        aid = pend.get("aid")

        # v2.1 state graph: hash both frames
        h_prev = self.graph.hash_grid(prev)
        h_cur = self.graph.hash_grid(cur)
        diff_px = int(frame_diff(prev, cur).sum())

        # v2.1 transient detection: revert to state-2-ago with small diff
        is_revert = (h_cur is not None and h_prev != h_cur
                     and self.graph.is_revert(h_cur, diff_px))

        # v2.1 observe the new state (records visit + recent)
        if h_cur is not None:
            self.graph.observe(h_cur)

        effect, npx = classify_effect(prev, cur, self.level, levels_now,
                                      transient=is_revert)
        self.ledger.append("EFFECT_CLASSIFIED", action=aid, effect=effect,
                           changed=npx, level=levels_now,
                           replay=bool(pend.get("replay")))

        # v2.1 record transition (settled only; transients never recorded).
        # FIX (audit B2): NO_EFFECT pairs are recorded as SELF-LOOPS so the
        # graph counts them as tried — otherwise novelty_action re-proposes
        # the same dead action forever (the v1 _pending bug reborn: a frozen
        # game burned 71/80 actions on ACTION1).
        if (h_prev is not None and h_cur is not None
                and effect != EFFECT_TRANSIENT):
            key = _action_key(pend)
            if key is not None and (h_prev != h_cur
                                    or effect == EFFECT_NO_EFFECT):
                self.graph.record_transition(h_prev, key, h_cur)

        # v2.1 transients don't update rules; they're animation noise
        if effect == EFFECT_TRANSIENT:
            return

        # prediction validation
        verified = False
        predicted = pend.get("predicted")
        if predicted is not None and effect in (EFFECT_LOCAL, EFFECT_GLOBAL):
            ok = float((predicted == cur).mean()) if predicted.shape == cur.shape else 0.0
            if ok >= 0.97:
                verified = True

        if aid is not None:
            self.level_actions.append({
                "aid": aid, "data": pend.get("data") or {},
                "effect": effect, "verified": verified})
            if verified:
                self.level_verified = True
        if verified:
            self.explore_streak = 0   # progress: exploration budget resets

        if pend.get("replay"):
            expected = pend.get("expected")
            if effect != expected and effect != EFFECT_LEVEL:
                self._replay_abort(cause=f"expected {expected}, saw {effect}")
                return
            if effect == EFFECT_LEVEL:
                self.replay = None
                self.ledger.append("REPLAY_STEP_DONE", effect=effect)
                self._on_level_advanced(levels_now, pend)
                return
            self.ledger.append("REPLAY_STEP_OK", effect=effect)
            return

        if effect == EFFECT_LEVEL:
            self._on_level_advanced(levels_now, pend)
            return

        kind = pend.get("kind")
        if effect == EFFECT_NO_EFFECT:
            self._on_no_effect(pend)
            if kind == "probe":
                self._learn_direction(prev, cur, aid)
        elif kind == "click":
            self._on_click_feedback(pend, effect, verified)
        elif kind == "probe":
            self._learn_direction(prev, cur, aid)
        elif kind == "move":
            self._on_move_feedback(pend, prev, cur, aid, verified)
        # v2.1: "explore" actions update the graph but no rules

    def _on_level_advanced(self, levels_now, pend):
        seq = [a for a in self.level_actions if a["effect"] != EFFECT_NO_EFFECT]
        if 0 < len(self.level_actions) <= SEQ_STORE_CAP and seq:
            self.knowledge.sequences[self.level] = {
                "actions": seq,
                "evidence_class": (EVIDENCE_VERIFIED if self.level_verified
                                   else EVIDENCE_CORROBORATED),
                "aborts": 0, "stale": False}
            self.ledger.append("WIN_SEQUENCE_STORED", level=self.level,
                               n_actions=len(seq))
        rule_name = pend.get("rule")
        if self.mode == "move" and self.knowledge.move:
            rec = self.knowledge.move["record"]
            rec.success += 1
            if rec.evidence_class is None:
                rec.evidence_class = EVIDENCE_CORROBORATED
        elif rule_name:
            rec = self.knowledge.rule(rule_name)
            rec.success += 1
            if rec.evidence_class is None:
                rec.evidence_class = EVIDENCE_CORROBORATED
        self.ledger.append("LEVEL_ADVANCED", to=levels_now)
        self.level = levels_now
        self._clear_level_state()

    def _on_no_effect(self, pend):
        aid = pend.get("aid")
        kind = pend.get("kind")
        if kind == "click" and pend.get("sig"):
            self.clicked_sig.add(pend["sig"])
            if len(self.clicked_sig) > BLACKLIST_CAP:
                self.clicked_sig.pop()
        elif kind in ("simple", None) and self.mode != "move" and aid is not None:
            key = (self.level, aid)
            if key not in self.blacklist and len(self.blacklist) < BLACKLIST_CAP:
                self.blacklist.add(key)
                self.ledger.append("ACTION_BLACKLISTED", action=aid,
                                   level=self.level)
        if kind == "click" and pend.get("rule"):
            rec = self.knowledge.rule(pend["rule"])
            rec.fail += 1
        if kind == "move":
            self.stuck += 1

    def _on_click_feedback(self, pend, effect, verified):
        rule_name = pend.get("rule")
        if not rule_name:
            return
        rec = self.knowledge.rule(rule_name)
        if verified:
            rec.success += 1
            if rec.evidence_class != EVIDENCE_VERIFIED:
                rec.evidence_class = EVIDENCE_VERIFIED
                if self.verbose:
                    print(f"    [v3] VERIFIED '{rule_name}'")
            rec.cite(self.ledger.append("RULE_VERIFIED", rule=rule_name))
        elif pend.get("predicted") is not None:
            rec.fail += 1
            if rec.evidence_class == EVIDENCE_VERIFIED:
                rec.status = STATUS_STALE
                self.ledger.append("RULE_STALE", rule=rule_name,
                                   cause="prediction mismatch")
            self._demote(rule_name, why="prediction mismatch")
        else:
            rec.fail += 1

    def _on_move_feedback(self, pend, prev, cur, aid, verified):
        if self.knowledge.move is None:
            return
        rec = self.knowledge.move["record"]
        if verified:
            rec.success += 1
            if rec.evidence_class != EVIDENCE_VERIFIED:
                rec.evidence_class = EVIDENCE_VERIFIED
        self._learn_direction(prev, cur, aid)
        moved = bool(frame_diff(prev, cur).any())
        self.stuck = 0 if moved else self.stuck + 1

    def _learn_direction(self, prev, cur, aid):
        if aid is None or aid in self.dirs:
            return
        hole = (prev != 0) & (cur == 0)
        fill = (prev == 0) & (cur != 0)
        if hole.any() and fill.any():
            hr, hc = np.nonzero(hole)
            fr, fc = np.nonzero(fill)
            dr = int(round(fr.mean() - hr.mean()))
            dc = int(round(fc.mean() - hc.mean()))
            self.dirs[aid] = (int(np.sign(dr)) if abs(dr) > 1 else 0,
                              int(np.sign(dc)) if abs(dc) > 1 else 0)
            self.player_color = int(np.bincount(prev[hole]).argmax())
            if self.knowledge.move is None:
                self.knowledge.move = {
                    "dirs": {}, "player_color": None,
                    "record": RuleRecord("move", "control")}
            self.knowledge.move["dirs"] = dict(self.dirs)
            self.knowledge.move["player_color"] = self.player_color
        else:
            self.probe_attempts[aid] = self.probe_attempts.get(aid, 0) + 1
            if self.probe_attempts[aid] >= PROBE_ATTEMPT_CAP:
                self.dirs[aid] = (0, 0)

    # ------------------------------------------------------------- replay
    def _replay_step(self, grid, frame, avail):
        seq = self.replay["actions"]
        idx = self.replay["idx"]
        if idx >= len(seq):
            self.replay = None
            return (self._click_policy(grid, frame, avail) if self.mode == "click"
                    else self._move_policy(grid, frame, avail))
        stored = seq[idx]
        aid, data = stored["aid"], dict(stored["data"])
        self.replay["idx"] = idx + 1
        self.pending = {
            "aid": aid, "data": data, "kind": "replay", "replay": True,
            "replay_idx": idx, "expected": stored["effect"], "predicted": None}
        return aid, data, {"strategy": "verified_replay",
                           "step": f"{idx + 1}/{len(seq)}"}

    def _replay_abort(self, cause):
        seqrec = self.knowledge.sequences.get(self.level)
        if seqrec is not None:
            seqrec["aborts"] = seqrec.get("aborts", 0) + 1
            if seqrec["aborts"] >= REPLAY_ABORT_STALE:
                seqrec["stale"] = True
                self.ledger.append("SEQUENCE_STALE", level=self.level,
                                   cause=cause)
        self.ledger.append("REPLAY_ABORT", level=self.level, cause=cause)
        self.replay = None

    # ------------------------------------------------------------- click games
    def _select_targets(self, sel, objs):
        if sel == "rarest_color":
            counts = {}
            for o in objs:
                counts[o["color"]] = counts.get(o["color"], 0) + o["size"]
            if not counts:
                return []
            rare = min(counts, key=lambda c: (counts[c], c))
            return [o for o in objs if o["color"] == rare]
        if sel == "smallest_object":
            return [min(objs, key=lambda o: (o["size"], o["bbox"]))]
        if sel == "corner_sample_color":
            corner = [o for o in objs
                      if o["bbox"][0] < CELL and o["bbox"][1] < CELL]
            if not corner:
                return []
            sample = corner[0]
            return [o for o in objs
                    if o["color"] == sample["color"] and o["id"] != sample["id"]]
        if sel == "largest_object":
            return [max(objs, key=lambda o: o["size"])]
        return []

    def _click_policy(self, grid, frame, avail, _depth=0):
        objs = find_objects(grid)
        if not objs:
            return self._no_signal_explore(grid, avail)
        active = [s for s in self.sel_order
                  if self.knowledge.rules.get(s) is None
                  or self.knowledge.rules[s].status != STATUS_QUARANTINED]
        if not active:
            active = list(self.sel_order)
        sel = active[0]
        if not self.queue:
            targets = self._select_targets(sel, objs)
            counts = {}
            for o in objs:
                counts[o["color"]] = counts.get(o["color"], 0) + 1
            targets = sorted(targets, key=lambda o: (counts[o["color"]], o["size"],
                                                     o["bbox"]))
            self.queue = targets
            self.queue_ptr = 0
        target = None
        while self.queue_ptr < len(self.queue):
            cand = self.queue[self.queue_ptr]
            cy, cx = cand["centroid"]
            cell = (int(cy) // CELL, int(cx) // CELL)
            if (cand["color"], cell) not in self.clicked_sig:
                target = cand
                break
            self.queue_ptr += 1
        if target is None:
            if _depth >= len(CLICK_SELECTORS):
                # v2.1 M2' FIX: design §4.2 — no-signal click exploration
                # must be SALIENCE-ORDERED. The received code's blind cell
                # sweeps lost lp85's level-1 clear in the 25-game head-to-
                # head (v2.0's proven fallback clicked salient objects).
                return self._salience_explore(grid, objs, avail)
            self._demote(sel, why="all candidates clicked, no progress")
            return self._click_policy(grid, frame, avail, _depth + 1)
        mask = np.zeros_like(grid, dtype=bool)
        for (r, c) in target["pixels"]:
            mask[r, c] = True
        predicted = prim_recolor_region(grid, mask, 0)
        cy, cx = target["centroid"]
        x = min(max(int(round(cx)), 0), 63)
        y = min(max(int(round(cy)), 0), 63)
        sig = (target["color"], (int(cy) // CELL, int(cx) // CELL))
        self.pending = {"aid": 6, "kind": "click", "rule": sel,
                        "predicted": predicted, "sig": sig,
                        "data": {"x": x, "y": y, "game_id": self.game_id}}
        self.queue_ptr += 1
        rec = self.knowledge.rule(sel)
        return 6, dict(self.pending["data"]), {
            "strategy": "program_synthesis_v3", "hypothesis": sel,
            "evidence_class": rec.evidence_class}

    # ------------------------------------------------------------- move games
    def _move_policy(self, grid, frame, avail):
        avail = [a for a in avail if a != 0]
        unprobed = [a for a in avail
                    if a not in self.dirs
                    and self.probe_attempts.get(a, 0) < PROBE_ATTEMPT_CAP]
        if unprobed:
            aid = unprobed[0]
            self.pending = {"aid": aid, "kind": "probe", "predicted": None,
                            "data": {"game_id": self.game_id}}
            return aid, {"game_id": self.game_id}, {
                "strategy": "program_synthesis_v3", "phase": "probe controls"}
        objs = find_objects(grid)
        if not objs or self.player_color is None:
            return self._no_signal_explore(grid, avail)
        players = [o for o in objs if o["color"] == self.player_color]
        others = [o for o in objs if o["color"] != self.player_color]
        if not players or not others:
            return self._no_signal_explore(grid, avail)
        player = players[0]
        goal = salience_rank(others)[0][0]
        py, px = player["centroid"]
        gy, gx = goal["centroid"]

        def dist_after(a):
            dr, dc = self.dirs.get(a, (0, 0))
            return abs(py + dr * CELL - gy) + abs(px + dc * CELL - gx)

        ordered = sorted(avail, key=dist_after)
        if self.stuck >= 2:
            ordered = ordered[1:] + ordered[:1]
            self.stuck = 0
        aid = ordered[0]
        dr, dc = self.dirs.get(aid, (0, 0))
        mask = np.zeros_like(grid, dtype=bool)
        for (r, c) in player["pixels"]:
            mask[r, c] = True
        predicted = prim_move_region(grid, mask, dr * CELL, dc * CELL)
        self.pending = {"aid": aid, "kind": "move", "predicted": predicted,
                        "data": {"game_id": self.game_id}}
        return aid, {"game_id": self.game_id}, {
            "strategy": "program_synthesis_v3",
            "controls": {str(k): v for k, v in self.dirs.items()}}

    # ------------------------------------------------------------- v2.1 explore
    def _salience_explore(self, grid, objs, avail):
        """M2' FIX (design §4.2): salience-ordered no-signal click
        exploration — v2.0's proven exhaustion fallback, restored, with
        graph bookkeeping (objects already tried at this state are
        skipped). These clicks are v2-baseline behaviour and deliberately
        do NOT count against EXPLORATION_CAP (worst case = v2.0 exactly).
        Blind cell enumeration remains the last resort."""
        ranked, _ = salience_rank(objs)
        h = self.graph.hash_grid(grid)
        tried = self.graph.transitions.get(h, {}) if h is not None else {}
        tried_cells = {k[1:] for k in tried
                       if isinstance(k, tuple) and k[0] == 6}
        for o in ranked:
            cy, cx = o["centroid"]
            cell = (int(cy) // CELL, int(cx) // CELL)
            if cell in tried_cells:
                continue
            if (o["color"], cell) in self.clicked_sig:
                continue
            x = min(max(int(round(cx)), 0), 63)
            y = min(max(int(round(cy)), 0), 63)
            self.pending = {"aid": 6, "kind": "explore", "predicted": None,
                            "sig": (o["color"], cell),
                            "data": {"x": x, "y": y, "game_id": self.game_id}}
            return 6, dict(self.pending["data"]), {
                "strategy": "salience_explore", "object": o["id"],
                "color": o["color"], "cell": cell,
                "note": "salience-ordered no-signal click"}
        # every salient object tried at this state: blind cell sweep
        return self._no_signal_explore(grid, avail)

    def _no_signal_explore(self, grid, avail):
        """Novelty-seeking fallback. Uses the state-action graph to pick
        an untried action at the current state, or the least-visited
        successor. Never returns None unless avail is empty."""
        if not avail:
            return 0, {"game_id": self.game_id}, {"note": "no actions"}

        def _rotate(note):
            # FIX (audit B2/B3): bounded fallback — never repeat one dead
            # action forever; rotate through the legal actions instead
            aid = int(avail[self._explore_rr % len(avail)])
            self._explore_rr += 1
            data = {"game_id": self.game_id}
            if aid == 6:
                data = {"x": 32, "y": 32, "game_id": self.game_id}
            self.pending = {"aid": aid, "kind": "explore",
                            "predicted": None, "data": data}
            return aid, dict(data), {
                "strategy": "state_graph_explore", "action": aid,
                "note": note}

        # FIX (audit B3): exploration budget cap — after EXPLORATION_CAP
        # consecutive explore actions without progress, degrade to the
        # bounded rotation for the rest of the level (design §4.3)
        if self.explore_blocked:
            return _rotate("exploration capped")

        h = self.graph.hash_grid(grid)
        choice = self.graph.novelty_action(h, avail)

        # Novelty might return (6, r, c) tuple for untried click cells
        if isinstance(choice, tuple) and len(choice) == 3 and choice[0] == 6:
            _, r, c = choice
            x = min(max(c * CELL + CELL // 2, 0), 63)
            y = min(max(r * CELL + CELL // 2, 0), 63)
            cell = (r, c)
            self.pending = {"aid": 6, "kind": "explore", "predicted": None,
                            "sig": cell,
                            "data": {"x": x, "y": y, "game_id": self.game_id}}
            self.explore_streak += 1
            if self.explore_streak >= EXPLORATION_CAP:
                self.explore_blocked = True
                self.ledger.append("EXPLORATION_CAPPED",
                                   streak=self.explore_streak)
            return 6, dict(self.pending["data"]), {
                "strategy": "state_graph_explore", "action": 6,
                "cell": cell, "state": h,
                "note": "novel click cell"}
        if choice is not None:
            aid = int(choice)
            data = {"game_id": self.game_id}
            if aid == 6:
                data = {"x": 32, "y": 32, "game_id": self.game_id}
            self.pending = {"aid": aid, "kind": "explore", "predicted": None,
                            "data": data}
            self.explore_streak += 1
            if self.explore_streak >= EXPLORATION_CAP:
                self.explore_blocked = True
                self.ledger.append("EXPLORATION_CAPPED",
                                   streak=self.explore_streak)
            return aid, dict(data), {
                "strategy": "state_graph_explore", "action": aid,
                "state": h, "note": "novel action"}

        # Graph exhausted at this state: bounded rotation
        return _rotate("graph exhausted")

    # ------------------------------------------------------------- helpers
    def _demote(self, sel, why):
        if sel in self.sel_order:
            self.sel_order.remove(sel)
            self.sel_order.append(sel)
        self.queue, self.queue_ptr = [], 0
        rec = self.knowledge.rule(sel)
        rec.rotations += 1
        if rec.rotations >= 2 and rec.success == 0:
            rec.status = STATUS_QUARANTINED
            self.ledger.append("RULE_QUARANTINED", rule=sel, cause=why)
        else:
            self.ledger.append("RULE_DEMOTED", rule=sel, cause=why)

    def _clear_level_state(self):
        self.queue, self.queue_ptr = [], 0
        self.clicked_sig = set()
        self.blacklist = set()
        self.pending = None
        self.stuck = 0
        self.level_actions = []
        self.level_verified = False
        self.replay = None
        # v2.1: recent buffer restarts fresh on new level; the exploration
        # cap also resets (design §4.3: cap is per level)
        self.explore_streak = 0
        self.explore_blocked = False
        if self.graph is not None:
            self.graph.clear_recent()

    def _grid(self, frame):
        g = merged_view(frame)
        if g is None:
            g = _grid_of(frame)
        return g

    def _available(self, frame):
        try:
            avail = list(getattr(frame, "available_actions", None) or [])
            return [int(a) for a in avail]
        except Exception:
            return []


# ============================================================================
# SECTION 6 — KIT-FACING AGENT (never-hang wrapper + protocol discipline)
# ============================================================================
# FIX (audit B1): the ARC-AGI-3-Agents framework drives agents via
# Agent.main(); when the framework is importable we subclass its base so
# main()/frames/counter all work (local `make play-local` and the Kaggle
# gateway). Standalone (no framework), the class duck-types the same
# surface. Restored verbatim from the M2-proven v2 wrapper.
try:
    from agents.agent import Agent as _KitAgent   # ARC-AGI-3-Agents framework
    KIT_FRAMEWORK = True
except Exception:
    _KitAgent = object
    KIT_FRAMEWORK = False


class ProgramSynthesisV2Agent(_KitAgent):
    """MyAgent for the ARC-AGI-3 Kaggle starter kit.

    - choose_action / is_done match the official Agent ABC surface; when the
      framework is present the inherited main() loop drives this class.
    - Every decide path is wrapped: any exception or missing field falls
      back to the cheapest legal action (RESET when required) — it can
      degrade, never hang, never raise.
    - Constructor accepts any kit-style arguments (game_id is sniffed from
      kwargs or the second positional argument when present).
    - v2.2: per-game telemetry (K2). Observes frames via choose_action and
      keeps k5_per_game.csv current. Fully guarded: a telemetry failure is
      logged once and telemetry disables itself; gameplay is unaffected.
    """

    MAX_ACTIONS = 80
    TIME_BUDGET_S = 1.0

    def __init__(self, *args, **kwargs):
        verbose = bool(kwargs.pop("verbose", False))
        self._kit = False
        if _KitAgent is not object:
            try:
                super().__init__(*args, **kwargs)
                self._kit = True
            except Exception:
                self._kit = False   # bad args for the kit signature: standalone
        if not self._kit:
            self.agent_name = (kwargs.get("agent_name")
                               or (args[2] if len(args) > 2 else None)
                               or "ProgramSynthesisV4")
        gid = (kwargs.get("game_id")
               or (args[1] if len(args) > 1 and isinstance(args[1], str) else "")
               or "")
        self.core = ProgramSynthesisV2Core(verbose=verbose)
        self.action_counter = 0     # bumped by the harness (official main())
        self._steps = 0             # internal mirror, bumped by choose_action
        # v2.2 telemetry state
        self._k5_games = {}
        self._k5_io_ok = True
        self.begin_episode(gid or "")

    # -- lifecycle -----------------------------------------------------------
    def begin_episode(self, game_id):
        self.core.new_episode(game_id)

    # -- v2.2 telemetry (K2): observe-only; must never influence decisions ----
    def _k5_dir(self):
        import os
        return (os.environ.get("K5_TELEMETRY_DIR") or "/kaggle/working")

    def _k5_note(self, frame):
        """Update the row for the current game and flush the CSV.

        Called from choose_action AFTER the action was decided (never before),
        so telemetry cannot feed back into play. CSV content is a pure
        function of the observed frame sequence + action counts: same run,
        same bytes. Flushed every call (write early, write often).
        """
        if not self._k5_io_ok:
            return
        try:
            gid = str(getattr(frame, "game_id", "") or "")
            if not gid:
                return
            row = self._k5_games.get(gid)
            if row is None:
                row = {"game_id": gid, "levels_completed": 0, "win_levels": 0,
                       "win": 0, "actions_used": 0, "decision_events": 0,
                       "final_state": ""}
                self._k5_games[gid] = row
            state = _state_name(getattr(frame, "state", None))
            row["levels_completed"] = max(
                row["levels_completed"],
                int(getattr(frame, "levels_completed", 0) or 0))
            row["win_levels"] = max(
                row["win_levels"], int(getattr(frame, "win_levels", 0) or 0))
            row["win"] = 1 if state == "WIN" else row["win"]
            row["actions_used"] = max(self.action_counter, self._steps)
            row["decision_events"] = len(self.core.ledger)
            if state:
                row["final_state"] = state
            self._k5_flush()
        except Exception:
            self._k5_io_ok = False
            try:
                print("[k5] telemetry disabled after an I/O error (gameplay unaffected)")
            except Exception:
                pass

    def _k5_flush(self):
        import csv
        import hashlib
        import json
        import os
        d = self._k5_dir()
        os.makedirs(d, exist_ok=True)
        cols = ["run_variant", "run_agent_sha256", "run_knobs_sha256",
                "game_id", "levels_completed", "win_levels", "win",
                "actions_used", "decision_events", "final_state"]
        variant = str(K5_RUN_INFO.get("variant") or "")
        asha = str(K5_RUN_INFO.get("agent_sha256") or "")
        ksha = hashlib.sha256(json.dumps(K5_RUN_INFO.get("knobs") or {},
                                         sort_keys=True).encode()).hexdigest()
        tmp = os.path.join(d, "k5_per_game.csv.tmp")
        final = os.path.join(d, "k5_per_game.csv")
        # Merge-on-write: the harness may build a fresh MyAgent per game, so
        # self._k5_games never accumulates across games. Re-read whatever is
        # already on disk and union it in before writing, so the CSV always
        # covers every game of the run. Counters take max (monotonic), win
        # is monotonic too, final_state prefers the live row. A corrupt or
        # absent prior file is ignored: telemetry degrades, never breaks.
        rows = {gid: dict(row) for gid, row in self._k5_games.items()}
        try:
            import io
            with open(final) as fh:
                body = "".join(ln for ln in fh if not ln.startswith("#"))
            if body.strip():
                for old in csv.DictReader(io.StringIO(body)):
                    gid = old.get("game_id") or ""
                    if not gid:
                        continue
                    cur = rows.get(gid)
                    if cur is None:
                        rows[gid] = old
                    else:
                        for k in ("levels_completed", "win_levels", "win",
                                  "actions_used", "decision_events"):
                            try:
                                cur[k] = max(int(cur[k] or 0), int(old[k] or 0))
                            except (TypeError, ValueError):
                                pass
                        if not cur.get("final_state") and old.get("final_state"):
                            cur["final_state"] = old["final_state"]
        except Exception:
            pass
        with open(tmp, "w", newline="") as fh:
            fh.write("# k5 telemetry v1 (a FILTER/observation aid, not a score)\n")
            fh.write("# variant=%s agent_sha256=%s knobs_sha256=%s\n"
                     % (variant, asha, ksha[:16]))
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for gid in sorted(rows):
                row = dict(rows[gid])
                row.update(run_variant=variant, run_agent_sha256=asha,
                           run_knobs_sha256=ksha[:16])
                w.writerow(row)
        os.replace(tmp, final)
        info = dict(K5_RUN_INFO)
        info["knobs_sha256"] = ksha
        with open(os.path.join(d, "k5_run_info.json"), "w") as fh:
            json.dump(info, fh, indent=1, sort_keys=True)

    # -- official ABC surface --------------------------------------------------
    def is_done(self, frames, latest_frame):
        try:
            frame = _pick_frame(frames, latest_frame)
            # Telemetry here too: the terminal (WIN) frame is only ever
            # offered to is_done, never to choose_action, so noting here is
            # the only way the final level total reaches the CSV row.
            self._k5_note(frame)
            if _state_name(getattr(frame, "state", None)) == "WIN":
                return True
            return (self.action_counter >= self.MAX_ACTIONS
                    or self._steps >= self.MAX_ACTIONS)
        except Exception:
            return False

    def choose_action(self, frames, latest_frame):
        try:
            # kit main() passes the converted current observation as arg 2
            # (frames[0] is a dummy); runner mode passes frames[-1] itself
            frame = _pick_frame(frames, latest_frame)
            t0 = time.perf_counter()
            aid, data, reasoning = self.core.next_action(frame)
            action = self._make_action(aid, data, reasoning)
            self._steps += 1
            if time.perf_counter() - t0 > self.TIME_BUDGET_S:
                self.core.ledger.append(
                    "SLOW_DECISION",
                    seconds=round(time.perf_counter() - t0, 3))
            self._k5_note(frame)   # telemetry AFTER the decision (observe-only)
            return action
        except Exception:
            self._steps += 1
            return self._fallback(frames, latest_frame)

    @staticmethod
    def _make_action(aid, data, reasoning):
        action = GameAction.from_id(int(aid))
        if data:
            try:
                action.set_data({k: (int(v) if k in ("x", "y") else v)
                                 for k, v in data.items()})
            except Exception:
                pass
        try:
            action.reasoning = reasoning
        except Exception:
            pass
        return action

    def _fallback(self, frames, latest_frame):
        try:
            state = _state_name(getattr(latest_frame, "state", None))
            if state in ("NOT_PLAYED", "GAME_OVER"):
                return self._make_action(0, {"game_id": self.core.game_id},
                                         {"strategy": "fallback_reset"})
            avail = []
            try:
                avail = [int(a) for a in
                         (getattr(latest_frame, "available_actions", None) or [])]
            except Exception:
                avail = []
            if not avail:
                avail = [1, 2, 3, 4, 6]
            aid = avail[0]
            if aid == 6:
                data = {"x": 32, "y": 32, "game_id": self.core.game_id}
            else:
                data = {"game_id": self.core.game_id}
            return self._make_action(aid, data, {"strategy": "safe_fallback"})
        except Exception:
            try:
                return GameAction.RESET
            except Exception:
                return None


MyAgent = ProgramSynthesisV2Agent
