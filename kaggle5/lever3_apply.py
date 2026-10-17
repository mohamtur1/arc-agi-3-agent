"""lever3_apply.py — Injects CompactStateGraph memory and action pruning into the harness."""
import os
import sys
from pathlib import Path

WORKING_DIR = os.environ.get("WORKING_DIR", "/kaggle/working")
BUNDLE_DIR = os.environ.get("BUNDLE_DIR", "/kaggle/working/taaf-kaggle-source-bundle")

AGENT_DIR = Path(BUNDLE_DIR) / "src/ARC3-Inference/inference/agent"
SANDBOX_FILE = AGENT_DIR / "python_tool_sandbox.py"
PROMPTS_FILE = AGENT_DIR / "prompts.py"

COMPACT_STATE_GRAPH_SRC = '''
class CompactStateGraph:
    """Lightweight in-sandbox state graph and transition tracker."""
    def __init__(self):
        self.transitions = {}
        self.dead_ends = set()
        self.visit_counts = {}
        self._last_state = None
        self._last_action = None

    @staticmethod
    def hash_grid(grid):
        if not grid:
            return 0
        raw = bytearray()
        for r in grid:
            raw.extend(r)
        h = 0xcbf29ce484222325
        for b in raw:
            h ^= b
            h = (h * 0x100000001b3) & 0xffffffffffffffff
        return h

    def tried(self, s_hash):
        return [(act, outcome) for act, _, outcome in self.transitions.get(s_hash, [])]

    def is_cycle_or_dead_end(self, s_hash, action_repr):
        for act, next_h, out in self.transitions.get(s_hash, []):
            if str(act) == str(action_repr):
                if out in ("dead_end", "loop", "no_op") or next_h in self.dead_ends or next_h == s_hash:
                    return True
        return False

    def record(self, grid, action_repr):
        s_hash = self.hash_grid(grid)
        self._last_state = s_hash
        self._last_action = str(action_repr)
        self.visit_counts[s_hash] = self.visit_counts.get(s_hash, 0) + 1

    def update_result(self, new_grid, outcome="active"):
        if self._last_state is None:
            return
        new_hash = self.hash_grid(new_grid)
        s_hash = self._last_state
        act = self._last_action
        if new_hash == s_hash:
            outcome = "no_op"
        elif outcome in ("game_over", "lost"):
            self.dead_ends.add(new_hash)
            outcome = "dead_end"
        if s_hash not in self.transitions:
            self.transitions[s_hash] = []
        self.transitions[s_hash].append((act, new_hash, outcome))
        self._last_state = None
        self._last_action = None

_GLOBAL_EPISODIC_MEMORY = CompactStateGraph()
'''

def apply_lever3():
    if not SANDBOX_FILE.is_file():
        raise FileNotFoundError(f"Sandbox file not found: {SANDBOX_FILE}")

    sb_text = SANDBOX_FILE.read_text(encoding="utf-8")
    if "CompactStateGraph" not in sb_text:
        sb_text = COMPACT_STATE_GRAPH_SRC + "\n" + sb_text

        grid_hook = "def grid(self):"
        memory_prop = """
        @property
        def memory(self):
            return _GLOBAL_EPISODIC_MEMORY
"""
        if grid_hook in sb_text:
            sb_text = sb_text.replace(grid_hook, memory_prop + "\n        " + grid_hook, 1)
        else:
            raise RuntimeError("lever3 FAILED: unable to locate FrameView grid hook in sandbox")

        SANDBOX_FILE.write_text(sb_text, encoding="utf-8")
        print("lever 3: CompactStateGraph injected into python_tool_sandbox.py")
    else:
        print("lever 3: CompactStateGraph already present in sandbox")

    if PROMPTS_FILE.is_file():
        pr_text = PROMPTS_FILE.read_text(encoding="utf-8")
        memory_instruction = (
            "- `current_frame.memory` is an episodic state graph. "
            "Call `current_frame.memory.tried(current_frame.memory.hash_grid(current_frame.grid))` before choosing an action. "
            "If an action appears with outcome 'loop', 'no_op', or 'dead_end', PRUNE it and pick another action.\\n"
        )
        if "current_frame.memory" not in pr_text:
            target = "STRUCTURED_RUNTIME_STATE_ADDENDUM = ("
            pr_text = pr_text.replace(target, target + f'\\n    "{memory_instruction}"')
            PROMPTS_FILE.write_text(pr_text, encoding="utf-8")
            print("lever 3: prompt addendum updated with action pruning directives")

if __name__ == "__main__":
    apply_lever3()
