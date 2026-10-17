# CELL 4B-2 of 2 — apply and verify the Lever-3 harness patch.
# Run right after the 4b-1 patch-writer cell. Fails loudly if the bundle is
# not the exact state this patch was verified against.
#
# lever2-grid.patch now carries the lever-2 `.grid` property AND Lever-3
# (deterministic episodic memory + the hard prune gate). The name is kept so
# every existing fetch path, pin and ledger row stays valid.
#
# VARIANT
#   lever3 : memory + hard prune gate + prompt directives  (default)
#   kernel : memory view present, gate off, prompt untouched (A/B rollback)
import hashlib
import os
import subprocess
import sys
from pathlib import Path

VARIANT = "lever3"  # k5-variant-line

AGENT_REL = Path("inference") / "agent" / "python_tool_sandbox.py"


def _looks_like_src(path) -> bool:
    try:
        return bool(path) and (Path(path) / "ARC3-Inference" / AGENT_REL).is_file()
    except OSError:
        return False


def _search_for_src(max_depth: int = 6):
    """Find the bundle's src/ by looking for the harness itself."""
    import os as _os
    roots = ["/kaggle/working", "/kaggle/temp", _os.getcwd()]
    skip = {"__pycache__", ".git", "node_modules", ".ipynb_checkpoints"}
    for root in roots:
        if not _os.path.isdir(root):
            continue
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, _ in _os.walk(root):
            if dirpath.count("/") - base_depth >= max_depth:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in skip]
            if "python_tool_sandbox.py" in _os.listdir(dirpath):
                # <src>/ARC3-Inference/inference/agent/python_tool_sandbox.py
                # <src>/ARC3-Inference/inference/agent -> parents[2] is <src>
                src = Path(dirpath).parents[2] if len(Path(dirpath).parts) > 2 else None
                if _looks_like_src(src):
                    return src
    return None


def _resolve_src():
    """Three tiers, because this cell runs inside the notebook namespace.

    The kernel's own setup cell defines a Python global BUNDLE_DIR. Reading
    only the ENVIRONMENT variable was the bug: the global exists, the env var
    does not, so the applier fell back to a hard-coded guess and died. A cell
    sees globals first; a standalone run has to fall back to env, then to a
    filesystem search.
    """
    seen = []
    g = globals().get("BUNDLE_DIR")
    if g is not None:
        seen.append(("kernel global BUNDLE_DIR", Path(g)))
    if os.environ.get("BUNDLE_DIR"):
        seen.append(("env BUNDLE_DIR", Path(os.environ["BUNDLE_DIR"])))
    seen.append(("default", Path("/kaggle/working/taaf-kaggle-source-bundle")))
    for label, base in seen:
        for candidate in (base / "src", base):
            if _looks_like_src(candidate):
                print("lever 3: harness located via %s -> %s" % (label, candidate))
                return candidate
    found = _search_for_src()
    if found is not None:
        print("lever 3: harness located by filesystem search -> %s" % found)
        return found
    raise FileNotFoundError(
        "lever 3 FAILED: could not find the harness bundle. Looked at "
        "kernel global BUNDLE_DIR=%r, env BUNDLE_DIR=%r, the default path, "
        "and a bounded search of /kaggle/working, /kaggle/temp and cwd for "
        "ARC3-Inference/inference/agent/python_tool_sandbox.py. Run this cell "
        "after the kernel's setup cell that defines BUNDLE_DIR."
        % (globals().get("BUNDLE_DIR"), os.environ.get("BUNDLE_DIR")))


WORKING_DIR = Path(globals().get("WORKING_DIR")
                   or os.environ.get("WORKING_DIR") or "/kaggle/working")
if not WORKING_DIR.is_dir():
    WORKING_DIR = Path("/kaggle/working")
PATCH = WORKING_DIR / "lever2-grid.patch"
SRC = _resolve_src()
AGENT = SRC / "ARC3-Inference" / "inference" / "agent"

# Post-image git blob hashes taken from the patch's own index lines.
POST = {
    "ARC3-Inference/inference/agent/prompts.py":
        "fd749e93e7be9d827d63d13351513de24ab8b68c",
    "ARC3-Inference/inference/agent/python_tool_sandbox.py":
        "cf3b6a86b742527fe4194fa2a335afc380c9bcff",
    "ARC3-Inference/inference/agent/tool_agent.py":
        "bb3a5a658ad9e8b08b8fe2072dd8c9a0045e39e8",
    "ARC3-Inference/inference/agent/episodic_memory.py":
        "79683696f1c5c6b98f627e0a0a3b3fb0fe87f9aa",
    "ARC3-Inference/README.md":
        "9f95aa3a74cd17e80b027d91ab2874b52f1a0e4c",
}

CAPABILITIES = {
    "python_tool_sandbox.py": ["def grid(self)", "class MemoryView", "current_frame.memory = memory_view"],
    "episodic_memory.py": ["class CompactStateGraph", "def gate(", "EPISODIC_MEMORY_GATE"],
    "tool_agent.py": ['"memory": self._memory.to_payload()', "k5_pruned", "self._memory.observe("],
    "prompts.py": ["EPISODIC_MEMORY_ADDENDUM"],
}


def blob_hash(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def snapshot() -> dict:
    return {
        rel: (blob_hash(SRC / rel) if (SRC / rel).is_file() else "MISSING")
        for rel in POST
    }


def core_present() -> bool:
    return all((AGENT / name).is_file() for name in CAPABILITIES) and (
        "class CompactStateGraph" in (AGENT / "episodic_memory.py").read_text()
    )


if not SRC.is_dir():
    raise FileNotFoundError("lever 3 FAILED: resolved src is not a directory: "
                            + str(SRC))
if not PATCH.is_file():
    raise FileNotFoundError("run the 4b-1 cell first: " + str(PATCH))

if core_present():
    print("lever 3: already applied, nothing to do (core memory module present)")
else:
    try:
        subprocess.run(
            ["git", "apply", "-v", str(PATCH)],
            cwd=str(SRC), check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip()[-600:]
        raise RuntimeError(
            "lever 3 FAILED: the patch did not apply. This bundle is not the "
            "state the patch was built against. git said:\n" + tail
        ) from None
    after = snapshot()
    bad = [name for name, digest in after.items() if POST[name] != digest]
    if bad:
        raise RuntimeError("lever 3 FAILED hash check: " + ", ".join(bad))
    print("lever 3: applied and hash-verified (%d files)" % len(POST))

# Ordering guard: editing files after the harness is imported changes nothing.
import sys as _sys
_loaded = [m for m in ("inference.agent.tool_agent",
                       "inference.agent.python_tool_sandbox",
                       "inference.agent.prompts") if m in _sys.modules]
if _loaded:
    print("lever 3: WARNING the harness was ALREADY IMPORTED before this cell "
          "(%s). File edits cannot change a loaded module, so this run may not "
          "see Lever-3. Move the lever cells above the harness import."
          % ", ".join(_loaded))
else:
    print("lever 3: ordering OK (harness not imported yet, patched files will load)")

# Capability proof: the patch must be live, not merely present.
missing = []
for name, markers in CAPABILITIES.items():
    text = (AGENT / name).read_text(encoding="utf-8")
    missing += ["%s:%s" % (name, marker) for marker in markers if marker not in text]
if missing:
    raise RuntimeError("lever 3 FAILED capability check: " + ", ".join(missing))
print("lever 3: capability check OK (grid, MemoryView, CompactStateGraph, gate, prompt)")

# ------------------------------------------------------------------ variant
gate_file = AGENT / "episodic_memory.py"
prompt_file = AGENT / "prompts.py"
tool_file = AGENT / "tool_agent.py"

if VARIANT == "kernel":
    gate_text = gate_file.read_text(encoding="utf-8")
    if "EPISODIC_MEMORY_GATE = True" in gate_text:
        gate_file.write_text(
            gate_text.replace("EPISODIC_MEMORY_GATE = True",
                              "EPISODIC_MEMORY_GATE = False", 1),
            encoding="utf-8")
        print("lever 3: variant=kernel -> prune gate disabled (memory stays readable)")
    prompt_text = prompt_file.read_text(encoding="utf-8")
    if "    prompt += EPISODIC_MEMORY_ADDENDUM\n" in tool_file.read_text(encoding="utf-8"):
        tool_text = tool_file.read_text(encoding="utf-8")
        tool_file.write_text(
            tool_text.replace("    prompt += EPISODIC_MEMORY_ADDENDUM\n",
                              "    pass  # k5 variant=kernel: memory is advisory only\n", 1),
            encoding="utf-8")
        print("lever 3: variant=kernel -> memory directives removed from the prompt")
elif VARIANT != "lever3":
    raise RuntimeError("unknown variant %r (expected 'lever3' or 'kernel')" % VARIANT)
else:
    print("lever 3: variant=lever3 -> memory + hard prune gate active")

print("lever 3: bundle ready at %s" % SRC)
