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

WORKING_DIR = Path(os.environ.get("WORKING_DIR", "/kaggle/working"))
BUNDLE_DIR = Path(os.environ.get("BUNDLE_DIR",
                                 "/kaggle/working/taaf-kaggle-source-bundle"))
PATCH = WORKING_DIR / "lever2-grid.patch"
SRC = BUNDLE_DIR / "src"
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
    raise FileNotFoundError("run the section 1 setup cell first: " + str(SRC))
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
