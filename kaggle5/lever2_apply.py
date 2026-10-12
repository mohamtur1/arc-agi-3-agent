# CELL 4B-2 of 2 — apply lever 2 to the harness source, then verify.
# Run right after the 4b-1 patch-writer cell. Fails loudly if the
# bundle is not the exact state this patch was verified against.
import hashlib
import subprocess
from pathlib import Path

PATCH = Path(WORKING_DIR) / "lever2-grid.patch"
AGENT = Path(BUNDLE_DIR) / "src/ARC3-Inference/inference/agent"
SCOPE = "ARC3-Inference/inference/agent/*"

# Post-image git blob hashes, recorded in the patch's own index lines.
POST = {
    "prompts.py": "3053e65dea2e8bc804c6321866dcedb64ba196ce",
    "python_tool_sandbox.py": "71f7172e9ee7555e44bc5e5f6743a2567b0f95b8",
    "tool_agent.py": "6b21669ee7c17ab757cea202085f65707515efd6",
}


def blob_hash(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def snapshot():
    return {name: blob_hash(AGENT / name) for name in POST}


if not AGENT.is_dir():
    raise FileNotFoundError("run the section 1 setup cell first: " + str(AGENT))
if not PATCH.is_file():
    raise FileNotFoundError("run the 4b-1 cell first: " + str(PATCH))

if snapshot() == POST:
    print("lever 2: already applied, nothing to do")
else:
    try:
        subprocess.run(
            ["git", "apply", "--include=" + SCOPE, "-v", str(PATCH)],
            cwd=str(Path(BUNDLE_DIR) / "src"), check=True,
            capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or exc.stdout or "").strip()[-400:]
        raise RuntimeError(
            "lever 2 FAILED: the grid patch did not apply. This bundle is "
            "not the state the patch was verified against. git said:\n" + tail
        ) from None
    after = snapshot()
    if after != POST:
        bad = [n for n in POST if after[n] != POST[n]]
        raise RuntimeError("lever 2 FAILED hash check: " + ", ".join(bad))
    print("lever 2: applied and hash-verified")

# The capability itself: the sandbox must expose a numeric grid property.
boot = (AGENT / "python_tool_sandbox.py").read_text()
if "def grid(self)" not in boot:
    raise RuntimeError("lever 2 FAILED: grid property missing from sandbox")
print("lever 2: current_frame.grid is live for python tool calls")
