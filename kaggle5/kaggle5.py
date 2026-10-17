"""kaggle5 — the between-game learning loop driver. Pushes daily, never submits.

Runs INSIDE a scheduled Kaggle notebook (CPU-only, internet ON) and drives the
competition kernel (GPU, internet OFF) purely through the Kaggle CLI.

What a day looks like now
  1. pull the ledger (dataset download; falls back loudly if unavailable)
  2. reconcile: read yesterday's submission rows back and attach them to the
     pending experiment (per-game telemetry when the kernel exposes it;
     absent / zero / errored submissions are all recorded as data)
  3. plan: fold the ledger, pick the next one-knob experiment from the fixed
     queue (with its documented backtrack rule), and log exactly why
  4. filter: run the local toy gym (k5_gym.py) — a smoke filter, never a
     score — and skip obviously-broken variants before they burn a day
  5. write the pending row (append-only, durable BEFORE the push)
  6. push the kernel with the knob cell injected — AND STOP. The machine
     never calls `competitions submit`. The human finalizes in the Kaggle
     app; the run prints exactly which variant was prepared and what to tap.

Stages
  check    free: pull, verify inputs, probe CLI flags, dry-run the injection,
           scan for the K5 knob channel, then probe submission-limits.
  plan     reconcile + print what today's experiment would be. Pushes nothing.
  gym      run the local filter for an ad-hoc --set KNOB=VALUE (offline OK).
  push     manual one-shot: ad-hoc experiment via --set, or lever-only repush
  score    reconcile + print the experiment table. Pushes nothing.
  daily    the scheduled cell: the full loop above, idempotent within a day.
  submit / all   REMOVED. They print why and exit non-zero. A human submits.

`daily` keeps no hidden state: everything lives in the append-only ledger
(experiments.jsonl + game_rows.jsonl), which is synced to a private Kaggle
dataset at the start and end of every run — and once more right before the
push, so a crash can never lose a pending row.

Run `python kaggle5.py --self-test` to exercise the whole loop offline
against _stub_kaggle.py: multi-day reconciliation, backtracking, dummies,
stale rows, refusals. No auth, no network, no Kaggle.

One honest fixed fact: this system measures. It does not compound. A knob
sweep is expected to find small deltas (and some negative ones); the value
is one trustworthy measurement per day and a ledger that explains itself.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG — edit only this block
# --------------------------------------------------------------------------

OWNER = "mohamedmalikturay"
KERNEL_SLUG = "arc-agi-3-milestone-2-solution-3e9adb"
KERNEL_URL = "https://www.kaggle.com/code/%s/%s" % (OWNER, KERNEL_SLUG)
COMPETITION = "arc-prize-2026-arc-agi-3"
SUBMISSION_FILE = "submission.parquet"

LEDGER_DATASET = "%s/kaggle5-ledger" % OWNER
LEDGER_FILES = ("experiments.jsonl", "game_rows.jsonl")

AGENT_FILE = "program_synthesis_agent_v4.py"
AGENT_REPO_URL = ("https://raw.githubusercontent.com/mohamtur1/4CBOn2/main/Kaggle/"
                  + AGENT_FILE)
AGENT_SHA256 = "8f447e454134d4a55dd0dfe3227c6273fa70c595f24773a0c6c84e1ce7b440a0"
GYM_FILE = "k5_gym.py"
KNOB_JSON_NAME = "k5_knobs.json"
AGENT_CHANNEL_MARKER = "K5_KNOBS_JSON"   # present in any agent with the K1 loader

DEVICE = "franzen-harness-v1"
ANCHOR_VARIANT = "franzen-base"
HOOK_HEADING_SUBSTR = "customization hook"

LEVER_TAG = "lever-2"
LEVER_PATCH_FILE = "lever2-grid.patch"     # the git patch that IS the lever

# The lever patches harness SOURCE, so it must land after section 1 (which
# copies the bundle and runs `git apply harness-changes.patch`) and before
# section 3 (which installs the runtime). Section 2's heading is the insert
# point; verified against the real notebook, which puts the lever at cells 5-7.
INJECT_AFTER_HEADING = "1. environment and submission mode"
INJECT_BEFORE_HEADING = "2. precaching"

# The applier cell reads these two names. They are defined in section 1.
REQUIRED_UPSTREAM_NAMES = ("WORKING_DIR", "BUNDLE_DIR")

# Only used when machine_shape comes back empty. Verified value, taken from the
# notebook's own metadata. The CLI docs write it NvidiaRtxPro6000.
FALLBACK_ACCELERATOR = "nvidiaRtxPro6000"

# The 5 inputs. push refuses if any is missing.
EXPECTED_COMPETITION_SOURCES = ["arc-prize-2026-arc-agi-3"]
EXPECTED_DATASET_SOURCES = [
    "dfranzen/pennyroyal-v253",
    "dfranzen/taaf-kaggle-source-bundle-copy",
]
EXPECTED_MODEL_SOURCES = [
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround",
    "dfranzen/albucino-qwen3-8-flash-next-drafter",
]

STATUS_POLL_SECONDS = 60
STATUS_MAX_SECONDS = 9 * 3600 + 1800       # ARC-AGI-3 notebooks cap at 9 h
SCORE_POLL_SECONDS = 120
SCORE_MAX_SECONDS = 6 * 3600

STALE_AFTER_DAYS = 2                  # pending this long unanswered -> stale

# Scheduled runs fire in UTC. Freetown is UTC+0 all year, so 0 is correct there.
LOCAL_UTC_OFFSET_HOURS = 0

# --------------------------------------------------------------------------
# THE KNOBS — full default state (must mirror the agent module exactly;
# self-test imports the agent and refuses to run on drift). GRID/CELL are
# structural and are permanently OFF the sweep list.
# --------------------------------------------------------------------------

KNOB_SURFACE = {
    "analyzer_timeout": {"type": "float", "min": 0.0, "default": 900.0},
    "max_runtime_s_per_game": {"type": "float", "min": 0.0, "default": 31920.0},
    "max_actions_per_game": {"type": "int_or_none", "min": 1, "default": None},
    "concurrency": {"type": "int", "min": 1, "default": 120},
}

DEFAULT_KNOBS = {
    "analyzer_timeout": 900.0,
    "max_runtime_s_per_game": 31920.0,
    "max_actions_per_game": None,
    "concurrency": 120,
}

EXPERIMENT_QUEUE = [
    ("analyzer_timeout", [1200.0],
     "up: 900.0 -> 1200.0 gives analyzer 33% more time per puzzle; respects |log2(1200/900)| <= 1"),
    ("max_actions_per_game", [500],
     "cap: None -> 500 caps the uncapped action side to prevent runaway infinite loops"),
    ("max_runtime_s_per_game", [31320.0],
     "trim: 31920 -> 31320 trims ~2% (600s / 10m buffer) per game"),
    ("concurrency", [60],
     "down: 120 -> 60 halves concurrency to reduce memory pressure / contention"),
]

# Historical anchors, seeded once into an empty ledger. Scores from the
# owner's first two submissions (v1 -> 0.06, v2 -> 0.07); agent file hashes
# were not recorded at the time, which is exactly why the ledger exists now.
IMPORTED_HISTORY = [
    {"variant": "v1-imported", "score": 0.06,
     "note": "historical import: fixed-bag-of-transforms agent; predates the loop"},
    {"variant": "v2-imported", "score": 0.07,
     "note": "historical import: learning machinery (EpisodeLedger/GameKnowledge/"
             "RuleRecord); predates the loop"},
]

# CLI subcommands + flags this driver relies on, probed against the INSTALLED
# cli (bug D: no inherited flag assumptions; verified present on 2.2.4 but the
# probe checks whatever is actually in the container, every run).
REQUIRED_CLI = {
    ("kernels", "pull"): ["-p", "-m"],
    ("kernels", "push"): ["-p", "--accelerator"],
    ("kernels", "status"): [],
    ("kernels", "files"): ["--format"],
    ("kernels", "output"): ["-p"],
    ("competitions", "submissions"): ["--format", "--page-size"],
    ("competitions", "submission-limits"): ["--json"],
    ("datasets", "download"): ["-p", "--unzip"],
    ("datasets", "create"): ["-p"],
    ("datasets", "version"): ["-p", "-m", "-t"],
}

# --------------------------------------------------------------------------


def say(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    say("")
    say("STOP — " + msg)
    raise SystemExit(1)


def norm(s: str) -> str:
    return re.sub(r"[\s\-_:]+", " ", (s or "").strip().lower())


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def canon(value) -> str:
    """Canonical string for a knob value (list vs tuple safe)."""
    return json.dumps(value, sort_keys=True)


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

def check_auth() -> None:
    """Report the credential path. The abrasive version lives in the
    notebook's auth cell (bug A); here we only describe what was found. The
    last resort is Kaggle's ambient notebook token — never an abort."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        say("auth: KAGGLE_API_TOKEN is set")
        return
    if os.environ.get("KAGGLE_API_V1_TOKEN"):
        say("auth: KAGGLE_API_V1_TOKEN is set (Kaggle-injected v1 token)")
        return
    for p in ("~/.kaggle/access_token", "~/.kaggle/access_token.txt",
              "~/.kaggle/kaggle.json"):
        f = Path(os.path.expanduser(p))
        if f.is_file() and f.read_text().strip():
            say("auth: using %s" % p)
            return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        say("auth: using legacy KAGGLE_USERNAME/KAGGLE_KEY")
        return
    say("auth: no explicit credentials. Inside a Kaggle notebook the SDK reads")
    say("      the token Kaggle injects, so this is expected and may still work.")


def kaggle(args: list[str], capture: bool = True, timeout: int = 900,
           allow_fail: bool = False, quiet: bool = False) -> subprocess.CompletedProcess:
    """Run the kaggle CLI.

    KAGGLE_BIN overrides the command (space separated) — used for the test
    double, and handy for pinning a specific install.
    """
    override = os.environ.get("KAGGLE_BIN", "").strip()
    if override:
        base = override.split()
    elif shutil.which("kaggle"):
        base = [shutil.which("kaggle")]
    else:
        base = [sys.executable, "-m", "kaggle"]
    cmd = base + args
    say("$ " + " ".join(cmd))
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
        timeout=timeout,
    )
    if capture and r.stdout and not quiet:
        say(r.stdout.rstrip())
    if r.returncode != 0 and not allow_fail:
        if quiet and r.stdout:
            say(r.stdout.rstrip())
        die("kaggle %s exited %d" % (args[0], r.returncode))
    return r


def probe_cli_flags() -> None:
    """Bug D: verify every subcommand + flag this driver uses against the
    CLI actually installed in the container, not against a version note."""
    if os.environ.get("KAGGLE_BIN", "").strip():
        say("  cli flag probe: skipped (KAGGLE_BIN override in use)")
        return
    missing = []
    for (grp, sub), flags in sorted(REQUIRED_CLI.items()):
        r = kaggle([grp, sub, "--help"], allow_fail=True, quiet=True)
        if r.returncode != 0:
            missing.append("%s %s (subcommand absent)" % (grp, sub))
            continue
        for fl in flags:
            if fl not in (r.stdout or ""):
                missing.append("%s %s %s" % (grp, sub, fl))
    if missing:
        die("the installed kaggle CLI does not provide everything this driver "
            "needs:\n  " + "\n  ".join(missing) +
            "\n  Do not patch around this silently: fix the driver or pin the CLI.")
    say("  cli flag probe: all %d subcommands and flags present on the "
        "installed CLI" % len(REQUIRED_CLI))


def probe_submission_limits() -> None:
    """Bug E: `competitions submission-limits` (present on 2.2.4 — verified),
    best-effort and non-blocking: printed for the human, never parsed into
    control flow."""
    r = kaggle(["competitions", "submission-limits", COMPETITION, "--json"],
               allow_fail=True, quiet=True)
    out = (r.stdout or "").strip()
    if r.returncode != 0 or not out:
        say("  submission-limits: unavailable on this CLI/competition; "
            "assuming the usual 1/day (informational only)")
        return
    say("  submission-limits (informational): %s" % out[:400])


def _json_from(stdout: str):
    try:
        return json.loads(stdout[stdout.index("["): stdout.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------
# pull + guards
# --------------------------------------------------------------------------

def pull(work: Path) -> tuple[Path, dict]:
    work.mkdir(parents=True, exist_ok=True)
    out = kaggle(["kernels", "pull", "-p", str(work), "-m",
                  f"{OWNER}/{KERNEL_SLUG}"]).stdout
    meta_path = work / "kernel-metadata.json"
    if not meta_path.is_file():
        die("pull produced no kernel-metadata.json:\n" + out)
    meta = json.loads(meta_path.read_text())
    code = work / meta.get("code_file", "")
    if not code.is_file():
        die("pull produced no notebook file (expected %s)" % code.name)
    return code, meta


def guard_metadata(meta: dict) -> None:
    say("")
    say("--- data sources on the pulled kernel ---")
    for key in ("competition_sources", "dataset_sources", "kernel_sources",
                "model_sources"):
        say("  %-22s %s" % (key, meta.get(key) or []))
    say("  %-22s %s" % ("enable_gpu", meta.get("enable_gpu")))
    say("  %-22s %s" % ("enable_internet", meta.get("enable_internet")))
    say("  %-22s %r" % ("machine_shape", meta.get("machine_shape")))

    def missing(expected: list[str], actual: list[str]) -> list[str]:
        have = [norm(x).lstrip("c/") for x in (actual or [])]
        out = []
        for e in expected:
            ne = norm(e)
            if not any(h == ne or h.startswith(ne + "/") for h in have):
                out.append(e)
        return out

    problems = []
    if m := missing(EXPECTED_COMPETITION_SOURCES, meta.get("competition_sources")):
        problems.append("competition_sources missing %s" % m)
    if m := missing(EXPECTED_DATASET_SOURCES, meta.get("dataset_sources")):
        problems.append("dataset_sources missing %s" % m)
    if m := missing(EXPECTED_MODEL_SOURCES, meta.get("model_sources")):
        problems.append("model_sources missing %s" % m)
    if meta.get("enable_internet") is not False:
        problems.append("enable_internet is %r, must be false"
                        % meta.get("enable_internet"))

    if problems:
        die(
            "the pulled kernel is not configured the way your baseline is.\n"
            "  " + "\n  ".join(problems) + "\n"
            "  Pushing now would silently drop inputs and burn a run."
        )
    # The real notebook carries accelerator=nvidiaRtxPro6000 while enable_gpu is
    # false. The accelerator field is what selects hardware, so this is only a note.
    if not meta.get("enable_gpu"):
        say("  note: enable_gpu is falsy, machine_shape=%r. The accelerator field"
            % meta.get("machine_shape"))
        say("        selects hardware on this kernel, so that combination is expected.")
    say("  -> all 5 inputs present, internet off. OK to push.")


def guard_slug(meta: dict) -> None:
    title = (meta.get("title") or "").strip()
    kslug = (meta.get("id") or "").split("/")[-1]
    if not title or not kslug:
        return
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")
    if slug and slug != kslug.lower():
        say("  note: title %r slugifies to %r but the id slug is %r."
            % (title, slug, kslug))
        say("        The id in the metadata wins; check the URL the push prints.")


def needs_accelerator_flag(meta: dict) -> bool:
    return not (meta.get("machine_shape") or "").strip()


def find_hook_heading(nb: dict) -> int | None:
    for i, c in enumerate(nb.get("cells", [])):
        if _is_md(c):
            t = cell_text(c).lower()
            if HOOK_HEADING_SUBSTR in t:
                return i
    return None


def has_knob_channel(nb: dict) -> bool:
    for c in nb.get("cells", []):
        if AGENT_CHANNEL_MARKER in cell_text(c):
            return True
    return find_hook_heading(nb) is not None


def is_hand_lever_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = cell_text(cell)
    if "Lever 2 — add .grid" in src or "sb = python_tool_sandbox._SANDBOX_BOOTSTRAP" in src:
        return True
    if "from inference.agent import prompts, python_tool_sandbox, tool_agent" in src:
        return True
    return False


def is_empty_code_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = cell_text(cell).strip()
    outputs = cell.get("outputs", [])
    return not src and not outputs


def is_misclassified_markdown(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    src = cell_text(cell).strip()
    if not src:
        return False
    if src.startswith("%%") or "writefile" in src or "_K5" in src or LEVER_TAG in src:
        return False
    lines = [l.strip() for l in src.splitlines() if l.strip()]
    if not lines:
        return False
    first = lines[0]
    is_md_header = first.startswith(("# ", "## ", "### ", "**", "- "))
    has_bold_or_update = any(l.startswith(("**", "* ", "- ", "> ")) or "update (v" in l.lower() for l in lines)
    if not (is_md_header and has_bold_or_update):
        return False
    try:
        compile(src, "<cell>", "exec")
        return False
    except SyntaxError:
        return True


def sanitize_notebook(nb: dict) -> dict:
    nb = json.loads(json.dumps(nb))
    cells = nb.get("cells", [])
    drop = set()
    for i, c in enumerate(cells):
        if is_empty_code_cell(c):
            say("  [sanitizer] removed empty output-less code cell at index %d" % i)
            drop.add(i)
        elif is_hand_lever_cell(c):
            say("  [sanitizer] removed hand lever-2 prototype cell at index %d" % i)
            drop.add(i)
        elif is_misclassified_markdown(c):
            say("  [sanitizer] converted misclassified markdown cell at index %d to markdown" % i)
            c["cell_type"] = "markdown"
            c.pop("execution_count", None)
            c.pop("outputs", None)
    if drop:
        nb["cells"] = [c for i, c in enumerate(cells) if i not in drop]
    return nb


def find_knob_cell_index(nb: dict) -> int | None:
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code" and (knob_marker() in cell_text(c) or AGENT_CHANNEL_MARKER in cell_text(c)):
            return i
    return None


# --------------------------------------------------------------------------
# notebook surgery
# --------------------------------------------------------------------------

def cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def _is_md(cell: dict) -> bool:
    return cell.get("cell_type") == "markdown"


def _heading_match(cell: dict, target: str, mode: str) -> str | None:
    if not _is_md(cell):
        return None
    for line in cell_text(cell).splitlines():
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            continue
        head = norm(stripped.lstrip("#"))
        if (mode == "eq" and head == target) or (mode == "prefix"
                                                 and head.startswith(target)):
            return line.strip()
    return None


def find_injection_point(nb: dict, explicit_index: int | None) -> int:
    """Index to insert at: end of the anchor section, i.e. just before the next
    markdown heading. Inserting before the next heading rather than at
    `heading + 1` keeps this correct when a section holds several code cells."""
    cells = nb.get("cells", [])
    if explicit_index is not None:
        if not 0 <= explicit_index <= len(cells):
            die("--inject-at %d out of range (notebook has %d cells)"
                % (explicit_index, len(cells)))
        return explicit_index

    hit = None
    for i, c in enumerate(cells):
        line = _heading_match(c, norm(INJECT_AFTER_HEADING), "eq")
        if line:
            hit = ("after %r" % line, i)
            break
    if hit is None:
        for i, c in enumerate(cells):
            line = _heading_match(c, norm(INJECT_BEFORE_HEADING), "prefix")
            if line:
                say("  fallback anchor: heading %r at cell %d -> insert at %d"
                    % (line, i, i))
                return i
    if hit is None:
        die(
            "no anchor found. Run stage=check, read the heading map, then pass --inject-at N.\n"
            "  Expected a markdown heading matching %r or starting %r."
            % (INJECT_AFTER_HEADING, INJECT_BEFORE_HEADING)
        )

    label, i = hit
    for j in range(i + 1, len(cells)):
        if _is_md(cells[j]):
            say("  anchor: %s at cell %d, next heading at cell %d -> insert at %d"
                % (label, i, j, j))
            return j
    say("  anchor: %s at cell %d, no later heading -> insert at end (%d)"
        % (label, i, len(cells)))
    return len(cells)


def guard_upstream_names(nb: dict, at: int) -> None:
    """The applier cell reads WORKING_DIR and BUNDLE_DIR. If section 1 does not
    define them the lever would NameError nine hours into a run."""
    before = "".join(cell_text(c) for c in nb.get("cells", [])[:at])
    absent = [n for n in REQUIRED_UPSTREAM_NAMES
              if not re.search(r"\b%s\s*=" % n, before)]
    if absent:
        die(
            "the cells before the injection point do not define: %s\n"
            "  The lever applier reads those names. Check that section 1 is intact,\n"
            "  or pass --inject-at N to place the lever after the cell that defines them."
            % ", ".join(absent)
        )
    say("  upstream names present before the lever: %s"
        % ", ".join(REQUIRED_UPSTREAM_NAMES))


def applier_marker(lever_tag: str) -> str:
    """Goes in the applier cell. The patch cell cannot carry a comment marker:
    `%%writefile` is a cell magic and must be the first line, so a leading
    comment would turn it into plain text and write nothing."""
    return "# ===== %s apply (kaggle5) =====" % lever_tag


def knob_marker() -> str:
    return "# ===== k5 knobs (kaggle5) ====="


def _is_writefile(cell: dict, filename: str) -> bool:
    lines = cell_text(cell).splitlines()
    return bool(lines) and lines[0].startswith("%%writefile") \
        and filename in lines[0]


def lever_anchor(nb: dict, lever_tag: str) -> int | None:
    """Index of the applier cell, or None. The patch cell is the one before it."""
    mk = applier_marker(lever_tag)
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code" and mk in cell_text(c):
            return i
    return None


def lever_cells_present(nb: dict, lever_tag: str) -> list[int]:
    i = lever_anchor(nb, lever_tag)
    if i is None:
        return []
    cells = nb.get("cells", [])
    out = []
    if i > 0 and _is_writefile(cells[i - 1], LEVER_PATCH_FILE):
        out.append(i - 1)
    out.append(i)
    if i + 1 < len(cells) and knob_marker() in cell_text(cells[i + 1]):
        out.append(i + 1)
    return out


VARIANT_LINE = 'VARIANT = "lever3"  # k5-variant-line'


def set_variant(nb: dict, variant: str) -> dict:
    """Bake the requested variant into the applier cell (idempotent).

    lever3 : memory + hard prune gate + prompt directives (default)
    kernel : memory view present, gate off, prompt untouched (A/B rollback)
    """
    if variant == "lever3":
        return nb
    anchor = lever_anchor(nb, LEVER_TAG)
    if anchor is None:
        die("variant rewrite: no applier cell found in the pulled notebook")
    cell = nb["cells"][anchor]
    src = cell_text(cell)
    if VARIANT_LINE not in src:
        die("variant rewrite: the applier cell has no %r line; refusing to "
            "rewrite blindly" % VARIANT_LINE)
    cell["source"] = src.replace(
        VARIANT_LINE, 'VARIANT = "%s"  # k5-variant-line' % variant, 1)
    say("  variant=%s baked into the applier cell" % variant)
    return nb


def build_lever_cells(patch_text: str, applier: str, lever_tag: str) -> list[dict]:
    if ("'" * 3) in patch_text:
        die(LEVER_PATCH_FILE
            + " contains a triple quote; cannot embed it in a writefile cell")
    body = patch_text if patch_text.endswith(chr(10)) else patch_text + chr(10)
    write = "%%writefile /kaggle/working/" + LEVER_PATCH_FILE + chr(10) + body
    app = applier_marker(lever_tag) + chr(10) + applier.strip() + chr(10)
    return [
        {"cell_type": "code", "metadata": {"trusted": True,
                                           "kaggle5": lever_tag + "-patch"},
         "execution_count": None, "outputs": [], "source": write},
        {"cell_type": "code", "metadata": {"trusted": True,
                                           "kaggle5": lever_tag + "-apply"},
         "execution_count": None, "outputs": [], "source": app},
    ]


def build_knob_cell(payload: dict) -> dict:
    src = (
        knob_marker() + chr(10) +
        "# Injected after ## 7. Customization hook" + chr(10) +
        "# Applies assignments allowlist-only, loudly refuses unknown names / uncaps / bools," + chr(10) +
        "# dies if the upstream harness renamed an attribute, prints '[k5] bm.solver.x: old -> new'" + chr(10) +
        "# per knob, and writes /kaggle/working/k5_knobs.json." + chr(10) +
        "import json as _k5_json" + chr(10) +
        "import pathlib as _k5_pathlib" + chr(10) + chr(10) +
        "# JSON literal aliases so valid JSON is valid Python at runtime" + chr(10) +
        "null = None" + chr(10) +
        "false = False" + chr(10) +
        "true = True" + chr(10) + chr(10) +
        "_K5 = " + json.dumps(payload, indent=2, sort_keys=True) + chr(10) + chr(10) +
        "_K5_SURFACE = {" + chr(10) +
        "    'analyzer_timeout': float," + chr(10) +
        "    'max_runtime_s_per_game': float," + chr(10) +
        "    'max_actions_per_game': (int, type(None))," + chr(10) +
        "    'concurrency': int," + chr(10) +
        "}" + chr(10) + chr(10) +
        "if 'bm' not in globals() or not hasattr(bm, 'solver'):" + chr(10) +
        "    raise RuntimeError('[k5] bm.solver not found in globals')" + chr(10) + chr(10) +
        "for _k, _v in _K5.get('knobs', {}).items():" + chr(10) +
        "    if _k not in _K5_SURFACE:" + chr(10) +
        "        raise ValueError(f'[k5] Refusing unknown knob: {_k}')" + chr(10) +
        "    if isinstance(_v, bool):" + chr(10) +
        "        raise TypeError(f'[k5] Refusing boolean knob value: {_k}={_v}')" + chr(10) +
        "    if not hasattr(bm.solver, _k):" + chr(10) +
        "        raise AttributeError(f'[k5] Upstream harness has no attribute bm.solver.{_k}')" + chr(10) +
        "    _old = getattr(bm.solver, _k)" + chr(10) +
        "    if _old is not None and _v is None:" + chr(10) +
        "        raise ValueError(f'[k5] Refusing uncap for {_k}')" + chr(10) +
        "    setattr(bm.solver, _k, _v)" + chr(10) +
        "    print(f'[k5] bm.solver.{_k}: {_old} -> {_v}')" + chr(10) + chr(10) +
        "_k5_pathlib.Path('/kaggle/working').mkdir(parents=True, exist_ok=True)" + chr(10) +
        "_k5_pathlib.Path('/kaggle/working/" + KNOB_JSON_NAME + "').write_text(" + chr(10) +
        "    _k5_json.dumps(_K5, indent=1, sort_keys=True)" + chr(10) +
        ")" + chr(10) +
        "print('[k5] run binding: variant=%r changed=%r'" + chr(10) +
        "      % (_K5.get('variant'), _K5.get('changed')))" + chr(10) +
        "del _k5_json, _k5_pathlib, _K5, _K5_SURFACE" + chr(10)
    )
    return {"cell_type": "code",
            "metadata": {"trusted": True, "kaggle5": "knobs"},
            "execution_count": None, "outputs": [], "source": src}


def inject(nb: dict, patch_text: str, applier: str, lever_tag: str,
           explicit_index: int | None = None,
           knob_payload: dict | None = None) -> dict:
    nb = json.loads(json.dumps(nb))
    nb = sanitize_notebook(nb)

    lever_cells = build_lever_cells(patch_text, applier, lever_tag)
    present = lever_cells_present(nb, lever_tag)
    if present:
        say("  %s already present at cells %s — replacing in place, no duplicates"
            % (lever_tag, present))
        drop = set(present)
        keep = [c for i, c in enumerate(nb["cells"]) if i not in drop]
        at = min(present)
        keep[at:at] = lever_cells
        nb["cells"] = keep
    else:
        at = find_injection_point(nb, explicit_index)
        guard_upstream_names(nb, at)
        nb["cells"][at:at] = lever_cells
        say("  inserted %s at cells %d-%d" % (lever_tag, at, at + len(lever_cells) - 1))

    if knob_payload is not None:
        k_cell = build_knob_cell(knob_payload)
        existing_k_idx = find_knob_cell_index(nb)
        if existing_k_idx is not None:
            nb["cells"][existing_k_idx] = k_cell
            say("  replaced existing knob cell at cell %d in place" % existing_k_idx)
        else:
            h_idx = find_hook_heading(nb)
            if h_idx is not None:
                code_idx = None
                for j in range(h_idx + 1, len(nb["cells"])):
                    if _is_md(nb["cells"][j]):
                        break
                    if nb["cells"][j].get("cell_type") == "code":
                        code_idx = j
                ins_idx = (code_idx + 1) if code_idx is not None else (h_idx + 1)
                nb["cells"].insert(ins_idx, k_cell)
                say("  inserted knob cell at cell %d (after hook heading)" % ins_idx)
            else:
                anchor = lever_anchor(nb, lever_tag)
                ins_idx = (anchor + 1) if anchor is not None else len(nb["cells"])
                nb["cells"].insert(ins_idx, k_cell)
                say("  inserted knob cell at cell %d" % ins_idx)

    return nb


def verify_injection(nb: dict, patch_text: str, lever_tag: str,
                     knob_payload: dict | None = None) -> None:
    anchor = lever_anchor(nb, lever_tag)
    if anchor is None:
        die("post-check: no applier cell found")
    cells = nb.get("cells", [])
    if anchor == 0 or not _is_writefile(cells[anchor - 1], LEVER_PATCH_FILE):
        die("post-check: the cell before the applier is not a %%writefile "
            + LEVER_PATCH_FILE + " cell")
    if patch_text.strip() not in cell_text(cells[anchor - 1]):
        die("post-check: patch body not intact in the writefile cell")
    apply_src = cell_text(cells[anchor])
    if "git" not in apply_src or "apply" not in apply_src:
        die("post-check: the applier cell does not git-apply the patch")
    say("  post-check OK: writefile cell + applier, patch intact")

    if knob_payload is not None:
        k_idx = find_knob_cell_index(nb)
        if k_idx is None:
            die("post-check: expected knob cell not found")
        knob_src = cell_text(cells[k_idx])
        m = re.search(r"_K5 = (\{.*?\})" + chr(10) + chr(10) + "_K5_SURFACE", knob_src, re.S)
        if not m:
            die("post-check: could not find _K5 payload dictionary in knob cell")
        try:
            got_payload = json.loads(m.group(1))
        except json.JSONDecodeError as exc:
            die("post-check: knob cell payload does not parse as JSON: %s" % exc)
        if got_payload.get("format") != 2:
            die("post-check: expected format 2 payload in knob cell, got %r" % got_payload.get("format"))
        say("  knob cell present and parses OK")


# --------------------------------------------------------------------------
# stateless cycle discovery + submission table (read-only)
# --------------------------------------------------------------------------

def kernel_status() -> str:
    out = kaggle(["kernels", "status", f"{OWNER}/{KERNEL_SLUG}"],
                 quiet=True).stdout
    m = re.search(r'has status "([^"]+)"', out)
    return (m.group(1) if m else out.strip()).lower()


def kernel_has_submission_file(version: int | None = None) -> bool:
    spec = f"{OWNER}/{KERNEL_SLUG}" + (f"/{version}" if version else "")
    r = kaggle(["kernels", "files", spec, "--format", "json"],
               allow_fail=True, quiet=True)
    if r.returncode != 0 or not r.stdout.strip():
        return False
    try:
        data = _json_from(r.stdout)
        if isinstance(data, list):
            return any(f.get("name") == SUBMISSION_FILE for f in data)
    except Exception:
        pass
    return False


def submissions() -> list[dict]:
    r = kaggle(["competitions", "submissions", COMPETITION, "--format", "json",
                "--page-size", "20"], allow_fail=True, quiet=True)
    if r.returncode != 0:
        say("  competitions submissions: unavailable (rc=%d) — starting with empty submissions" % r.returncode)
        if r.stdout and r.stdout.strip():
            say("  " + r.stdout.strip().splitlines()[0])
        return []
    data = _json_from(r.stdout)
    if not isinstance(data, list):
        say("  competitions submissions: response not a JSON list — treating as empty")
        return []
    return data


def local_today(offset_hours: int = LOCAL_UTC_OFFSET_HOURS) -> str:
    env = os.environ.get("K5_TODAY", "").strip()   # testing hook, documented
    if env:
        return env
    return (datetime.now(timezone.utc)
            + timedelta(hours=offset_hours)).strftime("%Y-%m-%d")


def date_of(sub: dict) -> str:
    return str(sub.get("date") or "")[:10]


def to_float(x) -> float | None:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _days_old(ymd: str, today: str) -> int:
    try:
        a = datetime.strptime(ymd, "%Y-%m-%d")
        b = datetime.strptime(today, "%Y-%m-%d")
        return (b - a).days
    except ValueError:
        return 999


# --------------------------------------------------------------------------
# PIECE ONE — the ledger that survives the night
#
# Append-only JSON lines, event-sourced: nothing is ever rewritten, new facts
# are new lines. `experiments.jsonl` carries one experiment row and its life
# events (pushed -> scored/zero/error / filtered / stale / abandoned).
# `game_rows.jsonl` carries the per-game telemetry. Every experiment row
# records the FULL knob state plus the agent source hash, so any row is
# reproducible from the ledger alone.
# --------------------------------------------------------------------------

EXP_FIELDS = ("exp", "variant", "device", "date", "changed", "knobs", "agent_sha256",
              "reason")


class Ledger:
    def __init__(self, ldir: Path):
        self.dir = ldir
        self.dir.mkdir(parents=True, exist_ok=True)
        for name in LEDGER_FILES:
            f = self.dir / name
            if not f.exists():
                f.write_text("")

    def path(self, name: str) -> Path:
        return self.dir / name

    # ---- raw IO (append-only; read tolerates and counts bad lines) --------
    def read_lines(self, name: str) -> list[str]:
        p = self.path(name)
        return [ln for ln in p.read_text().splitlines() if ln.strip()]

    def events(self, name: str = "experiments.jsonl") -> list[dict]:
        out, bad = [], 0
        for ln in self.read_lines(name):
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    out.append(obj)
                    continue
            except json.JSONDecodeError:
                pass
            bad += 1
        if bad:
            say("  ledger: %d unreadable line(s) in %s (kept untouched, skipped)"
                % (bad, name))
        return out

    def append(self, obj: dict, name: str = "experiments.jsonl") -> None:
        obj = dict(obj)
        obj.setdefault("at",
                       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        line = json.dumps(obj, sort_keys=True)
        with self.path(name).open("a") as fh:
            fh.write(line + "\n")

    # ---- remote sync ------------------------------------------------------
    def merge_remote_dir(self, remote: Path) -> int:
        """Union of exact lines, local order first, remote additions appended.
        Append-only sets make this idempotent and order-stable."""
        added = 0
        for name in LEDGER_FILES:
            rp = remote / name
            if not rp.is_file():
                continue
            have = set(self.read_lines(name))
            new = [ln for ln in rp.read_text().splitlines()
                   if ln.strip() and ln not in have]
            if new:
                with self.path(name).open("a") as fh:
                    for ln in new:
                        fh.write(ln + "\n")
                added += len(new)
        return added

    def sync_in(self) -> None:
        """Pull the ledger dataset. Absent dataset / no auth / first run are
        all normal: the local files are the source of truth either way."""
        tmp = Path(tempfile.mkdtemp(prefix="k5_ledger_in_"))
        r = kaggle(["datasets", "download", "-d", LEDGER_DATASET,
                    "-p", str(tmp), "--unzip", "-q"],
                   allow_fail=True, quiet=True)
        if r.returncode != 0:
            say("  ledger sync-in: dataset %s not downloaded (first run? no "
                "auth yet?) — starting from the local copy" % LEDGER_DATASET)
            return
        added = self.merge_remote_dir(tmp)
        say("  ledger sync-in: merged %d remote line(s) from %s"
            % (added, LEDGER_DATASET))

    def sync_out(self, note: str) -> None:
        """Push a new dataset version. Tried, in order: datasets version ->
        datasets create -> GitHub commit (only if a token env var exists) ->
        a loud manual instruction. The local copy is always intact regardless.
        """
        meta = {
            "title": "kaggle5-ledger",
            "id": LEDGER_DATASET,
            "licenses": [{"name": "CC0-1.0"}],
        }
        (self.dir / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
        (self.dir / "datasets-metadata.json").write_text(json.dumps(meta, indent=1))
        r = kaggle(["datasets", "version", "-p", str(self.dir),
                    "-m", note[:250], "-t", "-q"], allow_fail=True, quiet=True)
        if r.returncode == 0:
            say("  ledger sync-out: new dataset version (%s)" % note)
            return
        r = kaggle(["datasets", "create", "-p", str(self.dir), "-t", "-q"],
                   allow_fail=True, quiet=True)
        if r.returncode == 0:
            say("  ledger sync-out: dataset created (%s)" % LEDGER_DATASET)
            return
        if r.stdout and r.stdout.strip():
            say("  ledger create note: %s" % r.stdout.strip().splitlines()[0])
        if self._github_fallback(note):
            return
        say("")
        say("!" * 66)
        say("LEDGER SYNC FAILED — the ledger is safe locally at")
        say("  %s" % self.dir)
        say("but it did NOT reach the dataset or GitHub. Rescue it by hand:")
        say("  1. open this run's output files on kaggle.com (they persist), or")
        say("  2. paste the two .jsonl files into the kaggle5-ledger dataset, or")
        say("  3. run `kaggle datasets version -p <dir> -m manual -t` anywhere.")
        say("!" * 66)

    def _github_fallback(self, note: str) -> bool:
        """Commit the ledger to the repo. Only attempted when a token was
        deliberately provided as an env var; never silently skipped — the
        caller announces loudly when this path is used."""
        token = (os.environ.get("K5_GITHUB_TOKEN")
                 or os.environ.get("GITHUB_TOKEN") or "").strip()
        if not token:
            return False
        say("  ledger sync-out: dataset path failed; committing to GitHub "
            "(K5_GITHUB_TOKEN is set)")
        ok = True
        for name in LEDGER_FILES:
            ok = self._github_put(
                "mohamtur1/arc-agi-3-agent", "kaggle5/ledger/" + name,
                self.path(name).read_bytes(), "kaggle5 ledger: " + note[:100],
                token) and ok
        if ok:
            say("  ledger sync-out: committed to GitHub kaggle5/ledger/")
        else:
            say("  ledger sync-out: GitHub commit failed too")
        return ok

    @staticmethod
    def _github_put(repo: str, path: str, content: bytes, message: str,
                    token: str) -> bool:
        import base64
        api = "https://api.github.com/repos/%s/contents/%s" % (repo, path)
        headers = {"Authorization": "Bearer " + token,
                   "Accept": "application/vnd.github+json",
                   "User-Agent": "kaggle5-ledger"}
        try:
            req = urllib.request.Request(api, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                sha = json.loads(resp.read().decode()).get("sha")
        except Exception:
            sha = None
        body = {"message": message, "branch": "main",
                "content": base64.b64encode(content).decode()}
        if sha:
            body["sha"] = sha
        try:
            req = urllib.request.Request(
                api, data=json.dumps(body).encode(), method="PUT",
                headers=dict(headers, **{"Content-Type": "application/json"}))
            with urllib.request.urlopen(req, timeout=30) as resp:
                return 200 <= resp.status < 300
        except Exception as exc:
            say("    github put %s failed: %r" % (path, exc))
            return False

    # ---- fold: event log -> current view of the world ---------------------
    def fold(self) -> "View":
        view = View()
        for ev in self.events("experiments.jsonl"):
            kind = ev.get("kind")
            if kind == "import":
                view.imports.append(ev)
            elif kind == "experiment":
                exp = {k: ev.get(k) for k in EXP_FIELDS}
                exp.update(status="pending", score=None, version=None,
                           outcome=None, adopted=None, ref=None, per_game=[])
                view.experiments[ev["exp"]] = exp
                view.order.append(ev["exp"])
            elif kind == "pushed":
                exp = view.experiments.get(ev.get("exp"))
                if exp is not None:
                    exp["version"] = ev.get("version")
            elif kind == "score":
                exp = view.experiments.get(ev.get("exp"))
                if exp is not None:
                    exp.update(status=ev.get("outcome_status", "scored"),
                               score=ev.get("score"), ref=ev.get("ref"),
                               outcome=ev.get("verdict"),
                               adopted=ev.get("adopted"))
                    if ev.get("adopted"):
                        view.incumbent_knobs = dict(exp.get("knobs") or {})
                        view.incumbent_score = ev.get("score")
                        view.anchor_done = True
                    if ev.get("knob") is not None and (
                            ev.get("verdict") in ("kept", "worse", "equal")
                            or ev.get("score") is not None):
                        view.tried.add((canon(ev.get("knob")),
                                        canon(ev.get("value"))))
                    view.matched_refs.add(str(ev.get("ref")))
            elif kind == "gym":
                exp = view.experiments.get(ev.get("exp"))
                if exp is not None and exp["status"] == "pending":
                    exp["status"] = "filtered"
                if ev.get("verdict") == "FAIL":
                    view.tried.add((canon(ev.get("knob")), canon(ev.get("value"))))
            elif kind == "stale":
                exp = view.experiments.get(ev.get("exp"))
                if exp is not None and exp["status"] == "pending":
                    exp["status"] = "stale"
            elif kind == "abandon":
                exp = view.experiments.get(ev.get("exp"))
                if exp is not None and exp["status"] == "pending":
                    exp["status"] = "abandoned"
        for ev in self.events("game_rows.jsonl"):
            exp = view.experiments.get(ev.get("exp"))
            if exp is not None:
                exp["per_game"] = list(ev.get("rows") or [])
                exp["per_game_dummy"] = bool(ev.get("dummy"))
        return view


class View:
    """The folded state. Everything the planner needs, nothing else."""
    def __init__(self):
        self.imports = []
        self.experiments = {}
        self.order = []
        self.incumbent_knobs = dict(DEFAULT_KNOBS)
        self.incumbent_score = None
        self.anchor_done = False
        self.tried = set()          # {(canon(knob), canon(value))} measured-or-filtered
        self.matched_refs = set()

    def open_pending(self):
        for eid in self.order:
            exp = self.experiments[eid]
            if exp["status"] == "pending":
                return exp
        return None

    def next_exp_id(self) -> int:
        n = 0
        for eid in self.order:
            m = re.match(r"exp-(\d+)$", eid)
            if m:
                n = max(n, int(m.group(1)))
        return n + 1

    def table(self) -> str:
        lines = ["  %-8s %-9s %-26s %-24s %-9s %s"
                 % ("exp", "status", "change", "agent", "score", "verdict")]
        for eid in self.order:
            e = self.experiments[eid]
            ch = ", ".join("%s %r->%r" % (k, ov, nv)
                           for k, (ov, nv) in (e.get("changed") or {}).items()) \
                or "(baseline)"
            sc = "-" if e.get("score") is None else "%.4g" % e["score"]
            lines.append("  %-8s %-9s %-26s %-24s %-9s %s"
                         % (eid, e["status"], ch[:26],
                            str(e.get("agent_sha256") or "?")[:12] + "…",
                            sc, e.get("outcome") or ""))
        return "\n".join(lines)


# --------------------------------------------------------------------------
# PIECE TWO — read yesterday's score back
#
# The submission artifact is per-game (submission.parquet: one row per game),
# so an aggregate alone would throw away most of the signal. Two sources:
#   aggregate: `competitions submissions` — the public score per submission.
#   per-game : `kernels output <kernel>/<version>` -> k5_per_game.csv, the
#              telemetry file the v4 agent writes during the rerun.
# Parquet is never parsed here (stdlib-only contract); if only the parquet
# is present we record per-game as unavailable and say so.
# --------------------------------------------------------------------------

MSG_RE_TEMPLATE = r"kaggle5\s+{variant}\s+v(\d+)"
SENTINEL_GAME_IDS = {"1", "0", "dummy", "test", "example", "sample"}


def rows_look_dummy(parsed: list[dict]) -> bool:
    """The interactive (Phase-A) artifact is a hardcoded one-row submission
    so the commit succeeds; real scoring only happens on competition rerun.
    Dummy rows are labelled, never folded into results. Rule: a single row
    is always the dummy; otherwise only file-wide sentinel ids count, so a
    real game that happens to reuse a sentinel-looking id is never thrown
    away by mistake."""
    if len(parsed) <= 1:
        return True
    ids = [str(rw.get("game_id", "")).strip().lower() for rw in parsed]
    return bool(ids) and all(i in SENTINEL_GAME_IDS for i in ids)


def fetch_per_game(exp_id: str, version: int, led: Ledger) -> None:
    """Best-effort per-game telemetry for a scored experiment. Chain of
    custody: only ever read the output of the version that was pushed for
    THIS experiment and then competition-rerun (a completed submission was
    just reconciled against it). A Phase-A (interactive) output contains a
    hardcoded DUMMY submission — the sentinel filter labels such rows and
    they are never folded into results."""
    tmp = Path(tempfile.mkdtemp(prefix="k5_out_"))
    r = kaggle(["kernels", "output", f"{OWNER}/{KERNEL_SLUG}/{version}",
                "-p", str(tmp), "-q"], allow_fail=True, quiet=True)
    csv_path = tmp / "k5_per_game.csv"
    info_path = tmp / "k5_run_info.json"
    if r.returncode != 0 or not csv_path.is_file():
        led.append({"kind": "note", "exp": exp_id,
                    "text": "per-game telemetry unavailable for v%d "
                            "(old agent, or rerun outputs not exposed); "
                            "aggregate score was still recorded" % version})
        say("  per-game: no k5_per_game.csv for v%d (recorded as unavailable)"
            % version)
        return
    rows = []
    for ln in csv_path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        rows.append(ln)
    if not rows:
        dummy, parsed = True, []
    else:
        try:
            parsed = list(csv.DictReader(rows))
        except csv.Error:
            parsed = []
        dummy = rows_look_dummy(parsed)
    led.append({"kind": "game_rows", "exp": exp_id, "version": version,
                "source": "kernel_output", "dummy": dummy, "rows": parsed},
               name="game_rows.jsonl")
    if dummy:
        say("  per-game: rows for v%d look like the interactive DUMMY — "
            "labelled dummy=true, not folded into results" % version)
    if info_path.is_file():
        try:
            info = json.loads(info_path.read_text())
        except json.JSONDecodeError:
            info = None
        if info:
            led.append({"kind": "note", "exp": exp_id, "text":
                        "run self-report: agent_sha256=%s variant=%r applied=%r"
                        % (str(info.get("agent_sha256"))[:16],
                           info.get("variant"),
                           info.get("applied_overrides"))})


def reconcile(led: Ledger, subs: list[dict], today: str) -> int:
    """Attach new submission outcomes to pending experiments. Returns the
    number of score events written. Matching: (1) the message I ask the
    human to paste (variant+version, strongest), (2) FIFO by date for
    anything unmatched (submission date must be on/after the push date).
    Zero and errored submissions are recorded as data."""
    view = led.fold()
    pend_ids = [eid for eid in view.order
                if view.experiments[eid]["status"] == "pending"]
    fresh = []
    for s in subs:
        ref = str(s.get("ref", ""))
        if ref and ref in view.matched_refs:
            continue
        status = str(s.get("status") or "").lower()
        if status not in ("complete", "error"):
            continue
        fresh.append(s)
    fresh.sort(key=date_of)                      # chronological application

    written = 0
    for s in fresh:
        desc = str(s.get("description") or "")
        ref = str(s.get("ref", ""))
        status = str(s.get("status") or "").lower()
        score = to_float(s.get("publicScore"))
        pick = None
        for eid in pend_ids:
            exp = view.experiments[eid]
            m = re.search(MSG_RE_TEMPLATE.format(variant=re.escape(
                str(exp["variant"]))), desc)
            if m and (exp["version"] is None
                      or int(m.group(1)) == int(exp["version"])):
                pick = eid
                break
        if pick is None:
            for eid in pend_ids:                 # FIFO fallback, date-guarded
                if date_of(s) >= (view.experiments[eid]["date"] or ""):
                    pick = eid
                    break
        if pick is None:
            led.append({"kind": "note", "text":
                        "submission ref=%s (%s, score=%s) matched no pending "
                        "experiment — recorded as a note only"
                        % (ref, desc[:60], s.get("publicScore"))})
            continue
        exp = view.experiments[pick]
        pend_ids.remove(pick)

        inc = view.incumbent_score
        if status == "error":
            outcome_status, verdict, adopted = "error", "worse", False
        else:
            outcome_status = "zero" if score == 0.0 else "scored"
            if inc is None:
                verdict, adopted = "anchor", True
            elif score is None:
                verdict, adopted = "unknown", False
            elif score >= inc - 1e-9:
                verdict = "kept" if score > inc + 1e-9 else "equal"
                adopted = True
            else:
                verdict, adopted = "worse", False

        changed = exp.get("changed") or {}
        knob = next(iter(changed), None)
        ev = {"kind": "score", "exp": pick, "ref": ref,
              "submission_date": date_of(s), "status": status,
              "outcome_status": outcome_status, "score": score,
              "incumbent_score_before": inc, "verdict": verdict,
              "adopted": adopted,
              "knob": knob, "value": (changed.get(knob) or [None, None])[1]
              if knob else None}
        led.append(ev)
        written += 1
        say("  reconciled %s: submission ref=%s status=%s score=%s "
            "(incumbent was %s) -> %s%s"
            % (pick, ref, status,
               ("%.4g" % score) if score is not None else "?",
               ("%.4g" % inc) if inc is not None else "-",
               verdict,
               " — this value is the new incumbent" if adopted else ""))
        if view.experiments[pick]["version"] is not None and status != "error":
            fetch_per_game(pick, int(view.experiments[pick]["version"]), led)
        # keep the local view current for the next match in this batch
        view = led.fold()

    # stale watch: pendings that outlived their welcome
    for eid in list(pend_ids):
        exp = view.experiments[eid]
        age = _days_old(exp["date"], today)
        if age > STALE_AFTER_DAYS:
            led.append({"kind": "stale", "exp": eid,
                        "reason": "pending %d days with no submission "
                                  "(>%d). Tombstoned; the value may be "
                                  "re-planned later." % (age, STALE_AFTER_DAYS)})
            say("  %s is %d days pending with no submission -> stale" % (eid, age))
    return written


# --------------------------------------------------------------------------
# PIECE THREE — pick the next experiment from the ledger
# Fixed queue, one knob at a time, documented backtrack. The choice is a pure
# function of the ledger, and the reason is written into the pending row.
# --------------------------------------------------------------------------

def plan_next(view: View, today: str):
    """Return the plan dict, or None when the queue is exhausted."""
    if not view.anchor_done:
        return {"kind": "anchor", "variant": ANCHOR_VARIANT,
                "changed": {}, "knobs": dict(DEFAULT_KNOBS),
                "reason": "anchor: first measured run of %s; knobs at the "
                          "documented defaults; binds the ledger to the kernel's "
                          "own bytes before any knob moves" % DEVICE}

    for knob, ladder, why in EXPERIMENT_QUEUE:
        dead = False
        for value in ladder:
            key = (canon(knob), canon(value))
            if key in view.tried:
                continue
            if dead:
                continue
            # was any EARLIER entry on this ladder scored worse?
            for eid in view.order:
                e = view.experiments[eid]
                if (e["status"] in ("scored", "zero")
                        and e.get("outcome") == "worse"):
                    ch = e.get("changed") or {}
                    if knob in ch:
                        dead = True
            if dead:
                continue
            old = view.incumbent_knobs.get(knob, DEFAULT_KNOBS.get(knob))
            knobs = dict(view.incumbent_knobs)
            knobs[knob] = value
            if canon(old) == canon(value):
                view.tried.add(key)
                continue
            eid = "exp-%03d" % view.next_exp_id()
            return {"kind": "experiment", "exp": eid, "variant": eid,
                    "changed": {knob: [old, value]}, "knobs": knobs,
                    "reason": "queue: %s | %s" % (knob, why)}
    return None


def block_experiment_event(plan: dict, today: str) -> dict:
    """The pending row. Written BEFORE the push and synced, so a crash can
    never lose what today was supposed to measure."""
    eid = plan.get("exp") or "exp-%03d" % 0
    ev = {"kind": "experiment", "exp": plan.get("exp", eid),
          "variant": plan["variant"], "device": DEVICE, "date": today,
          "changed": plan["changed"], "knobs": plan["knobs"],
          "agent_file": AGENT_FILE, "agent_sha256": AGENT_SHA256,
          "reason": plan["reason"]}
    if plan["kind"] == "anchor":
        ev["exp"] = "exp-000"
        ev["variant"] = ANCHOR_VARIANT
    return ev


def knob_payload_for(plan: dict, today: str) -> dict:
    return {"format": 2, "variant": plan["variant"], "device": DEVICE,
            "date": today, "changed": plan["changed"], "reason": plan["reason"],
            "knobs": plan["knobs"]}


def validate_static_gate(incumbent_knobs: dict, plan: dict) -> tuple[bool, str]:
    """Static gate: RULES ONLY.
    - exactly one knob changed (unless anchor)
    - knob in surface
    - no booleans
    - never uncap (None -> value allowed, value -> None refused)
    - |log2(new/old)| <= 1 for positive numerics
    - canon-equal refused
    THE HARNESS HAS NO LOCAL PROXY (wallet-budget LLM solver, 9h GPU run).
    Never present filter results as predictions or scores.
    """
    variant = plan.get("variant", "")
    changed = plan.get("changed", {})
    new_knobs = plan.get("knobs", {})

    if variant == ANCHOR_VARIANT or plan.get("kind") == "anchor" or not changed:
        if new_knobs != DEFAULT_KNOBS:
            return False, "anchor knobs do not match baseline surface defaults"
        return True, "anchor accepted: full surface at baselines"

    if len(changed) != 1:
        return False, f"gate refused: expected exactly 1 knob changed, got {len(changed)} ({list(changed.keys())})"

    knob, (old_val, new_val) = next(iter(changed.items()))
    if knob not in KNOB_SURFACE:
        return False, f"gate refused: knob {knob!r} not in KNOB_SURFACE {sorted(KNOB_SURFACE.keys())}"

    if isinstance(new_val, bool) or isinstance(old_val, bool):
        return False, f"gate refused: boolean values not permitted on {knob!r} ({new_val!r})"

    if canon(old_val) == canon(new_val):
        return False, f"gate refused: proposed value {new_val!r} is canonically equal to old value {old_val!r}"

    if old_val is not None and new_val is None:
        return False, f"gate refused: uncap attempted on {knob!r} ({old_val!r} -> None is refused)"

    if (isinstance(old_val, (int, float)) and old_val > 0 and
        isinstance(new_val, (int, float)) and new_val > 0):
        ratio = float(new_val) / float(old_val)
        log2_val = abs(math.log2(ratio))
        if log2_val > 1.0 + 1e-9:
            return False, f"gate refused: |log2({new_val}/{old_val})| = {log2_val:.4f} > 1.0"

    return True, f"gate passed: knob {knob} valid ({old_val} -> {new_val})"


# --------------------------------------------------------------------------
# PIECE TWO-B — the local filter. A smoke test in front of the real queue;
# never presented, stored, or read as a leaderboard number.
# --------------------------------------------------------------------------

def gym_json(out: str) -> dict:
    """Extract the gym's summary JSON from stdout. The agent prints its
    '[k5] knob overrides' line(s) too, so try full text, then text without
    those lines, then from the last line that opens a brace."""
    for cand in (out,
                 "\n".join(ln for ln in out.splitlines()
                           if not ln.startswith("[k5]"))):
        try:
            return json.loads(cand)
        except (ValueError, json.JSONDecodeError):
            pass
        if "\n{" in cand:
            try:
                return json.loads(cand[cand.rindex("\n{") + 1:])
            except (ValueError, json.JSONDecodeError):
                pass
    raise ValueError("no JSON object found in gym output")


def run_gym_gate(lever_dir: Path, incumbent_knobs: dict, plan: dict) -> dict:
    """Run k5_gym.py: incumbent vs candidate on five toy games. Returns a
    dict with verdict PASS/FAIL. A crash, a hang, a lost toy win, or a 2+
    toy-level drop FAILs the filter; anything else passes. Filter numbers
    are stored under local_filter in the ledger and are labelled there as
    NOT a score."""
    gym = lever_dir / GYM_FILE
    agent = lever_dir / AGENT_FILE
    if not gym.is_file() or not agent.is_file():
        return {"verdict": "SKIP",
                "verdict_reasons": ["gym or agent file missing next to kaggle5.py"]}
    tmp = Path(tempfile.mkdtemp(prefix="k5_gym_"))
    inc = tmp / "incumbent.json"
    cand = tmp / "candidate.json"
    inc.write_text(json.dumps({"variant": "incumbent",
                               "knobs": incumbent_knobs}, indent=1))
    cand.write_text(json.dumps({"variant": plan["variant"],
                                "knobs": plan["knobs"]}, indent=1))
    cmd = [sys.executable, str(gym), "--agent", str(agent),
           "--base-knobs", str(inc), "--knobs", str(cand), "--json"]
    say("$ " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return {"verdict": "FAIL", "verdict_reasons": ["gym ran over 600s (hang?)"]}
    out = r.stdout or ""
    try:
        data = gym_json(out)
    except ValueError:
        blob = out + (r.stderr or "")
        # A missing dependency means the FILTER could not run. That is an
        # environment fact, not evidence about the variant. Conflating the two
        # silences the loop forever on any host without the gym's imports
        # (a clean CI runner, for example), so it gets its own verdict.
        if "ModuleNotFoundError" in blob or "ImportError" in blob:
            missing = sorted(set(
                line.split("No module named", 1)[1].strip().strip("'\"")
                for line in blob.splitlines() if "No module named" in line))
            return {"verdict": "UNAVAILABLE",
                    "verdict_reasons": [
                        "gym could not run on this host: missing %s"
                        % (", ".join(m for m in missing if m) or "a dependency"),
                        blob[-300:]]}
        return {"verdict": "FAIL",
                "verdict_reasons": ["gym output unreadable", blob[-300:]]}
    say("  gym filter (a smoke test, NOT a score): verdict=%s %s"
        % (data.get("verdict"), "; ".join(data.get("verdict_reasons") or [])))
    for tag in ("incumbent", "candidate"):
        blk = data.get(tag) or {}
        say("    %-9s toy-levels=%s toy-wins=%s wall=%ss"
            % (tag, blk.get("levels"), blk.get("wins"), blk.get("wall_seconds")))
    return data


# --------------------------------------------------------------------------
# agent file guard — the pin that keeps source drift loud instead of silent
# --------------------------------------------------------------------------

def guard_agent_file(lever_dir: Path) -> str:
    p = lever_dir / AGENT_FILE
    if not p.is_file():
        die("%s not found next to kaggle5.py (fetch or embed step must place "
            "it there)" % AGENT_FILE)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != AGENT_SHA256:
        die(
            "%s sha256 %s… != pinned %s…\n"
            "  The agent changed without a re-pin. That is how the 0.06->0.07\n"
            "  delta became guesswork. If this change is intended: update\n"
            "  AGENT_SHA256 in kaggle5.py CONFIG (and AGENT_FILE if the name\n"
            "  changed), commit, and let tomorrow's run pick it up."
            % (AGENT_FILE, digest[:16], AGENT_SHA256[:16]))
    say("  agent pin OK: %s sha256 %s…" % (AGENT_FILE, digest[:16]))
    return digest


def check_defaults_drift(lever_dir: Path) -> None:
    """DEFAULT_KNOBS must match the baseline solver attributes — the anchor
    run must mean 'no overrides' (franzen-base)."""
    expected = {
        "analyzer_timeout": 900.0,
        "max_runtime_s_per_game": 31920.0,
        "max_actions_per_game": None,
        "concurrency": 120,
    }
    drift = {k: (DEFAULT_KNOBS.get(k), v)
             for k, v in expected.items()
             if canon(DEFAULT_KNOBS.get(k)) != canon(v)}
    if drift:
        die("DEFAULT_KNOBS drifted from baseline solver surface: %r" % drift)
    say("  defaults check OK: DEFAULT_KNOBS == baseline solver attributes")


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

VERSION_RE = re.compile(r"Kernel version (\d+) successfully")


def parse_push_version(stdout: str) -> int:
    m = VERSION_RE.search(stdout or "")
    if not m:
        die("no version number in the push output. Look for "
            "'Kernel version N successfully pushed'.")
    return int(m.group(1))


def print_human_card(plan: dict, version: int, today: str) -> None:
    changed = plan.get("changed") or {}
    delta = ", ".join("%s %r -> %r" % (k, ov, nv)
                      for k, (ov, nv) in changed.items()) or "(no knob change — anchor)"
    say("")
    say("=" * 66)
    say("PUSH COMPLETE — THE MACHINE STOPS HERE. IT NEVER SUBMITS.")
    say("=" * 66)
    say("  variant prepared : %s" % plan["variant"])
    say("  knob changed     : %s" % delta)
    say("  why (from ledger): %s" % plan["reason"])
    say("  kernel version   : v%d" % version)
    if plan.get("gym"):
        say("  local filter     : %s (a smoke test, NOT a score)"
            % plan["gym"].get("verdict"))
    say("")
    say("  YOUR THREE TAPS (the daily ~1-slot allowance is yours to spend):")
    say("  1. open  %s" % KERNEL_URL)
    say("  2. wait for version v%d to finish running, then" % version)
    say("     '...' menu -> Submit to %s -> pick version v%d"
        % (COMPETITION, version))
    say("  3. paste this exact message (it is how tomorrow matches the score):")
    say("       kaggle5 %s v%d" % (plan["variant"], version))
    say("")
    say("  Tomorrow's scheduled run reads the result and plans from it.")
    say("  If you skip today, this driver WAITS (no new push) for %d days,"
        % STALE_AFTER_DAYS)
    say("  then marks the row stale and moves on.")
    say("=" * 66)


def do_check(work: Path, patch_path: Path, applier_path: Path,
             inject_at: int | None, lever_dir: Path,
             variant: str = "lever3") -> None:
    probe_cli_flags()
    guard_agent_file(lever_dir)
    check_defaults_drift(lever_dir)
    code, meta = pull(work)
    guard_metadata(meta)
    guard_slug(meta)

    nb = json.loads(code.read_text())
    cells = nb.get("cells", [])
    say("")
    say("--- notebook map: %d cells ---" % len(cells))
    for i, c in enumerate(cells):
        first = cell_text(c).strip().splitlines()[0] if cell_text(c).strip() else ""
        say("  [%3d] %-4s %7dch  %s"
            % (i, "md" if _is_md(c) else "code", len(cell_text(c)), first[:66]))

    say("")
    say("  knob channel present in pulled notebook: %s"
        % ("YES" if has_knob_channel(nb) else
           "NO — ensure section 7 Customization hook heading is present"))

    try:
        at = find_injection_point(nb, inject_at)
    except SystemExit:
        say("no anchor matched — pass --inject-at N using the map above, then re-run check")
        return
    guard_upstream_names(nb, at)

    say("lever cells already present: %s" % (lever_cells_present(nb, LEVER_TAG) or "no"))

    if not patch_path.is_file() or not patch_path.read_text().strip():
        die("%s missing or empty — the lever is the patch, it must exist"
            % patch_path.name)
    applier = applier_path.read_text() if applier_path.is_file() else ""
    if not applier.strip():
        die("%s missing or empty" % applier_path.name)

    demo = {"kind": "experiment", "exp": "exp-000", "variant": "check-dry-run",
            "changed": {"EXPLORATION_CAP": [12, 24]}, "reason": "dry run",
            "knobs": dict(DEFAULT_KNOBS, EXPLORATION_CAP=24)}
    nb2 = inject(nb, patch_path.read_text(), applier, LEVER_TAG, inject_at,
                 knob_payload=knob_payload_for(demo, local_today()))
    nb2 = set_variant(nb2, variant)
    verify_injection(nb2, patch_path.read_text(), LEVER_TAG,
                     knob_payload=knob_payload_for(demo, local_today()))
    where = lever_cells_present(nb2, LEVER_TAG)
    say("dry-run injection: %d cells -> %d cells, lever+knobs at cells %s"
        % (len(cells), len(nb2["cells"]), where))
    say("")
    probe_submission_limits()
    say("")
    say("nothing was pushed. Next: stage=plan")


def do_push(work: Path, patch_path: Path, applier_path: Path,
            inject_at: int | None, allow_same: bool,
            plan: dict | None = None, allow_inert: bool = False,
            variant: str = "lever3") -> int:
    patch_text = patch_path.read_text() if patch_path.is_file() else ""
    applier = applier_path.read_text() if applier_path.is_file() else ""
    if not patch_text.strip():
        die("%s missing or empty" % patch_path.name)
    if not applier.strip():
        die("%s missing or empty" % applier_path.name)

    knob_payload = knob_payload_for(plan, local_today()) if plan else None

    code, meta = pull(work)
    nb = json.loads(code.read_text())
    say("pulled %s: %d cells" % (code.name, len(nb.get("cells", []))))
    guard_metadata(meta)
    guard_slug(meta)

    if plan and plan.get("changed") and not has_knob_channel(nb):
        msg = ("the pulled notebook has no knob channel (marker %r not found "
               "and hook heading not found), so %r would change NOTHING.\n"
               "  Pushing it would spend a run to produce zero information.\n"
               "  Fix: ensure section '## 7. Customization hook' is present in the kernel.\n"
               "  Or pass --allow-inert to push an inert variant anyway."
               % (AGENT_CHANNEL_MARKER, plan["variant"]))
        if allow_inert:
            say("  WARNING --allow-inert: " + msg.replace("\n", "\n  "))
        else:
            die(msg)
    elif plan and plan.get("changed") and allow_inert:
        pass

    before = json.dumps(nb, sort_keys=True)
    nb2 = inject(nb, patch_text, applier, LEVER_TAG, inject_at,
                 knob_payload=knob_payload)
    nb2 = set_variant(nb2, variant)
    verify_injection(nb2, patch_text, LEVER_TAG, knob_payload=knob_payload)

    if before == json.dumps(nb2, sort_keys=True):
        if not allow_same:
            die("the injected notebook is identical to what is already on Kaggle.\n"
                "  A push would create a version with no change. Edit the lever, or pass --allow-same.")
        say("  --allow-same given: pushing an unchanged notebook")

    code.write_text(json.dumps(nb2, indent=1))
    say("  wrote %s (%d cells)" % (code.name, len(nb2["cells"])))

    args = ["kernels", "push", "-p", str(work)]
    if needs_accelerator_flag(meta):
        say("  machine_shape empty on the pulled metadata; passing --accelerator %s"
            % FALLBACK_ACCELERATOR)
        args += ["--accelerator", FALLBACK_ACCELERATOR]
    out = kaggle(args, timeout=600).stdout
    return parse_push_version(out)


def do_wait(max_seconds: int = STATUS_MAX_SECONDS) -> bool:
    say("waiting on %s/%s (poll %ds, max %.1fh)"
        % (OWNER, KERNEL_SLUG, STATUS_POLL_SECONDS, max_seconds / 3600))
    deadline = time.time() + max_seconds
    last = None
    while time.time() < deadline:
        st = kernel_status()
        if st != last:
            say("  [%s] %s" % (time.strftime("%H:%M:%S"), st))
            last = st
        if st in ("complete", "cancelled", "error"):
            if st != "complete":
                die("the run ended as %r — do NOT submit. Read it with:\n"
                    "  kaggle kernels logs %s/%s" % (st, OWNER, KERNEL_SLUG))
            return True
        time.sleep(STATUS_POLL_SECONDS)
    say("still %r after %.1fh" % (last, max_seconds / 3600))
    return False


def refuse_submit(stage: str) -> None:
    say("")
    say("=" * 66)
    say("THE MACHINE NEVER PRESSES THE BUTTON.")
    say("=" * 66)
    say(" stage=%r no longer exists. `competitions submit` is not in this" % stage)
    say(" driver at all: grep the source. Every submission is finalized by a")
    say(" human in the Kaggle app, exactly like the first two.")
    say("")
    say(" To submit today's pushed version: open %s" % KERNEL_URL)
    say(" -> '...' -> Submit to the competition -> choose the version -> go.")
    raise SystemExit(1)


def do_daily(work: Path, patch_path: Path, applier_path: Path,
             inject_at: int | None, lever_dir: Path, skip_gym: bool,
             allow_inert: bool, variant: str = "lever3") -> None:
    """The scheduled loop. Idempotent within a day; safe to interrupt at
    any line; never submits."""
    today = local_today()
    say("=" * 66)
    say("daily run for %s (local, UTC%+d)" % (today, LOCAL_UTC_OFFSET_HOURS))
    say("=" * 66)

    probe_cli_flags()
    agent_sha = guard_agent_file(lever_dir)
    check_defaults_drift(lever_dir)

    led = Ledger(work / "ledger")
    led.sync_in()

    if not any(ev.get("kind") == "import" for ev in led.events()):
        for h in IMPORTED_HISTORY:
            eid = "import-%s" % h["variant"]
            led.append({"kind": "import", "exp": eid, "variant": h["variant"],
                        "score": h["score"], "date": "historical",
                        "note": h["note"]})
        say("  ledger: seeded %d historical anchors (v1=0.06, v2=0.07; "
            "source hashes unrecorded at the time — that is why this file "
            "now exists)" % len(IMPORTED_HISTORY))

    # ---- 1) read yesterday back ----
    say("")
    say("--- 1) reconcile: reading yesterday's submission(s) back ---")
    n = reconcile(led, submissions(), today)
    if n:
        led.sync_out("reconcile %s: %d score row(s)" % (today, n))   # write early
    say("  %d new score row(s) reconciled" % n)
    probe_submission_limits()

    view = led.fold()

    # ---- 2) hold / stale window for an unanswered pending row ----
    pend = view.open_pending()
    if pend is not None:
        st = kernel_status()
        ver = pend.get("version")
        has_sub = kernel_has_submission_file(ver)
        say("  pushed kernel status: %r (version %s), has submission file: %s"
            % (st, ver, has_sub))
        if any(w in st for w in ("running", "queued")) or "running" in st or "queued" in st:
            say("")
            say("--- pending row %s (%s) is still running on GPU (status %r) ---"
                % (pend["exp"], pend["variant"], st))
            say("  pushed as version v%s on %s; hold until it finishes."
                % (ver, pend.get("date")))
            say("  The driver pushes nothing today, on purpose.")
            led.sync_out("daily %s: holding on %s (running)" % (today, pend["exp"]))
            return
        elif not has_sub:
            say("")
            say("  pushed kernel version v%s completed without producing %s (status %r)."
                % (ver, SUBMISSION_FILE, st))
            say("  auto-abandoning %s to unblock re-planning." % pend["exp"])
            led.append({"kind": "abandon", "exp": pend["exp"],
                        "reason": "kernel run completed without %s (status %s)"
                                  % (SUBMISSION_FILE, st)})
            led.sync_out("abandon %s (kernel failed)" % pend["exp"])
            view = led.fold()
        else:
            say("")
            say("--- pending row %s (%s) is ready for manual submission ---"
                % (pend["exp"], pend["variant"]))
            say("  pushed as version v%s on %s; finished with %s."
                % (ver, pend.get("date"), SUBMISSION_FILE))
            say("  Either the human has not submitted (your move — see the card in")
            say("  that day's run log) or scoring is still in flight. The driver")
            say("  pushes nothing today, on purpose.")
            led.sync_out("daily %s: holding on %s (awaiting human submit)"
                         % (today, pend["exp"]))
            return

    # ---- 3) plan ----
    say("")
    say("--- 3) plan ---")
    plan = plan_next(view, today)
    if plan is None:
        say("  experiment queue exhausted. The incumbent IS the answer this")
        say("  system has to give. Incumbent knobs and score:")
        say("    score %.4g  knobs %s"
            % (view.incumbent_score, json.dumps(view.incumbent_knobs,
                                                sort_keys=True)))
        say("")
        say(view.table())
        say("")
        say("  nothing pushed. The loop stops here by design — no silent")
        say("  re-runs, no invented experiments.")
        led.sync_out("daily %s: queue exhausted" % today)
        return
    say("  planned %s: %s" % (plan["variant"],
                              json.dumps(plan["changed"])))
    say("  reason: %s" % plan["reason"])

    # ---- 4) local filter (piece two-b): a smoke test, never a score ----
    if plan["kind"] != "anchor" and not skip_gym:
        say("")
        say("--- 4) local filter (toy games; filters only the obviously-broken) ---")
        gate = run_gym_gate(lever_dir, view.incumbent_knobs, plan)
        plan["gym"] = gate
        knob, val = next(iter(plan["changed"].items()))
        led.append({"kind": "gym", "exp": plan.get("exp", plan["variant"]),
                    "variant": plan["variant"], "knob": knob, "value": val[1],
                    "verdict": gate.get("verdict"),
                    "verdict_reasons": gate.get("verdict_reasons"),
                    "note": "local_filter: toy-gym smoke counters, NOT a "
                            "leaderboard score",
                    "summary": {t: {k: (gate.get(t) or {}).get(k)
                                    for k in ("levels", "wins", "wall_seconds")}
                                for t in ("incumbent", "candidate")}})
        if gate.get("verdict") == "UNAVAILABLE":
            say("  " + "!" * 62)
            say("  FILTER UNAVAILABLE — this says nothing about the variant.")
            say("  reason: " + "; ".join((gate.get("verdict_reasons") or [""])[:1]))
            say("  Proceeding WITHOUT the filter, exactly as --skip-gym would.")
            say("  Install the gym's imports to restore the filter.")
            say("  " + "!" * 62)
        if gate.get("verdict") == "FAIL":
            say("  FILTERED: %s rejected before spending a day: %s"
                % (plan["variant"], "; ".join(gate.get("verdict_reasons") or [])))
            say("  recorded as tried-without-measurement; tomorrow plans the")
            say("  next queue entry. Nothing pushed today.")
            led.sync_out("daily %s: %s filtered" % (today, plan["variant"]))
            return

    # ---- 5) pending row FIRST (durable), then push, then stop ----
    say("")
    say("--- 5) pending row + push + STOP ---")
    ev = block_experiment_event(plan, today)
    led.append(ev)
    led.sync_out("pending %s for %s (pre-push)" % (ev["exp"], today))
    version = do_push(work, patch_path, applier_path, inject_at,
                      allow_same=False, plan=plan, allow_inert=allow_inert,
                      variant=variant)
    led.append({"kind": "pushed", "exp": ev["exp"], "version": version})
    led.sync_out("pushed v%d for %s" % (version, ev["exp"]))
    say("PUSHED version %d of %s/%s" % (version, OWNER, KERNEL_SLUG))
    print_human_card(plan, version, today)


def do_plan(lever_dir: Path, work: Path, today: str) -> None:
    led = Ledger(work / "ledger")
    led.sync_in()
    reconcile(led, submissions(), today)
    view = led.fold()
    say("")
    say(view.table())
    say("")
    plan = plan_next(view, today)
    if plan is None:
        say("queue exhausted; incumbent score %.4g" % (view.incumbent_score or -1))
        return
    say("next experiment: %s  %s" % (plan["variant"], json.dumps(plan["changed"])))
    say("reason: %s" % plan["reason"])
    say("nothing pushed (stage=plan is read-only).")


def do_score_read(lever_dir: Path, work: Path, today: str) -> None:
    led = Ledger(work / "ledger")
    led.sync_in()
    n = reconcile(led, submissions(), today)
    say("reconciled %d new score row(s)" % n)
    view = led.fold()
    say("")
    say(view.table())
    say("")
    say("incumbent score: %s" % (("%.4g" % view.incumbent_score)
                                 if view.incumbent_score is not None else "-"))
    led.sync_out("score read %s" % today)


def parse_set(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        if "=" not in p:
            die("--set expects KNOB=JSON_VALUE, got %r" % p)
        k, v = p.split("=", 1)
        k = k.strip()
        if k not in DEFAULT_KNOBS:
            die("--set %s: not a sweepable knob (sweepable: %s)"
                % (k, ", ".join(sorted(DEFAULT_KNOBS))))
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError as exc:
            die("--set %s: value %r is not JSON (%s)" % (k, v, exc))
    return out


# --------------------------------------------------------------------------
# self test — the whole loop, offline, against _stub_kaggle.py
# --------------------------------------------------------------------------

FIXTURE_HEADINGS = [
    "# About this notebook",
    "## 0. Harness patch",
    "## 1. Environment and submission mode",
    "## 2. Precaching",
    "## 3. Install the ARC runtime",
    "## 4. Import the bundled source",
    "## 5. Start serving",
    "## 6. Load the benchmark",
    "## 7. Customization hook",
    "## 8. Run the system monitor",
]

PATCH = ("diff --git a/ARC3-Inference/inference/agent/prompts.py "
         "b/ARC3-Inference/inference/agent/prompts.py\n"
         "index 52351db..3053e65 100644\n"
         "--- a/ARC3-Inference/inference/agent/prompts.py\n"
         "+++ b/ARC3-Inference/inference/agent/prompts.py\n"
         "@@ -139,7 +139,7 @@\n"
         '-    "- The raw numeric grid is intentionally not exposed.\\n"\n'
         '+    "- `current_frame.grid` is the exact numeric color grid.\\n"\n')

APPLIER = ("import subprocess\n"
           "VARIANT = \"lever3\"  # k5-variant-line\n"
           "subprocess.run(['git', 'apply', '-v', str(PATCH)], "
           "cwd=str(BUNDLE_DIR / 'src'), check=True)\n"
           "print('lever 3 applied')\n")


def _fixture() -> dict:
    cells = [{"cell_type": "markdown", "metadata": {},
              "source": FIXTURE_HEADINGS[0] + "\n\nprose"}]
    for h in FIXTURE_HEADINGS[1:]:
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": h + "\n\nprose"})
        body = "import os\n"
        if h.startswith("## 1."):
            body = ("import subprocess\n"
                    "WORKING_DIR = Path('/kaggle/working')\n"
                    "BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')\n"
                    "subprocess.run(['git', 'apply', '/kaggle/harness-changes.patch'], check=True)\n")
        elif h.startswith("## 7."):
            body = ("# Make one-off changes to `bm`, `bm.games`, or `bm.solver` here before the run starts.\n"
                    "class _Solver:\n"
                    "    analyzer_timeout = 900.0\n"
                    "    max_runtime_s_per_game = 31920.0\n"
                    "    max_actions_per_game = None\n"
                    "    concurrency = 120\n"
                    "class _BM:\n"
                    "    solver = _Solver()\n"
                    "bm = _BM()\n")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": body})
    return {"cells": cells, "metadata": {"kaggle": {"accelerator": "nvidiaRtxPro6000"}},
            "nbformat": 4, "nbformat_minor": 4}


def _t(ok: bool, label: str) -> None:
    if not ok:
        raise AssertionError("self-test failed: " + label)
    say("  " + label + " OK")


def _test_injection() -> None:
    say("self-test: injection surgery")
    nb = _fixture()
    expect = FIXTURE_HEADINGS.index("## 2. Precaching") * 2 - 1
    at = find_injection_point(nb, None)
    _t(at == expect, "anchor: insert at %d" % at)

    # lever-only (legacy) mode: exactly 2 cells
    nb2 = inject(nb, PATCH, APPLIER, LEVER_TAG)
    verify_injection(nb2, PATCH, LEVER_TAG)
    _t(len(lever_cells_present(nb2, LEVER_TAG)) == 2
       and len(nb2["cells"]) == len(nb["cells"]) + 2,
       "lever-only injection adds exactly 2 cells")

    # with the knob payload: format 2, hook placement
    payload = {"format": 2, "variant": "exp-001", "device": DEVICE,
               "date": "2026-10-16",
               "changed": {"analyzer_timeout": [900.0, 1200.0]},
               "reason": "t",
               "knobs": dict(DEFAULT_KNOBS, analyzer_timeout=1200.0)}
    nb3 = inject(nb, PATCH, APPLIER, LEVER_TAG, knob_payload=payload)
    verify_injection(nb3, PATCH, LEVER_TAG, knob_payload=payload)

    # variant baking: lever3 is a no-op, kernel rewrites exactly one line
    anchor = lever_anchor(nb3, LEVER_TAG)
    before_src = cell_text(nb3["cells"][anchor])
    _t('VARIANT = "lever3"' in before_src, "applier cell carries the variant line")
    nb_kept = set_variant(json.loads(json.dumps(nb3)), "lever3")
    _t(cell_text(nb_kept["cells"][lever_anchor(nb_kept, LEVER_TAG)]) == before_src,
       "variant=lever3 leaves the applier cell byte-identical")
    nb_kernel = set_variant(json.loads(json.dumps(nb3)), "kernel")
    kernel_src = cell_text(nb_kernel["cells"][lever_anchor(nb_kernel, LEVER_TAG)])
    _t('VARIANT = "kernel"' in kernel_src and 'VARIANT = "lever3"' not in kernel_src,
       "variant=kernel bakes the kernel variant into the applier cell")
    _t(kernel_src.count("k5-variant-line") == 1,
       "variant rewrite does not duplicate the marker")

    # hook placement test
    h_idx = find_hook_heading(nb3)
    k_idx = find_knob_cell_index(nb3)
    _t(h_idx is not None and k_idx is not None and k_idx > h_idx,
       "hook placement: knob cell placed after '## 7. Customization hook'")

    # payload AST round-trip
    import ast
    knob_src = cell_text(nb3["cells"][k_idx])
    parsed_ast = ast.parse(knob_src)
    m = re.search(r"_K5 = (\{.*?\})" + chr(10) + chr(10) + "_K5_SURFACE", knob_src, re.S)
    _t(bool(parsed_ast) and m is not None and json.loads(m.group(1))["format"] == 2,
       "payload AST round-trip")

    # idempotent replace
    _t(json.dumps(inject(nb3, PATCH, APPLIER, LEVER_TAG, knob_payload=payload),
                  sort_keys=True) == json.dumps(nb3, sort_keys=True),
       "re-inject with same payload is a no-op")
    nb4 = inject(nb3, PATCH, APPLIER, LEVER_TAG,
                 knob_payload=dict(payload, variant="exp-002"))
    _t(find_knob_cell_index(nb4) is not None
       and len(nb4["cells"]) == len(nb3["cells"]),
       "re-inject replaces in place, never duplicates")

    # sanitizer on a live-shaped fixture
    nb_live = _fixture()
    nb_live["cells"].insert(4, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "   " + chr(10) + chr(10)})
    proto_src = ("# Lever 2 — add .grid to the sandbox; prompt tweaks are best" + chr(10) +
                 "import re" + chr(10) +
                 "from inference.agent import prompts, python_tool_sandbox, tool_agent" + chr(10) +
                 "sb = python_tool_sandbox._SANDBOX_BOOTSTRAP" + chr(10))
    nb_live["cells"].insert(12, {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": proto_src})
    nb_san = sanitize_notebook(nb_live)
    _t(not any(is_empty_code_cell(c) for c in nb_san["cells"])
       and not any(is_hand_lever_cell(c) for c in nb_san["cells"]),
       "sanitizer on a live-shaped fixture")

    # refusal firing when the hook is renamed
    nb_renamed = _fixture()
    for c in nb_renamed["cells"]:
        if _is_md(c) and "customization hook" in cell_text(c).lower():
            c["source"] = "## 7. Something else"
    _t(not has_knob_channel(nb_renamed), "refusal firing when the hook is renamed")

    nb5 = _fixture()
    for c in nb5["cells"]:
        c["source"] = cell_text(c).replace("BUNDLE_DIR = ", "_unused = ")
    try:
        inject(nb5, PATCH, APPLIER, LEVER_TAG)
    except SystemExit:
        say("  upstream-name guard rejects a section 1 that lost BUNDLE_DIR OK")
    else:
        raise AssertionError("upstream-name guard did not fire")

    _t(parse_push_version("Kernel version 17 successfully pushed.  x") == 17,
       "push version parse")


def _test_ledger_and_planner(tmp: Path) -> None:
    say("self-test: ledger, reconcile, planner")
    led = Ledger(tmp / "led")
    today = "2026-10-16"

    view0 = led.fold()
    plan = plan_next(view0, today)
    _t(plan["kind"] == "anchor" and plan["variant"] == ANCHOR_VARIANT
       and plan["changed"] == {}, "empty ledger plans the anchor first")

    ev0 = block_experiment_event(plan, today)
    _t(set(ev0["knobs"]) == set(DEFAULT_KNOBS) and ev0["device"] == DEVICE
       and ev0["reason"], "pending row carries full knob state + device")

    led.append(ev0)
    led.append({"kind": "pushed", "exp": "exp-000", "version": 5})
    subs = [{"ref": 1001, "date": "2026-10-15 09:00:00",
             "description": "kaggle5 %s v5" % ANCHOR_VARIANT, "status": "complete",
             "publicScore": 0.07}]
    n = reconcile(led, subs, today)
    view1 = led.fold()
    _t(n == 1 and view1.incumbent_score == 0.07 and view1.anchor_done,
       "anchor reconciliation: message-matched, 0.07 becomes incumbent")
    _t(view1.experiments["exp-000"]["status"] == "scored",
       "anchor status scored")

    plan1 = plan_next(view1, today)
    _t(plan1 is not None and plan1["changed"] == {"analyzer_timeout": [900.0, 1200.0]}
       and plan1["exp"] == "exp-001",
       "first queue step: analyzer_timeout 900.0->1200.0")
    _t(plan1["reason"].startswith("queue: analyzer_timeout"),
       "plan reason names the knob and cites the queue")

    led.append(block_experiment_event(plan1, "2026-10-16"))
    led.append({"kind": "pushed", "exp": "exp-001", "version": 6})
    subs1 = [{"ref": 1002, "date": "2026-10-16 09:00:00",
              "description": "kaggle5 exp-001 v6", "status": "complete",
              "publicScore": 0.05}]
    reconcile(led, subs1, "2026-10-16")
    view2 = led.fold()
    _t(view2.incumbent_score == 0.07,
       "worse result (0.05 < 0.07): incumbent unchanged")
    _t((canon("analyzer_timeout"), canon(1200.0)) in view2.tried,
       "tried set records the measured value")

    plan2 = plan_next(view2, "2026-10-17")
    _t(plan2 is not None and plan2["changed"] == {"max_actions_per_game": [None, 500]},
       "backtrack rule: worse kills the rest of analyzer_timeout's ladder; next knob is max_actions_per_game")

    led.append(block_experiment_event(plan2, "2026-10-17"))
    led.append({"kind": "pushed", "exp": "exp-002", "version": 7})
    subs2 = [{"ref": 1003, "date": "2026-10-17 09:00:00",
              "description": "unlabelled submission", "status": "complete",
              "publicScore": 0.07}]
    reconcile(led, subs2, "2026-10-17")
    view3 = led.fold()
    _t(view3.experiments["exp-002"]["outcome"] == "equal"
       and view3.incumbent_knobs["max_actions_per_game"] == 500,
       "FIFO fallback matched the unlabelled submission; equal adopts")
    _t("1003" in view3.matched_refs,
       "matched refs accumulate (no double counting)")
    _t(reconcile(led, subs2, "2026-10-17") == 0,
       "re-reconcile is a no-op (idempotent)")

    plan3 = plan_next(view3, "2026-10-18")
    led.append(block_experiment_event(plan3, "2026-10-18"))
    led.append({"kind": "pushed", "exp": "exp-003", "version": 8})
    subs3 = [{"ref": 1004, "date": "2026-10-18 09:00:00",
              "description": "kaggle5 exp-003 v8", "status": "complete",
              "publicScore": 0.0}]
    reconcile(led, subs3, "2026-10-18")
    view4 = led.fold()
    _t(view4.experiments["exp-003"]["status"] == "zero",
       "a zero is recorded as data, never dropped")

    led.append({"kind": "experiment", "exp": "exp-099", "variant": "exp-099",
                "date": "2026-10-01", "changed": {}, "knobs": dict(DEFAULT_KNOBS),
                "reason": "old"})
    reconcile(led, [], "2026-10-10")
    view5 = led.fold()
    _t(view5.experiments["exp-099"]["status"] == "stale",
       "pending > 2 days -> stale tombstone")

    dummy_rows = [{"game_id": "ls20-dummy", "levels_completed": "1"}]
    _t(rows_look_dummy(dummy_rows),
       "dummy: single row or file-wide sentinels labelled; mixed files kept")

    for knob, ladder, _why in EXPERIMENT_QUEUE:
        for val in ladder:
            view5.tried.add((canon(knob), canon(val)))
    _t(plan_next(view5, "2026-10-19") is None,
       "queue exhausted -> plan None (the loop stops by design)")

    before = (tmp / "led" / "experiments.jsonl").read_text()
    _ = led.fold()
    _t((tmp / "led" / "experiments.jsonl").read_text() == before,
       "reading/folding never rewrites the ledger (append-only)")


def _test_static_gate() -> None:
    say("self-test: static gate edge cases")
    incumbent = dict(DEFAULT_KNOBS)

    # 1. multi-knob changed
    p_multi = {"variant": "exp-bad1", "changed": {"analyzer_timeout": [900.0, 1200.0], "concurrency": [120, 60]}, "knobs": {}}
    ok, msg = validate_static_gate(incumbent, p_multi)
    _t(not ok and "expected exactly 1 knob changed" in msg, "multi-knob plan refused")

    # 2. unknown knob
    p_unk = {"variant": "exp-bad2", "changed": {"unknown_knob": [1, 2]}, "knobs": {}}
    ok, msg = validate_static_gate(incumbent, p_unk)
    _t(not ok and "not in KNOB_SURFACE" in msg, "unknown knob refused")

    # 3. boolean value
    p_bool = {"variant": "exp-bad3", "changed": {"concurrency": [120, True]}, "knobs": {}}
    ok, msg = validate_static_gate(incumbent, p_bool)
    _t(not ok and "boolean" in msg, "boolean value refused")

    # 4. uncap value -> None
    p_uncap = {"variant": "exp-bad4", "changed": {"max_actions_per_game": [500, None]}, "knobs": {}}
    ok, msg = validate_static_gate(incumbent, p_uncap)
    _t(not ok and "uncap" in msg, "uncap (value -> None) refused")

    # 5. uncap None -> value allowed
    p_cap = {"variant": "exp-ok1", "changed": {"max_actions_per_game": [None, 500]}, "knobs": dict(incumbent, max_actions_per_game=500)}
    ok, msg = validate_static_gate(incumbent, p_cap)
    _t(ok, "uncap (None -> value) allowed")

    # 6. |log2| > 1
    p_ratio = {"variant": "exp-bad5", "changed": {"analyzer_timeout": [900.0, 2000.0]}, "knobs": {}}
    ok, msg = validate_static_gate(incumbent, p_ratio)
    _t(not ok and "|log2" in msg, "ratio > 2.0 (|log2| > 1) refused")

    # 7. canon-equal
    p_eq = {"variant": "exp-bad6", "changed": {"analyzer_timeout": [900.0, 900.0]}, "knobs": {}}
    ok, msg = validate_static_gate(incumbent, p_eq)
    _t(not ok and "canonically equal" in msg, "canon-equal change refused")


def _test_gym_verdicts(lever_dir: Path) -> None:
    """A filter that cannot run must never be mistaken for a rejection."""
    say("self-test: gym verdict classification")
    tmp = Path(tempfile.mkdtemp(prefix="k5_gymv_"))
    (tmp / AGENT_FILE).write_text("x = 1\n")   # run_gym_gate needs this name
    plan = {"variant": "t", "knobs": {}, "changed": {"analyzer_timeout": [1.0, 2.0]}}

    gym = tmp / GYM_FILE
    gym.write_text(
        "import sys" + chr(10) +
        "import numpy_that_does_not_exist" + chr(10) +
        "print('{}')" + chr(10))
    v = run_gym_gate(tmp, {}, plan)
    _t(v.get("verdict") == "UNAVAILABLE",
       "a gym that cannot import its deps is UNAVAILABLE, not FAIL")
    _t(any("numpy_that_does_not_exist" in str(r)
           for r in v.get("verdict_reasons") or []),
       "UNAVAILABLE names the missing module")

    gym.write_text(
        "import sys" + chr(10) +
        "sys.stderr.write('Traceback...\nRuntimeError: real crash\n')" +
        chr(10) + "sys.exit(3)" + chr(10))
    v = run_gym_gate(tmp, {}, plan)
    _t(v.get("verdict") == "FAIL",
       "a gym that really crashes is still FAIL (the filter still bites)")

    gym.write_text(
        "import json" + chr(10) +
        "print(json.dumps({'verdict': 'PASS', 'verdict_reasons': [],"
        " 'incumbent': {'levels': 5, 'wins': 3, 'wall_seconds': 1.0},"
        " 'candidate': {'levels': 5, 'wins': 3, 'wall_seconds': 1.0}}))" + chr(10))
    v = run_gym_gate(tmp, {}, plan)
    _t(v.get("verdict") == "PASS", "a healthy gym still passes")
    shutil.rmtree(tmp, ignore_errors=True)


def _test_guards(lever_dir: Path) -> None:
    say("self-test: guards and refusals")
    _src = Path(__file__).read_text()
    _bad1 = "competitions" + chr(34) + ", " + chr(34) + "submit"
    _bad2 = "competitions" + chr(39) + ", " + chr(39) + "submit"
    _t(_bad1 not in _src and _bad2 not in _src,
       "driver source contains no call-shaped `competitions submit` text")
    try:
        refuse_submit("submit")
    except SystemExit as exc:
        _t(exc.code == 1, "--stage submit refuses loudly (exit 1)")
    else:
        raise AssertionError("refuse_submit did not raise")
    try:
        refuse_submit("all")
    except SystemExit:
        say("  --stage all refuses too OK")
    else:
        raise AssertionError("refuse_submit did not raise")

    good = {"competition_sources": EXPECTED_COMPETITION_SOURCES,
            "dataset_sources": EXPECTED_DATASET_SOURCES,
            "model_sources": EXPECTED_MODEL_SOURCES,
            "enable_internet": False, "enable_gpu": False,
            "machine_shape": "nvidiaRtxPro6000"}
    guard_metadata(good)
    say("  metadata guard accepts the real config OK")

    if (lever_dir / AGENT_FILE).is_file():
        guard_agent_file(lever_dir)
        check_defaults_drift(lever_dir)
        say("  agent pin + defaults-drift guards pass on the shipped agent OK")


def _test_stub_end_to_end(lever_dir: Path, tmp: Path) -> None:
    say("self-test: multi-day offline end-to-end against _stub_kaggle.py")
    stub = lever_dir / "_stub_kaggle.py"
    for f in (stub, lever_dir / LEVER_PATCH_FILE, lever_dir / "lever3_apply.py"):
        if not f.is_file():
            say("  stub end-to-end: SKIPPED (%s missing beside kaggle5.py)"
                % f.name)
            return
    env_base = dict(os.environ)
    env_base["KAGGLE_BIN"] = "%s %s" % (sys.executable, stub)
    env_base["STUB_STATE"] = str(tmp / "state.json")
    env_base["STUB_LOG"] = str(tmp / "stub.log")
    env_base["STUB_DATA"] = str(tmp / "stubdata")

    def run(day, stage="daily", extra=()):
        env = dict(env_base, K5_TODAY=day)
        r = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--stage", stage,
             "--work", str(tmp / "work"), *extra],
            env=env, capture_output=True, text=True, timeout=600)
        return r

    def human_submits(ref, day, desc, score, version, per_game=True):
        state = json.loads((tmp / "state.json").read_text())
        state["subs"].insert(0, {
            "ref": ref, "fileName": SUBMISSION_FILE,
            "date": day + " 09:00:00", "description": desc,
            "status": "complete", "publicScore": score, "privateScore": ""})
        if per_game:
            state.setdefault("k5_outputs", {})[str(version)] = {
                "k5_per_game.csv": (
                    "# k5 telemetry v1" + chr(10) +
                    "run_variant,run_agent_sha256,run_knobs_sha256,game_id," +
                    "levels_completed,win_levels,win,actions_used," +
                    "decision_events,final_state" + chr(10) +
                    "%(v)s,abc123,def456,ls20-%(v)s,1,3,0,80,212,NOT_FINISHED" + chr(10) +
                    "%(v)s,abc123,def456,ft09-%(v)s,2,2,1,37,140,WIN" + chr(10)
                    % {"v": desc.split()[1]}),
                "k5_run_info.json": json.dumps(
                    {"variant": desc.split()[1], "device": DEVICE,
                     "applied_overrides": {}}),
            }
        (tmp / "state.json").write_text(json.dumps(state))

    def stub_versions():
        return json.loads((tmp / "state.json").read_text())["version"]

    def ledger_lines():
        return (tmp / "work" / "ledger" / "experiments.jsonl").read_text()

    # ---- day 1: anchor ----
    r = run("2026-10-15")
    _t(r.returncode == 0, "day1 daily exits 0")
    _t('"exp": "exp-000"' in ledger_lines() and '"kind": "pushed"'
       in ledger_lines(), "day1: anchor pending row + pushed row")
    _t("THE MACHINE STOPS HERE" in r.stdout, "day1: human card printed")
    v1 = stub_versions()
    human_submits(1001, "2026-10-15", "kaggle5 %s v%d" % (ANCHOR_VARIANT, v1), 0.07, v1)

    # ---- day 2: reconcile anchor, plan + push exp-001 ----
    r = run("2026-10-16")
    _t(r.returncode == 0, "day2 daily exits 0")
    _t("anchor" in r.stdout and "exp-001" in r.stdout
       and "analyzer_timeout" in r.stdout,
       "day2: anchor reconciled; exp-001 (analyzer_timeout 900.0->1200.0) planned")
    v2 = stub_versions()
    _t(v2 == v1 + 1, "day2: exactly one new push")
    human_submits(1002, "2026-10-16", "kaggle5 exp-001 v%d" % v2, 0.05, v2)

    # ---- day 3: worse -> backtrack rule picks max_actions_per_game ----
    r = run("2026-10-17")
    _t(r.returncode == 0, "day3 daily exits 0")
    _t("worse" in r.stdout and "max_actions_per_game" in r.stdout,
       "day3: 0.05 < 0.07 recorded as data; backtrack plans max_actions_per_game")
    v3 = stub_versions()

    # ---- idempotency: same day again -> hold, no push ----
    r = run("2026-10-17")
    _t(r.returncode == 0 and "pushes nothing today" in r.stdout,
       "day3 second run: holds on the unanswered pending row")
    _t(stub_versions() == v3, "day3 second run: no extra push")

    # ---- per-game rows landed for the reconciled experiments ----
    game_rows = (tmp / "work" / "ledger" / "game_rows.jsonl").read_text()
    _t("ls20-" in game_rows and "ft09-" in game_rows,
       "per-game telemetry rows stored (not just an aggregate)")
    _t('"dummy": false' in game_rows, "real rows labelled dummy=false")

    # ---- the machine never submitted, across all three days ----
    log = (tmp / "stub.log").read_text() if (tmp / "stub.log").is_file() else ""
    _t("competitions submit" not in log, "never-submits: zero submit calls "
                                         "across the whole simulated week")
    _t("kernels push" in log and "datasets version" in log,
       "pushes and ledger syncs did happen")

    # ---- abandoned pending -> re-plan next day ----
    env = dict(env_base, K5_TODAY="2026-10-17")
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--abandon", "--work", str(tmp / "work")],
                       env=env, capture_output=True, text=True, timeout=60)
    _t(r.returncode == 0, "--abandon tombstones the pending row")


def self_test(lever_dir: Path) -> None:
    _test_injection()
    _test_ledger_and_planner(Path(tempfile.mkdtemp(prefix="k5_st_")))
    _test_static_gate()
    _test_gym_verdicts(lever_dir)
    _test_guards(lever_dir)
    _test_stub_end_to_end(lever_dir, Path(tempfile.mkdtemp(prefix="k5_e2e_")))
    say("")
    say("SELF TEST PASSED")



def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="check",
                    choices=["check", "plan", "gym", "push", "score", "daily",
                             "submit", "all"])
    ap.add_argument("-v", "--version", type=int, default=0)
    ap.add_argument("--lever-dir", default=None,
                    help="folder holding the patch, applier, agent and gym")
    ap.add_argument("--work", default=None)
    ap.add_argument("--inject-at", type=int, default=None)
    ap.add_argument("--allow-same", action="store_true")
    ap.add_argument("--allow-inert", action="store_true",
                    help="push even when the knob channel is missing (inert variant)")
    ap.add_argument("--skip-gym", action="store_true")
    ap.add_argument("--set", action="append", default=[],
                    metavar="KNOB=JSON", help="ad-hoc override for stage=gym/push")
    ap.add_argument("--variant", default="lever3", choices=["lever3", "kernel"],
                    help="lever3 = memory + hard prune gate (default); "
                         "kernel = memory present, gate off, prompt untouched")
    ap.add_argument("--abandon", action="store_true",
                    help="tombstone the current pending row (human decided "
                         "not to submit it) and exit")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    default_work = ("/kaggle/working/kaggle5" if Path("/kaggle/working").is_dir()
                    else str(Path.cwd() / "kaggle5_work"))
    work = Path(a.work or default_work)
    lever_dir = Path(a.lever_dir) if a.lever_dir else (
        Path(__file__).resolve().parent
        if Path(__file__).resolve().parent.is_dir() else Path.cwd())
    patch_path = lever_dir / LEVER_PATCH_FILE
    applier_path = lever_dir / "lever3_apply.py"

    if a.self_test:
        self_test(lever_dir)
        return

    say("kaggle5  stage=%s  kernel=%s/%s  comp=%s"
        % (a.stage, OWNER, KERNEL_SLUG, COMPETITION))
    check_auth()

    if a.abandon:
        led = Ledger(work / "ledger")
        led.sync_in()
        view = led.fold()
        pend = view.open_pending()
        if pend is None:
            say("no pending row to abandon.")
            return
        led.append({"kind": "abandon", "exp": pend["exp"],
                    "reason": "human flagged --abandon (decided not to submit "
                              "this variant)"})
        led.sync_out("abandon %s" % pend["exp"])
        say("abandoned %s. Tomorrow's run will plan from the ledger." % pend["exp"])
        return

    if a.stage == "check":
        do_check(work, patch_path, applier_path, a.inject_at, lever_dir,
                 variant=a.variant)
    elif a.stage == "plan":
        do_plan(lever_dir, work, local_today())
    elif a.stage == "gym":
        guard_agent_file(lever_dir)
        over = parse_set(a.set)
        knobs = dict(DEFAULT_KNOBS, **over)
        plan = {"variant": "adhoc-gym", "changed":
                {k: [DEFAULT_KNOBS[k], v] for k, v in over.items()},
                "knobs": knobs, "reason": "ad-hoc --set", "exp": "adhoc"}
        gate = run_gym_gate(lever_dir, dict(DEFAULT_KNOBS), plan)
        if gate.get("verdict") in (None, "FAIL"):
            raise SystemExit(1)
    elif a.stage == "push":
        if a.set:
            over = parse_set(a.set)
            knobs = dict(DEFAULT_KNOBS, **over)
            plan = {"kind": "experiment", "variant": "manual-" + local_today(),
                    "changed": {k: [DEFAULT_KNOBS[k], v] for k, v in over.items()},
                    "knobs": knobs, "reason": "manual ad-hoc push via --set",
                    "exp": "manual-" + local_today()}
            led = Ledger(work / "ledger")
            led.sync_in()
            ev = block_experiment_event(plan, local_today())
            led.append(ev)
            led.sync_out("manual pending %s" % ev["exp"])
            ver = do_push(work, patch_path, applier_path, a.inject_at,
                          a.allow_same, plan=plan, allow_inert=a.allow_inert,
                          variant=a.variant)
            led.append({"kind": "pushed", "exp": ev["exp"], "version": ver})
            led.sync_out("manual pushed v%d" % ver)
            print_human_card(plan, ver, local_today())
        else:
            say("note: lever-only repush, OUTSIDE the experiment loop "
                "(no pending row written).")
            ver = do_push(work, patch_path, applier_path, a.inject_at,
                          a.allow_same, variant=a.variant)
            say("PUSHED version %d. The loop is untouched; use --set to push "
                "an ad-hoc experiment instead." % ver)
    elif a.stage == "score":
        do_score_read(lever_dir, work, local_today())
    elif a.stage == "daily":
        do_daily(work, patch_path, applier_path, a.inject_at, lever_dir,
                 skip_gym=a.skip_gym, allow_inert=a.allow_inert,
                 variant=a.variant)
    elif a.stage in ("submit", "all"):
        refuse_submit(a.stage)


if __name__ == "__main__":
    main()
