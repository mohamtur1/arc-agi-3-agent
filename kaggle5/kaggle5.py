"""kaggle5 — inject the lever patch, push, submit, score. Optionally every day.

Runs INSIDE a Kaggle notebook. The notebook is CPU-only and can be scheduled.

Stages
  check    pull the target kernel, verify its inputs, print the heading map,
           dry-run the injection. Changes nothing.
  push     pull, inject, verify, push a new version. Kaggle then runs it (Phase A).
  wait     poll until that run finishes.
  submit   submit a finished version to the competition. BURNS A DAILY SLOT.
  score    read the public score back, compare to BASELINE_SCORE.
  daily    one cell for a scheduled notebook. Works out by itself whether today
           needs a push, a submit, a score read, or nothing. Idempotent.
  all      push -> wait -> submit -> score, in one go.

`daily` keeps no state of its own. It derives where the cycle is from two
queries: which kernel versions have a submission.parquet output, and which
versions already appear in the competition's submission list. That means a
scheduled run can be interrupted at any point and resume correctly the next day.

Run `python kaggle5.py --self-test` to exercise the logic offline: no auth, no
network, no Kaggle.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG — edit only this block
# --------------------------------------------------------------------------

OWNER = "mohamedmalikturay"
KERNEL_SLUG = "arc-agi-3-milestone-2-solution-3e9adb"
COMPETITION = "arc-prize-2026-arc-agi-3"
SUBMISSION_FILE = "submission.parquet"
BASELINE_SCORE = 25.83

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
VERSION_SCAN_BACK = 12                     # how many recent versions to probe

# Scheduled runs fire in UTC. Freetown is UTC+0 all year, so 0 is correct there.
LOCAL_UTC_OFFSET_HOURS = 0

# --------------------------------------------------------------------------


def say(msg: str = "") -> None:
    print(msg, flush=True)


def die(msg: str) -> None:
    say("")
    say("STOP — " + msg)
    raise SystemExit(1)


def norm(s: str) -> str:
    return re.sub(r"[\s\-_:]+", " ", (s or "").strip().lower())


# --------------------------------------------------------------------------
# CLI plumbing
# --------------------------------------------------------------------------

def check_auth() -> None:
    if os.environ.get("KAGGLE_API_TOKEN"):
        say("auth: KAGGLE_API_TOKEN is set")
        return
    for p in ("~/.kaggle/access_token", "~/.kaggle/access_token.txt", "~/.kaggle/kaggle.json"):
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
    base = override.split() if override else [shutil.which("kaggle") or "kaggle"]
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
        die("kaggle %s exited %d" % (args[0], r.returncode))
    return r


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
    out = kaggle(["kernels", "pull", "-p", str(work), "-m", f"{OWNER}/{KERNEL_SLUG}"]).stdout
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
    for key in ("competition_sources", "dataset_sources", "kernel_sources", "model_sources"):
        say("  %-22s %s" % (key, meta.get(key) or []))
    say("  %-22s %s" % ("enable_gpu", meta.get("enable_gpu")))
    say("  %-22s %s" % ("enable_internet", meta.get("enable_internet")))
    say("  %-22s %r" % ("machine_shape", meta.get("machine_shape")))

    def missing(expected: list[str], actual: list[str]) -> list[str]:
        have = {norm(x).lstrip("c/") for x in (actual or [])}
        return [e for e in expected if norm(e) not in have]

    problems = []
    if m := missing(EXPECTED_COMPETITION_SOURCES, meta.get("competition_sources")):
        problems.append("competition_sources missing %s" % m)
    if m := missing(EXPECTED_DATASET_SOURCES, meta.get("dataset_sources")):
        problems.append("dataset_sources missing %s" % m)
    if m := missing(EXPECTED_MODEL_SOURCES, meta.get("model_sources")):
        problems.append("model_sources missing %s" % m)
    if meta.get("enable_internet") is not False:
        problems.append("enable_internet is %r, must be false" % meta.get("enable_internet"))

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
        say("  note: title %r slugifies to %r but the id slug is %r." % (title, slug, kslug))
        say("        The id in the metadata wins; check the URL the push prints.")


def needs_accelerator_flag(meta: dict) -> bool:
    return not (meta.get("machine_shape") or "").strip()


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
        if (mode == "eq" and head == target) or (mode == "prefix" and head.startswith(target)):
            return line.strip()
    return None


def find_injection_point(nb: dict, explicit_index: int | None) -> int:
    """Index to insert at: end of the anchor section, i.e. just before the next
    markdown heading. Inserting before the next heading rather than at
    `heading + 1` keeps this correct when a section holds several code cells."""
    cells = nb.get("cells", [])
    if explicit_index is not None:
        if not 0 <= explicit_index <= len(cells):
            die("--inject-at %d out of range (notebook has %d cells)" % (explicit_index, len(cells)))
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
                say("  fallback anchor: heading %r at cell %d -> insert at %d" % (line, i, i))
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
            say("  anchor: %s at cell %d, next heading at cell %d -> insert at %d" % (label, i, j, j))
            return j
    say("  anchor: %s at cell %d, no later heading -> insert at end (%d)" % (label, i, len(cells)))
    return len(cells)


def guard_upstream_names(nb: dict, at: int) -> None:
    """The applier cell reads WORKING_DIR and BUNDLE_DIR. If section 1 does not
    define them the lever would NameError nine hours into a run."""
    before = "".join(cell_text(c) for c in nb.get("cells", [])[:at])
    absent = [n for n in REQUIRED_UPSTREAM_NAMES if not re.search(r"\b%s\s*=" % n, before)]
    if absent:
        die(
            "the cells before the injection point do not define: %s\n"
            "  The lever applier reads those names. Check that section 1 is intact,\n"
            "  or pass --inject-at N to place the lever after the cell that defines them."
            % ", ".join(absent)
        )
    say("  upstream names present before the lever: %s" % ", ".join(REQUIRED_UPSTREAM_NAMES))


def applier_marker(lever_tag: str) -> str:
    """Goes in the applier cell. The patch cell cannot carry a comment marker:
    `%%writefile` is a cell magic and must be the first line, so a leading
    comment would turn it into plain text and write nothing."""
    return "# ===== %s apply (kaggle5) =====" % lever_tag


def _is_writefile(cell: dict, filename: str) -> bool:
    lines = cell_text(cell).splitlines()
    return bool(lines) and lines[0].startswith("%%writefile") and filename in lines[0]


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
    if i > 0 and _is_writefile(cells[i - 1], LEVER_PATCH_FILE):
        return [i - 1, i]
    return [i]


def build_lever_cells(patch_text: str, applier: str, lever_tag: str) -> list[dict]:
    """The two cells, mirroring the working with_lever2 notebook: write the patch
    to disk, then git-apply it and hash-verify."""
    # Built from parts so this file never contains the literal sequence, which
    # would break embedding it inside a notebook cell.
    if ("'" * 3) in patch_text:
        die(LEVER_PATCH_FILE + " contains a triple quote; cannot embed it in a writefile cell")
    body = patch_text if patch_text.endswith("\n") else patch_text + "\n"
    # Not %-formatted: `%%` would collapse to a single `%` and break the magic.
    write = "%%writefile /kaggle/working/" + LEVER_PATCH_FILE + "\n" + body
    app = applier_marker(lever_tag) + "\n" + applier.strip() + "\n"
    return [
        {"cell_type": "code", "metadata": {"trusted": True, "kaggle5": lever_tag + "-patch"},
         "execution_count": None, "outputs": [], "source": write},
        {"cell_type": "code", "metadata": {"trusted": True, "kaggle5": lever_tag + "-apply"},
         "execution_count": None, "outputs": [], "source": app},
    ]


def inject(nb: dict, patch_text: str, applier: str, lever_tag: str,
           explicit_index: int | None = None) -> dict:
    nb = json.loads(json.dumps(nb))
    new = build_lever_cells(patch_text, applier, lever_tag)

    present = lever_cells_present(nb, lever_tag)
    if present:
        say("  %s already present at cells %s — replacing in place, no duplicates"
            % (lever_tag, present))
        drop = set(present)
        keep = [c for i, c in enumerate(nb["cells"]) if i not in drop]
        at = min(present)
        keep[at:at] = new
        nb["cells"] = keep
        return nb

    at = find_injection_point(nb, explicit_index)
    guard_upstream_names(nb, at)
    nb["cells"][at:at] = new
    say("  inserted %s at cells %d-%d" % (lever_tag, at, at + 1))
    return nb


def verify_injection(nb: dict, patch_text: str, lever_tag: str) -> None:
    anchor = lever_anchor(nb, lever_tag)
    if anchor is None:
        die("post-check: no applier cell found")
    cells = nb.get("cells", [])
    if anchor == 0 or not _is_writefile(cells[anchor - 1], LEVER_PATCH_FILE):
        die("post-check: the cell before the applier is not a %%writefile " + LEVER_PATCH_FILE + " cell")
    if patch_text.strip() not in cell_text(cells[anchor - 1]):
        die("post-check: patch body not intact in the writefile cell")
    apply_src = cell_text(cells[anchor])
    if "git" not in apply_src or "apply" not in apply_src:
        die("post-check: the applier cell does not git-apply the patch")
    if len(lever_cells_present(nb, lever_tag)) != 2:
        die("post-check: expected exactly 2 lever cells")
    say("  post-check OK: writefile cell + applier, patch intact")


# --------------------------------------------------------------------------
# stateless cycle discovery
# --------------------------------------------------------------------------

def kernel_status() -> str:
    out = kaggle(["kernels", "status", f"{OWNER}/{KERNEL_SLUG}"], quiet=True).stdout
    m = re.search(r'has status "([^"]+)"', out)
    return (m.group(1) if m else out.strip()).lower()


def versions_with_output(highest_hint: int | None = None) -> list[int]:
    """Probe recent versions for one that produced SUBMISSION_FILE.

    There is no CLI call that returns a kernel's current version number, so we
    probe `kernels files <kernel>/<n>` downward from a hint. A version that has
    never run returns an error, which we treat as "no output".
    """
    start = highest_hint if highest_hint else None
    if start is None:
        start = _guess_highest_version()
    found = []
    for v in range(start, max(0, start - VERSION_SCAN_BACK), -1):
        r = kaggle(["kernels", "files", f"{OWNER}/{KERNEL_SLUG}/{v}", "--format", "json"],
                   allow_fail=True, quiet=True)
        if r.returncode != 0:
            continue
        data = _json_from(r.stdout or "")
        if isinstance(data, list) and any(SUBMISSION_FILE in str(f.get("name", "")) for f in data):
            found.append(v)
    return sorted(found)


def _guess_highest_version() -> int:
    """Push output is the only reliable version source; if we have no hint,
    probe upward from 1 until the kernel stops answering."""
    last = 1
    for v in range(1, 200):
        r = kaggle(["kernels", "files", f"{OWNER}/{KERNEL_SLUG}/{v}", "--format", "json"],
                   allow_fail=True, quiet=True)
        if r.returncode != 0 and "No files found" not in (r.stdout or ""):
            break
        last = v
    return last


def submissions() -> list[dict]:
    out = kaggle(["competitions", "submissions", COMPETITION, "--format", "json",
                  "--page-size", "20"], quiet=True).stdout
    data = _json_from(out)
    if not isinstance(data, list):
        die("could not parse `competitions submissions` as JSON:\n" + out[:800])
    return data


def local_today(offset_hours: int = LOCAL_UTC_OFFSET_HOURS) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=offset_hours)).strftime("%Y-%m-%d")


def date_of(sub: dict) -> str:
    return str(sub.get("date") or "")[:10]


def submitted_versions(subs: list[dict]) -> set[int]:
    """Versions already spent, read out of the description tag we write."""
    out = set()
    for s in subs:
        m = re.search(r"\bv(\d+)\b", str(s.get("description") or ""))
        if m:
            out.add(int(m.group(1)))
    return out


def submitted_today(subs: list[dict], today: str) -> list[dict]:
    return [s for s in subs if date_of(s) == today]


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

VERSION_RE = re.compile(r"Kernel version (\d+) successfully")


def parse_push_version(stdout: str) -> int:
    m = VERSION_RE.search(stdout or "")
    if not m:
        die("no version number in the push output. Look for "
            "'Kernel version N successfully pushed' and run stage=submit -v N.")
    return int(m.group(1))


def do_check(work: Path, patch_path: Path, applier_path: Path, inject_at: int | None) -> None:
    code, meta = pull(work)
    guard_metadata(meta)
    guard_slug(meta)

    nb = json.loads(code.read_text())
    cells = nb.get("cells", [])
    say("")
    say("--- notebook map: %d cells ---" % len(cells))
    for i, c in enumerate(cells):
        first = cell_text(c).strip().splitlines()[0] if cell_text(c).strip() else ""
        say("  [%3d] %-4s %7dch  %s" % (i, "md" if _is_md(c) else "code", len(cell_text(c)), first[:66]))

    say("")
    try:
        at = find_injection_point(nb, inject_at)
    except SystemExit:
        say("no anchor matched — pass --inject-at N using the map above, then re-run check")
        return
    guard_upstream_names(nb, at)

    say("lever cells already present: %s" % (lever_cells_present(nb, LEVER_TAG) or "no"))

    if not patch_path.is_file() or not patch_path.read_text().strip():
        die("%s missing or empty — the lever is the patch, it must exist" % patch_path.name)
    applier = applier_path.read_text() if applier_path.is_file() else ""
    if not applier.strip():
        die("%s missing or empty" % applier_path.name)

    nb2 = inject(nb, patch_path.read_text(), applier, LEVER_TAG, inject_at)
    verify_injection(nb2, patch_path.read_text(), LEVER_TAG)
    where = lever_cells_present(nb2, LEVER_TAG)
    say("dry-run injection: %d cells -> %d cells, lever at cells %s" % (len(cells), len(nb2["cells"]), where))
    say("")
    say("nothing was pushed. Next: stage=push")


def do_push(work: Path, patch_path: Path, applier_path: Path, inject_at: int | None,
            allow_same: bool) -> int:
    patch_text = patch_path.read_text() if patch_path.is_file() else ""
    applier = applier_path.read_text() if applier_path.is_file() else ""
    if not patch_text.strip():
        die("%s missing or empty" % patch_path.name)
    if not applier.strip():
        die("%s missing or empty" % applier_path.name)

    code, meta = pull(work)
    nb = json.loads(code.read_text())
    say("pulled %s: %d cells" % (code.name, len(nb.get("cells", []))))
    guard_metadata(meta)
    guard_slug(meta)

    before = json.dumps(nb, sort_keys=True)
    nb2 = inject(nb, patch_text, applier, LEVER_TAG, inject_at)
    verify_injection(nb2, patch_text, LEVER_TAG)

    if before == json.dumps(nb2, sort_keys=True):
        if not allow_same:
            die("the injected notebook is identical to what is already on Kaggle.\n"
                "  A push would create a version with no change. Edit the lever, or pass --allow-same.")
        say("  --allow-same given: pushing an unchanged notebook")

    code.write_text(json.dumps(nb2, indent=1))
    say("  wrote %s (%d cells)" % (code.name, len(nb2["cells"])))

    args = ["kernels", "push", "-p", str(work)]
    if needs_accelerator_flag(meta):
        say("  machine_shape empty on the pulled metadata; passing --accelerator %s" % FALLBACK_ACCELERATOR)
        args += ["--accelerator", FALLBACK_ACCELERATOR]
    out = kaggle(args, timeout=600).stdout
    version = parse_push_version(out)
    say("")
    say("PUSHED version %d of %s/%s" % (version, OWNER, KERNEL_SLUG))
    say("Phase A is running. Next: stage=wait")
    return version


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
            say("")
            say("Phase A COMPLETE. Safe to submit.")
            return True
        time.sleep(STATUS_POLL_SECONDS)
    say("still %r after %.1fh — resume later with stage=wait" % (last, max_seconds / 3600))
    return False


def do_submit(version: int, tag: str) -> None:
    if version <= 0:
        die("stage=submit needs a version. Pass -v N (the number stage=push printed).")
    st = kernel_status()
    if st != "complete":
        die("the kernel's latest run is %r, not complete. Submitting now would fail." % st)
    subs = submissions()
    if version in submitted_versions(subs):
        say("version %d is already in the submission list — not submitting twice." % version)
        return
    say("")
    say(">>> THIS BURNS ONE DAILY SUBMISSION <<<")
    kaggle(["competitions", "submit", COMPETITION,
            "-f", SUBMISSION_FILE,
            "-k", f"{OWNER}/{KERNEL_SLUG}",
            "-v", str(version),
            "-m", "kaggle5 %s v%d" % (tag, version)],
           capture=False, timeout=900)
    say("")
    say("submitted v%d. Next: stage=score" % version)


def newest_submission(subs: list[dict]) -> dict | None:
    return subs[0] if subs else None


def do_score(wait: bool, version: int, tag: str) -> float | None:
    needle = "kaggle5 %s v%d" % (tag, version) if version > 0 else ""
    deadline = time.time() + (SCORE_MAX_SECONDS if wait else 0)
    last_status = None
    while True:
        subs = submissions()
        pick = next((s for s in subs if needle in (s.get("description") or "")), None) if needle else None
        pick = pick or newest_submission(subs)
        if pick is None:
            if not wait:
                die("no submissions returned for %s" % COMPETITION)
        else:
            status = str(pick.get("status") or "").lower()
            score = pick.get("publicScore")
            if status != last_status:
                say("  [%s] ref=%s status=%s score=%s file=%s"
                    % (time.strftime("%H:%M:%S"), pick.get("ref"), status, score, pick.get("fileName")))
                last_status = status
            if status in ("complete", "error"):
                say("")
                if status == "error":
                    say("SUBMISSION ERRORED — no score to compare.")
                    return None
                return report_score(score)
        if not wait or time.time() >= deadline:
            say("not scored yet. Re-run stage=score later.")
            return None
        time.sleep(SCORE_POLL_SECONDS)


def report_score(score) -> float | None:
    try:
        val = float(score)
    except (TypeError, ValueError):
        say("submission complete but publicScore is %r — read it on kaggle.com" % score)
        return None
    delta = val - BASELINE_SCORE
    say("SCORE  %.4f   (baseline %.4f)   delta %+.4f  %s"
        % (val, BASELINE_SCORE, delta,
           "BETTER" if delta > 0 else ("same" if delta == 0 else "worse")))
    return val


def do_daily(work: Path, patch_path: Path, applier_path: Path, inject_at: int | None,
             wait_here: bool) -> None:
    """One cell for a scheduled CPU notebook. Stateless and idempotent."""
    today = local_today()
    say("=" * 62)
    say("daily run for %s (local, UTC%+d)" % (today, LOCAL_UTC_OFFSET_HOURS))
    say("=" * 62)

    subs = submissions()
    todays = submitted_today(subs, today)
    if todays:
        say("already submitted today (%d submission(s)). Latest:" % len(todays))
        for s in todays:
            say("  ref=%s status=%s score=%s  %s"
                % (s.get("ref"), s.get("status"), s.get("publicScore"), s.get("description")))
        newest = newest_submission(subs)
        if newest is not None:
            report_score(newest.get("publicScore"))
        say("")
        say("nothing to do. The slot for %s is spent." % today)
        return

    spent = submitted_versions(subs)
    say("submissions on record for versions: %s" % (sorted(spent) or "none"))

    st = kernel_status()
    say("kernel latest run status: %s" % st)

    # A push may be in flight. Its output is not ready yet, and the version
    # numbers below still describe the previous run, so settle that first.
    if st in ("running", "queued"):
        say("a run is in flight. Waiting on it before deciding anything.")
        if not do_wait():
            say("resuming tomorrow.")
            return
        st = kernel_status()

    with_output = versions_with_output()
    say("versions with a %s output: %s" % (SUBMISSION_FILE, with_output or "none"))

    # ONLY the newest version is a candidate. Submitting an older one would
    # spend the daily slot on code you have already moved past.
    newest = max(with_output) if with_output else 0
    if newest and newest not in spent:
        say("")
        say("newest finished version v%d has not been submitted. Submitting it." % newest)
        do_submit(newest, LEVER_TAG)
        do_score(wait=False, version=newest, tag=LEVER_TAG)
        return
    if newest:
        say("newest finished version v%d is already spent." % newest)
    else:
        say("no version has produced %s yet." % SUBMISSION_FILE)

    say("pushing a new version.")
    if not wait_here:
        say("this scheduled run does not block for the 9 h run. It pushed; tomorrow's")
        say("run will find the finished version and submit it.")
    version = do_push(work, patch_path, applier_path, inject_at, allow_same=False)
    say("")
    say("pushed v%d. Phase A is running." % version)
    if wait_here:
        if do_wait():
            do_submit(version, LEVER_TAG)
            do_score(wait=False, version=version, tag=LEVER_TAG)


# --------------------------------------------------------------------------
# self test
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
           "subprocess.run(['git', 'apply', '-v', str(PATCH)], "
           "cwd=str(BUNDLE_DIR / 'src'), check=True)\n"
           "print('lever 2 applied')\n")


def _fixture() -> dict:
    cells = [{"cell_type": "markdown", "metadata": {}, "source": FIXTURE_HEADINGS[0] + "\n\nprose"}]
    for h in FIXTURE_HEADINGS[1:]:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": h + "\n\nprose"})
        body = "import os\n"
        if h.startswith("## 1."):
            body = ("import subprocess\n"
                    "WORKING_DIR = Path('/kaggle/working')\n"
                    "BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')\n"
                    "subprocess.run(['git', 'apply', '/kaggle/harness-changes.patch'], check=True)\n")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": body})
    return {"cells": cells, "metadata": {"kaggle": {"accelerator": "nvidiaRtxPro6000"}},
            "nbformat": 4, "nbformat_minor": 4}


def self_test() -> None:
    nb = _fixture()

    # cell 0 is the unpaired "# About this notebook"; every later section is a
    # markdown cell plus a code cell, so section N starts at 2*N - 1.
    expect = FIXTURE_HEADINGS.index("## 2. Precaching") * 2 - 1
    at = find_injection_point(nb, None)
    assert at == expect, (at, expect)
    say("  anchor: insert at %d (expected %d) OK" % (at, expect))

    nb2 = inject(nb, PATCH, APPLIER, LEVER_TAG)
    verify_injection(nb2, PATCH, LEVER_TAG)
    assert len(nb2["cells"]) == len(nb["cells"]) + 2, "expected 2 cells added"
    where = lever_cells_present(nb2, LEVER_TAG)
    assert where == [expect, expect + 1], where
    say("  insert: %d -> %d cells, lever at %s OK" % (len(nb["cells"]), len(nb2["cells"]), where))

    before = "".join(cell_text(c) for c in nb2["cells"][:expect])
    after = "".join(cell_text(c) for c in nb2["cells"][expect + 2:])
    assert "## 1. Environment and submission mode" in before
    assert "harness-changes.patch" in before, "the harness patch cell must precede the lever"
    assert "## 2. Precaching" not in before
    assert "## 3. Install the ARC runtime" in after, "the install must still follow the lever"
    say("  order: harness patch before, runtime install after OK")

    nb3 = inject(nb2, PATCH + "# changed\n", APPLIER, LEVER_TAG)
    assert len(nb3["cells"]) == len(nb2["cells"]), "re-inject must not duplicate"
    assert "# changed" in "".join(cell_text(nb3["cells"][i]) for i in lever_cells_present(nb3, LEVER_TAG))
    verify_injection(nb3, PATCH + "# changed", LEVER_TAG)
    say("  idempotent re-inject: still 2 cells, body updated OK")

    nb4 = _fixture()
    nb4["cells"] = [c for c in nb4["cells"]
                    if "## 1. Environment and submission mode" not in cell_text(c)]
    at4 = find_injection_point(nb4, None)
    assert cell_text(nb4["cells"][at4]).startswith("## 2."), at4
    say("  fallback anchor (heading 1 absent): insert at %d OK" % at4)

    nb5 = _fixture()
    for c in nb5["cells"]:
        c["source"] = cell_text(c).replace("BUNDLE_DIR = ", "_unused = ")
    try:
        inject(nb5, PATCH, APPLIER, LEVER_TAG)
    except SystemExit:
        say("  upstream-name guard rejects a section 1 that lost BUNDLE_DIR OK")
    else:
        raise AssertionError("upstream-name guard did not fire")

    assert json.dumps(inject(nb2, PATCH, APPLIER, LEVER_TAG), sort_keys=True) == \
        json.dumps(nb2, sort_keys=True)
    say("  no-change detection OK")

    assert parse_push_version("Kernel version 17 successfully pushed.  x") == 17
    say("  push version parse OK")

    subs = [
        {"ref": 3, "date": "2026-10-12 09:00:00", "description": "kaggle5 lever-2 v9", "publicScore": 26.1},
        {"ref": 2, "date": "2026-10-11 09:00:00", "description": "kaggle5 lever-2 v8", "publicScore": 25.9},
        {"ref": 1, "date": "2026-10-03 09:00:00", "description": "Duck harness copy", "publicScore": 25.83},
    ]
    assert submitted_versions(subs) == {8, 9}, submitted_versions(subs)
    assert len(submitted_today(subs, "2026-10-12")) == 1
    assert len(submitted_today(subs, "2026-10-13")) == 0
    assert date_of(subs[0]) == "2026-10-12"
    say("  submission history: version tags, today filter, date parse OK")

    bad = {"competition_sources": [], "dataset_sources": [], "model_sources": [],
           "enable_internet": False, "enable_gpu": False}
    try:
        guard_metadata(bad)
    except SystemExit:
        say("  metadata guard rejects a kernel with no inputs OK")
    else:
        raise AssertionError("guard_metadata did not reject a kernel with no inputs")

    good = {"competition_sources": EXPECTED_COMPETITION_SOURCES,
            "dataset_sources": EXPECTED_DATASET_SOURCES,
            "model_sources": EXPECTED_MODEL_SOURCES,
            "enable_internet": False, "enable_gpu": False,
            "machine_shape": "nvidiaRtxPro6000"}
    guard_metadata(good)
    assert needs_accelerator_flag(good) is False
    assert needs_accelerator_flag(dict(good, machine_shape="")) is True
    say("  metadata guard accepts the real config; enable_gpu=false does not block OK")

    say("")
    say("SELF TEST PASSED")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="check",
                    choices=["check", "push", "wait", "submit", "score", "daily", "all"])
    ap.add_argument("-v", "--version", type=int, default=0)
    ap.add_argument("--tag", default=LEVER_TAG)
    ap.add_argument("--lever-dir", default=None, help="folder holding the patch + applier")
    ap.add_argument("--work", default=None)
    ap.add_argument("--inject-at", type=int, default=None)
    ap.add_argument("--allow-same", action="store_true")
    ap.add_argument("--no-wait-score", action="store_true")
    ap.add_argument("--wait-here", action="store_true",
                    help="stage=daily blocks for the 9 h run instead of resuming tomorrow")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return

    default_work = "/kaggle/working/kaggle5" if Path("/kaggle/working").is_dir() else str(Path.cwd() / "kaggle5_work")
    work = Path(a.work or default_work)
    lever_dir = Path(a.lever_dir) if a.lever_dir else (
        Path(__file__).resolve().parent if Path(__file__).resolve().parent.is_dir() else Path.cwd())
    patch_path = lever_dir / LEVER_PATCH_FILE
    applier_path = lever_dir / "lever2_apply.py"

    say("kaggle5  stage=%s  kernel=%s/%s  comp=%s" % (a.stage, OWNER, KERNEL_SLUG, COMPETITION))
    check_auth()

    if a.stage == "check":
        do_check(work, patch_path, applier_path, a.inject_at)
    elif a.stage == "push":
        do_push(work, patch_path, applier_path, a.inject_at, a.allow_same)
    elif a.stage == "wait":
        do_wait()
    elif a.stage == "submit":
        do_submit(a.version, a.tag)
    elif a.stage == "score":
        do_score(wait=not a.no_wait_score, version=a.version, tag=a.tag)
    elif a.stage == "daily":
        do_daily(work, patch_path, applier_path, a.inject_at, wait_here=a.wait_here)
    elif a.stage == "all":
        v = do_push(work, patch_path, applier_path, a.inject_at, a.allow_same)
        if do_wait():
            do_submit(v, a.tag)
            do_score(wait=not a.no_wait_score, version=v, tag=a.tag)


if __name__ == "__main__":
    main()
