"""kaggle5 — inject one lever cell into your ARC-AGI-3 notebook, push, submit, score.

Designed to run INSIDE a Kaggle notebook session on an Android phone.

Stages:
  check    pull the notebook + metadata, print the map, change nothing, push nothing
  push     pull, inject, verify, push a new version (this RUNS the notebook, up to 9h)
  wait     poll `kaggle kernels status` until the pushed version finishes
  submit   submit that version to the competition (BURNS ONE DAILY SUBMISSION)
  score    read the public score back and compare to BASELINE
  all      push -> wait -> submit -> score

Nothing is pushed and nothing is submitted unless you ask for that stage.

Run `python kaggle5.py --self-test` to exercise the injector offline (no auth, no network).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG — edit only this block
# --------------------------------------------------------------------------

OWNER = "mohamedmalikturay"
KERNEL_SLUG = "arc-agi-3-milestone-2-solution"          # <-- your fork's slug
COMPETITION = "arc-prize-2026-arc-agi-3"
SUBMISSION_FILE = "submission.parquet"                   # verified: ARC-AGI-3 output file
BASELINE_SCORE = 25.83
LEVER_TAG = "lever-2"                                    # one token, no spaces
LEVER_FILE = "lever2.py"                                 # the .py holding the lever cell body

# Anchors used to place the lever cell.
INJECT_AFTER_HEADING = "4. import the bundled source"    # markdown heading to insert AFTER
INJECT_BEFORE_HEADING = "5."                             # fallback: insert BEFORE first heading starting with this

# The 5 inputs the Franzen/Duck harness needs. push aborts if any is missing.
EXPECTED_COMPETITION_SOURCES = ["arc-prize-2026-arc-agi-3"]
EXPECTED_DATASET_SOURCES = [
    "dfranzen/pennyroyal-v253",
    "dfranzen/taaf-kaggle-source-bundle-copy",
]
EXPECTED_MODEL_SOURCES = [
    "dfranzen/intel-qwen3.8-flash-next-w4a16-autoround",
    "dfranzen/albucino-qwen3-8-flash-next-drafter",
]

# Only used if the pulled metadata has enable_gpu=true but an empty machine_shape
# (i.e. the GPU was set in the UI and never round-tripped through the API).
FALLBACK_ACCELERATOR = "NvidiaRtxPro6000"

STATUS_POLL_SECONDS = 60
STATUS_MAX_SECONDS = 10 * 3600          # 10h; ARC-AGI-3 notebooks are capped at 9h
SCORE_POLL_SECONDS = 120
SCORE_MAX_SECONDS = 6 * 3600

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
# auth
# --------------------------------------------------------------------------

def check_auth() -> None:
    """Refuse to go further if the CLI has no credentials."""
    token = os.environ.get("KAGGLE_API_TOKEN", "")
    if token:
        say("auth: KAGGLE_API_TOKEN is set (%d chars)" % len(token))
        return
    for p in ("~/.kaggle/access_token", "~/.kaggle/access_token.txt", "~/.kaggle/kaggle.json"):
        f = Path(os.path.expanduser(p))
        if f.is_file() and f.read_text().strip():
            say("auth: using %s" % p)
            return
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        say("auth: using legacy KAGGLE_USERNAME/KAGGLE_KEY")
        return
    die(
        "no Kaggle credentials found.\n"
        "  Token: https://www.kaggle.com/settings  (API -> Create New Token)\n"
        "  then:  import os; os.environ['KAGGLE_API_TOKEN']='KGAT_...'\n"
        "  (inside a Kaggle notebook the SDK may already inject one; run check first)"
    )


def kaggle(args: list[str], capture: bool = True, timeout: int = 900) -> subprocess.CompletedProcess:
    """Run the kaggle CLI. capture=False streams straight to the notebook output."""
    cmd = [shutil.which("kaggle") or "kaggle"] + args
    say("$ " + " ".join(cmd))
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
        timeout=timeout,
    )
    if capture and r.stdout:
        say(r.stdout.rstrip())
    if r.returncode != 0:
        die("kaggle %s exited %d" % (args[0], r.returncode))
    return r


# --------------------------------------------------------------------------
# pull
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


# --------------------------------------------------------------------------
# metadata guard — the MCP failure mode, defended against
# --------------------------------------------------------------------------

def guard_metadata(meta: dict) -> None:
    say("")
    say("--- data sources on the pulled kernel ---")
    for key in ("competition_sources", "dataset_sources", "kernel_sources", "model_sources"):
        say("  %-22s %s" % (key, meta.get(key) or []))
    say("  %-22s %s" % ("enable_gpu", meta.get("enable_gpu")))
    say("  %-22s %s" % ("enable_internet", meta.get("enable_internet")))
    say("  %-22s %r" % ("machine_shape", meta.get("machine_shape")))
    say("  %-22s %r" % ("docker_image", meta.get("docker_image")))

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
    if not meta.get("enable_gpu"):
        problems.append("enable_gpu is falsy — this harness needs the GPU")

    if problems:
        die(
            "the pulled kernel is not configured the way your baseline is.\n"
            "  " + "\n  ".join(problems) + "\n"
            "  Pushing now would silently drop inputs and burn a run. Fix on kaggle.com first."
        )
    say("  -> all 5 inputs present, internet off, GPU on. OK to push.")


def needs_accelerator_flag(meta: dict) -> bool:
    return bool(meta.get("enable_gpu")) and not (meta.get("machine_shape") or "").strip()


def guard_slug(meta: dict) -> None:
    """A title that does not slugify to the id makes Kaggle create a NEW kernel
    instead of a new version of the existing one. Refuse rather than fork by accident."""
    title = (meta.get("title") or "").strip()
    kslug = (meta.get("id") or "").split("/")[-1]
    if not title or not kslug:
        return
    slug = re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", title.lower())).strip("-")
    if slug and slug != kslug.lower():
        say("  note: title %r slugifies to %r, id slug is %r" % (title, slug, kslug))
        say("        Kaggle warns on this; the id in the metadata is what wins, so the push")
        say("        should still land on %s/%s. Check the printed URL after pushing." % (OWNER, KERNEL_SLUG))


# --------------------------------------------------------------------------
# injection
# --------------------------------------------------------------------------

def cell_text(cell: dict) -> str:
    src = cell.get("source", "")
    return src if isinstance(src, str) else "".join(src)


def headings(nb: dict) -> list[tuple[int, str]]:
    out = []
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "markdown":
            first = cell_text(c).strip().splitlines()[0] if cell_text(c).strip() else ""
            out.append((i, first))
    return out


def find_injection_point(nb: dict, explicit_index: int | None) -> int:
    """Return the index the lever cell should be inserted at."""
    cells = nb.get("cells", [])
    if explicit_index is not None:
        if not 0 <= explicit_index <= len(cells):
            die("INJECT_AT_INDEX=%d out of range (notebook has %d cells)" % (explicit_index, len(cells)))
        return explicit_index

    target = norm(INJECT_AFTER_HEADING)
    for i, c in enumerate(cells):
        if c.get("cell_type") != "markdown":
            continue
        for line in cell_text(c).splitlines():
            if line.lstrip().startswith("#") and norm(line.lstrip("#")) == target:
                say("  anchor: heading %r at cell %d -> insert at %d" % (line.strip(), i, i + 1))
                return i + 1

    prefix = norm(INJECT_BEFORE_HEADING)
    for i, c in enumerate(cells):
        if c.get("cell_type") != "markdown":
            continue
        for line in cell_text(c).splitlines():
            if line.lstrip().startswith("#") and norm(line.lstrip("#")).startswith(prefix):
                say("  fallback anchor: heading %r at cell %d -> insert at %d" % (line.strip(), i, i))
                return i

    die(
        "no anchor found. Run stage=check, read the heading map, then set INJECT_AT_INDEX.\n"
        "  Expected a markdown heading matching %r or starting %r." % (INJECT_AFTER_HEADING, INJECT_BEFORE_HEADING)
    )
    raise AssertionError  # unreachable


def marker(lever_tag: str, action: str = "ADD") -> str:
    return f"# ===== {action} {lever_tag} (kaggle5) ====="


def lever_already_present(nb: dict, lever_tag: str) -> int | None:
    for i, c in enumerate(nb.get("cells", [])):
        if c.get("cell_type") == "code" and marker(lever_tag) in cell_text(c):
            return i
    return None


def build_lever_cell(lever_body: str, lever_tag: str) -> dict:
    body = f"{marker(lever_tag)}\n{lever_body.strip()}\n"
    # kaggle kernels push joins list-of-lines source into one string anyway;
    # emit a plain string so what we write is what the server stores.
    return {
        "cell_type": "code",
        "metadata": {"trusted": True, "kaggle5": lever_tag},
        "execution_count": None,
        "outputs": [],
        "source": body,
    }


def inject(nb: dict, lever_body: str, lever_tag: str, explicit_index: int | None = None) -> dict:
    """Return a new notebook dict with exactly one lever cell in the right place."""
    nb = json.loads(json.dumps(nb))  # deep copy

    existing = lever_already_present(nb, lever_tag)
    if existing is not None:
        say("  %s already present at cell %d — replacing in place (no duplicate)" % (lever_tag, existing))
        nb["cells"][existing] = build_lever_cell(lever_body, lever_tag)
        return nb

    at = find_injection_point(nb, explicit_index)
    nb["cells"].insert(at, build_lever_cell(lever_body, lever_tag))
    say("  inserted %s at cell %d" % (lever_tag, at))
    return nb


def verify_injection(nb: dict, lever_body: str, lever_tag: str) -> None:
    hits = [i for i, c in enumerate(nb.get("cells", [])) if marker(lever_tag) in cell_text(c)]
    if len(hits) != 1:
        die("post-check: found %d lever cells, expected exactly 1 (at %s)" % (len(hits), hits))
    cell = nb["cells"][hits[0]]
    if lever_body.strip() not in cell_text(cell):
        die("post-check: lever body not intact in cell %d" % hits[0])
    if cell.get("cell_type") != "code":
        die("post-check: lever cell is %r, expected 'code'" % cell.get("cell_type"))
    say("  post-check OK: exactly one %s cell, body intact, cell_type=code" % lever_tag)


# --------------------------------------------------------------------------
# push / wait / submit / score
# --------------------------------------------------------------------------

VERSION_RE = re.compile(r"Kernel version (\d+) successfully")


def parse_push_version(stdout: str) -> int:
    m = VERSION_RE.search(stdout or "")
    if not m:
        die(
            "could not read a version number out of the push output.\n"
            "  Look for 'Kernel version N successfully pushed' above, then run stage=submit -v N."
        )
    return int(m.group(1))


def do_push(work: Path, lever_path: Path, inject_at: int | None, allow_same: bool) -> int:
    lever_body = lever_path.read_text()
    if not lever_body.strip():
        die("%s is empty — nothing to inject" % lever_path.name)

    code, meta = pull(work)
    nb = json.loads(code.read_text())
    say("pulled %s: %d cells" % (code.name, len(nb.get("cells", []))))

    guard_metadata(meta)
    guard_slug(meta)

    before = json.loads(json.dumps(nb))
    nb2 = inject(nb, lever_body, LEVER_TAG, inject_at)
    verify_injection(nb2, lever_body, LEVER_TAG)

    if json.dumps(before, sort_keys=True) == json.dumps(nb2, sort_keys=True):
        if not allow_same:
            die(
                "the injected notebook is byte-identical to what is already on Kaggle.\n"
                "  Pushing would create a version with no change and could burn a run.\n"
                "  Edit %s, or pass --allow-same to push anyway." % lever_path.name
            )
        say("  --allow-same given: pushing an unchanged notebook")

    code.write_text(json.dumps(nb2, indent=1))
    say("  wrote %s (%d cells)" % (code.name, len(nb2["cells"])))

    args = ["kernels", "push", "-p", str(work)]
    if needs_accelerator_flag(meta):
        say("  machine_shape is empty on the pulled metadata; passing --accelerator %s" % FALLBACK_ACCELERATOR)
        args += ["--accelerator", FALLBACK_ACCELERATOR]
    out = kaggle(args, timeout=600).stdout
    version = parse_push_version(out)
    say("")
    say("PUSHED version %d of %s/%s" % (version, OWNER, KERNEL_SLUG))
    say("Phase A (Save & Run All) is now running. Track it with:  stage=wait")
    return version


def kernel_status() -> str:
    out = kaggle(["kernels", "status", f"{OWNER}/{KERNEL_SLUG}"]).stdout
    m = re.search(r'has status "([^"]+)"', out)
    return (m.group(1) if m else out.strip()).lower()


def do_wait() -> None:
    say("waiting on %s/%s (poll %ds, max %dh)" % (OWNER, KERNEL_SLUG, STATUS_POLL_SECONDS, STATUS_MAX_SECONDS // 3600))
    deadline = time.time() + STATUS_MAX_SECONDS
    last = None
    while time.time() < deadline:
        st = kernel_status()
        if st != last:
            say("  [%s] %s" % (time.strftime("%H:%M:%S"), st))
            last = st
        if st in ("complete", "cancelled", "error"):
            if st != "complete":
                die("the pushed run ended as %r — do NOT submit. Read the log with:\n  kaggle kernels logs %s/%s"
                    % (st, OWNER, KERNEL_SLUG))
            say("")
            say("Phase A COMPLETE. Safe to submit. Next:  stage=submit")
            return
        time.sleep(STATUS_POLL_SECONDS)
    die("still %r after %dh. Re-run stage=wait later." % (last, STATUS_MAX_SECONDS // 3600))


def do_submit(version: int, tag: str, no_wait: bool) -> None:
    if version <= 0:
        die("stage=submit needs the version. Pass -v N (the number stage=push printed).")
    st = kernel_status()
    if st != "complete":
        die("the kernel's latest run is %r, not complete. Submitting now would fail. Run stage=wait first." % st)

    say("")
    say(">>> THIS BURNS ONE DAILY SUBMISSION <<<")
    kaggle(
        [
            "competitions", "submit", COMPETITION,
            "-f", SUBMISSION_FILE,
            "-k", f"{OWNER}/{KERNEL_SLUG}",
            "-v", str(version),
            "-m", f"kaggle5 {tag} v{version}",
        ],
        capture=False,
        timeout=900,
    )
    say("")
    say("submitted v%d. Next:  stage=score" % version)


def fetch_submissions() -> list[dict]:
    out = kaggle(["competitions", "submissions", COMPETITION, "--format", "json", "--page-size", "20"]).stdout
    try:
        data = json.loads(out[out.index("["): out.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        die("could not parse `competitions submissions` as JSON:\n" + out[:800])
    return data if isinstance(data, list) else []


def do_score(wait: bool, version: int, tag: str) -> None:
    needle = f"kaggle5 {tag} v{version}" if version > 0 else ""
    deadline = time.time() + (SCORE_MAX_SECONDS if wait else 0)
    last_status = None
    while True:
        subs = fetch_submissions()
        pick = None
        if needle:
            pick = next((s for s in subs if needle in (s.get("description") or "")), None)
        if pick is None and subs:
            pick = subs[0]
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
                    say("SUBMISSION ERRORED — no slot result to compare.")
                    return
                try:
                    val = float(score)
                except (TypeError, ValueError):
                    say("submission complete but publicScore is %r — read it on kaggle.com" % score)
                    return
                delta = val - BASELINE_SCORE
                say("SCORE  %.4f   (baseline %.4f)   delta %+.4f  %s"
                    % (val, BASELINE_SCORE, delta, "BETTER" if delta > 0 else ("same" if delta == 0 else "worse")))
                return
        if not wait or time.time() >= deadline:
            say("not scored yet. Re-run stage=score later.")
            return
        time.sleep(SCORE_POLL_SECONDS)


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------

def do_check(work: Path, lever_path: Path, inject_at: int | None) -> None:
    code, meta = pull(work)
    guard_metadata(meta)
    guard_slug(meta)

    nb = json.loads(code.read_text())
    cells = nb.get("cells", [])
    say("")
    say("--- notebook map: %d cells ---" % len(cells))
    for i, c in enumerate(cells):
        if c.get("cell_type") == "markdown":
            first = cell_text(c).strip().splitlines()[0] if cell_text(c).strip() else ""
            say("  [%3d] md   %s" % (i, first[:78]))
        else:
            first = cell_text(c).strip().splitlines()[0] if cell_text(c).strip() else ""
            say("  [%3d] code %s" % (i, first[:78]))

    say("")
    try:
        at = find_injection_point(nb, inject_at)
    except SystemExit:
        say("no anchor matched — set INJECT_AT_INDEX from the map above and re-run stage=check")
        return

    existing = lever_already_present(nb, LEVER_TAG)
    say("lever cell present already: %s" % (existing if existing is not None else "no"))

    body = lever_path.read_text() if lever_path.is_file() else ""
    if body.strip():
        nb2 = inject(nb, body, LEVER_TAG, inject_at)
        verify_injection(nb2, body, LEVER_TAG)
        say("dry-run injection: %d cells -> %d cells, lever at cell %d"
            % (len(cells), len(nb2["cells"]), lever_already_present(nb2, LEVER_TAG)))
        say("")
        say("--- first 6 lines of what would be injected ---")
        for line in cell_text(nb2["cells"][lever_already_present(nb2, LEVER_TAG)]).splitlines()[:6]:
            say("  | " + line)
    else:
        say("%s is empty — write the lever body into it before stage=push" % lever_path.name)

    say("")
    say("nothing was pushed. Next:  stage=push")


# --------------------------------------------------------------------------
# self test
# --------------------------------------------------------------------------

FIXTURE_HEADINGS = [
    "# 1. Setup",
    "# 2. Install",
    "# 3. Configure",
    "# 4. Import the bundled source",
    "# 5. Run the agent",
    "# 6. Write the submission",
]


def _fixture() -> dict:
    cells = []
    for h in FIXTURE_HEADINGS:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": h + "\n\nprose"})
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": "print(%r)\n" % h})
    return {"cells": cells, "metadata": {"kaggle": {"accelerator": "nvidiaRtxPro6000"}},
            "nbformat": 4, "nbformat_minor": 4}


def self_test() -> None:
    body = "def _grid(self):\n    return self._board\nFrameView.grid = property(_grid)\n"
    nb = _fixture()

    at = find_injection_point(nb, None)
    expect = FIXTURE_HEADINGS.index("# 4. Import the bundled source") * 2 + 1
    assert at == expect, (at, expect)
    say("  anchor: insert at %d (expected %d) OK" % (at, expect))

    nb2 = inject(nb, body, LEVER_TAG)
    verify_injection(nb2, body, LEVER_TAG)
    assert len(nb2["cells"]) == len(nb["cells"]) + 1
    assert lever_already_present(nb2, LEVER_TAG) == expect
    say("  insert: %d -> %d cells, marker at %d OK" % (len(nb["cells"]), len(nb2["cells"]), expect))

    # everything before the lever still belongs to section 4 or earlier;
    # everything from the next markdown heading on must be section 5.
    before = "".join(cell_text(c) for c in nb2["cells"][:expect])
    assert "4. Import the bundled source" in before, "section 4 not before the lever"
    assert "5. Run the agent" not in before, "section 5 leaked in before the lever"
    after = [cell_text(c) for c in nb2["cells"][expect + 1:]]
    assert "# 5. Run the agent" not in before, "section 5 leaked in before the lever"
    next_md = next(t for t in after if t.strip().startswith("#"))
    assert next_md.strip().splitlines()[0] == "# 5. Run the agent", next_md[:60]
    say("  neighbours: section 4 before, section 5 is the next heading after OK")

    # idempotent: second inject replaces, never duplicates
    nb3 = inject(nb2, body + "# changed\n", LEVER_TAG)
    assert len(nb3["cells"]) == len(nb2["cells"])
    assert "# changed" in cell_text(nb3["cells"][expect])
    verify_injection(nb3, body + "# changed", LEVER_TAG)
    say("  idempotent re-inject: still 1 cell, body updated OK")

    # fallback anchor when the exact heading is missing
    nb4 = _fixture()
    nb4["cells"] = [c for c in nb4["cells"]
                    if "4. Import the bundled source" not in cell_text(c)]
    at4 = find_injection_point(nb4, None)
    assert cell_text(nb4["cells"][at4]).startswith("# 5."), at4
    say("  fallback anchor (heading 4 absent): insert at %d, before section 5 OK" % at4)

    # no-change detection
    assert json.dumps(inject(nb2, body, LEVER_TAG), sort_keys=True) == json.dumps(nb2, sort_keys=True)
    say("  no-change detection OK")

    # version parsing
    assert parse_push_version("Kernel version 17 successfully pushed.  Please check progress at https://x") == 17
    say("  push version parse OK")

    # metadata guard must reject a kernel missing an input
    bad = {"competition_sources": [], "dataset_sources": [], "model_sources": [],
           "enable_internet": False, "enable_gpu": True}
    try:
        guard_metadata(bad)
    except SystemExit:
        say("  metadata guard rejects missing inputs OK")
    else:
        raise AssertionError("guard_metadata did not reject a kernel with no inputs")

    say("")
    say("SELF TEST PASSED")


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="check",
                    choices=["check", "push", "wait", "submit", "score", "all"])
    ap.add_argument("-v", "--version", type=int, default=0, help="kernel version (from stage=push)")
    ap.add_argument("--tag", default=LEVER_TAG, help="description tag used to find the score later")
    ap.add_argument("--lever", default=None, help="path to the lever .py (default: %s next to this file)" % LEVER_FILE)
    ap.add_argument("--work", default=None, help="scratch dir (default: /kaggle/working/kaggle5)")
    ap.add_argument("--inject-at", type=int, default=None, help="explicit cell index, overrides heading anchors")
    ap.add_argument("--allow-same", action="store_true", help="push even if nothing changed")
    ap.add_argument("--no-wait-score", action="store_true", help="stage=score returns after one poll")
    ap.add_argument("--self-test", action="store_true", help="exercise the injector offline and exit")
    a = ap.parse_args()

    if a.self_test:
        self_test()
        return

    default_work = "/kaggle/working/kaggle5" if Path("/kaggle/working").is_dir() else str(Path.cwd() / "kaggle5_work")
    work = Path(a.work or default_work)
    lever_dir = Path(__file__).resolve().parent if Path(__file__).resolve().parent.is_dir() else Path.cwd()
    lever_path = Path(a.lever) if a.lever else lever_dir / LEVER_FILE

    say("kaggle5  stage=%s  kernel=%s/%s  comp=%s" % (a.stage, OWNER, KERNEL_SLUG, COMPETITION))
    check_auth()

    if a.stage == "check":
        do_check(work, lever_path, a.inject_at)
    elif a.stage == "push":
        do_push(work, lever_path, a.inject_at, a.allow_same)
    elif a.stage == "wait":
        do_wait()
    elif a.stage == "submit":
        do_submit(a.version, a.tag, no_wait=False)
    elif a.stage == "score":
        do_score(wait=not a.no_wait_score, version=a.version, tag=a.tag)
    elif a.stage == "all":
        v = do_push(work, lever_path, a.inject_at, a.allow_same)
        do_wait()
        do_submit(v, a.tag, no_wait=False)
        do_score(wait=not a.no_wait_score, version=v, tag=a.tag)


if __name__ == "__main__":
    main()
