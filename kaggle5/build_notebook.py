#!/usr/bin/env python3
"""Build 4CBOn2_Kaggle5.ipynb from its sources. Never hand-edit the notebook.

Sources
  kaggle5.py                 the driver (embedded + fetched at run time)
  lever2-grid.patch          the lever (fixed artifact; never retyped)
  lever3_apply.py            the applier cell (lever-2 grid + lever-3 memory)
  k5_gym.py                 the local filter gym
  program_synthesis_agent_v4.py  the agent, mirrored from 4CBOn2/Kaggle at
                             build time and PINNED by the sha256 inside
                             kaggle5.py. If the upstream file drifts this
                             build fails loudly instead of embedding stale
                             bytes. Override the mirror path with
                             K5_AGENT_SRC=/local/file when offline.

Preflight runs kaggle5.py --self-test (the whole offline loop) and refuses
to build on failure, then verifies every embedded source is byte-identical
to its file, and fails loudly on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "4CBOn2_Kaggle5.ipynb"

# Where the notebook fetches its sources at run time (repo = source of truth).
REPO = "mohamtur1/arc-agi-3-agent"
BRANCH = "main"
REL = "kaggle5"
AGENT_REPO = "mohamtur1/4CBOn2"
AGENT_REL = "Kaggle"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True},
            "execution_count": None, "outputs": [], "source": src}


def md(src: str) -> dict:
    commented = "\n".join("# " + l if l.strip() else "#" for l in src.splitlines())
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "execution_count": None,
        "outputs": [],
        "source": commented + "\npass\n"
    }


Q3 = "'" * 3   # built, never written literally: this file's text is not embedded


def embed(name: str, body: str) -> dict:
    """A cell that writes `body` to disk only if the fetch above did not.
    Always prints which side won and the sha256 of the file in play."""
    if Q3 in body:
        raise SystemExit(name + " contains a triple quote; cannot embed")
    tmpl = (
        "import hashlib, pathlib\n"
        "D = pathlib.Path('/kaggle/working/kaggle5'); D.mkdir(parents=True, exist_ok=True)\n"
        "t = D / {name!r}\n"
        "EMBEDDED = r" + Q3 + "\n{body}" + Q3 + ".lstrip('\\n')\n"
        "if not t.is_file() or t.stat().st_size < 100:\n"
        "    t.write_text(EMBEDDED)\n"
        "    print('EMBEDDED FALLBACK used ->', t.name, t.stat().st_size, 'bytes')\n"
        "else:\n"
        "    cur = t.read_text()\n"
        "    print('using fetched', t.name, t.stat().st_size, 'bytes',\n"
        "          '(identical to embedded)' if cur == EMBEDDED else\n"
        "          '(DIFFERS from embedded: the repo fetch is the source of truth)')\n"
        "print('  sha256', hashlib.sha256(t.read_bytes()).hexdigest()[:16], t.name)\n"
    )
    src = tmpl.format(name=name, body=body)
    return {"cell_type": "code",
            "metadata": {"trusted": True, "kaggle5_embed": name},
            "execution_count": None, "outputs": [], "source": src}


# --------------------------------------------------------------------------
# gather sources
# --------------------------------------------------------------------------

driver = (HERE / "kaggle5.py").read_text()
patch = (HERE / "lever2-grid.patch").read_text()
applier = (HERE / "lever3_apply.py").read_text()
gym = (HERE / "k5_gym.py").read_text()

# lever2-grid.patch is a FIXED artifact (byte-identical to
# 4CBOn2/Kaggle/franzen-grid-interface.patch). It is never retyped,
# reformatted or rewrapped, so its digest is pinned here and the build
# fails loudly on any drift — including a one-byte accident.
PATCH_SHA256 = "eb25c2d6480ed36eba40e9a05bc358ac2d59995856604d0023d0d67f62bb2cc9"
if hashlib.sha256(patch.encode()).hexdigest() != PATCH_SHA256:
    raise SystemExit(
        "LEVER PATCH DRIFT — refusing to build. lever2-grid.patch is a fixed\n"
        "  artifact. Expected sha256 %s, got %s.\n"
        "  Re-extract it from source (4CBOn2/Kaggle/franzen-grid-interface.patch)\n"
        "  and re-verify the hash; never hand-edit it."
        % (PATCH_SHA256, hashlib.sha256(patch.encode()).hexdigest()))
print("lever patch pin OK: sha256 %s…" % PATCH_SHA256[:16])

agent_name = re.search(r'AGENT_FILE = "([^"]+)"', driver).group(1)
agent_pin = re.search(r'AGENT_SHA256 = "([0-9a-f]{64})"', driver).group(1)
agent_url = ("https://raw.githubusercontent.com/%s/%s/%s/%s"
             % (AGENT_REPO, BRANCH, AGENT_REL, agent_name))

mirror = HERE / agent_name
local_override = os.environ.get("K5_AGENT_SRC", "").strip()
if local_override:
    agent_bytes = pathlib.Path(local_override).read_bytes()
    mirror.write_bytes(agent_bytes)
elif mirror.is_file():
    agent_bytes = mirror.read_bytes()
    fresh_url = agent_url
    try:
        got = urllib.request.urlopen(fresh_url, timeout=30).read()
        agent_bytes = got          # the repo stays the source of truth
        mirror.write_bytes(got)
    except Exception as exc:
        print("build: could not refresh %s (%r); building from the local mirror"
              % (agent_url, exc))
else:
    agent_bytes = urllib.request.urlopen(agent_url, timeout=60).read()
    mirror.write_bytes(agent_bytes)

if hashlib.sha256(agent_bytes).hexdigest() != agent_pin:
    raise SystemExit(
        "AGENT PIN MISMATCH — refusing to build.\n"
        "  %s sha256 %s\n  != pin  %s (AGENT_SHA256 in kaggle5.py)\n"
        "  This is the anti-silent-drift guard doing its job. If the agent "
        "changed\n  on purpose, re-pin kaggle5.py first, then rebuild."
        % (agent_name, hashlib.sha256(agent_bytes).hexdigest(), agent_pin))
agent = agent_bytes.decode()
print("agent mirror OK: %s sha256 %s… (matches the driver pin)"
      % (agent_name, agent_pin[:16]))

r = subprocess.run([sys.executable, str(HERE / "kaggle5.py"), "--self-test"],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit("kaggle5.py --self-test FAILED, not building:\n"
                     + r.stdout + r.stderr)
print("preflight: kaggle5.py --self-test passed")

fetch_list = [f for f in ("kaggle5.py", "lever2-grid.patch", "lever3_apply.py",
                          "k5_gym.py")]

# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------

cells = [
    code(
        "# ==============================================================================\n"
        "# 4CBOn2 — Kaggle5: the between-game learning loop\n"
        "#\n"
        "# This notebook is the daily driver. On a schedule it: pulls the ledger,\n"
        "# reads yesterday's score back, picks the next one-knob experiment from\n"
        "# the fixed queue, smoke-filters it on five toy games, writes the pending\n"
        "# row, pushes the GPU kernel and stops.\n"
        "#\n"
        "# THE MACHINE NEVER SUBMITS. Finalize submissions manually in the Kaggle app.\n"
        "# Accelerator: None (CPU-only) | Internet: ON\n"
        "# ==============================================================================\n"
        "print('4CBOn2 Kaggle5: Daily driver initialized. Ready.')"
    ),

    md("## 1 — The Kaggle CLI\n\n"
       "Install it only if the image lacks it; the driver probes every flag it"
       " relies on against the CLI actually present, each run (>= 2.2,< 3: flags"
       " verified live on 2.2.4)."),
    code("import subprocess, sys\n"
         "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade',\n"
         "                'kaggle>=2.2,<3'], check=True)\n"
         "print(subprocess.run(['kaggle', '--version'], capture_output=True,\n"
         "                     text=True).stdout.strip())"),

    md("## 2 — Get the driver, the lever, the gym and the agent\n\n"
       "The GitHub repo stays the source of truth; anything fetched is hashed and"
       " printed. Files that fail to fetch fall back to the embedded byte-verified"
       " copies in section 2b — **loudly**."),
    code(
        "import hashlib, os, pathlib\n"
        "D = pathlib.Path('/kaggle/working/kaggle5'); D.mkdir(parents=True, exist_ok=True)\n"
        "BASE_K5 = 'https://raw.githubusercontent.com/%s/%s/%s'\n"
        "BASE_4CB = 'https://raw.githubusercontent.com/%s/%s/%s'\n"
        "FAILED = []\n"
        "for url, name in [(f'{BASE_K5}/{f}', f) for f in %r] + [(f'{BASE_4CB}/%s', %r)]:\n"
        "    rc = os.system(f'curl -fsSL {url} -o {D}/{name}')\n"
        "    if rc == 0:\n"
        "        data = (D / name).read_bytes()\n"
        "        print('fetched  %%-32s %%7d bytes  sha256 %%s'\n"
        "              %% (name, len(data), hashlib.sha256(data).hexdigest()[:16]))\n"
        "    else:\n"
        "        FAILED.append(name)\n"
        "        print('FETCH FAILED (rc=%%d) %%s -> embedded fallback will be used' %% (rc, name))\n"
        "print()\n"
        "if FAILED:\n"
        "    print('!' * 60)\n"
        "    print('FETCH FAILURES:', ', '.join(FAILED))\n"
        "    print('using the embedded byte-verified copies in the cells below')\n"
        "    print('!' * 60)\n"
        "else:\n"
        "    print('all 5 files fetched from GitHub (source of truth).')\n"
        "print(sorted(p.name for p in D.iterdir()))"
        % (REPO, BRANCH, REL, AGENT_REPO, BRANCH, AGENT_REL,
           fetch_list, agent_name, agent_name)
    ),

    md("## 2b — Embedded fallbacks\n\n"
       "Each cell writes its file only if the fetch above did not, then prints"
       " the sha256 of whatever bytes are in play."),
    embed("kaggle5.py", driver),
    embed("lever2-grid.patch", patch),
    embed("lever3_apply.py", applier),
    embed("k5_gym.py", gym),
    embed(agent_name, agent),

    md("## 3 — Auth (never aborts)\n\n"
       "Order: `KAGGLE_API_TOKEN`, then `KAGGLE_API_V1_TOKEN` (what Kaggle"
       " injects itself — mapped into the modern name), then token files or"
       " `KAGGLE_USERNAME`+`KAGGLE_KEY`, then an interactive prompt **only when"
       " a TTY exists**, and finally Kaggle's ambient notebook auth. A scheduled"
       " (papermill) run has no stdin, so no code path here may raise."),
    code(
        "import os, pathlib, sys\n"
        "def _have_file():\n"
        "    for p in ('~/.kaggle/access_token', '~/.kaggle/access_token.txt',\n"
        "              '~/.kaggle/kaggle.json'):\n"
        "        f = pathlib.Path(os.path.expanduser(p))\n"
        "        if f.is_file() and f.read_text().strip():\n"
        "            return p\n"
        "    return None\n"
        "if os.environ.get('KAGGLE_API_TOKEN'):\n"
        "    print('auth: KAGGLE_API_TOKEN is set')\n"
        "elif os.environ.get('KAGGLE_API_V1_TOKEN'):\n"
        "    os.environ['KAGGLE_API_TOKEN'] = os.environ['KAGGLE_API_V1_TOKEN']\n"
        "    print('auth: mapped KAGGLE_API_V1_TOKEN (Kaggle-injected) into KAGGLE_API_TOKEN')\n"
        "elif _have_file():\n"
        "    print('auth: using', _have_file())\n"
        "elif os.environ.get('KAGGLE_USERNAME') and os.environ.get('KAGGLE_KEY'):\n"
        "    print('auth: using KAGGLE_USERNAME/KAGGLE_KEY')\n"
        "elif sys.stdin is not None and sys.stdin.isatty():\n"
        "    import getpass\n"
        "    tok = getpass.getpass('Paste a KGAT_ token (blank = use ambient auth): ').strip()\n"
        "    if tok:\n"
        "        os.environ['KAGGLE_API_TOKEN'] = tok\n"
        "    print('continuing with whatever the environment provides')\n"
        "else:\n"
        "    print('auth: no token and no TTY (scheduled run) — using Kaggle '\n"
        "          'ambient notebook auth and CONTINUING')"
    ),

    md("## 4 — Free dry run\n\n"
       "Probes the CLI, verifies the 5 inputs + agent pin + defaults, dry-runs"
       " the lever+knob injection, scans the fork for the K1 knob channel, then"
       " probes submission-limits (informational). **Pushes nothing.**"),
    code("%cd /kaggle/working/kaggle5\n!python kaggle5.py --stage check"),

    md("## 5 — Read-only: what would today do?\n\n"
       "Reconciles any new submission against the pending row and prints the"
       " experiment table + the next plan. Pushes nothing."),
    code("!python kaggle5.py --stage plan"),

    md("## 6 — THE DAILY CELL (this is the one you schedule)\n\n"
       "| situation | what it does |\n"
       "|---|---|\n"
       "| yesterday's submission scored | reconciles, then plans from it |\n"
       "| pending row still unanswered | prints why, pushes nothing |\n"
       "| variant planned | toy-filter -> pending row -> push -> **stop** |\n"
       "| filter rejects the variant | records it, pushes nothing, moves on tomorrow |\n"
       "| queue exhausted | prints the incumbent, pushes nothing, ever |\n"
       "\n"
       "A scheduled run never blocks 9 h and never submits. It emails you the"
       " run log; the bottom of a push log is the human card (what to tap)."),
    code("!python kaggle5.py --stage daily"),

    md("## 7 — Manual ops (optional, run only by hand when needed)\n"),
    code("# Optional manual ops — uncomment if needed:\n"
         "# !python kaggle5.py --stage plan\n"
         "# !python kaggle5.py --stage score\n"
         "# !python kaggle5.py --abandon\n"
         "pass"),

    md(
        "## The machine never submits — what you do instead\n"
        "\n"
        "After a push, the run prints the card. The taps:\n"
        "\n"
        "1. open the fork: https://www.kaggle.com/code/mohamedmalikturay/arc-agi-3-milestone-2-solution-3e9adb\n"
        "2. wait for the pushed version to finish, then '...' -> **Submit to"
        " arc-prize-2026-arc-agi-3** -> pick the pushed version\n"
        "3. paste the message the card prints (`kaggle5 <variant> v<N>`) — it is"
        " how tomorrow's run matches the score to the experiment with certainty\n"
        "\n"
        "Skip a day? The driver waits ~2 days, marks the row stale, and moves on"
        " — nothing is silently dropped. Want to skip deliberately? Run"
        " `--abandon` from cell 7 or tell the next run to.\n"
        "\n"
        "## Scheduling it\n"
        "\n"
        "Settings -> **Schedule a notebook to run** -> Daily. CPU-only notebooks"
        " can be scheduled (this one is); runs fire roughly on the UTC date;"
        " every run emails you its log, pass or fail; save a version first — the"
        " schedule runs the saved notebook.\n"
        "\n"
        "LOCAL_UTC_OFFSET_HOURS in kaggle5.py is 0 (Freetown is UTC+0 all year)."
    ),
]

nb = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3",
                       "name": "python3"},
        "language_info": {"name": "python", "file_extension": ".py",
                          "mimetype": "text/x-python"},
        "kaggle": {
            "accelerator": "none",       # required for scheduling
            "isInternetEnabled": True,   # required for the API + GitHub
            "isGpuEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells,
}

OUT.write_text(json.dumps(nb, indent=1))
print("wrote", OUT.name, OUT.stat().st_size, "bytes,", len(cells), "cells")

# ---- verify the artefact -------------------------------------------------
loaded = json.loads(OUT.read_text())
assert len(loaded["cells"]) == len(cells)
for c in loaded["cells"]:
    assert isinstance(c["source"], str) and c["source"]
    assert c["cell_type"] in ("code", "markdown")
acc = loaded["metadata"]["kaggle"]
assert acc["accelerator"] == "none" and acc["isGpuEnabled"] is False, \
    "must be CPU to be schedulable"
assert acc["isInternetEnabled"] is True, "driver needs internet"
print("verify: %d cells, CPU-only, internet on (schedulable)" % len(loaded["cells"]))


def extract(name: str) -> str:
    hits = [c for c in loaded["cells"]
            if c.get("metadata", {}).get("kaggle5_embed") == name]
    assert len(hits) == 1, "expected exactly one embed cell for %r, got %d" \
        % (name, len(hits))
    src = hits[0]["source"]
    return src.split("r" + Q3 + "\n", 1)[1].rsplit(Q3, 1)[0]


d = extract("kaggle5.py")
a = extract("lever3_apply.py")
pb = extract("lever2-grid.patch")
g = extract("k5_gym.py")
ag = extract(agent_name)
for label, text in (("kaggle5.py", d), ("lever3_apply.py", a),
                    ("k5_gym.py", g), (agent_name, ag)):
    compile(text, label, "exec")
assert d.strip() == driver.strip(), "embedded driver differs from source"
assert a.strip() == applier.strip(), "embedded applier differs from source"
assert pb.strip() == patch.strip(), "embedded patch differs from source"
assert g.strip() == gym.strip(), "embedded gym differs from source"
assert ag.strip() == agent.strip(), "embedded agent differs from mirror"
print("verify: driver %d, applier %d, patch %d, gym %d, agent %d chars -"
      " all five byte-identical to source" % (len(d), len(a), len(pb), len(g), len(ag)))

fetch_src = next(c for c in loaded["cells"] if c["cell_type"] == "code"
                 and "BASE_K5 =" in c["source"])["source"]
assert "{{BASE}}" not in fetch_src and "{BASE_K5}" in fetch_src, \
    "bug B regression: fetch cell must interpolate BASE"
# bug A regression rule: getpass must never appear without a TTY guard,
# in ANY cell (the fatal version called it on a papermill run with no stdin)
for c in loaded["cells"]:
    if c["cell_type"] == "code" and "getpass" in c["source"]:
        assert "isatty()" in c["source"], \
            "bug A regression: getpass without a TTY guard in a code cell"
print("verify: bug A (auth never aborts) and bug B (BASE interpolation, loud"
      " fallback) fixes present in the artefact")
