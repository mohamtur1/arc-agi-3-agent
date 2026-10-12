#!/usr/bin/env python3
"""Build 4CBOn2_Kaggle5.ipynb from kaggle5.py + lever2-grid.patch + lever2_apply.py.

Never hand-edit the notebook. Edit the sources, re-run this. It self-tests the
driver first and refuses to build if that fails.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "4CBOn2_Kaggle5.ipynb"

# Where the driver fetches its own sources at run time.
REPO = "mohamtur1/arc-agi-3-agent"
BRANCH = "main"
REL = "kaggle5"


def code(src: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True}, "execution_count": None,
            "outputs": [], "source": src}


def md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": src}


Q3 = "'" * 3   # built, never written literally: this file's text is not embedded


Q3 = "'" * 3   # built, never written literally


def embed(name: str, body: str) -> dict:
    """A cell that writes `body` to disk only if the fetch above did not."""
    if Q3 in body:
        raise SystemExit(name + " contains a triple quote; cannot embed")
    tmpl = (
        "import pathlib\n"
        "D = pathlib.Path('/kaggle/working/kaggle5'); D.mkdir(parents=True, exist_ok=True)\n"
        "t = D / {name!r}\n"
        "if not t.is_file() or t.stat().st_size < 100:\n"
        "    t.write_text(r" + Q3 + "\n{body}" + Q3 + ".lstrip('\\n'))\n"
        "    print('embedded fallback ->', t.name, t.stat().st_size, 'bytes')\n"
        "else:\n"
        "    print('using fetched', t.name, t.stat().st_size, 'bytes')\n"
    )
    src = tmpl.format(name=name, body=body)
    return {"cell_type": "code",
            "metadata": {"trusted": True, "kaggle5_embed": name},
            "execution_count": None, "outputs": [], "source": src}


driver = (HERE / "kaggle5.py").read_text()
patch = (HERE / "lever2-grid.patch").read_text()
applier = (HERE / "lever2_apply.py").read_text()

r = subprocess.run([sys.executable, str(HERE / "kaggle5.py"), "--self-test"],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit("kaggle5.py --self-test FAILED, not building:\n" + r.stdout + r.stderr)
print("preflight: kaggle5.py --self-test passed")

cells = [
    md(
        "# 4CBOn2 — Kaggle5\n"
        "\n"
        "Injects the lever-2 patch into the ARC-AGI-3 notebook, pushes it, submits it, "
        "reads the score. Can run **once a day on a schedule**.\n"
        "\n"
        "## Two ways to use it\n"
        "\n"
        "**Automatic.** Cell 5 is the whole loop in one cell. Set this notebook to run "
        "daily and it works out by itself whether today needs a push, a submit, a score "
        "read, or nothing at all. It is idempotent: run it twice in a day and the second "
        "run does nothing.\n"
        "\n"
        "**Manual.** Cells 6-10 are the same steps one at a time, for when you want to "
        "watch it.\n"
        "\n"
        "## Settings for this notebook\n"
        "\n"
        "- **Accelerator: none.** This notebook only makes API calls. It must be CPU, "
        "because Kaggle will only schedule CPU notebooks.\n"
        "- **Internet: ON.** It needs to reach the Kaggle API and GitHub.\n"
        "- The notebook it *pushes* keeps GPU + internet off. Those come from the pulled "
        "metadata and are verified before every push.\n"
        "\n"
        "## What it will and will not spend\n"
        "\n"
        "| action | cost |\n"
        "|---|---|\n"
        "| `check` | free |\n"
        "| `push` | one GPU run of the target kernel, up to 9 h |\n"
        "| `submit` | **one daily submission slot** |\n"
        "| `score` | free |\n"
        "\n"
        "It refuses to push if any of the 5 inputs is missing, if internet is on, or if "
        "the injection post-check fails. It refuses to submit unless the run is complete. "
        "And it only ever submits the **newest** finished version — never an older one."
    ),

    md("## 1 — Install the Kaggle CLI\n\n"
       "The driver avoids main-only flags, so the plain PyPI wheel works. "
       "Needs internet on."),
    code("!pip install -q --upgrade kaggle 2>&1 | tail -2\n!kaggle --version"),

    md("## 2 — Get the driver and the lever\n\n"
       "Pulls from the repo so the repo stays the source of truth. Edit `lever2-grid.patch` "
       "in git and the next run picks it up with no re-pasting."),
    code(
        "import os, pathlib\n"
        "D = pathlib.Path('/kaggle/working/kaggle5'); D.mkdir(parents=True, exist_ok=True)\n"
        f"BASE = 'https://raw.githubusercontent.com/{REPO}/{BRANCH}/{REL}'\n"
        "for f in ('kaggle5.py', 'lever2-grid.patch', 'lever2_apply.py'):\n"
        "    rc = os.system(f'curl -fsSL {{BASE}}/{{f}} -o {{D}}/{{f}}')\n"
        "    print(f, 'fetched' if rc == 0 else 'FETCH FAILED -> embedded fallback will be used')\n"
        "print(sorted(p.name for p in D.iterdir()))"
    ),

    md("## 2b — Embedded fallbacks\n\n"
       "Each cell writes its file only if the fetch above did not. If you have not pushed "
       "the repo yet, these carry the whole thing."),
    embed("kaggle5.py", driver),
    embed("lever2-grid.patch", patch),
    embed("lever2_apply.py", applier),

    md("## 3 — Auth\n\n"
       "Inside a Kaggle notebook the SDK reads the token Kaggle injects itself, so this "
       "usually needs nothing. If it does, the modern CLI wants `KAGGLE_API_TOKEN` — a "
       "`KGAT_...` token from https://www.kaggle.com/settings (API -> Create New Token). "
       "`kaggle.json` is the legacy path."),
    code(
        "import os, pathlib, getpass\n"
        "have = bool(os.environ.get('KAGGLE_API_TOKEN'))\n"
        "for p in ('~/.kaggle/access_token', '~/.kaggle/kaggle.json'):\n"
        "    f = pathlib.Path(os.path.expanduser(p))\n"
        "    have = have or (f.is_file() and bool(f.read_text().strip()))\n"
        "if have:\n"
        "    print('credentials already present — nothing to paste')\n"
        "else:\n"
        "    tok = getpass.getpass('Paste your KGAT_ token (blank to rely on the injected one): ').strip()\n"
        "    if tok:\n"
        "        os.environ['KAGGLE_API_TOKEN'] = tok\n"
        "    print('continuing with whatever the environment provides')"
    ),

    md("## 4 — Free dry run\n\n"
       "Pulls the kernel, verifies all 5 inputs, prints the heading map, and shows where "
       "the lever would land. **Pushes nothing.** Run this first, every time."),
    code("%cd /kaggle/working/kaggle5\n!python kaggle5.py --stage check"),

    md("## 5 — THE DAILY CELL\n\n"
       "This is the one you schedule. It is stateless: it asks Kaggle which versions have "
       "a `submission.parquet` and which are already in the submission list, then does the "
       "one right thing.\n"
       "\n"
       "| situation | what it does |\n"
       "|---|---|\n"
       "| already submitted today | nothing, prints the score |\n"
       "| a run is in flight | waits for it, then decides |\n"
       "| newest version finished and unspent | submits it, reads the score |\n"
       "| newest version already spent | pushes a new version, exits |\n"
       "\n"
       "A scheduled run does **not** block for the 9-hour run. It pushes and exits; the "
       "next day's run finds the finished version and submits it. That costs one day of "
       "latency on the first cycle only. Add `--wait-here` to block instead."
    ),
    code("!python kaggle5.py --stage daily"),

    md("## 6-10 — The same steps by hand\n\nFor watching it happen, or debugging one stage."),
    code("# 6. push: inject + push a new version. Prints the version number.\n!python kaggle5.py --stage push"),
    code("# 7. wait: poll Phase A until it finishes\n!python kaggle5.py --stage wait"),
    code("# 8. submit: BURNS ONE DAILY SLOT. Put the version number after -v.\n!python kaggle5.py --stage submit -v 0"),
    code("# 9. score: read it back, compare to 25.83\n!python kaggle5.py --stage score -v 0"),
    code("# 10. all: push -> wait -> submit -> score in one go\n!python kaggle5.py --stage all"),

    md(
        "## Scheduling it\n"
        "\n"
        "Kaggle can run a notebook on a schedule. Settings panel -> "
        "\"**Schedule a notebook to run**\" -> Frequency: **Daily**.\n"
        "\n"
        "Constraints that matter here:\n"
        "\n"
        "- **CPU notebooks only.** This one is CPU. Do not turn on the GPU or the "
        "schedule option disappears.\n"
        "- Runs fire roughly on the date in **UTC**, not at an exact time. Set "
        "`LOCAL_UTC_OFFSET_HOURS` in `kaggle5.py` if your local day differs from UTC. "
        "Freetown is UTC+0, so 0 is correct.\n"
        "- Save a version first. A schedule runs the saved notebook, not your unsaved edits.\n"
        "- You get an email for every scheduled run, pass or fail.\n"
        "\n"
        "## Changing the lever\n"
        "\n"
        "Edit `kaggle5/lever2-grid.patch` in the repo and push. The next scheduled run "
        "fetches it. Nothing here needs re-pasting.\n"
        "\n"
        "The applier hash-pins the files it patches, so a patch that does not match the "
        "bundle fails loudly instead of half-applying."
    ),
]

nb = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"},
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
assert acc["accelerator"] == "none" and acc["isGpuEnabled"] is False, "must be CPU to be schedulable"
assert acc["isInternetEnabled"] is True, "driver needs internet"
print("verify: %d cells, CPU-only, internet on (schedulable)" % len(loaded["cells"]))


def extract(name: str) -> str:
    hits = [c for c in loaded["cells"] if c.get("metadata", {}).get("kaggle5_embed") == name]
    assert len(hits) == 1, "expected exactly one embed cell for %r, got %d" % (name, len(hits))
    src = hits[0]["source"]
    return src.split("r" + Q3 + "\n", 1)[1].rsplit(Q3, 1)[0]


d = extract("kaggle5.py")
a = extract("lever2_apply.py")
pb = extract("lever2-grid.patch")
compile(d, "kaggle5.py", "exec")
compile(a, "lever2_apply.py", "exec")
assert d.strip() == driver.strip(), "embedded driver differs from source"
assert a.strip() == applier.strip(), "embedded applier differs from source"
assert pb.strip() == patch.strip(), "embedded patch differs from source"
print("verify: driver %d chars, applier %d chars, patch %d chars - all three byte-identical to source"
      % (len(d), len(a), len(pb)))
