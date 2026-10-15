#!/usr/bin/env python3
"""Stub `kaggle` CLI. Lets kaggle5.py's real stage functions run offline.

Drive it with:  KAGGLE_BIN="python3 /path/_stub_kaggle.py"

Environment:
  STUB_LOG    append-only log of every invocation   (default /tmp/kaggle5_stub.log)
  STUB_STATE  json state file                       (default /tmp/kaggle5_stub_state.json)
  STUB_DATA   dir backing the datasets endpoints    (default /tmp/kaggle5_stub_data)

The test harness drives time and outcomes by editing the state file between
runs (stub versions of "the human submitted in the app" and "the rerun
produced telemetry"): keys "subs" and "k5_outputs".
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(os.environ.get("STUB_LOG", "/tmp/kaggle5_stub.log"))
STATE = Path(os.environ.get("STUB_STATE", "/tmp/kaggle5_stub_state.json"))
DATA = Path(os.environ.get("STUB_DATA", "/tmp/kaggle5_stub_data"))

NB_HEADINGS = [
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

META = {
    "id": "mohamedmalikturay/arc-agi-3-milestone-2-solution-3e9adb",
    "id_no": 13999001,
    "title": "ARC AGI 3 Milestone 2 Solution 3e9adb",
    "code_file": "arc-agi-3-milestone-2-solution-3e9adb.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": False,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": [],
    "dataset_sources": ["dfranzen/pennyroyal-v253", "dfranzen/taaf-kaggle-source-bundle-copy"],
    "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": ["dfranzen/intel-qwen3.8-flash-next-w4a16-autoround",
                      "dfranzen/albucino-qwen3-8-flash-next-drafter"],
    "docker_image": "",
    "machine_shape": "nvidiaRtxPro6000",
}


def rec(*a):
    with LOG.open("a") as f:
        f.write(" ".join(a) + "\n")


def state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"version": 4, "status": "complete", "subs": [], "pushed": None,
            "k5_outputs": {}, "submission_limits": True}


def save(d):
    STATE.write_text(json.dumps(d, indent=1))


def notebook():
    cells = [{"cell_type": "markdown", "metadata": {}, "source": NB_HEADINGS[0] + "\n\nprose"}]
    for h in NB_HEADINGS[1:]:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": h + "\n\nprose"})
        if h.startswith("## 0."):
            body = "%%writefile /kaggle/harness-changes.patch\ndiff --git a/x b/x\n"
        elif h.startswith("## 1."):
            body = ("import subprocess\n"
                    "WORKING_DIR = Path('/kaggle/working')\n"
                    "BUNDLE_DIR = Path('/kaggle/taaf-kaggle-source-share')\n"
                    "subprocess.run(['git', 'apply', '/kaggle/harness-changes.patch'], check=True)\n")
        elif h.startswith("## 4."):
            # stands in for the bundled v4 agent: carries the K1 marker so the
            # driver's knob-channel scan sees what a v4-bearing fork shows.
            body = ("# the bundled agent reads overrides from K5_KNOBS_JSON (K1)\n"
                    "print('import bundled source')\n")
        elif h.startswith("## 7."):
            body = ("# Make one-off changes to bm, bm.games, or bm.solver here before the run starts.\n"
                    "class _Solver:\n"
                    "    analyzer_timeout = 900.0\n"
                    "    max_runtime_s_per_game = 31920.0\n"
                    "    max_actions_per_game = None\n"
                    "    concurrency = 120\n"
                    "class _BM:\n"
                    "    solver = _Solver()\n"
                    "bm = _BM()\n")
        else:
            body = "print(%r)\n" % h
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": body})
    return {"cells": cells,
            "metadata": {"kaggle": {"accelerator": "nvidiaRtxPro6000",
                                    "isInternetEnabled": False, "isGpuEnabled": False}},
            "nbformat": 4, "nbformat_minor": 4}


def main():
    args = sys.argv[1:]
    rec(*args)

    if args[:2] == ["kernels", "pull"]:
        p = Path(args[args.index("-p") + 1])
        p.mkdir(parents=True, exist_ok=True)
        (p / META["code_file"]).write_text(json.dumps(notebook(), indent=1))
        (p / "kernel-metadata.json").write_text(json.dumps(META, indent=2))
        print("Source code and metadata downloaded to %s" % p)
        return 0

    if args[:2] == ["kernels", "push"]:
        p = Path(args[args.index("-p") + 1])
        nb = json.loads((p / META["code_file"]).read_text())
        texts = []
        for c in nb["cells"]:
            s = c.get("source", "")
            texts.append(s if isinstance(s, str) else "".join(s))
        s = state()
        s["version"] += 1
        s["status"] = "running"
        s["pushed"] = {"cells": len(nb["cells"]),
                       "has_knob_payload": any("_K5 = {" in t or "_K5 =" in t for t in texts),
                       "sources": json.loads((p / "kernel-metadata.json").read_text())}
        save(s)
        print("Kernel version %d successfully pushed.  Please check progress at https://www.kaggle.com/code/x"
              % s["version"])
        return 0

    if args[:2] == ["kernels", "status"]:
        print('%s has status "%s"' % (args[2], state()["status"]))
        return 0

    if args[:2] == ["kernels", "files"]:
        spec = args[2]                       # owner/slug or owner/slug/version
        s = state()
        parts = spec.split("/")
        if len(parts) == 3:
            try:
                v = int(parts[2])
            except ValueError:
                print("invalid version", file=sys.stderr)
                return 2
            if v > s["version"]:
                print("404 - Not Found", file=sys.stderr)
                return 1
        print(json.dumps([{"name": "submission.parquet", "size": 4096,
                           "creationDate": "2026-10-12T09:00:00Z"}]))
        return 0

    if args[:2] == ["kernels", "output"]:
        spec = args[2]
        p = Path(args[args.index("-p") + 1]) if "-p" in args else Path(".")
        p.mkdir(parents=True, exist_ok=True)
        s = state()
        parts = spec.split("/")
        v = parts[2] if len(parts) == 3 else None
        files = (s.get("k5_outputs") or {}).get(str(v or ""))
        if not files:
            print("No output files found for %s" % spec, file=sys.stderr)
            return 1
        for name, content in files.items():
            (p / name).write_text(content)
        print("Downloaded %d file(s) to %s" % (len(files), p))
        return 0

    if args[:2] == ["competitions", "submit"]:
        # still modelled so the never-submit test can prove the driver
        # never calls it: every invocation is logged above.
        s = state()
        v = args[args.index("-v") + 1]
        s["subs"].insert(0, {
            "ref": 99000000 + len(s["subs"]),
            "fileName": args[args.index("-f") + 1],
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "description": args[args.index("-m") + 1],
            "status": "complete",
            "publicScore": 27.41,
            "privateScore": "",
        })
        s["submitted_version"] = v
        save(s)
        print("Submission ref: %d" % s["subs"][0]["ref"])
        print("0 submissions remaining today.")
        return 0

    if args[:2] == ["competitions", "submissions"]:
        print(json.dumps(state()["subs"]))
        return 0

    if args[:2] == ["competitions", "submission-limits"]:
        if not state().get("submission_limits", True):
            print("404 - not available for this competition", file=sys.stderr)
            return 1
        print(json.dumps({"maxDailySubmissions": 1,
                          "submissionsUsedToday": len(state()["subs"]),
                          "note": "stub-shaped payload; the driver prints it "
                                  "verbatim and never parses it"}))
        return 0

    if args[:2] == ["datasets", "download"]:
        # -d owner/name ... -p dir --unzip
        store = DATA / "kaggle5-ledger"
        p = Path(args[args.index("-p") + 1]) if "-p" in args else Path(".")
        if not store.is_dir():
            print("404 - dataset not found", file=sys.stderr)
            return 1
        p.mkdir(parents=True, exist_ok=True)
        for f in store.iterdir():
            if f.is_file():
                shutil.copy2(f, p / f.name)
        print("Downloaded dataset to %s" % p)
        return 0

    if args[:2] in (["datasets", "create"], ["datasets", "version"]):
        p = Path(args[args.index("-p") + 1])
        store = DATA / "kaggle5-ledger"
        store.mkdir(parents=True, exist_ok=True)
        for f in p.iterdir():
            if f.is_file():
                shutil.copy2(f, store / f.name)
        sub = args[1]
        print("Your private Dataset is being created"
              if sub == "create" else "Dataset version is being created.")
        return 0

    print("stub: unhandled args %s" % args, file=sys.stderr)
    return 2


sys.exit(main())
