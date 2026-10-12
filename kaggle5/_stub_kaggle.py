#!/usr/bin/env python3
"""Stub `kaggle` CLI. Lets kaggle5.py's real stage functions run offline.

Drive it with:  KAGGLE_BIN="python3 /path/_stub_kaggle.py"

Environment:
  STUB_LOG    append-only log of every invocation   (default /tmp/kaggle5_stub.log)
  STUB_STATE  json state file                       (default /tmp/kaggle5_stub_state.json)
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(os.environ.get("STUB_LOG", "/tmp/kaggle5_stub.log"))
STATE = Path(os.environ.get("STUB_STATE", "/tmp/kaggle5_stub_state.json"))

NB_HEADINGS = [
    "# About this notebook",
    "## 0. Harness patch",
    "## 1. Environment and submission mode",
    "## 2. Precaching",
    "## 3. Install the ARC runtime",
    "## 4. Import the bundled source",
    "## 5. Start serving",
    "## 6. Load the benchmark",
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
    return {"version": 4, "status": "complete", "subs": [], "pushed": None}


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
        s = state()
        s["version"] += 1
        s["status"] = "running"
        s["pushed"] = {"cells": len(nb["cells"]),
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

    if args[:2] == ["competitions", "submit"]:
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

    print("stub: unhandled args %s" % args, file=sys.stderr)
    return 2


sys.exit(main())
