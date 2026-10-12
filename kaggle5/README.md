# kaggle5

Injects the lever-2 patch into the ARC-AGI-3 notebook, pushes it, submits it,
reads the score. Can do that **once a day on a schedule**.

Lives in `mohamtur1/arc-agi-3-agent/kaggle5/`. This folder is what *initiates
submission*; the agent logic stays in `program_synthesis_agent.py` beside it.

```
kaggle5/
├── kaggle5.py             the driver. The only file that calls `kaggle`.
├── lever2-grid.patch      THE LEVER. A git patch. Edit this to change behaviour.
├── lever2_apply.py        applies it and hash-verifies. Rarely edited.
├── build_notebook.py      regenerates the notebook from the three above
├── 4CBOn2_Kaggle5.ipynb   generated. Paste into Kaggle once.
└── _stub_kaggle.py        CLI test double, for offline tests
```

The notebook is a generated artifact. Edit the sources, re-run
`build_notebook.py`, push to git. The notebook fetches all three from the repo
at run time, so a pushed edit is live with no re-pasting.

## Why the lever is a patch and not a cell

`lever2-grid.patch` is a git diff against the harness source, applied after
section 1's `git apply harness-changes.patch`. A patch carries its own context
lines, so it either applies or fails loudly. Runtime string replacement of
prompt text does not: the harness source mixes 10 real em-dashes with 9 ASCII
`--`, and one is unspaced (`predictions—for`). Matching that by hand is how
silent failures happen.

The applier hash-pins the three files it patches, so a patch that does not
match the bundle raises instead of half-applying.

## Stages

```
python kaggle5.py --stage check     # pull, verify inputs, print the map, push nothing
python kaggle5.py --stage push      # inject + push. Prints the version number.
python kaggle5.py --stage wait      # poll Phase A (up to 9 h)
python kaggle5.py --stage submit -v N
python kaggle5.py --stage score  -v N
python kaggle5.py --stage daily     # the whole cycle, one cell, idempotent
python kaggle5.py --stage all       # push -> wait -> submit -> score
```

## The daily stage

`daily` keeps no state. It asks Kaggle two questions — which versions have a
`submission.parquet`, and which are already in the submission list — and does
the one right thing:

| situation | action |
|---|---|
| already submitted today | nothing; prints the score |
| a run is in flight | waits, then decides |
| newest version finished and unspent | submits it, reads the score |
| newest version already spent | pushes a new version, exits |

It only ever submits the **newest** finished version. An earlier version of
this driver picked the highest *unsubmitted* one, which resurrected a stale v2
and burned the slot on code already moved past. Caught in testing.

A scheduled run does not block for the 9-hour run: it pushes and exits, and the
next day's run submits. One day of latency on the first cycle only. Pass
`--wait-here` to block instead.

## Scheduling

Kaggle runs notebooks on a schedule: Settings -> "Schedule a notebook to run"
-> Frequency: Daily.

- **CPU notebooks only.** The driver notebook is CPU-only for this reason. The
  notebook it *pushes* keeps `nvidiaRtxPro6000` and internet off — those come
  from the pulled metadata and are verified before every push.
- Runs fire roughly on the date in **UTC**. Set `LOCAL_UTC_OFFSET_HOURS` in
  `kaggle5.py` if your local day differs. Freetown is UTC+0.
- Save a version first; the schedule runs the saved notebook.

## Guards

`push` refuses if: any of the 5 inputs is missing from the pulled metadata,
internet is on, the heading anchor is not found, the cells before the injection
point do not define `WORKING_DIR` / `BUNDLE_DIR`, the post-injection re-read
does not find exactly two lever cells with an intact patch, or nothing changed.

`submit` refuses unless `kernels status` reports `complete`, and skips a version
already in the submission list.

`enable_gpu: false` does **not** block. The real kernel carries
`accelerator: nvidiaRtxPro6000` with `enable_gpu` false; the accelerator field
selects the hardware.

## Config

Top of `kaggle5.py`:

```
OWNER         = "mohamedmalikturay"
KERNEL_SLUG   = "arc-agi-3-milestone-2-solution-3e9adb"
COMPETITION   = "arc-prize-2026-arc-agi-3"
SUBMISSION_FILE = "submission.parquet"
BASELINE_SCORE  = 25.83
LEVER_TAG     = "lever-2"
INJECT_AFTER_HEADING  = "1. environment and submission mode"
INJECT_BEFORE_HEADING = "2. precaching"
FALLBACK_ACCELERATOR  = "nvidiaRtxPro6000"
LOCAL_UTC_OFFSET_HOURS = 0
```

The lever must land after section 1 (which runs `git apply
harness-changes.patch`) and before section 3 (which installs the runtime).
Verified against the real notebook, which puts it at cells 5-7.

## Auth

`KAGGLE_API_TOKEN` — a `KGAT_...` token from https://www.kaggle.com/settings
(API -> Create New Token). Inside a Kaggle notebook the SDK reads the token
Kaggle injects, so it usually needs nothing.

Legacy fallbacks, in the order the SDK checks: `~/.kaggle/access_token`,
`~/.kaggle/kaggle.json`, then `KAGGLE_USERNAME` + `KAGGLE_KEY`.

## Tests

```
python kaggle5.py --self-test
```

Offline, no auth, no network. Covers anchor resolution, fallback anchor,
two-cell insert, idempotent re-inject, the upstream-name guard, no-change
detection, version parsing, and both directions of the metadata guard.

To drive the real stage functions against the stub:

```
KAGGLE_BIN="python3 $PWD/_stub_kaggle.py" KAGGLE_API_TOKEN=KGAT_stub \
    python kaggle5.py --stage daily --work /tmp/w
```

`KAGGLE_BIN` overrides the CLI path (space-separated), which is how the stub is
driven without depending on an exec bit surviving a copy.

## CLI version

Targets the Kaggle image's CLI (2.0.2 was observed in the notebook). Avoids
`--wait`, `--no-run` (main-only) and `competitions submission-limits`
(added in 2.2.4). `pip install kaggle` is enough.

## After editing

```
python build_notebook.py
```

Runs `--self-test` first and refuses to build if it fails. Then verifies the
notebook re-parses, is CPU-only with internet on, and that the three embedded
fallbacks are byte-identical to their sources.
