# kaggle5 — the between-game learning loop (v5: harness epoch)

This folder drives the between-game learning loop for the Kaggle competition
**ARC Prize 2026 — ARC-AGI-3** (`arc-prize-2026-arc-agi-3`). A scheduled CPU
Kaggle notebook ("K5") drives the GPU competition kernel purely through the
Kaggle CLI, running one deterministic one-knob experiment per day, recorded
in an append-only ledger.

**THE MACHINE NEVER SUBMITS.** It pushes a kernel version, prints a "human card"
(variant, knob changed, reason, version number, three taps), and stops. The
user spends the daily slot in the Kaggle app and pastes the message
`kaggle5 <variant> v<N>` so tomorrow's reconcile matches the score to the
experiment with certainty (FIFO-by-date is the fallback). This rule is absolute
and is itself self-tested.

Lives in `mohamtur1/arc-agi-3-agent/kaggle5/`.

```
kaggle5/
├── kaggle5.py                    the driver. The only file that calls `kaggle`.
│                                 Also: ledger, reconciler, planner, static gate.
├── program_synthesis_agent_v4.py FROZEN lineage: pinned (AGENT_SHA256) mirror
│                                 of the v4 agent from 4CBOn2/Kaggle/. Delivered
│                                 and pinned but NEVER wired into the live kernel.
├── k5_gym.py                     FROZEN lineage: local toy filter gym. Never
│                                 runs in v5 (harness has no local proxy).
├── lever2-grid.patch             THE LEVER. Pinned fixed artifact (sha256).
├── lever2_apply.py               applier (applies and hash-verifies all 4 hunks).
├── build_notebook.py             notebook generator (preflight self-test gates build).
├── 4CBOn2_Kaggle5.ipynb          generated deliverable to import into Kaggle.
└── _stub_kaggle.py               Kaggle CLI test double for offline tests.
```

## The Architecture Decision

The device k5 serves is the **Franzen LLM-harness kernel**:
`mohamedmalikturay/arc-agi-3-milestone-2-solution-3e9adb`
(dfranzen's benchmark fork + the lever layer).

The older MyAgent program-synthesis lineage is **FROZEN** as published work
in the repos — files `program_synthesis_agent_v4.py` (pinned `AGENT_SHA256`
`8f447e454134d4a55dd0dfe3227c6273fa70c595f24773a0c6c84e1ce7b440a0`), `k5_gym.py`,
and the Kaggle4 notebooks. They are delivered and pinned but **NEVER** wired into
the live kernel; the toy gym never runs in v5. The earlier "paste v4 agent
into the fork" step is a documented DEAD END — do not resurrect it (the live
kernel is an LLM tool-agent; a pasted v4 cell would execute nothing while
unlocking inert knob pushes — exactly what the design protects against).

## Knob Surface & Baselines

Exactly four `bm.solver` attributes (baselines are the submission-effective values):

| knob | type | baseline | role |
|---|---|---|---|
| `analyzer_timeout` | float+ | 900.0 | Per-game timeout for code analysis |
| `max_runtime_s_per_game` | float+ | 31920.0 | Notebook runtime budget per game (532*60) |
| `max_actions_per_game` | int\|null | None | Cap on total agent action steps |
| `concurrency` | int+ | 120 | Parallel solver workers |

Day 1 is the **ANCHOR**: full surface at baselines, pushed and measured, so the
ledger binds to this kernel's own bytes (variant name `franzen-base`, device
`franzen-harness-v1`).

## The Fixed Experiment Queue

One unmeasured step per knob, respecting the static gate's `|log2(new/old)| <= 1`
clamp:

1. `analyzer_timeout`: `900.0 -> 1200.0` (up: 33% more time for analyzer)
2. `max_actions_per_game`: `None -> 500` (capping the uncapped side)
3. `max_runtime_s_per_game`: `31920.0 -> 31320.0` (~2% trim, 10m buffer)
4. `concurrency`: `120 -> 60` (down: reduce contention / memory pressure)

Backtrack rule: WORSE reverts and kills that knob's ladder; EQUAL adopts and continues.

## Knob Channel Topology & Sanitizer

- **Topology**: The driver injects ONE self-contained knob cell immediately
  after the kernel's `## 7. Customization hook` markdown heading (after the hook's
  code cell, before the next heading). The cell carries its own surface map in
  the payload (format 2), applies assignments allowlist-only, loudly refuses
  unknown names / uncaps / bools, dies if the upstream harness renamed an
  attribute, prints `[k5] bm.solver.x: old -> new` per knob, and writes
  `/kaggle/working/k5_knobs.json`.
- **Push refusal predicate**: `has_knob_channel = "K5_KNOBS_JSON" marker present OR hook heading found`.
  If neither, knob pushes die with the fix explained.
- **Sanitizer**: Runs at every push (loud): removes hand lever-2 prototype cells
  (inline monkeypatch edits) and empty output-less code cells; the canonical
  hash-verified lever pair replaces them. Lever pair = `%%writefile lever2-grid.patch`
  cell + applier cell (`lever2_apply.py` applies and hash-verifies all 4 hunks;
  patch pinned sha256 `143ae716d99374cccbe6f25b607c3f6bf6e2eb9fa50299b9a05ff5e3cfd09b69`).

## Static Gate

The static gate is **RULES ONLY**:
- Exactly one knob changed
- Knob in surface (`KNOB_SURFACE`)
- No booleans
- Never uncap (`None -> value` allowed, `value -> None` refused)
- `|log2(new/old)| <= 1` for positive numerics
- Canon-equal refused

**THE HARNESS HAS NO LOCAL PROXY** (wallet-budget LLM solver, 9h GPU run) — the
frozen toy gym said nothing about it. Filter results are never presented as
predictions or scores.

## A Day in the Loop

1. **pull the ledger** — `datasets download` of private dataset `mohamedmalikturay/kaggle5-ledger`
   (line-merged union).
2. **reconcile** — match yesterday's submission (message match `kaggle5 <variant> v<N>`,
   FIFO fallback; zero/error/absent all recorded as data; dummy rows labelled).
3. **plan** — fixed queue + documented backtrack.
4. **static gate** — rules check.
5. **pending row** — written and synced BEFORE the push (durable).
6. **push** — push the GPU kernel, print the human card, **and stop**.

## Your Three Taps After a Push

1. Open https://www.kaggle.com/code/mohamedmalikturay/arc-agi-3-milestone-2-solution-3e9adb
2. Wait for the pushed version to finish running, then `...` → **Submit to
   arc-prize-2026-arc-agi-3** → pick the pushed version.
3. Paste the message printed on the card: `kaggle5 <variant> v<N>`.

Skip a day: waits ~2 days, marks `stale`, moves on.
Deliberate skip: `python kaggle5.py --abandon`.

## Ledger

Append-only `experiments.jsonl` + `game_rows.jsonl`. Pending rows carry full
surface state + device (`franzen-harness-v1`). If dataset path fails, tries
GitHub commit (only with deliberately set `K5_GITHUB_TOKEN`) else prints loud
rescue block; never pretends a sync succeeded.
Seeded history: `v1-imported 0.06`, `v2-imported 0.07`.

## Self-Test

```bash
python3 kaggle5.py --self-test
```
Offline, no auth, no network. Exercises injection surgery, hook placement,
payload AST round-trip, idempotent replace, sanitizer, refusal on missing hook,
ledger, reconcile, planner, static gate edge cases, guards, and multi-day
simulation against `_stub_kaggle.py` asserting **zero** `competitions submit` calls.
