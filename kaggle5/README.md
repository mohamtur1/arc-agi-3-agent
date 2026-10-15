# kaggle5 — the between-game learning loop

The agent already learns *within* a game. This folder makes it learn *between*
games: one deterministic, knob-at-a-time experiment per day, measured, and
written into an append-only ledger that explains every decision months later.

**The machine never submits.** It pushes the GPU kernel, prints a card naming
the variant, the knob changed, and the reason — and stops. A human spends the
daily slot in the Kaggle app, exactly like the first two submissions.

Lives in `mohamtur1/arc-agi-3-agent/kaggle5/`.

```
kaggle5/
├── kaggle5.py                the driver. The only file that calls `kaggle`.
│                             Also: ledger, reconciler, planner, gym gate.
├── program_synthesis_agent_v4.py   the agent. v3 + the K1 knob channel + K2
│                             telemetry. Identical behaviour to v3 when no
│                             knob file is present (trace-verified). The
│                             source of truth is 4CBOn2/Kaggle/; this mirror
│                             is what build_notebook.py embeds, pinned by
│                             AGENT_SHA256 inside kaggle5.py.
├── k5_gym.py                 the local filter: five deterministic toy games.
│                             A smoke filter, NEVER a leaderboard score.
├── lever2-grid.patch         THE LEVER. A fixed artifact (pinned by sha256).
├── lever2_apply.py           applies it and hash-verifies all 4 hunks.
├── build_notebook.py         regenerates the notebook from the five above
├── 4CBOn2_Kaggle5.ipynb      generated. Paste into Kaggle once.
└── _stub_kaggle.py           CLI test double, for offline tests
```

The notebook is a generated artifact. Edit the sources, run
`build_notebook.py`, push to git. At run time the notebook re-fetches every
file from the repos (with loud per-file hashes) and only falls back to its
embedded byte-verified copies on failure — loudly.

## A day in the loop

1. **pull the ledger** — `datasets download` of `mohamedmalikturay/kaggle5-ledger`
   (auto-created on first use, private). Local copy + remote are line-merged;
   append-only, so merging is a union.
2. **reconcile** — read yesterday's submission(s) back: aggregate from
   `competitions submissions`, per-game rows from `kernels output <v>` (the
   v4 agent writes `k5_per_game.csv` + a self-report during the rerun).
   Matched to the pending row by the message the human pastes
   (`kaggle5 <variant> v<N>`), falling back to FIFO-by-date. Absent, zero,
   and errored submissions are all recorded as data. Interactive-run DUMMY
   artifacts (single hardcoded row) are labelled `dummy`, never folded in.
3. **plan** — fold the ledger, walk the fixed queue (below), one knob at a
   time, with the documented backtrack rule. The reason is written into the
   pending row.
4. **filter** — run `k5_gym.py`: incumbent vs candidate on five toy games.
   Only the obviously-broken fail (crash / hang / lost toy win / 2+ toy
   levels). Filter counters are stored in the ledger under `local_filter`,
   permanently labelled NOT a score.
5. **pending row** — written and synced BEFORE the push, so a crash can
   never lose what today was supposed to measure.
6. **push + stop** — inject the lever cells plus the `k5_knobs.json` writer
   cell (the K1 channel), push, print the human card. Exit.

Interrupted anywhere, the next run resumes from the ledger. `daily` is
idempotent within a day: an unanswered pending row means "print the card
state and push nothing".

### Your three taps after a push notification

1. open https://www.kaggle.com/code/mohamedmalikturay/arc-agi-3-milestone-2-solution-3e9adb
2. wait for the pushed version to finish, then `...` → **Submit to
   arc-prize-2026-arc-agi-3** → pick the pushed version
3. paste the message the card prints (`kaggle5 <variant> v<N>`) — that's how
   the next run matches the score to the experiment with certainty (FIFO by
   date is the fallback)

Skip a day: the driver waits ~2 days, tombstones the row `stale` (never
silently drops it), and moves on. Skip deliberately: `python kaggle5.py
--abandon`.

## The one-time step (do this once, in the app)

The knob channel only exists in v4. **Paste `program_synthesis_agent_v4.py`
into the fork**, replacing the v3 agent, and save a version. Until you do,
`check` will report `K1 channel: NO`, and any knob-changing push is REFUSED
(inert variants would burn a day for zero information — the refusal text says
this too). v4 with no knob file is behaviour-identical to v3, so this swap
changes nothing until the first experiment arrives.

Also upload `program_synthesis_agent_v4.py` to `4CBOn2/Kaggle/` — that repo
is the source of truth; the mirror here is pinned (`AGENT_SHA256`).

## The ledger

Two append-only JSON-lines files, synced to the private dataset
`mohamedmalikturay/kaggle5-ledger` at the start and end of every run and once
more pre-push:

- `experiments.jsonl` — experiment rows (`kind: experiment`) carry date,
  variant, `changed: {knob: [old, new]}`, the FULL 10-knob state, the agent
  file name + sha256, and the one-line reason. Life events are separate
  appended rows: `pushed`, `score` (with `ref`, `incumbent_score_before`,
  `verdict`, `adopted`), `gym` (filter), `stale`, `abandon`, `import`.
- `game_rows.jsonl` — the per-game telemetry rows for scored experiments,
  with provenance (`source: kernel_output`, `version`) and the `dummy` label.

Nothing is ever rewritten; fold logic derives the current picture. If the
dataset path fails, the driver tries a GitHub commit (only when
`K5_GITHUB_TOKEN`/`GITHUB_TOKEN` was deliberately set) and otherwise prints a
loud rescue block. It never pretends a sync succeeded.

Seeded once into an empty ledger: `v1-imported 0.06`, `v2-imported 0.07`
(hashes unrecorded at the time — which is why the ledger exists).

## The experiment queue

Fixed order, one knob at a time. Ladder entries are tried in order;
EQUAL-OR-BETTER adopts the value and continues the ladder; WORSE reverts and
kills the rest of that knob's ladder (knob declared settled). Ranges were
chosen by reading the code that consumes each knob:

| # | knob | ladder | why (code-grounded) |
|---|---|---|---|
| 1 | EXPLORATION_CAP | 24, 6 | up first: more graph novelty before the bounded rotation (`_no_signal_explore`) |
| 2 | GRAPH_HASH_STRIDE | 2, 8 | down first: 32x32 vs 16x16 hash lattice, finer state identity |
| 3 | PROBE_ATTEMPT_CAP | 2, 5 | down first: every probe is a burned action in move games |
| 4 | GRAPH_RECENT_CAP | 1 | the on/off A/B: `is_revert` reads `recent[-2]` EXCLUSIVELY, so 6/12/24 are provably identical — the only real alternative is OFF (cap 1) |
| 5 | REPLAY_ABORT_STALE | 3, 1 | up first: trust stored wins one abort longer |
| 6 | SEQ_STORE_CAP | 80, 40 | up first: 80 == MAX_ACTIONS, so no win is truncated from replay |
| 7 | BLACKLIST_CAP | 128, 32 | up first: eviction is an unordered `set.pop()`; 128 mostly ends thrash |
| 8 | GLOBAL_CHANGE_FRACTION | 0.15, 0.05 | down only: `n > FRAC*4096 or n > 1024` clamps the fraction term at 0.25, so upward sweeps are provably inert and were removed |
| 9 | CLICK_SELECTORS | swap probe | one adjacent swap of the hypothesis priority order |

`GRID`/`CELL` are structural and permanently off the list. When the queue is
exhausted the driver prints the incumbent and pushes nothing, ever — the loop
stops by design rather than inventing experiments.

## Honest expectations (the ceiling)

This system measures; it does not compound. Knob experiments are expected to
find small deltas in the 0.06–0.07 regime, and some will be worse — a worse
number is still one clean data point, because the agent is deterministic. The
0.06 → 0.07 step came from a hand-written architectural change, not from
sweeping. What you are buying here is one trustworthy measurement per day, a
per-game map of what each knob touches, and a ledger that explains itself.
The same machinery can later drive targeted v5 architectural experiments —
the knob JSON is just the first payload — but nobody should expect daily
compounding from constants.

## Guards

`push` refuses when: any of the 5 inputs is missing from the pulled metadata;
internet is on; the heading anchor is missing; `WORKING_DIR`/`BUNDLE_DIR` are
not defined above the injection point; the post-injection re-read fails
(patch + applier [+ knobs] cells, payload round-trips); the notebook was
already identical (no `--allow-same`); the agent file's sha256 differs from
the `AGENT_SHA256` pin; `DEFAULT_KNOBS` drifted from the agent module; or a
knob experiment would be inert (no K1 channel in the fork) without
`--allow-inert`. Every run starts by probing every CLI subcommand + flag the
driver relies on against the installed CLI (bug-D rule: no inherited flag
assumptions).

`check` additionally prints the pulled notebook map, whether the K1 channel
is present, and a best-effort `submission-limits` probe (informational only).

## Stages

```
python kaggle5.py --stage check    # free dry run after probes
python kaggle5.py --stage plan     # reconcile + print today's plan (read-only)
python kaggle5.py --stage gym --set EXPLORATION_CAP=20   # ad-hoc toy filter
python kaggle5.py --stage push [--set KNOB=JSON]         # manual push
python kaggle5.py --stage score    # reconcile + experiment table
python kaggle5.py --stage daily    # the scheduled loop
python kaggle5.py --abandon        # tombstone the pending row
python kaggle5.py --self-test      # everything below, offline
python kaggle5.py --stage submit   # refuses loudly, exit 1 (as does `all`)
```

## Auth

`KAGGLE_API_TOKEN`, then `KAGGLE_API_V1_TOKEN` (what Kaggle injects; the
notebook maps it into the modern name), then `~/.kaggle/access_token[.txt]` /
`~/.kaggle/kaggle.json`, then `KAGGLE_USERNAME`+`KAGGLE_KEY`, then a
TTY-guarded prompt (interactive only), and finally Kaggle's ambient notebook
auth. No path may abort a scheduled run (bug-A rule).

## Tests

```
python kaggle5.py --self-test
```

Offline, no auth, no network. Covers: injection surgery (lever-only and
lever+knobs, idempotent replace); the ledger (append-only, fold, full-knob +
hash reproducibility of pending rows); reconcile (message match, FIFO
fallback, idempotency, zero and error recorded as data, stale tombstones,
dummy labelling); the planner (anchor first, queue order, worse-kills-ladder
backtrack, exhaustion); the guards and the submit/all refusals; and a full
multi-day simulation against `_stub_kaggle.py` (anchor → exp-001 → worse →
backtrack → same-day hold), which also asserts **zero** `competitions submit`
calls across the whole simulated week and that per-game rows landed.

To drive stage functions against the stub by hand:

```
KAGGLE_BIN="python3 $PWD/_stub_kaggle.py" K5_TODAY=2026-10-15 \
    python kaggle5.py --stage daily --work /tmp/w
```

`KAGGLE_BIN` overrides the CLI path; `K5_TODAY` overrides the local date
(testing hook, also documented in the driver).

## CLI version

Verified live against Kaggle CLI **2.2.4** (an earlier note that claimed
2.0.2 was wrong): `kernels pull/push/status/files/output`,
`competitions submissions (--format json --page-size)`,
`competitions submission-limits (--json)` (it exists), and
`datasets download/create/version` (`-t` so .jsonl files are not "converted";
create is private by default; metadata file is `datasets-metadata.json`).
The probe re-verifies all of this every run, so a future CLI bump fails
loudly at the flag that changed instead of silently misreading output.

## After editing

```
python build_notebook.py
```

Runs `--self-test` first and refuses to build if it fails; pins the lever
patch by sha256; mirrors + pins the agent (`K5_AGENT_SRC=/path` for offline
builds); then verifies the notebook re-parses, is CPU-only with internet on,
and that all five embedded fallbacks are byte-identical to their sources.
Bug-A and bug-B regression assertions are part of the verify step.
