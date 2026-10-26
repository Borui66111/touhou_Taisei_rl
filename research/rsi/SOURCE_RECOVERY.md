# E5 exact-source recovery: offline, fail-closed path

**Canonical writable branch:** `research/rsi-automation-v1` only. **Status (2026-10-26):** source recovery NOT yet achieved.

## Why this exists

The E5 native runner could compile authentic Taisei in earlier runs, but the pinned upstream `https://github.com/touhourl/taisei-sim.git` now fails unauthenticated access (Git exit 128; GitHub API 404). The exact lock is `my_test/UPSTREAM.lock`, commit `789200ff255c681b3ed4230a3f26f0b8c12c78fe`. The HEAD-access preflight intentionally blocks CI before expensive setup. Changing the simulator to a different revision would invalidate the matched-seed experiment.

A previously saved **complete Git checkout** is a legitimate recovery source if and only if it contains the exact commit, its original origin identity, clean tracked/untracked state, and all recursive submodules at their recorded revisions. A ZIP of working-tree files without Git object history is insufficient.

## Safe local verification (no network and no gameplay)

From the canonical repository root, with an independently recovered checkout at `/path/to/taisei-sim-pinned`:

```bash
python research/rsi/verify_pinned_local_checkout.py \
  --lock my_test/UPSTREAM.lock \
  --checkout /path/to/taisei-sim-pinned \
  --out research_evidence/local_source_verification.json
```

The command does **not** fetch, clone, modify, build, or execute the game. Exit 0 means only that the exact pinned Git source identity and recursive submodule states were verified locally; exit 2 means blocked. The report always sets `native_game_executed=false`. The checker is stricter than the remote-HEAD preflight: it checks the actual locked commit and submodules, not mere URL reachability.

If the source is later placed in `my_test/upstream/taisei-sim-pinned`, re-run verification there before using the existing `my_test/scripts/build_taisei_sim.sh`. That build script has independent commit/origin/submodule checks and must remain the final authority. Do not disable its checks, overwrite an existing checkout, or run E5 before the full native smoke and model-hash gates succeed.

**Important:** GitHub-hosted runners do not automatically inherit a researcher's local checkout. This change creates a verified local recovery path, **not** a new GitHub Actions source mirror, authenticated access, or a completed E5 run. Any future artifact transport or mirror must preserve the Git object identities and be reviewed before enabling the native workflow.

## Research decision

- **Hypothesis:** A complete existing pinned Git checkout can restore native source availability without weakening source identity or switching simulator revisions.
- **Action:** Implemented offline verifier and fail-closed source-only regression tests.
- **Observation:** Tests exercise exact identity, bad lock, bad origin, bad HEAD, dirty checkout, missing/mismatched recursive submodules, missing manifest, and non-Git directory. No actual recovered pinned checkout has been supplied or verified.
- **Interpretation:** A safe recovery route is available for inspection; actual source access and native gameplay remain blocked.
- **Decision:** Do not run native E5 or train a candidate until the exact checkout and submodules are verified. No champion promotion, sealed seeds, new branches, or paid infrastructure.
- **Next:** Obtain an authorized saved checkout; run this checker; then execute one serialized E5 checkpoint replay (development seeds 42132–42133, <=12,000 evaluation decisions per policy, <=60 minutes) and inspect full action/state traces.
