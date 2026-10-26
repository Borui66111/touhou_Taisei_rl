# Taisei RSI automation: pilot state

## Purpose
A bounded GitHub Actions pilot verifies that the genuine pinned Taisei simulator can be built, executes a 300-frame authentic-native smoke test, runs 1,024 PPO training decisions for a skill-aware model, and evaluates that checkpoint against random and geometric-heuristic policies on the same four **development** seeds (42100-42103).

## Scope and safeguards
- Research branch: `research/rsi-automation-v1`; base snapshot: `d463a2fd`.
- Workflow: `.github/workflows/taisei-rsi-pilot.yml`.
- Entry point: `research/rsi/run_pilot.sh`; result validation: `research/rsi/summarize_pilot.py`.
- Actual Taisei native library is mandatory; the job aborts on build/load/smoke failure. No toy fallback.
- Native simulator pinned via `my_test/UPSTREAM.lock`.
- This is **NOT** autonomous parameter search, a promotion decision, or proof of RSI improvement. It establishes working automation and comparable evidence.
- Final sealed seeds **43000-43099 are never evaluated or trained on**.
- The small 4-seed evaluation is diagnostic only; it cannot support a robust improvement claim.
- No secrets, tokens, or write access required by the workflow; GitHub Action has `contents: read`.
- Artifacts retained 14 days; copy meaningful results to permanent versioned evidence if research requires retention.
- The current skill-action mask is known to exclude early discharge of an already active Power Surge; fixing that behavior and reward-scale calibration should precede *scientific* RSI model comparisons.
- Do not automatically promote any pilot model to the champion.

## Operational steps
1. In GitHub Actions, open *Taisei RSI pilot (native, fail closed)* on the research branch.
2. Inspect failures in `native_build.log`, `native_smoke.log`, `pilot_training.stdout.log` in that order.
3. Verify `summary.json` and the generated checkpoint hash before making any claims.
4. Once infrastructure passes, add controlled one-factor experiments with matched seeds, multiple independent training seeds, explicit model identity, and champion non-regression checks.
5. Keep all RSI candidate comparisons on a development branch and do not touch `main` or final seeds.

## Current state
- Prior native smoke failed at SDL3 configuration because X11/Wayland development headers were missing. The research pilot now installs those development packages, but **successful build and training remain unverified until Actions completes**.
- The current agent-side container cannot directly compile the engine; the GitHub runner is the execution backend.

## Build troubleshooting 2026-10-25

- GitHub Actions run 40122026719 reached SDL3 CMake but failed due missing XTEST development package (`libxtst-dev`). No smoke or PPO run completed.
- Targeted workflow dependency fix: commit `48493153b9dc7dc5f2d6911c31b740fa9c611160` on `research/rsi-automation-v1`; subsequent run still needs native build/smoke/train verification.
- New independent competence research branch `research/rsi-competence-v1` contains the skill-oriented design and descriptive comparison evaluator. It must not overwrite this runtime branch or reuse the sealed final seeds.

## SDL3 static/PIC native build follow-up — 2026-10-25

- Repaired XTEST dependency run 40122420549 proceeded past SDL3 CMake and failed when Meson refused non-PIC `SDL3_static` while linking `libtaisei_sim.so` (pinned wrapper forces static SDL).
- Added `research/rsi/build_native_with_shared_sdl3.sh`: builds the SDL source tag named in Taisei's wrapper as an external **shared** library in the temporary CI runner, without modifying the pinned game source checkout. `run_pilot.sh` now supplies the package path and runtime loader path.
- Run 40122713708 is the first attempt of the shared-SDL workaround. Its native smoke, training and results are **unverified** until its job actually completes.

## User-directed canonical branch policy — 2026-10-25

- **Default for ALL future Taisei RSI development:** `research/rsi-automation-v1` (the branch containing this file).
- Standing instructions are in `research/rsi/BRANCH_POLICY.md`: **do not create an experimental branch** absent a verified technical reason documented in the trajectory, with an integration plan and review where consequential.
- Prior work in `research/rsi-easy-curriculum-fix`, `research/rsi-competence-v1`, `research/rsi-meta-controller-v1` and other experiment branches is historical source/evidence, **not automatically merged** into this currently older branch. Before reusing fixes, carefully review and integrate compatible, verified changes into this selected branch without rewriting history.
- Daily autonomous RSI research and hourly milestone monitor have both been updated to target this canonical branch. Check GitHub Actions branch triggers after each workflow integration.

## 2026-10-26 — E2 prediction-based defensive movement research (pending native comparison)

- **Research hypothesis:** published no-skill PPO policies exhibit strong boundary-pushing proxies and limited responsiveness; a transparent native-observation-only risk teacher with explicit projected collision and wall penalties may produce better native survival than the current geometric heuristic. No gain has yet been measured.
- **Source:** `research/rsi/risk_aware_teacher.py`; source-only guard tests: `test_risk_aware_teacher.py`, `test_e2_teacher_compare.py`. Tests passed in workflow [40212565935](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40212565935). This is not gameplay validation.
- **Native protocol:** `research/rsi/run_e2_teacher.sh`, `research/rsi/e2_evaluate_teacher.py`, `research/rsi/e2_compare_teacher.py`, workflow `.github/workflows/rsi-e2-teacher.yml`. Genuine pinned Taisei Stage1 Easy, Reimu A, 16 development seeds 42100-42115, equal initial conditions, no neural skill head or Bomb/Special actions, fixed controller coefficients, 60-minute job budget, no auto-promotion or sealed final seeds.
- **Active gate:** [native pilot 40212525844](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40212525844) was still in progress at latest inspection. Do **not** start the E2 native comparison while another native job is active.
- **CI failure diagnosed:** previous native pilot [40211694671](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40211694671) failed resource packing because installing `zstandard` did not satisfy Taisei's `backports.zstd` import. The canonical pilot workflow now installs/tests `backports.zstd` and **only responds to an explicit `PILOT_RUN_REQUEST.md`**, avoiding native builds after ordinary research edits. The ongoing older job predates this repair and can still fail.
- **Next:** after any active native job finishes, submit `E2_TEACHER_RUN_REQUEST.md` on this same branch to launch the reproducible E2 native evaluation, check the result artifacts and append observed outcomes. If teacher is better, study controlled distillation/auxiliary policy training as a separate L2/L0 challenger. If worse, falsify/revise the predictive movement objective before PPO HPO.
- **Live replay reference:** `research/rsi/evidence/rendered_challenger_replay_20261026.md` identifies 1016 uninterrupted Left input frames, 7 Bomb uses and no clear, but does not prove actual wall contact or checkpoint identity.

### E2 native teacher execution submitted — 2026-10-26

- Explicit request committed as `research/rsi/E2_TEACHER_RUN_REQUEST.md` (`9d391c69`) on this same canonical branch; the new opt-in GitHub Actions job is [40213025937](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40213025937).
- At latest inspection, source preflight and dependency installation passed. Real native build, deterministic smoke and paired 16-seed teacher evaluation are in progress; **no native E2 outcome or RL improvement is established yet**.
- The prior unintended native pilot [40212525844](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40212525844) failed for the same missing `backports.zstd` import and has finished. Pilot trigger/dependency have been corrected in subsequent commits. No overlapping native experiment was deliberately launched.
- Follow-up: download native E2 artifacts, verify 16 paired real-game episodes, extract mean aliveness/clear/deaths/resources and limitations, append a result event; then decide whether to pursue teacher-distilled PPO or revise the planner.

## E2 verified result and E3 learning-transfer experiment — 2026-10-26

- E2 native run [40213025937](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40213025937) **succeeded**: 16 paired Stage1 Easy seeds, no Bombs or Surge activations; existing heuristic mean alive 1,659.875 vs predictive risk teacher 7,598.1875 (+5,938.3125 alive frames; paired game-seed 95% bootstrap CI [5,083.94,6,787.94]). Both 0/16 clears; teacher two 10k-frame timeouts. Teacher is **handcrafted, not a learned neural improvement**. Permanent report: research/rsi/evidence/native_e2_teacher_40213025937.json.
- Evidence suggests explicit risk-conditioned and boundary-aware movement is a promising controller strategy, but we must validate whether a 528-feature PPO actor can learn it.
- E3 prepared: research/rsi/e3_distill_native.py, e3_compare.py, run_e3_distill.sh, test_e3_distill_plan.py and the explicit-trigger workflow .github/workflows/rsi-e3-distill.yml. Pilot native **training** budget=12,000 teacher demonstrations plus 8,000 PPO rollout decisions = 20,000 per candidate; 60-minute runner; native Stage1 Easy; independent teacher demonstration seed 48030 and PPO seed 48020.
- E3 outputs are a behavior-cloned legacy 18-movement policy and its PPO-fine-tuned successor, both evaluated on 16 development seeds 42100-42115. No Bomb/Special actions, no promotion, no final sealed seeds 43000-43099.
- Explicit E3 run request is required after confirming no active native job. Future experiment should add synchronized vulnerable-threat and border traces; mean alive frames alone cannot prove actual dodging, and 10k timeouts are not clears.

## E3 completed; E4 learning reweighting — 2026-10-26

- E3 recovery native run [40219788151](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40219788151) **passed** (fresh OS process resolved prior PPO segfault). BC 12,000 real-game teacher decisions and 16 epochs resulted in 83.9% random-frame imitation accuracy; PPO 8,000 native decisions then completed. Native 16-seed BC mean alive=1,346.5, PPO=1,047.56, both 0/16 clears, mean deaths=3, zero Bombs and zero boss damage. **PPO worsened paired survival by −298.94 alive frames** (95% bootstrap evaluation-seed CI [−401.81,−234.38]). No neural champion promotion; permanent artifact report `research/rsi/evidence/native_e3_bc_ppo_40219788151.json`.
- **Observed transfer failure:** successful teacher decisions did not generalize through 528-input MLP. E3 demonstration action 0 (stay) accounts for 8,321/12,000=69.3%. Offline saved-checkpoint holdout macro recall=40.6% across 14 present action classes; nonstay accuracy=60.3%. Random-frame split shares correlated temporal context and does not establish generalization.
- **E4 new hypothesis:** class-frequency-aware imitation reduces frequent-stay dominance without more native training interactions. Fixed 16-epoch objective = 0.5 ordinary CE + 0.5 capped sqrt-inverse-frequency weighted CE on same E3 dataset/seed. An exploratory offline run improved macro class recall to ~49.2% (baseline 40.6%) but no improvement in overall frame accuracy (~83.9%). May help or harm actual game survival.
- E4 source: `research/rsi/e4_rebalanced_bc.py`, `test_e4_rebalanced_bc.py`, `run_e4_bc.sh`, `e4_eval_teacher.py`, `e4_compare.py`; opt-in CI `.github/workflows/rsi-e4-bc.yml`. Reuses immutable E3 artifact and checkpoint hashes, evaluates original BC, weighted BC and unchanged teacher on **fresh** Easy development seeds 42132..42147. Sealed 43000..43099 remain untouched, no extra native training decisions, no automatic policy promotion. Run and record before choosing E5.


## E4 native matched-seed result archived — 2026-10-26 (supersedes E4 pending above)

- **Verified native run:** [40220975924](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40220975924) completed successfully from commit `b4974034299e18703d02bf771fe40a941ee0524f`. Pinned real Taisei Stage1 Easy, 300-frame smoke and 600-frame deterministic replay passed. No new native *training* interactions: reweighted BC reused E3's immutable 12,000 native teacher decisions; all three controllers evaluated on 16 paired **fresh development** seeds 42132–42147. Archived artifact [12609314728](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40220975924/artifacts/12609314728) includes native JSONL and checkpoints (ZIP SHA256 `1dc6ab9ffe7d5ab88d30c162246d0b2c7fe66c29556fb751b54886b4d2430789`).
- **Observed outcome:** original BC 1305.625 mean alive frames; frequency-aware BC 1387.5625 (**+81.9375** paired frames; evaluation-seed bootstrap 95% interval **[-31.125, 194]**, includes zero). Both neural checkpoints: **0/16 clears, 3 mean deaths, 0 Bombs, 0 boss damage**. Unchanged handcrafted risk teacher: 7252.1875 alive frames, 0/16 clears, 2 timeouts (not clears). Offline macro recall on supported actions improved 40.56%→49.17%, but correlated-frame accuracy remained 83.92%; offline recall is not gameplay evidence.
- **Decision:** E4 did not establish reliable learned native survival or dodging improvement. No champion promotion, no new PPO/HPO or overlapping native job. A simple class-reweighting repeat has low expected marginal return while teacher–learner gap remains ~5864.6 frames. Archive: `research/rsi/evidence/native_e4_rebalanced_bc_40220975924.json`; append-only event `T-20261026-32`.
- **Highest-value next action (L2/L3, cheap):** audit E3 native demonstration **trajectory coverage** (distinct episodes/resets, temporally correlated Stay/action proportions, hazard/risk contexts) and test whether a trajectory-disjoint validation split is possible. If only ~1 complete episode is represented, preregister diverse-seed, risk-balanced *native* teacher data collection within 20k decisions before any E5 neural training. Later test temporal/risk representation and actual vulnerable-hit/clear metrics; never interpret 10k-frame truncation as a clear.
- **Evidence boundary:** the E4 result was verified from completed CI job logs and emitted summary plus artifact metadata; the artifact's per-episode JSONL was not independently re-parsed in the archival turn. The existing UKS project note is older; GitHub canonical branch and trajectory supersede stale UKS snapshots until synchronized.

## E5 native replay and demonstration diversity diagnostic — 2026-10-26

- **Canonical branch only:** `research/rsi-automation-v1`; no new research branch, main merge, force-push, or champion overwrite.
- Recovered actual E3 artifact 12609092648 and E4 artifact 12609314728. Independently re-parsed dataset/logs and all 48 E4 native episode JSONL entries. Source SHA-256 checks pass. Recovered **one completed teacher episode (7,857 decisions, seed 48030) plus one unfinished (4,143 decisions, seed 48031)**; total 69.34% Stay. Episode-disjoint statistical validation is not credible with only two seed episodes.
- Verified archival E4 paired survival delta +81.9375 alive frames (10 positive, 6 negative, n=16), 0/16 neural clears and no neural boss damage. This is NOT a reproducible learned improvement; earlier 95% bootstrap interval includes zero. Full report: `research/rsi/evidence/e5_archived_native_reaudit_20261026.json`; append-only trajectory event `T-20261026-33`.
- Added `e5_audit_replay.py`, `test_e5_audit_replay.py`, `run_e5_native_replay.sh` and opt-in `.github/workflows/rsi-e5-native-replay.yml`. Three local source-fixture tests passed; no new native gameplay or training was claimed. The existing 600-frame deterministic engine smoke does not verify learned checkpoint replay.
- **E5 native gate:** After explicit request, GitHub Actions will use only immutable E3/E4 source checkpoints and real native Taisei, stage 1 Easy, development seeds 42132 and 42133. Evaluate each learned checkpoint twice using independent OS processes; compare every action and synchronized pre/post state plus terminal digests. Max per policy 2 seeds × 2 passes × 3000 frames = **12,000 native evaluation decisions**, zero training decisions, <=60-minute job, serial native concurrency, no sealed seeds.
- **Next if replay passes:** archive pass evidence, then preregister independently seeded native teacher demonstrations and a genuinely episode-separated holdout before training. If replay fails, analyze divergence; do not promote E4 or spend extra PPO/HPO budget.

## 2026-10-26 — L4 ledger CI recovery verified; E5 native source access blocked

- Canonical writable branch remains `research/rsi-automation-v1` under `research/rsi/BRANCH_POLICY.md`. No new branches, champion updates, main merges, sealed seed access, or native training were performed.
- **Verified governance improvement:** research ledger historical entries `T-20261026-33/34` lacked `metrics` and used historical evidence tags `tests/request`; preserving immutable lines required exact-identity exceptions in the validator. All new events remain strictly checked. [CI run 40257430687](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40257430687) passed 49 source-level tests (14 trajectory plus 35 other checks), append-only verification, digest generation and artifact upload. Ledger events `T-20261026-35/36` record hypothesis, failure diagnosis, patch and verified outcome.
- **E5 remains blocked, not failed gameplay:** [native replay run 40253496010](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40253496010) recovered immutable E3/E4 artifacts and built shared SDL3 but failed `git clone` of pinned `https://github.com/touhourl/taisei-sim.git` with `fatal: could not read Username for 'https://github.com'`. GitHub connector lookup currently returns 404. Prior [E4 native run 40220975924](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40220975924) cloned the same pinned source successfully on 2026-10-25. Access/visibility change is suspected, **not confirmed**.
- **Decision/next:** No repeat native job while source access is unverified. Next cheap L4 step: add a fail-fast pinned-source availability preflight, then recover an authorized copy of the exact upstream commit or restore access; local execution is preferred where feasible. Only then repeat the capped E5 action-level native replay on development seeds 42132–42133. No gameplay competence improvement was established.


## 2026-10-26 — E3/E4 native evidence escrow verified (L4; event T-20261026-41)

- **Why:** E3 artifact 12609092648 and E4 artifact 12609314728 were set to expire on **2026-11-15 UTC**. They contained the original native teacher demonstrations, BC/PPO checkpoints, native evaluation JSONL and training/build logs. Expiry would jeopardize E5 checkpoint replay even after simulator source recovery.
- **Action:** Downloaded and independently SHA256-verified both original ZIPs (E3 `ecb660decee9706dd62c8a98023d0fe0979b578c24eb6a037c089f1014b169d1`, E4 `1dc6ab9ffe7d5ab88d30c162246d0b2c7fe66c29556fb751b54886b4d2430789`). Added a fail-closed escrow copier, three source-only regression tests and one explicitly requested non-native GitHub Actions archival workflow.
- **Verified:** [Escrow run 40304908681](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40304908681) **SUCCESS**, fast-forward commit [39247a55](https://github.com/Borui66111/touhou_Taisei_rl/commit/39247a55afc8905b4507cded7e0be5549c2d7c85) on canonical `research/rsi-automation-v1`. All **33 original evidence files** (E3=17, E4=16) and per-file SHA256 manifest are now Git-tracked at `research/rsi/evidence/escrow_e3_e4_20261026_v1/`. [Post-ledger CI 40305021546](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40305021546) passed **76 source-only tests**, append-only validation and digest.
- **Research boundary:** Archival did **not** execute Taisei, restore the locked simulator, or establish improved neural survival, dodging, boss progress, item collection or clears. Runner still reports pinned upstream `git ls-remote` exit 128. No sealed seeds, champion, main merge, new branch, paid infrastructure or native job.
- **Next:** Recover and verify the exact pinned Git source and recursive submodules. If old Actions artifacts expire, adapt E5's recovery steps to use escrowed canonical checkpoint files with their SHA256 checks; then run the already capped two-process E5 native replay on development seeds 42132–42133. No additional PPO/HPO until reproducibility is established.
