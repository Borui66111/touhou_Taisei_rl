# Alice RSI Research OS — recursive strategy, experiment, and policy improvement

**Version:** 0.1 (2026-10-25, Asia/Singapore) · **Status:** active protocol; native learning not yet validated.
**Scope:** authentic Taisei, repository `Borui66111/touhou_Taisei_rl`; canonical writable branch `research/rsi-automation-v1` (see `research/rsi/BRANCH_POLICY.md`). Older branches are read-only evidence sources.
**Canonical durable trajectory:** `research/rsi/trajectory/events.jsonl`; machine-validated, append-only in intention, immutable event IDs and source links. Every entry distinguishes observed results from proposals and unverified reports.

## 1. Mission, nonnegotiables, and what counts as progress

Mission: learn useful *gameplay competence* — dodging while vulnerable, death-free spell/stage clears, pickup of actual items, boss progress without destructive resource spam — through measurable experiments on genuine native Taisei. No toy simulator may substitute for native-game evidence. Retain sealed final evaluation seeds `43000–43099` until a frozen final design. No policy may replace the champion without evidence, clear lineage, and an approval gate.

Progress has **two independent axes**:
1. **Execution/infrastructure progress**: source synchronization, dependencies, native build, deterministic replay, training readiness, trace validity.
2. **Scientific/behavioral progress**: genuine win rate, first-hit time, risk-conditioned dodge competence, pickups, boss progress, resource economy, cross-stage generalization.

A successful CI/unit test is not evidence of learned gameplay; a trajectory event can be useful even if it records a failed hypothesis.

## 2. Nested recursion — what can be revised?

| Loop | Optimized object | Typical intervention | Measurement/exit trigger |
| --- | --- | --- | --- |
| **L0: policy learning** | Neural weights `theta` | PPO gradients from genuine game transitions | learning curve vs matched baseline |
| **L1: training/search** | optimizer, schedules, seeds, curriculum budget, exploration | LR/entropy/HPO, rollout lengths, phase balance | controlled multi-seed outcome with equal budget |
| **L2: agent strategy** | state representation, action semantics, rewards, architecture, task formulation | items/lasers, threat prediction, skill discharge, recurrence, hierarchical action | changed *capability*, ablation + native behavioral evidence |
| **L3: scientific strategy** | which hypotheses/questions and experiments are run | falsify reward hypothesis; choose diagnostics before training; pivot from PPO if needed | information gained per unit cost, hypothesis elimination |
| **L4: RSI governance** | failure routing, stopping/pause/review gates, selection policy | change promotion rules, allocate budgets, inspect trajectory/meta-failures | trend in validated improvements and scientific waste |

These loops interact; they are **not** a forced pipeline where tuning L1 must be exhausted before thinking at L2 or L3. Recurrent lack of learning, repeated implementation failure or non-comparable evidence can trigger a higher-level pivot. A major strategy change may need a small factorial test, rather than falsely attributing improvement to one variable.

## 3. Every experiment has a full scientific trajectory

Minimum record fields: `event_id`, `timestamp_utc`, `project`, `level`, `kind`, `status`, `parent_ids`, `hypothesis`, `intervention`, `expectation`, `observation`, `interpretation`, `evidence`, `limits`, `decision`, `next_action`, `cost`.

Event kinds: `hypothesis`, `experiment_plan`, `execution`, `measurement`, `diagnosis`, `decision`, `pivot`, `human_review`, `checkpoint`. Event statuses: `proposed`, `running`, `succeeded`, `failed`, `blocked`, `superseded`, `review_required`. **Historical snapshots are never edited to retrospectively claim success.** Corrections are appended as new events that reference the earlier IDs.

For experiments, preserve:
- **Provenance**: exact code and pinned native simulator commits, native resource version, policy/checkpoint SHA-256, environment/observation/action/reward versions, stage, difficulty, seeds, RNG, training steps, eval mode, per-episode JSONL/trace/replay/checkpoint.
- **Pre-registration**: initial idea and causal hypothesis, baseline/control, independent variable, expected change, alternate outcomes that would falsify the hypothesis, budget, seed partition and comparison rules.
- **Observed**: actual execution outcomes, sample sizes, failure signatures, confidence intervals/uncertainty; never fill missing fields with guessed values.
- **Interpretation**: distinguish whether result supports hypothesis, falsifies it, or remains indeterminate due to invalid environment or low N.
- **Next action and dependency**: one concrete follow-up and review triggers, not an unbounded agent command.

Git commits, workflow run pages, archived JSONL and artifacts are **evidence references**, not replacements for these records. GitHub Actions artifacts expire; meaningful reports/checkpoint manifests need durable versioned storage.

## 4. Progress landscape and marginal returns

After a batch, compute a capability vector (with uncertainty and provenance):

```
M = {
   "death_free_clear_rate": ..., "first_hit_frame": ...,
   "vulnerable_risk_survival": ..., "confirmed_pickups": ...,
   "boss_phase_clear": ..., "bombs_per_clear": ...,
   "replay_action_stability": ...,
}
```

Unavailable metrics are `null`, never zero. Report **best validated champion**, **new candidate**, **improvement distribution**, **compute/time cost**, **confidence**, and **unresolved failures**. Define marginal efficiency only where common evaluation exists, e.g.
`efficiency = paired_improvement_in_death_free_clear_probability / incremental_GPU_hours`.
Do not divide by zero or manufacture a statistic from four pilot episodes; otherwise use a qualitative ranking plus an uncertainty label.

Research continuation recommendation:
```
utility(next_test) = estimated probability of resolving a high-impact uncertainty
                    * expected benefit if resolved
                    / estimated compute + engineering + risk cost
```
This is a *decision heuristic*, not calibrated Bayesian expected utility. Compare plausible alternative hypotheses and the value of a cheap diagnostic against another long training run. Track failed experiments as informative when they eliminate plausible explanations.

## 5. Never stop learning ≠ never stop a run

**Default is continuation of the research agenda, not infinite unbounded compute.**
After every event with a verified outcome, queue the *next justified action*:
- **Build/ABI/physics invalid** → L3 diagnostic, small targeted fix, rerun smoke only, log blocker.
- **Bad/missing evidence or unmatched seeds** → instrumentation/manifest repair before interpreting efficacy.
- **Trained but fails competence** → inspect trajectories; test a causal strategy hypothesis (L2) or reward ablation.
- **Progress plateaus despite valid training** → investigate distribution shift/algorithm limits, escalate L1→L2→L3.
- **Meaningful reproducible gain** → cross-seed check, more difficult authentic curriculum, then next challenger.
- **All currently bounded work exhausted** → write a reasoned backlog and explicit human-review request; mark `review_required`, not `complete`.

**Human interventions requested only when necessary:** spending/compute limit exceeded; strategy redefines the objective; unsafe/irreversible repo edits; changing final-test protocol; access/credentials needed; conflicting evidence; promotion of major architecture changes; plateau with multiple similarly plausible paths. Human can review a progress digest at any time. Absence of human input is *not* permission for unlimited training or API spending.

**Standing budget defaults (modifiable only by human):** at most one native experiment at a time; run only after native-smoke gate; pilot <= 20k environment decisions per candidate, <= 60 minutes per CI job before explicit approval; no automatic permanent champion promotion; no new cloud service provision or paid resources. Stop expensive execution on infrastructure failure; continue cheap diagnosis and research planning. Keep final seeds sealed.

## 6. Scheduler architecture and autonomy boundary

- **Deterministic components we can execute today:** offline ledger validation, regression tests, reporting, CI checks, measured paired metrics, next-action selection from known blocked/ready state.
- **Native components (conditional):** real-Taisei build, PPO train, eval and replay export *only after smoke gate and runner succeed*. Each episode must be traceable to real native library.
- **Research reasoning:** causal diagnosis, literature synthesis, architectural pivots, experiment ideation require deliberate agent analysis and/or human review; a YAML GitHub workflow cannot independently perform open-ended reasoning.
- **Routine continuation:** scheduled assistant follow-up checks the repo and advances one bounded, verified action; no promise that a transient chat container runs forever. Durable state in GitHub + UKS enables cross-session reentry. Worker events must be recorded before a new turn/worker supersedes them.

## 7. Research priorities (dynamically revisable)

P0: Native simulator compilation and deterministic smoke, still blocked on the latest runner.
P1: True checkpointer + per-episode provenance, comparable deterministic baselines, actual first-hit and item-pickup instrumentation.
P2: Skill competence gate on one authentic Easy spell; verify reward-only ablation vs stable baseline.
P3: Controlled policy/observation strategy: true item features, risk-aware movement, tactical skill and selective collection.
P4: Curriculum across authentic spells/stages; multiple seeds, confidence and robust performance.
P5: RSI meta-level experiments: analyze history of succeeded/failed pivots, measure wasted compute and probability of champion improvement, tune supervisor decisions.

## 8. Transfer to UKS

`research/rsi/trajectory/events.jsonl` is the **detailed, append-only evidence log**; UKS holds the **compact discoverable synthesis**: principles, source pointers, milestone snapshots, failed assumptions, evolving capabilities and meta-lessons. Link the Git revision/experiment ID for each nontrivial claim. Avoid copying all run logs into UKS. Keep open questions and bad outcomes visible, not erased by newer plans.

Other persistent source-of-truth docs: `research/rsi/STATE.md`, `CHANGELOG.md`, `COMPETENCE_SPEC.md`. These documents guide decisions but do not replace measured training artifacts.


## 9. Autonomous actual-game recording and visual replay review

Every supported policy test must include an **automated play → record → verify → inspect → compare** cycle as specified in [REPLAY_VERIFICATION.md](REPLAY_VERIFICATION.md). This is a required evidence and troubleshooting capability, not a routine human manual review request.

- Run the **genuine** game on development seeds with exact model/loader/engine identities. Record complete native action/state/terminal logs, a valid playable game replay and, when a compatible rendered native launcher exists, genuine MP4 or frame capture. A headless FFI rollout is not itself a rendered recording.
- Automatically check replay/video contents and sync; reject header-only or empty captures (including the previous 44-byte .tsr). Replay native input/actions in an independent process; require complete seed/episode/terminal coverage.
- Inspect available genuine video/frames, concentrating on collisions, dodging, wall pinning, pickups, boss transitions and resource actions; preserve timestamped examples of failure and success. Do **not** call input traces, charts or generated visuals true game footage.
- If no genuine renderer/video or multimodal viewing is available, mark visual inspection **NOT EXECUTED**. Native telemetry can still support numerical game metrics but not claims of direct visual corroboration. Identify precisely the missing interface; attempt feasible automatic integration before requesting human review.
- Use identical conditions and matched development seeds for policy-versus-baseline comparison. Never promote a model on video anecdotes or offline BC accuracy alone.

Existing E3/E4 escrow artifacts are native checkpoints/logs/JSONL **without recorded MP4 or .tsr**, so previous native outcomes are not retroactively relabeled as visually inspected. Visual replay is an additional verification gate, not a claim that the original E3/E4 native measurements were fabricated.

## 10. Local execution operator and scheduled-task reliability (2026-10-28)

The **primary backend is local computation**, not GitHub Actions. The tested coordinator is `research/rsi/local_native_operator.py`, with source-only regression tests at `test_local_native_operator.py` and deployment instructions `LOCAL_OPERATOR.md`. It independently verifies original E3/E4 native evidence, strictly gates exact simulator origin/commit/submodule provenance and canonical branch, checks build prerequisites, and conditionally performs bounded E5 two-process native checkpoint replay, with single-run locking, unique output dirs, event journal, a 60-minute ceiling and idempotent success registry. It never dispatches GitHub Actions. Genuine video/TSR capture and visual analysis are **separate** required gates under `REPLAY_VERIFICATION.md` and must not be falsely inferred from headless native metrics or synthetic media fixtures.

The packaged runnable offline environment is persisted in the user Library at `/AI/Taisei_RSI_Local_Automation_2026-10-28/Taisei_RSI_Local_Automation_v2_2026-10-28.zip` (SHA-256 `c2252a50a18ebc350750596083e730abd9517c79e8cea2cd627404e4fc611b95`). Each scheduled invocation must materialize/verify it if no previous local workspace exists; a temporary assistant container cannot be assumed to stay alive between schedule executions. On 2026-10-28, 13 local-coordinator and 6 E6 source tests passed, and independent bundle extraction and hash verification passed. The authentic fork at `789200ff255c681b3ed4230a3f26f0b8c12c78fe` remains inaccessible (original remote 404 and local DNS failure); fresh game training, native replay and video capture are **not** verified. Evidence: `research/rsi/evidence/local_operator_debug_20261028_v2.json`, ledger event `T-20261027-47`.

**Stopping rule:** Do not endlessly repeat unavailable-source Git clones/pip installs or blindly repeat previously regressed E3 training. While blocked, perform independent offline analysis and source recovery; notify a precise single decision when source access requires the user. Once native E5 determinism and captured-renderer parity pass, prioritize an episode-diverse teacher demonstration and a bounded, matched-seed E7 challenger, not superficial CI success.
