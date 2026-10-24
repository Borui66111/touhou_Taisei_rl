# Alice Competence-First RSI — Research and Engineering Specification v1

**Status:** proposed, not a verified model improvement. **Date:** 2026-10-25.
**Repository:** `Borui66111/touhou_Taisei_rl`; branch `research/rsi-competence-v1`; ancestor `28e26ae850693cfe73cdf872ad84a4df3e91e749`.
**Aim:** demonstrate *genuine gameplay skills* in authentic Taisei, not increased reward, model entropy, or simulated toy scores.

## 1. Operational definition of competence

Alice must exhibit **(S) survival**, **(D) risk-responsive dodging**, **(C) intentional collection**, and eventually **(O) offensive progress** and **(K) resource management**. Every claim needs behavioral evidence and paired baselines.

| Skill | Real gameplay test | Primary observable | Invalid substitute |
| --- | --- | --- | --- |
| S — survival | True Taisei spell-practice / Stage 1 runs with fixed setup and different development seeds | Death-free encounter clear rate; first-hit frame for failed runs; death count | Reward sum, artificially long unfinished episode |
| D — dodge | Repeated *vulnerable* bullet encounters, action+player+hazard trace | Avoided actual collisions, conditional near-threat survival, distance to nearest future trajectory, no prolonged wall pinning | Number of direction changes or entropy |
| C — collect | Native item pickups across naturally spawned items with player at risk | **Confirmed collection events** or per-type item collection counters; collection rate conditional on visibility/safety | Item disappeared, score increase alone, graze count |
| O — offense | Stage/phase progress with authentic boss health and clock | Phase clear, completion time conditional on clear, valid damage | Damage accumulated by Bomb spam alone |
| K — skills | Legitimate Bomb and Power Surge usage | Verified activations/discharges, death avoidance, resource efficiency | Key-press attempt count or activation without effect |

### Minimum meaningful capability gates (provisional preregistration targets)

- **G0 infrastructure:** native sim loads, deterministic reset and action replay validated; no toy fallback; first automated native pilot complete.
- **G1 nontrivial defense:** on a *verified named authentic Easy Spell Practice* instance, at least **30% death-free clears across 40 development seeds**, or predeclared similarly discriminating absolute bar if the encounter's mechanics make clear rate ill-posed; performance exceeds heuristic and random on paired trials with confidence intervals. Initial target, not yet achieved.
- **G2 robust dodging:** demonstrate positive paired improvement over baseline in vulnerable-state survival, and decreased collisions per threatening encounter; show 3-5 native traces, at least one failing case, and verify no mere invulnerability/Bomb exploit. Operationalize and validate the threatening-encounter counter using native collision data.
- **G3 intentional collection:** verified item pickup count beats stay-and-shoot and safety-matched heuristic, without materially increasing deaths; minimum 20 matched development episodes, multiple item opportunities. **Blocked until instrumentation for actual pickups is verified.**
- **G4 Stage 1:** repeatable clear/strong progress on full Stage 1 Easy without excessive resources across ≥40 matched seeds. Record resources and build identity.
- **G5 generalization:** unseen but related Taisei spells/stages, not memorized one specific pattern. Final sealed seeds 43000–43099 used only after design freeze; not for tuning.

Thresholds are engineering targets, not observed results. One good replay or 4 pilot episodes is not sufficient.

## 2. Native environment audit: established versus missing

Source `my_test/native_rl/env.py` already exposes `obs["bullets"]`, `obs["lasers"]`, `obs["items"]`, skill availability, Boss state, game status, and native seed. The current `torch_ppo.features()` and `skill_features()` **discard `obs["items"]`**; deliberately learning which visible item to collect is therefore not adequately observable. Items should enter a versioned observation encoder (top K with masks, relative position/velocity if available, item type, age, attraction, value), not be mixed with current checkpoint architecture without a new checkpoint format.

Game state exposes `items` as dynamic objects. **Native confirmed collection is not yet in the Python observation or episode report.** Falling item count and score are *not* adequate proof of pickups (offscreen despawn, item auto-collection, and scoring events can differ). Prefer adding an explicit pickup counter to the native sim API or a verified pickup event stream. Avoid ABI changes in the first automated pilot.

Current reward: `r = 1.0 * alive_post_frame + 0.02 * boss_damage + 0.001 * graze - 10 * deaths + terminal_bonus`, with per-frame observation. Across thousands of frames, survival reward can drown the death and clear terms; meanwhile `gamma=.995` discounts rewards 600 steps away to ~0.049. These are *diagnosed design risks*, not evidence that a particular policy failed because of them.

Current `skill_action_mask()` suppresses SPECIAL when surge is active. Native Taisei allows pressing Special again to discharge an active surge, so v2 mask is a **deliberately incomplete action semantics** until activation/discharge has been tested and implemented. Use a versioned action mask and native tests.

Current `heuristic_action()` uses fixed movement speeds and closest-approach approximations: useful as a weak, transparent baseline, not a validated optimal teacher.

## 3. Research ladder — independently testable changes

**E0: Establish comparable baselines.** Freeze task (stage, difficulty, practice, shot, initial lives/bombs, native build), 40 development seeds, deterministic greedy vs sampled policy distinction, one known checkpoint hash. Evaluate random, stay+shoot, heuristic, the published 200k masked PPO checkpoint, and the pilot policy if available. Each episode must record actual terminal outcome, deaths, first-death frame, alive time, Bomb uses, Power Surge activation/discharge, progress. Invalid/missing data fails closed. Include simple trace alignment to rendered replay.

**E1: Reward calibration only.** Keep observations, PPO network, stage, seeds, and training budget fixed. Compare current default reward against a survival/clear-first formulation. Proposed *illustration* for one-frame actions:
```
r_base = 0.002*alive_frame - 30*death_delta + 100*encounter_clear
         - 30*episode_loss + 0.001*valid_damage
```
Graze/item rewards disabled during defensive control. The factor of 0.002 makes 5,000 alive frames worth 10, less than a death (30). These are trial coefficients subject to ablation, **not already verified optimal weights**. Avoid active survival pay encouraging idle indefinite play: fixed episode timeouts and explicit successful completion priority; analyze clear-conditioned time separately. Use `gamma=.999` as a single independently controlled follow-up if horizon is too short; do not change gamma and reward simultaneously in a single challenger.

**E2: Threat-aware observation only.** Add actual player position vs edges, bullet time-to-closest approach and collision clearance, active laser proximity, nearby enemy threats. Preserve base observation and compare a versioned `features_v2` with one independent training change. Threat approximation is a proxy, not exact engine collision; validate against native hit signals. Consider permutation-invariant pooling (DeepSets) as a separate architecture ablation after plain MLP feature tests.

**E3: Learning to collect.** After verified pickup telemetry is added, add top-K item features and item action-value shaping, initially in a safe low-pressure real Taisei phase. Use item pickup reward *only on confirmed actual pickups*, optionally potential-based approach reward:
`F_t = gamma * Phi(s_{t+1}) - Phi(s_t)`, where `Phi(s)` increases for safely reachable desired pickups and is 0 on terminal states. This shape has known policy-invariance conditions for a fixed Markov task, but does **not** guarantee invariance if observation is partial, boundaries/terminations are mishandled or reward objectives change. Rank collection after survival; report collection conditional on not dying.

**E4: Movement stability and risk-responsive control.** Verify observation parity between Python and rendered C first; instrument logits, x/y, hazard clearance and wall contact. Only then test action persistence (2-4 frame hold, 1-frame native physics and reward measurement preserved via within-loop actions), short observation history, or recurrent network as *separate challengers*. Oscillation can be tactical; do not punish direction changes blindly.

**E5: Tactical skills.** Correct SPECIAL semantics (activate vs early discharge), test native event transitions, then compare no-skills, threshold-bomb baseline and learnable skills with same movement policy. Skill-aware success requires fewer deaths and sensible resources, not unlimited Bomb chaining.

## 4. A grounded dodging concept

For bullet relative displacement `p=(dx,dy)` and relative velocity `v=(vx,vy)`, compute `t*=clip(-p·v / (||v||²+epsilon),0,H)` and `d*=||p+t*v||`; subtract a conservative tested effective collision radius `R` to obtain predicted clearance `c=d*-R`. This is valid only as a *linear-motion proxy*; curves, homing bullets, laser geometry and invulnerability require native traces. Near-threat measures are active **only when Alice is vulnerable** and should never be called true dodges without evidence that a collision was prevented.

Example: Alice is at (240,500). A bullet is 30 pixels right and 40 above, moving downward 2 pixels/frame. If Alice stands still, the closest-approach time is 20 frames and minimum distance 30 pixels. Running right can reduce clearance by moving into the bullet path. A useful policy must value predicted crossing risk, not merely current pixel distance.

## 5. Curriculum in authentic Taisei

- **C0 engine identity:** correct stage/spell IDs (the uploaded replay includes Spell Practice ID `0x8104`, but verify the native library accepts the same ID), same build/game version and physics.
- **C1 defense:** short authentic Easy Spell Practice segments or a natural Stage 1 early section; prioritize no-hit survival, shooting can remain automatic.
- **C2 offense:** add verified boss phases and faster clear to resolve pure-stalling behaviors.
- **C3 collect:** natural actual item windows with certified pickup telemetry; demand survival-aware collection.
- **C4 mixed:** multiple real attacks/stages, balanced hazards and item opportunities; prevent overfitting to a specific seed or opening animation.

A curriculum is a sequence of authentic game tasks or stage selections; **never fabricate a substitute physics simulator and label that result Taisei competence**.

## 6. Controlled RSI outer loop

Inner loop: PPO updates policy parameters `theta` on native game transitions. Outer loop: search over *training procedure choices* `eta`, one preregistered hypothesis/challenger at a time. Each candidate has immutable `{code_commit, native_commit, task_id, training_seed, eval_seed_set, model_hash, config, budget}`.

Decision rules: (a) no regression in death rate beyond predeclared tolerance; (b) evidence-backed improvement in clear rate/phase progress/first-death time; (c) no excessive resource exploitation; (d) equal environment interaction budget and at least 3 independent training seeds when making a research claim. Use paired bootstrap confidence intervals; do not infer statistical significance from a point estimate. One challenger changes **one factor** (reward, observation, temporal architecture, action mask, or training budget). No open-ended autonomous code edits, no auto-promotion, no final-seed exposure, no reward-only winner selection.

Pseudocode:
```
champion = versioned_baseline
for hypothesis in registered_hypotheses:
    challenger = train_on_real_taisei(hypothesis, fixed_budget, training_seeds)
    report = evaluate_paired(champion, challenger, dev_seeds)
    if validity_checks_pass(report) and improvement_is_supported(report):
        propose_promotion_for_review(challenger)  # no silent overwrite
    else:
        reject_or_investigate(challenger)
freeze_method_and_champion()
evaluate_once_on_sealed_final_seeds()
```

## 7. Next execution checklist

- [ ] Repair/check native GitHub Actions build gate and save validated smoke logs.
- [ ] Verify 1024-step automated PPO pilot actually trained and generated a checkpoint + 4 matched-seed baseline records; do **not** call it learned competence.
- [ ] Add run manifest and first-death/progress telemetry; verify stage/spell replay compatibility.
- [ ] Freeze the published 200k model and run E0 matched baseline before introducing challengers.
- [ ] Implement E1 reward-only challenger with controlled training budget, saved model, and comparable evaluation.
- [ ] Implement E2 observation, E3 pickup instrumentation/feature/reward in separate versioned arms.
- [ ] Demonstrate at least one sustained, reproducible actual dodge/clear/collection skill before claiming autonomous RSI success.

## Research motivation / methodological foundations

- Ng, Harada, Russell (1999), *Policy invariance under reward transformations* — why careless reward shaping changes the optimum.
- Agarwal et al. (2021), *Deep RL at the Edge of the Statistical Precipice* — why multiple seeds and uncertainty intervals matter.
- Patterson et al. (2024), *Empirical Design in Reinforcement Learning* — controlled ablations, baselines, experimenter bias.

## Current evidence (not newly observed in this design document)

- Branch `feat/native-rl-workbench` at `d463a2fd` has a 200k masked skill PPO checkpoint and native evaluation JSONLs.
- In the 16-run masked candidate evaluation: 0 clears, 3 deaths per run, mean 2712.625 alive frames.
- First native GitHub workflow failed SDL3 X11/Wayland dependencies; repaired build/pilot is in progress separately.
