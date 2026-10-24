# RSI strategy research: next falsifiable native-Ta​​isei hypotheses
Date: 2026-10-25 SGT
Status: PRE-REGISTERED RESEARCH OPTIONS; no challenger training or gameplay improvements claimed.
Canonical ledger: research/rsi/trajectory/events.jsonl. Previous 200k PPO checkpoint and 0/48 clear result are source-reported native evaluations, not independently reproduced in our CI runner.

## Root problem: useful defensive competence versus numeric reward

The existing controller contains top-64 threat-sorted bullet rows, laser points and global states. PPO's fixed-size MLP processes them with one-frame action choices; previous rendered replays included near-constant movement and rapid direction alternation. Native item observations are not fed into the PPO encoder, and item pickup events are not independently proven. A claim that reward tweaking alone fixes dodging is therefore too strong: failures may derive from deficient temporal structure, risk features, short training budget, task choice, or live-deployment parity.

The sequential research design below prioritizes isolating causes over running many underpowered PPO configurations.

## Competing hypotheses and discriminating interventions

| ID | Hypothesis | First intervention | Prediction if true | Evidence that would weaken it | Cost/order |
|---|---|---|---|---|---|
| H0 | Native training/evaluation path is invalid | Build, ABI/version check, two fixed-seed deterministic repeat episodes + checksums | Native engine repeatably executes & traces real actions | ABI or source version mismatch, crashed game, inconsistent native digest under identical inputs | First; build only |
| H1 | Current reward promotes frame accumulation/resource spam over clears | E1 control current reward vs survival/clear-focused coefficients, paired task+training seed budget | More death-free clears, delayed first hit and/or fewer deaths without more skill spam | No competence gains despite sufficient training; worse survival, despite adjusted reward | After H0 |
| H2 | MLP sees current relative bullets but lacks an adequate anticipatory collision representation | E2 add per-candidate horizon clearance/time-to-impact and wall proximity, keep RL algorithm/reward fixed | Improved near-threat avoidance and lower wall-pin duration | No improvement under hazard-matched seeds; precision/recall of threat proxy poor | After H0 and H1 diagnostics |
| H3 | First-frame movement oscillation is due to missing temporal context | Short action-history/stacked frames, then recurrent PPO separately | Better action stability, improved conditional survival with no new skills | Same oscillation, more deaths or only high-frequency suppression without actual avoidance | Later L2 |
| H4 | Fully model-free PPO exploration is inefficient in dense bullet fields | Rule-based threat-aware safety teacher or policy distillation; validate teacher first | Faster time-to-first-survival skill than random-initialized PPO at equal native decisions | Teacher itself fails or student never exceeds teacher on held-out patterns | after E0 baseline |
| H5 | Collection is unlearnable without item state | Add versioned top-K item features AND verified pickup count, then compare reward-only pickup vs no pickup | Real item pickups rise at matched death cost | No increase in verified pickups, or only increased risky deaths | after native pickup instrumentation |
| H6 | One spell overfits and will not transfer | Repeat on distinct spells and seeds with stage identity held out | Positive held-out phase survival under difficulty shift | Severe overfitting despite training-seed success | late-stage generalization |

For H2: use a conservative predicted clearance `d*(a)=min_{t∈[0,H]} ||p_b + t(v_b-v_a)|| - R_b - R_p` for each bullet and **candidate move**. This proxy assumes locally linear dynamics; curved/aimed bullets and laser geometry demand validation against actual native traces and damage states. Do not call it an oracle or exact collision prediction. One-frame reward should never privilege high-frequency direction switching as an end in itself.

## Candidate learning approaches: explicit alternatives

A. **Existing MLP-PPO**: baseline; simplest to validate and compare, but sample inefficiency and lack of temporal memory are hypotheses, not demonstrated causes.
B. **Teacher-distilled/residual RL**: evaluate existing geometric heuristic on the same authentic Easy task before deciding whether it can teach useful maneuvers. If teacher is weak, imitation can cap success; do not assume BC is beneficial.
C. **Spatial occupancy/PCCM + CNN-PPO**: inspired by the public LunAI bullet-hell project, which reports multi-scale, multi-frame hazard maps, PPO and curriculum in a different pygame game. Transfer to Taisei is an idea, **not evidence of improved Taisei performance**.
D. **Short-history or recurrent PPO**: may learn attack phases, bullet movement and action persistence; higher data cost and ablation complexity.
E. **Model-based short-horizon risk controller**: conservative safety baseline and hybrid mentor for movement; goal must remain beating baseline by valid learned competence, not merely demonstrating scripted avoidance.

## Controlled E0 protocol before new 200k runs

- Confirm native engine version and replay fidelity. Exact stage=1, difficulty=1 corresponds to Easy in pinned src/difficulty.h (`D_Easy=1`, `D_Normal=2`). Do **not** mix Easy with earlier pilot default `difficulty=2`.
- Identify stable, genuinely supported Easy Spell Practice ID with native probe before registering curriculum.
- Evaluate frozen random, stay+shoot, existing heuristic, archived checkpoint on development seeds. Record exact checkpoint hash, game commit, native build, stage/spell, practice flag, starting lives/bombs, action stride, training seed and deterministic/sampled policy mode.
- Telemetry: first hit/death frame; vulnerable (non-invulnerable) frames; death-free encounter completion; resource spends, action oscillations, wall-pinning; item pickup only once native confirmed telemetry exists. Use missing, not fake zero, for unavailable signals.
- After low-cost smoke, compare H1 reward challenger first on a short training budget that is identical between arms, then increase only if enough learning signal. Pre-register 3 independent training seeds and 40 development evaluation seeds for scientific promotion, with paired intervals and correct task identity; do not use final 43000–43099.
- If the run is invalid or reward-only experiments are inconclusive, update the trajectory and choose among H2/H4, **not** simply repeat with more compute by default.

## Sources and limits

1. Pinned Taisei source: https://github.com/touhourl/taisei-sim/blob/789200ff255c681b3ed4230a3f26f0b8c12c78fe/src/difficulty.h (Easy=1 Normal=2)
2. Project real PPO: https://github.com/Borui66111/touhou_Taisei_rl/tree/feat/native-rl-workbench/my_test
3. Bullet-hell transfer *candidate*: https://github.com/Tweet-META/LunAI — pygame Touhou-style game; cannot be cited as native Taisei success.
4. Patterson et al. (2024), Empirical Design in RL: https://www.jmlr.org/beta/papers/v25/23-0183.html — variance, baselines, HPO/experimenter bias.
5. Agarwal et al. (2021), Deep RL at the Edge of the Statistical Precipice: https://arxiv.org/abs/2108.13264 — few-run evaluation uncertainty.
6. Allik et al. (2022), Guided Projectile PPO with Curriculum: https://arl.devcom.army.mil/arlreport/arl-tr-9603/ — curriculum helps sparse rewards in different RL control; transfer hypothesis only.

## Next decision from evidence

Until H0 passes, prioritize native build and fixed-state replay. Do not launch full training or modify the champion. After H0 passes, freeze a verified Easy curriculum and run E0 baselines and the 1,024-step pipeline pilot; if healthy, authorize controlled H1. After every run, append hypothesis, intervention, observed results (including negative), interpretation, costs and next action to ledger; do not rewrite earlier findings.
