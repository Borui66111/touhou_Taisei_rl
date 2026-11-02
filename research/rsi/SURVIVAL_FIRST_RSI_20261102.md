# Taisei RSI — survival and dodging first (2026-11-02)

## User-defined priority; overrides prior task-order advice

1. **P0: survive through real movement and deliberate bullet/laser dodging**, with accurate boundaries. Avoid unproductive repeated movement toward a wall, timeouts, or counting Bomb protection as a dodge.
2. **P1: skills** (Bomb / Power Surge) after the movement baseline is measured; then learn use as a resource-aware defensive/offensive tool.
3. **P2: collectibles** (versioned item observations + confirmed pickup events and reward) after survival improvement.
4. **P3: boss progress, generalization, efficiency, other aspects** after P0–P2; offense/timeouts are still necessary guardrails against stalling.

Existing `rl_skill_v2` + `TSKILL2` (546 input / 18 movement / 3 skill) is the **only live policy interface**. No legacy second controller; never silently reinterpret E1's feature layout. Source changes default to `research/rsi-automation-v1`; Windows integration remains on `feat/native-rl-workbench` only by explicit branch-policy exception.

## Critical boundary insight

Taisei movement is constrained to a playable area. A network may select an identical directional action for hundreds of frames, but after reaching the edge the character cannot continue moving. **Action selection != actual displacement**. This can look like intentional navigation in logs while producing no useful evasive movement.

For a vulnerable frame with player position `(x,y)` in pixels and commanded direction `(dx,dy)`, a potentially wasted command is:

- near left boundary **and** `dx < 0`; near right **and** `dx > 0`;
- near top **and** `dy < 0`; near bottom **and** `dy > 0`;
- **confirmed low progress** along the command vector using *post-step* `(x',y')` is stronger evidence than a near-wall command alone.

The analysis uses provisional playable bounds `(16,464,16,544)` based on the E2 source and a 12-pixel margin. **Verify those limits against the exact pinned native game** before interpreting thresholds as accurate collision geometry. Diagonal travel tangential to a wall may still be useful: the audit measures projected progress and does *not* automatically call it stalled.

Bullet proximity is different from true collision: current linear closest-approach is a proxy; only native death/hit transitions establish actual impacts. Exclude invulnerable/respawning samples from the vulnerable-state denominator. Do not infer that an action change proves a successful dodge.

## Newly implemented, no model change

- `my_test/native_rl/torch_ppo.py::evaluate(..., force_skill_none=True)` forces the skill action to `None` for actual skill-v2 E1 checkpoints, while leaving the **movement head unchanged**. It refuses this setting on a legacy movement-only or planner arm. The immutable original `.pt` is unchanged.
- `my_test/scripts/evaluate_native.py --no-skills` exposes that movement-only native evaluation, and optional `--trace-out` logs the **pre and post normalized player position**, skill action, hazards, life transitions, and invulnerability for every decision.
- `research/rsi/survival_dodge_audit.py` rejects skill-consuming, malformed, empty and sealed-seed traces; measures repeated outward commands, empirically stalled wall-directed commands, near-wall fraction, threat-proxy conjunction and max identical-movement streak. It reads JSONL, not a game binary or imaginary episode.
- `research/rsi/test_survival_dodge_audit.py` covers left/right/top/bottom constraints, wall escapes, tangential diagonal motion, lack of real displacement, invulnerability/death exclusion, missing telemetry, sealed seeds, and fail-closed parsing.
- `research/rsi/wall_veto.py` is a strictly **opt-in experimental safeguard**, *not* a new model/ABI: when E1 commands outward at an edge, it substitutes E2 risk-aware teacher movement. Only eligible in `evaluate_native.py --no-skills --edge-veto`; never silently enabled during training or rendered Windows play. It records original vs executed movement and correction count. `test_wall_veto.py` tests boundary conditions with synthetic observation fixtures; its unit pass is not evidence that the safeguard improves dodging.

These are **source-tested measurement tools**, not claims that E1 learned to avoid bullets. They do not repair the separate empty `.tsr` replay writer.

## Your October 25 local model identity

Your Windows file `my_test\\res\\Taisei_RL_Checkpoints_Local_Demo_2026-10-25\\challenger_seed48020.pt` is the **E1 skill-aware survival-first PPO challenger**, not a new E7 or movement-only E1 checkpoint. Provenance: original [native experiment 40187753883](https://github.com/Borui66111/touhou_Taisei_rl/actions/runs/40187753883), 20,000 training decisions, seed 48020, Stage 1 Easy, evaluation seeds 42110–42125. Expected SHA-256 `be320e7764b236b6d2b8c42eb26524fdd57345fe2182cc916ca53f0cdf8d4478`, full `SkillActorCritic` 546 inputs + 18 movement/3 skill actions, mask v2. Measured 4,477 mean alive frames (control 3,454), 0/16 clears, 3 deaths, 8.9375 Bomb uses per episode. This is evidence of a survival difference **with skills**, not evidence of competent dodging. The no-skill native evaluation below is intended to reveal whether movement itself learned threat avoidance.

Local verification from `my_test`:

```powershell
(Get-FileHash '.\\res\\Taisei_RL_Checkpoints_Local_Demo_2026-10-25\\challenger_seed48020.pt' -Algorithm SHA256).Hash
```

Only run the native arms if the file matches that SHA, with a valid exact pinned native library; avoid confusing the original Windows rendered game executable with a native `libtaisei_sim` testing library.

## Next bounded authentic experiment (three independent native jobs)

Do not execute if the exact pinned Taisei native library/resources and source manifest are unavailable. Run a native provenance gate first; the commands below verify path existence but **do not independently prove the binary was built from the pinned source**. Follow `research/rsi/BRANCH_POLICY.md` and the native source/provenance preflight. Use the **same** authentic Stage 1 Easy task, one-frame actions, 16 fixed DEVELOPMENT seeds `42152..42167` (not the sealed final 43000–43099); use distinct new output filenames. Never overwrite existing JSONL. All native arms must have **zero verified Bomb use** and compatible task/build identity.

From the repository root (replace the **example placeholders** with the proven local paths):

```bash
# Arm A: archived E1 skill-aware challenger, but force all skills NONE;
# do not re-export or fine-tune it.
python my_test/scripts/evaluate_native.py \
  --library /exact/pinned/libtaisei_sim.so \
  --resource-path /exact/pinned/resources \
  --stage 1 --difficulty 1 --policy ppo \
  --model /trusted/verified/challenger_seed48020.pt \
  --no-skills --seed 42152 --episodes 16 --max-frames 10000 \
  --out /new/results/e1_no_skills_42152_42167.jsonl \
  --trace-out /new/results/e1_no_skills_trace_42152_42167.jsonl

# Independent native process, same checkpoint/task/seed set: E1 with only the
# experimental wall-outward veto enabled; compare with E1 baseline above.
python my_test/scripts/evaluate_native.py \
  --library /exact/pinned/libtaisei_sim.so \
  --resource-path /exact/pinned/resources \
  --stage 1 --difficulty 1 --policy ppo \
  --model /trusted/verified/challenger_seed48020.pt \
  --no-skills --edge-veto --seed 42152 --episodes 16 --max-frames 10000 \
  --out /new/results/e1_wall_veto_42152_42167.jsonl \
  --trace-out /new/results/e1_wall_veto_trace_42152_42167.jsonl

# Independent native process, same task/seed set: E2 handcrafted risk teacher,
# used as a *non-neural* movement baseline.
python my_test/scripts/evaluate_native.py \
  --policy risk_teacher --library /exact/pinned/libtaisei_sim.so \
  --resource-path /exact/pinned/resources \
  --stage 1 --difficulty 1 \
  --seed 42152 --episodes 16 --max-frames 10000 \
  --out /new/results/e2_teacher_42152_42167.jsonl \
  --trace-out /new/results/e2_teacher_trace_42152_42167.jsonl

# Structural trace checks; read-only, no game execution.
python research/rsi/survival_dodge_audit.py \
  /new/results/e1_no_skills_trace_42152_42167.jsonl \
  --out /new/results/e1_no_skills_wall_audit.json
python research/rsi/survival_dodge_audit.py \
  /new/results/e1_wall_veto_trace_42152_42167.jsonl \
  --out /new/results/e1_wall_veto_wall_audit.json
python research/rsi/survival_dodge_audit.py \
  /new/results/e2_teacher_trace_42152_42167.jsonl \
  --out /new/results/e2_teacher_wall_audit.json
```

The risk teacher is a **handcrafted evaluation baseline, not a trained E7 checkpoint**. Verify all three arms are run from independent fresh processes because Taisei native engine global state is not safely reentrant. Run another original heuristic baseline if time permits.

## Acceptance and research choices

**G0 telemetry:** Actual native source manifest and executable hashes, correct seed/start/terminal, per-action trace with pre/post actual position, actual outcomes and no skills, both arms same task. Any missing data -> **NOT VERIFIED**.

**G1 survival:** actual first death frame, death-free progress/clear, death count and vulnerable alive frames; handle 10k timeouts separately (never count them as clears). On paired 16 seeds report distributions, not only one mean. Full research conclusions need independent training seeds and predeclared criteria.

**G2 dodging and boundary:** report outward commands while at the edge, measured low-displacement stalls, maximum identical-direction and wall-pinned streaks, threatened-state behavior, and *actual* deaths during vulnerable bullet encounters. Measure whether E1 movement-only improves or regresses relative to E2 and simple heuristic. A long timeout or high entropy is NOT a successful dodge.

**G3 conditional improvement:** only after G0–G2 pass, collect multiple independent actual E2 teacher episodes and implement one bounded defensive student under **the same skill-v2 network/TSKILL2**. Use honest episode-disjoint validation, frame diversity and risk-specific weighting. Fix observation parity in C/native before any live Windows efficacy claim. Test boundary-aware action veto as a **separate optional ablation** only if repeated outward motion is measured; do not silently add an untested filter to the live game.

Skills, collectibles, and offensive optimization remain next-phase goals, not concurrent distracting sweeps. The scheduler's manager review should report whether work is resolving G0–G2 and stop endless replay/debug churn when real native traces can answer the survival question first. A genuine playable replay and MP4 are still required for independent *visual* corroboration, even if native survival metrics are already measured.
