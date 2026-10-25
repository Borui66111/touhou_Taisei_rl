# Explicit E4 frequency-aware Alice learning experiment

- Date: 2026-10-26 SGT
- Canonical branch: research/rsi-automation-v1; no new branch
- Experiment ID: E4_AUTHENTIC_TAISEI_FREQUENCY_AWARE_BC
- Scientific motivation: E3 completed BC + PPO with real Taisei, yet 0/16 clears and mean survival regressed from 1346.5 (BC) to 1047.56 (BC+PPO). Teacher reference on E2 was far stronger (7598.19 mean), while 69.3% E3 demonstrations are Stay and rare dodge actions have low holdout recall.
- E4 intervention: same 12000 genuine native E3 state/action demonstrations and original BC baseline checkpoint from archived GitHub Actions run 40219788151, pinned by SHA. Same 528-input/18-movement MLP, training seed 48030, 16 epochs. Class-frequency-aware imitation = 50% original CE + 50% normalized capped inverse-sqrt-frequency weighted CE. No PPO updates.
- Training-game interactions E4: ZERO new native training interactions (reuse exact archived source). Prior 12k teacher interactions are provenance, not E4 new compute.
- Offline metric: fixed candidate exploratory random-frame holdout macro recall 40.6% -> 49.2%, while frame accuracy ~83.9% unchanged. This does not certify generalization and may induce unsafe movement.
- Native evaluation: pinned authentic Taisei full Stage 1 Easy, no skill actions, new development seeds 42132..42147 (16 episodes each) for original E3 BC, E4 balanced BC and unchanged E2 risk teacher. All paired, same build/game state, 10k-frame caps.
- Metrics: death-free clears, alive frames, deaths, boss damage, resource use, truncation; paired bootstrap across game seeds, checkpoint/dataset hashes. Zero successful clears must be reported honestly.
- Budget: one GitHub Actions job, 60min max. Never touch sealed final seeds 43000..43099. No champion promotion and no automatic merge.
- Condition for E5: actual E4 native evaluation is required; if action-balanced BC regresses, investigate temporal/diverse demonstrations and action-conditioned risk representations, not repeated reward tuning.
- Launch trigger: addition of this file runs .github/workflows/rsi-e4-bc.yml.
