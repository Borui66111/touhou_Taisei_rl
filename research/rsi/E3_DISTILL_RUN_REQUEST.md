# Explicit E3 native teacher imitation + PPO run request

- Experiment: E3_NATIVE_TEACHER_DISTILL_V1
- Canonical branch: research/rsi-automation-v1; no new branch
- Motivation: E2 40213025937 genuine native risk teacher mean alive 7598 vs geometric heuristic 1660 over 16 matched Easy development seeds, but 0/16 clears. Test whether a neural actor can learn teacher's defensive behavior.
- Actual training environment: pinned authentic native Taisei, full Stage 1 Easy, Reimu A, 1 physics frame per movement decision
- Demonstration: 12000 teacher-controlled native decisions, from seed 48030 (following sequential episode seeds only after terminal)
- Neural policy: legacy 528-feature MLP with 18 movement actions, no Bomb/Special decisions
- Behavior cloning: supervised cross-entropy on source-observation/teacher-action dataset, no extra game interactions
- PPO: warm-start from imitation weights and train 8000 native decisions with seed 48020, same fixed reward formulation as existing legacy PPO
- Combined native training interaction ceiling: exactly 20000
- Evaluation: imitation-only and fine-tuned PPO on the same 16 Easy development game seeds 42100-42115, 10000-frame episode cap
- Research metrics: stage clears, survival frames, deaths, timeouts, boss damage, Bomb/Special usage, BC action accuracy, per-run checkpoint SHA and dataset hash
- Candidate accepted only for continued study, never automatically champion-promoted. No claim that BC holdout accuracy equals actual game competence
- Native build/smoke/deterministic replay required; no toy fallback
- Sealed 43000-43099 seeds never used
- 60-minute CI timeout, one native job at a time. No paid infrastructure or other branch.
- Trigger: this file creation launches .github/workflows/rsi-e3-distill.yml.
