# E5 native learned-policy replay — single explicit execution request

Experiment: E5_TWO_PROCESS_NATIVE_POLICY_REPLAY_V1
Requested: 2026-10-26, Asia/Singapore
Canonical branch: research/rsi-automation-v1
Workflow: .github/workflows/rsi-e5-native-replay.yml
Scientific parent: trajectory T-20261026-33

## Preregistered question
Do the immutable E3 original BC and E4 rebalanced BC checkpoints produce exactly the same native Taisei movement actions and synchronized player/threat/post-action states when re-evaluated on the same game seeds in independent fresh OS processes?

## Protocol
- Genuine pinned Taisei Stage 1 Easy, Reimu A, movement-only 18 actions, greedy (non-sampled) policy.
- Inputs: immutable E3 run 40219788151 and E4 run 40220975924 archive artifacts; fail closed if SHA-256 mismatches.
- Two development seeds: 42132 and 42133, two independent evaluations per neural checkpoint, 3000 maximum frames per episode.
- Per candidate policy: <= 2 seeds x 2 runs x 3000 = 12000 environment decisions; 0 new training decisions.
- Native smoke before evaluation; <=60-minute job; shared native-serial concurrency key, no intentional overlapping native execution.
- Independent reparse of 16-seed E4 historical native episode outcomes is a separate evidence integrity check, NOT a new evaluation.
- Verify per-action movement and corresponding pre/post native trace JSONL exactly (including player coordinates/projectile state) and terminal gameplay digest for each repeat.
- Preserve traces, model identities, manifest, native build logs, audit report, and failure evidence.
- Success means repeatability of checkpoint-driven native behavior ONLY. It cannot establish better gameplay, dodging, item collection or survival beyond original matched-seed evidence.

Never touch final seeds 43000–43099, champion, main, paid services, or toy gameplay. Any replay divergence blocks model promotion and further HPO until diagnosed.

## Decision rule
PASS: Archive reproducibility evidence; next design diverse independent native teacher-data collection and trajectory-disjoint evaluation.
FAIL/BLOCKED: Archive first divergent action/seed or build/ABI issue, fix that narrowly, then repeat at same budget.
