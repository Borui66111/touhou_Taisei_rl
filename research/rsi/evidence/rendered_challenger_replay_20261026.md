# Taisei challenger rendered replay audit — 2026-10-26

**Evidence:** decoded .tsr inputs and stage metadata, not gameplay video, AI policy identity, or player/bullet trajectories.

## Source identity and validation
- File: taisei_20261026_00-02-57-864_stg1_reimuA_easy(1).tsr, 17,701 bytes.
- SHA-256: 8675a534b83ac39b8dd9f8aea967f9efa8c67425dc5e16cdb195f04223ea3011.
- Taisei 1.5.0, zstd replay struct version 14; decoded metadata checksum valid.
- Stage 1 Easy (difficulty=1), Reimu A (character/shot=0/0), seed=6003837114837005130; starting lives=8, Bombs=3, power=400.
- Stage clear bit absent; last replay event frame=3885 (~64.75s at nominal 60FPS); lives used=3; Bombs used=7; continues=0; final score=1,000,382.
- 13,544 replay events, including 13 desync-check records; recorded checks are not verification of synced rendered playback.

## Recorded movement decisions
3,881 decoded frames with Shoot pressed (no player position data).

| Movement command | Frames | Percent |
|---|---:|---:|
| Left (unfocused) | 1,771 | 45.63% |
| Up + Focus | 924 | 23.81% |
| Right + Focus | 747 | 19.25% |
| Up (unfocused) | 331 | 8.53% |
| Down + Left + Focus | 83 | 2.14% |
| Down + Right (unfocused) | 25 | 0.64% |

- Six distinct command combinations, empirical movement entropy 1.9354 bits, 89 changes, Focus input on 45.19% of analyzed frames.
- Longest identical command: Left continuously on frames 2357–3372 (1016 frames or ~16.93 seconds). Possible boundary saturation or control collapse, **not proven** without position and hazard traces.
- Bomb press events: 740, 1062, 1404, 1926, 2357, 3417, 3717. Seven consumed bombs verified by metadata. Special presses at 0 and 411; successful activation/discharge not verifiable from replay alone.

## Earlier Stage 1 Easy replay (context, not paired)
October 23 replay SHA-256 9024536e75af05c005c537bb90dd140a956dedb31714084cb108d6befc26dcc0: last frame=3086 (~51.43 seconds), lives used=5, bombs used=1, score=42,807, movement commands=2, movement entropy=0.2205 bits, Focus 100%.
The new replay lasts longer, loses fewer lives, has more diverse commands, but spends six additional Bombs; crucially, seed and possibly controller/model differ. No causal improvement in dodging is established. Neither replay cleared Stage 1.

## Attribution and next experiment
- The local worker reported the intended skill-v2 demo accepts TSKILL2 policy binaries; prior TPPOv1 movement-only exports are incompatible.
- The replay alone does not contain the running binary hash, checkpoint ID, accepted loader format, AI inference/action count, player coordinates, bullets, enemy state, item pickups, collision events, or exact cause the run ended. Do not silently attribute Bomb/Special key presses to the challenger network.
- Require an unrotated live-controller log for this exact run with TSKILL2 acceptance, policy binary SHA-256 and run seed, and a synchronized rendered or native trace.
- Investigate long Left segment (frames 2357–3372) and each Bomb event for wall-contact, invulnerability, threat conditions and actual model skill decisions.
- Use matched Stage 1 Easy game seeds/starting resources against a no-skill and heuristic baseline, and reserve sealed seeds 43000–43099.

**Decision:** record as an input-behavior observation only; no champion promotion or claim of demonstrated survival/dodging/item collection.

Conversation artifacts: REPLAY_AUDIT_2026-10-26.md, audit_metrics.json, decoded_input_frames.csv, action_runs.csv, skill_key_events.csv, challenger_input_timeline.png, Taisei_Challenger_Replay_Audit.zip.
