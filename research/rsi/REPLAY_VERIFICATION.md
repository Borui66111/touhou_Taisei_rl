# RSI Autonomous Gameplay Recording and Replay-Inspection Gate

**Status:** Required verification protocol; recording/rendering integration is **not yet validated**.  
**Canonical branch:** `research/rsi-automation-v1` (see `BRANCH_POLICY.md`).  
**Simulator:** genuine pinned Taisei, not a toy renderer or simulated substitute.

## Why this gate exists

Source tests, successful PPO execution, imitation accuracy, native summary JSONL, decoded input-only `.tsr` records and even deterministic engine smoke do **not** establish that a policy is dodging bullets or collecting items as intended. The 44-byte header-only `.tsr` incident showed that an apparently existing replay can contain **zero gameplay**. RSI automation must actively produce and inspect real-game recordings when the runtime supports them, instead of routinely assigning gameplay verification to the user.

## Mandatory automated closed-loop sequence

1. **Authenticate the game and policy.** Read `BRANCH_POLICY.md`; check the pinned simulator commit and recursive submodules/resources. Record exact source commit, game binary/native library hash, model/checkpoint SHA-256, accepted loader format and acceptance log, seed, stage/difficulty, shot, initial lives/Bombs/power, reward/observation/action implementation, and whether actions are greedy or sampled. Reject incompatible `TPPOv1` / `TSKILL2` loaders.
2. **Actually execute native Taisei.** Run the named candidate and matched baseline on **development** seeds only with identical starting conditions. Native smoke/determinism and policy inference must pass. Record actions, pre/post logical frames, game digest, player coordinates/aliveness, enemy/projectile and item state when available, hits/deaths/respawns, pickups, boss progress, Bomb/Special activations, truncations and true stage clears. Keep frame-indexed logs aligned to the recording; do not infer missing facts.
3. **Capture actual gameplay.** Prefer both (a) the game's complete native replay file and (b) rendered gameplay MP4 or original PNG frame sequence produced by **the genuine Taisei renderer**, if supported. Save capture logs and exit status; capture graceful termination and unexpected death/time-limit endpoints. Native headless stepping alone does **not** generate a genuine video. Never pass off plots, overlays, synthetic graphics, or input timelines as rendered game footage.
4. **Validate capture before analysis.** Reject absent, zero-byte, header-only/empty-payload, truncated or unreadable recordings. Use `inspect_rendered_tsr.py` as a *preflight*, not proof of playability. Parse replay metadata/events with a genuine decoder where available. Verify seed/stage, input count, event continuity, first/last frame, event checksums, expected terminal frame, player outcome and resource accounting. Confirm video is actually decodable with ffprobe/ffmpeg or equivalent; verify timestamp/frame coverage against telemetry. If a field cannot be verified, mark it `unknown`.
5. **Independent replay.** Load the recorded game/checkpoint in a fresh process and compare action and native pre/post-state streams, terminal digests and replay status using `e5_audit_replay.py` or stronger native instrumentation. Replay-header validity and identical *incomplete* traces are insufficient. Explicitly record any desynchronization.
6. **Automatically inspect visual evidence.** Sample frames from **real recorded video** at fixed intervals and around detected events (first hit, deaths, dangerous bullet patterns, long repeated action, near-boundary behavior, item spawn/pickup, boss phase, Bomb/Special). If model vision/video tooling is available, examine the frames/video and write concrete timestamped observations; when absent, preserve video and annotate `visual_inspection_not_performed`, not a fabricated audit. Cross-check visual judgments against synchronized native state. Bullet proximity and movement-direction changes alone are not confirmed dodges.
7. **Compare and decide.** On matched development seeds, examine clears, first-hit time, vulnerable survival, deaths, verified pickups, boss damage/progress, Bomb economy, action collapse and wall pinning. Include representative **failure** clips, not just impressive successes; count time-limit truncation separately from clears. Never claim learned policy improvement on video anecdotes alone: require comparable native metrics and uncertainty. No champion promotion without a separate approval gate.

## Minimum artifact bundle (per candidate × development seed)

```
<experiment>/<policy_sha_prefix>/<seed>/
  provenance.json             # source, simulator, loader, policy hashes and settings
  native_episode.json         # authentic terminal metrics; end reason explicit
  actions_states.jsonl        # synchronized action/pre/post state or documented limitations
  recording.tsr               # if genuine renderer supports replay capture
  gameplay.mp4                # if actual rendered capture works; else genuine PNG frames
  capture.log                 # commands, ffmpeg/game exit statuses and errors
  capture_validation.json     # empty/truncated/metadata/coverage checks
  replay_verification.json    # independent process comparison, or explicit BLOCKED
  visual_review.json          # timestamps, inspected genuine frame SHA and observed events
  audit_conclusion.json       # factual observations, uncertainty, next action
```

If a file is unavailable, record an explicit `not_available` reason in `provenance.json` / audit rather than creating a placeholder disguised as evidence. Every produced file should have SHA-256 and byte count in an immutable manifest. Preserve snapshots/links and experiment IDs in the append-only trajectory.

## Evidence levels — do not collapse them

| Level | Verified | Not established |
| --- | --- | --- |
| Source preflight | Tests, input formats | Simulator was played |
| Native episode | Real engine metrics on known seeds | Replay reproducible or visually inspected |
| Deterministic native replay | Complete trace and terminal agreement across fresh processes | Genuine displayed footage or visually verified hazards |
| Captured rendered footage | Authentic visual output decodes and matches run provenance | Anyone or any model inspected it |
| Visual audit | Timestamped actual frame/video examination corroborated by native telemetry | Causal learning improvement without paired baselines |
| Matched competence result | Pre-registered native comparison with uncertainty | Generalization beyond tested development seeds |

**Human escalation policy:** The automation owns all feasible capture, playback, frame extraction, inspection and failure diagnosis. Ask for intervention only for unrecoverable credentials/locked exact source, inaccessible hardware/runtime, paid services, destructive authorization, or unresolved identity conflict. State the *precise* missing prerequisite. Never ask the user to play/record the game simply because manual review would be easier.

**Current blocker (2026-10-27):** Previous runs reported unavailable exact simulator source at `789200ff255c681b3ed4230a3f26f0b8c12c78fe`. Existing `inspect_rendered_tsr.py` is header-only and E5 is state/action replay, not genuine rendered-video capture. Actual game recording and visual inspection cannot be claimed until a compatible native **rendered** run is executed and its footage examined. The branch policy, training budget caps, and sealed seeds `43000–43099` remain in force.
