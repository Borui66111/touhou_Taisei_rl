# E3/E4 immutable evidence escrow — one-time explicit request

**Canonical writable branch:** `research/rsi-automation-v1` only.
**Scope:** Source-only archival of completed authentic-native E3/E4 artifacts before they expire on 2026-11-15 UTC.

- E3: run 40219788151, artifact 12609092648, ZIP SHA256 `ecb660decee9706dd62c8a98023d0fe0979b578c24eb6a037c089f1014b169d1`.
- E4: run 40220975924, artifact 12609314728, ZIP SHA256 `1dc6ab9ffe7d5ab88d30c162246d0b2c7fe66c29556fb751b54886b4d2430789`.
- Hash-check E3 demonstrations and BC/PPO checkpoints and E4 checkpoint and evaluation JSONL, then preserve the original evidence files and a full SHA256 manifest under `research/rsi/evidence/escrow_e3_e4_20261026_v1/`.
- One GitHub Actions source-only job, 15-minute timeout, fast-forward-only commit on the same branch, no native Taisei execution or new policy training.
- Never access sealed seeds 43000–43099, champion, or main; never create another branch or force-push.
- Source checkout is still unavailable; escrow is not source recovery or gameplay improvement.
- If workflow cannot push, preserve diagnostics and use the independently verified user-facing ZIP backup.

This file is an opt-in trigger, not a claim that the workflow succeeded.
