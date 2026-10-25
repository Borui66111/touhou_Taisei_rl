# Taisei RSI — Default Branch and Experimental Work Policy

**Effective date:** 2026-10-25
**User-selected primary working branch:** `research/rsi-automation-v1`
**Repository:** https://github.com/Borui66111/touhou_Taisei_rl/tree/research/rsi-automation-v1

## Standing rule

All new Taisei RSI research, experiment definitions, source fixes, build workflows, evaluation scripts, progress reports, and trajectory/decision records **must be committed to `research/rsi-automation-v1` by default**.

Do **not** create a new branch for each experimental hypothesis, parameter sweep, curriculum run, troubleshooting fix, or checkpoint. Use versioned experiment IDs, configuration files, output directories, commits, checkpoints, and an append-only experiment ledger on this same branch to isolate work. Preserve previous results as evidence rather than overwriting them.

## Exceptions — require a verified reason

Use another branch only for a concrete, documented technical reason, such as:
- incompatible or destructive changes that would endanger the known working experiment pipeline;
- an independently reviewed integration task with a justified branch or PR isolation requirement;
- an existing external branch that cannot be safely incorporated without a controlled merge.

Before creating an exception branch, document its reason, the alternatives considered, expected return/integration path, and relation to the main RSI branch in the research trajectory; seek human review for consequential deviations.

Do not silently keep developing previously created experimental branches. Historical branches remain valid evidence sources, but future edits start from the selected branch and selectively integrate *verified* earlier work after inspecting diffs, build compatibility, and experimental provenance. Do not force-push, overwrite history, or merge `main` by default.

## Automation requirement

Scheduled research and monitoring must read this policy first, treat `research/rsi-automation-v1` as the canonical writable branch, and record all operations (hypothesis, training, evaluation, errors, next action) with source/run/checkpoint identifiers. Do not assume that a GitHub Actions workflow configured to trigger on a different branch will run after work is committed here: adapt and verify its triggers when integrating.

This is a branch-selection policy; it does not waive the existing safety constraints: authentic Taisei only; bounded compute; matched-seed evaluation; sealed final seeds `43000–43099`; no autonomous champion overwrite, unsafe rewrite, or unsupported gameplay claims.
