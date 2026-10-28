# Taisei RSI: second empty replay + observed no-dodge/no-pickup failure (28 October 2026)

## Observations and evidence classification

The user manually observed rendered gameplay that **failed to dodge** and **did not move toward collectibles**. This is credible user observation, not yet machine-verified movement traces. The recorded `.tsr` was 44 bytes and **byte-for-byte identical** to the earlier failed replay (SHA-256 `3eb7427f4c3372d997c3718c699803c16bb42fead5a0f644ff800bed48958943`). Both compressed sections begin with empty Zstandard frames whose standalone frame prefixes decode to zero bytes. Stage/outcome/actions cannot be recovered from the file. Its presence does not prove the game's replay writer was initialized.

## Confirmed model design defect: items omitted

The native simulator exposes `obs['items']` including item type and location, but `skill_features(obs)` uses only player, boss, bullets, lasers, counts, timing, and skill state. All item locations are discarded before the 546-float model input. Native reward gives survival, graze, boss damage, death/loss penalties and a win bonus, **no explicit pickup reward**. In a source-layout checkpoint probe using the authentic E1 `.pt`, modifying only item positions left both movement and skill logits exactly unchanged. This is **structural item blindness**, not proof about any particular live game state.

## Dodge causes remain hypotheses

Bullets and their velocities are in the model input, but failure to dodge could reflect lack of varied training contexts, reward/exploration failure, or a native-to-rendered observation scaling/order mismatch. A user-reported failure without frame-synchronized logs cannot distinguish these causes. Do not infer the replay failure was caused by the earlier atlas initialization problem; gameplay was reportedly seen on this attempt.

## Next actions and experiment design

1. Fix replay capture at its source. Record actual policy-loaded SHA, ordered decision frames, renderer build SHA, replay-start event, replay-save exit and size, plus the matching game log. Reject 44-byte replay automatically. Capture real rendered video (not synthetic plot) to avoid a single-point-of-failure in `.tsr` saving.
2. Verify native-versus-C rendered **observation and action parity** before interpreting a dodge failure as learning failure.
3. Preserve the **one** live controller (`TSKILL2`, 546 features, 18 movements + 3 skills, mask-v2). Item awareness needs a **versioned observation schema** and matching native plus renderer feature changes within that same controller, followed by new training. Never silently reinterpret old E1 weights. A possible controlled proposal is to reserve a small number of 7-float slots for nearest items, taken from the tail of the 64 projectile slots; study the survival tradeoff before adopting it.
4. Add actual pickup event instrumentation and a small survival-safe pickup reward, verify it triggers only on real pickups, then compare native runs on matched development seeds. Prioritize diverse hazard/teacher data and failure clips to improve dodging.

**No new native game simulation or visual footage inspection was performed in this diagnosis.**
