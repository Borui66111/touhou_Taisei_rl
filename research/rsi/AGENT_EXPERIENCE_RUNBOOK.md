# Agent Experience Learning — executable L4 RSI pilot

**Status (2026-10-28):** Local evidence ledger, transparent lesson routing and instrumented benchmark scorer implemented and source-tested. **No independent agent before/after trial or actual policy improvement demonstrated.**

## Purpose and separation of concerns

The agent's own trajectory supplies *experience*, not automatically learning. Preserve four different artifacts:

1. **G0 raw facts:** immutable-ish project events, actual run/tool trace, user's correction, source/model hashes; keep primary source outside summarized UKS.
2. **G1 incident:** evidence-scoped causal hypotheses, original misconception, corrections, exceptions, actual cost, falsification tests.
3. **G2 candidate skill:** short trigger, actionable better decision, limits; retrieve before analogous actions without blindly copying the case.
4. **G3 evaluation:** agent action traces and independent human/agent audit, matched control/intervention tasks, failures, cost and negative transfer.

**Raw history must not be rewritten.** The hash chain detects changes relative to an externally preserved head hash. It does **not** provide cryptographic immutability if an attacker can rewrite the whole ledger and its head. Anchor the head in a trusted immutable commit/run manifest.

The executable helper (`agent_experience.py`) is intentionally small and deterministic. It does not invoke a language model or GitHub Actions, and it cannot promote a lesson. Retrieval is an *explicit keyword matcher*, not a learned semantic model. Actual agent decisions must come from an external execution system and be linked to observed tool/action trace evidence; a trace field claiming independence is not proof of independent verification.

## Source files

- `agent_experience.py` — validation, SHA-chained append, candidate retrieval, action-trace structural scoring and fail-closed promotion gate.
- `agent_lesson_registry_v1.json` — eight **candidate**, not validated, lessons derived from AIF-001/AIF-002 and earlier project corrections.
- `agent_failure_benchmark_v1.jsonl` — original 16 **proposed, unexecuted** Taisei failure scenarios, preserved without relabeling as run trials.
- `test_agent_experience.py` — 24 local tests including tampering, duplicate and timestamp checks, evidence/benchmark guardrails, false positive retrieval and synthetic-vs-independent distinction.

## Command line

From the directory containing the four files:

```bash
python -m unittest -v test_agent_experience
python agent_experience.py validate-benchmark --benchmark agent_failure_benchmark_v1.jsonl
python agent_experience.py retrieve --registry agent_lesson_registry_v1.json \
  --task 'A GUI agent judge reports SUCCESS but no execution evidence exists'
python agent_experience.py validate-ledger --ledger experience_events.jsonl
python agent_experience.py append --ledger experience_events.jsonl --event new_event.json
python agent_experience.py score --benchmark agent_failure_benchmark_v1.jsonl \
  --traces observed_agent_traces.jsonl --out score_20261028_run01.json
python agent_experience.py gate --baseline baseline_score.json \
  --treatment lesson_score.json
```

The `append` command expects a standalone JSON event with fields in the skill's experience schema. `score` expects each trace to contain `case_id`, unique `run_id`, `source_kind` (synthetic fixture / self-reported / independent claim), `actions` with ACT/ASK/VERIFY/CORRECT/REPLAN/STOP decisions, actual observation flags and action evidence refs, and optional reported hard failures. **Only** structural decision-label coverage is computed. Semantic success and real task outcome remain **NOT_PERFORMED**. The gate always blocks promotion without independent blind semantic adjudication and matched real-task outcomes. `--out` refuses to overwrite previous reports.

## Minimal actual experience capture (not simulated game learning)

Event ID `EX-TAISEI-20261028-01` describes the verified transition from a stored proposed meta-skill to a tested runnable prototype. It records the original defect (lessons not executable/evaluable), actual test outcomes and the remaining generalization gap. Its event status remains **candidate**. It must not be conflated with a future independently verified change to agent behavior.

## Next meaningful experiment — preregister before measuring

**Hypothesis:** retrieving the *right* G2 lesson before acting reduces avoidable user corrections and proxy-evidence overclaims, compared with an otherwise identical agent without the retrieval step.

**Design:** prepare an unseen set of mixed-domain task variants (Taisei, MobileJail, FastBench, scheduling, deployment) including correct ASK/STOP controls. Randomize baseline versus lesson-retrieval access, freeze prompts and tool permissions, compare matched scenarios, and audit actual actions using an evaluator blind to condition. Prespecify primary metrics: critical-failure frequency, correct first action, independently verified end-task completion, unnecessary human requests, correction count, tool/compute cost, and negative transfer. Retain per-run tool traces and all failures. Do **not** call this a successful learned ability based on perfect matches to development/test fixtures or an improved written protocol.

**RSI integration:** after significant corrections/milestones, capture G0/G1, retrieve G2 before next analogous high-stakes decision, run G3 regression/adjudication, and append its evidence. No uncontrolled auto-promotion of rules, policies, or champion models.
