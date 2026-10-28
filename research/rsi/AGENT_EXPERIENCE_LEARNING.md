# Agent skill — Experience → Lesson → Test → Policy → Transfer

**Skill ID:** `AGENT-META-LESSON-001`  
**Created:** 2026-10-28 (SGT)  
**Scope:** UKS post-mission learning, agent design, failure/vulnerability benchmarks, RSI L3/L4 governance and cross-project transfer  
**Maturity:** **protocol specified and case captured**, NOT demonstrated autonomously learned behavioral improvement  
**Core insight:** The agent's own collaboration history—wrong assumptions, user corrections, experiments, failed attempts, successful recoveries—is a dataset of **experience**. The aim is not to remember more text; it is to improve *future decisions and actions*.

## 1. Desired capability, not just a notebook

After substantial work, the agent should become better at:
- modeling **the ultimate user goal**, and detecting when a useful proxy is silently becoming the endpoint;
- selecting between **ACT / ASK / VERIFY / CORRECT / REPLAN / STOP** under constraints;
- diagnosing why a chosen strategy failed and which untested assumptions mattered;
- recovering from tool/environment changes without repeat failures or fabricated completion;
- generating and testing causal hypotheses instead of running an endless collection of familiar checks;
- **recalling the right lesson before a similar failure**, including analogous-but-not-identical tasks;
- preserving scientific integrity and *not* confusing activity, tests, accepted proofs and real target success.

This is an **agent architecture / decision-policy** capability that can exist without model-weights updates. Memory and protocol changes are candidates; sustained behavioral improvement requires independent tests.

## 2. Three granularities of memory, with a fourth evaluation layer

| Layer | What to retain | Storage | Retrieval/use |
|---|---|---|---|
| **G0 raw trajectory** | Source events, full commands/tool traces, run IDs, timestamps, checkpoint/source hashes, user corrections, failed paths, raw results, invalidated hypotheses | Project-native append-only ledger / immutable raw artifacts | Evidence audit/reproduction; preserve negative runs without editing history |
| **G1 case / episode** | Local goal, actual behavior, mismatch, user correction, alternative action, source evidence, hypotheses, cost, validity limits, outcome | UKS failure catalog case; project short incident report | Reconstruct failure mechanisms; create benchmark cases |
| **G2 reusable skill** | General rule/operator, trigger, exception, stop condition, counterfactual reasoning, evidence standard, cross-domain examples | UKS Methods; RSI research OS references; agent operating policy | *Before* action selection on a semantically related task |
| **G3 evaluation / promotion** | Adversarial regression probes, control cases, pass/fail and false positives, tool traces, residual errors, drift/rollback records | Versioned benchmark suite + evaluation ledger | Decide whether a lesson deserves integration into future action policy |

**Distillation must never destroy G0.** Preserve causal uncertainty; G2 is a summary hypothesis, not a replacement for evidence. UKS is discoverable, source-native ledgers remain authoritative for project execution.

## 3. When should the agent capture an experience?

**Triggers with high learning value:** a direct user correction or repeated reframing; a significant failed assumption; surprising success/failure under a counterfactual; a repeated/expensive blocker; discovered failure of a verification gate; a high-cost near miss; conflicting evidence; a scientific negative result that changes next strategy; a benchmark/test that exposes systematic weakness; a cross-project pattern. Capture success cases too (e.g., an inexpensive diagnostic avoided unneeded training).

**Do not persist everything:** routine greetings and deterministic shell outputs, low-impact one-off slips without generalizable insight, duplicate near-identical failures, sensitive/confidential raw content absent necessity, unsupported causal claims, and unverified mechanisms masquerading as lessons. Preserve needed source pointers rather than copying logs into UKS.

**Ranking suggestion (decision aid, not a learned calibrated model):** prioritize by user correction/goal relevance × severity/risk × recurrence × generalizability × available evidence × expected future reuse, weighed against storage/retrieval cost. When multiple notes repeat the same mechanism, link them and update a canonical abstract skill rather than create contradictory directives.

## 4. Exact capture schema

```yaml
experience_id: EX-<project>-<date>-<sequence>
mission: {ultimate_goal: ..., success_criteria: ..., real_constraints: ...}
context: {source_of_truth: ..., branch_revision: ..., environment: ..., available_tools: ...}
trigger: {type: user_correction|failure|surprise|success|near_miss, quote_or_reference: ...}
trajectory:
  - {step: ..., agent_intent: ..., action: ..., observed_result: ..., cost: ..., source_ref: ...}
original_agent_belief: ...
user_intent_or_correction: ...
intent_delta: ...
observed_mismatch: ...
causal_hypotheses: [ {hypothesis: ..., confidence: low|medium|high, test: ...} ]
evidence_status: observed|reported|inferred|unknown
alternative_actions: [ {choice: ACT|ASK|VERIFY|CORRECT|REPLAN|STOP, when: ..., counterfactual_outcome: ...} ]
lesson: {pattern: ..., trigger: ..., mechanism: ..., actionable_rule: ..., exceptions: ..., stop_condition: ...}
benchmark: {positive_case: ..., negative_case: ..., hidden_variant: ..., pass_condition: ..., false_positive_guard: ...}
validation: {status: proposed|unit_tested|integration_tested|behaviorally_evaluated|promoted|retired, evidence_ref: ...}
links: {raw_ledger: ..., related_cases: [...], skills: [...], project_state: ...}
next_action: ...
```

Sources distinguish **user firsthand observation**, **tool-executed evidence**, **developer inference**, and **historical narrative**. If a correction is retroactive, do not rewrite the agent's earlier belief: append a correction event referencing it. Version the skill and keep superseded rules discoverable.

## 5. Post-mission reflective loop

1. **Reconstruct:** What exactly was the user attempting? What would true 'done' have looked like? Which assumption about their goal or the environment was active at each decision?
2. **Locate divergence:** Identify earliest avoidable wrong action, verification gap, tool-choice error, or missed user correction. Separate real external constraints from agent-caused misalignment.
3. **Explain and challenge:** Generate at least one alternative explanation; what observation would falsify the favored causal story? Do not equate chronology with causation.
4. **Counterfactual:** What smaller/different action could have prevented wasted effort? What if the user had *not* corrected the agent? What if a superficially similar future task really requires ASK instead of ACT?
5. **Distill:** Produce one retrieval-friendly failure pattern and one operational rule; list exceptions and proof boundary. Prefer an existing related UKS note over proliferation.
6. **Instrument:** Turn it into a regression test (or benchmark scenario), including confusing similar variants, a negative/positive control and measurable success criterion.
7. **Integrate:** Add to UKS Methods and project RSI operating policy only after versioning and consistency check; bind to relevant task triggers, not indiscriminately to every prompt.
8. **Validate in action:** Run the skill on held-out tasks; check whether the agent actually *chooses* better actions and uses tools differently. Record false positives and cost, not just adherence to new wording.
9. **Update/retire:** Keep the earlier evidence, merge redundant rules, supersede unhelpful instructions. High-value repeated failure can reopen a supposedly 'fixed' lesson.

**Trigger cadence:** lightweight reflection after a significant milestone/correction; deeper review at task completion, failed experiment, repeated pattern or weekly RSI governance review. Do not perform a costly full retrospective after every trivial exchange.

## 6. Learn from what, exactly? Five sources of supervision

- **Outcome supervision:** Did actual target performance improve, fail, regress or remain unknown? Was the evaluation valid?
- **Interaction supervision:** Did the user have to correct, repeat, verify manually, supply unnecessary inputs, or reject a proxy completion?
- **Process supervision:** Did the agent choose a good first action, verify source freshness, respect tool boundaries, avoid repeated blockers, and know when to stop?
- **Epistemic supervision:** Was confidence warranted? Were unavailable observations labeled unknown rather than guessed? Did the agent separate cause and correlation?
- **Cost supervision:** Avoidable delay, duplicated experiments, compute, tool calls, churn, data loss and user burden.

A correction often teaches **what objective was mis-specified**, not merely that the last answer was phrased badly. Preserve the intent delta and decision opportunity at which the correction should have been applied.

## 7. Incorporation into UKS and RSI (architecture)

**UKS role — memory/retrieval:** Store G1 concise cases and G2 reusable skills in `/Knowledge/05_Methods`; link them through the catalog and Knowledge Index. For task routing, retrieve by semantic triggers and known failure signatures, expand only when useful, and check provenance/freshness with the current live project. UKS must not silently assume old snapshots remain authoritative. Add a bounded retrieval check before an analogous mission, not blanket loading of the full history.

**RSI L0/L1:** Weights, rollouts, training losses, HPO and performance; these produce experience but do not alone prove meaningful competence.

**RSI L2:** Observation/action/reward/agent architecture; some failures require redesign rather than longer training.

**RSI L3:** Research experiment-selection policy: reconstruct failed assumption, inspect evidence, choose diagnosis vs retraining vs pivot. Keep falsification and a budget.

**RSI L4:** *Self-improve the researcher*. Analyze agent/user correction traces and governance failures; learn triggers for ACT/ASK/VERIFY/CORRECT/REPLAN/STOP; test whether changing retrieval, verification gates or work allocation improves downstream outcomes and reduces user rework. **Do not mistake a new long prompt or unit-tested protocol for proven learning.**

**Cross-project transfer:** Taisei real-game fidelity → MobileJail authentic app-security checks rather than judge labels; E3 class-recall vs native competence → FastBench executable baseline fidelity vs WAM claims; replay validity vs visible behavior → agent GUI execution/observation alignment; repeated GitHub/local environment confusion → any ephemeral vs persistent deployment.

## 8. Benchmark: can the agent generalize the lesson?

Use the companion Taisei case's T01–T14 probes and add cross-domain hidden variants. Evaluate **before/after** the skill or use a matched ablation:

- **Goal-faithfulness:** chooses actions directly serving ultimate user objective rather than easiest proxy.
- **Evidence calibration:** separates source test, offline metric, native behavior, visual inspection, and causal generalization.
- **Autonomy vs escalation:** acts when authorized and tools available; asks when credentials, irreversible changes, missing information, or unsafe execution really demand it.
- **Error recovery:** detects contradictory facts, rechecks live authority, bounds retries and avoids duplicated work.
- **Trajectory quality:** retains original error, correction time, reason for choosing next step, actual action and outcome.
- **Transfer:** recognizes same mechanism in a differently worded domain or unseen workflow.
- **Efficiency:** fewer avoidable actions/turns/user interventions, without sacrificing correctness or creating harmful over-action.

**Pass criterion proposal:** zero critical evidence fabrications or sealed-protocol violations; no unnecessary user replay delegation when a genuine capture tool is available; distinguish genuine blockers from fixable errors; improve action-quality outcomes on held-out cases without significant increase in harmful false-positive escalations or compute. Report per-case behavior, not only aggregate score. This is a proposed rubric, **not an observed benchmark result**.

## 9. Safety against learning the wrong lesson

- **Overfitting to corrections:** sometimes asking the human is optimal; never learn "never ask" from repeated requests for autonomy.
- **Wrong causal story:** user dissatisfaction is a signal to investigate, not proof of mechanism.
- **Memory contamination:** confidently worded old assistant claims must not override current Git or experiment artifacts.
- **Instruction inflation:** pruning, testability and transfer matter more than the number of explicit rules.
- **Experience laundering:** do not convert a planned tool call, passed unit test, or offline archive into a claimed authentic action.
- **Hindsight rewrite:** keep the *belief as held at the time*, the user correction and actual observed result; append revisions rather than editing failure away.
- **Privacy/provenance:** record minimal user-specific detail necessary; share artifacts/URLs only in authorized scopes.
- **Capacity limits:** ChatGPT schedules may start with a fresh sandbox; persistence to UKS/library/repo must be explicit and verified. A note existing is not evidence it is retrieved or obeyed on future turns.

## 10. Concrete first integration / next falsifiable test

**This integration (2026-10-28):** Record AIF-002 with an actual correction trajectory, add this skill to UKS and RSI research OS, link in the AI Failure Catalog and Knowledge Index, and prepare candidate benchmark probes. Mark the skill as *specified and stored*, not behaviorally validated.

**Next test:** replay or simulate a new work request in which convenient CI/offline progress conflicts with target real-world evidence; observe agent's first 3 decisions, if/when it acts/asks, its evidence labels and plan. Compare to baseline without the skill. Run an unrelated second project variant. If the agent still asks for needless manual verification, claims proxy success, or fails to retrieve the lesson, append a new G0 event and revise G2 after investigating the mechanism.

**One-line operating rule:** **Preserve the original trajectory → identify the earliest avoidable mismatch → distill a transferable, falsifiable lesson → retrieve it at the next relevant decision → test behavior, not merely memory.**
