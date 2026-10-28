# Agent Meta-RSI — lesson retrieval v2 experimental record

**Date:** 2026-10-28 (Asia/Singapore)  
**Category:** L4 agent decision tooling; retrieval only, **not** observed agent learning  
**Canonical branch:** `research/rsi-automation-v1`  
**Source package:** UKS Library `/Knowledge/05_Methods/Agent_Experience_Runtime_2026-10-28/v2/Agent_Meta_RSI_Retrieval_v2_20261028.zip`  
**Package SHA-256:** `a2c2f982b3a8abd44cc032cf86279ca59975f159185669c0e508385119779b6d`

## Falsifiable hypothesis

Literal single-word lesson triggers both miss analogous failures and trigger irrelevant lessons for unrelated educational questions. A task-intent gate and relation-bearing evidence cues should improve relevance on scenarios not inspected while implementing the replacement.

## Execution order / provenance

1. Reran old 24 source tests (all passed); verified original 16 benchmark records remain unexecuted proposals.
2. Created a 24-case diagnostic corpus (12 operational positives, 12 educational negative controls; SHA-256 `069208c2cea606d267d8149eef8d93bde77e352daffe391480853368aeef4b15`).
3. Tested original lexical routing. It retrieved a relevant lesson in 3/12 positive cases, and wrongly triggered on 8/12 controls.
4. Developed conservative `meta_router_v2.py` using these development cases; on this **seen** set it obtained 12/12 positive hits, 0/12 control false triggers. These numbers are not out-of-sample evidence.
5. Wrote and hashed a separate **prospective** 16-case set (8 positives, 8 controls), **before inspecting** v2 predictions. Corpus SHA-256 `1c9b7c0e833d983bfb77aee35f49d1e8315d90edbf8c9cfcc3408042df5b9528`; router SHA-256 `ae55cdf50a7d993326a74287e92a28ee053155931d0cac0a05f24cdb2b2e1b38`; evaluation script SHA-256 `a7bf2430852ba356a50c76d585f11ae8ddffea2164501839921f42e9463256d1`; registry SHA-256 `23afca95011991661b2d7b9b114668c64af49456d1095863508d69f776a428fd`.
6. Executed both routers **once** on prospective set. Ran 37/37 local Python source-level tests. ZIP was re-extracted in an independent directory and the SHA-256 manifest and 37 tests passed.

| Prospectively authored 16-case comparison | Original keyword subsets | Relation gate v2 |
|---|---:|---:|
| Positive cases with ≥1 relevant lesson | 4/8 | **8/8** |
| Negative controls correctly rejected | 6/8 | **7/8** |
| Lesson-label micro recall | 33.3% | **91.7%** |
| Lesson-label micro precision | 57.1% | **64.7%** |
| Micro F1 | 0.421 | **0.759** |

The new router still returns extra unrelated lessons and fires on one classroom statistics question. Both corpus labels and router were authored by the same experimenter; no independent evaluator or agent execution cohort was involved. **The prospective set is now spent** and cannot be reused as a new unbiased holdout following further tuning.

## Evidence limits

- Improved **retrieval routing**, not live agent actions, calibrated behavior, research competence, or Taisei game performance.
- No controlled agent before/after cohort, real tool-trace grader, user correction reduction, model-weight update, native simulation or video inspection occurred in this experiment.
- Statistical confidence is low at 8 positive scenarios; do not promote rules solely on this test.
- Avoid converting source unit tests or self-selected example improvements into claims of agent learning.

## Next executable experiment

Collect actual tool-observed agent decisions on matched, preregistered cross-project tasks with and without lesson retrieval, including cases where ASK or STOP is correct. Use immutable source/run IDs, independently adjudicated task outcomes, actual tool-call provenance, user correction burden and cost. If no independent agent runner is available, build only the next small instrument that addresses a concrete missing piece and explicitly report why promotion stays blocked. For router precision tuning, prepare a **new** development set and a **fresh** prospective challenge suite before final evaluation.

Task recurrence is authorized through the separate daily **Agent Meta-RSI Experiments** automation; native Taisei research remains a distinct enabled local-first automation. A scheduled ChatGPT task is not an always-on local process.
