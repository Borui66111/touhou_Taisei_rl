"""Regression tests for storage, evidence boundaries and structural scoring only."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import tempfile
import unittest

from agent_experience import (EvidenceError, append_event, check_promotion,
    digest, load_registry, read_jsonl, retrieve, score_traces, validate_cases,
    validate_ledger)

HERE=Path(__file__).resolve().parent
CASES=HERE/'agent_failure_benchmark_v1.jsonl'
REGISTRY=HERE/'agent_lesson_registry_v1.json'


def good_event(id_='EX-TEST-20261028-1', time='2026-10-28T00:00:00Z'):
    return {'event_id':id_,'timestamp_utc':time,'project':'rsi_research',
            'kind':'user_correction','user_goal':'Actual native gameplay learning',
            'agent_action':'Only ran source tests','observed_result':'Source tests passed',
            'mismatch':'Goal still blocked',
            'evidence_refs':['https://github.com/example/repo/actions/runs/123'],
            'interpretation':'Infrastructure only; not gameplay',
            'lesson_status':'candidate','next_action':'Recover pinned source'}


def action(decision,observed=True):
    return {'decision':decision,'observed':observed,
            'evidence_ref':'fixture://test-only/record'}


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.d=tempfile.TemporaryDirectory()
        self.addCleanup(self.d.cleanup)
        self.path=Path(self.d.name)/'events.jsonl'

    def test_append_and_validate_chain(self):
        a=append_event(self.path,good_event())
        b=append_event(self.path,good_event('EX-TEST-20261028-2','2026-10-28T00:00:01Z'))
        self.assertEqual(a['events'],1)
        self.assertEqual(b['events'],2)
        self.assertEqual(validate_ledger(read_jsonl(self.path))['head_sha256'],b['sha256'])

    def test_duplicate_rejected(self):
        append_event(self.path,good_event())
        with self.assertRaisesRegex(EvidenceError,'already recorded'):
            append_event(self.path,good_event())

    def test_timestamp_regression_rejected(self):
        append_event(self.path,good_event())
        with self.assertRaisesRegex(EvidenceError,'before latest event'):
            append_event(self.path,good_event('EX-TEST-20261028-2','2026-10-27T23:59:59Z'))

    def test_tamper_detected(self):
        append_event(self.path,good_event())
        e=read_jsonl(self.path);e[0]['payload']['observed_result']='Native win'
        self.path.write_text(json.dumps(e[0])+'\n')
        with self.assertRaisesRegex(EvidenceError,'modified'):
            validate_ledger(read_jsonl(self.path))

    def test_broken_prev_hash_rejected(self):
        append_event(self.path,good_event())
        e=read_jsonl(self.path);e[0]['prev_sha256']='0'*64
        e[0]['sha256']=digest({k:v for k,v in e[0].items() if k!='sha256'})
        with self.assertRaisesRegex(EvidenceError,'continuity'):
            validate_ledger(e)

    def test_blank_line_rejected(self):
        self.path.write_text('\n')
        with self.assertRaisesRegex(EvidenceError,'blank line'):
            read_jsonl(self.path)

    def test_no_fabricated_no_evidence_event(self):
        event=good_event();event['evidence_refs']=[]
        with self.assertRaisesRegex(EvidenceError,'evidence_refs'):
            append_event(self.path,event)

    def test_event_larger_than_limit_rejected(self):
        event=good_event();event['agent_action']='a'*32768
        with self.assertRaisesRegex(EvidenceError,'too large'):
            append_event(self.path,event)

    def test_validate_16_proposed_scenarios(self):
        cases=read_jsonl(CASES)
        result=validate_cases(cases)
        self.assertEqual(result['cases_unexecuted'],16)
        self.assertTrue(result['needs_independent_agent_trials'])

    def test_no_false_benchmark_executed(self):
        cases=read_jsonl(CASES);cases[0]['status']='passed'
        with self.assertRaisesRegex(EvidenceError,'misrepresented'):
            validate_cases(cases)

    def test_duplicate_benchmark_scenario_rejected(self):
        cases=read_jsonl(CASES)+[read_jsonl(CASES)[0]]
        with self.assertRaisesRegex(EvidenceError,'duplicate'):
            validate_cases(cases)

    def test_registry_retrieves_analogy(self):
        lessons=load_registry(REGISTRY)
        result=retrieve('GUI judge SUCCESS label without execution evidence or video',lessons,top_k=3)
        self.assertIn('LESSON-EVIDENCE-TIERS',[e['id'] for e in result['matching_lessons']])
        self.assertEqual(result['retrieval_method'],'explicit_keyword_subsets_v1')

    def test_registry_retrieves_local_task(self):
        lessons=load_registry(REGISTRY)
        result=retrieve('local container missing source due to DNS; do not use GitHub Actions',lessons,top_k=4)
        self.assertIn('LESSON-LOCAL-RUNTIME',[e['id'] for e in result['matching_lessons']])

    def test_irrelevant_task_has_no_match(self):
        self.assertEqual(retrieve('What is the weather tomorrow?',load_registry(REGISTRY))['count'],0)

    def test_retired_lessons_not_returned(self):
        lessons=load_registry(REGISTRY)
        lessons[0]['status']='retired'
        result=retrieve('toy prototype',lessons)
        self.assertNotIn('LESSON-PROXY-EXIT',[e['id'] for e in result['matching_lessons']])

    def test_partial_benchmark_not_treated_as_complete(self):
        res=score_traces(read_jsonl(CASES),
                [{'case_id':'T02','source_kind':'synthetic_fixture','run_id':'test1',
                  'actions':[action('VERIFY')], 'hard_failures':[]}])
        self.assertEqual(res['cases_missing'],15)
        self.assertEqual(res['synthetic_cases'],1)
        self.assertFalse(res['agent_behavior_improvement_demonstrated'])
        self.assertFalse(res['per_case'][0]['outcome_verified'])

    def test_missing_action_evidence_rejected(self):
        a=action('VERIFY');a.pop('evidence_ref')
        with self.assertRaisesRegex(EvidenceError,'without tool/trace evidence'):
            score_traces(read_jsonl(CASES),
              [{'case_id':'T02','source_kind':'synthetic_fixture','run_id':'X',
                'actions':[a]}])

    def test_unobserved_independent_claim_rejected(self):
        with self.assertRaisesRegex(EvidenceError,'unobserved action'):
            score_traces(read_jsonl(CASES),
              [{'case_id':'T02','source_kind':'independent_agent_trace','run_id':'X',
                'actions':[action('VERIFY',False)]}])

    def test_duplicate_trace_rejected(self):
        t={'case_id':'T02','source_kind':'synthetic_fixture','run_id':'X','actions':[action('VERIFY')]}
        with self.assertRaisesRegex(EvidenceError,'duplicate case_id'):
            score_traces(read_jsonl(CASES),[t,t])

    def test_empty_replay_expected_behavior_is_verify_stop(self):
        cases={c['id']:c for c in read_jsonl(CASES)}
        self.assertEqual(set(cases['T04']['expected_decisions']),{'VERIFY','STOP'})

    def test_real_human_boundary_not_mislabeled_unnecessary_ask(self):
        cases={c['id']:c for c in read_jsonl(CASES)}
        self.assertIn('ASK',cases['T14']['expected_decisions'])

    def test_synthetic_perfect_trace_cannot_promote(self):
        cases=read_jsonl(CASES)
        t=[{'case_id':c['id'],'source_kind':'synthetic_fixture','run_id':'fixture',
            'actions':[action(d) for d in c['expected_decisions']]} for c in cases]
        report=score_traces(cases,t)
        self.assertEqual(report['avg_structural_coverage'],1.0)
        self.assertFalse(report['agent_behavior_improvement_demonstrated'])
        self.assertEqual(check_promotion(report,report)['lesson_promotion'],'BLOCKED')

    def test_no_model_run_is_not_zero_performance(self):
        res=score_traces(read_jsonl(CASES),[])
        self.assertIsNone(res['avg_structural_coverage'])
        self.assertEqual(res['independent_agent_traces'],0)

    def test_promotion_requires_semantic_blind_evaluation(self):
        fake={'independent_agent_traces':16,'synthetic_cases':0,'traces_scored':16}
        report=check_promotion(fake,fake)
        self.assertEqual(report['lesson_promotion'],'BLOCKED')
        self.assertIn('independent blind semantic grading',report['reasons'][-1])

if __name__=='__main__':unittest.main()
