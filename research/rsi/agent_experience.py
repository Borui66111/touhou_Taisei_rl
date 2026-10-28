#!/usr/bin/env python3
"""Evidence-scoped agent experience capture, retrieval and benchmark tracing.

Does NOT call an LLM, train weights, drive a native environment or certify agent
behavior. Scores observed decision records only; synthetic examples stay synthetic.
All JSONL edits are append-only and SHA-256 chained; no in-place promotion.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
try:
    import fcntl  # POSIX advisory locking
except ImportError:  # pragma: no cover - Windows
    fcntl = None
    import msvcrt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ACTIONS = frozenset({'ACT', 'ASK', 'VERIFY', 'CORRECT', 'REPLAN', 'STOP'})
KINDS = frozenset({'user_correction', 'failed_attempt', 'verified_success',
                   'diagnostic', 'lesson_candidate', 'benchmark_outcome', 'revision'})
EVENT_SCHEMA = 'agent_experience_v1'
TRACE_KINDS = frozenset({'synthetic_fixture', 'independent_agent_trace', 'self_reported_agent_trace'})
HEX64 = re.compile(r'^[0-9a-f]{64}$')
TOKEN = re.compile(r'[a-z0-9]+')


class EvidenceError(ValueError):
    """Fail closed on incomplete evidence, duplicate identity or broken lineage."""


def canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(obj: Any) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def read_jsonl(path: Path, *, required: bool = True) -> list[dict]:
    if not path.exists():
        if required:
            raise EvidenceError(f'missing file: {path}')
        return []
    records = []
    with path.open('r', encoding='utf-8') as stream:
        for line_num, line in enumerate(stream, 1):
            if not line.strip():
                raise EvidenceError(f'blank line at {path}:{line_num}')
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f'corrupt JSON at {path}:{line_num}') from exc
            if not isinstance(obj, dict):
                raise EvidenceError(f'non-object JSONL at {path}:{line_num}')
            records.append(obj)
    return records


def _timestamp(value: str) -> datetime:
    try:
        stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset() != timezone.utc.utcoffset(stamp):
            raise ValueError('not UTC')
        return stamp
    except (ValueError, AttributeError, TypeError) as exc:
        raise EvidenceError('timestamp must be a valid UTC ISO-8601 string') from exc


def _validate_event_data(event: dict) -> None:
    mandatory = {'event_id', 'timestamp_utc', 'project', 'kind', 'user_goal',
                 'agent_action', 'observed_result', 'mismatch', 'evidence_refs',
                 'interpretation', 'lesson_status', 'next_action'}
    if mandatory - event.keys():
        raise EvidenceError('missing event fields: ' + ', '.join(sorted(mandatory-event.keys())))
    for key in mandatory - {'evidence_refs'}:
        if not isinstance(event[key], str) or not event[key].strip():
            raise EvidenceError(f'empty/invalid {key}')
    if not re.fullmatch(r'EX-[A-Za-z0-9_-]{5,96}', event['event_id']):
        raise EvidenceError('invalid event_id')
    _timestamp(event['timestamp_utc'])
    if event['kind'] not in KINDS:
        raise EvidenceError('unrecognized event kind')
    if event['lesson_status'] not in {'candidate', 'proposed', 'unit_tested',
                                      'behaviorally_evaluated', 'rejected', 'retired'}:
        raise EvidenceError('invalid lesson status')
    refs = event['evidence_refs']
    if not isinstance(refs, list) or not refs or not all(isinstance(x, str) and x.strip() for x in refs):
        raise EvidenceError('evidence_refs must be nonempty reference strings')
    # No raw user secrets or large traces are meant to live here; retain pointers.
    if len(canonical(event)) > 32768:
        raise EvidenceError('event too large: store raw trace as a linked artifact')


def validate_ledger(records: list[dict]) -> dict:
    prev = None
    last_time = None
    ids = set()
    for index, entry in enumerate(records):
        if entry.get('schema') != EVENT_SCHEMA:
            raise EvidenceError(f'{index}: schema mismatch')
        if 'payload' not in entry or not isinstance(entry['payload'], dict):
            raise EvidenceError(f'{index}: missing event payload')
        payload = entry['payload']
        _validate_event_data(payload)
        if payload['event_id'] in ids:
            raise EvidenceError(f'{index}: duplicate event ID')
        ids.add(payload['event_id'])
        time = _timestamp(payload['timestamp_utc'])
        if last_time and time < last_time:
            raise EvidenceError(f'{index}: historical timestamp regressed')
        last_time = time
        if entry.get('prev_sha256') != prev:
            raise EvidenceError(f'{index}: chain continuity violated')
        if not isinstance(entry.get('sha256'), str) or not HEX64.fullmatch(entry['sha256']):
            raise EvidenceError(f'{index}: invalid chain digest')
        expected = digest({k: v for k, v in entry.items() if k != 'sha256'})
        if expected != entry['sha256']:
            raise EvidenceError(f'{index}: event has been modified or corrupted')
        prev = expected
    return {'status': 'valid', 'events': len(records), 'head_sha256': prev,
            'last_event_id': records[-1]['payload']['event_id'] if records else None}


def append_event(path: Path, payload: dict) -> dict:
    """Fail-closed append protected by a local POSIX/Windows advisory file lock."""
    _validate_event_data(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + '.lock')
    with lock_path.open('a+b') as lock:
        if fcntl is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        else:  # pragma: no cover - Windows
            # On Windows lock exactly one byte starting at offset 0.
            lock.seek(0, os.SEEK_END)
            if lock.tell() == 0:
                lock.write(b'\0'); lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        try:
            return _append_locked(path,payload)
        finally:
            if fcntl is None:  # pragma: no cover - Windows
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def _append_locked(path: Path, payload: dict) -> dict:
    records = read_jsonl(path, required=False)
    state = validate_ledger(records)
    if any(e['payload']['event_id'] == payload['event_id'] for e in records):
        raise EvidenceError('event ID already recorded; do not overwrite history')
    if records and _timestamp(payload['timestamp_utc']) < _timestamp(records[-1]['payload']['timestamp_utc']):
        raise EvidenceError('event time before latest event')
    entry = {'schema': EVENT_SCHEMA, 'prev_sha256': state['head_sha256'],
             'payload': payload}
    entry['sha256'] = digest(entry)
    # A partial final write fails closed on future reads; never silently repair.
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o640)
    try:
        data = canonical(entry) + b'\n'
        n = os.write(fd, data)
        if n != len(data):
            raise OSError('short append write; inspect and recover from original artifacts')
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'status': 'appended', 'event_id': payload['event_id'],
            'sha256': entry['sha256'], 'events': state['events']+1}

def validate_cases(records: list[dict]) -> dict:
    ids = set()
    for row in records:
        key = row.get('id')
        if not isinstance(key, str) or not re.fullmatch(r'T\d{2,4}', key) or key in ids:
            raise EvidenceError('missing, malformed or duplicate benchmark case ID')
        ids.add(key)
        if row.get('status') != 'proposed_unexecuted':
            raise EvidenceError(f'{key}: benchmark specification misrepresented as executed')
        if not isinstance(row.get('context'), str) or len(row['context']) < 15:
            raise EvidenceError(f'{key}: invalid context')
        expected = row.get('expected_decisions')
        if not isinstance(expected, list) or not expected or any(a not in ACTIONS for a in expected):
            raise EvidenceError(f'{key}: invalid expected decision set')
        if len(set(expected)) != len(expected):
            raise EvidenceError(f'{key}: duplicate expected decision')
        if not row.get('success_oracle') or not row.get('hard_fail_if'):
            raise EvidenceError(f'{key}: missing oracle/guardrail')
    if not records:
        raise EvidenceError('benchmark has no cases')
    return {'status': 'valid_specification_only', 'cases': len(ids),
            'cases_unexecuted': len(ids), 'needs_independent_agent_trials': True}


def load_registry(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or data.get('schema') != 'agent_lesson_registry_v1':
        raise EvidenceError('invalid lesson registry schema')
    lessons = data.get('lessons')
    if not isinstance(lessons, list) or not lessons:
        raise EvidenceError('lesson registry empty')
    keys = set()
    for lesson in lessons:
        key = lesson.get('id')
        if not isinstance(key, str) or key in keys:
            raise EvidenceError('duplicate/missing lesson ID')
        keys.add(key)
        for field in ('name','operational_rule','evidence_ref','status','exceptions'):
            if not isinstance(lesson.get(field), str) or not lesson[field]:
                raise EvidenceError(f'{key}: missing {field}')
        if lesson['status'] not in {'candidate','unit_tested','behaviorally_evaluated','retired'}:
            raise EvidenceError(f'{key}: invalid status')
        terms = lesson.get('triggers')
        if not isinstance(terms, list) or not terms or any(not isinstance(t,str) or not t.strip() for t in terms):
            raise EvidenceError(f'{key}: missing retrieval triggers')
    return lessons


def retrieve(task: str, lessons: list[dict], top_k: int = 3) -> dict:
    """Simple transparent lexical routing; no claim of semantic generalization."""
    words = set(TOKEN.findall(task.lower()))
    if top_k < 1 or top_k > 20:
        raise EvidenceError('top_k must be 1..20')
    matches = []
    for item in lessons:
        if item['status'] == 'retired':
            continue
        hits = []
        for trig in item['triggers']:
            bits = TOKEN.findall(trig.lower())
            if bits and set(bits).issubset(words):
                hits.append(trig)
        if hits:
            matches.append({'id':item['id'],'name':item['name'],
                            'hits':sorted(hits), 'hit_count':len(hits),
                            'rule':item['operational_rule'],
                            'exceptions':item['exceptions'],
                            'status':item['status'], 'evidence_ref':item['evidence_ref']})
    matches.sort(key=lambda x:(-x['hit_count'], x['id']))
    return {'matching_lessons':matches[:top_k], 'count':len(matches),
            'retrieval_method':'explicit_keyword_subsets_v1',
            'warning':'Candidate advice only; verify live project facts and exceptions. A lexical miss does not imply no relevant lesson.'}


def score_traces(cases: list[dict], traces: list[dict]) -> dict:
    """Instrumented action coverage only; cannot judge semantic correctness."""
    validate_cases(cases)
    by_case = {c['id']:c for c in cases}
    seen = set()
    rows = []
    for trace in traces:
        case_id = trace.get('case_id')
        if case_id not in by_case or case_id in seen:
            raise EvidenceError('unknown or duplicate case_id in trace')
        seen.add(case_id)
        kind = trace.get('source_kind')
        if kind not in TRACE_KINDS:
            raise EvidenceError(f'{case_id}: missing credible trace source_kind')
        if not isinstance(trace.get('run_id'), str) or not trace['run_id'].strip():
            raise EvidenceError(f'{case_id}: run_id required')
        actions = trace.get('actions')
        if not isinstance(actions,list) or not actions:
            raise EvidenceError(f'{case_id}: no observed/claimed actions')
        for a in actions:
            if not isinstance(a,dict) or a.get('decision') not in ACTIONS:
                raise EvidenceError(f'{case_id}: invalid decision action')
            if not isinstance(a.get('evidence_ref'),str) or not a['evidence_ref']:
                raise EvidenceError(f'{case_id}: action without tool/trace evidence reference')
            if not isinstance(a.get('observed'),bool):
                raise EvidenceError(f'{case_id}: observed must be boolean')
        if any(not a['observed'] for a in actions) and kind=='independent_agent_trace':
            raise EvidenceError(f'{case_id}: unobserved action claimed as independently traced')
        recorded_hard = trace.get('hard_failures',[])
        if not isinstance(recorded_hard,list) or not all(isinstance(x,str) for x in recorded_hard):
            raise EvidenceError(f'{case_id}: invalid hard_failures')
        found = set(a['decision'] for a in actions if a['observed'])
        required = set(by_case[case_id]['expected_decisions'])
        extra = found-required
        rows.append({'case_id':case_id,'source_kind':kind,
                     'observed_decisions':sorted(found),
                     'required_decisions':sorted(required),
                     'required_coverage':round(len(found&required)/len(required),4),
                     'extra_decisions':sorted(extra),
                     'hard_failures_reported':recorded_hard,
                     'oracle_semantics_grading':'NOT_PERFORMED',
                     'outcome_verified':False})
    independent = [r for r in rows if r['source_kind']=='independent_agent_trace']
    return {'status':'instrumentation_scored_only', 'benchmark_cases':len(cases),
            'traces_scored':len(rows),'cases_missing':len(cases)-len(rows),
            'synthetic_cases':sum(r['source_kind']=='synthetic_fixture' for r in rows),
            'independent_agent_traces':len(independent),
            'avg_structural_coverage':round(sum(r['required_coverage'] for r in rows)/len(rows),4) if rows else None,
            'agent_behavior_improvement_demonstrated':False,
            'reason':'Structural decision labels cannot prove real tool actions, scientific correctness, or before/after transfer; independent blind semantic grading and matched cohorts still required.',
            'per_case':rows}


def check_promotion(baseline: dict, treatment: dict) -> dict:
    """Strict evidence gate; no automatic integration/promotion occurs."""
    need = ['independent_agent_traces','synthetic_cases','traces_scored']
    if any(k not in baseline or k not in treatment for k in need):
        raise EvidenceError('missing benchmark report fields')
    reasons = []
    if baseline['synthetic_cases'] or treatment['synthetic_cases']:
        reasons.append('synthetic traces cannot establish agent improvement')
    if min(baseline['independent_agent_traces'],treatment['independent_agent_traces'])<5:
        reasons.append('need >=5 independent agent traces in each matched cohort')
    if baseline['traces_scored']!=treatment['traces_scored']:
        reasons.append('unmatched cohort sizes')
    # Even a perfect action-label trace is not an audited task outcome.
    reasons.append('independent blind semantic grading and matched real task outcomes missing from structural scorer')
    return {'lesson_promotion':'BLOCKED', 'reasons':reasons,
            'policy_automatically_modified':False}


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    cmds=p.add_subparsers(dest='command',required=True)
    for name in ('validate-benchmark','score'):
        s=cmds.add_parser(name);s.add_argument('--benchmark',type=Path,required=True)
        if name=='score':
            s.add_argument('--traces',type=Path,required=True)
    s=cmds.add_parser('validate-ledger');s.add_argument('--ledger',type=Path,required=True)
    s=cmds.add_parser('append');s.add_argument('--ledger',type=Path,required=True)
    s.add_argument('--event',type=Path,required=True)
    s=cmds.add_parser('retrieve');s.add_argument('--registry',type=Path,required=True)
    s.add_argument('--task',required=True);s.add_argument('--top-k',type=int,default=3)
    s=cmds.add_parser('gate');s.add_argument('--baseline',type=Path,required=True)
    s.add_argument('--treatment',type=Path,required=True)
    for c in cmds.choices.values():
        c.add_argument('--out',type=Path)
    args=p.parse_args(argv)
    try:
        if args.command=='validate-benchmark':
            out=validate_cases(read_jsonl(args.benchmark))
        elif args.command=='validate-ledger':
            out=validate_ledger(read_jsonl(args.ledger))
        elif args.command=='append':
            out=append_event(args.ledger,json.loads(args.event.read_text(encoding='utf-8')))
        elif args.command=='retrieve':
            out=retrieve(args.task,load_registry(args.registry),args.top_k)
        elif args.command=='score':
            out=score_traces(read_jsonl(args.benchmark),read_jsonl(args.traces))
        elif args.command=='gate':
            out=check_promotion(json.loads(args.baseline.read_text(encoding='utf-8')),
                                json.loads(args.treatment.read_text(encoding='utf-8')))
        else:
            raise AssertionError('unknown subcommand')
        rendered=json.dumps(out,indent=2,sort_keys=True,ensure_ascii=False)+'\n'
        if args.out:
            args.out.parent.mkdir(parents=True,exist_ok=True)
            if args.out.exists():
                raise EvidenceError('output already exists; version results rather than overwriting')
            with args.out.open('x',encoding='utf-8') as stream:
                stream.write(rendered)
        print(rendered,end='')
        return 0
    except (EvidenceError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({'status':'blocked','reason':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
