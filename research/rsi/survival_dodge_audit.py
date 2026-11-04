#!/usr/bin/env python3
"""Analyze *real* native Taisei evaluation traces for boundary-pinning and hazard response.

This is a read-only measurement tool, not a controller, replay, simulator, or
proof that a particular dodge avoided a collision. No synthetic episodes enter
reported results. Requires per-action JSONL from evaluate_native.py --trace-out
with post-action player coordinates (see the paired instrumentation change).

Usage: python research/rsi/survival_dodge_audit.py trace.jsonl --out audit.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

WIDTH, HEIGHT = 480.0, 560.0
BOUNDS = (16.0, 464.0, 16.0, 544.0)  # pixel-space; source should be checked against pinned C build
DIRECTIONS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0),
              (-1, -1), (1, -1), (-1, 1), (1, 1))


class InvalidTrace(ValueError):
    pass


def _finite_number(x, label):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise InvalidTrace(f'{label} must be a finite number')
    return float(x)


def _norm_position(state, where):
    if not isinstance(state, dict):
        raise InvalidTrace(f'missing {where} state')
    x = _finite_number(state.get('player_x'), f'{where}.player_x') * WIDTH
    y = _finite_number(state.get('player_y'), f'{where}.player_y') * HEIGHT
    if not (-2.0 <= x <= WIDTH+2.0 and -2.0 <= y <= HEIGHT+2.0):
        raise InvalidTrace(f'implausible {where} normalized position: ({x}, {y})')
    return x, y


def _check_step(row):
    if not isinstance(row, dict):
        raise InvalidTrace('step must be a JSON object')
    if isinstance(row.get('seed'), bool) or not isinstance(row.get('seed'), int):
        raise InvalidTrace('missing integer game seed')
    if 43000 <= row['seed'] <= 43099:
        raise InvalidTrace('sealed final seed data are not allowed for RSI diagnostics')
    if isinstance(row.get('action_index'), bool) or not isinstance(row.get('action_index'), int) or row['action_index'] < 0:
        raise InvalidTrace('invalid action_index')
    a = row.get('movement_action')
    if isinstance(a, bool) or not isinstance(a, int) or not 0 <= a < 18:
        raise InvalidTrace('movement_action must be integer 0..17')
    if row.get('skill_action') not in (0, None):
        raise InvalidTrace('defensive movement audit must have skill None, not Bomb/Special')
    pre, post = row.get('pre'), row.get('post')
    px, py = _norm_position(pre, 'pre')
    nx, ny = _norm_position(post, 'post')
    for label, obj in [('pre', pre), ('post', post)]:
        if not isinstance(obj.get('alive'), bool):
            raise InvalidTrace(f'missing boolean {label}.alive')
        if isinstance(obj.get('deaths'), bool) or not isinstance(obj.get('deaths'), int) or obj['deaths'] < 0:
            raise InvalidTrace(f'invalid {label}.deaths')
    if post['deaths'] < pre['deaths']:
        raise InvalidTrace('death counter cannot decrease within step')
    if pre.get('invulnerable') is not None and not isinstance(pre['invulnerable'], bool):
        raise InvalidTrace('pre.invulnerable must be bool if present')
    nearest = pre.get('nearest_projectile')
    if nearest is not None and not isinstance(nearest, dict):
        raise InvalidTrace('nearest_projectile must be a dict or null')
    return px, py, nx, ny, a


def _threat_proxy(pre, *, clearance_px=18.0, horizon_frames=20.0):
    """Only a linear nearest-projectile threat proxy, NOT actual collision."""
    p = pre.get('nearest_projectile')
    if not p:
        return False
    d = _finite_number(p.get('predicted_closest_separation_px'), 'projectile closest separation')
    t = _finite_number(p.get('time_to_closest_frames'), 'projectile closest time')
    halfwidth = _finite_number(p.get('collision_width_px'), 'projectile collision_width_px')
    halfheight = _finite_number(p.get('collision_height_px'), 'projectile collision_height_px')
    if min(d,t,halfwidth,halfheight) < 0:
        raise InvalidTrace('negative projectile geometry')
    # Approximate effective clearance is conservative, and needs comparison
    # against native hit events; the solver is not a ground-truth oracle.
    return (t <= horizon_frames and d <= 6.0 + max(halfwidth, halfheight) + clearance_px)


def analyze(rows, *, edge_margin_px=12.0, min_progress_px=0.25):
    if not isinstance(rows, list) or not rows:
        raise InvalidTrace('empty trace; no actual actions to audit')
    if not (0 <= edge_margin_px < 48 and 0 <= min_progress_px <= 5):
        raise InvalidTrace('invalid wall/progress thresholds')
    counts = dict(total_steps=0, eligible_vulnerable_steps=0, boundary_steps=0,
                  outward_command_steps=0, outward_command_under_threat_steps=0,
                  linear_threat_steps=0, pinned_commands=0,
                  death_increments=0, skill_requested=0,
                  pre_vulnerable_decisions=0, pre_vulnerable_death_transitions=0,
                  pre_vulnerable_projectile_proxy_steps=0,
                  pre_vulnerable_projectile_proxy_death_transitions=0)
    per_seed = {}
    first_vulnerable_death_by_seed = {}
    distinct_actions = set()
    max_same_action = 0
    same_action = 0
    previous = None
    outward_streak = 0
    max_outward_streak = 0
    stall_streak = 0
    max_stall_streak = 0
    for row in rows:
        px, py, nx, ny, action = _check_step(row)
        seed = row['seed']
        seq = row['action_index']
        if seed not in per_seed:
            if seq != 0:
                raise InvalidTrace(f'first action for seed {seed} is not index 0')
            per_seed[seed] = -1
            previous = None
            same_action = outward_streak = stall_streak = 0
        if seq != per_seed[seed]+1:
            raise InvalidTrace(f'nonconsecutive action index for seed {seed}: {seq}')
        per_seed[seed] = seq
        if previous == (seed, action):
            same_action += 1
        else:
            same_action = 1
        previous = (seed, action)
        max_same_action = max(max_same_action, same_action)
        distinct_actions.add(action)
        counts['total_steps'] += 1
        pre, post = row['pre'], row['post']
        death_increment = post['deaths'] - pre['deaths']
        counts['death_increments'] += death_increment
        # Attribute fatal actions using the PRE-action vulnerable state. Conditioning
        # on post.alive/deaths here would silently discard the fatal decision.
        pre_vulnerable = pre['alive'] and not pre.get('invulnerable', False)
        if pre_vulnerable:
            counts['pre_vulnerable_decisions'] += 1
            danger_at_decision = _threat_proxy(pre)  # bullet proxy, NOT laser geometry
            counts['pre_vulnerable_projectile_proxy_steps'] += int(danger_at_decision)
            if death_increment:
                counts['pre_vulnerable_death_transitions'] += 1
                counts['pre_vulnerable_projectile_proxy_death_transitions'] += int(danger_at_decision)
                first_vulnerable_death_by_seed.setdefault(str(seed), seq)
        eligible = (pre['alive'] and post['alive'] and
                    not pre.get('invulnerable', False) and
                    pre['deaths'] == post['deaths'])
        if not eligible:
            outward_streak = stall_streak = 0
            continue
        counts['eligible_vulnerable_steps'] += 1
        dx, dy = DIRECTIONS[action % 9]
        left, right, top, bottom = BOUNDS
        near_left = px <= left+edge_margin_px
        near_right = px >= right-edge_margin_px
        near_top = py <= top+edge_margin_px
        near_bottom = py >= bottom-edge_margin_px
        boundary = near_left or near_right or near_top or near_bottom
        counts['boundary_steps'] += int(boundary)
        outward = ((near_left and dx < 0) or (near_right and dx > 0)
                   or (near_top and dy < 0) or (near_bottom and dy > 0))
        counts['outward_command_steps'] += int(outward)
        danger = _threat_proxy(pre)
        counts['linear_threat_steps'] += int(danger)
        counts['outward_command_under_threat_steps'] += int(outward and danger)
        if outward:
            outward_streak += 1
        else:
            outward_streak = 0
        max_outward_streak = max(max_outward_streak, outward_streak)
        # A stuck command is empirically defined as ineffective translation
        # along its requested travel vector, not simply remaining near the edge.
        length = math.hypot(dx,dy)
        projected = ((nx-px)*dx + (ny-py)*dy)/length if length else 0.0
        pinned = bool(outward and length and projected < min_progress_px)
        counts['pinned_commands'] += int(pinned)
        if pinned:
            stall_streak += 1
        else:
            stall_streak = 0
        max_stall_streak = max(max_stall_streak, stall_streak)
    eligible = counts['eligible_vulnerable_steps']
    pre_vulnerable_total = counts['pre_vulnerable_decisions']
    counts['fraction_pre_vulnerable_decisions_with_death_increment'] = (
        counts['pre_vulnerable_death_transitions'] / pre_vulnerable_total
        if pre_vulnerable_total else None)
    counts.update(
        distinct_movement_actions=len(distinct_actions),
        max_identical_movement_action_streak=max_same_action,
        max_outward_wall_streak=max_outward_streak,
        max_wall_pinned_streak=max_stall_streak,
        fraction_vulnerable_steps_near_boundary=(counts['boundary_steps']/eligible if eligible else None),
        fraction_vulnerable_steps_command_outward=(counts['outward_command_steps']/eligible if eligible else None),
        fraction_vulnerable_steps_pinned=(counts['pinned_commands']/eligible if eligible else None),
        fraction_threat_proxy_steps_command_outward=(counts['outward_command_under_threat_steps']/counts['linear_threat_steps']
                                                    if counts['linear_threat_steps'] else None),
    )
    return {
        'schema':'taisei_survival_dodge_wall_audit_v1',
        'classification':'read_only_native_trace_diagnostic_not_gameplay_improvement',
        'scope':'one-frame, no-skill, actual native evaluation traces',
        'game_seeds':sorted(per_seed),
        'first_vulnerable_death_action_index_by_seed':first_vulnerable_death_by_seed,
        'bounds_px':{'left':BOUNDS[0],'right':BOUNDS[1],'top':BOUNDS[2],'bottom':BOUNDS[3]},
        'edge_margin_px':edge_margin_px,'min_projected_progress_px':min_progress_px,
        'metrics':counts,
        'limits':['Boundary collision box must be checked in pinned native C source.',
                  'Linear projectile closest approach is a proxy, not verified collision risk.',
                  'Fatal decisions count toward pre-vulnerable death attribution, but not post-displacement/wall-pinning metrics because death/respawn may teleport the player.',
                  'Death increments do not identify bullet versus laser impact; laser geometry and damage source remain unavailable.',
                  'Per-seed trace continuity and real native game/library provenance need independent validation.',
                  'Logit/command or motion diversity alone is not a genuine dodge.',
                  'No replay/video or learned-model promotion is established by this audit.']
    }


def read_jsonl(path):
    out=[]
    with Path(path).open(encoding='utf-8') as f:
        for line_no, line in enumerate(f,1):
            if not line.strip():
                raise InvalidTrace(f'blank line at {line_no}')
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise InvalidTrace(f'bad JSON on line {line_no}') from e
    return out


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',help='real one-frame no-skill native trace JSONL')
    p.add_argument('--out', required=True, help='fresh JSON report, never overwrite')
    a=p.parse_args(argv)
    dest=Path(a.out)
    if dest.exists():p.error(f'output exists; refusing overwrite: {dest}')
    report=analyze(read_jsonl(a.trace))
    dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,sort_keys=True)
        f.write('\n')
    print(json.dumps(report,indent=2,sort_keys=True))


if __name__=='__main__':main()
