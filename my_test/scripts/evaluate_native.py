#!/usr/bin/env python3
"""Independent episode metrics from the actual native engine only."""
import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.env import EnvConfig
from native_rl.planner import heuristic_action
from native_rl.torch_ppo import evaluate, mean_alive_frames


def open_trace_file(trace_path, summary_path):
    """Open an opt-in JSONL trace exclusively, rejecting summary collisions."""
    trace_path = pathlib.Path(trace_path)
    summary_path = pathlib.Path(summary_path)
    if trace_path.resolve() == summary_path.resolve():
        raise ValueError('--trace-out must differ from --out')
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    return trace_path.open('x', encoding='utf-8')


def write_trace_row(trace_file, row):
    trace_file.write(json.dumps(row) + '\n')
    trace_file.flush()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--resource-path')
    parser.add_argument('--stage', type=int, default=1)
    parser.add_argument('--difficulty', type=int, default=2)
    parser.add_argument('--practice', action='store_true',
                        help='Enable spell-practice mode (default: off)')
    parser.add_argument('--max-frames', type=int, default=10000)
    parser.add_argument('--model')
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--sampling-seed', type=int,
                        help='Seed Torch action sampling for sampled PPO evaluation')
    parser.add_argument('--policy', choices=['ppo', 'heuristic', 'random'], default='ppo')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=91000)
    parser.add_argument('--out', default='results/native_evaluation.jsonl')
    parser.add_argument('--trace-out', help='Optional exclusive per-action PPO JSONL trace')
    args = parser.parse_args(argv)
    if args.policy == 'ppo' and not args.model:
        parser.error('--model required for PPO')
    if args.sampling_seed is not None and (args.policy != 'ppo' or not args.sampled):
        parser.error('--sampling-seed requires --policy ppo and --sampled')

    cfg = EnvConfig(library_path=args.library, resource_path=args.resource_path,
                    stage_id=args.stage, difficulty=args.difficulty,
                    practice=args.practice, max_frames=args.max_frames)
    if args.policy == 'random':
        rng = random.Random(args.seed)
        plan = lambda _: rng.randrange(18)
    elif args.policy == 'heuristic':
        plan = heuristic_action
    else:
        plan = None

    def summarize(rows):
        summary = {
            'policy': args.policy, 'episodes': len(rows),
            'win_rate': sum(int(x['won']) for x in rows) / len(rows),
            'clear_rate': sum(int(x['cleared']) for x in rows) / len(rows),
            'truncation_rate': sum(int(x['truncated']) for x in rows) / len(rows),
            'mean_boss_damage': sum(x['boss_damage'] for x in rows) / len(rows),
            'mean_reward': sum(x['reward'] for x in rows) / len(rows),
            'mean_graze': sum(x['graze'] for x in rows) / len(rows),
            'mean_deaths': sum(x['deaths'] for x in rows) / len(rows),
            'mean_frames': sum(x['frames'] for x in rows) / len(rows),
            'mean_survival_frames': mean_alive_frames(rows),
            'mean_score': sum(x['score'] for x in rows) / len(rows),
            'mean_point_item_value': sum(x['point_item_value'] for x in rows) / len(rows),
            'mean_bombs_used': sum(x['bombs_used'] for x in rows) / len(rows),
            'mean_logical_frame': sum(x['logical_frame'] for x in rows) / len(rows),
        }
        print(json.dumps(summary, indent=2))

    def write_reports(rows):
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as report_file:
            for row in rows:
                report_file.write(json.dumps(row) + '\n')
        summarize(rows)

    if args.trace_out:
        with open_trace_file(args.trace_out, args.out) as trace_file:
            def record_trace(row):
                write_trace_row(trace_file, row)
            rows = evaluate(cfg, args.model, episodes=args.episodes, seed=args.seed,
                            sampled=args.sampled, planner=plan,
                            trace_callback=record_trace,
                            sampling_seed=args.sampling_seed)
    else:
        rows = evaluate(cfg, args.model, episodes=args.episodes, seed=args.seed,
                        sampled=args.sampled, planner=plan,
                        sampling_seed=args.sampling_seed)
    write_reports(rows)


if __name__ == '__main__':
    main()
