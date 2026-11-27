#!/usr/bin/env python3
"""E7: DAgger distillation of the risk-aware teacher into the live TSKILL2 policy.

Why: E3/E4 behaviour cloning used ~12k teacher decisions from one or two teacher
episodes and the student collapsed (~1.3k alive frames vs the teacher's ~7.4k):
classic covariate shift. The teacher (research/rsi/risk_aware_teacher.py) is a pure
function of the observation, so it can label the states the *student* visits.

Algorithm (Ross et al. 2011, DAgger):
  iteration 0: the teacher drives; every state is labelled with the teacher action.
  iteration i: the executed action is the teacher's with prob beta_i = beta0**i,
               otherwise the student's (argmax); every state is labelled by the teacher.
  after each iteration the student is retrained on the aggregate dataset.

Interface: the live `rl_skill_v2` / TSKILL2 SkillActorCritic (546 inputs, 18 movement,
3 skill actions, width 128) saved with native_rl.torch_ppo.checkpoint, so
my_test/scripts/evaluate_native.py --no-skills evaluates it unchanged. The skill head
is trained towards NONE (skills are out of scope for P0, survival-first).

Task: Stage 1, Easy, Reimu A, one-frame actions, max 10k frames per episode.
Seeds: training episodes use seed, seed+1, ...; development evaluation seeds
42152-42167 and sealed 43000-43099 are refused.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "my_test"))
sys.path.insert(0, str(ROOT / "research/rsi"))

from native_rl.env import EnvConfig, TaiseiEnv  # noqa: E402
from native_rl.torch_ppo import (SkillActorCritic, checkpoint, skill_features,  # noqa: E402
                                 SKILL_ACTION_MASK_VERSION, SKILL_NONE)
from risk_aware_teacher import risk_aware_action  # noqa: E402

INPUT_DIM, MOVES, SKILLS, WIDTH = 546, 18, 3, 128
FORBIDDEN_SEEDS = set(range(42152, 42168)) | set(range(43000, 43100))


def student_action(model, x):
    with torch.no_grad():
        logits, _, _ = model(torch.as_tensor(x).unsqueeze(0))
    return int(logits.argmax(-1).item())


def collect(env, model, beta, n_steps, next_seed, rng):
    """Run episodes until n_steps decisions; returns arrays + rollout stats."""
    X, Y = [], []
    episodes, alive_frames, agree = [], 0, 0
    obs, _ = env.reset(seed=next_seed)
    seed = next_seed
    ep_alive = 0
    for _ in range(n_steps):
        x = skill_features(obs)
        if x.shape[0] != INPUT_DIM:
            raise RuntimeError(f"TSKILL2 feature width {x.shape[0]} != {INPUT_DIM}")
        a_teacher = risk_aware_action(obs)
        if beta >= 1.0 or rng.random() < beta:
            a_exec = a_teacher
        else:
            a_exec = student_action(model, x)
        agree += int(a_exec == a_teacher)
        X.append(x)
        Y.append(a_teacher)
        obs, _, term, trunc, info = env.step(a_exec)
        ep_alive += int(info.get("alive", obs["skill_state"][0] > 0))
        if term or trunc:
            episodes.append({"seed": seed, "alive_frames": ep_alive, "deaths": obs["deaths"],
                             "status": info["status"], "truncated": bool(trunc)})
            seed += 1
            if seed in FORBIDDEN_SEEDS:
                raise RuntimeError(f"training seed walked into a forbidden seed: {seed}")
            ep_alive = 0
            obs, _ = env.reset(seed=seed)
    # the open episode is discarded from the stats (not from the data)
    return (np.asarray(X, np.float32), np.asarray(Y, np.int64),
            {"episodes": episodes, "executed_teacher_agreement": agree / n_steps}, seed + 1)


def fit(model, opt, X, Y, epochs, batch, gen):
    model.train()
    Xt, Yt = torch.as_tensor(X), torch.as_tensor(Y)
    skill_target = torch.full((batch,), SKILL_NONE, dtype=torch.long)
    n = len(Xt)
    losses = []
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n - batch + 1, batch):
            idx = perm[i:i + batch]
            move_logits, skill_logits, _ = model(Xt[idx])
            loss = F.cross_entropy(move_logits, Yt[idx]) + 0.1 * F.cross_entropy(skill_logits, skill_target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
    model.eval()
    with torch.no_grad():
        acc = float((model(Xt[:200000])[0].argmax(-1) == Yt[:200000]).float().mean())
    return float(np.mean(losses[-50:])), acc


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--library", required=True)
    p.add_argument("--resource-path", required=True)
    p.add_argument("--seed", type=int, required=True, help="training seed (episodes use seed, seed+1, ...)")
    p.add_argument("--iterations", type=int, default=6)
    p.add_argument("--steps-per-iteration", type=int, default=25000)
    p.add_argument("--beta0", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", required=True, help="checkpoint .pt (must not exist)")
    p.add_argument("--log", required=True, help="JSONL log (must not exist)")
    a = p.parse_args(argv)
    if a.seed in FORBIDDEN_SEEDS:
        p.error("training seed is an evaluation or sealed seed")
    out, log = pathlib.Path(a.out), pathlib.Path(a.log)
    if out.exists() or log.exists():
        p.error("refusing to overwrite existing output")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = random.Random(a.seed)
    gen = torch.Generator().manual_seed(a.seed)
    model = SkillActorCritic(INPUT_DIM, MOVES, SKILLS, WIDTH)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    cfg = EnvConfig(library_path=a.library, resource_path=a.resource_path, stage_id=1,
                    difficulty=1, practice=False, max_frames=10000, frames_per_action=1)
    Xs, Ys = [], []
    next_seed = a.seed
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with TaiseiEnv(cfg) as env, open(log, "x", encoding="utf-8") as lf:
        for it in range(a.iterations):
            beta = 1.0 if it == 0 else a.beta0 ** it
            X, Y, stats, next_seed = collect(env, model, beta, a.steps_per_iteration, next_seed, rng)
            Xs.append(X)
            Ys.append(Y)
            loss, acc = fit(model, opt, np.concatenate(Xs), np.concatenate(Ys), a.epochs, a.batch, gen)
            rec = {"iteration": it, "beta": beta, "dataset": int(sum(len(y) for y in Ys)),
                   "train_loss": loss, "train_accuracy": acc, "elapsed_s": round(time.time() - t0, 1),
                   "rollout_completed_episodes": len(stats["episodes"]),
                   "rollout_mean_alive_frames": (float(np.mean([e["alive_frames"] for e in stats["episodes"]]))
                                                 if stats["episodes"] else None),
                   "executed_teacher_agreement": stats["executed_teacher_agreement"],
                   "episodes": stats["episodes"],
                   "teacher_action_histogram": np.bincount(Y, minlength=MOVES).tolist()}
            lf.write(json.dumps(rec) + "\n")
            lf.flush()
            print(json.dumps({k: v for k, v in rec.items() if k not in ("episodes", "teacher_action_histogram")}),
                  flush=True)
    checkpoint(model, opt, out, cfg, updates=a.iterations, steps=int(sum(len(y) for y in Ys)),
               training_state={"algorithm": "dagger", "teacher": "risk_aware_teacher.risk_aware_action",
                               "training_seed": a.seed, "iterations": a.iterations,
                               "steps_per_iteration": a.steps_per_iteration, "beta0": a.beta0,
                               "episode_seeds_last": next_seed - 1},
               skill_action_mask_version=SKILL_ACTION_MASK_VERSION)
    print(json.dumps({"checkpoint": str(out), "decisions": int(sum(len(y) for y in Ys)),
                      "elapsed_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
