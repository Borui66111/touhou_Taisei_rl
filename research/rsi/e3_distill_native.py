#!/usr/bin/env python3
"""E3: teacher imitation -> PPO fine-tuning on *genuine* native Taisei.

Fixed pilot: 12k teacher-controlled native decisions + 8k PPO decisions.
This is no guarantee of performance, no toy fallback, no skill/Bomb actions,
and no automatic champion promotion. Legacy 528-input movement-only policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"my_test"))
import numpy as np
import torch
from torch import nn
from native_rl.env import EnvConfig,TaiseiEnv
from native_rl.torch_ppo import ActorCritic,features,train
from risk_aware_teacher import risk_aware_action

DEMO_STEPS=12000
PPO_STEPS=8000
TOTAL_NATIVE_BUDGET=20000
BC_TRAINING_SEED=48030
PPO_TRAINING_SEED=48020
N_ACTIONS=18
N_FEATURES=528


def checked_plan(demo_steps,ppo_steps,difficulty,stage,demo_seed,ppo_seed):
    if (demo_steps,ppo_steps)!=(DEMO_STEPS,PPO_STEPS):
        raise ValueError("E3 preregistered pilot must allocate 12k demo + 8k PPO decisions")
    if demo_steps+ppo_steps>TOTAL_NATIVE_BUDGET:
        raise ValueError("E3 candidate exceeds native interaction budget")
    if (stage,difficulty)!=(1,1):
        raise ValueError("E3 task is actual Taisei Stage1 Easy")
    if (demo_seed,ppo_seed)!=(BC_TRAINING_SEED,PPO_TRAINING_SEED):
        raise ValueError("E3 training seeds fixed by preregistration")
    if 43000<=demo_seed<=43099 or 43000<=ppo_seed<=43099:
        raise ValueError("Sealed final seeds forbidden")
    return {"experiment":"E3_NATIVE_TEACHER_DISTILL_V1",
            "native_interactions_max":demo_steps+ppo_steps,
            "demonstrations":demo_steps,"ppo_decisions":ppo_steps,
            "demo_seed":demo_seed,"ppo_seed":ppo_seed,
            "stage_id":stage,"difficulty_id":difficulty,
            "action_space":"legacy 18 movement actions; no Bomb/Special",
            "model_input_dim":N_FEATURES,
            "replay_final_seeds":"43000-43099 sealed",
            "hypothesis":"BC initialization from an effective native risk-aware teacher improves the resulting PPO policy's actual survival without consumable spamming",
            "limitations":"One teacher dataset, one training initialization, 16 development game seeds; no generalization or auto promotion"}


def collect_demonstrations(cfg,steps,seed):
    state_vectors=[]
    labels=[]
    episodes=0
    with TaiseiEnv(cfg) as env:
        obs,_=env.reset(seed=seed)
        for _ in range(steps):
            x=features(obs)
            if x.shape!=(N_FEATURES,) or not np.isfinite(x).all():
                raise ValueError("Unexpected or nonfinite native observation feature vector")
            a=int(risk_aware_action(obs))
            state_vectors.append(x.copy())
            labels.append(a)
            obs,_,terminated,truncated,_=env.step(a)
            if terminated or truncated:
                episodes+=1
                obs,_=env.reset(seed=seed+episodes)
    return np.stack(state_vectors),np.asarray(labels,dtype=np.int64),episodes


def bc_pretrain(states,actions,seed=48030,epochs=16,batch_size=256):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=ActorCritic(N_FEATURES,actions=N_ACTIONS,width=128)
    optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    x=torch.from_numpy(states.astype(np.float32,copy=False))
    labels=torch.from_numpy(actions.astype(np.int64,copy=False))
    n=len(actions)
    rng=np.random.default_rng(seed)
    perm=rng.permutation(n)
    train_indices=perm[:int(.9*n)]
    validation_indices=perm[int(.9*n):]
    history=[]
    for epoch in range(epochs):
        shuffled=rng.permutation(train_indices)
        model.train()
        weighted_loss=0.0
        for batch in range(0,len(shuffled),batch_size):
            ids=shuffled[batch:batch+batch_size]
            logits,_=model(x[ids])
            loss=nn.functional.cross_entropy(logits,labels[ids])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite BC loss")
            optim.zero_grad(set_to_none=True);loss.backward()
            optim.step()
            weighted_loss+=float(loss.item())*len(ids)
        model.eval()
        with torch.no_grad():
            val_logits,_=model(x[validation_indices])
            accuracy=float((val_logits.argmax(1)==labels[validation_indices]).float().mean().item())
        history.append({"epoch":epoch+1,"train_loss":weighted_loss/len(train_indices),
                        "random_frame_holdout_accuracy":accuracy})
    return model,history


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--library",required=True)
    p.add_argument("--resource-path",required=True)
    p.add_argument("--outdir",required=True)
    p.add_argument("--preflight-only",action="store_true")
    p.add_argument("--demo-steps",type=int,default=DEMO_STEPS)
    p.add_argument("--ppo-steps",type=int,default=PPO_STEPS)
    p.add_argument("--demo-seed",type=int,default=BC_TRAINING_SEED)
    p.add_argument("--ppo-seed",type=int,default=PPO_TRAINING_SEED)
    p.add_argument("--stage",type=int,default=1)
    p.add_argument("--difficulty",type=int,default=1)
    a=p.parse_args(argv)
    plan=checked_plan(a.demo_steps,a.ppo_steps,a.difficulty,a.stage,a.demo_seed,a.ppo_seed)
    plan.update({"code_sha":os.environ.get("GITHUB_SHA","unknown"),
                 "status":"preflight_only" if a.preflight_only else "executing_native"})
    print(json.dumps(plan,indent=2),flush=True)
    if a.preflight_only:return
    if not Path(a.library).is_file() or not Path(a.resource_path).exists():
        raise FileNotFoundError("Authentic native simulator/resources required (no toy fallback)")
    out=Path(a.outdir)
    out.mkdir(parents=True,exist_ok=True)
    dataset=out/"teacher_demonstrations_12000.npz"
    bc_model=out/"teacher_bc_seed48030.pt"
    ppo_model=out/"teacher_bc_ppo_seed48020.pt"
    log=out/"ppo_finetune.jsonl"
    meta=out/"e3_manifest.json"
    bc_log=out/"bc_history.jsonl"
    if any(path.exists() for path in (dataset,bc_model,ppo_model,meta,bc_log,log)):
        raise FileExistsError("Experiment evidence exists: never silently overwrite")
    cfg=EnvConfig(library_path=a.library,resource_path=a.resource_path,
                  stage_id=1,difficulty=1,practice=False,
                  max_frames=10000,frames_per_action=1)
    states,labels,n_complete=collect_demonstrations(cfg,a.demo_steps,a.demo_seed)
    np.savez_compressed(dataset,states=states,actions=labels)
    plan.update({"teacher_demo_completed_episodes":n_complete,
                 "demo_action_frequencies":np.bincount(labels,minlength=N_ACTIONS).tolist(),
                 "dataset_sha256":hashlib.sha256(dataset.read_bytes()).hexdigest()})
    model,history=bc_pretrain(states,labels,seed=a.demo_seed)
    with bc_log.open("x",encoding="utf-8") as f:
        for row in history:f.write(json.dumps(row)+"\n")
    payload={"model":model.state_dict(),"input_dim":N_FEATURES,"actions":N_ACTIONS,
             "width":128,"policy_type":"legacy","source":"E3_teacher_behavior_cloning",
             "native_demo_steps":DEMO_STEPS,"demo_seed":a.demo_seed}
    torch.save(payload,bc_model)
    plan.update({"bc_checkpoint_sha256":hashlib.sha256(bc_model.read_bytes()).hexdigest(),
                 "bc_final_train_loss":history[-1]["train_loss"],
                 "bc_final_frame_holdout_accuracy":history[-1]["random_frame_holdout_accuracy"],
                 "status":"bc_completed_ppo_pending"})
    meta.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    train(cfg,outfile=str(ppo_model),total_steps=a.ppo_steps,rollout_steps=512,
          learning_rate=3e-4,seed=a.ppo_seed,device="cpu",gamma=.995,
          policy_type="legacy",skill_action_mask=False,init=str(bc_model),
          log_path=str(log))
    plan.update({"status":"bc_and_ppo_completed_unpromoted",
                 "ppo_checkpoint_sha256":hashlib.sha256(ppo_model.read_bytes()).hexdigest()})
    meta.write_text(json.dumps(plan,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":plan["status"],"native_budget":a.demo_steps+a.ppo_steps,
                      "bc_checkpoint_sha256":plan["bc_checkpoint_sha256"],
                      "ppo_checkpoint_sha256":plan["ppo_checkpoint_sha256"]}),flush=True)


if __name__=="__main__":
    main()
