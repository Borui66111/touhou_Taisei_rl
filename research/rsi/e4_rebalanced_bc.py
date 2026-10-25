#!/usr/bin/env python3
"""E4: frequency-aware imitation on E3 authentic Taisei demonstrations.

No environment steps or fabricated physics. Source input MUST match the E3
artifact SHA; same 528-dimensional legacy actor and 18 movement actions.
The frame-level split is temporally correlated, so offline validation is
diagnostic only; native matched-seed evaluation is indispensable.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"my_test"))
from native_rl.torch_ppo import ActorCritic

DATASET_SHA = "02dd11feb60e0dcb5710ae8af3cf33dca7e7e77756d6467f822ad94d0f5a0d8d"
BASE_BC_SHA = "87b43918d0181355435f105e3fcec44d7fd314c12aec6978b63ec33e9955cf54"
SEED=48030
EPOCHS=16
BATCH=256
INPUT_DIM=528
N_ACTIONS=18


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_weights(train_labels):
    labels=np.asarray(train_labels)
    if labels.ndim != 1 or labels.size==0 or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError("integer action labels required")
    if not np.all((0<=labels)&(labels<N_ACTIONS)):
        raise ValueError("action index outside 0..17")
    counts=np.bincount(labels,minlength=N_ACTIONS)
    n_active=int((counts>0).sum())
    w=np.zeros(N_ACTIONS,dtype=np.float32)
    w[counts>0]=np.sqrt(len(labels)/(n_active*counts[counts>0]))
    w=np.clip(w,0.5,3.0)
    w/=np.sum(counts*w)/len(labels)
    return w,counts


def diagnostics(model,x,labels,indices):
    model.eval()
    with torch.no_grad():
        logits,_=model(x[indices])
        pred=logits.argmax(dim=1).cpu().numpy()
    target=labels[indices]
    ok=pred==target
    counts=np.bincount(target,minlength=N_ACTIONS)
    hits=np.bincount(target[ok],minlength=N_ACTIONS)
    return {
        "frame_accuracy":float(ok.mean()),
        "always_stay_accuracy":float((target==0).mean()),
        "stay_recall":float(ok[target==0].mean()) if np.any(target==0) else None,
        "nonstay_accuracy":float(ok[target!=0].mean()) if np.any(target!=0) else None,
        "macro_recall_supported_labels":float(np.mean(hits[counts>0]/counts[counts>0])),
        "per_action":[{"action":i,"support":int(counts[i]),
                       "recall":float(hits[i]/counts[i]) if counts[i] else None}
                      for i in range(N_ACTIONS)],
    }


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True)
    p.add_argument("--baseline-bc",required=True)
    p.add_argument("--outdir",required=True)
    a=p.parse_args(argv)
    if checksum(a.dataset)!=DATASET_SHA or checksum(a.baseline_bc)!=BASE_BC_SHA:
        p.error("E3 native demonstration data or checkpoint checksum mismatch")
    out=Path(a.outdir)
    out.mkdir(parents=True,exist_ok=True)
    output=out/"teacher_bc_rebalanced_seed48030.pt"
    report=out/"e4_offline_bc_report.json"
    if output.exists() or report.exists():
        p.error("Refusing to overwrite existing candidate/evidence")
    z=np.load(a.dataset,allow_pickle=False)
    states=z["states"]
    labels=z["actions"]
    if states.shape!=(12000,INPUT_DIM) or labels.shape!=(12000,) or not np.isfinite(states).all():
        p.error("Unexpected genuine E3 data dimensions or nonfinite input")
    if np.any((labels<0)|(labels>=N_ACTIONS)):
        p.error("Label outside 18 movement actions")
    torch.set_num_threads(min(torch.get_num_threads(),2))
    torch.manual_seed(SEED);random.seed(SEED);np.random.seed(SEED)
    rng=np.random.default_rng(SEED)
    perm=rng.permutation(len(labels))
    tr=perm[:int(.9*len(perm))]
    val=perm[int(.9*len(perm)):]
    class_weights,counts=make_weights(labels[tr])
    model=ActorCritic(INPUT_DIM,actions=N_ACTIONS,width=128)
    optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    x=torch.from_numpy(states.astype(np.float32,copy=False))
    y=torch.from_numpy(labels.astype(np.int64,copy=False))
    w=torch.from_numpy(class_weights)
    history=[]
    for epoch in range(EPOCHS):
        order=rng.permutation(tr)
        model.train()
        total=0.
        for k in range(0,len(order),BATCH):
            ix=order[k:k+BATCH]
            logits,_=model(x[ix])
            per=nn.functional.cross_entropy(logits,y[ix],reduction="none")
            # Fixed before testing; retains a 50% unweighted objective to
            # discourage rare-movement overfitting.
            loss=(per*(.5+.5*w[y[ix]])).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite supervised imitation loss")
            optim.zero_grad(set_to_none=True);loss.backward();optim.step()
            total+=float(loss.detach())*len(ix)
        row={"epoch":epoch+1,"weighted_loss":total/len(tr),
             **{k:v for k,v in diagnostics(model,x,labels,val).items()
                if k!="per_action"}}
        history.append(row)
    old_payload=torch.load(a.baseline_bc,map_location="cpu",weights_only=True)
    old=ActorCritic(INPUT_DIM,actions=N_ACTIONS,width=128)
    old.load_state_dict(old_payload["model"])
    base_diag=diagnostics(old,x,labels,val)
    new_diag=diagnostics(model,x,labels,val)
    payload={"model":model.state_dict(),"input_dim":INPUT_DIM,"actions":N_ACTIONS,
             "width":128,"policy_type":"legacy","source":"E4_class_rebalanced_BC",
             "dataset_sha256":DATASET_SHA,"baseline_bc_sha256":BASE_BC_SHA,
             "teacher_demo_steps":12000,"training_seed":SEED,
             "training_epochs":EPOCHS}
    torch.save(payload,output)
    result={"experiment":"E4_NATIVE_DATASET_CLASS_AWARE_IMITATION",
            "status":"offline_learner_created_native_gameplay_not_yet_tested",
            "data_sha256":DATASET_SHA,"baseline_bc_sha256":BASE_BC_SHA,
            "candidate_sha256":checksum(output),
            "teacher_game_interactions_this_run":0,
            "training_frame_count":len(tr),"holdout_frame_count":len(val),
            "loss":"0.5 * unweighted CE + 0.5 * normalized capped sqrt-inverse-class-frequency CE",
            "seed":SEED,"epochs":EPOCHS,"train_class_counts":counts.tolist(),
            "class_weights":class_weights.tolist(),
            "baseline_holdout":base_diag,"candidate_holdout":new_diag,
            "history":history,
            "comparison_is_training_distribution_only":True,
            "limitations":["Random-frame split from ~1.2 real teacher episodes is temporally correlated.",
                           "Class-weighted accuracy/macro recall is not native survival competence.",
                           "The weighting rule is exploratory and may degrade genuine gameplay.",
                           "Requires 16 paired native development-game evaluations before any neural improvement claim."],
            "promote_policy":False}
    report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","candidate_sha256",
              "baseline_holdout","candidate_holdout")},indent=2),flush=True)


if __name__=="__main__":
    main()
