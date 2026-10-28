#!/usr/bin/env python3
"""Bridge *historical* 528-feature movement BC weights to ONE skill-v2 live ABI.

This is a faithful movement-head embedding, NOT a trained skill policy or a
replacement renderer. New training must use native 546/18+3 SkillActorCritic.
Only trust .pt checkpoints from authorized sources: torch.load uses pickle.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

LEGACY_INPUTS = 528
CANONICAL_INPUTS = 546
ACTIONS = 18
SKILLS = 3
WIDTH = 128
MASK_VERSION = "taisei_skill_mask_v2_respawn"
ADAPTER_KIND = "legacy528_prefix_zero_pad_skill_none_v1"
# Bomb and Special must remain impossible even if the runtime samples skills.
# exp(-1e6) underflows to zero in float32; None has the maximum finite logit.
NEVER_SKILL_LOGIT = -1_000_000.0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _legacy_state(obj):
    import torch
    if not isinstance(obj, dict) or any(obj.get(k) != v for k, v in (
            ('policy_type', 'legacy'), ('input_dim', LEGACY_INPUTS),
            ('actions', ACTIONS), ('width', WIDTH))):
        raise ValueError('not an exact legacy 528-input ActorCritic checkpoint')
    state = obj.get('model')
    expected = {
        'base.0.weight': (128, 528), 'base.0.bias': (128,),
        'base.2.weight': (128, 128), 'base.2.bias': (128,),
        'actor.weight': (18, 128), 'actor.bias': (18,),
        'critic.weight': (1, 128), 'critic.bias': (1,),
    }
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('legacy checkpoint missing or has unexpected model tensors')
    for k, shape in expected.items():
        v = state[k]
        if (not isinstance(v, torch.Tensor) or tuple(v.shape) != shape or
                v.dtype != torch.float32 or not torch.isfinite(v).all()):
            raise ValueError(f'non-finite, wrong dtype or shape in {k}')
    return state


def _adapt_payload(checkpoint: Path, *, source_sha: str):
    import torch
    obj = torch.load(checkpoint, map_location='cpu', weights_only=False)
    state = _legacy_state(obj)
    new_state = {k: v.detach().cpu().clone().contiguous() for k, v in state.items()}
    new_state['base.0.weight'] = torch.cat([
        state['base.0.weight'].detach().cpu(),
        torch.zeros((WIDTH, CANONICAL_INPUTS - LEGACY_INPUTS), dtype=torch.float32)
    ], dim=1).contiguous()
    new_state['skill_actor.weight'] = torch.zeros((SKILLS, WIDTH), dtype=torch.float32)
    new_state['skill_actor.bias'] = torch.tensor([0.0, NEVER_SKILL_LOGIT, NEVER_SKILL_LOGIT], dtype=torch.float32)
    return {
        'policy_type': 'skill', 'input_dim': CANONICAL_INPUTS, 'actions': ACTIONS,
        'skills': SKILLS, 'width': WIDTH, 'skill_action_mask_version': MASK_VERSION,
        'model': new_state, 'adapter_kind': ADAPTER_KIND,
        'source_checkpoint_sha256': source_sha,
        'source_checkpoint_name': checkpoint.name,
        'adapter_warning': 'movement weights only; 18 new inputs ignored; skills always NONE; not retrained',
    }


def verify_adapter(original: Path, adapted: Path, *, trials: int = 16) -> dict:
    """Verify source lineage, exact parameter embedding and forward movement parity."""
    import numpy as np
    import torch
    original, adapted = Path(original), Path(adapted)
    orig_hash = sha256_file(original)
    o = torch.load(original, map_location='cpu', weights_only=False)
    old = _legacy_state(o)
    obj = torch.load(adapted, map_location='cpu', weights_only=False)
    if (not isinstance(obj, dict) or obj.get('source_checkpoint_sha256') != orig_hash
            or obj.get('adapter_kind') != ADAPTER_KIND or
            any(obj.get(k) != v for k, v in [('policy_type','skill'),('input_dim',546),
                                               ('actions',18),('skills',3),('width',128),
                                               ('skill_action_mask_version',MASK_VERSION)])):
        raise ValueError('adapted checkpoint provenance/contract mismatch')
    new = obj.get('model')
    if not isinstance(new,dict) or set(new) != set(old)|{'skill_actor.weight','skill_actor.bias'}:
        raise ValueError('adapted checkpoint state keys mismatch')
    if not torch.equal(new['base.0.weight'][:,:528], old['base.0.weight']) or not torch.count_nonzero(new['base.0.weight'][:,528:]) == 0:
        raise ValueError('legacy input prefix weights modified or 18 trailing inputs not zero')
    if not all(torch.equal(new[k], old[k]) for k in old if k != 'base.0.weight'):
        raise ValueError('legacy actor/hidden/critic parameters changed')
    if not torch.count_nonzero(new['skill_actor.weight']) == 0 or not torch.equal(
            new['skill_actor.bias'], torch.tensor([0, NEVER_SKILL_LOGIT, NEVER_SKILL_LOGIT], dtype=torch.float32)):
        raise ValueError('adapted skill head may activate skills')
    rng = np.random.default_rng(20261028)
    max_diff = 0.0
    for i in range(trials):
        x = (np.zeros(546, dtype=np.float32) if i==0 else
             np.linspace(-1, 1, 546, dtype=np.float32) if i==1 else
             rng.normal(0, 0.7, 546).astype(np.float32))
        with torch.no_grad():
            x528 = torch.from_numpy(x[:528].copy())
            x546 = torch.from_numpy(x)
            h_old = torch.tanh(torch.nn.functional.linear(x528,old['base.0.weight'],old['base.0.bias']))
            h_old = torch.tanh(torch.nn.functional.linear(h_old,old['base.2.weight'],old['base.2.bias']))
            logits_old = torch.nn.functional.linear(h_old,old['actor.weight'],old['actor.bias'])
            h_new = torch.tanh(torch.nn.functional.linear(x546,new['base.0.weight'],new['base.0.bias']))
            h_new = torch.tanh(torch.nn.functional.linear(h_new,new['base.2.weight'],new['base.2.bias']))
            logits_new = torch.nn.functional.linear(h_new,new['actor.weight'],new['actor.bias'])
            skills_new = torch.nn.functional.linear(h_new,new['skill_actor.weight'],new['skill_actor.bias'])
        diff = (logits_old-logits_new).abs().max().item()
        max_diff = max(max_diff,diff)
        if not torch.allclose(logits_old,logits_new,rtol=2e-5,atol=2e-5) or logits_old.argmax() != logits_new.argmax():
            raise ValueError('adapted movement deviates from original checkpoint')
        if skills_new.argmax().item() != 0 or not torch.equal(skills_new, new['skill_actor.bias']):
            raise ValueError('adapted skill head is not fixed to NONE')
    return {
        'source_checkpoint_sha256': orig_hash,
        'adapted_checkpoint_sha256': sha256_file(adapted),
        'adapter_kind': ADAPTER_KIND,
        'movement_parity_vectors': trials,
        'movement_max_logit_difference': max_diff,
        'skill_policy': 'always_NONE', 'live_interface': 'TSKILL2_546_18_3',
    }


def adapt(checkpoint: Path, destination: Path) -> dict:
    """Exclusive, idempotent creation. Never silently modify an adapted checkpoint."""
    import torch
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination).resolve()
    source_sha = sha256_file(checkpoint)
    if destination == checkpoint:
        raise ValueError('adapted output must differ from original checkpoint')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return {'status':'already_identical', 'path':str(destination), **verify_adapter(checkpoint,destination)}
    payload = _adapt_payload(checkpoint,source_sha=source_sha)
    with tempfile.TemporaryDirectory(prefix='.w2ctrl-legacy-',dir=destination.parent) as scratch:
        # Same basename as destination: avoids torch zip root variance on retries.
        temp = Path(scratch)/destination.name
        torch.save(payload,temp)
        verification = verify_adapter(checkpoint,temp)
        try:
            os.link(temp,destination)
            status='adapted'
        except FileExistsError:
            status='already_identical'
    return {'status':status, 'path':str(destination), **verify_adapter(checkpoint,destination)}
