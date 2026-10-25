"""Standalone PyTorch PPO for the authentic native Taisei environment.

No toy physics, no Gymnasium or SB3 dependency. Full episodes must come from
TaiseiEnv and its native C backend. PPO numerical tests are not game validation.
"""
from __future__ import annotations
from dataclasses import asdict
import json
import math
import os
import pathlib
import random
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam
from .env import EnvConfig, TaiseiEnv

SKILL_ACTION_MASK_VERSION = 'taisei_skill_mask_v2_respawn'
SKILL_NONE, SKILL_BOMB, SKILL_SPECIAL = 0, 1, 2
SKILL_BOMB_STOCK_INDEX = 3
SKILL_STORED_POWER_INDEX = 5
SKILL_BOMB_ACTIVE_INDEX = 7
SKILL_SURGE_ACTIVE_INDEX = 8
SKILL_ALIVE_INDEX = 0
SKILL_BOSS_PHASE_INDEX = 4
SKILL_EXTRA_PHASE = 5
SKILL_RESPAWN_TIMER_INDEX = 2
SKILL_SPECIAL_POWER_THRESHOLD = 0.5


def skill_action_mask(obs):
    """Return v2 [none, bomb, special] validity from verified Reimu A state."""
    state = obs['skill_state']
    alive = bool(state[SKILL_ALIVE_INDEX])
    bomb_active = bool(state[SKILL_BOMB_ACTIVE_INDEX])
    surge_active = bool(state[SKILL_SURGE_ACTIVE_INDEX])
    bombs_available = state[SKILL_BOMB_STOCK_INDEX] > 0
    respawn_timer = obs['timing'][SKILL_RESPAWN_TIMER_INDEX]
    extra_spell = int(obs['boss'][SKILL_BOSS_PHASE_INDEX]) == SKILL_EXTRA_PHASE
    special_power = state[SKILL_STORED_POWER_INDEX] >= SKILL_SPECIAL_POWER_THRESHOLD
    return [True,
            bombs_available and not bomb_active and respawn_timer <= 0 and not extra_spell,
            alive and not bomb_active and not surge_active and special_power]


_skill_action_mask_for_obs = skill_action_mask


def apply_skill_action_mask(skill_logits, skill_masks):
    """Set unavailable skill logits to -inf; none-valid keeps Categorical finite."""
    masks = torch.as_tensor(skill_masks, dtype=torch.bool, device=skill_logits.device)
    if masks.shape != skill_logits.shape:
        raise ValueError('Skill action mask shape must match skill logits')
    if not torch.all(masks.any(dim=-1)):
        raise ValueError('Every skill action mask must leave at least one action valid')
    return skill_logits.masked_fill(~masks, -torch.inf)


def features(obs):
    return np.asarray(obs['player'] + obs['boss'] +
        [v for bullet in obs['bullets'] for v in bullet] +
        [v for laser in obs['lasers'] for v in laser] +
        [obs['projectile_count']/1000.0, obs['laser_count']/50.0], dtype=np.float32)


def skill_features(obs):
    """Legacy threat vector followed by survival/skill timing; never reward telemetry."""
    return np.concatenate((features(obs), np.asarray(
        obs['timing'] + obs['skill_state'], dtype=np.float32)))


class ActorCritic(nn.Module):
    def __init__(self, input_dim, actions=18, width=128):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(),
                                  nn.Linear(width, width), nn.Tanh())
        self.actor = nn.Linear(width, actions)
        self.critic = nn.Linear(width, 1)

    def forward(self, obs):
        z = self.base(obs)
        return self.actor(z), self.critic(z).squeeze(-1)


class SkillActorCritic(nn.Module):
    """Independent movement and consumable-skill categorical heads."""
    def __init__(self, input_dim, actions=18, skills=3, width=128):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(input_dim, width), nn.Tanh(),
                                  nn.Linear(width, width), nn.Tanh())
        self.actor = nn.Linear(width, actions)
        self.skill_actor = nn.Linear(width, skills)
        self.critic = nn.Linear(width, 1)

    def forward(self, obs):
        z = self.base(obs)
        return self.actor(z), self.skill_actor(z), self.critic(z).squeeze(-1)

    @staticmethod
    def joint_logprob_entropy(move_logits, skill_logits, movement, skill,
                              skill_masks=None):
        joint_logp, entropy, _, _ = SkillActorCritic.joint_logprob_entropy_components(
            move_logits, skill_logits, movement, skill, skill_masks)
        return joint_logp, entropy

    @staticmethod
    def joint_logprob_entropy_components(move_logits, skill_logits, movement,
                                         skill, skill_masks=None):
        move_dist=Categorical(logits=move_logits)
        if skill_masks is not None:
            skill_logits=apply_skill_action_mask(skill_logits, skill_masks)
        skill_dist=Categorical(logits=skill_logits)
        movement_entropy=move_dist.entropy()
        skill_entropy=skill_dist.entropy()
        return (move_dist.log_prob(movement)+skill_dist.log_prob(skill),
                movement_entropy+skill_entropy, movement_entropy, skill_entropy)


def gae(rewards, values, next_values, terminated, truncated,
        gamma=0.995, lam=0.95):
    """GAE with terminal zero-bootstrap, time-limit value-bootstrap,
    and *no* recursion across a reset boundary (including time limits).
    """
    adv = np.empty(len(rewards), dtype=np.float32)
    trace = 0.0
    for t in range(len(rewards)-1, -1, -1):
        delta = float(rewards[t]) + gamma*(1.0-float(terminated[t]))*float(next_values[t]) - float(values[t])
        if terminated[t] or truncated[t]:
            trace = 0.0
        trace = delta + gamma*lam*trace
        adv[t] = trace
    return adv, adv+np.asarray(values, dtype=np.float32)


def _rng_state():
    return {'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state()}


def checkpoint(model, optimizer, outfile, config, updates, steps,
               training_state=None, skill_action_mask_version=None):
    outfile = pathlib.Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    tmp = outfile.with_name(outfile.name + '.tmp')
    is_skill=isinstance(model,SkillActorCritic)
    state={'episodes':0,'wins':0,'deaths':0,'grazes':0,**_rng_state()}
    if training_state:
        state.update(training_state)
    payload={'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'input_dim': model.base[0].in_features,
                'actions': model.actor.out_features,
                'width': model.base[0].out_features,
                'policy_type':'skill' if is_skill else 'legacy',
                'config': asdict(config), 'updates': updates, 'steps': steps,
                'training_state':state}
    if is_skill: payload['skills']=model.skill_actor.out_features
    if is_skill and skill_action_mask_version is not None:
        payload['skill_action_mask_version']=skill_action_mask_version
    torch.save(payload, tmp)
    os.replace(tmp, outfile)


def _resume_checkpoint(path, cfg, policy_type, total_steps, resume_log,
                       skill_action_mask_version=None):
    path=pathlib.Path(path)
    if not path.is_file(): raise ValueError(f'Resume checkpoint does not exist: {path}')
    try: obj=torch.load(path,map_location='cpu',weights_only=False)
    except Exception as exc: raise ValueError(f'Cannot read resume checkpoint: {exc}') from exc
    expected_policy={'policy_type':policy_type,
        'input_dim':8+6+cfg.max_projectiles*7+cfg.max_laser_samples*4+2+(18 if policy_type=='skill' else 0),
        'actions':18,'width':128}
    if policy_type=='skill':expected_policy['skills']=3
    if policy_type=='skill':
        actual_mask_version=obj.get('skill_action_mask_version')
        if actual_mask_version != skill_action_mask_version:
            raise ValueError('Resume checkpoint skill action mask version mismatch: '
                             f'{actual_mask_version!r} != {skill_action_mask_version!r}')
    for key,value in expected_policy.items():
        actual=obj.get(key,'legacy' if key=='policy_type' else None)
        if actual!=value:raise ValueError(f'Resume checkpoint {key} mismatch: {actual!r} != {value!r}')
    saved_config=obj.get('config')
    expected_config=asdict(cfg)
    if not isinstance(saved_config,dict):raise ValueError('Resume checkpoint env config mismatch')
    path_fields={'library_path','resource_path','storage_path'}
    config_matches=saved_config.keys()==expected_config.keys()
    if config_matches:
        for key,expected in expected_config.items():
            if key in path_fields:
                continue
            if saved_config[key]!=expected:
                config_matches=False;break
    if not config_matches:raise ValueError('Resume checkpoint env config mismatch')
    try:
        steps=int(obj['steps']);updates=int(obj['updates'])
    except (KeyError,TypeError,ValueError) as exc:raise ValueError('Resume checkpoint lacks valid steps/updates') from exc
    if not 0 <= steps < total_steps:raise ValueError('Resume checkpoint steps must satisfy 0 <= steps < total_steps')
    try:
        probe=(SkillActorCritic(obj['input_dim'],obj['actions'],obj['skills'],obj['width'])
               if policy_type=='skill' else ActorCritic(obj['input_dim'],obj['actions'],obj['width']))
        probe.load_state_dict(obj['model'])
        probe_optimizer=Adam(probe.parameters(),lr=3e-4)
        probe_optimizer.load_state_dict(obj['optimizer'])
        for param,state_item in probe_optimizer.state.items():
            for key,value in state_item.items():
                if key in ('exp_avg','exp_avg_sq','max_exp_avg_sq') and value.shape!=param.shape:
                    raise ValueError(f'Invalid optimizer {key} shape')
    except Exception as exc:raise ValueError(f'Cannot restore resume model/optimizer state: {exc}') from exc
    state=obj.get('training_state')
    if state is None or not all(k in state for k in ('episodes','wins','deaths','grazes')):
        if not resume_log:raise ValueError('Legacy checkpoint requires --resume-log with paired counters')
        try:
            rows=[json.loads(line) for line in pathlib.Path(resume_log).read_text(encoding='utf-8').splitlines() if line.strip()]
            row=rows[-1]
            if int(row['steps'])!=steps or int(row['updates'])!=updates:raise ValueError('Resume-log steps/updates do not match checkpoint')
            state={k:int(row[k]) for k in ('episodes','wins','deaths','grazes')}
        except ValueError:raise
        except Exception as exc:raise ValueError(f'Invalid --resume-log counters: {exc}') from exc
    else:
        state=dict(state)
    state['rng_reset']=not all(k in state for k in ('python_rng_state','numpy_rng_state','torch_rng_state'))
    return obj,state,steps,updates


def load_model(checkpoint_path, device='cpu'):
    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if obj.get('policy_type','legacy') != 'legacy':
        raise ValueError('load_model only accepts legacy checkpoints; use load_policy')
    model = ActorCritic(obj['input_dim'], obj['actions'], obj['width']).to(device)
    model.load_state_dict(obj['model'])
    model.eval()
    return model, obj


def load_policy(checkpoint_path, device='cpu'):
    obj=torch.load(checkpoint_path,map_location=device,weights_only=False)
    policy_type=obj.get('policy_type','legacy')
    if policy_type=='legacy':
        model=ActorCritic(obj['input_dim'],obj['actions'],obj['width']).to(device)
    elif policy_type=='skill':
        model=SkillActorCritic(obj['input_dim'],obj['actions'],obj['skills'],obj['width']).to(device)
    else:
        raise ValueError(f"Unsupported checkpoint policy_type: {policy_type}")
    model.load_state_dict(obj['model'])
    model.eval()
    obj['policy_type']=policy_type
    return model,obj


def optimize(model, optimizer, rollout, gamma=0.995, gae_lambda=0.95,
             epochs=4, batch_size=64, clip=0.2, vf_coef=0.5,
             entropy_coef=0.01, max_grad_norm=0.5):
    advantages, returns = gae(rollout['rewards'], rollout['values'],
        rollout['next_values'], rollout['terminated'], rollout['truncated'],
        gamma=gamma, lam=gae_lambda)
    device = next(model.parameters()).device
    x = torch.as_tensor(np.stack(rollout['states']), dtype=torch.float32, device=device)
    skill_policy=isinstance(model,SkillActorCritic)
    if skill_policy:
        actions=np.asarray(rollout['actions'],dtype=np.int64)
        a=torch.as_tensor(actions[:,0],dtype=torch.long,device=device)
        skill_a=torch.as_tensor(actions[:,1],dtype=torch.long,device=device)
        skill_masks=(torch.as_tensor(rollout['skill_masks'],dtype=torch.bool,device=device)
                     if 'skill_masks' in rollout else None)
    else:
        a = torch.as_tensor(rollout['actions'], dtype=torch.long, device=device)
    oldlog = torch.as_tensor(rollout['logprobs'], dtype=torch.float32, device=device)
    ret = torch.as_tensor(returns, dtype=torch.float32, device=device)
    adv = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    adv = (adv-adv.mean()) / (adv.std(unbiased=False)+1e-8)
    indices = np.arange(len(a))
    losses=[]
    model.train()
    for _ in range(epochs):
        np.random.shuffle(indices)
        for begin in range(0,len(indices),batch_size):
            ix = indices[begin:begin+batch_size]
            if skill_policy:
                logits,skill_logits,value=model(x[ix])
                (joint_logp,entropy_values,movement_entropy_values,
                 skill_entropy_values)=model.joint_logprob_entropy_components(
                    logits,skill_logits,a[ix],skill_a[ix],
                    skill_masks[ix] if skill_masks is not None else None)
                entropy=entropy_values.mean()
            else:
                logits, value = model(x[ix]); dist=Categorical(logits=logits)
                joint_logp=dist.log_prob(a[ix]);entropy=dist.entropy().mean()
            ratio = torch.exp(joint_logp-oldlog[ix])
            surrogate = torch.minimum(ratio*adv[ix], ratio.clamp(1-clip,1+clip)*adv[ix])
            actor_loss=-surrogate.mean()
            critic_loss=((value-ret[ix])**2).mean()
            loss=actor_loss + vf_coef*critic_loss - entropy_coef*entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)
            optimizer.step()
            if skill_policy:
                losses.append((float(actor_loss.detach()),float(critic_loss.detach()),
                               float(entropy.detach()),
                               float(movement_entropy_values.mean().detach()),
                               float(skill_entropy_values.mean().detach())))
            else:
                losses.append((float(actor_loss.detach()),float(critic_loss.detach()),float(entropy.detach())))
    stats={'policy_loss':float(np.mean([v[0] for v in losses])),
           'value_loss':float(np.mean([v[1] for v in losses])),
           'entropy':float(np.mean([v[2] for v in losses]))}
    if skill_policy:
        stats['movement_entropy']=float(np.mean([v[3] for v in losses]))
        stats['skill_entropy']=float(np.mean([v[4] for v in losses]))
    return stats


def train(cfg:EnvConfig, *, outfile='results/ppo_taisei_native.pt',
          total_steps=100000, rollout_steps=512, learning_rate=3e-4,
          seed=42, device='cpu', gamma=0.995, lam=0.95,
          init=None, resume=None, resume_log=None, log_path=None, policy_type='legacy',
          skill_action_mask=False):
    if skill_action_mask and policy_type != 'skill':
        raise ValueError('--skill-action-mask requires --policy-type skill')
    skill_mask_version=SKILL_ACTION_MASK_VERSION if skill_action_mask else None
    if init and resume:raise ValueError('--init and --resume are mutually exclusive')
    if resume:
        if log_path is not None:
            checkpoint_path=pathlib.Path(outfile).resolve()
            log_file_path=pathlib.Path(log_path).resolve()
            if os.path.normcase(str(checkpoint_path))==os.path.normcase(str(log_file_path)):
                raise ValueError('Resume log/checkpoint output path collision')
        if pathlib.Path(outfile).exists():raise ValueError(f'Resume output checkpoint already exists: {outfile}')
        if log_path and pathlib.Path(log_path).exists():raise ValueError(f'Resume output log already exists: {log_path}')
        resume_obj,resume_state,steps,updates=_resume_checkpoint(
            resume,cfg,policy_type,total_steps,resume_log,skill_mask_version)
        torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    else:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if cfg.frames_per_action < 1:
        raise ValueError('frames_per_action must be >=1')
    # gamma is per *action*, not per internal physics frame.
    if cfg.frames_per_action != 1:
        print('WARNING: gamma is discounted per action, not per simulation frame')
    with TaiseiEnv(cfg) as env:
        obs, _ = env.reset(seed=seed + resume_state['episodes'] if resume else seed)
        feature_fn=skill_features if policy_type=='skill' else features
        if policy_type not in ('legacy','skill'):raise ValueError('policy_type must be legacy or skill')
        x = feature_fn(obs)
        model = (SkillActorCritic(len(x)) if policy_type=='skill' else ActorCritic(len(x))).to(device)
        if resume:
            if len(x)!=resume_obj['input_dim']:raise ValueError('Checkpoint observation dimension mismatch')
            model.load_state_dict(resume_obj['model'])
        elif init:
            existing, meta = load_policy(init, device)
            if meta['policy_type']!=policy_type:raise ValueError('Checkpoint policy type mismatch')
            if meta['input_dim']!=len(x):raise ValueError('Checkpoint observation dimension mismatch')
            model.load_state_dict(existing.state_dict())
        optimizer=Adam(model.parameters(),lr=learning_rate)
        if resume:
            try:optimizer.load_state_dict(resume_obj['optimizer'])
            except Exception as exc:raise ValueError(f'Cannot restore resume optimizer state: {exc}') from exc
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    state=optimizer.state.get(param,{})
                    for key,value in state.items():
                        if torch.is_tensor(value):state[key]=value.to(device)
            if resume_state['rng_reset']:
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            else:
                try:
                    random.setstate(resume_state['python_rng_state'])
                    np.random.set_state(resume_state['numpy_rng_state'])
                    torch.set_rng_state(resume_state['torch_rng_state'].cpu())
                except Exception as exc:raise ValueError(f'Invalid checkpoint RNG state: {exc}') from exc
            episodes,wins,deaths,grazes=(int(resume_state[k]) for k in ('episodes','wins','deaths','grazes'))
            boundary={'event':'resume_boundary','steps':steps,'updates':updates,
                      'episodes':episodes,'fresh_episode':True,'discarded_partial_episode':True,
                      'rng_reset':resume_state['rng_reset']}
            print(json.dumps(boundary),flush=True)
        else:
            steps=0;updates=0;episodes=0;wins=0;deaths=0;grazes=0
        while steps<total_steps:
            b={key:[] for key in ('states','actions','logprobs','values',
                                  'next_values','rewards','terminated','truncated')}
            if skill_action_mask: b['skill_masks']=[]
            for _ in range(min(rollout_steps,total_steps-steps)):
                with torch.no_grad():
                    if policy_type=='skill':
                        logits,skill_logits,value=model(torch.as_tensor(x,dtype=torch.float32,device=device).unsqueeze(0))
                        current_mask=_skill_action_mask_for_obs(obs) if skill_action_mask else None
                        if current_mask is not None:
                            skill_logits=apply_skill_action_mask(skill_logits,[current_mask])
                        move_dist=Categorical(logits=logits);skill_dist=Categorical(logits=skill_logits)
                        action=move_dist.sample();skill=skill_dist.sample()
                        log_prob=move_dist.log_prob(action)+skill_dist.log_prob(skill)
                    else:
                        logits, value = model(torch.as_tensor(x, dtype=torch.float32, device=device).unsqueeze(0))
                        dist=Categorical(logits=logits);action=dist.sample();log_prob=dist.log_prob(action)
                env_action=(int(action.item()),int(skill.item())) if policy_type=='skill' else int(action.item())
                next_obs, reward, term, trunc, info = env.step(env_action)
                next_x=feature_fn(next_obs)
                with torch.no_grad():
                    next_output=model(torch.as_tensor(next_x,dtype=torch.float32,device=device).unsqueeze(0))
                    next_value=next_output[2] if policy_type=='skill' else next_output[1]
                b['states'].append(x.copy());b['actions'].append(
                    [int(action.item()),int(skill.item())] if policy_type=='skill' else int(action.item()))
                if skill_action_mask: b['skill_masks'].append(current_mask)
                b['logprobs'].append(float(log_prob.item()))
                b['values'].append(float(value.item()))
                b['next_values'].append(float(next_value.item()))
                b['rewards'].append(float(reward))
                b['terminated'].append(bool(term));b['truncated'].append(bool(trunc))
                steps+=1
                deaths+=int(info.get('death_delta',0)); grazes+=int(info.get('graze_delta',0))
                if term or trunc:
                    episodes+=1;wins+=int(info['won'])
                    next_obs, _=env.reset(seed=seed+episodes)
                    next_x=feature_fn(next_obs)
                x=next_x
            stats=optimize(model, optimizer, b, gamma=gamma, gae_lambda=lam)
            updates+=1
            row={'updates':updates,'steps':steps,'episodes':episodes,'wins':wins,
                 'deaths':deaths,'grazes':grazes,**stats}
            if resume and updates==resume_obj['updates']+1:
                row.update({'event':'resume_boundary','fresh_episode':True,
                            'discarded_partial_episode':True,'rng_reset':resume_state['rng_reset']})
            print(json.dumps(row),flush=True)
            if log_path:
                pathlib.Path(log_path).parent.mkdir(parents=True,exist_ok=True)
                with open(log_path,'a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            state={'episodes':episodes,'wins':wins,'deaths':deaths,'grazes':grazes}
            if resume and resume_state['rng_reset']:state['rng_reset']=True
            checkpoint(model,optimizer,outfile,cfg,updates,steps,training_state=state,
                       skill_action_mask_version=skill_mask_version)
    return outfile


def mean_alive_frames(reports):
    """Mean per-episode alive frames, or None if any count is unknown."""
    if not reports or any(row.get('alive_frames') is None for row in reports):
        return None
    return sum(row['alive_frames'] for row in reports) / len(reports)


def _nearest_projectile_trace(obs):
    """Decode closest-approach-sorted observation rows into pixel geometry."""
    player = obs.get('player', ())
    if len(player) < 4:
        return None
    player_vx, player_vy = float(player[2]) * 20.0, float(player[3]) * 20.0
    candidates = []
    for bullet in obs.get('bullets', ()):
        if len(bullet) < 7 or float(bullet[6]) != 1.0:
            continue
        dx, dy = float(bullet[0]) * 480.0, float(bullet[1]) * 560.0
        vx = float(bullet[2]) * 20.0 - player_vx
        vy = float(bullet[3]) * 20.0 - player_vy
        speed_squared = vx * vx + vy * vy
        time_to_closest = (min(30.0, max(0.0, -(dx * vx + dy * vy) / speed_squared))
                           if speed_squared > 1e-9 else 0.0)
        current_separation = math.hypot(dx, dy)
        closest_separation = math.hypot(dx + vx * time_to_closest,
                                        dy + vy * time_to_closest)
        candidates.append((closest_separation, current_separation, time_to_closest,
                           float(bullet[4]) * 30.0, float(bullet[5]) * 30.0))
    if not candidates:
        return None
    closest, current, time_to_closest, width, height = min(candidates,
                                                           key=lambda item: item[:2])
    return {
        'current_separation_px': current,
        'predicted_closest_separation_px': closest,
        'time_to_closest_frames': time_to_closest,
        'collision_width_px': width,
        'collision_height_px': height,
    }


def evaluate(cfg:EnvConfig, checkpoint_path, episodes=20,seed=90000,
             sampled=False, device='cpu', planner=None, trace_callback=None,
             sampling_seed=None):
    """Evaluate episodes; frames is elapsed, alive_frames counts alive one-frame
    post-action native states (reset excluded). For action strides >1, exact
    intermediate alive frames are unavailable, so alive_frames is None.
    """
    if sampling_seed is not None and (not sampled or planner is not None):
        raise ValueError('sampling_seed requires sampled PPO evaluation')
    model=None
    policy_meta={}
    if not planner:
        model,policy_meta=load_policy(checkpoint_path,device)
    if sampling_seed is not None:
        # Model construction/loading consumes Torch RNG; reseed only after load.
        torch.manual_seed(sampling_seed)
    policy_type='legacy' if planner else ('skill' if isinstance(model,SkillActorCritic) else 'legacy')
    feature_fn=skill_features if policy_type=='skill' else features
    reports=[]
    with TaiseiEnv(cfg) as env:
        for e in range(episodes):
            obs,_=env.reset(seed=seed+e)
            first=obs;total_reward=0; boss_damage=0.0
            alive_frames=0 if cfg.frames_per_action == 1 else None
            skill_events=[];bomb_attempts=0;bomb_uses_confirmed=0
            special_attempts=0;special_activations=0
            surge_active_frames=0 if cfg.frames_per_action == 1 else None
            action_index=0
            while True:
                x=feature_fn(obs)
                move_entropy=move_max_probability=move_argmax=None
                skill_entropy=skill_max_probability=skill_argmax=None
                if planner:
                    action=planner(obs)
                else:
                    with torch.no_grad():
                        out=model(torch.as_tensor(x,dtype=torch.float32,device=device).unsqueeze(0))
                        if policy_type=='skill':
                            logits,skill_logits,_=out
                            version=policy_meta.get('skill_action_mask_version')
                            if version not in (None,SKILL_ACTION_MASK_VERSION):
                                raise ValueError(f'Unsupported skill action mask version: {version}')
                            if version==SKILL_ACTION_MASK_VERSION:
                                skill_logits=apply_skill_action_mask(skill_logits,[_skill_action_mask_for_obs(obs)])
                            if trace_callback is not None:
                                move_dist=Categorical(logits=logits)
                                skill_dist=Categorical(logits=skill_logits)
                                move_entropy=float(move_dist.entropy().item())
                                move_max_probability=float(move_dist.probs.max(dim=1).values.item())
                                move_argmax=int(logits.argmax(dim=1).item())
                                skill_entropy=float(skill_dist.entropy().item())
                                skill_max_probability=float(skill_dist.probs.max(dim=1).values.item())
                                skill_argmax=int(skill_logits.argmax(dim=1).item())
                            move=int(Categorical(logits=logits).sample().item()) if sampled else int(logits.argmax(dim=1).item())
                            skill=int(Categorical(logits=skill_logits).sample().item()) if sampled else int(skill_logits.argmax(dim=1).item())
                            action=(move,skill)
                        else:
                            logits,_=out
                            if trace_callback is not None:
                                move_dist=Categorical(logits=logits)
                                move_entropy=float(move_dist.entropy().item())
                                move_max_probability=float(move_dist.probs.max(dim=1).values.item())
                                move_argmax=int(logits.argmax(dim=1).item())
                            action=int(Categorical(logits=logits).sample().item()) if sampled else int(logits.argmax(dim=1).item())
                pre_state=None
                if trace_callback is not None:
                    player=obs.get('player', [])
                    move_action=int(action[0] if isinstance(action,(tuple,list)) else action)
                    skill_action=(int(action[1]) if policy_type=='skill' and
                                  isinstance(action,(tuple,list)) else None)
                    pre_state={
                        'logical_frame':int(obs.get('logical_frame',0)),
                        'player_x':float(player[0]) if len(player)>0 else None,
                        'player_y':float(player[1]) if len(player)>1 else None,
                        'alive':bool(obs.get('skill_state',[False])[0]),
                        'deaths':int(obs.get('deaths',0)),
                        'projectile_count':int(obs.get('projectile_count',0)),
                        'laser_count':int(obs.get('laser_count',0)),
                        'nearest_projectile':_nearest_projectile_trace(obs),
                    }
                obs,reward,term,trunc,info=env.step(action)
                if trace_callback is not None:
                    trace_callback({
                        'seed':seed+e,'episode_index':e,'action_index':action_index,
                        'pre':pre_state,'movement_action':move_action,'skill_action':skill_action,
                        'movement_argmax':move_argmax,'movement_entropy':move_entropy,
                        'movement_max_probability':move_max_probability,
                        'skill_argmax':skill_argmax,'skill_entropy':skill_entropy,
                        'skill_max_probability':skill_max_probability,
                        'post':{
                            'logical_frame':int(info.get('logical_frame_after',obs.get('logical_frame',0))),
                            'alive':bool(info.get('player_alive_after',obs.get('skill_state',[False])[0])),
                            'deaths':int(obs.get('deaths',0)),
                            'boss_damage':float(info.get('boss_damage',0.0)),
                        },
                    })
                action_index+=1
                total_reward+=reward
                if alive_frames is not None:
                    # Count resulting native alive-state samples: reset is excluded,
                    # each one-frame action contributes exactly 0 or 1 alive frame.
                    alive_frames+=int(bool(obs['skill_state'][0]))
                requested=int(info.get('requested_skill',0))
                used_delta=int(info.get('bombs_used_delta',0))
                bomb_transition=(info.get('bomb_active_before') is not None and
                    info.get('bomb_active_after') is not None and
                    info['bomb_active_before'] != info['bomb_active_after'])
                surge_transition=(info.get('power_surge_active_before') is not None and
                    info.get('power_surge_active_after') is not None and
                    info['power_surge_active_before'] != info['power_surge_active_after'])
                if requested or used_delta or bomb_transition or surge_transition:
                    skill_events.append({
                        'logical_frame':int(info.get('logical_frame_after',obs.get('logical_frame',0))),
                        'logical_frame_before':info.get('logical_frame_before'),
                        'logical_frame_after':info.get('logical_frame_after'),
                        'requested_skill':requested,
                        'player_alive_before':info.get('player_alive_before'),
                        'player_alive_after':info.get('player_alive_after'),
                        'bombs_before':info.get('bombs_before'),
                        'bombs_after':info.get('bombs_after'),
                        'bombs_used_before':info.get('bombs_used_before'),
                        'bombs_used_after':info.get('bombs_used_after'),
                        'bombs_used_delta':used_delta,
                        'bomb_use_confirmed':used_delta > 0,
                        'bomb_active_transition':[info.get('bomb_active_before'),info.get('bomb_active_after')],
                        'power_surge_active_transition':[info.get('power_surge_active_before'),
                            info.get('power_surge_active_after')],
                        'power_surge_positive_transition':[info.get('power_surge_positive_before'),
                            info.get('power_surge_positive_after')],
                        'power_surge_negative_transition':[info.get('power_surge_negative_before'),
                            info.get('power_surge_negative_after')],
                    })
                bomb_attempts += int(requested == 1)
                special_attempts += int(requested == 2)
                bomb_uses_confirmed += max(0,used_delta)
                special_activations += int(
                    info.get('power_surge_active_before') == 0 and
                    info.get('power_surge_active_after') == 1)
                if surge_active_frames is not None and info.get('logical_frame_after') is not None:
                    if (int(info['logical_frame_after'])-int(info.get('logical_frame_before',0)) == 1
                            and bool(info.get('power_surge_active_after',0))):
                        surge_active_frames += 1
                boss_damage+=max(0.0,float(info.get('boss_damage',0.0)))
                if term or trunc:
                    reports.append({'seed':seed+e,'won':bool(info['won']),'cleared':bool(term and info['won']),
                        'frames':info['frames'],'alive_frames':alive_frames,
                        'boss_damage':boss_damage,
                        'graze':obs['graze']-first['graze'],'deaths':obs['deaths']-first['deaths'],
                        'score':int(obs.get('score',0)),
                        'point_item_value':int(obs.get('point_item_value',0)),
                        'bombs_used':int(obs.get('bombs_used',0)),
                        'logical_frame':int(obs.get('logical_frame',info['frames'])),
                        'skill_events':skill_events,'bomb_attempts':bomb_attempts,
                        'bomb_uses_confirmed':bomb_uses_confirmed,
                        'special_attempts':special_attempts,
                        'special_activations':special_activations,
                        'surge_active_frames':surge_active_frames,
                        'reward':round(total_reward,3),'terminated':term,'truncated':trunc,
                        'digest':info['digest']});break
    return reports
