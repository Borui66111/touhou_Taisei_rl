#!/usr/bin/env python3
"""E6: episode-disjoint offline imitation audit on E3 authentic Taisei data.

Train only on the complete 7,857-frame native teacher episode (seed 48030).
Hold out the 4,143-frame *partial* second episode (seed 48031) in entirety.
Compare standard BC with E4-style class-reweighted BC at identical seeds,
initializations, frame budgets and epochs. Never claim native gameplay gains.

This runs PyTorch on *archived genuine native observations*, never a toy game.
No new environment interactions, champion changes, or final seed access.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import pathlib
import struct
import sys
import zipfile

DATASET_SHA = '02dd11feb60e0dcb5710ae8af3cf33dca7e7e77756d6467f822ad94d0f5a0d8d'
N_FRAMES = 12000
TRAIN_END = 7857
HOLDOUT_START = TRAIN_END
HOLDOUT_N = 4143
TRAIN_GAME_SEED = 48030
HOLDOUT_GAME_SEED = 48031
MODEL_SEEDS = (48030, 48032, 48033)
N_FEATURES = 528
N_ACTIONS = 18
EPOCHS = 16
BATCH = 256
LR = 0.001
EXPECTED_ACTION_COUNTS = (8321,156,0,124,138,16,24,0,0,0,109,123,1173,1097,171,177,194,177)
EXPECTED_TRAIN_COUNTS = (5391,122,0,110,66,4,22,0,0,0,70,70,779,722,108,122,141,130)
VARIANTS = ('plain_ce', 'half_class_weighted_ce')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_split(n, boundary=TRAIN_END):
    if n != N_FRAMES or boundary != TRAIN_END or boundary + HOLDOUT_N != n:
        raise ValueError('E6 split must preserve exact E3 episode boundary (7857/4143)')
    return (range(0, boundary), range(boundary, n))


def read_native_labels_without_numpy(dataset):
    """Source-only CI can independently audit .npz labels without torch/numpy."""
    if file_sha256(dataset) != DATASET_SHA:
        raise ValueError('Not the immutable E3 native demonstration dataset')
    with zipfile.ZipFile(dataset) as archive:
        if set(archive.namelist()) != {'states.npy', 'actions.npy'}:
            raise ValueError('Unexpected native demonstration fields')
        raw = archive.read('actions.npy')
    if not raw.startswith(b'\x93NUMPY') or raw[6] not in (1,2):
        raise ValueError('Invalid native action .npy header')
    header_size = 2 if raw[6] == 1 else 4
    offset = 8 + header_size
    nheader = struct.unpack_from('<H' if header_size == 2 else '<I',raw,8)[0]
    header = ast.literal_eval(raw[offset:offset+nheader].decode('latin1'))
    if (header.get('shape') != (N_FRAMES,) or header.get('fortran_order') is not False
            or header.get('descr') not in ('<i8','|i8','=i8')):
        raise ValueError('Unexpected action label shape/encoding')
    data = raw[offset+nheader:]
    if len(data) != N_FRAMES * 8:
        raise ValueError('Action label byte count mismatch')
    actions = [v[0] for v in struct.iter_unpack('<q',data)]
    if any(not 0 <= v < N_ACTIONS for v in actions):
        raise ValueError('Invalid native action index')
    counts = tuple(actions.count(i) for i in range(N_ACTIONS))
    tr_counts = tuple(actions[:TRAIN_END].count(i) for i in range(N_ACTIONS))
    if counts != EXPECTED_ACTION_COUNTS or tr_counts != EXPECTED_TRAIN_COUNTS:
        raise ValueError('Native teacher action distribution/episode split mismatch')
    return actions


def diagnostic(y_true, y_pred, bullet_count, density_threshold):
    """Agreement with recorded teacher actions, not actual dodge success."""
    if not (len(y_true) == len(y_pred) == len(bullet_count)) or not y_true:
        raise ValueError('Empty or misaligned action/observation vectors')
    if any(not isinstance(v,int) or not 0 <= v < N_ACTIONS for v in [*y_true,*y_pred]):
        raise ValueError('Invalid teacher or model action')
    if not isinstance(density_threshold,(int,float)) or not 0 <= density_threshold < 1:
        raise ValueError('Invalid training-derived bullet density threshold')
    def acc(indices):
        return sum(y_true[i] == y_pred[i] for i in indices) / len(indices) if indices else None
    n = len(y_true)
    all_ix = list(range(n))
    nonstay = [i for i in all_ix if y_true[i] != 0]
    high = [i for i in all_ix if bullet_count[i] >= density_threshold]
    high_nonstay = [i for i in high if y_true[i] != 0]
    active = [k for k in range(N_ACTIONS) if k in y_true]
    recall = [acc([i for i in all_ix if y_true[i] == k]) for k in active]
    return {
        'frame_accuracy': acc(all_ix),
        'always_stay_accuracy': sum(v == 0 for v in y_true) / n,
        'nonstay_accuracy': acc(nonstay),
        'macro_recall_supported_labels': sum(recall) / len(recall),
        'predicted_stay_fraction': sum(v == 0 for v in y_pred) / n,
        'high_bullet_density_threshold': float(density_threshold),
        'high_bullet_density_support': len(high),
        'high_density_nonstay_support': len(high_nonstay),
        'high_density_nonstay_accuracy': acc(high_nonstay),
        'high_density_frame_accuracy': acc(high),
    }


def class_weights(labels):
    import numpy as np
    counts = np.bincount(labels,minlength=N_ACTIONS)
    active = int((counts>0).sum())
    weights = np.zeros(N_ACTIONS,dtype=np.float32)
    weights[counts>0] = np.sqrt(len(labels)/(active*counts[counts>0]))
    weights = np.clip(weights,0.5,3.0)
    weights /= np.sum(counts*weights)/len(labels)
    return weights


def run(dataset, output):
    import numpy as np
    import torch
    from torch import nn
    # Actual repository ActorCritic architecture. No native simulator imported or executed.
    root = pathlib.Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(root/'my_test'))
    from native_rl.torch_ppo import ActorCritic

    read_native_labels_without_numpy(dataset)
    with np.load(dataset,allow_pickle=False) as z:
        if set(z.files) != {'states','actions'}:
            raise ValueError('Unexpected genuine native feature fields')
        states = z['states']
        labels = z['actions']
    if (states.shape != (N_FRAMES,N_FEATURES) or labels.shape != (N_FRAMES,)
            or not np.isfinite(states).all() or not np.issubdtype(labels.dtype,np.integer)):
        raise ValueError('Unexpected native feature shape/values')
    validate_split(len(labels))
    if (np.bincount(labels[:TRAIN_END],minlength=N_ACTIONS) != EXPECTED_TRAIN_COUNTS).any():
        raise ValueError('Teacher episode split corrupted')
    torch.set_num_threads(min(torch.get_num_threads(),2))
    torch.use_deterministic_algorithms(True)
    x = torch.from_numpy(states.astype(np.float32,copy=False))
    y = torch.from_numpy(labels.astype(np.int64,copy=False))
    threshold = float(np.quantile(states[:TRAIN_END,-2],0.75))
    weights = torch.from_numpy(class_weights(labels[:TRAIN_END]))
    report = {
        'schema':'taisei_rsi_e6_episode_disjoint_offline_v1',
        'status':'offline_imitation_diagnostics_only',
        'canonical_branch':'research/rsi-automation-v1',
        'source':'E3 authentic native Taisei teacher demonstrations (permanent E3 escrow)',
        'dataset_sha256':DATASET_SHA,
        'training_episode':{'game_seed':TRAIN_GAME_SEED,'frames':TRAIN_END,'completed':True},
        'heldout_episode':{'game_seed':HOLDOUT_GAME_SEED,'frames':HOLDOUT_N,'completed':False},
        'architecture':'ActorCritic 528-input 18 movement-action 128-width, no Bomb/Special',
        'model_seeds':list(MODEL_SEEDS),
        'training_epochs':EPOCHS,
        'learning_rate':LR,
        'batch_size':BATCH,
        'train_action_counts':list(EXPECTED_TRAIN_COUNTS),
        'heldout_action_counts':(np.bincount(labels[TRAIN_END:],minlength=N_ACTIONS)).tolist(),
        'density_threshold_train_q75':threshold,
        'experiment_design':'matched-initialization two-arm BC; entire second native episode held out; one completed training episode',
        'variant_definitions':{'plain_ce':'unweighted categorical cross entropy',
                               'half_class_weighted_ce':'0.5*CE + 0.5*train-frequency sqrt-inverse-class-weighted CE, normalized/capped as E4'},
        'native_training_decisions':0,
        'native_evaluation_decisions':0,
        'source_checkout_recovered':False,
        'native_game_executed':False,
        'gameplay_improvement_verified':False,
        'promotion_allowed':False,
        'results':[],
        'limitations':['Holdout is ONE partial game episode; no independent multi-episode uncertainty or stage generalization.',
                       'Actions are evaluated only on recorded teacher states (behavioral cloning distribution); closed-loop covariate shift is not measured.',
                       'Teacher action agreement and projectile count are proxies, not native collision avoidance, collection or stage clearing.',
                       'All E6 policies are movement-only and incompatible with skill-v2 live-demo loader without separately verified conversion.',
                       'No native E5 replay until exact pinned source and recursive submodules are recovered.'],
    }
    output = pathlib.Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for seed in MODEL_SEEDS:
        for variant in VARIANTS:
            torch.manual_seed(seed)
            np.random.seed(seed)
            rng = np.random.default_rng(seed)
            model = ActorCritic(N_FEATURES,actions=N_ACTIONS,width=128)
            optimizer = torch.optim.Adam(model.parameters(),lr=LR)
            history=[]
            for epoch in range(EPOCHS):
                order = rng.permutation(TRAIN_END)
                model.train()
                total_loss = 0.0
                for start in range(0,TRAIN_END,BATCH):
                    idx = order[start:start+BATCH]
                    logits,_ = model(x[idx])
                    per = nn.functional.cross_entropy(logits,y[idx],reduction='none')
                    loss = (per if variant=='plain_ce' else per*(.5+.5*weights[y[idx]])).mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite E6 imitation loss')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach())*len(idx)
                history.append(total_loss/TRAIN_END)
            model.eval()
            with torch.no_grad():
                logits,_ = model(x[TRAIN_END:])
                pred = logits.argmax(dim=1).cpu().tolist()
            result = diagnostic(labels[TRAIN_END:].tolist(),pred,states[TRAIN_END:,-2].tolist(),threshold)
            ckpt = output/f'e6_{variant}_seed{seed}.pt'
            payload = {'model':model.state_dict(),'input_dim':N_FEATURES,'actions':N_ACTIONS,'width':128,
                       'policy_type':'legacy','source':'E6_episode_disjoint_native_teacher_offline',
                       'dataset_sha256':DATASET_SHA,'training_game_seed':TRAIN_GAME_SEED,
                       'heldout_game_seed':HOLDOUT_GAME_SEED,'train_frames':TRAIN_END,
                       'train_seed':seed,'loss_variant':variant,'epochs':EPOCHS}
            torch.save(payload,ckpt)
            reloaded = torch.load(ckpt,map_location='cpu',weights_only=True)
            verifier = ActorCritic(N_FEATURES,actions=N_ACTIONS,width=128)
            verifier.load_state_dict(reloaded['model'])
            verifier.eval()
            with torch.no_grad():
                replay_pred = verifier(x[TRAIN_END:])[0].argmax(dim=1).tolist()
            if pred != replay_pred:
                raise ValueError('Saved E6 checkpoint inference mismatch')
            report['results'].append({'seed':seed,'variant':variant,'checkpoint':ckpt.name,
                                      'checkpoint_sha256':file_sha256(ckpt),
                                      'last_train_loss':history[-1],**result})
    paired=[]
    for seed in MODEL_SEEDS:
        base=next(r for r in report['results'] if r['seed']==seed and r['variant']=='plain_ce')
        weighted=next(r for r in report['results'] if r['seed']==seed and r['variant']=='half_class_weighted_ce')
        paired.append({'seed':seed,
                       'delta_frame_accuracy':weighted['frame_accuracy']-base['frame_accuracy'],
                       'delta_nonstay_accuracy':weighted['nonstay_accuracy']-base['nonstay_accuracy'],
                       'delta_macro_recall':weighted['macro_recall_supported_labels']-base['macro_recall_supported_labels'],
                       'delta_high_density_nonstay_accuracy':weighted['high_density_nonstay_accuracy']-base['high_density_nonstay_accuracy']})
    report['paired_deltas_weighted_minus_plain'] = paired
    report['decision'] = 'No E6 policy promotion or native gain claim. Use heldout results to prioritize multi-episode native teacher coverage and a later matched-seed closed-loop evaluation once exact pinned source is restored.'
    report['software']={'python':sys.version.split()[0],'torch':torch.__version__,'numpy':np.__version__}
    (output/'e6_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    args=p.parse_args(argv)
    report=run(args.dataset,args.output)
    print(json.dumps({'status':report['status'],'paired_deltas':report['paired_deltas_weighted_minus_plain'],
                      'report':str(args.output/'e6_report.json')},indent=2))


if __name__=='__main__':
    main()
