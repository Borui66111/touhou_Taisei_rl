#!/usr/bin/env python3
"""Compare actual native Taisei digests on repeated seeded input traces.

No mocks and no replacement simulator; fails fast if library cannot load.
"""
import argparse,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.taisei_ffi import NativeTaisei,STATUS_RUNNING
from native_rl.env import action_buttons
p=argparse.ArgumentParser()
p.add_argument('--library',required=True);p.add_argument('--resource-path')
p.add_argument('--stage',type=int,default=1);p.add_argument('--difficulty',type=int,default=2)
p.add_argument('--practice',action='store_true',help='Enable spell-practice mode (default: off)')
p.add_argument('--frames',type=int,default=600);p.add_argument('--seed',type=int,default=1402)
a=p.parse_args()

def run(sim):
    state,_=sim.reset(seed=a.seed,stage_id=a.stage,difficulty=a.difficulty,practice=a.practice)
    seq=[]
    # The same action sequence for both runs; do not depend on observations.
    for i in range(a.frames):
        if state.episode_status!=STATUS_RUNNING:break
        action=[0,3,0,4,9,0,1,2][(i//40)%8]
        state,arrays=sim.step(action_buttons(action))
        seq.append([int(state.logical_frame),int(state.gameplay_digest),
                    int(state.projectile_count),int(state.graze),int(state.deaths)])
    return seq

with NativeTaisei(a.library,resource_path=a.resource_path) as sim:
    one=run(sim)
    two=run(sim)
if one!=two:
    bad=next((i for i,(x,y) in enumerate(zip(one,two)) if x!=y),None)
    print(json.dumps({'result':'FAIL','frames_first':len(one),'frames_second':len(two),
                      'first_diverging_index':bad},indent=2))
    sys.exit(1)
if len(one)<2:sys.exit('FAIL: too few frames to establish determinism')
print(json.dumps({'result':'PASS','identical_frames':len(one),'seed':a.seed,
     'stage':a.stage,'final_gameplay_digest':one[-1][1],
     'observed_max_projectiles':max(x[2] for x in one)},indent=2))
