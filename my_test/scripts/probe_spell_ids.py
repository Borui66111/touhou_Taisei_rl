#!/usr/bin/env python3
"""Find real spell-practice episode IDs in Taisei's native game registry.

Taisei stageinfo.c computes IDs as 0x8000 | (origin_stage << 8) | spell_id.
This script probes IDs by asking the native engine; it does NOT invent patterns.
"""
import argparse,pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from native_rl.taisei_ffi import NativeTaisei
p=argparse.ArgumentParser()
p.add_argument('--library',required=True);p.add_argument('--resource-path')
p.add_argument('--origin-stage',type=int,default=1,choices=range(1,7))
p.add_argument('--difficulty',type=int,default=2)
p.add_argument('--max-offset',type=int,default=100)
a=p.parse_args()
found=[]
with NativeTaisei(a.library,resource_path=a.resource_path) as sim:
    for off in range(a.max_offset):
        id_=0x8000 | (a.origin_stage<<8) | off
        try:
            st,_=sim.reset(seed=1402,stage_id=id_,difficulty=a.difficulty,practice=True)
        except RuntimeError as e:
            if 'Unknown stage ID' in str(e):continue
            # A registered stage may require additional resources. Do not hide this.
            print(f'{id_}: non-lookup error: {e}',file=sys.stderr)
            raise
        print(f'id={id_} stage_type={st.stage_type} diff={st.difficulty} phase={st.boss.phase_type}')
        found.append(id_)
print(f'Found {len(found)} registered spell-practice stage IDs in inspected range.')
if not found:sys.exit('No spells found. Increase max-offset or inspect the native stage registry.')
