"""Transparent, inexpensive baseline controller using native bullet observations.
Not expert-level and not a replacement for the actual game physics.
"""
from __future__ import annotations
import math
from .env import DIRECTIONS
from .taisei_ffi import BUTTON_UP, BUTTON_DOWN, BUTTON_LEFT, BUTTON_RIGHT


def heuristic_action(obs, horizon=16):
    # Playfield units per *frame*. This is a heuristic, not authoritative
    # for all Taisei characters / focus speeds.
    px=float(obs['player'][0])*480
    py=float(obs['player'][1])*560
    boss_dx=float(obs['boss'][1])*480
    is_boss=bool(obs['boss'][0])
    best=-float('inf'); selected=0
    for a in range(18):
        d=DIRECTIONS[a%9]
        dx=float(bool(d&BUTTON_RIGHT))-float(bool(d&BUTTON_LEFT))
        dy=float(bool(d&BUTTON_DOWN))-float(bool(d&BUTTON_UP))
        norm=math.hypot(dx,dy) or 1
        speed=2.5 if a>=9 else 5.5
        vx,vy=dx/norm*speed,dy/norm*speed
        nx=max(5,min(475,px+vx*horizon))
        ny=max(5,min(555,py+vy*horizon))
        # Encourage alignment under a boss; slight preference for survival zone.
        score=(-abs(boss_dx + px -nx)*0.035 if is_boss else -abs(nx-240)*0.008) \
            -abs(ny-490)*0.006
        for b in obs['bullets']:
            if not b[-1]:continue  # ignore padding
            rx=b[0]*480;ry=b[1]*560
            bvx=b[2]*20;bvy=b[3]*20
            rvx=bvx-vx;rvy=bvy-vy
            t=max(0,min(horizon,-(rx*rvx+ry*rvy)/(rvx*rvx+rvy*rvy+1e-6)))
            sep=math.hypot(rx+rvx*t,ry+rvy*t)
            danger_radius=6.0+max(abs(b[4]),abs(b[5]))*30
            score-=120.0*math.exp(-max(0,sep-danger_radius)/10)
        for q in obs['lasers']:
            if not q[-1]:continue
            sep=math.hypot(q[0]*480-vx*horizon,q[1]*560-vy*horizon)
            score-=60*math.exp(-max(0,sep-q[2]*30)/14)
        if score>best:best=score;selected=a
    return selected
