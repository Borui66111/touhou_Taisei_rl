"""Opt-in, source-auditable Taisei native evaluation action veto.

This is an experimental *movement filter*, NOT a new policy ABI, trained
checkpoint, a second live game controller, or a proven dodge oracle. It may be
used only in a controlled skill-NONE native matched-seed comparison. The
risk-aware teacher is a handcrafted fallback only on outward wall commands.
"""
from __future__ import annotations
import math

DIRECTIONS = ((0,0),(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(1,-1),(-1,1),(1,1))
BOUNDS = (16.0,464.0,16.0,544.0)


def outward_command(obs, action, margin=8.0):
    if isinstance(action, bool) or not isinstance(action,int) or not 0 <= action <18:
        raise ValueError('movement action must be int 0..17')
    if isinstance(margin,bool) or not isinstance(margin,(int,float)) or not 0 <= margin <= 32:
        raise ValueError('invalid boundary margin')
    player=obs.get('player') if isinstance(obs,dict) else None
    if not isinstance(player,(list,tuple)) or len(player)<2:
        raise ValueError('missing normalized player state')
    x,y = player[:2]
    if any(isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) for q in (x,y)):
        raise ValueError('nonfinite normalized player position')
    px,py=float(x)*480.0,float(y)*560.0
    left,right,top,bottom=BOUNDS
    dx,dy=DIRECTIONS[action%9]
    return ((px<=left+margin and dx<0) or (px>=right-margin and dx>0)
            or (py<=top+margin and dy<0) or (py>=bottom-margin and dy>0))


def edge_veto(obs, action, *, margin=8.0, fallback=None):
    """Replace ONLY an outward edge command; otherwise preserve actor choice.

    Fallback defaults to fixed E2 risk teacher. A fallback that still points
    outward causes a clear error, not a fabricated improvement or blind move.
    """
    if not outward_command(obs,action,margin):
        return action
    if fallback is None:
        from risk_aware_teacher import risk_aware_action
        fallback = risk_aware_action
    proposed = fallback(obs)
    if isinstance(proposed,bool) or not isinstance(proposed,int) or not 0 <= proposed < 18:
        raise ValueError('risk-aware fallback returned invalid movement')
    if outward_command(obs,proposed,margin):
        raise ValueError('risk-aware fallback still points outside the playable area')
    return proposed
