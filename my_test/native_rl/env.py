"""Gymnasium-compatible (five-value step) adapter over the real Taisei game runtime.

Intentionally no fallback/toy physics: without a compiled upstream engine, it fails.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from .taisei_ffi import (NativeTaisei, BUTTON_UP, BUTTON_DOWN, BUTTON_LEFT,
    BUTTON_RIGHT, BUTTON_FOCUS, BUTTON_SHOT, BUTTON_BOMB, BUTTON_SPECIAL, PROJECTILE_ENEMY,
    STATUS_WON, STATUS_LOST, STATUS_RUNNING, STATUS_ABORTED, STATUS_ERROR)

DIRECTIONS = (0, BUTTON_UP, BUTTON_DOWN, BUTTON_LEFT, BUTTON_RIGHT,
              BUTTON_UP | BUTTON_LEFT, BUTTON_UP | BUTTON_RIGHT,
              BUTTON_DOWN | BUTTON_LEFT, BUTTON_DOWN | BUTTON_RIGHT)

def action_buttons(action: int, auto_shoot: bool = True) -> int:
    if not 0 <= action < 18:
        raise ValueError("Action index must be 0..17")
    return DIRECTIONS[action % 9] | (BUTTON_FOCUS if action >= 9 else 0) | (BUTTON_SHOT if auto_shoot else 0)

@dataclass
class EnvConfig:
    library_path: str
    resource_path: str | None = None
    storage_path: str | None = None
    stage_id: int = 1
    difficulty: int = 2
    character: int = 0
    shot_mode: int = 0
    practice: bool = False
    frames_per_action: int = 1
    max_frames: int = 30000
    max_projectiles: int = 64
    max_laser_samples: int = 16
    survival_weight: float = 1.0
    damage_weight: float = 0.02
    graze_weight: float = 0.001
    death_penalty: float = -10.0
    loss_penalty: float = -10.0
    win_bonus: float = 20.0
    no_bomb_bonus: float = 0.0

class TaiseiEnv:
    def __init__(self, cfg: EnvConfig, native=None):
        if cfg.frames_per_action < 1 or cfg.max_frames < 1:
            raise ValueError("frame counts must be positive")
        if cfg.max_projectiles < 1 or cfg.max_laser_samples < 1:
            raise ValueError("observation capacities must be positive")
        self.cfg = cfg
        self.native = native if native is not None else NativeTaisei(cfg.library_path,
            resource_path=cfg.resource_path, storage_path=cfg.storage_path)
        self.prev = None
        self.elapsed = 0

    def _observe(self, snapshot):
        st, arrays = snapshot
        p, b = st.player, st.boss
        px, py = p.position.x, p.position.y
        # Sort by predicted closest approach, not just current distance.
        threats = []
        for z in arrays["projectiles"]:
            if z.category != PROJECTILE_ENEMY:
                continue
            dx, dy = z.position.x-px, z.position.y-py
            vx, vy = z.velocity.x-p.velocity.x, z.velocity.y-p.velocity.y
            v2 = vx*vx+vy*vy
            t = min(30.0, max(0.0, -(dx*vx+dy*vy)/v2)) if v2 > 1e-9 else 0.0
            closest = math.hypot(dx+vx*t,dy+vy*t)
            threats.append((closest, math.hypot(dx,dy), z))
        threats.sort(key=lambda x:(x[0],x[1]))
        bullets = []
        for _,__,z in threats[:self.cfg.max_projectiles]:
            bullets.append([(z.position.x-px)/480,(z.position.y-py)/560,
                z.velocity.x/20,z.velocity.y/20,z.collision_size.x/30,
                z.collision_size.y/30,1.0])
        while len(bullets)<self.cfg.max_projectiles:
            bullets.append([0.0]*7)
        laser_points = []
        for laser in arrays["lasers"]:
            if not laser.collision_active:
                continue
            for q in arrays["laser_points"][laser.first_point:laser.first_point+laser.point_count]:
                laser_points.append((math.hypot(q.position.x-px,q.position.y-py),q))
        laser_points.sort(key=lambda item:item[0])
        lasers = [[(q.position.x-px)/480,(q.position.y-py)/560,q.half_width/30,1.0]
                  for _,q in laser_points[:self.cfg.max_laser_samples]]
        while len(lasers) < self.cfg.max_laser_samples:
            lasers.append([0.0]*4)
        items=[]
        for item in arrays["items"]:
            items.append([float(item.item_type), (item.position.x-px)/480,
                (item.position.y-py)/560, float(item.age_frames)/3600,
                float(item.collect_frame)/3600, float(item.attracted),
                float(item.pickup_value)])
        timing=[float(st.logical_frame)/3600.0,float(p.death_timer)/120.0,
            float(p.respawn_timer)/120.0,float(p.recovery_timer)/120.0,
            float(p.bomb_progress),float(p.bomb_trigger_frame)/3600.0,
            float(p.bomb_end_frame)/3600.0]
        skill_state=[float(p.alive),float(p.invulnerable),float(p.recovering),
            float(p.bombs)/8.0,float(p.bomb_fragments)/8.0,
            float(p.stored_power)/400.0,float(p.effective_power)/400.0,
            float(p.bomb_active),float(p.power_surge_active),
            float(p.power_surge_positive)/100.0,float(p.power_surge_negative)/100.0]
        return {"player": [px/480,py/560,p.velocity.x/20,p.velocity.y/20,
                    float(p.focused),float(p.lives),float(p.bombs),float(p.invulnerable)],
                "boss": [float(b.active),(b.position.x-px)/480,(b.position.y-py)/560,
                    b.hp/max(b.max_hp,1.0),float(b.phase_type),float(b.remaining_timeout_frames)/3600],
                "bullets":bullets,"lasers":lasers,"projectile_count":int(st.projectile_count),
                "laser_count":int(st.laser_count),"graze":int(st.graze),
                "deaths":int(st.deaths),"status":int(st.episode_status),
                "gameplay_digest":int(st.gameplay_digest),"timing":timing,
                "skill_state":skill_state,"items":items,"score":int(st.score),
                "point_item_value":int(p.point_item_value),"bombs_used":int(st.bombs_used),
                "logical_frame":int(st.logical_frame)}

    def reset(self, seed=1234):
        self.elapsed = 0
        self.prev = self.native.reset(seed=seed,stage_id=self.cfg.stage_id,
            difficulty=self.cfg.difficulty,character=self.cfg.character,
            shot_mode=self.cfg.shot_mode,practice=self.cfg.practice)
        obs = self._observe(self.prev)
        return obs,{"seed":seed,"stage_id":self.cfg.stage_id}

    def step(self, action):
        if self.prev is None:
            raise RuntimeError("Reset before step")
        before = self.prev[0]
        if before.episode_status != STATUS_RUNNING:
            raise RuntimeError("Game episode is no longer running; reset")
        frames=min(self.cfg.frames_per_action,self.cfg.max_frames-self.elapsed)
        if frames <= 0:
            raise RuntimeError("Time limit reached; reset")
        skill = 0
        move = action
        if isinstance(action, (tuple, list)):
            if len(action) != 2:
                raise ValueError("Skill action must be (movement, skill)")
            move, skill = action
            if not 0 <= int(skill) <= 2:
                raise ValueError("Skill index must be 0 (none), 1 (bomb), or 2 (special)")
            move = int(move)
            skill = int(skill)
        buttons = action_buttons(move)
        if skill == 1:
            buttons |= BUTTON_BOMB
        elif skill == 2:
            buttons |= BUTTON_SPECIAL
        after=self.native.step(buttons,frames=frames)
        self.elapsed += frames
        st=after[0]
        if st.episode_status in (STATUS_ABORTED, STATUS_ERROR):
            raise RuntimeError(f"Native Taisei episode aborted/errored: status={st.episode_status}")
        survival_frames=max(0,int(st.logical_frame)-int(before.logical_frame))
        player_before, player_after = before.player, st.player
        bombs_used_delta = int(st.bombs_used) - int(before.bombs_used)
        skill_telemetry = {
            "requested_skill": int(skill),
            "logical_frame_before": int(before.logical_frame),
            "logical_frame_after": int(st.logical_frame),
            "player_alive_before": bool(player_before.alive),
            "player_alive_after": bool(player_after.alive),
            "bombs_before": int(player_before.bombs),
            "bombs_after": int(player_after.bombs),
            "bombs_used_before": int(before.bombs_used),
            "bombs_used_after": int(st.bombs_used),
            "bombs_used_delta": bombs_used_delta,
            "bomb_use_confirmed": bombs_used_delta > 0,
            "bomb_active_before": int(player_before.bomb_active),
            "bomb_active_after": int(player_after.bomb_active),
            "power_surge_active_before": int(player_before.power_surge_active),
            "power_surge_active_after": int(player_after.power_surge_active),
            "power_surge_positive_before": float(player_before.power_surge_positive),
            "power_surge_positive_after": float(player_after.power_surge_positive),
            "power_surge_negative_before": float(player_before.power_surge_negative),
            "power_surge_negative_after": float(player_after.power_surge_negative),
        }
        # At one-frame action boundaries, the resulting native alive flag tells
        # whether this frame was survived. Wider strides hide intermediate states.
        survival_reward=(survival_frames*self.cfg.survival_weight
            if self.cfg.frames_per_action == 1 and bool(st.player.alive) else 0.0)
        reward=(survival_reward
               +max(0,int(st.graze)-int(before.graze))*self.cfg.graze_weight
               +max(0.0,float(st.boss.recent_damage))*self.cfg.damage_weight
               +self.cfg.death_penalty*max(0,int(st.deaths)-int(before.deaths)))
        terminated=st.episode_status in (STATUS_WON,STATUS_LOST)
        truncated=not terminated and self.elapsed>=self.cfg.max_frames
        if st.episode_status==STATUS_WON:
            reward+=self.cfg.win_bonus
        if terminated and st.episode_status==STATUS_LOST:
            reward+=self.cfg.loss_penalty
        obs=self._observe(after)
        info={"frames":self.elapsed,"survival_reward":float(survival_reward),
              "boss_damage":float(st.boss.recent_damage),
              "graze_delta":int(st.graze)-int(before.graze),
              "death_delta":int(st.deaths)-int(before.deaths),
              "won":st.episode_status==STATUS_WON,"digest":int(st.gameplay_digest),
              "status":int(st.episode_status), **skill_telemetry}
        self.prev=after
        return obs,float(reward),bool(terminated),bool(truncated),info

    def close(self): self.native.close()
    def __enter__(self): return self
    def __exit__(self,*_): self.close()
