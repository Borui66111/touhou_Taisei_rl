"""Optional Gymnasium wrapper. `pip install gymnasium numpy stable-baselines3`."""
from __future__ import annotations
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError("pip install gymnasium numpy stable-baselines3") from exc
from .env import TaiseiEnv, EnvConfig

def flatten(obs):
    return np.asarray(obs['player']+obs['boss']+
        [x for bullet in obs['bullets'] for x in bullet] +
        [x for laser in obs['lasers'] for x in laser] +
        [obs['projectile_count']/1000,obs['laser_count']/50],dtype=np.float32)

class TaiseiGym(gym.Env):
    metadata={"render_modes":[]}
    def __init__(self, cfg: EnvConfig):
        self.native_env=TaiseiEnv(cfg)
        self.observation_space=spaces.Box(low=-1e7,high=1e7,
            shape=(8+6+cfg.max_projectiles*7+cfg.max_laser_samples*4+2,),dtype=np.float32)
        self.action_space=spaces.Discrete(18)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        obs,info=self.native_env.reset(seed=int(self.np_random.integers(1,2**31)) if seed is None else int(seed))
        return flatten(obs),info
    def step(self,action):
        o,r,d,t,i=self.native_env.step(int(action))
        return flatten(o),r,d,t,i
    def close(self): self.native_env.close()
