"""ctypes binding for touhourl/taisei-sim ABI v1 (verified against upstream taisei_sim.h).

No imitation of gameplay physics is present here. All dynamics come from the native
Taisei game library. Native ABI sizes are checked before any gameplay calls.
"""
from __future__ import annotations
import ctypes as C
import os

API_VERSION = 1
DEFAULT_I32 = -2147483647
DEFAULT_U64 = 0xFFFFFFFFFFFFFFFF
BUTTON_UP, BUTTON_DOWN, BUTTON_LEFT, BUTTON_RIGHT = 1, 2, 4, 8
BUTTON_FOCUS, BUTTON_SHOT, BUTTON_BOMB, BUTTON_SPECIAL = 16, 32, 64, 128
STATUS_RUNNING, STATUS_WON, STATUS_LOST, STATUS_ABORTED, STATUS_ERROR = 1, 2, 3, 4, 5
PROJECTILE_ENEMY = 1

class Vec2(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double)]

class GlobalConfig(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("api_version", C.c_uint32),
                ("resource_path", C.c_char_p), ("storage_path", C.c_char_p),
                ("cache_path", C.c_char_p), ("flags", C.c_uint32), ("reserved", C.c_uint32)]

class SimConfig(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("flags", C.c_uint32),
                ("laser_sample_step", C.c_double), ("reserved", C.c_uint32)]

class EpisodeConfig(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("stage_id", C.c_uint16),
                ("difficulty", C.c_uint8), ("player_character", C.c_uint8),
                ("shot_mode", C.c_uint8), ("practice_mode", C.c_uint8),
                ("reserved0", C.c_uint16), ("rng_seed", C.c_uint64),
                ("start_time", C.c_uint64), ("initial_lives", C.c_int32),
                ("initial_bombs", C.c_int32), ("initial_life_fragments", C.c_int32),
                ("initial_bomb_fragments", C.c_int32), ("initial_power", C.c_int32),
                ("initial_point_item_value", C.c_int32), ("initial_score", C.c_uint64),
                ("initial_graze", C.c_uint32), ("reserved1", C.c_uint32)]

class Action(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("buttons", C.c_uint32)]

class PlayerState(C.Structure):
    _fields_ = [("position", Vec2), ("previous_position", Vec2), ("velocity", Vec2),
                ("input_flags", C.c_uint32), ("focused", C.c_uint32),
                ("shooting", C.c_uint32), ("lives", C.c_int32),
                ("bombs", C.c_int32), ("life_fragments", C.c_int32),
                ("bomb_fragments", C.c_int32), ("stored_power", C.c_int32),
                ("effective_power", C.c_int32), ("point_item_value", C.c_uint32),
                ("score", C.c_uint64), ("graze", C.c_uint32), ("voltage", C.c_uint32),
                ("invulnerable", C.c_uint32), ("recovering", C.c_uint32),
                ("alive", C.c_uint32), ("death_timer", C.c_int32),
                ("respawn_timer", C.c_int32), ("recovery_timer", C.c_int32),
                ("bomb_active", C.c_uint32), ("bomb_progress", C.c_double),
                ("bomb_trigger_frame", C.c_int32), ("bomb_end_frame", C.c_int32),
                ("power_surge_active", C.c_uint32), ("power_surge_positive", C.c_float),
                ("power_surge_negative", C.c_float), ("player_character", C.c_uint8),
                ("shot_mode", C.c_uint8), ("reserved", C.c_uint16)]

class BossState(C.Structure):
    _fields_ = [("active", C.c_uint32), ("position", Vec2), ("velocity", Vec2),
                ("hp", C.c_float), ("max_hp", C.c_float),
                ("invulnerable", C.c_uint32), ("phase_index", C.c_int32),
                ("phase_type", C.c_uint32), ("attack_id", C.c_uint16),
                ("spell_id", C.c_uint16), ("spell_active", C.c_uint32),
                ("phase_start_frame", C.c_int32), ("phase_end_frame", C.c_int32),
                ("phase_timeout_frames", C.c_int32), ("remaining_timeout_frames", C.c_int32),
                ("spell_failed_frame", C.c_int32), ("spell_failed", C.c_uint32),
                ("spell_captured", C.c_uint32), ("recent_damage", C.c_float)]

class Projectile(C.Structure):
    _fields_ = [("spawn_id", C.c_uint32), ("category", C.c_uint32),
                ("position", Vec2), ("previous_position", Vec2), ("velocity", Vec2),
                ("collision_size", Vec2), ("damage", C.c_float), ("angle", C.c_float),
                ("age_frames", C.c_int32), ("flags", C.c_uint32),
                ("damage_type", C.c_uint32), ("clear_flags", C.c_uint32)]

class Enemy(C.Structure):
    _fields_ = [("spawn_id", C.c_uint32), ("position", Vec2), ("velocity", Vec2),
                ("hp", C.c_float), ("max_hp", C.c_float), ("hit_radius", C.c_float),
                ("hurt_radius", C.c_float), ("age_frames", C.c_int32),
                ("flags", C.c_uint32), ("damageable", C.c_uint32), ("harmful", C.c_uint32)]

class Item(C.Structure):
    _fields_ = [("spawn_id", C.c_uint32), ("position", Vec2), ("velocity", Vec2),
                ("item_type", C.c_uint32), ("age_frames", C.c_int32),
                ("collect_frame", C.c_int32), ("attracted", C.c_uint32),
                ("pickup_value", C.c_float)]

class Laser(C.Structure):
    _fields_ = [("spawn_id", C.c_uint32), ("origin", Vec2), ("age_frames", C.c_int32),
                ("width", C.c_float), ("speed", C.c_float), ("timespan", C.c_float),
                ("death_time", C.c_float), ("time_shift", C.c_float),
                ("collision_active", C.c_uint32), ("unclearable", C.c_uint32),
                ("clear_flags", C.c_uint32), ("first_point", C.c_uint32),
                ("point_count", C.c_uint32)]

class LaserPoint(C.Structure):
    _fields_ = [("position", Vec2), ("half_width", C.c_float),
                ("time", C.c_float), ("flags", C.c_uint32)]

class State(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("api_version", C.c_uint32),
                ("stage_id", C.c_uint16), ("stage_type", C.c_uint16),
                ("difficulty", C.c_uint8), ("player_character", C.c_uint8),
                ("shot_mode", C.c_uint8), ("practice_mode", C.c_uint8),
                ("logical_frame", C.c_uint64), ("initial_rng_seed", C.c_uint64),
                ("start_time", C.c_uint64), ("episode_status", C.c_uint32),
                ("stage_clear", C.c_uint32), ("game_over", C.c_uint32),
                ("reserved0", C.c_uint32), ("score", C.c_uint64),
                ("graze", C.c_uint32), ("voltage", C.c_uint32),
                ("deaths", C.c_uint32), ("bombs_used", C.c_uint32),
                ("continues_used", C.c_uint32), ("gameover_frame", C.c_int32),
                ("reserved2", C.c_int32), ("player", PlayerState),
                ("boss", BossState), ("projectile_count", C.c_uint32),
                ("enemy_count", C.c_uint32), ("item_count", C.c_uint32),
                ("laser_count", C.c_uint32), ("laser_point_count", C.c_uint32),
                ("reserved1", C.c_uint32), ("gameplay_digest", C.c_uint64)]

class Buffers(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("reserved", C.c_uint32),
                ("projectiles", C.POINTER(Projectile)), ("projectile_capacity", C.c_uint32),
                ("enemies", C.POINTER(Enemy)), ("enemy_capacity", C.c_uint32),
                ("items", C.POINTER(Item)), ("item_capacity", C.c_uint32),
                ("lasers", C.POINTER(Laser)), ("laser_capacity", C.c_uint32),
                ("laser_points", C.POINTER(LaserPoint)), ("laser_point_capacity", C.c_uint32)]

SIZE_CHECKS = {"vec2": Vec2, "global_config": GlobalConfig, "config": SimConfig,
    "episode_config": EpisodeConfig, "action": Action, "state": State,
    "state_buffers": Buffers, "player_state": PlayerState, "boss_state": BossState,
    "projectile_state": Projectile, "enemy_state": Enemy, "item_state": Item,
    "laser_state": Laser, "laser_point": LaserPoint}

def constructed(cls, **kwargs):
    obj = cls(**kwargs)
    if hasattr(obj, "struct_size"):
        obj.struct_size = C.sizeof(cls)
    return obj

class NativeTaisei:
    """One Taisei simulation per process, per upstream API restriction."""
    def __init__(self, library_path: str, resource_path: str | None = None,
                 storage_path: str | None = None):
        if not os.path.isfile(library_path):
            raise FileNotFoundError(f"Taisei native simulation not found: {library_path}")
        self.lib = C.CDLL(os.path.abspath(library_path))
        self.handle = C.c_void_p()
        self._initialized = False
        self._configure_calls()
        version = self.lib.taisei_sim_api_version()
        if version != API_VERSION:
            raise RuntimeError(f"Taisei API {version} != supported API {API_VERSION}")
        for name, typ in SIZE_CHECKS.items():
            size = getattr(self.lib, "taisei_sim_sizeof_" + name)()
            if size != C.sizeof(typ):
                raise RuntimeError(f"Taisei ABI mismatch for {name}: C={size}, Python={C.sizeof(typ)}")
        cfg = constructed(GlobalConfig, api_version=API_VERSION,
            resource_path=os.fsencode(resource_path) if resource_path else None,
            storage_path=os.fsencode(storage_path) if storage_path else None)
        self._check(self.lib.taisei_sim_global_init(C.byref(cfg)))
        self._initialized = True
        conf = constructed(SimConfig)
        try:
            self._check(self.lib.taisei_sim_create(C.byref(conf), C.byref(self.handle)))
        except Exception:
            self.close()
            raise

    def _configure_calls(self):
        lib = self.lib
        for name in ["taisei_sim_api_version"]:
            getattr(lib, name).restype = C.c_uint32
        for name in SIZE_CHECKS:
            getattr(lib, "taisei_sim_sizeof_" + name).restype = C.c_size_t
        for name, args in {
            "taisei_sim_global_init": [C.POINTER(GlobalConfig)],
            "taisei_sim_global_shutdown": [],
            "taisei_sim_create": [C.POINTER(SimConfig), C.POINTER(C.c_void_p)],
            "taisei_sim_reset": [C.c_void_p, C.POINTER(EpisodeConfig)],
            "taisei_sim_step": [C.c_void_p, C.POINTER(Action), C.c_uint32],
            "taisei_sim_get_state": [C.c_void_p, C.POINTER(State), C.POINTER(Buffers)],
            "taisei_sim_destroy": [C.c_void_p],
            "taisei_sim_save_replay": [C.c_void_p, C.c_char_p],
        }.items():
            fn = getattr(lib, name)
            fn.argtypes = args
            fn.restype = C.c_int32
        lib.taisei_sim_last_error.argtypes = [C.c_void_p]
        lib.taisei_sim_last_error.restype = C.c_char_p

    def _check(self, result: int):
        if result != 0:
            msg = self.lib.taisei_sim_last_error(self.handle)
            raise RuntimeError(f"Taisei error {result}: {(msg or b'unknown').decode(errors='replace')}")

    def reset(self, *, seed: int, stage_id: int = 1, difficulty: int = 2,
              character: int = 0, shot_mode: int = 0, practice: bool = False):
        config = constructed(EpisodeConfig, stage_id=stage_id, difficulty=difficulty,
            player_character=character, shot_mode=shot_mode, practice_mode=int(practice),
            rng_seed=seed, start_time=DEFAULT_U64,
            initial_lives=DEFAULT_I32, initial_bombs=DEFAULT_I32,
            initial_life_fragments=DEFAULT_I32, initial_bomb_fragments=DEFAULT_I32,
            initial_power=DEFAULT_I32, initial_point_item_value=DEFAULT_I32)
        self._check(self.lib.taisei_sim_reset(self.handle, C.byref(config)))
        return self.get_state()

    def step(self, buttons: int, frames: int = 1):
        act = constructed(Action, buttons=buttons)
        self._check(self.lib.taisei_sim_step(self.handle, C.byref(act), frames))
        return self.get_state()

    def get_state(self):
        state = constructed(State)
        # Initial snapshot obtains exact counts; reallocate on every frame to avoid stale capacity.
        ret = self.lib.taisei_sim_get_state(self.handle, C.byref(state), None)
        if ret not in (0, -11):
            self._check(ret)
        arrays = {"projectiles": (Projectile * state.projectile_count)(),
                  "enemies": (Enemy * state.enemy_count)(),
                  "items": (Item * state.item_count)(),
                  "lasers": (Laser * state.laser_count)(),
                  "laser_points": (LaserPoint * state.laser_point_count)()}
        buffers = constructed(Buffers, **{
            "projectiles": arrays["projectiles"], "projectile_capacity": state.projectile_count,
            "enemies": arrays["enemies"], "enemy_capacity": state.enemy_count,
            "items": arrays["items"], "item_capacity": state.item_count,
            "lasers": arrays["lasers"], "laser_capacity": state.laser_count,
            "laser_points": arrays["laser_points"], "laser_point_capacity": state.laser_point_count})
        self._check(self.lib.taisei_sim_get_state(self.handle, C.byref(state), C.byref(buffers)))
        return state, arrays

    def save_replay(self, path: str):
        self._check(self.lib.taisei_sim_save_replay(self.handle, os.fsencode(path)))

    def close(self):
        if self.handle.value:
            self._check(self.lib.taisei_sim_destroy(self.handle))
            self.handle = C.c_void_p()
        if self._initialized:
            self._check(self.lib.taisei_sim_global_shutdown())
            self._initialized = False

    def __enter__(self): return self
    def __exit__(self, *_): self.close()
