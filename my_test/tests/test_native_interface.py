import ctypes as C
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from native_rl.env import TaiseiEnv,EnvConfig,action_buttons
from native_rl.taisei_ffi import (Action,State,Projectile,Vec2,PlayerState,BossState,Buffers,SIZE_CHECKS,BUTTON_FOCUS,BUTTON_SHOT,BUTTON_BOMB,BUTTON_SPECIAL,STATUS_RUNNING,STATUS_WON)

class FakeNative:
    """Interface test only; does NOT simulate or represent actual Taisei gameplay."""
    def __init__(self):
        self.i=0
        self.reset_kwargs=None
        self.calls=[]
    def _state(self,i):
        st=State()
        st.episode_status=STATUS_WON if i==2 else STATUS_RUNNING
        st.logical_frame=i
        st.graze=i
        st.deaths=0
        st.boss.active=1
        st.boss.max_hp=100
        st.boss.hp=100-i
        st.boss.recent_damage=1 if i else 0
        st.player.lives=3
        st.player.alive=1
        st.player.bombs=0
        st.bombs_used=0
        st.player.position=Vec2(240,500)
        st.boss.position=Vec2(240,70)
        st.projectile_count=1
        p=Projectile()
        p.category=1
        p.position=Vec2(200,400)
        p.velocity=Vec2(2,2)
        return st,{'projectiles':[p],'items':[],'lasers':[],'laser_points':[]}
    def reset(self,**kwargs):
        self.reset_kwargs=kwargs
        self.i=0
        return self._state(0)
    def step(self,buttons,frames):
        self.calls.append((buttons,frames))
        self.i+=1
        return self._state(self.i)
    def close(self):pass

class TestNativeInterface(unittest.TestCase):
    def test_skill_actions_are_one_call_pulses_and_leave_legacy_actions_unchanged(self):
        self.assertEqual(BUTTON_BOMB,64)
        self.assertEqual(BUTTON_SPECIAL,128)
        native=FakeNative()
        with TaiseiEnv(EnvConfig(library_path='fake',frames_per_action=3),native=native) as env:
            env.reset()
            env.step((0,1))
            self.assertEqual(native.calls[-1],(BUTTON_SHOT|BUTTON_BOMB,3))
            env.step((9,2))
            self.assertEqual(native.calls[-1],(BUTTON_SHOT|BUTTON_FOCUS|BUTTON_SPECIAL,3))
            env.reset()
            env.step(0)
            self.assertEqual(native.calls[-1],(BUTTON_SHOT,3))

    def test_survival_reward_is_only_awarded_while_player_is_alive(self):
        class AliveTransitionNative(FakeNative):
            def _state(self, i):
                result = super()._state(i)
                result[0].player.alive = int(i == 1)
                return result

        self.assertEqual(EnvConfig(library_path='fake').survival_weight, 1.0)
        native=AliveTransitionNative()
        with TaiseiEnv(EnvConfig(library_path='fake'),native=native) as env:
            env.reset()
            _, alive_reward, _, _, alive_info = env.step(0)
            _, dead_reward, _, _, dead_info = env.step(0)

        self.assertAlmostEqual(alive_info['survival_reward'], 1.0)
        self.assertEqual(dead_info['survival_reward'], 0.0)
        # Existing graze/damage/death and terminal terms remain independent.
        self.assertAlmostEqual(alive_reward, 1.0+.001+.02)
        self.assertAlmostEqual(dead_reward, .001+.02+20.0)

    def test_bomb_request_without_stock_is_attempted_but_not_confirmed(self):
        with TaiseiEnv(EnvConfig(library_path='fake'),native=FakeNative()) as env:
            env.reset()
            *_, info = env.step((0, 1))
        self.assertEqual(info['requested_skill'], 1)
        self.assertEqual(info['bombs_before'], 0)
        self.assertEqual(info['bombs_after'], 0)
        self.assertEqual(info['bombs_used_delta'], 0)
        self.assertFalse(info['bomb_use_confirmed'])

    def test_native_bombs_used_increment_confirms_use(self):
        class BombNative(FakeNative):
            def _state(self, i):
                result = super()._state(i)
                if i == 0:
                    result[0].player.bombs = 2
                else:
                    result[0].player.bombs = 1
                    result[0].bombs_used = 1
                    result[0].player.bomb_active = 1
                return result

        with TaiseiEnv(EnvConfig(library_path='fake'),native=BombNative()) as env:
            env.reset()
            *_, info = env.step((0, 1))
        self.assertEqual(info['bombs_used_delta'], 1)
        self.assertTrue(info['bomb_use_confirmed'])
        self.assertEqual((info['bomb_active_before'], info['bomb_active_after']), (0, 1))

    def test_special_request_is_distinct_from_surge_activation_and_charge_transitions(self):
        class SurgeNative(FakeNative):
            def _state(self, i):
                result = super()._state(i)
                p = result[0].player
                if i == 0:
                    p.stored_power = 400
                    p.power_surge_active = 0
                else:
                    p.stored_power = 200
                    p.power_surge_active = 1
                    p.power_surge_positive = 99
                    p.power_surge_negative = 0
                return result

        with TaiseiEnv(EnvConfig(library_path='fake'),native=SurgeNative()) as env:
            env.reset()
            *_, info = env.step((0, 2))
        self.assertEqual(info['requested_skill'], 2)
        self.assertEqual((info['power_surge_active_before'], info['power_surge_active_after']), (0, 1))
        self.assertEqual((info['power_surge_positive_before'], info['power_surge_positive_after']), (0, 99))
        self.assertEqual((info['power_surge_negative_before'], info['power_surge_negative_after']), (0, 0))

    def test_deactivation_and_charge_change_are_retained_without_request(self):
        class SurgeNative(FakeNative):
            def _state(self, i):
                result = super()._state(i)
                p = result[0].player
                p.power_surge_active = int(i == 0)
                p.power_surge_positive = 90 if i == 0 else 0
                p.power_surge_negative = 10 if i == 0 else 0
                return result

        with TaiseiEnv(EnvConfig(library_path='fake'),native=SurgeNative()) as env:
            env.reset()
            *_, info = env.step(0)
        self.assertEqual(info['requested_skill'], 0)
        self.assertEqual((info['power_surge_active_before'], info['power_surge_active_after']), (1, 0))
        self.assertEqual((info['power_surge_positive_before'], info['power_surge_positive_after']), (90, 0))

    def test_action_count_and_focus(self):
        self.assertEqual(len({action_buttons(i) for i in range(18)}),18)
        self.assertFalse(action_buttons(0)&BUTTON_FOCUS)
        self.assertTrue(action_buttons(9)&BUTTON_FOCUS)
        self.assertTrue(action_buttons(0)&BUTTON_SHOT)
        with self.assertRaises(ValueError):action_buttons(18)
    def test_struct_size(self):
        self.assertEqual(C.sizeof(Vec2),16)
        self.assertEqual(C.sizeof(Action),8)
        self.assertGreater(C.sizeof(State),C.sizeof(PlayerState))
        self.assertGreater(len(SIZE_CHECKS),10)
    def test_training_gym_step_contract(self):
        c=EnvConfig(library_path='not-used-in-fake',max_frames=5)
        with TaiseiEnv(c,native=FakeNative()) as env:
            o,i=env.reset(seed=4)
            self.assertEqual(len(o['bullets']),64)
            self.assertEqual(len(o['lasers']),16)
            self.assertEqual(o['graze'],0)
            o,r,d,t,i=env.step(0)
            self.assertAlmostEqual(r,1.0+.001+.02)
            self.assertFalse(d)
            o,r,d,t,i=env.step(0)
            self.assertTrue(d)
            self.assertTrue(i['won'])
            self.assertAlmostEqual(r,21.021)
            with self.assertRaises(RuntimeError): env.step(0)
    def test_practice_mode_defaults_false_on_reset(self):
        native=FakeNative()
        with TaiseiEnv(EnvConfig(library_path='not-used-in-fake'),native=native) as env:
            env.reset(seed=7)
        self.assertIs(native.reset_kwargs['practice'],False)

    def test_practice_mode_forwarded_on_reset(self):
        native=FakeNative()
        with TaiseiEnv(EnvConfig(library_path='not-used-in-fake',practice=True),native=native) as env:
            env.reset(seed=7)
        self.assertIs(native.reset_kwargs['practice'],True)

    def test_no_game_fallback(self):
        from native_rl.taisei_ffi import NativeTaisei
        with self.assertRaises(FileNotFoundError): NativeTaisei('/definitely/no/taisei-sim.so')

if __name__=='__main__':unittest.main()
