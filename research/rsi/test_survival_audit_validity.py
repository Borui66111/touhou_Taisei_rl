"""RSI-AUDIT-VALIDITY regressions. Synthetic trace ROWS only; not gameplay evidence."""
import importlib.util
import math
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from survival_dodge_audit import analyze, InvalidTrace


def row(i=0, *, x=16.0, y=270.0, action=3, invulnerable=False, alive_after=True,
        deaths=0, post_deaths=0, threat=False, laser=None, laser_count=0, status='absent'):
    near = ({'predicted_closest_separation_px': 7.0, 'time_to_closest_frames': 8.0,
             'collision_width_px': 3.0, 'collision_height_px': 3.0} if threat else None)
    pre = {'player_x': x/480, 'player_y': y/560, 'alive': True, 'deaths': deaths,
           'nearest_projectile': near, 'laser_count': laser_count,
           'laser_geometry_status': status, 'nearest_laser_point': laser}
    if invulnerable is not Ellipsis:
        pre['invulnerable'] = invulnerable
    return {'seed': 42152, 'action_index': i, 'movement_action': action, 'skill_action': 0,
            'pre': pre,
            'post': {'player_x': x/480, 'player_y': y/560, 'alive': alive_after,
                     'deaths': post_deaths}}


class UnknownVulnerability(unittest.TestCase):
    def test_missing_invulnerable_is_unknown_not_vulnerable(self):
        m = analyze([row(invulnerable=Ellipsis), row(1, invulnerable=Ellipsis,
                                                    alive_after=False, post_deaths=1)])['metrics']
        self.assertEqual(m['unknown_vulnerability_decisions'], 2)
        self.assertEqual(m['unknown_vulnerability_death_transitions'], 1)
        self.assertEqual(m['pre_vulnerable_decisions'], 0)
        self.assertEqual(m['eligible_vulnerable_steps'], 0)
        self.assertEqual(m['pre_vulnerable_death_transitions'], 0)

    def test_null_invulnerable_is_unknown(self):
        m = analyze([row(invulnerable=None)])['metrics']
        self.assertEqual(m['unknown_vulnerability_decisions'], 1)
        self.assertEqual(m['pre_vulnerable_decisions'], 0)

    def test_known_false_still_vulnerable(self):
        m = analyze([row()])['metrics']
        self.assertEqual(m['pre_vulnerable_decisions'], 1)
        self.assertEqual(m['unknown_vulnerability_decisions'], 0)


class FatalOutwardCommands(unittest.TestCase):
    def test_fatal_outward_wall_command_counted(self):
        # left wall, action 3 = left (outward), dies on this decision
        m = analyze([row(threat=True, alive_after=False, post_deaths=1)])['metrics']
        self.assertEqual(m['fatal_boundary_decisions'], 1)
        self.assertEqual(m['fatal_outward_command_decisions'], 1)
        self.assertEqual(m['fatal_outward_command_under_threat_decisions'], 1)
        self.assertEqual(m['fraction_fatal_decisions_command_outward'], 1.0)
        self.assertEqual(m['outward_command_steps'], 0)  # survivor metrics unchanged
        self.assertEqual(m['pinned_commands'], 0)

    def test_fatal_inward_and_midfield_not_outward(self):
        m = analyze([row(action=4, alive_after=False, post_deaths=1),
                     row(1, x=240, action=3, deaths=1, post_deaths=2, alive_after=False)])['metrics']
        self.assertEqual(m['pre_vulnerable_death_transitions'], 2)
        self.assertEqual(m['fatal_boundary_decisions'], 1)
        self.assertEqual(m['fatal_outward_command_decisions'], 0)
        self.assertEqual(m['fraction_fatal_decisions_command_outward'], 0.0)

    def test_no_deaths_fraction_none(self):
        self.assertIsNone(analyze([row()])['metrics']['fraction_fatal_decisions_command_outward'])


class LaserGeometry(unittest.TestCase):
    near_laser = {'current_separation_px': 10.0, 'half_width_px': 4.0}
    far_laser = {'current_separation_px': 200.0, 'half_width_px': 4.0}

    def test_near_laser_death_attributed_to_laser_proxy_not_bullet(self):
        m = analyze([row(x=240, laser=self.near_laser, laser_count=1, status='point_samples',
                         alive_after=False, post_deaths=1)])['metrics']
        self.assertEqual(m['pre_vulnerable_laser_proxy_death_transitions'], 1)
        self.assertEqual(m['pre_vulnerable_projectile_proxy_death_transitions'], 0)
        self.assertEqual(m['pre_vulnerable_death_transitions_no_threat_proxy'], 0)
        self.assertEqual(m['pre_vulnerable_death_transitions_lasers_unmeasured'], 0)

    def test_far_laser_not_proxy(self):
        m = analyze([row(x=240, laser=self.far_laser, laser_count=1, status='point_samples')])['metrics']
        self.assertEqual(m['pre_vulnerable_laser_proxy_steps'], 0)

    def test_legacy_trace_lasers_present_but_unmeasured_flagged(self):
        r = row(x=240, laser_count=3, status=None, alive_after=False, post_deaths=1)
        del r['pre']['laser_geometry_status'], r['pre']['nearest_laser_point']
        m = analyze([r])['metrics']
        self.assertEqual(m['pre_vulnerable_lasers_unmeasured_steps'], 1)
        self.assertEqual(m['pre_vulnerable_death_transitions_lasers_unmeasured'], 1)
        self.assertEqual(m['pre_vulnerable_death_transitions_no_threat_proxy'], 1)

    def test_no_lasers_not_flagged(self):
        m = analyze([row(x=240)])['metrics']
        self.assertEqual(m['pre_vulnerable_lasers_unmeasured_steps'], 0)

    def test_invalid_status_and_geometry_rejected(self):
        with self.assertRaisesRegex(InvalidTrace, 'laser_geometry_status'):
            analyze([row(status='segments')])
        with self.assertRaisesRegex(InvalidTrace, 'negative laser'):
            analyze([row(laser={'current_separation_px': -1.0, 'half_width_px': 1.0},
                         laser_count=1, status='point_samples')])

    def test_schema_bumped(self):
        self.assertEqual(analyze([row()])['schema'], 'taisei_survival_dodge_wall_audit_v2')


@unittest.skipUnless(importlib.util.find_spec('torch'), 'torch not installed')
class TraceWriter(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(HERE.parents[1] / 'my_test'))
        from native_rl import torch_ppo
        self.tp = torch_ppo

    def test_nearest_laser_decoding(self):
        obs = {'laser_count': 2, 'lasers': [[0.0, 0.0, 0.0, 0.0], [30/480, 40/560, 6/30, 1.0]]}
        q = self.tp._nearest_laser_trace(obs)
        self.assertTrue(math.isclose(q['current_separation_px'], 50.0))
        self.assertTrue(math.isclose(q['half_width_px'], 6.0))
        self.assertEqual(self.tp._laser_geometry_status(obs), 'point_samples')

    def test_laser_status_absent_and_unmeasured(self):
        self.assertEqual(self.tp._laser_geometry_status({'laser_count': 0, 'lasers': []}), 'absent')
        self.assertIsNone(self.tp._laser_geometry_status({'laser_count': 1, 'lasers': [[0, 0, 0, 0]]}))


class TraceWriterSource(unittest.TestCase):
    """Source-level guard that does not need torch."""
    def test_writer_emits_none_for_unknown_invulnerability(self):
        src = (HERE.parents[1] / 'my_test/native_rl/torch_ppo.py').read_text(encoding='utf-8')
        self.assertIn("'invulnerable':bool(player[7]) if len(player)>7 else None", src)
        self.assertNotIn("if len(player)>7 else False", src)
        self.assertIn("'laser_geometry_status':_laser_geometry_status(obs)", src)


if __name__ == '__main__':
    unittest.main()
