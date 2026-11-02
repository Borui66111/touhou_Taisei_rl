"""Unit tests use synthetic trace ROWS ONLY; not synthetic game evidence."""
import unittest
import tempfile
import pathlib
import sys
import json

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from survival_dodge_audit import analyze, InvalidTrace, read_jsonl


def row(seed=42160, i=0, x=16.0, y=270.0, nx=None, ny=None, action=3,
        threat=False, alive=True, invulnerable=False, skill=0, deaths=0, post_deaths=None):
    if nx is None:nx=x
    if ny is None:ny=y
    near=({'predicted_closest_separation_px':7.,'time_to_closest_frames':8.,
           'collision_width_px':3.,'collision_height_px':3.} if threat else None)
    return {'seed':seed,'action_index':i,'movement_action':action,'skill_action':skill,
            'pre':{'player_x':x/480.,'player_y':y/560.,'alive':alive,
                   'invulnerable':invulnerable,'deaths':deaths,'nearest_projectile':near},
            'post':{'player_x':nx/480.,'player_y':ny/560.,'alive':alive,
                    'deaths':deaths if post_deaths is None else post_deaths}}


class DodgeAuditTests(unittest.TestCase):
    def test_left_wall_repeated_action_pinned(self):
        result=analyze([row(i=i,threat=True) for i in range(24)])['metrics']
        self.assertEqual(result['outward_command_steps'],24)
        self.assertEqual(result['outward_command_under_threat_steps'],24)
        self.assertEqual(result['max_outward_wall_streak'],24)
        self.assertEqual(result['max_wall_pinned_streak'],24)
        self.assertEqual(result['pinned_commands'],24)
        self.assertEqual(result['death_increments'],0)

    def test_left_wall_escape_counts_as_correct_direction(self):
        result=analyze([row(x=16,y=270,nx=21,action=4)])['metrics']
        self.assertEqual(result['boundary_steps'],1)
        self.assertEqual(result['outward_command_steps'],0)
        self.assertEqual(result['pinned_commands'],0)

    def test_right_wall_outward_and_top_bottom(self):
        events=[row(i=0,x=464,action=4),row(i=1,x=240,y=16,action=1),
                row(i=2,x=240,y=544,action=2)]
        self.assertEqual(analyze(events)['metrics']['outward_command_steps'],3)

    def test_diagonal_tangential_motion_not_called_stalled(self):
        # At left edge, a down+left diagonal is partly clipped but still moves down.
        report=analyze([row(x=16, y=270, nx=16, ny=273, action=7)])['metrics']
        self.assertEqual(report['outward_command_steps'],1)
        self.assertEqual(report['pinned_commands'],0)

    def test_near_edge_but_inward_action_not_wall_pinned(self):
        out=analyze([row(x=20,y=270,nx=25,action=4)])['metrics']
        self.assertEqual(out['outward_command_steps'],0)

    def test_action_persistent_not_outward_in_midfield(self):
        out=analyze([row(i=i,x=240, nx=236,action=3) for i in range(15)])['metrics']
        self.assertEqual(out['max_identical_movement_action_streak'],15)
        self.assertEqual(out['outward_command_steps'],0)

    def test_invulnerable_and_respawn_ignored(self):
        out=analyze([row(i=0,invulnerable=True),row(i=1,alive=False),
                     row(i=2,deaths=0,post_deaths=1)])['metrics']
        self.assertEqual(out['eligible_vulnerable_steps'],0)
        self.assertEqual(out['outward_command_steps'],0)
        self.assertEqual(out['death_increments'],1)

    def test_no_skill_taint_allowed(self):
        with self.assertRaisesRegex(InvalidTrace,'skill None'):
            analyze([row(skill=1)])

    def test_sealed_seed_rejected(self):
        with self.assertRaisesRegex(InvalidTrace,'sealed'):
            analyze([row(seed=43002)])

    def test_missing_post_position_fails(self):
        r=row()
        del r['post']['player_x']
        with self.assertRaisesRegex(InvalidTrace,'post.player_x'):
            analyze([r])

    def test_nonconsecutive_trace_rejected(self):
        with self.assertRaisesRegex(InvalidTrace,'nonconsecutive'):
            analyze([row(i=0),row(i=2)])

    def test_empty_trace_rejected(self):
        with self.assertRaisesRegex(InvalidTrace,'empty'):
            analyze([])

    def test_missing_nearest_projectile_is_unknown_not_threat(self):
        report=analyze([row()])
        self.assertEqual(report['metrics']['linear_threat_steps'],0)
        self.assertIsNone(report['metrics']['fraction_threat_proxy_steps_command_outward'])

    def test_synthetic_rows_are_not_claimed_real(self):
        self.assertIn('not_gameplay_improvement',analyze([row()])['classification'])

    def test_read_jsonl_and_blank_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'trace.jsonl'
            path.write_text(json.dumps(row())+'\n',encoding='utf-8')
            self.assertEqual(len(read_jsonl(path)),1)
            path.write_text(json.dumps(row())+'\n\n',encoding='utf-8')
            with self.assertRaisesRegex(InvalidTrace,'blank'):
                read_jsonl(path)


if __name__=='__main__': unittest.main()
