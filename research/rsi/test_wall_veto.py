"""Synthetic observation unit tests; no claim of real Taisei gameplay improvement."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from wall_veto import outward_command,edge_veto


def state(x=240,y=280):
    return {'player':[x/480,y/560,0,0,0,3,3,0],
            'boss':[0,0,0,0,0,0],
            'bullets':[], 'lasers':[]}

class WallVetoTests(unittest.TestCase):
    def test_at_left_or_right_outward_detected(self):
        self.assertTrue(outward_command(state(16),3))
        self.assertFalse(outward_command(state(16),4))
        self.assertTrue(outward_command(state(464),4))
        self.assertFalse(outward_command(state(464),3))

    def test_top_bottom_and_diagonals(self):
        self.assertTrue(outward_command(state(240,16),1))
        self.assertTrue(outward_command(state(240,544),2))
        self.assertTrue(outward_command(state(16,100),5))
        self.assertFalse(outward_command(state(16,100),8))

    def test_not_at_wall_retains_action_exact(self):
        obs=state(240,280)
        self.assertEqual(edge_veto(obs,3,fallback=lambda _:1),3)

    def test_outward_replaced_by_provided_test_fallback(self):
        obs=state(16,280)
        replacement=edge_veto(obs,3,fallback=lambda _:4)
        self.assertFalse(outward_command(obs,replacement))

    def test_never_replaces_focus_when_inside(self):
        self.assertEqual(edge_veto(state(),12,fallback=lambda _:0),12)

    def test_fallback_cannot_command_outward(self):
        with self.assertRaisesRegex(ValueError,'outside'):
            edge_veto(state(16),3,fallback=lambda _:3)

    def test_reject_invalid_fallback(self):
        with self.assertRaisesRegex(ValueError,'invalid movement'):
            edge_veto(state(16),3,fallback=lambda _:18)

    def test_reject_invalid_action_and_observation(self):
        for a in (-1,18,True,2.1):
            with self.assertRaises(ValueError):edge_veto(state(),a)
        with self.assertRaises(ValueError):edge_veto({'player':[]},3)

    def test_boundary_inclusive_margin(self):
        self.assertTrue(outward_command(state(24),3,margin=8))
        self.assertFalse(outward_command(state(26),3,margin=8))

if __name__=='__main__':unittest.main()
