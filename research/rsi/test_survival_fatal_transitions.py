"""Source-only regression cases; no native gameplay evidence."""
import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from survival_dodge_audit import analyze

def row(i=0, *, alive=True, invulnerable=False, deaths=0, post_deaths=0, threat=True):
    near = {'predicted_closest_separation_px':7.0,'time_to_closest_frames':8.0,'collision_width_px':3.0,'collision_height_px':3.0} if threat else None
    return {'seed':42152,'action_index':i,'movement_action':3,'skill_action':0,
        'pre':{'player_x':16/480,'player_y':270/560,'alive':True,'invulnerable':invulnerable,'deaths':deaths,'nearest_projectile':near},
        'post':{'player_x':16/480,'player_y':270/560,'alive':alive,'deaths':post_deaths}}

class FatalTransitions(unittest.TestCase):
    def test_fatal_action_in_pre_vulnerable_denominator(self):
        result=analyze([row(alive=False,post_deaths=1)])
        m=result['metrics']
        self.assertEqual(m['pre_vulnerable_decisions'],1)
        self.assertEqual(m['pre_vulnerable_death_transitions'],1)
        self.assertEqual(m['pre_vulnerable_projectile_proxy_death_transitions'],1)
        self.assertEqual(m['eligible_vulnerable_steps'],0)
        self.assertEqual(m['pinned_commands'],0)
        self.assertEqual(result['first_vulnerable_death_action_index_by_seed'],{'42152':0})
    def test_invulnerable_death_excluded(self):
        m=analyze([row(alive=False,post_deaths=1,invulnerable=True)])['metrics']
        self.assertEqual(m['pre_vulnerable_decisions'],0)
        self.assertEqual(m['death_increments'],1)
        self.assertIsNone(m['fraction_pre_vulnerable_decisions_with_death_increment'])
    def test_no_bullet_proxy_attribution_without_bullet(self):
        m=analyze([row(alive=False,post_deaths=1,threat=False)])['metrics']
        self.assertEqual(m['pre_vulnerable_death_transitions'],1)
        self.assertEqual(m['pre_vulnerable_projectile_proxy_death_transitions'],0)
    def test_mixed_alive_and_fatal(self):
        m=analyze([row(),row(1,alive=False,post_deaths=1)])['metrics']
        self.assertEqual(m['pre_vulnerable_decisions'],2)
        self.assertEqual(m['eligible_vulnerable_steps'],1)
        self.assertEqual(m['fraction_pre_vulnerable_decisions_with_death_increment'],0.5)
        self.assertEqual(m['pinned_commands'],1)

if __name__=='__main__':unittest.main()
