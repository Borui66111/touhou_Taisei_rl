"""Source-level tests for E2 trace diagnostic. No fake native gameplay."""
import json
import tempfile
from pathlib import Path
import unittest
from e2_trace_diagnostics import read_trace, summarize_trace


def action(seed, i, movement=8, x=.5, y=.8, deaths=0, nearest=None):
    return {"seed":seed,"action_index":i,"movement_action":movement,
            "skill_action":None,
            "pre":{"logical_frame":i,"player_x":x,"player_y":y,
                   "deaths":deaths,"nearest_projectile":nearest},
            "post":{"logical_frame":i+1,"deaths":deaths}}


class E2Tests(unittest.TestCase):
    def test_episode_integrity_and_action_proxies(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"trace.jsonl"
            rows=[]
            for seed in range(42100,42104):
                rows += [action(seed,0,8,x=.99,y=.96,nearest={"predicted_closest_separation_px":8}),
                         action(seed,1,8,x=.99,y=.96,nearest={"predicted_closest_separation_px":8}),
                         action(seed,2,3,x=.01,y=.9,nearest=None)]
            path.write_text("\n".join(map(json.dumps,rows))+"\n")
            report=summarize_trace(path)
            self.assertEqual(report["seed_reports"]["42100"]["recorded_actions"],3)
            self.assertEqual(report["seed_reports"]["42100"]["longest_identical_action_run"],2)
            self.assertEqual(report["seed_reports"]["42100"]["first_recorded_death_increment_frame"],None)
            self.assertGreater(report["aggregate_mean"]["border_pushing_proxy_fraction"],0)

    def test_sealed_final_seed_refusal(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"bad.jsonl"
            path.write_text(json.dumps(action(43000,0))+"\n")
            with self.assertRaisesRegex(ValueError,"sealed seed"):
                read_trace(path)

    def test_missing_episode_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"incomplete.jsonl"
            path.write_text(json.dumps(action(42100,0))+"\n")
            with self.assertRaisesRegex(ValueError,"four complete"):
                read_trace(path)


if __name__=="__main__":
    unittest.main()
