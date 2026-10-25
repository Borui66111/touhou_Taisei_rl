"""Pure-record regression tests for E2 teacher evaluation. No fake gameplay."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e2_compare_teacher import compare, read, SEEDS


def row(seed, alive=1000, bombs=0, cleared=False):
    return dict(seed=seed, alive_frames=alive, deaths=3,
                bombs_used=bombs, boss_damage=0.0, cleared=cleared)


class TeacherComparisonTests(unittest.TestCase):
    def make_file(self, records):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "e2.jsonl"
        p.write_text("".join(json.dumps(x)+"\n" for x in records), encoding="utf-8")
        return p

    def test_equal_controls_yield_zero_paired_difference(self):
        base = self.make_file([row(s) for s in SEEDS])
        cand = self.make_file([row(s) for s in SEEDS])
        report = compare(base, cand)
        self.assertEqual(report["paired_alive_delta"]["mean"], 0)
        self.assertEqual(report["paired_clear_delta"], 0)
        self.assertFalse(report["promote_policy"])
        self.assertEqual(report["candidate"]["mean_bombs_used"], 0)

    def test_positive_delta_calculated_on_paired_seeds(self):
        base = self.make_file([row(s, alive=100) for s in SEEDS])
        cand = self.make_file([row(s, alive=175) for s in SEEDS])
        result = compare(base, cand)
        self.assertEqual(result["paired_alive_delta"]["mean"], 75)
        self.assertEqual(result["paired_alive_delta"]["eval_seed_bootstrap_95pct_ci"], [75, 75])

    def test_no_skill_resources_enforced(self):
        records = [row(s) for s in SEEDS]
        records[3]["bombs_used"] = 1
        with self.assertRaisesRegex(ValueError, "no Bombs"):
            read(self.make_file(records))

    def test_seed_swap_rejected(self):
        records = [row(s) for s in SEEDS]
        records[0]["seed"] = 42000
        with self.assertRaisesRegex(ValueError, "exact 16"):
            read(self.make_file(records))

    def test_missing_alive_frames_rejected(self):
        records = [row(s) for s in SEEDS]
        records[2]["alive_frames"] = None
        with self.assertRaisesRegex(ValueError, "Exact one-frame"):
            read(self.make_file(records))


if __name__ == "__main__":
    unittest.main()
