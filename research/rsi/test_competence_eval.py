"""Regression tests for competence evaluator: pure records, no substitute simulator."""
import json
import tempfile
import unittest
from pathlib import Path
from competence_eval import compare, paired_ci, read_episode_rows, report_metrics


def row(seed, *, alive=500, deaths=0, clears=False, bombs=0):
    return dict(seed=seed, cleared=clears, won=clears, deaths=deaths,
                alive_frames=alive, frames=alive+20, bombs_used=bombs,
                boss_damage=120.0)


class CompetenceEvaluationTests(unittest.TestCase):
    def test_paired_basic(self):
        baseline = {42001: row(42001), 42002: row(42002, alive=400)}
        candidate = {42001: row(42001, alive=600, clears=True),
                     42002: row(42002, alive=450)}
        report = compare(baseline, candidate)
        self.assertEqual(report["candidate"]["death_free_clears"], 1)
        self.assertEqual(report["paired_differences_candidate_minus_baseline"]["alive_frames"]["mean"], 75)
        self.assertFalse(report["promotion_allowed"])
        self.assertEqual(report["baseline"]["confirmed_item_pickups"], None)

    def test_sealed_final_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x.jsonl"
            path.write_text(json.dumps(row(43042)) + "\n")
            with self.assertRaisesRegex(ValueError, "sealed"):
                read_episode_rows(path)

    def test_duplicate_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x.jsonl"
            data = json.dumps(row(42001))
            path.write_text(data + "\n" + data + "\n")
            with self.assertRaisesRegex(ValueError, "duplicated"):
                read_episode_rows(path)

    def test_missing_metrics_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x.jsonl"
            path.write_text('{"seed":42001}\n')
            with self.assertRaisesRegex(ValueError, "missing"):
                read_episode_rows(path)

    def test_unequal_seed_sets_rejected(self):
        with self.assertRaisesRegex(ValueError, "same exact"):
            compare({42001: row(42001)}, {42002: row(42002)})

    def test_broken_alive_frames_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "x.jsonl"
            value = row(42001, alive=700)
            value["frames"] = 500
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "exceed"):
                read_episode_rows(path)

    def test_bootstrap_reproducible_and_constant(self):
        self.assertEqual(paired_ci([5, 5, 5]), [5, 5])
        self.assertEqual(paired_ci([1, 3, 5]), paired_ci([1, 3, 5]))

    def test_pickup_none_until_native_instrumentation(self):
        rows = {42001: row(42001), 42002: row(42002)}
        self.assertIsNone(report_metrics(rows)["confirmed_item_pickups"])
        for r in rows.values():
            r["confirmed_item_pickups"] = 3
        self.assertEqual(report_metrics(rows)["confirmed_item_pickups"], 6)

    def test_death_free_requires_both_clear_and_zero_deaths(self):
        rows = {42001: row(42001, clears=True, deaths=1),
                42002: row(42002, clears=True, deaths=0),
                42003: row(42003, clears=False, deaths=0)}
        self.assertEqual(report_metrics(rows)["death_free_clears"], 1)


if __name__ == "__main__":
    unittest.main()
