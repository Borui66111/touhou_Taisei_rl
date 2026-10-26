"""Tests for research event provenance and append-only integrity."""
from __future__ import annotations
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ledger import append_event, read_ledger, summary, validate


def event(key="T-TEST-01", parent=None, kind="diagnosis", status="proposed",
          when="2026-10-25T01:00:00Z"):
    return {
        "event_id": key, "timestamp_utc": when,
        "project": "touhou_taisei_rl", "level": "L3", "kind": kind,
        "status": status, "parent_ids": parent or [],
        "hypothesis": "Proposed theory", "intervention": "Small independent change",
        "expectation": "Observable prediction", "observation": "Not yet tested",
        "interpretation": "Unknown", "evidence": [],
        "limits": ["Requires native evidence"], "decision": "Do not promote",
        "next_action": "Run a diagnostic", "review_required": False,
        "metrics": {}, "cost": {"compute_minutes": None, "engineering_minutes": None},
    }


class LedgerTest(unittest.TestCase):
    def test_initial_real_history_validates(self):
        rows = read_ledger(HERE / "events.jsonl")
        self.assertGreaterEqual(len(rows), 9)
        self.assertEqual(len({x["event_id"] for x in rows}), len(rows))

    def test_linked_append_preserves_old_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "events.jsonl"
            first = event()
            append_event(p, first)
            old = p.read_bytes()
            append_event(p, event("T-TEST-02", parent=["T-TEST-01"],
                                  when="2026-10-25T01:10:00Z"))
            self.assertTrue(p.read_bytes().startswith(old))
            self.assertEqual(len(read_ledger(p)), 2)

    def test_two_immutable_legacy_events_have_unknown_metrics(self):
        rows = read_ledger(HERE / "events.jsonl")
        legacy = {r["event_id"]: r for r in rows if r["event_id"] in {"T-20261026-33", "T-20261026-34"}}
        self.assertEqual(set(legacy), {"T-20261026-33", "T-20261026-34"})
        self.assertTrue(all("metrics" not in row for row in legacy.values()))
        report = summary(rows)
        self.assertEqual(report["total_events"], len(rows))
        self.assertIsNone(report["validated_competence_improvement"])

    def test_missing_metrics_still_rejected_for_new_events(self):
        e = event()
        e.pop("metrics")
        with self.assertRaisesRegex(ValueError, "missing fields.*metrics"):
            validate([e])

    def test_legacy_exception_requires_exact_identity(self):
        e = event(key="T-20261026-33")
        e.pop("metrics")
        with self.assertRaisesRegex(ValueError, "missing fields.*metrics"):
            validate([e])

    def test_duplicate_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate([event(), event()])

    def test_forward_parent_rejected(self):
        with self.assertRaisesRegex(ValueError, "forward"):
            validate([event(parent=["T-FUTURE"])])

    def test_time_cannot_regress(self):
        with self.assertRaisesRegex(ValueError, "regress"):
            validate([event(), event("T-TEST-02", when="2026-10-24T01:00:00Z")])

    def test_sealed_training_or_eval_seed_rejected(self):
        e = event()
        e["metrics"] = {"eval_seeds": [42001, 43099]}
        with self.assertRaisesRegex(ValueError, "sealed"):
            validate([e])

    def test_claims_need_evidence(self):
        e = event(kind="measurement", status="succeeded")
        with self.assertRaisesRegex(ValueError, "without evidence"):
            validate([e])

    def test_summary_never_conflates_ci_success_with_game_competence(self):
        a = event(kind="execution", status="succeeded")
        r = summary([a])
        self.assertEqual(r["native_reexecuted_measurements"], 0)
        self.assertIsNone(r["validated_competence_improvement"])
        self.assertTrue(r["continuous_agenda"])
        self.assertFalse(r["auto_promotion"])

    def test_invalid_metrics_no_fabricated_values(self):
        a = event()
        report = summary([a])
        self.assertNotIn("death_free_clear_rate", report)
        self.assertIsNone(report["validated_competence_improvement"])

    def test_correction_requires_new_event(self):
        e0 = event()
        e1 = event("T-TEST-02", parent=["T-TEST-01"],
                   when="2026-10-25T01:10:00Z")
        e1.update(kind="decision", observation="Correct old interpretation by referencing original event")
        self.assertEqual(validate([e0, e1])["events"], 2)


if __name__ == "__main__":
    unittest.main()
