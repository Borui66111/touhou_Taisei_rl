"""Static, dependency-free safeguards for every bounded authentic-native workflow.

These tests do not execute game physics. They ensure an inaccessible locked
upstream fails *before* artifacts, package installation, and native work.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = {
    "rsi-e2-teacher.yml": "rsi_e2_teacher_v1",
    "rsi-e3-distill.yml": "rsi_e3_distill_v1",
    "rsi-e4-bc.yml": "rsi_e4_class_aware_bc",
    "taisei-rsi-pilot.yml": "rsi_pilot",
    "rsi-e5-native-replay.yml": "rsi_e5_native_replay_v1",
}
GATE = "python research/rsi/preflight_native_upstream.py"
EXPENSIVE = ("actions/download-artifact@v4", "sudo apt-get update",
             "bash ../research/rsi/run_")


class NativeWorkflowPreflightTests(unittest.TestCase):
    def workflow(self, name):
        return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")

    def test_all_native_workflows_fail_fast_before_expensive_work(self):
        for name, output_dir in WORKFLOWS.items():
            with self.subTest(workflow=name):
                source = self.workflow(name)
                self.assertEqual(source.count(GATE), 1)
                gate_index = source.index(GATE)
                self.assertLess(source.index("actions/setup-python@v5"), gate_index)
                self.assertIn("--lock my_test/UPSTREAM.lock", source)
                self.assertIn(
                    f"--out my_test/results/{output_dir}/upstream_preflight.json",
                    source,
                )
                for expensive in EXPENSIVE:
                    if expensive in source:
                        self.assertLess(gate_index, source.index(expensive),
                                        f"{name}: expensive step before access gate")
                # No error masking: a nonzero preflight must fail its Actions step.
                gate_step = source[source.rfind("- name:", 0, gate_index):]
                gate_step = gate_step.split("\n      - name:", 1)[0]
                self.assertNotIn("|| true", gate_step)
                self.assertNotIn("continue-on-error: true", gate_step)

    def test_all_native_jobs_are_serialized_and_time_bounded(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                source = self.workflow(name)
                self.assertIn("group: taisei-native-serial", source)
                self.assertIn("cancel-in-progress: false", source)
                self.assertIn("timeout-minutes: 60", source)
                self.assertIn("workflow_dispatch:", source)
                self.assertIn("permissions:", source)
                self.assertIn("contents: read", source)

    def test_push_triggers_only_target_canonical_branch(self):
        for name in WORKFLOWS:
            with self.subTest(workflow=name):
                source = self.workflow(name)
                if "\n  push:" in source:
                    self.assertIn("branches: [research/rsi-automation-v1]", source)
                self.assertNotIn("branches: [main]", source)

    def test_failures_preserve_preflight_evidence(self):
        for name, output_dir in WORKFLOWS.items():
            with self.subTest(workflow=name):
                source = self.workflow(name)
                self.assertIn("actions/upload-artifact@v4", source)
                self.assertIn("if: always()", source)
                self.assertIn(f"my_test/results/{output_dir}/", source)


if __name__ == "__main__":
    unittest.main()
