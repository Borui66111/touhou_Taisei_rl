"""No-network, no-native tests for exact-identity recovered checkout verification."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from verify_pinned_local_checkout import EXPECTED_COMMIT, EXPECTED_URL, verify


class LocalCheckoutVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.lock = self.root / "UPSTREAM.lock"
        self.lock.write_text(f"REPO_URL={EXPECTED_URL}\nCOMMIT={EXPECTED_COMMIT}\n")
        self.checkout = self.root / "source"
        self.checkout.mkdir()
        (self.checkout / ".gitmodules").write_text('[submodule "s"]\n path = s\n url = x\n')
        self.good = [str(self.checkout.resolve()), EXPECTED_URL, EXPECTED_COMMIT,
                     "commit", "", " " + "a"*40 + " s (heads/main)"]

    def run_mock(self, responses):
        calls = []
        def runner(cmd, **kwargs):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, responses[len(calls)-1], "")
        return runner, calls

    def test_exact_offline_checkout_accepted_without_claiming_gameplay(self):
        runner, calls = self.run_mock(self.good)
        r = verify(self.lock, self.checkout, runner=runner)
        self.assertEqual(r["status"], "verified")
        self.assertTrue(r["locked_commit_verified"])
        self.assertTrue(r["recursive_submodules_verified"])
        self.assertFalse(r["native_game_executed"])
        self.assertEqual(r["submodule_count"], 1)
        self.assertEqual(len(calls), 6)
        self.assertTrue(all(x[0] == "git" and x[1] == "-C" for x in calls))
        self.assertFalse(any(x in cmd for cmd in calls for x in ("fetch", "clone", "checkout", "update")))

    def test_lock_mismatch_never_invokes_git(self):
        self.lock.write_text("REPO_URL=https://example.invalid\nCOMMIT=123\n")
        runner = Mock()
        self.assertEqual(verify(self.lock, self.checkout, runner=runner)["status"], "blocked")
        runner.assert_not_called()

    def test_missing_checkout_never_invokes_git(self):
        runner = Mock()
        self.assertEqual(verify(self.lock, self.root / "absent", runner=runner)["status"], "blocked")
        runner.assert_not_called()

    def test_wrong_origin_commit_or_dirty_tree_fail(self):
        for index, value in ((1, "https://example.invalid"), (2, "f"*40), (4, " M src/player.c")):
            with self.subTest(index=index):
                answers = list(self.good)
                answers[index] = value
                runner, _ = self.run_mock(answers)
                self.assertEqual(verify(self.lock, self.checkout, runner=runner)["status"], "blocked")

    def test_missing_or_mismatched_submodules_fail(self):
        for line in ("", "-"+"a"*40+" s", "+"+"a"*40+" s", "U"+"a"*40+" s", " invalid"):
            with self.subTest(line=line):
                answers = list(self.good)
                answers[-1] = line
                runner, _ = self.run_mock(answers)
                self.assertEqual(verify(self.lock, self.checkout, runner=runner)["status"], "blocked")

    def test_missing_submodule_manifest_fails(self):
        (self.checkout / ".gitmodules").unlink()
        runner, _ = self.run_mock(self.good)
        self.assertEqual(verify(self.lock, self.checkout, runner=runner)["status"], "blocked")

    def test_non_git_checkout_fails_with_real_git(self):
        self.assertEqual(verify(self.lock, self.checkout)["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
