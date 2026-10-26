"""Source-only preflight tests: mocked git, no native or network traffic."""
import pathlib
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from preflight_native_upstream import EXPECTED_COMMIT, EXPECTED_URL, check_upstream


class UpstreamPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.lock = pathlib.Path(self.tmp.name) / "UPSTREAM.lock"
        self.lock.write_text(f"REPO_URL={EXPECTED_URL}\nCOMMIT={EXPECTED_COMMIT}\n")

    def test_reachable_head_does_not_claim_pinned_commit(self):
        run = Mock(return_value=subprocess.CompletedProcess([], 0, "a" * 40 + "\tHEAD\n", ""))
        result = check_upstream(self.lock, runner=run)
        self.assertEqual(result["status"], "reachable")
        self.assertFalse(result["locked_commit_verified"])
        self.assertFalse(result["native_game_executed"])
        args, kwargs = run.call_args
        self.assertEqual(args[0][-3:], ["--exit-code", EXPECTED_URL, "HEAD"])
        self.assertEqual(kwargs["env"]["GIT_TERMINAL_PROMPT"], "0")
        self.assertEqual(kwargs["timeout"], 20)

    def test_unavailable_upstream_fails_closed(self):
        run = Mock(return_value=subprocess.CompletedProcess([], 128, "", "fatal: authentication required"))
        result = check_upstream(self.lock, runner=run)
        self.assertEqual(result["status"], "blocked")
        self.assertIn("exit 128", result["reason"])
        self.assertNotIn("fatal:", str(result))

    def test_lock_mismatch_prevents_git_access(self):
        self.lock.write_text(f"REPO_URL={EXPECTED_URL}\nCOMMIT={'f'*40}\n")
        run = Mock()
        self.assertEqual(check_upstream(self.lock, runner=run)["status"], "blocked")
        run.assert_not_called()

    def test_timeout_and_malformed_head_fail_closed(self):
        run = Mock(side_effect=subprocess.TimeoutExpired(["git"], 20))
        self.assertEqual(check_upstream(self.lock, runner=run)["status"], "blocked")
        run = Mock(return_value=subprocess.CompletedProcess([], 0, "not-a-sha\tHEAD\n", ""))
        self.assertEqual(check_upstream(self.lock, runner=run)["status"], "blocked")

    def test_missing_lock_fails_closed(self):
        self.lock.unlink()
        run = Mock()
        self.assertEqual(check_upstream(self.lock, runner=run)["status"], "blocked")
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
