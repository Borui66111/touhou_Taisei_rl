"""Raw launcher must reject noncanonical policies before spawning a game."""
from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import run_live_ppo_windows as launcher


class RawLauncherPolicyValidationTests(unittest.TestCase):
    def test_malformed_and_tppov1_policies_are_rejected_before_subprocess(self):
        payloads = {
            "malformed_tskill2": b"TSKILL2\0bad header",
            "legacy_tppov1": b"TPPOv1\0" + bytes(64),
        }
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory) / "policy.bin"
            for label, payload in payloads.items():
                with self.subTest(policy=label):
                    policy.write_bytes(payload)
                    stderr = io.StringIO()
                    with patch.object(launcher.subprocess, "run") as game_process, \
                            redirect_stderr(stderr):
                        status = launcher.main(["--rl-policy", str(policy)])
                    self.assertNotEqual(status, 0)
                    self.assertIn("canonical TSKILL2", stderr.getvalue())
                    game_process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
