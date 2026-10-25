"""Sanity checks for first native-game reward-only RSI challenger."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from train_reward_ablation import CONTROL, CHALLENGER, main, reward_config


class RewardAblationTests(unittest.TestCase):
    def test_reward_arm_keys_match_and_are_independently_owned(self):
        self.assertEqual(set(CONTROL), set(CHALLENGER))
        a = reward_config("control")
        b = reward_config("challenger")
        self.assertNotEqual(a, b)
        a["survival_weight"] = -999
        self.assertEqual(CONTROL["survival_weight"], 1.0)

    def test_preflight_does_not_need_or_fake_native_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()) as captured:
                main(["--variant", "challenger", "--library", "/definitely/absent/native.so",
                      "--output-dir", tmp, "--preflight-only", "--seed", "48020",
                      "--steps", "1024"])
            manifest = json.loads(captured.getvalue())
            self.assertEqual(manifest["status"], "untrained_preflight")
            self.assertEqual(manifest["frames_per_action"], 1)
            self.assertEqual(manifest["reward"]["survival_weight"], 0.002)
            self.assertFalse(list(Path(tmp).glob("*.pt")))

    def test_sealed_seed_not_usable_as_training_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--variant", "control", "--library", "/invalid",
                          "--output-dir", tmp, "--preflight-only",
                          "--seed", "43042"])

    def test_change_is_reward_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = {}
            for arm in ("control", "challenger"):
                with contextlib.redirect_stdout(io.StringIO()) as capture:
                    main(["--variant", arm, "--library", "/not-present",
                          "--output-dir", tmp, "--preflight-only"])
                results[arm] = json.loads(capture.getvalue())
            c = results["control"].copy()
            d = results["challenger"].copy()
            del c["variant"]; del c["reward"]
            del d["variant"]; del d["reward"]
            self.assertEqual(c, d)


if __name__ == "__main__":
    unittest.main()
