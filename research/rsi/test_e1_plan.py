"""E1 source-level experiment gates; not a substitute for real-game tests."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from e1_train_movement import main, reward

RUNNER=(Path(__file__).resolve().parent/"run_e1_movement.sh").read_text()

class TestE1(unittest.TestCase):
    def test_reward_bundle_only(self):
        a=reward("baseline");b=reward("survival_first")
        self.assertEqual(set(a),set(b))
        self.assertNotEqual(a,b)
        a["survival_weight"]=-100
        self.assertEqual(reward("baseline")["survival_weight"],1.0)

    def test_identical_training_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            data={}
            for arm in ("baseline","survival_first"):
                with contextlib.redirect_stdout(io.StringIO()):
                    data[arm]=main(["--arm",arm,"--library","/missing-native.so",
                        "--resource-path","/missing-assets","--outdir",tmp,
                        "--steps","20000","--seed","48020","--stage","1",
                        "--difficulty","1","--preflight-only"])
            a=data["baseline"].copy();b=data["survival_first"].copy()
            for key in ("arm","reward"):
                del a[key];del b[key]
            self.assertEqual(a,b)
            self.assertEqual(a["policy_type"],"legacy_18_movement_no_skills")
            self.assertFalse(list(Path(tmp).iterdir()))

    def test_sealed_training_seeds_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--arm","baseline","--library","none",
                          "--resource-path","none","--outdir",tmp,
                          "--seed","43001","--preflight-only"])

    def test_wrong_difficulty_forbidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--arm","baseline","--library","none",
                          "--resource-path","none","--outdir",tmp,
                          "--difficulty","2","--preflight-only"])

    def test_budget_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["--arm","baseline","--library","none",
                          "--resource-path","none","--outdir",tmp,
                          "--steps","20001","--preflight-only"])

    def test_native_runner_task_and_seed_identity(self):
        self.assertIn("--stage 1 --difficulty 1",RUNNER)
        self.assertIn("--seed 42100",RUNNER)
        self.assertIn("--episodes 16",RUNNER)
        self.assertIn("--seed 48020",RUNNER)
        self.assertIn("--steps 20000",RUNNER)
        self.assertIn("check_native_determinism.py",RUNNER)
        self.assertIn('libtaisei_sim.so',RUNNER)
        self.assertNotIn("--difficulty 2",RUNNER)
        self.assertNotIn("--policy-type skill",RUNNER)
        self.assertNotIn("--skill-action-mask",RUNNER)
        self.assertNotIn("toy",RUNNER.lower().replace("no toy fallback",""))
        self.assertTrue(RUNNER.count("meson devenv -C")>=5)

if __name__=="__main__":
    unittest.main()
