"""E3 source and experiment-budget tests; not evidence of real-game learning."""
import contextlib
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from e3_distill_native import checked_plan, main, N_FEATURES, ppo_subprocess_command


class E3PlanTests(unittest.TestCase):
    def test_exact_pilot_budget(self):
        p=checked_plan(12000,8000,1,1,48030,48020)
        self.assertEqual(p["native_interactions_max"],20000)
        self.assertEqual(p["model_input_dim"],528)

    def test_over_budget_rejected(self):
        with self.assertRaisesRegex(ValueError,"12k demo"):
            checked_plan(15000,8000,1,1,48030,48020)

    def test_wrong_difficulty_rejected(self):
        with self.assertRaisesRegex(ValueError,"Stage1 Easy"):
            checked_plan(12000,8000,2,1,48030,48020)

    def test_wrong_training_seeds_rejected(self):
        with self.assertRaisesRegex(ValueError,"fixed"):
            checked_plan(12000,8000,1,1,43042,48020)

    def test_ppo_uses_separate_native_process_with_pinned_budget(self):
        cmd=ppo_subprocess_command("/native/libtaisei_sim.so","/native/resources",
                                   "/out/bc.pt","/out/ppo.pt","/out/ppo.jsonl",8000,48020)
        self.assertTrue(cmd[1].endswith("my_test/scripts/train_torch_ppo.py"))
        for key,value in (("--steps","8000"),("--seed","48020"),
                          ("--difficulty","1"),("--stage","1"),
                          ("--policy-type","legacy"),("--init","/out/bc.pt"),
                          ("--out","/out/ppo.pt")):
            self.assertEqual(cmd[cmd.index(key)+1],value)
        self.assertNotIn("--skill-action-mask",cmd)

    def test_preflight_does_not_claim_training_or_require_native(self):
        with tempfile.TemporaryDirectory() as d:
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):
                main(["--library","/definitely/not/taisei.so",
                      "--resource-path","/definitely/not/taisei_resources",
                      "--outdir",d,"--preflight-only"])
            obj=json.loads(stream.getvalue())
            self.assertEqual(obj["status"],"preflight_only")
            self.assertEqual(obj["native_interactions_max"],20000)
            self.assertFalse(list(Path(d).glob("*.pt")))


if __name__=="__main__":
    unittest.main()
