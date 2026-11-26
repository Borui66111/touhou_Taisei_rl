import json
import pathlib
import tempfile
import unittest

import reward_audit as ra

REPO = pathlib.Path(__file__).resolve().parents[2]
W = dict(survival_weight=1.0, damage_weight=0.02, graze_weight=0.001,
         death_penalty=-10.0, loss_penalty=-10.0, win_bonus=20.0, no_bomb_bonus=0.0)


class RewardAuditTest(unittest.TestCase):
    def test_reads_defaults_and_flags_unread_field(self):
        w, unread = ra.env_reward_config(REPO / "my_test/native_rl/env.py")
        self.assertEqual(w["death_penalty"], -10.0)
        self.assertIn("no_bomb_bonus", unread)
        self.assertNotIn("death_penalty", unread)

    def test_detects_field_once_step_reads_it(self):
        src = ("class EnvConfig:\n" + "".join(f"    {k}: float = {v}\n" for k, v in W.items())
               + "class TaiseiEnv:\n    def step(self):\n        return self.cfg.no_bomb_bonus + "
               + " + ".join(f"self.cfg.{k}" for k in W if k != "no_bomb_bonus") + "\n")
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "env.py"
            p.write_text(src)
            self.assertEqual(ra.env_reward_config(p)[1], [])

    def test_decompose_exact_and_terminal(self):
        ep = dict(alive_frames=100, boss_damage=50.0, graze=3, deaths=3, frames=300,
                  terminated=True, won=False, reward=100 + 1.0 + 0.003 - 30 - 10)
        r = ra.decompose(ep, W)
        self.assertAlmostEqual(r["residual"], 0.0, places=9)
        self.assertEqual(r["non_alive_frames"], 200)
        ep_trunc = dict(ep, terminated=False, reward=ep["reward"] + 10)
        self.assertAlmostEqual(ra.decompose(ep_trunc, W)["residual"], 0.0, places=9)

    def test_discounted_costs(self):
        c = ra.discounted_costs(W, 0.995, 0.0)
        self.assertAlmostEqual(c["effective_horizon_frames"], 200.0, places=6)
        self.assertAlmostEqual(c["nonfinal_death_cost"], 10.0, places=9)
        self.assertAlmostEqual(c["final_death_cost"], 220.0, places=6)

    def test_escrow_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "e4").mkdir()
            (root / "e4" / "x.jsonl").write_text("{}\n")
            man = {"artifacts": {"e4": {"files": {"x.jsonl": {"sha256": "0" * 64}}}}}
            with self.assertRaises(ValueError):
                ra.verify_escrow(root, "e4/x.jsonl", man)

    def test_real_escrow_reconstructs(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "r.json"
            self.assertEqual(ra.main(["--repo", str(REPO), "--out", str(out)]), 0)
            rep = json.loads(out.read_text())
        self.assertEqual(rep["episodes_reconstructed"], 80)
        self.assertLess(rep["max_abs_residual_all"], 1e-2)
        self.assertFalse(rep["native_game_executed"])


if __name__ == "__main__":
    unittest.main()
