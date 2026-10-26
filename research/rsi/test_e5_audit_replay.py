"""E5 source-only fail-closed tests; fixtures never represent game physics."""
import json
from pathlib import Path
import tempfile
import unittest

from e5_audit_replay import validate_replay, REPLAY_SEEDS, episodes, max_run

class E5Tests(unittest.TestCase):
    def test_replay_matches_and_catches_divergence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            summaries=[];trace=[]
            for seed in REPLAY_SEEDS:
                summaries.append({"seed":seed,"digest":seed,"alive_frames":2,"deaths":0,
                                  "bombs_used":0,"boss_damage":0,"cleared":False,
                                  "frames":2,"truncated":False,"terminated":True,
                                  "score":8,"logical_frame":3})
                for idx in range(2):
                    trace.append({"seed":seed,"action_index":idx,"movement_action":0,
                                  "movement_argmax":0,"pre":{"player_x":0.},
                                  "post":{"alive":True}})
            for variant in ("a","b"):
                (root/(variant+".jsonl")).write_text("".join(json.dumps(v)+"\n" for v in summaries))
                (root/(variant+".trace.jsonl")).write_text("".join(json.dumps(v)+"\n" for v in trace))
            args=[root/"a.jsonl",root/"b.jsonl",root/"a.trace.jsonl",root/"b.trace.jsonl"]
            self.assertEqual(validate_replay(*args)["identical_actions_and_native_states"],4)
            wrong=[dict(v) for v in trace]
            wrong[2]["movement_action"]=1
            (root/"b.trace.jsonl").write_text("".join(json.dumps(v)+"\n" for v in wrong))
            with self.assertRaisesRegex(ValueError,"replay mismatch"):
                validate_replay(*args)
            (root/"b.trace.jsonl").write_text((root/"a.trace.jsonl").read_text())
            changed=[dict(v) for v in summaries];changed[0]["digest"]=999
            (root/"b.jsonl").write_text("".join(json.dumps(v)+"\n" for v in changed))
            with self.assertRaisesRegex(ValueError,"terminal replay mismatch"):
                validate_replay(*args)

    def test_sealed_seed_is_forbidden(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"wrong.jsonl"
            p.write_text(json.dumps({"seed":43001})+"\n")
            with self.assertRaisesRegex(ValueError,"sealed"):
                episodes(p,[43001])

    def test_identical_action_run(self):
        self.assertEqual(max_run([0,0,1,1,1,0]),3)
        self.assertEqual(max_run([]),0)

if __name__=="__main__":
    unittest.main()
