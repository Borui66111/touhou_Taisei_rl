"""E5 fail-closed replay tests. Synthetic JSON fixtures are NOT game simulations."""
import json
from pathlib import Path
import tempfile
import unittest

from e5_audit_replay import validate_replay, REPLAY_SEEDS, episodes, max_run


class E5Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.summaries=[]
        self.trace=[]
        for episode_index,seed in enumerate(REPLAY_SEEDS):
            self.summaries.append({
                "seed":seed,"digest":str(seed),"alive_frames":2,"deaths":0,
                "bombs_used":0,"boss_damage":0,"cleared":False,"frames":2,
                "truncated":False,"terminated":True,"score":8,"logical_frame":2,
            })
            for idx in range(2):
                self.trace.append({
                    "seed":seed,"episode_index":episode_index,"action_index":idx,
                    "movement_action":0,"movement_argmax":0,"skill_action":None,
                    "pre":{"logical_frame":idx,"alive":True,"deaths":0},
                    "post":{"logical_frame":idx+1,"alive":True,"deaths":0,
                            "boss_damage":0.0},
                })
        self.write()

    def write(self, summary_b=None, trace_b=None):
        for variant,rows,traces in (
            ("a",self.summaries,self.trace),
            ("b",self.summaries if summary_b is None else summary_b,
             self.trace if trace_b is None else trace_b),
        ):
            (self.root/(variant+".jsonl")).write_text(
                "".join(json.dumps(x)+"\n" for x in rows))
            (self.root/(variant+".trace.jsonl")).write_text(
                "".join(json.dumps(x)+"\n" for x in traces))

    def check(self):
        return validate_replay(
            self.root/"a.jsonl",self.root/"b.jsonl",
            self.root/"a.trace.jsonl",self.root/"b.trace.jsonl")

    def test_complete_identical_native_format_passes(self):
        self.assertEqual(self.check()["identical_actions_and_native_states"],4)

    def test_rejects_different_action_or_terminal_digest(self):
        trace=json.loads(json.dumps(self.trace))
        trace[2]["movement_action"]=1
        self.write(trace_b=trace)
        with self.assertRaisesRegex(ValueError,"replay mismatch"):
            self.check()
        summary=json.loads(json.dumps(self.summaries))
        summary[0]["digest"]="changed"
        self.write(summary_b=summary)
        with self.assertRaisesRegex(ValueError,"terminal replay mismatch"):
            self.check()

    def test_rejects_two_identical_but_incomplete_traces(self):
        self.trace=[r for r in self.trace if r["action_index"]==0]
        self.write()
        with self.assertRaisesRegex(ValueError,"coverage|seed order"):
            self.check()

    def test_rejects_missing_native_state_or_discontinuous_frames(self):
        self.trace[0]["post"].pop("logical_frame")
        self.write()
        with self.assertRaisesRegex(ValueError,"native state fields"):
            self.check()
        self.trace[0]["post"]["logical_frame"]=3
        self.write()
        with self.assertRaisesRegex(ValueError,"logical progression"):
            self.check()

    def test_rejects_non_argmax_and_skill_actions(self):
        self.trace[0]["movement_argmax"]=1
        self.write()
        with self.assertRaisesRegex(ValueError,"argmax"):
            self.check()
        self.trace[0]["movement_argmax"]=0
        self.trace[0]["skill_action"]=1
        self.write()
        with self.assertRaisesRegex(ValueError,"skill action"):
            self.check()

    def test_rejects_mismatched_alive_count_and_episode_index(self):
        self.summaries[0]["alive_frames"]=1
        self.write()
        with self.assertRaisesRegex(ValueError,"alive-frame"):
            self.check()
        self.summaries[0]["alive_frames"]=2
        self.trace[0]["episode_index"]=1
        self.write()
        with self.assertRaisesRegex(ValueError,"episode index"):
            self.check()

    def test_sealed_seed_is_forbidden(self):
        p=self.root/"wrong.jsonl"
        p.write_text(json.dumps({"seed":43001})+"\n")
        with self.assertRaisesRegex(ValueError,"sealed"):
            episodes(p,[43001])

    def test_identical_action_run(self):
        self.assertEqual(max_run([0,0,1,1,1,0]),3)
        self.assertEqual(max_run([]),0)


if __name__=="__main__":
    unittest.main()
