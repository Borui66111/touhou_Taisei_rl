"""Source-only checks for E6 genuine archived Taisei episode-disjoint BC audit."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from e6_episode_holdout import (DATASET_SHA, EXPECTED_ACTION_COUNTS,
    EXPECTED_TRAIN_COUNTS, HOLDOUT_N, N_FRAMES, TRAIN_END,
    diagnostic, file_sha256, read_native_labels_without_numpy, validate_split)

ROOT=Path(__file__).resolve().parents[0]
DATA=ROOT/'evidence/escrow_e3_e4_20261026_v1/e3/teacher_demonstrations_12000.npz'
REPORT=ROOT/'evidence/e6_episode_disjoint_bc_20261026.json'

class E6EpisodeHoldoutTests(unittest.TestCase):
    def test_real_native_archive_and_complete_episode_boundary(self):
        actions=read_native_labels_without_numpy(DATA)
        self.assertEqual(file_sha256(DATA),DATASET_SHA)
        train,heldout=validate_split(len(actions))
        self.assertEqual((len(train),len(heldout)),(TRAIN_END,HOLDOUT_N))
        self.assertEqual(tuple(actions.count(k) for k in range(18)),EXPECTED_ACTION_COUNTS)
        self.assertEqual(tuple(actions[:TRAIN_END].count(k) for k in range(18)),EXPECTED_TRAIN_COUNTS)
        self.assertFalse(set(train).intersection(heldout))

    def test_tampered_dataset_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/'tampered.npz'
            shutil.copyfile(DATA,target)
            with target.open('ab') as handle: handle.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'immutable E3 native'):
                read_native_labels_without_numpy(target)

    def test_boundary_shift_or_dataset_size_rejected(self):
        for n,b in [(12000,7856),(12000,7858),(11999,7857),(12001,7857)]:
            with self.subTest(n=n,b=b):
                with self.assertRaisesRegex(ValueError,'episode boundary'):
                    validate_split(n,b)

    def test_metrics_keep_stay_and_nonstay_separate(self):
        d=diagnostic([0,0,1,1,2,2],[0,0,0,1,0,2],
                     [.1,.2,.3,.4,.5,.6],.3)
        self.assertAlmostEqual(d['frame_accuracy'],4/6)
        self.assertAlmostEqual(d['nonstay_accuracy'],2/4)
        self.assertAlmostEqual(d['macro_recall_supported_labels'],(1+.5+.5)/3)
        self.assertAlmostEqual(d['predicted_stay_fraction'],4/6)
        self.assertEqual(d['high_bullet_density_support'],4)
        self.assertEqual(d['high_density_nonstay_support'],4)
        self.assertAlmostEqual(d['high_density_nonstay_accuracy'],.5)

    def test_invalid_actions_and_misalignment_rejected(self):
        cases=[([],[],[],.2),([0],[0],[],.2),([18],[0],[.1],.2),
               ([0],[-1],[.1],.2),([0],[0],[.1],-1)]
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError): diagnostic(*args)

    def test_offline_report_records_no_native_or_champion_claim(self):
        r=json.loads(REPORT.read_text())
        self.assertEqual(r['dataset_sha256'],DATASET_SHA)
        self.assertEqual(r['training_episode']['frames'],TRAIN_END)
        self.assertEqual(r['heldout_episode']['frames'],HOLDOUT_N)
        self.assertFalse(r['heldout_episode']['completed'])
        self.assertEqual(len(r['results']),6)
        self.assertEqual(len(r['paired_deltas_weighted_minus_plain']),3)
        for key in ('native_game_executed','gameplay_improvement_verified','promotion_allowed'):
            self.assertIs(r[key],False)
        self.assertEqual(r['native_training_decisions'],0)
        self.assertEqual(r['native_evaluation_decisions'],0)
        for result in r['results']:
            self.assertEqual(len(result['checkpoint_sha256']),64)
        for delta in r['paired_deltas_weighted_minus_plain']:
            base=next(x for x in r['results'] if x['seed']==delta['seed'] and x['variant']=='plain_ce')
            weighted=next(x for x in r['results'] if x['seed']==delta['seed'] and x['variant']=='half_class_weighted_ce')
            self.assertAlmostEqual(delta['delta_nonstay_accuracy'],weighted['nonstay_accuracy']-base['nonstay_accuracy'])

if __name__=='__main__':unittest.main()
