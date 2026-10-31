"""Source and legacy checkpoint gates for the single live ABI."""
from __future__ import annotations
import os
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
import w2ctrl
import legacy_to_skillv2 as adapter

SURVIVAL_FIRST_PT = (HERE.parent / 'res' / 'Taisei_RL_Checkpoints_Local_Demo_2026-10-25' /
                     'survival_first_seed48020.pt')
E4_PT = (HERE.parent / 'res' / 'ckpt' / 'Taisei_E4_2026-10-28' / 'models' /
         'e4_weighted_bc_seed48030.pt')


class LegacyAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.root = Path(cls.work.name)
        cls.original = cls.root/'historical_legacy.pt'
        torch.manual_seed(20261028)
        sizes = {
            'base.0.weight':(128,528),'base.0.bias':(128,),
            'base.2.weight':(128,128),'base.2.bias':(128,),
            'actor.weight':(18,128),'actor.bias':(18,),
            'critic.weight':(1,128),'critic.bias':(1,),
        }
        torch.save({'policy_type':'legacy','input_dim':528,'actions':18,'width':128,
                   'model':{k:torch.randn(v)*0.07 for k,v in sizes.items()}},cls.original)
        cls.original_hash=adapter.sha256_file(cls.original)

    def test_unknown_legacy_refused_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'policy.bin'
            with self.assertRaisesRegex(ValueError,'unverified legacy source'):
                w2ctrl.prepare(self.original,output)
            self.assertFalse(output.exists())
            self.assertFalse((Path(d)/'adapted_checkpoints').exists())

    def test_verified_survival_first_checkpoint_accepted(self):
        if not SURVIVAL_FIRST_PT.is_file():
            self.skipTest('local verified survival_first checkpoint is unavailable')
        self.assertEqual(adapter.sha256_file(SURVIVAL_FIRST_PT),
                         '5526812cb58988849165389db39eec50add78d0dfc41840f5eba1ecc491b93b8')
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'survival_first_TSKILL2.bin'
            result=w2ctrl.prepare(SURVIVAL_FIRST_PT,output,
                expected_sha='5526812cb58988849165389db39eec50add78d0dfc41840f5eba1ecc491b93b8')
            self.assertEqual(result['source_checkpoint_sha256'],
                             '5526812cb58988849165389db39eec50add78d0dfc41840f5eba1ecc491b93b8')
            self.assertEqual(result['adaptation']['adapter_kind'],adapter.ADAPTER_KIND)
            self.assertEqual(result['adaptation']['movement_parity_vectors'],16)
            self.assertEqual(result['adaptation']['skill_policy'],'always_NONE')
            provenance=result['adaptation']['approved_source_provenance']
            self.assertEqual(provenance['code_sha'],'07ad672a595093107277b2e1fa398e65c36b6b64')
            self.assertEqual(provenance['feature_contract']['legacy_prefix_features'],528)
            self.assertEqual(provenance['feature_contract']['canonical_input_features'],546)
            self.assertIn('skill_features()',provenance['feature_contract']['feature_order'])
            self.assertEqual(result['numerical_parity'],'pytorch_vs_numpy_verified')
            self.assertLess(result['max_absolute_logit_difference'],1e-5)
            parsed=w2ctrl.parse_policy(output)
            self.assertEqual(parsed['format'],'TSKILL2')
            self.assertEqual(parsed['checkpoint_sha256'],result['adaptation']['adapted_checkpoint_sha256'])

    def test_missing_manifest_refuses_allowlisted_checkpoint_without_output(self):
        if not SURVIVAL_FIRST_PT.is_file():
            self.skipTest('local verified survival_first checkpoint is unavailable')
        import shutil
        with tempfile.TemporaryDirectory() as d:
            copied=Path(d)/SURVIVAL_FIRST_PT.name
            shutil.copyfile(SURVIVAL_FIRST_PT,copied)
            output=Path(d)/'policy.bin'
            with self.assertRaisesRegex(ValueError,'manifest'):
                w2ctrl.prepare(copied,output)
            self.assertFalse(output.exists())
            self.assertFalse((Path(d)/'adapted_checkpoints').exists())

    def test_mismatched_manifest_refuses_allowlisted_checkpoint_without_output(self):
        if not SURVIVAL_FIRST_PT.is_file():
            self.skipTest('local verified survival_first checkpoint is unavailable')
        import shutil
        with tempfile.TemporaryDirectory() as d:
            copied=Path(d)/SURVIVAL_FIRST_PT.name
            shutil.copyfile(SURVIVAL_FIRST_PT,copied)
            manifest=Path(d)/'survival_first_seed48020_manifest.json'
            output=Path(d)/'policy.bin'
            cases=(
                {'checkpoint_sha256':'0'*64,'code_sha':'07ad672a595093107277b2e1fa398e65c36b6b64'},
                {'checkpoint_sha256':'5526812cb58988849165389db9eec50add78d0dfc41840f5eba1ecc491b93b8','code_sha':'0'*40},
            )
            for metadata in cases:
                manifest.write_text(json.dumps(metadata),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'manifest .* mismatch'):
                    w2ctrl.prepare(copied,output)
                self.assertFalse(output.exists())
                self.assertFalse((Path(d)/'adapted_checkpoints').exists())

    def test_preserve_movement_logits(self):
        with tempfile.TemporaryDirectory() as d:
            adapted=Path(d)/'adapted.pt'
            result=adapter.adapt(self.original,adapted)
            self.assertLess(result['movement_max_logit_difference'],1e-5)
            self.assertEqual(result['movement_parity_vectors'],16)

    def test_skill_none_is_constant(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            adapted=Path(d)/'adapted.pt'
            adapter.adapt(self.original,adapted)
            o=torch.load(adapted,map_location='cpu',weights_only=False)
            s=o['model']
            self.assertEqual(s['skill_actor.bias'].tolist(),[0.0,-1_000_000.0,-1_000_000.0])
            self.assertEqual(torch.count_nonzero(s['skill_actor.weight']).item(),0)
            self.assertEqual(torch.count_nonzero(s['base.0.weight'][:,528:]).item(),0)
            self.assertEqual(tuple(s['base.0.weight'].shape),(128,546))

    def test_original_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            adapter.adapt(self.original,Path(d)/'adapted.pt')
        self.assertEqual(adapter.sha256_file(self.original),self.original_hash)

    def test_adapter_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'adapted.pt'
            first=adapter.adapt(self.original,p)
            second=adapter.adapt(self.original,p)
            self.assertEqual(first['adapted_checkpoint_sha256'],second['adapted_checkpoint_sha256'])
            self.assertEqual(second['status'],'already_identical')

    def test_unverified_legacy_never_creates_output_on_retries(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'policy.bin'
            for _ in range(2):
                with self.assertRaisesRegex(ValueError,'unverified legacy source'):
                    w2ctrl.prepare(self.original,p)
            self.assertFalse(p.exists())
            self.assertFalse((Path(d)/'adapted_checkpoints').exists())

    def test_unverified_source_cannot_modify_existing_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'policy.bin';p.write_bytes(b'protected output')
            with self.assertRaisesRegex(ValueError,'unverified legacy source'):
                w2ctrl.prepare(self.original,p)
            self.assertEqual(p.read_bytes(),b'protected output')

    def test_wrong_source_hash_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'checkpoint SHA-256 mismatch'):
                w2ctrl.prepare(self.original,Path(d)/'policy.bin',expected_sha='0'*64)

    def test_malformed_legacy_source_refused_before_output(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.pt'
            output=Path(d)/'policy.bin'
            torch.save({'policy_type':'legacy','input_dim':528,'actions':18,'width':128},p)
            with self.assertRaisesRegex(ValueError,'unverified legacy source'):
                w2ctrl.prepare(p,output)
            self.assertFalse(output.exists())
            self.assertFalse((Path(d)/'adapted_checkpoints').exists())

    def test_modified_adapter_does_not_get_reused(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            target=Path(d)/'adapter.pt'
            adapter.adapt(self.original,target)
            obj=torch.load(target,map_location='cpu',weights_only=False)
            obj['model']['actor.bias'][0] += 0.01
            torch.save(obj,target)
            with self.assertRaisesRegex(ValueError,'parameters changed'):
                adapter.adapt(self.original,target)

    def test_unknown_checkpoint_family_refused(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'unknown.pt'
            torch.save({'policy_type':'old_skill','input_dim':546},p)
            with self.assertRaisesRegex(ValueError,'unsupported source policy_type'):
                w2ctrl.prepare(p,Path(d)/'policy.bin')

    @unittest.skipUnless(E4_PT.is_file() or os.getenv('TAISEI_E4_PT'),
                        'requires archived real E4 source')
    def test_real_E4_checkpoint_is_refused_before_output(self):
        p=Path(os.environ.get('TAISEI_E4_PT',E4_PT))
        self.assertEqual(adapter.sha256_file(p),
                         '06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc')
        with tempfile.TemporaryDirectory() as d:
            output=Path(d)/'e4.bin'
            with self.assertRaisesRegex(ValueError,'unverified legacy source'):
                w2ctrl.prepare(p,output)
            self.assertFalse(output.exists())
            self.assertFalse((Path(d)/'adapted_checkpoints').exists())

if __name__=='__main__':unittest.main(verbosity=2)
