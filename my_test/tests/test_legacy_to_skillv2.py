"""Source and real E4 checkpoint parity for historical models on ONE live ABI."""
from __future__ import annotations
import os
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
import w2ctrl
import legacy_to_skillv2 as adapter

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

    def test_one_live_format(self):
        policy = self.root/'test_legacy_TSKILL2.bin'
        result = w2ctrl.prepare(self.original,policy)
        self.assertEqual(result['status'],'exported')
        self.assertEqual(result['source_policy_type'],'legacy')
        self.assertEqual(result['live_controller_design'],'TSKILL2_546_18_3_only')
        self.assertEqual(result['adaptation']['skill_policy'],'always_NONE')
        self.assertEqual(result['source_checkpoint_sha256'],self.original_hash)
        self.assertEqual(w2ctrl.parse_policy(policy)['format'],'TSKILL2')
        self.assertEqual(w2ctrl.parse_policy(policy)['checkpoint_sha256'],result['adaptation']['adapted_checkpoint_sha256'])

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

    def test_policy_output_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'policy.bin'
            a=w2ctrl.prepare(self.original,p)
            b=w2ctrl.prepare(self.original,p)
            self.assertEqual(a['policy_sha256'],b['policy_sha256'])
            self.assertEqual(b['status'],'already_identical')
            self.assertEqual(a['adaptation']['adapted_checkpoint_sha256'],b['adaptation']['adapted_checkpoint_sha256'])

    def test_overwrite_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'policy.bin';p.write_bytes(b'protected output')
            with self.assertRaises(FileExistsError):w2ctrl.prepare(self.original,p)
            self.assertEqual(p.read_bytes(),b'protected output')

    def test_wrong_source_hash_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'checkpoint SHA-256 mismatch'):
                w2ctrl.prepare(self.original,Path(d)/'policy.bin',expected_sha='0'*64)

    def test_missing_legacy_weights_refused(self):
        import torch
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.pt'
            torch.save({'policy_type':'legacy','input_dim':528,'actions':18,'width':128},p)
            with self.assertRaisesRegex(ValueError,'model tensors'):
                w2ctrl.prepare(p,Path(d)/'policy.bin')

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

    @unittest.skipUnless(os.getenv('TAISEI_E4_PT'), 'set TAISEI_E4_PT to archived real E4 source')
    def test_real_E4_checkpoint_moves_exactly(self):
        p=Path(os.environ['TAISEI_E4_PT'])
        self.assertEqual(adapter.sha256_file(p),'06c344d1c9415d1277e89c7d7c1a4519f274952f159e1cc5095d7fecf696e6fc')
        with tempfile.TemporaryDirectory() as d:
            bin_path=Path(d)/'e4.bin'
            out=w2ctrl.prepare(p,bin_path)
            self.assertEqual(out['source_policy_type'],'legacy')
            self.assertEqual(out['adaptation']['skill_policy'],'always_NONE')
            self.assertLess(out['adaptation']['movement_max_logit_difference'],1e-5)
            self.assertEqual(out['policy_sha256'],w2ctrl.parse_policy(bin_path)['policy_sha256'])

if __name__=='__main__':unittest.main(verbosity=2)
