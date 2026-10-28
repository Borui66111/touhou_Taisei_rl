"""Portable source/fixture tests for w2ctrl. These do not run authentic Taisei."""
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'scripts' if (HERE.parent / 'scripts' / 'w2ctrl.py').is_file() else HERE.parent / 'my_test' / 'scripts'
sys.path.insert(0, str(SRC))
import w2ctrl as w
import export_live_skill_ppo_v2 as exporter


def log_lines(n, sha, seed=42132):
    rows=['RL skill-v2 policy loaded checkpoint_sha256='+sha]
    for i in range(n):
        rows.extend([
            f'RL run seed={seed+i} stage=1 difficulty=1 policy_type=skill_v2 loaded=1 checkpoint_sha256={sha}',
            f'RL decision frame=1 movement=5 skill=0 accepted=1 action_count=1 checkpoint_sha256={sha}',
            f'RL decision frame=2 movement=6 skill=1 accepted=1 action_count=2 checkpoint_sha256={sha}',
            f'RL policy outcome=defeat frames=300 actions=2 checkpoint_sha256={sha}',
            f'Saving stage{i}.tsr'])
    return rows


class PortableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.tmp=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.root=Path(cls.tmp.name)
        torch.manual_seed(20261028)
        state={name:torch.randn(shape,dtype=torch.float32)*0.07 for name,shape in zip(w.TENSOR_NAMES,w.SHAPES)}
        cls.cp=cls.root/'fixture.pt'
        torch.save({'policy_type':'skill','input_dim':546,'width':128,'actions':18,'skills':3,
                    'skill_action_mask_version':w.MASK_VERSION,'model':state},cls.cp)
        cls.sha=w.sha256_file(cls.cp)
        cls.policy=cls.root/'fixture_TSKILL2.bin'
        exporter.export(cls.cp,cls.policy)

    def test_main_exporter_exact_git_blob(self):
        b=(SRC/'export_live_skill_ppo_v2.py').read_bytes()
        sha=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
        self.assertEqual(sha,'c54158611b5e0a145620733fc24e99612b3e5329')

    def test_exported_binary_schema(self):
        p=w.parse_policy(self.policy,self.sha)
        self.assertEqual(p['format'],'TSKILL2')
        self.assertEqual(p['payload_bytes'],w.EXPECTED_PAYLOAD_BYTES)

    def test_two_heads_pytorch_numpy(self):
        result=w.verify_forward_logits(self.cp,self.policy,trials=16)
        self.assertLess(result['max_absolute_logit_difference'],1e-5)
        self.assertEqual(result['c_loader_parity'],'not_tested')

    def test_reexport_exactly_identical(self):
        out=self.root/'same.bin'
        exporter.export(self.cp,out)
        self.assertEqual(out.read_bytes(),self.policy.read_bytes())

    def test_generated_checkpoint_prepares(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst=Path(tmp)/'export.bin'
            self.assertEqual(w.prepare(self.cp,dst,self.sha)['status'],'exported')
            self.assertEqual(w.parse_policy(dst)['checkpoint_sha256'],self.sha)

    def test_reject_movement_only(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            cp=Path(tmp)/'legacy.pt'; torch.save({'policy_type':'legacy','input_dim':528,'actions':18,'width':128},cp)
            with self.assertRaisesRegex(ValueError,'architecture mismatch'):
                w.prepare(cp,Path(tmp)/'bad.bin')

    def test_duplicate_protection_and_idempotence(self):
        with tempfile.TemporaryDirectory() as tmp:
            dst=Path(tmp)/'p.bin';dst.write_bytes(self.policy.read_bytes())
            self.assertEqual(w.prepare(self.cp,dst)['status'],'already_identical')
            dst.write_bytes(b'prior experiment')
            with self.assertRaises(FileExistsError):w.prepare(self.cp,dst)
            self.assertEqual(dst.read_bytes(),b'prior experiment')

    def test_wrong_checkpoint_hash_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'checkpoint SHA-256 mismatch'):
                w.prepare(self.cp,Path(tmp)/'a.bin','0'*64)

    def test_truncation_and_trailing_bytes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.bin';b=self.policy.read_bytes()
            for raw in [b[:40],b[:-1],b+b'junk']:
                p.write_bytes(raw)
                with self.assertRaises(ValueError):w.parse_policy(p)

    def test_metadata_damage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.bin';b=bytearray(self.policy.read_bytes());b[49]^=1;p.write_bytes(b)
            with self.assertRaisesRegex(ValueError,'metadata hash'):
                w.parse_policy(p)

    def test_mask_version_rejected(self):
        b=self.policy.read_bytes(); n=struct.unpack('<I',b[8:12])[0]
        m=json.loads(b[48:48+n]);m['skill_action_mask_version']='wrong'
        canonical=json.dumps(m,sort_keys=True,separators=(',',':')).encode()
        payload=b[48+n:]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.bin'
            p.write_bytes(b[:8]+struct.pack('<II',len(canonical),len(payload))+hashlib.sha256(canonical).digest()+canonical+payload)
            with self.assertRaisesRegex(ValueError,'mask_version'):
                w.parse_policy(p)

    def test_single_run_logs(self):
        a=w.audit_lines(log_lines(1,self.sha),self.sha)
        self.assertEqual(a['status'],'matched_logged_control')
        self.assertEqual(a['decision_count'],2)
        self.assertEqual(a['replay_playability'],'not_verified')

    def test_multiple_run_logs(self):
        a=w.audit_lines(log_lines(8,self.sha),self.sha)
        self.assertEqual(a['status'],'matched_logged_control')
        self.assertEqual(a['decision_count'],16)
        self.assertEqual(a['run_count'],8)

    def test_prefixed_run_logs(self):
        a=w.audit_lines(['100 INFO file source: '+x+' [logger]' for x in log_lines(1,self.sha)],self.sha)
        self.assertEqual(a['status'],'matched_logged_control')

    def test_missing_policy_loaded_rejected(self):
        self.assertEqual(w.audit_lines(log_lines(1,self.sha)[1:],self.sha)['status'],'not_verified')

    def test_zero_decisions_rejected(self):
        rows=[x for x in log_lines(1,self.sha) if not x.startswith('RL decision')]
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_wrong_policy_decision_rejected(self):
        rows=log_lines(1,self.sha); rows[2]=rows[2].replace(self.sha,'0'*64)
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_sealed_seeds_rejected(self):
        a=w.audit_lines(log_lines(1,self.sha,seed=43021),self.sha)
        self.assertEqual(a['status'],'not_verified')
        self.assertTrue(any('sealed' in v for v in a['violations']))

    def test_repeated_action_counter_rejected(self):
        rows=log_lines(1,self.sha);rows[3]=rows[3].replace('action_count=2','action_count=1')
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_missing_outcome_rejected(self):
        rows=[r for r in log_lines(1,self.sha) if not r.startswith('RL policy outcome')]
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_unchanged_log_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';m={'sha256':{'policy':w.sha256_file(self.policy)},
                'policy_metadata':{'checkpoint_sha256':self.sha},
                'log_capture':{'status':'unknown','reason':'log_missing_before_launch'},'exit_status':0}
            p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'fresh game log'):
                w.audit_manifest(p,self.policy)

    def test_valid_synthetic_manifest_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';m={'sha256':{'policy':w.sha256_file(self.policy),'game':'1'*64},
                'policy_metadata':{'checkpoint_sha256':self.sha},
                'log_capture':{'status':'captured','relevant_lines':log_lines(2,self.sha)},
                'status':'complete','exit_status':0}
            p.write_text(json.dumps(m))
            out=w.audit_manifest(p,self.policy)
            self.assertEqual(out['status'],'matched_logged_control')
            self.assertEqual(out['visual_inspection'],'not_performed')

    def test_wrong_manifest_binary_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'manifest.json';p.write_text(json.dumps({'sha256':{'policy':'0'*64}}))
            with self.assertRaisesRegex(ValueError,'policy file SHA-256'):
                w.audit_manifest(p,self.policy)

    def test_policy_load_must_precede_run(self):
        rows=log_lines(1,self.sha)
        rows=[*rows[1:4],rows[0],*rows[4:]]
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_run_requires_valid_seed(self):
        rows=log_lines(1,self.sha)
        rows[1]=rows[1].replace('seed=42132','seed=unknown')
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_all_decisions_rejected_not_verified(self):
        rows=[x.replace('accepted=1','accepted=0') for x in log_lines(1,self.sha)]
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_frame_regression_not_verified(self):
        rows=log_lines(1,self.sha)
        rows[3]=rows[3].replace('frame=2','frame=1')
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_empty_outcome_not_verified(self):
        rows=log_lines(1,self.sha)
        rows[4]=rows[4].replace('outcome=defeat','something_else=defeat')
        self.assertEqual(w.audit_lines(rows,self.sha)['status'],'not_verified')

    def test_one_command_demo_exports_and_calls_live_launcher(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy=Path(tmp)/'experiment.bin'
            report=Path(tmp)/'audit.json'
            with mock.patch.object(w, 'run', return_value={'status':'matched_logged_control'}) as fn:
                result=w.demo(self.cp,policy=policy,report_out=report)
            self.assertEqual(result['status'],'matched_logged_control')
            self.assertEqual(w.parse_policy(policy)['checkpoint_sha256'], self.sha)
            fn.assert_called_once_with(policy,'gl33',report)

    def test_one_command_default_policy_is_content_addressed(self):
        with mock.patch.object(w, 'prepare', return_value={'status':'exported'}) as prep, \
             mock.patch.object(w, 'run', return_value={'status':'not_verified'}) as run:
            out=w.demo(self.cp)
        path=prep.call_args.args[1]
        self.assertIn(self.sha[:12],str(path))
        self.assertEqual(path.name, f'{self.cp.stem}_{self.sha[:12]}_TSKILL2.bin')
        self.assertEqual(out['status'],'not_verified')
        self.assertEqual(run.call_args.args[0],path)

    def test_one_command_rejects_mismatched_expected_hash_before_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(w, 'run') as fn:
                with self.assertRaisesRegex(ValueError,'checkpoint SHA-256 mismatch'):
                    w.demo(self.cp,policy=Path(tmp)/'bad.bin',expected_sha='0'*64)
            fn.assert_not_called()

    def test_one_command_rejects_missing_model_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(w, 'run') as fn:
                with self.assertRaises(FileNotFoundError):
                    w.demo(Path(tmp)/'missing.pt')
            fn.assert_not_called()

    def test_nonwindows_fails_closed(self):
        with mock.patch.object(w.sys,'platform','linux'):
            with self.assertRaisesRegex(RuntimeError,'Windows'):
                w.run(self.policy)


@unittest.skipUnless(os.getenv('TAISEI_E1_PT'), 'requires authentic archived E1 checkpoint')
class ActualCheckpointIntegration(unittest.TestCase):
    def test_export_actual_e1_challenger(self):
        cp=Path(os.environ['TAISEI_E1_PT'])
        if not cp.is_file():self.fail('configured E1 checkpoint missing')
        self.assertEqual(w.sha256_file(cp),'be320e7764b236b6d2b8c42eb26524fdd57345fe2182cc916ca53f0cdf8d4478')
        with tempfile.TemporaryDirectory() as tmp:
            o=Path(tmp)/'e1.bin'
            p=w.prepare(cp,o)
            self.assertEqual(p['policy_sha256'],'63e1acc2b27aa74bb766f079061b2f1b3e0e13097a4f885fcfc8396c14c53423')
            self.assertLess(p['max_absolute_logit_difference'],1e-5)

if __name__ == '__main__':unittest.main(verbosity=2)
