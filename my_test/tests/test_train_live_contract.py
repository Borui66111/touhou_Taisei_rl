"""Verify future training defaults to the single live-compatible policy family.

Uses an isolated fake native_rl module; no toy Taisei dynamics or fake game
results are created by this source-level contract test.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'train_torch_ppo.py'

class TrainingContractTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        env=types.ModuleType('native_rl.env')
        class EnvConfig:
            def __init__(self,**kw):self.settings=kw
        env.EnvConfig=EnvConfig
        core=types.ModuleType('native_rl.torch_ppo')
        core.SKILL_ACTION_MASK_VERSION='taisei_skill_mask_v2_respawn'
        def mock_train(cfg,**kw):
            self.calls.append((cfg,kw))
            return {'source_contract_only':True}
        core.train=mock_train
        self.patched=patch.dict(sys.modules,{'native_rl':types.ModuleType('native_rl'),
                                            'native_rl.env':env,'native_rl.torch_ppo':core})
        self.patched.start()
        spec=importlib.util.spec_from_file_location('training_contract_test_import',SCRIPT)
        self.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def tearDown(self):self.patched.stop()

    def test_default_is_exact_live_mask_v2_skill(self):
        self.module.main(['--library','missing-native-is-okay-for-source-stub'])
        self.assertEqual(len(self.calls),1)
        cfg,kw=self.calls[0]
        self.assertEqual(kw['policy_type'],'skill')
        self.assertTrue(kw['skill_action_mask'])
        self.assertNotIn('skill_action_mask_version',kw)
        self.assertEqual(cfg.settings['frames_per_action'],1)

    def test_explicit_legacy_rejected_before_native_run(self):
        with self.assertRaises(SystemExit) as ctx:
            self.module.main(['--library','dummy','--policy-type','legacy'])
        self.assertEqual(ctx.exception.code,2)
        self.assertEqual(self.calls,[])

    def test_mask_v3_rejected_before_native_run(self):
        with self.assertRaises(SystemExit) as ctx:
            self.module.main(['--library','dummy','--skill-action-mask-version','taisei_skill_mask_v3_surge_discharge'])
        self.assertEqual(ctx.exception.code,2)
        self.assertEqual(self.calls,[])

    def test_init_resume_exclusive(self):
        with self.assertRaises(SystemExit) as ctx:
            self.module.main(['--library','dummy','--init','a.pt','--resume','b.pt'])
        self.assertEqual(ctx.exception.code,2)
        self.assertEqual(self.calls,[])

    def test_native_and_checkpoint_paths_are_forwarded(self):
        self.module.main(['--library','/native/libtaisei.so', '--resource-path','/native/assets',
                          '--init','teacher.pt', '--steps','42', '--out','experiment.pt'])
        cfg,kw=self.calls[0]
        self.assertEqual(kw['init'],'teacher.pt')
        self.assertEqual(kw['outfile'],'experiment.pt')
        self.assertEqual(kw['total_steps'],42)
        self.assertEqual(cfg.settings['resource_path'],'/native/assets')

if __name__=='__main__':unittest.main()
