"""Regression: missing Taisei renderer atlas must fail before spawning a game.

These tests do not run Taisei or invent rendered gameplay.
"""
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scripts'))
import w2ctrl
import export_live_skill_ppo_v2 as exporter


class LauncherPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.td=tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.td.cleanup)
        cls.base=Path(cls.td.name)
        cls.scripts=cls.base/'scripts'; cls.scripts.mkdir()
        (cls.scripts/'run_live_ppo_windows.py').write_text('# test-only marker')
        cls.ckpt=cls.base/'skill.pt';cls.bin=cls.base/'skill.bin'
        state={k:torch.zeros(shape,dtype=torch.float32) for k,shape in zip(w2ctrl.TENSOR_NAMES,w2ctrl.SHAPES)}
        torch.save({'model':state,'policy_type':'skill','input_dim':546,'width':128,
                    'actions':18,'skills':3,'skill_action_mask_version':w2ctrl.MASK_VERSION},cls.ckpt)
        exporter.export(cls.ckpt,cls.bin)

    def test_missing_atlas_prevents_game_subprocess(self):
        root=self.base/'missing_assets';root.mkdir(exist_ok=True)
        fake=types.ModuleType('run_live_ppo_windows')
        fake.ROOT=root
        fake.PATHS={'resources':root/'stale'}
        with (patch.object(w2ctrl.sys,'platform','win32'),
              patch.object(w2ctrl,'HERE',self.scripts),
              patch.dict(sys.modules,{'run_live_ppo_windows':fake}),
              patch.object(w2ctrl.subprocess,'run') as run):
            with self.assertRaisesRegex(FileNotFoundError,'portraits.tsatlas.zst'):
                w2ctrl.run(self.bin)
            run.assert_not_called()

    def test_resource_override_is_forwarded_when_found(self):
        root=self.base/'valid_assets';root.mkdir(exist_ok=True)
        atlas=root/'portraits.tsatlas.zst';atlas.write_bytes(b'fake nonempty atlas for preflight only')
        fake=types.ModuleType('run_live_ppo_windows');fake.ROOT=root
        fake.PATHS={'resources':root/'stale'}
        def stop(*args,**kwargs):
            argv=args[0]
            self.assertIn('--resources',argv)
            self.assertEqual(argv[argv.index('--resources')+1],str(root.resolve()))
            raise RuntimeError('subprocess intercepted by test fixture')
        with (patch.object(w2ctrl.sys,'platform','win32'),
              patch.object(w2ctrl,'HERE',self.scripts),
              patch.dict(sys.modules,{'run_live_ppo_windows':fake}),
              patch.object(w2ctrl.subprocess,'run',side_effect=stop) as run):
            with self.assertRaisesRegex(RuntimeError,'subprocess intercepted'):
                w2ctrl.run(self.bin,resources=root)
            run.assert_called_once()


if __name__=='__main__': unittest.main()
