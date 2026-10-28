import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from taisei_resource_doctor import (REQUIRED_ATLAS, inspect_resources, select_resources,
                                    candidate_directories, main)


class ResourceDoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.default = self.root / 'windows-build' / 'wlive_agent_demo' / 'resources'
        self.other = self.root / 'rl_skill_v2' / 'resources'

    def set_asset(self, directory, name=REQUIRED_ATLAS, data=b'TEST_ONLY_NOT_REAL_TAISEI_ATLAS'):
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(data)

    def test_missing_dir(self):
        info = inspect_resources(self.default)
        self.assertEqual(info['status'], 'directory_missing')

    def test_empty_resource_dir(self):
        self.default.mkdir(parents=True)
        self.assertEqual(inspect_resources(self.default)['status'], 'atlas_missing')

    def test_nonzero_direct_atlas_required(self):
        self.set_asset(self.default, data=b'')
        self.assertFalse(inspect_resources(self.default)['atlas_present'])
        (self.default / REQUIRED_ATLAS).write_bytes(b'fixture')
        self.assertTrue(inspect_resources(self.default)['atlas_present'])

    def test_zip_package_resources(self):
        self.default.mkdir(parents=True)
        with zipfile.ZipFile(self.default/'00-taisei.zip', 'w') as z:
            z.writestr(REQUIRED_ATLAS, b'fixture')
        info = inspect_resources(self.default)
        self.assertEqual(info['asset_mode'], 'zip')
        self.assertTrue(info['atlas_present'])

    def test_invalid_zip_not_crash(self):
        self.default.mkdir(parents=True)
        (self.default / 'bad.zip').write_bytes(b'not-a-zip')
        self.assertEqual(inspect_resources(self.default)['status'], 'atlas_missing')

    def test_subdirectory_not_misidentified_as_root(self):
        self.set_asset(self.default/'nested')
        self.assertFalse(inspect_resources(self.default)['atlas_present'])

    def test_valid_configured_resource_is_retained(self):
        self.set_asset(self.default)
        self.set_asset(self.other)
        root, info = select_resources(self.root,self.default)
        self.assertEqual(root, self.default.resolve())
        self.assertEqual(info['selected_by'], 'configured_valid')

    def test_unique_known_alternate_autoselected(self):
        self.default.mkdir(parents=True)
        self.set_asset(self.other)
        root, info = select_resources(self.root,self.default)
        self.assertEqual(root, self.other.resolve())
        self.assertEqual(info['selected_by'], 'one_known_alternative')

    def test_ambiguous_alternates_are_blocked(self):
        self.default.mkdir(parents=True)
        self.set_asset(self.other)
        self.set_asset(self.root/'resources')
        with self.assertRaisesRegex(ValueError,'Several resource roots'):
            select_resources(self.root,self.default)

    def test_no_atlas_gives_diagnostic(self):
        with self.assertRaisesRegex(FileNotFoundError,'portraits.tsatlas.zst'):
            select_resources(self.root,self.default)

    def test_invalid_explicit_override_refused(self):
        self.set_asset(self.default)
        with self.assertRaisesRegex(FileNotFoundError,'Explicit --resources'):
            select_resources(self.root,self.default,self.other)

    def test_explicit_override_selected(self):
        self.set_asset(self.other)
        root, info=select_resources(self.root,self.default,self.other)
        self.assertEqual(root,self.other.resolve())
        self.assertEqual(info['selected_by'],'explicit_override')

    def test_no_network_no_asset_copy(self):
        self.assertLess(len(candidate_directories(self.root,self.default)),20)
        self.assertFalse(self.default.exists())
        self.assertEqual(main(['--workbench',str(self.root),'--configured',str(self.default)]),2)
        self.assertFalse(self.default.exists())


if __name__ == '__main__': unittest.main()

class W2CtrlResourceFlagTests(unittest.TestCase):
    def test_cli_demo_forwards_resource_override(self):
        import w2ctrl as w
        from unittest import mock
        import contextlib
        import io
        path = Path('sample.pt')
        with mock.patch.object(w,'demo',return_value={'status':'matched_logged_control'}) as mocked:
            with contextlib.redirect_stdout(io.StringIO()):
                code = w.main(['demo',str(path),'--resources','C:/my-compiled-game/resources'])
        self.assertEqual(code,0)
        self.assertEqual(mocked.call_args.kwargs['resources'],Path('C:/my-compiled-game/resources'))

    def test_cli_run_forwards_resource_override(self):
        import w2ctrl as w
        from unittest import mock
        import contextlib
        import io
        with mock.patch.object(w,'run',return_value={'status':'matched_logged_control'}) as mocked:
            with contextlib.redirect_stdout(io.StringIO()):
                code = w.main(['run','model.bin','--resources','C:/my-compiled-game/resources'])
        self.assertEqual(code,0)
        self.assertEqual(mocked.call_args.kwargs['resources'],Path('C:/my-compiled-game/resources'))
