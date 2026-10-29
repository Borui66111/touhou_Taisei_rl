"""Tests only patch safety and expected source transformations; NOT game build."""
from pathlib import Path
import importlib.util
import tempfile
import unittest
s=importlib.util.spec_from_file_location('patch',Path(__file__).with_name('instrument_replay_writer.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

EXCERPT=(
 'bool replay_save(Replay *rpy, const char *name) {\n'
 '\tSDL_IOStream *file = vfs_open(p, VFS_MODE_WRITE);\n'
 '\tif(!file)return false;\n'
 +m.NEEDLE+'\n}\n'
)
class PatchTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
  self.p=Path(self.t.name)/'replay.c';self.p.write_text(EXCERPT)
 def test_dry_run_is_read_only(self):
  before=self.p.read_bytes()
  self.assertEqual(m.inspect(self.p)['status'],'ready')
  self.assertEqual(self.p.read_bytes(),before)
  self.assertFalse((self.p.with_name('replay.c.rsi-preinstrument-backup')).exists())
 def test_apply_logs_stage_and_event_counts(self):
  r=m.apply(self.p);self.assertEqual(r['status'],'applied_rebuild_required')
  self.assertEqual(self.p.with_name('replay.c.rsi-preinstrument-backup').read_text(),EXCERPT)
  body=self.p.read_text()
  for x in ('RSI_REPLAY_WRITE_BEGIN','RSI_REPLAY_WRITE_END','RSI_REPLAY_WRITE_INVALID','rsi_events','SDL_CloseIO(file)','rsi_bytes <= 44'):
   self.assertIn(x,body)
  self.assertFalse(r['runtime_verified'])
 def test_idempotent(self):
  m.apply(self.p)
  self.assertEqual(m.apply(self.p)['status'],'already_instrumented')
 def test_unsupported_context_refused(self):
  self.p.write_text(EXCERPT.replace('SDL_CloseIO(file)','SDL_CloseIO(otherfile)'))
  self.assertEqual(m.apply(self.p)['status'],'source_signature_mismatch')
  self.assertFalse(self.p.with_name('replay.c.rsi-preinstrument-backup').exists())
 def test_existing_backup_refused(self):
  p=self.p.with_name('replay.c.rsi-preinstrument-backup');p.write_text('existing')
  self.assertEqual(m.apply(self.p)['status'],'backup_already_exists_refuse_overwrite')
  self.assertEqual(self.p.read_text(),EXCERPT)
 def test_missing_source(self):
  self.assertEqual(m.inspect(Path(self.t.name)/'missing.c')['status'],'missing_source')
 def test_duplicate_needles_refused(self):
  self.p.write_text(EXCERPT.replace(m.NEEDLE,m.NEEDLE+'\n'+m.NEEDLE))
  self.assertEqual(m.inspect(self.p)['status'],'source_signature_mismatch')
if __name__=='__main__':unittest.main()
