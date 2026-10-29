"""Source-level replay structure tests (synthetic fixtures are not gameplay)."""
import hashlib
import importlib.util
import subprocess
import struct
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('replay_forensics',HERE/'replay_forensics.py')
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
EMPTY=bytes.fromhex('28b52ffd2000010000')


def zstd(raw):
    p=subprocess.run(['zstd','-q','-c','-3'],input=raw,capture_output=True,check=True)
    return p.stdout


def make_meta(*, stage_count=1,event_count=3):
    b=bytearray()
    b+=struct.pack('<B',4)+b'Test'
    b+=struct.pack('<IH',0,stage_count)
    for i in range(stage_count):
        fields=[
            ('I',0),('H',1),('Q',123456),('Q',42132+i),('B',1),('H',0),
            ('Q',0),('B',0),('B',0),('B',0),('B',0),('B',0),
            ('H',240),('H',400),('H',100),('B',3),('H',0),('B',2),
            ('H',0),('B',0),('I',0),('I',1000),('Q',123),('B',0),
            ('B',0),('B',0),('H',event_count),('I',0),
        ]
        for kind,v in fields:b+=struct.pack('<'+kind,v)
    return bytes(b)


def make_tsr(meta,events):
    mc=zstd(meta)
    ec=zstd(events)
    ofs=21+len(mc)+4
    return mod.MAGIC+struct.pack('<H',0x8000|14)+bytes((1,5,0,0,0))+struct.pack('<I',ofs)+mc+b'\0'*4+ec+b'\x69'


class ReplayForensicsTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def file(self,data):
        p=self.root/'run.tsr';p.write_bytes(data);return p

    def test_known_user_empty_replay(self):
        path=Path('/mnt/data/taisei_20261028_17-33-21-938_stg1_reimuA_easy.tsr')
        if not path.exists():self.skipTest('User replay not auto-mounted')
        result=mod.analyze(path)
        self.assertEqual(result['status'],'invalid_empty_metadata_and_events')
        self.assertEqual(result['file_bytes'],44)
        self.assertEqual(result['metadata_alignment_gap_bytes'],4)
        self.assertEqual(result['sha256'],'3eb7427f4c3372d997c3718c699803c16bb42fead5a0f644ff800bed48958943')

    def test_known_two_user_replays_identical(self):
        a=Path('/mnt/data/taisei_20261028_16-34-53-521_stg1_reimuA_easy.tsr')
        b=Path('/mnt/data/taisei_20261028_17-33-21-938_stg1_reimuA_easy.tsr')
        if not a.exists() or not b.exists():self.skipTest('User recordings not available')
        self.assertEqual(a.read_bytes(),b.read_bytes())

    def test_synthetic_full_stage_decodes_not_playability(self):
        events=b''.join(struct.pack('<IBH',t,1,3) for t in (1,2,400))
        result=mod.analyze(self.file(make_tsr(make_meta(),events)))
        self.assertEqual(result['status'],'structurally_decoded_requires_game_playback')
        self.assertEqual(result['decoded_stage_count'],1)
        self.assertEqual(result['decoded_event_count'],3)
        self.assertEqual(result['stages'][0]['seed'],42132)
        self.assertEqual(result['stages'][0]['last_event_frame'],400)
        self.assertEqual(result['replay_playability'],'not_tested')

    def test_empty_events_with_valid_metadata_rejected(self):
        result=mod.analyze(self.file(make_tsr(make_meta(event_count=0),b'')))
        self.assertIn(result['status'],('invalid_empty_events','invalid_or_unverified_decoded_payload'))

    def test_valid_empty_stage_metadata_is_still_nonempty_compressed(self):
        # Even a broken no-stage save should produce playername+flags+stage count;
        # identical Zstd empty frames imply upstream writes never reached stream.
        meta=make_meta(stage_count=0)
        self.assertGreater(len(meta),6)
        self.assertNotEqual(zstd(meta), EMPTY)
        result=mod.analyze(self.file(make_tsr(meta,b'')))
        self.assertEqual(result['status'],'invalid_or_unverified_decoded_payload')
        self.assertIn('count invalid',result['decode_error'])

    def test_malformed_stage_count(self):
        events=struct.pack('<IBH',1,1,0)
        raw=make_meta()
        raw=raw[:9]+struct.pack('<H',0)+raw[11:]
        r=mod.analyze(self.file(make_tsr(raw,events)))
        self.assertEqual(r['status'],'invalid_or_unverified_decoded_payload')
        self.assertIn('count invalid',r['decode_error'])

    def test_truncated_metadata(self):
        r=mod.analyze(self.file(make_tsr(make_meta()[:-5],struct.pack('<IBH',1,1,0))))
        self.assertIn('truncated',r['decode_error'])

    def test_wrong_event_count(self):
        r=mod.analyze(self.file(make_tsr(make_meta(event_count=5),struct.pack('<IBH',1,1,0))))
        self.assertIn('length mismatch',r['decode_error'])

    def test_decreasing_event_frame(self):
        e=b''.join(struct.pack('<IBH',t,2,0) for t in (90,10,100))
        r=mod.analyze(self.file(make_tsr(make_meta(),e)))
        self.assertIn('decreasing',r['decode_error'])

    def test_corrupt_zstd_section(self):
        b=bytearray(make_tsr(make_meta(),struct.pack('<IBH',1,1,0)*3))
        b[24]^=0xff
        r=mod.analyze(self.file(b))
        self.assertEqual(r['status'],'invalid_or_unverified_decoded_payload')

    def test_invalid_header(self):
        r=mod.analyze(self.file(b'not a replay'))
        self.assertEqual(r['status'],'invalid_header')

    def test_missing_file(self):
        r=mod.analyze(self.root/'missing.tsr')
        self.assertEqual(r['status'],'missing_replay')

    def test_bounded_file_size(self):
        p=self.root/'huge.tsr'
        with p.open('wb') as f:f.truncate(mod.MAX_FILE_BYTES+1)
        r=mod.analyze(p)
        self.assertEqual(r['status'],'too_large_for_bounded_audit')

    def test_log_filter_preserves_writer_errors_and_save_line(self):
        p=self.root/'native.log'
        p.write_text('old unrelated\nSaving taisei.tsr\nZSTD_compressStream2 failed\nSDL_WriteIO returned 0\nVFS error: fail\n')
        result=mod.analyze(self.file(make_tsr(make_meta(),struct.pack('<IBH',2,1,3)*3)),log_path=p)
        lines=result['matching_log']['matching_lines']
        self.assertEqual(len(lines),4)

    def test_sentinel_read_only(self):
        b=make_tsr(make_meta(),struct.pack('<IBH',2,1,3)*3)
        p=self.file(b)
        _=mod.analyze(p)
        self.assertEqual(p.read_bytes(),b)

if __name__=='__main__':unittest.main()
