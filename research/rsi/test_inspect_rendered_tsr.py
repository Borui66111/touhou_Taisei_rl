"""Source-only tests for honest local Taisei replay capture preflight."""
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from inspect_rendered_tsr import MAGIC, ZSTD_EMPTY_FRAME, inspect

# Exact 44-byte uploaded artifact on 2026-10-26: header plus empty frames.
# This fixture is public replay *structure*, not user gameplay or model data.
HEADER_ONLY_HEX = (
    "686f6e6fe29da4756d690e80010500000022000000"
    "28b52ffd200001000000000000"
    "28b52ffd200001000069"
)

class LocalTSRPreflightTests(unittest.TestCase):
    def write(self, data):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        p=Path(tmp.name)/"capture.tsr"
        p.write_bytes(data)
        return p

    def test_uploaded_44_byte_recording_is_not_gameplay(self):
        blob=bytes.fromhex(HEADER_ONLY_HEX)
        self.assertEqual(len(blob),44)
        self.assertEqual(hashlib.sha256(blob).hexdigest(),
                         "3eb7427f4c3372d997c3718c699803c16bb42fead5a0f644ff800bed48958943")
        report=inspect(self.write(blob))
        self.assertEqual(report["status"],"invalid_empty_metadata")
        self.assertEqual(report["events_offset"],34)
        self.assertEqual(report["replay_struct_version"],14)
        self.assertFalse(report["playable_verified"])
        self.assertFalse(report["policy_checkpoint_verified"])

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(inspect(Path(d)/"missing.tsr")["status"],"missing_file")

    def test_truncated_before_header(self):
        self.assertEqual(inspect(self.write(MAGIC))["status"],"truncated_before_replay_header")

    def test_empty_events_fail_closed(self):
        b=MAGIC+struct.pack("<H",0x800e)+bytes([1,5,0,0,0])+struct.pack("<I",26)
        b+=b"\x28\xb5\x2f\xfd\x20"+ZSTD_EMPTY_FRAME
        self.assertEqual(inspect(self.write(b))["status"],"invalid_empty_events")

    def test_nonempty_looking_sections_do_not_claim_playability(self):
        b=MAGIC+struct.pack("<H",0x800e)+bytes([1,5,0,0,0])+struct.pack("<I",29)
        b+=b"\x28\xb5\x2f\xfd\x11\x22\x33\x44"+b"\x28\xb5\x2f\xfd\x44\x33"
        result=inspect(self.write(b))
        self.assertEqual(result["status"],"header_plausible_full_decode_required")
        self.assertFalse(result["playable_verified"])
        self.assertFalse(result["gameplay_events_verified"])

if __name__=="__main__":
    unittest.main()
