"""Self-contained no-game tests for policy file identity and demo compatibility."""
import hashlib
import sys
import tempfile
from pathlib import Path
import struct
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy_format_preflight import (
    LEGACY_BYTES, LEGACY_HEADER, LEGACY_MAGIC, inspect,
)


class PolicyFormatPreflightTests(unittest.TestCase):
    def make_blob(self, payload):
        d = tempfile.TemporaryDirectory()
        path = Path(d.name) / "policy.bin"
        path.write_bytes(payload)
        self.addCleanup(d.cleanup)
        return path

    def test_tppo_rejected_for_skill_v2(self):
        header = LEGACY_HEADER.pack(LEGACY_MAGIC, 528, 128, 18, bytes(32))
        p = self.make_blob(header + bytes(LEGACY_BYTES-len(header)))
        got = inspect(p, "skill-v2")
        self.assertEqual(got["status"], "incompatible_demo_expects_TSKILL2")
        self.assertFalse(got["compatible_family"])
        self.assertEqual(got["input_dim"], 528)

    def test_tppo_legacy_with_allowlist_warning(self):
        header = LEGACY_HEADER.pack(LEGACY_MAGIC, 528, 128, 18, bytes(32))
        p = self.make_blob(header + bytes(LEGACY_BYTES-len(header)))
        got = inspect(p, "legacy-movement")
        self.assertTrue(got["fully_validated"])
        self.assertIn("allowlist", got["launch_recommendation"].lower())

    def test_tppo_truncated(self):
        p = self.make_blob(LEGACY_MAGIC)
        self.assertEqual(inspect(p, "legacy-movement")["status"], "truncated_tppo_header")

    def test_tppo_invalid_shape(self):
        header = LEGACY_HEADER.pack(LEGACY_MAGIC, 546, 128, 18, bytes(32))
        p = self.make_blob(header + bytes(LEGACY_BYTES-len(header)))
        self.assertEqual(inspect(p, "legacy-movement")["status"], "invalid_tppo_shape_or_size")

    def test_skill_family_recognized_but_not_falsely_certified(self):
        p = self.make_blob(b"TSKILL2\x00" + bytes(900))
        got = inspect(p, "skill-v2")
        self.assertTrue(got["compatible_family"])
        self.assertFalse(got["fully_validated"])
        self.assertEqual(got["launch_recommendation"],
                         "verify_local_demo_binary_schema_and_hash_before_launch")

    def test_skill_rejected_for_legacy(self):
        p = self.make_blob(b"TSKILL2\x00" + bytes(900))
        self.assertEqual(inspect(p, "legacy-movement")["status"],
                         "incompatible_demo_expects_TPPOv1")

    def test_unknown_format(self):
        p = self.make_blob(b"NOTAPOLICY")
        self.assertEqual(inspect(p, "skill-v2")["status"], "unknown_binary_format")

    def test_hash_guard(self):
        p = self.make_blob(b"TSKILL2\x00" + bytes(900))
        self.assertEqual(inspect(p, "skill-v2", "0"*64)["status"], "sha256_mismatch")

    def test_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(inspect(Path(d)/"absent.bin", "skill-v2")["status"], "file_missing")


if __name__ == "__main__":
    unittest.main()
