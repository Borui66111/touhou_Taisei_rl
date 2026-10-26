"""Source-only tests for E3/E4 artifact escrow. No gameplay simulation."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from escrow_native_artifacts import archive


class EvidenceEscrowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.e3, self.e4 = self.root / "e3", self.root / "e4"
        self.e3.mkdir()
        self.e4.mkdir()
        self.e3_file = self.e3 / "test.bin"
        self.e4_file = self.e4 / "test.bin"
        self.e3_file.write_bytes(b"authentic-e3-artifact-fixture")
        self.e4_file.write_bytes(b"authentic-e4-artifact-fixture")
        self.dest = self.root / "archived"
        self.expect = {
            "e3": {"test.bin": hashlib.sha256(self.e3_file.read_bytes()).hexdigest()},
            "e4": {"test.bin": hashlib.sha256(self.e4_file.read_bytes()).hexdigest()},
        }

    def run_archive(self):
        with patch("escrow_native_artifacts.EXPECTED", self.expect), patch(
                "escrow_native_artifacts.EXPECTED_COUNTS", {"e3": 1, "e4": 1}):
            return archive(self.e3, self.e4, self.dest)

    def test_verified_copy_is_append_only_and_not_native_execution(self):
        result = self.run_archive()
        self.assertFalse(result["native_game_executed"])
        self.assertFalse(result["source_checkout_recovered"])
        self.assertFalse(result["gameplay_improvement_claimed"])
        self.assertEqual((self.dest / "e3/test.bin").read_bytes(), self.e3_file.read_bytes())
        self.assertEqual(len(result["artifacts"]["e4"]["files"]), 1)
        with self.assertRaises(FileExistsError):
            self.run_archive()

    def test_mutated_critical_file_rejected_without_copy(self):
        self.e4_file.write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.run_archive()
        self.assertFalse(self.dest.exists())

    def test_missing_extra_or_symlink_files_rejected(self):
        self.e3_file.unlink()
        with self.assertRaisesRegex(ValueError, "file count/type mismatch"):
            self.run_archive()
        self.e3_file.write_bytes(b"authentic-e3-artifact-fixture")
        (self.e3 / "extra").write_bytes(b"extra")
        with self.assertRaisesRegex(ValueError, "file count/type mismatch"):
            self.run_archive()
        (self.e3 / "extra").unlink()
        self.e3_file.unlink()
        self.e3_file.symlink_to(self.e4_file)
        with self.assertRaisesRegex(ValueError, "file count/type mismatch"):
            self.run_archive()


if __name__ == "__main__":
    unittest.main()
