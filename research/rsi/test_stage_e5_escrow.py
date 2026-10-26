"""Fail-closed E5 permanent-escrow recovery tests; no native Taisei execution."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from stage_e5_escrow import stage, verify_escrow

ROOT = Path(__file__).resolve().parent / "evidence" / "escrow_e3_e4_20261026_v1"


class E5EscrowStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.out = self.base / "results"

    def editable(self):
        root = self.base / "escrow"
        shutil.copytree(ROOT, root)
        return root

    def test_stages_all_immutable_native_evidence_with_verified_hashes(self):
        report = stage(ROOT, self.out)
        self.assertEqual(report["verified_files"], 33)
        self.assertEqual(report["source_run_ids"], {"e3": 40219788151, "e4": 40220975924})
        self.assertFalse(report["native_game_executed"])
        self.assertFalse(report["native_replay_verified"])
        self.assertFalse(report["source_checkout_recovered"])
        self.assertFalse(report["gameplay_improvement_claimed"])
        self.assertEqual(len(list((self.out / "e3_artifact").iterdir())), 17)
        self.assertEqual(len(list((self.out / "e4_artifact").iterdir())), 16)
        self.assertEqual((self.out / "e4_artifact/teacher_bc_seed48030.pt").read_bytes(),
                         (ROOT / "e4/teacher_bc_seed48030.pt").read_bytes())
        with self.assertRaises(FileExistsError):
            stage(ROOT, self.out)

    def test_tampered_checkpoint_is_rejected_before_any_copy(self):
        root = self.editable()
        p = root / "e4/teacher_bc_rebalanced_seed48030.pt"
        p.write_bytes(p.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "SHA256/size mismatch"):
            stage(root, self.out)
        self.assertFalse((self.out / "e3_artifact").exists())
        self.assertFalse((self.out / "e4_artifact").exists())

    def test_missing_and_extra_files_are_rejected(self):
        root = self.editable()
        (root / "e3/e3_manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            stage(root, self.out)
        shutil.copyfile(ROOT / "e3/e3_manifest.json", root / "e3/e3_manifest.json")
        (root / "e4/unexpected.bin").write_bytes(b"new")
        with self.assertRaisesRegex(ValueError, "file set mismatch"):
            stage(root, self.out)

    def test_symlink_file_and_directory_rejected(self):
        root = self.editable()
        target = root / "e3/e3_manifest.json"
        target.unlink()
        target.symlink_to(ROOT / "e3/e3_manifest.json")
        with self.assertRaisesRegex(ValueError, "Unsafe or missing"):
            stage(root, self.out)
        target.unlink()
        shutil.copyfile(ROOT / "e3/e3_manifest.json", target)
        shutil.rmtree(root / "e4")
        (root / "e4").symlink_to(ROOT / "e4", target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
            stage(root, self.out)

    def test_wrong_commit_run_or_manifest_critical_hash_rejected(self):
        for mutation, expected in [
            (lambda m: m.update(pinned_native_commit="0" * 40), "provenance mismatch"),
            (lambda m: m["artifacts"]["e3"].update(run_id=123), "run/artifact identity mismatch"),
            (lambda m: m["artifacts"]["e4"]["files"]["teacher_bc_rebalanced_seed48030.pt"].update(sha256="0" * 64), "Critical E5 identity mismatch"),
        ]:
            with self.subTest(expected=expected):
                root = self.editable() if not (self.base / "escrow").exists() else self.base / "escrow"
                manifest_path = root / "manifest.json"
                original = manifest_path.read_text()
                data = json.loads(original)
                mutation(data)
                manifest_path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, expected):
                    stage(root, self.out)
                manifest_path.write_text(original)
                self.assertFalse((self.out / "e3_artifact").exists())

    def test_existing_target_is_never_overwritten(self):
        self.out.mkdir()
        target = self.out / "e3_artifact"
        target.mkdir()
        sentinel = target / "keep.txt"
        sentinel.write_text("important")
        with self.assertRaises(FileExistsError):
            stage(ROOT, self.out)
        self.assertEqual(sentinel.read_text(), "important")
        self.assertFalse((self.out / "e4_artifact").exists())

    def test_real_escrow_manifest_matches_every_committed_file(self):
        report = verify_escrow(ROOT)
        self.assertEqual(report["verified_files"], 33)
        self.assertGreater(report["verified_bytes"], 5_000_000)


if __name__ == "__main__":
    unittest.main()
