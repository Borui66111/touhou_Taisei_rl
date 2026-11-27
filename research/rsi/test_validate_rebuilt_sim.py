"""Source-only checks for the rebuilt libtaisei_sim (no native build, no gameplay).

Native validation is research/rsi/validate_rebuilt_sim.py; this guards the pinned inputs,
the ABI declaration and the recorded evidence so they cannot drift silently.
"""
import hashlib
import json
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
REBUILD = ROOT / "my_test/sim_rebuild"
LOCK = REBUILD / "UPSTREAM_PUBLIC.lock"
EVIDENCE = ROOT / "research/rsi/evidence/rebuilt_sim_validation_20261127.json"
sys.path.insert(0, str(ROOT / "my_test"))
sys.path.insert(0, str(ROOT / "research/rsi"))

# C struct in taisei_sim.h for each ctypes structure in native_rl/taisei_ffi.py
STRUCTS = {
    "Vec2": "TaiseiSimVec2", "GlobalConfig": "TaiseiSimGlobalConfig",
    "SimConfig": "TaiseiSimConfig", "EpisodeConfig": "TaiseiSimEpisodeConfig",
    "Action": "TaiseiSimAction", "PlayerState": "TaiseiSimPlayerState",
    "BossState": "TaiseiSimBossState", "Projectile": "TaiseiSimProjectileState",
    "Enemy": "TaiseiSimEnemyState", "Item": "TaiseiSimItemState",
    "Laser": "TaiseiSimLaserState", "LaserPoint": "TaiseiSimLaserPoint",
    "State": "TaiseiSimState", "Buffers": "TaiseiSimStateBuffers",
}


def read_lock():
    out = {}
    for line in LOCK.read_text().splitlines():
        if line.strip():
            k, v = line.split("=", 1)
            out[k] = v
    return out


def patch_text():
    return (REBUILD / read_lock()["PATCH"]).read_text()


def added_file(patch, path):
    m = re.search(r"diff --git a/%s b/%s\n.*?\n@@[^\n]*\n(.*?)(?=\ndiff --git |\Z)" %
                  (re.escape(path), re.escape(path)), patch, re.S)
    if not m:
        raise AssertionError(f"{path} not added by patch")
    return "\n".join(l[1:] for l in m.group(1).splitlines() if l.startswith("+"))


def c_struct_fields(header, name):
    m = re.search(r"typedef struct %s \{(.*?)\} %s;" % (name, name), header, re.S)
    if not m:
        raise AssertionError(f"struct {name} missing")
    body = re.sub(r"//[^\n]*", "", m.group(1))
    fields = []
    for decl in body.split(";"):
        decl = decl.strip()
        if not decl:
            continue
        decl = re.sub(r"^(const\s+)?[A-Za-z_][A-Za-z0-9_]*\s*\**", "", decl, count=1)
        fields += [re.sub(r"[*\s]", "", x) for x in decl.split(",") if x.strip()]
    return fields


class RebuiltSimSourceTests(unittest.TestCase):
    def test_lock_pins_public_upstream_and_patch_hash(self):
        lock = read_lock()
        self.assertEqual(lock["REPO_URL"], "https://github.com/taisei-project/taisei.git")
        self.assertRegex(lock["COMMIT"], r"^[0-9a-f]{40}$")
        digest = hashlib.sha256((REBUILD / lock["PATCH"]).read_bytes()).hexdigest()
        self.assertEqual(digest, lock["PATCH_SHA256"])
        subs = {k[len("SUBMODULE_"):]: v for k, v in lock.items() if k.startswith("SUBMODULE_")}
        # identical to the archived original build (escrow e3 native_build.log)
        log = (ROOT / "research/rsi/evidence/escrow_e3_e4_20261026_v1/e3/native_build.log").read_text()
        self.assertEqual(len(subs), 4)
        for path, sha in subs.items():
            self.assertIn(f"Submodule path '{path}': checked out '{sha}'", log)

    def test_patch_touches_only_expected_files(self):
        touched = set(re.findall(r"^diff --git a/(\S+) b/", patch_text(), re.M))
        self.assertEqual(touched, {
            "meson_options.txt", "src/dialog.c", "src/global.c", "src/main.c", "src/meson.build",
            "src/sim/meson.build", "src/sim/taisei_sim.c", "src/sim/taisei_sim.h",
            "src/sim_hooks.c", "src/sim_hooks.h", "src/stage.c"})

    def test_c_abi_field_order_matches_python_binding(self):
        import ctypes
        import native_rl.taisei_ffi as ffi
        header = added_file(patch_text(), "src/sim/taisei_sim.h")
        self.assertIn("#define TAISEI_SIM_API_VERSION 1", header)
        self.assertEqual(ffi.API_VERSION, 1)
        for py_name, c_name in STRUCTS.items():
            py_fields = [f[0] for f in getattr(ffi, py_name)._fields_]
            self.assertEqual(c_struct_fields(header, c_name), py_fields, c_name)
        for name in ffi.SIZE_CHECKS:
            self.assertIn(f"taisei_sim_sizeof_{name}(void)", header)
        self.assertEqual(ctypes.sizeof(ffi.Vec2), 16)

    def test_recorded_validation_evidence_passes_against_original_archives(self):
        report = json.loads(EVIDENCE.read_text())
        lock = read_lock()
        self.assertTrue(report["pass"])
        self.assertEqual(report["upstream_commit"], lock["COMMIT"])
        self.assertEqual(report["patch_sha256"], lock["PATCH_SHA256"])
        self.assertEqual(report["sealed_final_seeds_used"], 0)
        self.assertFalse(report["digest_compared"])
        checks = report["checks"]
        self.assertEqual(set(checks), {"smoke", "determinism", "demo_replay", "teacher_e4",
                                       "teacher_e2", "replay_roundtrip"})
        self.assertTrue(all(c["pass"] for c in checks.values()))
        self.assertEqual(checks["demo_replay"]["bitwise_identical_rows"], 12000)
        self.assertEqual(checks["demo_replay"]["max_abs_diff"], 0.0)
        self.assertEqual(checks["teacher_e4"]["n_field_mismatches"], 0)
        self.assertEqual(checks["teacher_e4"]["episodes"], 16)

    def test_digest_is_never_compared(self):
        from validate_rebuilt_sim import strip_digest
        self.assertEqual(strip_digest('{"frame": 1, "digest": 5, "graze": 0}'),
                         {"frame": 1, "graze": 0})
        self.assertEqual(strip_digest('RESET: {"seed": 3, "digest": 9}'), {"seed": 3})


if __name__ == "__main__":
    unittest.main()
