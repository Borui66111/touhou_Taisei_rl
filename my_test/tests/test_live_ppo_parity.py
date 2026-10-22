import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from native_rl.torch_ppo import ActorCritic
from scripts.export_live_ppo import export_checkpoint, infer_exported

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "live_ppo_infer_harness.c"


def test_native_c_logits_match_python_for_fixed_observation_vectors(tmp_path, patched_live_source):
    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    bundled_compiler = (ROOT / "toolchains" / "llvm-mingw-20261006" /
                        "llvm-mingw-20261006-ucrt-x86_64" / "bin" / "gcc.exe")
    if not compiler and bundled_compiler.is_file():
        compiler = str(bundled_compiler)
    if not compiler:
        pytest.skip("a C compiler is required for native inference parity; no installed compiler was found")
    torch.manual_seed(20261022)
    model = ActorCritic(528, 18, 128).eval()
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model.state_dict(), "input_dim": 528, "actions": 18,
                "width": 128, "config": {}}, checkpoint)
    policy = tmp_path / "policy.bin"
    export_checkpoint(checkpoint, policy)
    executable = tmp_path / "infer.exe"
    subprocess.run([compiler, "-std=c11", "-O2", "-I", str(patched_live_source),
                    str(HARNESS), "-lm", "-o", str(executable)], check=True)
    rng = np.random.default_rng(5581)
    vectors = [np.zeros(528, dtype=np.float32),
               np.linspace(-2, 2, 528, dtype=np.float32),
               rng.normal(0, 0.75, 528).astype(np.float32)]
    for index, vector in enumerate(vectors):
        vector_file = tmp_path / f"vector-{index}.bin"
        vector.astype("<f4").tofile(vector_file)
        output = subprocess.run([str(executable), str(policy), str(vector_file)],
                                check=True, capture_output=True, text=True)
        actual = np.asarray(json.loads(output.stdout), dtype=np.float32)
        expected = infer_exported(policy, vector)
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        assert int(actual.argmax()) == int(expected.argmax())


def test_native_policy_identity_is_fail_closed_and_action_mapping_is_exact(patched_live_source):
    source = (patched_live_source / "live_ppo.c").read_text(encoding="utf-8")
    assert "sha256_digest(b,FILESIZE" in source
    digest_bytes = bytes.fromhex("476ca4e755f24d351ac68da07719ed21821f84113cbc532b26dd70bd2208ddc1")
    import re
    encoded = re.search(r"expected\[32\]=\{([^}]+)", source).group(1)
    assert bytes(int(byte, 16) for byte in encoded.split(",")) == digest_bytes
    assert "case 0:break;case 1:d|=INFLAG_UP" in source
    assert "case 8:d|=INFLAG_DOWN|INFLAG_RIGHT" in source


def test_action_buttons_contract_includes_neutral_and_focus_variants():
    from native_rl.env import action_buttons
    assert [action_buttons(i) for i in range(18)] == [
        32, 33, 34, 36, 40, 37, 41, 38, 42,
        48, 49, 50, 52, 56, 53, 57, 54, 58,
    ]
