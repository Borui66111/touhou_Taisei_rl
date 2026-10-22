import struct

import numpy as np
import pytest
import torch

from native_rl.env import action_buttons
from native_rl.torch_ppo import ActorCritic, features
from scripts.export_live_ppo import export_checkpoint, infer_exported


def test_checkpoint_export_matches_python_logits_and_greedy_action(tmp_path):
    torch.manual_seed(77)
    model = ActorCritic(528, 18, 128).eval()
    checkpoint = tmp_path / "source.pt"
    torch.save({"model": model.state_dict(), "input_dim": 528, "actions": 18,
                "width": 128, "config": {"stage_id": 1, "difficulty": 2,
                "character": 0, "shot_mode": 0, "practice": False,
                "frames_per_action": 1, "max_frames": 10000}}, checkpoint)
    exported = tmp_path / "policy.bin"
    export_checkpoint(checkpoint, exported)
    assert exported.read_bytes()[:8] == b"TPPOv1\0\0"
    rng = np.random.default_rng(91)
    vector = rng.normal(size=528).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(vector).unsqueeze(0))[0][0].numpy()
    actual = infer_exported(exported, vector)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    assert int(np.argmax(actual)) == int(np.argmax(expected))


def test_python_feature_and_action_contract_has_fixed_dimensions_and_mapping():
    obs = {"player": [0.0] * 8, "boss": [0.0] * 6,
           "bullets": [[0.0] * 7 for _ in range(64)],
           "lasers": [[0.0] * 4 for _ in range(16)],
           "projectile_count": 0, "laser_count": 0}
    assert features(obs).shape == (528,)
    assert [action_buttons(i) for i in range(18)] == [
        32, 33, 34, 36, 40, 37, 41, 38, 42,
        48, 49, 50, 52, 56, 53, 57, 54, 58,
    ]


def test_export_rejects_wrong_checkpoint_shape(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"model": {}, "input_dim": 527, "actions": 18, "width": 128,
                "config": {}}, bad)
    with pytest.raises(ValueError, match="528|shape|dimension"):
        export_checkpoint(bad, tmp_path / "bad.bin")
