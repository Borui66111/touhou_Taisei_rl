"""Export a checked PPO checkpoint to a fixed-width Taisei live-policy format.

Binary format: 8-byte magic, three little-endian uint32 dimensions, 32-byte
checkpoint SHA-256, then float32 arrays W1,b1,W2,b2,Wa,ba (PyTorch row-major).
"""
from __future__ import annotations

import hashlib
import pathlib
import struct

import numpy as np
import torch

from native_rl.torch_ppo import ActorCritic

MAGIC = b"TPPOv1\0\0"
HEADER = struct.Struct("<8sIII32s")
EXPECTED_CHECKPOINT_SHA256 = "872d1e438afdc1df7858a0306c6373596d3ff89978b98d9f1b0a3d2aa378839d"
EXPECTED_CONFIG = {
    "stage_id": 1, "difficulty": 2, "character": 0, "shot_mode": 0,
    "practice": False, "frames_per_action": 1, "max_frames": 10000,
}


def _load(checkpoint_path, expected_sha256=None):
    path = pathlib.Path(checkpoint_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"checkpoint SHA-256 mismatch: {digest}")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if (obj.get("input_dim"), obj.get("actions"), obj.get("width")) != (528, 18, 128):
        raise ValueError("checkpoint dimensions must be 528 inputs, 18 actions, width 128")
    config = obj.get("config")
    if expected_sha256 == EXPECTED_CHECKPOINT_SHA256 and any(
        config.get(key) != value for key, value in EXPECTED_CONFIG.items()
    ):
        raise ValueError(f"checkpoint config mismatch: {config!r}")
    model = ActorCritic(528, 18, 128).cpu().eval()
    try:
        model.load_state_dict(obj["model"], strict=True)
    except Exception as exc:
        raise ValueError(f"checkpoint model shape mismatch: {exc}") from exc
    return path, digest, model


def export_checkpoint(checkpoint_path, output_path, expected_sha256=None):
    path, digest, model = _load(checkpoint_path, expected_sha256)
    state = model.state_dict()
    names = ("base.0.weight", "base.0.bias", "base.2.weight", "base.2.bias",
             "actor.weight", "actor.bias")
    arrays = [state[name].detach().cpu().numpy().astype("<f4", copy=False) for name in names]
    out = pathlib.Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(HEADER.pack(MAGIC, 528, 128, 18, bytes.fromhex(digest)))
        for array in arrays:
            f.write(array.tobytes(order="C"))
    return {"checkpoint": str(path.resolve()), "sha256": digest,
            "input_dim": 528, "width": 128, "actions": 18,
            "bytes": out.stat().st_size}


def infer_exported(binary_path, vector):
    data = pathlib.Path(binary_path).read_bytes()
    if len(data) < HEADER.size:
        raise ValueError("truncated policy file")
    magic, n_in, width, n_actions, _ = HEADER.unpack_from(data)
    if (magic, n_in, width, n_actions) != (MAGIC, 528, 128, 18):
        raise ValueError("invalid policy header")
    dims = ((width, n_in), (width,), (width, width), (width,),
            (n_actions, width), (n_actions,))
    count = sum(int(np.prod(shape)) for shape in dims)
    if len(data) != HEADER.size + count * 4:
        raise ValueError("invalid policy payload size")
    values = np.frombuffer(data, dtype="<f4", offset=HEADER.size)
    arrays, offset = [], 0
    for shape in dims:
        size = int(np.prod(shape))
        arrays.append(values[offset:offset + size].reshape(shape))
        offset += size
    w1, b1, w2, b2, wa, ba = arrays
    x = np.asarray(vector, dtype=np.float32)
    if x.shape != (528,):
        raise ValueError("observation must have 528 float32 features")
    z1 = np.tanh(w1 @ x + b1)
    z2 = np.tanh(w2 @ z1 + b2)
    return wa @ z2 + ba
