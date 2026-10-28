#!/usr/bin/env python3
"""Export trusted local SkillActorCritic checkpoints to the v2 live format.

File layout (all integers little-endian): 8-byte magic ``TSKILL2\\0``;
uint32 metadata byte count; uint32 payload byte count; 32-byte SHA-256 of the
canonical UTF-8 metadata; metadata; little-endian IEEE754 float32 weights in
this exact order:
base.0.weight, base.0.bias, base.2.weight, base.2.bias,
actor.weight, actor.bias, skill_actor.weight, skill_actor.bias.
Metadata includes format/architecture/mask identifiers, SHA-256 of the source
checkpoint, and SHA-256 of the exact payload bytes. No critic is exported.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np
import torch

MAGIC = b"TSKILL2\0"
MASK_VERSION = "taisei_skill_mask_v2_respawn"
ORDER = ("base.0.weight", "base.0.bias", "base.2.weight", "base.2.bias",
         "actor.weight", "actor.bias", "skill_actor.weight", "skill_actor.bias")
SHAPES = ((128, 546), (128,), (128, 128), (128,), (18, 128), (18,),
          (3, 128), (3,))


def export(checkpoint: pathlib.Path, output: pathlib.Path) -> None:
    if not checkpoint.is_file():
        raise ValueError("checkpoint file does not exist")
    try:
        obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"cannot load local checkpoint: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("checkpoint must be a mapping")
    expected = {"policy_type": "skill", "input_dim": 546, "actions": 18,
                "skills": 3, "width": 128}
    for key, value in expected.items():
        if obj.get(key) != value:
            raise ValueError(f"architecture mismatch: {key}={obj.get(key)!r}; expected {value!r}")
    if obj.get("skill_action_mask_version") != MASK_VERSION:
        raise ValueError("skill action mask version missing or unsupported")
    state = obj.get("model")
    if not isinstance(state, dict):
        raise ValueError("model state payload missing")
    chunks = []
    for key, shape in zip(ORDER, SHAPES):
        tensor = state.get(key)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"malformed model tensor {key}")
        arr = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        if not np.isfinite(arr).all():
            raise ValueError(f"non-finite model tensor {key}")
        chunks.append(arr.astype("<f4", copy=False).tobytes(order="C"))
    payload = b"".join(chunks)
    metadata = {"format_version": 2, "policy_type": "SkillActorCritic",
                "input_dim": 546, "width": 128, "movement_actions": 18,
                "skill_actions": 3, "skill_action_mask_version": MASK_VERSION,
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest()}
    meta_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(meta_bytes) > 65535 or len(payload) > 0xffffffff:
        raise ValueError("export exceeds binary format limits")
    data = MAGIC + struct.pack("<II", len(meta_bytes), len(payload)) + hashlib.sha256(meta_bytes).digest() + meta_bytes + payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    args = parser.parse_args()
    try:
        export(args.checkpoint, args.output)
    except Exception as exc:
        print(f"export failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
