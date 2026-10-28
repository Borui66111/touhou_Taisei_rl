#!/usr/bin/env python3
"""Fail-closed Windows-to-live-control bridge for canonical TSKILL2 Taisei policies.

This is not a simulator and never claims to have viewed rendered frames. It
reuses the verified main-branch exporter and Windows launcher, and independently
audits the *fresh* launch manifest for matching checkpoint identities and real
logged control decisions, including multi-run sessions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import math
import struct
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
MAGIC = b"TSKILL2\0"
MASK_VERSION = "taisei_skill_mask_v2_respawn"
SHAPES = ((128, 546), (128,), (128, 128), (128,), (18, 128), (18,), (3, 128), (3,))
TENSOR_NAMES = ("base.0.weight", "base.0.bias", "base.2.weight", "base.2.bias",
                "actor.weight", "actor.bias", "skill_actor.weight", "skill_actor.bias")
EXPECTED_PAYLOAD_BYTES = 4 * sum(math.prod(s) for s in SHAPES)
FIELD_RE = re.compile(r"([a-z0-9_]+)=([^ ]+)")
HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
REQUIRED_DECISION_FIELDS = frozenset({"frame", "movement", "skill", "accepted", "action_count", "checkpoint_sha256"})
SEALED_MIN, SEALED_MAX = 43000, 43099


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_policy(path: Path, expected_checkpoint_sha: str | None = None) -> dict:
    """Check entire canonical binary, metadata and payload, not magic alone."""
    path = Path(path)
    b = path.read_bytes()
    if len(b) < 48 or b[:8] != MAGIC:
        raise ValueError("not a canonical TSKILL2 policy")
    meta_len, payload_len = struct.unpack("<II", b[8:16])
    if not (0 < meta_len <= 65535) or payload_len != EXPECTED_PAYLOAD_BYTES:
        raise ValueError("invalid metadata/payload size")
    if len(b) != 48 + meta_len + payload_len:
        raise ValueError("truncated or trailing policy bytes")
    meta_bytes, payload = b[48:48+meta_len], b[48+meta_len:]
    if hashlib.sha256(meta_bytes).digest() != b[16:48]:
        raise ValueError("metadata hash mismatch")
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid UTF-8 JSON metadata") from exc
    canonical = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if canonical != meta_bytes:
        raise ValueError("metadata is not canonical UTF-8 JSON")
    expected = {"format_version": 2, "policy_type": "SkillActorCritic", "input_dim": 546,
                "width": 128, "movement_actions": 18, "skill_actions": 3,
                "skill_action_mask_version": MASK_VERSION}
    for k, value in expected.items():
        if meta.get(k) != value:
            raise ValueError(f"incompatible TSKILL2 metadata: {k}={meta.get(k)!r}")
    cp_hash = meta.get("checkpoint_sha256")
    if not isinstance(cp_hash, str) or not HEX_64.fullmatch(cp_hash):
        raise ValueError("invalid embedded checkpoint SHA-256")
    if expected_checkpoint_sha is not None and cp_hash != expected_checkpoint_sha.lower():
        raise ValueError("checkpoint hash mismatch")
    if hashlib.sha256(payload).hexdigest() != meta.get("payload_sha256"):
        raise ValueError("payload hash mismatch")
    import numpy as np
    if not np.isfinite(np.frombuffer(payload, dtype="<f4")).all():
        raise ValueError("non-finite payload weights")
    return {"checkpoint_sha256": cp_hash, "policy_sha256": sha256_file(path),
            "policy_bytes": len(b), "metadata": meta,
            "format": "TSKILL2", "payload_bytes": payload_len}


def verify_forward_logits(checkpoint: Path, policy: Path, *, trials: int = 16) -> dict:
    """Compare both network heads, including argmax, against serialized weights.

    Does not validate the missing C skill-v2 game loader or visual observations.
    """
    import numpy as np
    import torch
    p = parse_policy(policy, sha256_file(checkpoint))
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or not isinstance(obj.get("model"), dict):
        raise ValueError("malformed trusted checkpoint")
    state = obj["model"]
    with Path(policy).open("rb") as f:
        f.seek(8)
        meta_len, payload_len = struct.unpack("<II", f.read(8))
        f.seek(48 + meta_len)
        raw = f.read(payload_len)
    data = np.frombuffer(raw, dtype="<f4")
    if not np.isfinite(data).all():
        raise ValueError("nonfinite binary weights")
    weights = []
    offset = 0
    for name, shape in zip(TENSOR_NAMES, SHAPES):
        n = int(np.prod(shape))
        if name not in state or tuple(state[name].shape) != shape:
            raise ValueError(f"checkpoint tensor mismatch: {name}")
        w = data[offset:offset+n].reshape(shape)
        ref = state[name].detach().cpu().numpy().astype(np.float32)
        if not np.array_equal(w, ref):
            raise ValueError(f"serialized tensor differs: {name}")
        weights.append(w)
        offset += n
    w1,b1,w2,b2,wm,bm,ws,bs = weights
    rng = np.random.default_rng(20261028)
    max_error = 0.0
    for i in range(trials):
        v = (np.zeros(546, dtype=np.float32) if i == 0 else
             np.linspace(-1, 1, 546, dtype=np.float32) if i == 1 else
             rng.normal(0, 0.7, size=546).astype(np.float32))
        hidden = np.tanh(w1 @ v + b1)
        hidden = np.tanh(w2 @ hidden + b2)
        mv, sk = wm @ hidden + bm, ws @ hidden + bs
        with torch.no_grad():
            x = torch.as_tensor(v)
            a = torch.tanh(torch.nn.functional.linear(x, state["base.0.weight"], state["base.0.bias"]))
            a = torch.tanh(torch.nn.functional.linear(a, state["base.2.weight"], state["base.2.bias"]))
            tm = torch.nn.functional.linear(a, state["actor.weight"], state["actor.bias"]).numpy()
            ts = torch.nn.functional.linear(a, state["skill_actor.weight"], state["skill_actor.bias"]).numpy()
        error = max(float(np.max(np.abs(mv-tm))),float(np.max(np.abs(sk-ts))))
        max_error = max(max_error,error)
        if not (np.allclose(mv,tm,rtol=2e-5,atol=2e-5) and
                np.allclose(sk,ts,rtol=2e-5,atol=2e-5) and
                int(mv.argmax()) == int(tm.argmax()) and int(sk.argmax()) == int(ts.argmax())):
            raise ValueError(f"network inference mismatch on trial {i}")
    return {"checkpoint_sha256":p["checkpoint_sha256"],"policy_sha256":p["policy_sha256"],
            "vectors_tested":trials,"max_absolute_logit_difference":max_error,
            "numerical_parity":"pytorch_vs_numpy_verified", "c_loader_parity":"not_tested"}


def prepare(checkpoint: Path, output: Path, expected_sha: str | None = None) -> dict:
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    check_hash = sha256_file(checkpoint)
    if expected_sha and check_hash != expected_sha.lower():
        raise ValueError("checkpoint SHA-256 mismatch; refusing export")
    try:
        from export_live_skill_ppo_v2 import export
    except ImportError as exc:
        raise RuntimeError("canonical TSKILL2 exporter missing") from exc
    import torch
    obj = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_policy_type = obj.get("policy_type") if isinstance(obj, dict) else None
    output.parent.mkdir(parents=True, exist_ok=True)
    export_checkpoint = checkpoint
    adaptation = None
    if source_policy_type == "legacy":
        # One runtime only: wrap 528 movement features into the 546-input
        # skill-v2 ABI, keeping the legacy movement logits and disabling skills.
        from legacy_to_skillv2 import adapt
        adapted_path = output.parent / "adapted_checkpoints" / (
            f"{checkpoint.stem}_{check_hash[:12]}_skillv2.pt")
        adaptation = adapt(checkpoint, adapted_path)
        export_checkpoint = adapted_path
    elif source_policy_type != "skill":
        raise ValueError(f"unsupported source policy_type={source_policy_type!r}; expected skill or exact legacy")
    tmp = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        export(export_checkpoint, tmp)
        parity = verify_forward_logits(export_checkpoint,tmp)
        if output.exists():
            if sha256_file(output) != sha256_file(tmp):
                raise FileExistsError(f"refusing to replace existing different policy: {output}")
            status = "already_identical"
        else:
            # Atomic, same-volume hardlink. Original checkpoint is never edited.
            os.link(tmp, output)
            status = "exported"
        return {"status": status, "checkpoint":str(checkpoint),"output":str(output),
                "source_checkpoint_sha256":check_hash,
                "source_policy_type":source_policy_type,
                "live_controller_design":"TSKILL2_546_18_3_only",
                "adaptation":adaptation,
                **parity}
    finally:
        tmp.unlink(missing_ok=True)


def _message(line: str) -> str:
    # Match the actual launcher log envelope, not arbitrary historic text.
    m=re.match(r"^\d+\s+[A-Z]+\s+\S+\s+[^:]+:\s+(.*?)\s+\[[^]]+\]$", line)
    return m.group(1) if m else line


def _fields(line: str) -> dict:
    return dict(FIELD_RE.findall(line))


def audit_lines(lines: list[str], expected_sha: str) -> dict:
    """Check session attribution without conflating multiple stages or old logs."""
    if not HEX_64.fullmatch(expected_sha):
        raise ValueError("expected checkpoint SHA-256 invalid")
    loaded = 0
    runs = []
    current = None
    violations = []
    replay_paths = []
    for lineno, raw in enumerate(lines,1):
        line = _message(raw)
        if line.startswith("RL skill-v2 policy loaded "):
            data = _fields(line)
            if current is not None:
                violations.append(f"line {lineno}: unexpected policy reload within a stage run")
            loaded += 1
            if data.get("checkpoint_sha256") != expected_sha:
                violations.append(f"line {lineno}: loaded checkpoint identity mismatch")
        elif line.startswith("RL run "):
            data = _fields(line)
            if not loaded:
                violations.append(f"line {lineno}: run occurred before policy-loaded confirmation")
            if current is not None:
                violations.append(f"line {lineno}: previous run has no terminal outcome")
            if data.get("loaded") != "1" or data.get("policy_type") != "skill_v2":
                violations.append(f"line {lineno}: missing verified skill-v2 loaded flag")
            if data.get("checkpoint_sha256") != expected_sha:
                violations.append(f"line {lineno}: run checkpoint identity mismatch")
            try:
                seed = int(data.get("seed", "-1"))
            except ValueError:
                seed = -1
            if seed < 0:
                violations.append(f"line {lineno}: missing/invalid native run seed")
            if SEALED_MIN <= seed <= SEALED_MAX:
                violations.append(f"line {lineno}: sealed evaluation seed in run")
            current = {"start":lineno, "seed":seed,"stage":data.get("stage"),
                       "difficulty":data.get("difficulty"),"decisions":0,
                       "accepted_decisions":0, "last_frame":None,
                       "first_action_count":None,"last_action_count":None,
                       "outcome":None,"replay_path":None}
            runs.append(current)
        elif line.startswith("RL decision "):
            data = _fields(line)
            if current is None:
                violations.append(f"line {lineno}: decision outside run")
                continue
            if not REQUIRED_DECISION_FIELDS.issubset(data):
                violations.append(f"line {lineno}: incomplete decision fields")
                continue
            if data.get("checkpoint_sha256") != expected_sha:
                violations.append(f"line {lineno}: decision checkpoint mismatch")
            try:
                movement = int(data["movement"]);skill = int(data["skill"])
                action_count = int(data["action_count"])
                frame = int(data["frame"])
                if frame < 0 or data.get("accepted") not in {"0", "1"}:
                    violations.append(f"line {lineno}: invalid frame/accepted flag")
                if current["last_frame"] is not None and frame <= current["last_frame"]:
                    violations.append(f"line {lineno}: non-increasing game frame")
                current["last_frame"] = frame
                current["accepted_decisions"] += int(data["accepted"] == "1")
                if not(0 <= movement < 18 and 0 <= skill < 3):
                    violations.append(f"line {lineno}: invalid action ID")
                if action_count <= 0 or (current["last_action_count"] is not None and
                                           action_count <= current["last_action_count"]):
                    violations.append(f"line {lineno}: repeated/out-of-order action count")
                current["last_action_count"] = action_count
                if current["first_action_count"] is None:
                    current["first_action_count"] = action_count
                current["decisions"] += 1
            except (ValueError,KeyError):
                violations.append(f"line {lineno}: invalid action numeric fields")
        elif line.startswith("RL policy outcome="):
            data = _fields(line)
            if current is None:
                violations.append(f"line {lineno}: outcome without run")
            else:
                if data.get("checkpoint_sha256") != expected_sha:
                    violations.append(f"line {lineno}: outcome checkpoint mismatch")
                current["outcome"] = data.get("outcome")
                if not current["outcome"]:
                    violations.append(f"line {lineno}: missing native outcome value")
                if not current["decisions"]:
                    violations.append(f"line {lineno}: no policy decisions before outcome")
                if not current["accepted_decisions"]:
                    violations.append(f"line {lineno}: no accepted policy decisions")
                current = None
        elif line.startswith("Saving "):
            if not runs:
                violations.append(f"line {lineno}: replay save without run")
            elif runs[-1]["outcome"] is None:
                violations.append(f"line {lineno}: replay save before run outcome")
            elif runs[-1]["replay_path"] is not None:
                violations.append(f"line {lineno}: duplicate replay save")
            else:
                path = line[len("Saving "):].strip()
                runs[-1]["replay_path"] = path
                replay_paths.append(path)
    if not loaded:
        violations.append("no policy-loaded event")
    if not runs:
        violations.append("no actual stage runs")
    if current is not None:
        violations.append("last run missing terminal outcome")
    if any(r["decisions"] == 0 for r in runs):
        violations.append("one or more runs have no actual policy decisions")
    if any(r["outcome"] is None for r in runs):
        violations.append("one or more runs lack terminal outcomes")
    return {"status":"matched_logged_control" if not violations else "not_verified",
            "loaded_events":loaded,"run_count":len(runs),"decision_count":sum(x["decisions"] for x in runs),
            "runs":runs,"replay_save_references":replay_paths,"violations":violations,
            "visual_inspection":"not_performed","replay_playability":"not_verified",
            "gameplay_learning":"not_established"}


def audit_manifest(path: Path, policy: Path) -> dict:
    policy = Path(policy).resolve()
    policy_info = parse_policy(policy)
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    if report.get("sha256",{}).get("policy") != policy_info["policy_sha256"]:
        raise ValueError("launcher manifest policy file SHA-256 mismatch")
    if report.get("policy_metadata",{}).get("checkpoint_sha256") != policy_info["checkpoint_sha256"]:
        raise ValueError("launcher manifest embedded checkpoint hash mismatch")
    capture=report.get("log_capture",{})
    if capture.get("status")!="captured":
        raise ValueError("fresh game log not captured: " + str(capture.get("reason","unknown")))
    if not isinstance(capture.get("relevant_lines"),list):
        raise ValueError("missing launcher fresh-log content")
    out=audit_lines(capture["relevant_lines"], policy_info["checkpoint_sha256"])
    out.update({"launcher_manifest":str(Path(path).resolve()),
                "policy_sha256":policy_info["policy_sha256"],
                "checkpoint_sha256":policy_info["checkpoint_sha256"],
                "launched_binary_sha256":report.get("sha256",{}).get("game"),
                "launcher_exit_status":report.get("exit_status")})
    if report.get("exit_status") != 0:
        out["status"]="not_verified"
        out["violations"].append("game launch did not exit successfully")
    if report.get("status") == "partial":
        out["status"]="not_verified"
        out["violations"].append("launcher classified evidence capture as partial")
    return out


def run(policy: Path, renderer: str = "gl33", report_out: Path | None = None,
        resources: Path | None = None) -> dict:
    if sys.platform != "win32":
        raise RuntimeError("rendered launcher requires Windows workstation; no fake game fallback")
    policy=Path(policy).resolve()
    parse_policy(policy)
    launcher=HERE / "run_live_ppo_windows.py"
    if not launcher.is_file():
        raise FileNotFoundError("missing actual main-branch run_live_ppo_windows.py")
    # Validate required rendered assets before launching the game. Without this,
    # a fatal startup error looks like a model policy with zero decisions.
    from taisei_resource_doctor import select_resources
    from run_live_ppo_windows import ROOT, PATHS
    resource_override = Path(resources).resolve() if resources is not None else None
    resource_root, resource_report = select_resources(
        ROOT, PATHS["resources"], resource_override)
    scratch=Path.home()/"AppData"/"Local"/"hermes"/"cache"/"scratch"
    previous={f.resolve() for f in scratch.glob("taisei-live-ppo-*.json")} if scratch.is_dir() else set()
    argv=[sys.executable,str(launcher),"--renderer",renderer,"--rl-policy",str(policy)]
    argv.extend(["--resources", str(resource_root)])
    proc=subprocess.run(argv, check=False,cwd=str(HERE.parent))
    recent=sorted({f.resolve() for f in scratch.glob("taisei-live-ppo-*.json")}-previous)
    if len(recent)!=1:
        raise RuntimeError(f"cannot identify exactly one new launcher manifest; found {len(recent)}")
    audit=audit_manifest(recent[0],policy)
    audit["resource_preflight"] = resource_report
    if proc.returncode!=0:
        audit["status"]="not_verified"
        audit["violations"].append("launcher subprocess returned nonzero")
    if report_out:
        report_out.parent.mkdir(parents=True,exist_ok=True)
        with Path(report_out).open("x",encoding="utf-8") as f:
            json.dump(audit,f,indent=2,sort_keys=True)
            f.write("\n")
    return audit


def demo(checkpoint: Path, *, policy: Path | None = None, renderer: str = "gl33",
         expected_sha: str | None = None, report_out: Path | None = None,
         resources: Path | None = None) -> dict:
    """Convert a trusted skill-v2 .pt and launch the existing rendered game.

    This is the one-command user path: the exported binary is tied to the
    checkpoint digest and never overwrites a different experiment's output.
    A real Windows game, its policy loader, and its resources must exist.
    """
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint missing: {checkpoint}")
    checkpoint_sha = sha256_file(checkpoint)
    if policy is None:
        policy = HERE.parent / "results" / "w2ctrl" / (
            f"{checkpoint.stem}_{checkpoint_sha[:12]}_TSKILL2.bin")
    exported = prepare(checkpoint, policy, expected_sha)
    if report_out is None:
        report_out = HERE.parent / "results" / "w2ctrl" / "audits" / (
            f"{checkpoint.stem}_{uuid.uuid4().hex}.json")
    result = (run(Path(policy), renderer, report_out) if resources is None else
              run(Path(policy), renderer, report_out, resources=resources))
    return {"status": result["status"], "model_conversion": exported,
            "live_control_audit": result, "audit_file": str(Path(report_out).resolve())}


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    exp=sub.add_parser("prepare",help="export verified 546x18+3 skill PT to canonical TSKILL2")
    exp.add_argument("checkpoint",type=Path);exp.add_argument("output",type=Path)
    exp.add_argument("--expected-sha256")
    inspect=sub.add_parser("inspect",help="validate canonical TSKILL2 binary metadata")
    inspect.add_argument("policy",type=Path)
    check=sub.add_parser("audit",help="audit launcher fresh-log manifest, including multi-run sessions")
    check.add_argument("manifest",type=Path);check.add_argument("policy",type=Path)
    launch=sub.add_parser("run",help="launch actual Windows build, then audit NEW game log")
    launch.add_argument("policy",type=Path);launch.add_argument("--renderer",default="gl33")
    launch.add_argument("--report",type=Path)
    launch.add_argument("--resources",type=Path,help="explicit compiled Taisei resource root")
    single=sub.add_parser("demo",help="ONE COMMAND: convert trusted skill PT and launch/audit Windows Taisei")
    single.add_argument("checkpoint",type=Path)
    single.add_argument("--policy",type=Path,help="optional target TSKILL2 path (default: content-addressed results/w2ctrl)")
    single.add_argument("--expected-sha256",help="optional trusted checkpoint digest")
    single.add_argument("--renderer",default="gl33")
    single.add_argument("--report",type=Path)
    single.add_argument("--resources",type=Path,help="explicit compiled Taisei resource root")
    args=p.parse_args(argv)
    try:
        if args.command=="prepare":
            result=prepare(args.checkpoint,args.output,args.expected_sha256)
        elif args.command=="inspect":
            result=parse_policy(args.policy)
        elif args.command=="audit":
            result=audit_manifest(args.manifest,args.policy)
        elif args.command=="demo":
            result=demo(args.checkpoint,policy=args.policy,renderer=args.renderer,
                        expected_sha=args.expected_sha256,report_out=args.report,
                        resources=args.resources)
        else:
            result=run(args.policy,args.renderer,args.report,resources=args.resources)
        print(json.dumps(result,indent=2,sort_keys=True))
        return 0 if result.get("status") not in {"not_verified"} else 2
    except (OSError,ValueError,RuntimeError,KeyError) as exc:
        print(f"w2ctrl refused: {type(exc).__name__}: {exc}",file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
