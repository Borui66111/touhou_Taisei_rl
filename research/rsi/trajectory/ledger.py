"""Append-only, provenance-aware RSI event ledger.

This library validates research process evidence; it does not execute Taisei,
train a policy, judge screenshots, or claim experimental improvement.
Requires only Python 3.11+ stdlib.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import pathlib
import sys

LEVELS = {"L0", "L1", "L2", "L3", "L4"}
KINDS = {"hypothesis", "experiment_plan", "execution", "measurement",
         "diagnosis", "decision", "pivot", "human_review", "checkpoint"}
STATUSES = {"proposed", "running", "succeeded", "failed", "blocked",
            "superseded", "review_required"}
FIELDS = {"event_id", "timestamp_utc", "project", "level", "kind", "status",
          "parent_ids", "hypothesis", "intervention", "expectation",
          "observation", "interpretation", "evidence", "limits",
          "decision", "next_action", "cost", "metrics", "review_required"}
# Initial history includes an optional missing decision field; normalize via this
# only when loading legacy initial seed events, without fabricating outcomes.
OPTIONAL_LEGACY_FIELDS = {"decision"}


def read_ledger(path):
    p = pathlib.Path(path)
    events = []
    if not p.exists():
        return events
    for line_no, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{p}:{line_no}: blank line in append-only ledger")
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p}:{line_no}: invalid JSON") from exc
        if not isinstance(event, dict):
            raise ValueError(f"{p}:{line_no}: event must be an object")
        events.append(event)
    validate(events)
    return events


def validate(events):
    ids = set()
    prev_date = None
    if not isinstance(events, list):
        raise ValueError("Ledger must be a list of events")
    for index, e in enumerate(events):
        if not isinstance(e, dict):
            raise ValueError(f"event {index}: not a JSON object")
        missing = FIELDS.difference(e) - OPTIONAL_LEGACY_FIELDS
        if missing:
            raise ValueError(f"event {index}: missing fields: {sorted(missing)}")
        key = e["event_id"]
        if not isinstance(key, str) or not key or key in ids:
            raise ValueError(f"event {index}: missing/duplicate event id")
        if e["level"] not in LEVELS or e["kind"] not in KINDS or e["status"] not in STATUSES:
            raise ValueError(f"{key}: unknown level/kind/status")
        if e["project"] != "touhou_taisei_rl":
            raise ValueError(f"{key}: unexpected project")
        try:
            d = datetime.fromisoformat(e["timestamp_utc"].replace("Z", "+00:00"))
            if d.utcoffset() is None or d.utcoffset().total_seconds() != 0:
                raise ValueError("not UTC")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: invalid UTC timestamp") from exc
        if prev_date is not None and d < prev_date:
            raise ValueError(f"{key}: timestamps must not regress")
        if not isinstance(e["parent_ids"], list) or any(
                p not in ids for p in e["parent_ids"]):
            raise ValueError(f"{key}: unknown/forward parent event")
        for k in ("hypothesis", "intervention", "expectation", "observation",
                  "interpretation", "next_action"):
            if not isinstance(e[k], str) or not e[k].strip():
                raise ValueError(f"{key}: missing meaningful {k}")
        if not isinstance(e["limits"], list) or not isinstance(e["review_required"], bool):
            raise ValueError(f"{key}: invalid limitations/review flag")
        if not isinstance(e["evidence"], list):
            raise ValueError(f"{key}: evidence must be a list")
        for ref in e["evidence"]:
            if (not isinstance(ref, dict)
                    or ref.get("type") not in {"commit", "workflow", "issue", "source", "branch", "artifact", "report"}
                    or not isinstance(ref.get("url"), str)
                    or not ref["url"].startswith("https://")):
                raise ValueError(f"{key}: malformed evidence reference")
        if not isinstance(e["cost"], dict) or not isinstance(e["metrics"], dict):
            raise ValueError(f"{key}: cost/metrics must be dictionaries")
        if e["kind"] == "measurement" and e["status"] == "succeeded" and not e["evidence"]:
            raise ValueError(f"{key}: successful measurement without evidence link")
        if any(43000 <= x <= 43099 for x in _seed_values(e.get("metrics", {}))):
            raise ValueError(f"{key}: sealed final seed in research metrics")
        ids.add(key)
        prev_date = d
    return {"events": len(events), "last_event": events[-1]["event_id"] if events else None}


def _seed_values(metrics):
    values = []
    for k in ("seed", "training_seed", "eval_seed", "eval_seeds", "training_seeds"):
        v = metrics.get(k)
        if isinstance(v, int) and not isinstance(v, bool):
            values.append(v)
        elif isinstance(v, list):
            values.extend(x for x in v if isinstance(x, int) and not isinstance(x, bool))
    return values


def append_event(path, event):
    """Validate full history and append one canonical JSON object; never rewrite history."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = read_ledger(path)
    validate(old + [event])
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    return {"new_event": event["event_id"], "total": len(old) + 1}


def summary(events):
    validate(events)
    by_level = {lvl: sum(e["level"] == lvl for e in events) for lvl in sorted(LEVELS)}
    by_status = {status: sum(e["status"] == status for e in events) for status in sorted(STATUSES)}
    # Never collapse successful *engineering* events into learned gameplay progress.
    native_verified = [e for e in events if e["kind"] == "measurement"
                       and e["metrics"].get("verified_environment") == "native_reexecuted_and_verified"]
    newest = events[-1] if events else None
    pending = [e for e in events if e["status"] in {"proposed", "blocked", "review_required"}]
    return {
        "total_events": len(events),
        "levels": by_level, "status_counts": by_status,
        "native_reexecuted_measurements": len(native_verified),
        "latest_event": newest["event_id"] if newest else None,
        "last_recorded_next_action": newest["next_action"] if newest else None,
        "open_research_threads": [{"event_id": e["event_id"], "level": e["level"],
                                   "hypothesis": e["hypothesis"], "next_action": e["next_action"]}
                                  for e in pending],
        "validated_competence_improvement": None,
        "reason": "No comparable, native-reexecuted champion/challenger evidence with provenance is recorded in the initial ledger.",
        "continuous_agenda": True,
        "auto_promotion": False,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["validate", "summarize", "append"])
    p.add_argument("--ledger", required=True)
    p.add_argument("--event-json", help="new event JSON path for append command")
    p.add_argument("--out", help="optional JSON summary path")
    a = p.parse_args(argv)
    if a.command == "append":
        if not a.event_json:
            p.error("--event-json required to append")
        event = json.loads(pathlib.Path(a.event_json).read_text(encoding="utf-8"))
        result = append_event(a.ledger, event)
    else:
        events = read_ledger(a.ledger)
        result = validate(events) if a.command == "validate" else summary(events)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if a.out:
        path = pathlib.Path(a.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
