#!/usr/bin/env python3
"""Lightweight local run tracker for Phase A experiments."""

import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix fallback
    fcntl = None


SUMMARY_FIELDS = [
    "run_id",
    "run_name",
    "status",
    "started_at",
    "finished_at",
    "duration_seconds",
    "git_hash",
    "command",
    "notes",
]


def utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def git_hash(project_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


@contextmanager
def locked_file(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(path.parent), prefix=f".{path.name}."
    ) as tmp:
        tmp.write(content)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def event_log_path(runs_dir: Path) -> Path:
    return runs_dir / "events.jsonl"


def summary_path(runs_dir: Path) -> Path:
    return runs_dir / "summary.csv"


def run_file_path(runs_dir: Path, run_id: str) -> Path:
    return runs_dir / f"{run_id}.json"


def load_run(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Run file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_run(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def append_event(runs_dir: Path, event_type: str, payload: Dict[str, Any]) -> None:
    record = {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "timestamp": utc_now(),
        **payload,
    }
    path = event_log_path(runs_dir)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with locked_file(lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def append_summary_row(runs_dir: Path, row: Dict[str, Any]) -> None:
    path = summary_path(runs_dir)
    lock_path = path.with_suffix(path.suffix + ".lock")
    normalized_row = {field: row.get(field, "") for field in SUMMARY_FIELDS}

    with locked_file(lock_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(normalized_row)


def build_command_snapshot(command_override: Optional[str]) -> Dict[str, Any]:
    command = command_override or os.environ.get("TRACK_RUN_COMMAND") or shlex.join(sys.argv)
    return {
        "command": command,
        "command_argv": sys.argv,
        "command_source": "override" if command_override else ("env" if os.environ.get("TRACK_RUN_COMMAND") else "argv"),
        "cwd": os.getcwd(),
        "python": sys.executable,
    }


def cmd_start(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    project_root = Path(args.project_root)
    run_id = args.run_id or f"phasea-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
    command_snapshot = build_command_snapshot(args.command)

    payload: Dict[str, Any] = {
        "run_id": run_id,
        "run_name": args.run_name,
        "status": "running",
        "started_at": utc_now(),
        "finished_at": None,
        "project_root": str(project_root),
        "git_hash": git_hash(project_root),
        "notes": args.notes,
        "config": json.loads(args.config_json) if args.config_json else {},
        "command": command_snapshot,
        "stages": [],
        "artifacts": [],
    }

    save_run(run_file_path(runs_dir, run_id), payload)
    append_event(runs_dir, "start", {"run_id": run_id, "run_name": args.run_name, "command": command_snapshot})
    print(run_id)


def cmd_stage(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    path = run_file_path(runs_dir, args.run_id)
    payload = load_run(path)

    stage_record = {
        "name": args.stage,
        "status": args.status,
        "timestamp": utc_now(),
        "message": args.message,
        "artifact": args.artifact,
    }
    payload["stages"].append(stage_record)
    if args.artifact:
        payload["artifacts"].append(args.artifact)

    save_run(path, payload)
    append_event(
        runs_dir,
        "stage",
        {
            "run_id": args.run_id,
            "stage": args.stage,
            "status": args.status,
            "message": args.message,
            "artifact": args.artifact,
        },
    )


def cmd_finish(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    path = run_file_path(runs_dir, args.run_id)
    payload = load_run(path)

    payload["status"] = args.status
    payload["finished_at"] = utc_now()

    started = dt.datetime.fromisoformat(payload["started_at"].replace("Z", "+00:00"))
    finished = dt.datetime.fromisoformat(payload["finished_at"].replace("Z", "+00:00"))
    duration = int((finished - started).total_seconds())

    save_run(path, payload)
    append_event(
        runs_dir,
        "finish",
        {
            "run_id": args.run_id,
            "status": args.status,
            "notes": args.notes or payload.get("notes", ""),
            "duration_seconds": duration,
        },
    )

    append_summary_row(
        runs_dir,
        {
            "run_id": payload["run_id"],
            "run_name": payload.get("run_name", "phase-a"),
            "status": args.status,
            "started_at": payload["started_at"],
            "finished_at": payload["finished_at"],
            "duration_seconds": duration,
            "git_hash": payload.get("git_hash", "unknown"),
            "command": payload.get("command", {}).get("command", ""),
            "notes": args.notes or payload.get("notes", ""),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track Phase A run metadata")
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start")
    start.add_argument("--runs-dir", default="runs/phase_a")
    start.add_argument("--project-root", default=".")
    start.add_argument("--run-id")
    start.add_argument("--run-name", default="phase_a_plain_cgmap_ffr_out")
    start.add_argument("--notes", default="")
    start.add_argument("--command", default="", help="Optional human-readable command for the run")
    start.add_argument("--config-json", default="")
    start.set_defaults(func=cmd_start)

    stage = sub.add_parser("stage")
    stage.add_argument("--runs-dir", default="runs/phase_a")
    stage.add_argument("--run-id", required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--status", required=True, choices=["started", "success", "failed", "skipped"])
    stage.add_argument("--message", default="")
    stage.add_argument("--artifact", default="")
    stage.set_defaults(func=cmd_stage)

    finish = sub.add_parser("finish")
    finish.add_argument("--runs-dir", default="runs/phase_a")
    finish.add_argument("--run-id", required=True)
    finish.add_argument("--status", required=True, choices=["success", "failed", "partial"])
    finish.add_argument("--notes", default="")
    finish.set_defaults(func=cmd_finish)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
