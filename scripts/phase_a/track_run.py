#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
from pathlib import Path


def utc_now():
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    start = sub.add_parser("start")
    start.add_argument("--runs-dir", default="runs/phase_a")
    start.add_argument("--run-id", required=False)

    stage = sub.add_parser("stage")
    stage.add_argument("--runs-dir", default="runs/phase_a")
    stage.add_argument("--run-id", required=True)
    stage.add_argument("--stage", required=True)
    stage.add_argument("--status", required=True)

    finish = sub.add_parser("finish")
    finish.add_argument("--runs-dir", default="runs/phase_a")
    finish.add_argument("--run-id", required=True)
    finish.add_argument("--status", required=True)

    args = parser.parse_args()
    runs = Path(args.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)

    if args.cmd == "start":
        run_id = args.run_id or f"phasea-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        payload = {"run_id": run_id, "started_at": utc_now(), "stages": []}
        (runs / f"{run_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(run_id)
        return

    path = runs / f"{args.run_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"run_id": args.run_id, "stages": []}

    if args.cmd == "stage":
        payload["stages"].append({"name": args.stage, "status": args.status, "timestamp": utc_now()})
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    if args.cmd == "finish":
        payload["status"] = args.status
        payload["finished_at"] = utc_now()
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
