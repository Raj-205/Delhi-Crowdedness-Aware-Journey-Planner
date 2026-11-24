"""OPTIONAL — Self-check: find out exactly which step is blocking the app.

Run this any time the UI says "Engine: NOT trained" or an endpoint returns 503.
Pure stdlib (works even when dependencies are missing). Never prints .env values.

    python scripts/doctor.py

Exit code 0 = everything ready, 1 = something is missing (the fix is printed).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED = BASE_DIR / "data" / "processed"
ETM_DIR = BASE_DIR / "data" / "raw" / "etm"
GTFS_DIR = BASE_DIR / "data" / "raw" / "gtfs"

REQUIRED_PKGS = ["polars", "lightgbm", "fastapi", "uvicorn", "httpx"]

# Artifact -> the command that produces it
ARTIFACTS = {
    "routing_index.pkl": "python scripts/build_part1_index.py",
    "etm_stop_load.parquet": "python scripts/preprocess_etm_data.py",
    "crowd_lgbm_model.txt": "python scripts/train_part2_crowd_model.py",
    "model_meta.json": "python scripts/train_part2_crowd_model.py",
    "hist_crowd_lookup.parquet": "python scripts/preprocess_etm_data.py",
    "hist_slot_crowd.parquet": "python scripts/train_part2_crowd_model.py",
}


def main() -> int:
    print("=" * 70)
    print("DELHI CROWD PLANNER - DOCTOR (self-check)")
    print("=" * 70)
    problems: list[str] = []
    fix_cmds: list[str] = []

    # 1) Python version
    v = sys.version_info
    if (v.major, v.minor) < (3, 10):
        problems.append(f"Python {v.major}.{v.minor} is too old (need >= 3.10)")
    else:
        print(f"[OK]   Python {v.major}.{v.minor}.{v.micro}")

    # 2) Dependencies
    missing_pkgs = []
    for pkg in REQUIRED_PKGS:
        try:
            importlib.import_module(pkg)
        except ModuleNotFoundError:
            missing_pkgs.append(pkg)
    if missing_pkgs:
        problems.append(f"missing packages: {', '.join(missing_pkgs)}")
        fix_cmds.append("pip install -r requirements.txt   # (inside the activated venv!)")
    else:
        print(f"[OK]   dependencies installed ({', '.join(REQUIRED_PKGS)})")

    # 3) Raw data present
    etm_files = ([p for p in ETM_DIR.glob("*")
                  if p.is_file() and not p.name.startswith(".")]
                 if ETM_DIR.is_dir() else [])
    gtfs_files = [p for p in GTFS_DIR.glob("*") if p.is_file()] if GTFS_DIR.is_dir() else []
    if etm_files:
        print(f"[OK]   ETM raw data: {[p.name for p in etm_files]}")
    else:
        problems.append("no ETM e-ticketing file in data/raw/etm/")
        fix_cmds.append("drop your real ETM file into data/raw/etm/  "
                        "(columns: passenger_count, amount, vehicle_number, ticket_type_id, "
                        "end_stop_name, route, start_stop_name, created_at, is_ac)")

    gtfs_names = [p.name for p in gtfs_files]
    has_stops = any(n in gtfs_names for n in ("stops.txt", "stops.csv")) or any(
        p.suffix == ".zip" for p in gtfs_files
    )
    if gtfs_files and has_stops:
        print(f"[OK]   GTFS raw data: {gtfs_names}")
    else:
        problems.append("stops.txt (or a GTFS .zip) missing in data/raw/gtfs/")
        fix_cmds.append("drop the GTFS bundle (shapes.zip) contents into data/raw/gtfs/  "
                        "(stops.txt, routes.txt, trips.txt, stop_times.txt, shapes.txt)")

    # 4) Processed artifacts
    missing_art = {a: cmd for a, cmd in ARTIFACTS.items() if not (PROCESSED / a).exists()}
    for a in ARTIFACTS:
        if a not in missing_art:
            print(f"[OK]   artifact data/processed/{a}")
    for a, cmd in missing_art.items():
        problems.append(f"artifact data/processed/{a} missing")
        if cmd not in fix_cmds:
            fix_cmds.append(cmd)

    # 4b) Stale-artifact check: raw ETM file newer than the training table means
    # the last preprocessing run did NOT include it (usually: it failed).
    stop_load = PROCESSED / "etm_stop_load.parquet"
    if stop_load.exists() and etm_files:
        newest_raw = max(p.stat().st_mtime for p in etm_files)
        if newest_raw > stop_load.stat().st_mtime:
            problems.append(
                "your ETM file in data/raw/etm/ is NEWER than "
                "data/processed/etm_stop_load.parquet - the training table does "
                "NOT include it (the last preprocessing run likely failed)")
            fix_cmds.append("python scripts/preprocess_etm_data.py  "
                            "(fix any [ERROR] it prints, then re-run "
                            "scripts/train_part2_crowd_model.py)")

    # 5) .env key (presence only, never print the value)
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        print("[OK]   .env present (GTFS-RT live tracking enabled if key is valid)")
    else:
        print("[i]    no .env -> live tracking in fallback mode (fine, not an error)")

    # Verdict
    print("-" * 70)
    if not problems:
        print("VERDICT: ALL GOOD - start the app with:")
        print("    uvicorn app.main:app --reload   ->  http://127.0.0.1:8000")
        print("Then /health must show \"model_trained\": true and the UI header pill")
        print("reads 'Crowd engine trained'.")
        return 0

    print(f"VERDICT: {len(problems)} blocker(s) found:")
    for p in problems:
        print(f"  - {p}")
    print("\nRun these NOW, in this order, and watch for a red traceback:")
    seen = set()
    for c in fix_cmds:
        if c not in seen:
            seen.add(c)
            print(f"    {c}")
    print("\nIf any command prints an error/traceback, copy that output back to me.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
