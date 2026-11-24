#!/usr/bin/env python3
"""
STEP C — Part 1: Static GTFS Canonical Index Builder (Fast Route Data Fetch).

Reads the Static DTC / Bus GTFS files YOU dropped into data/raw/gtfs/ and
compresses them into a single in-memory lookup artifact:

    data/processed/routing_index.pkl

The pickle stores (all plain dicts -> O(1) lookups, < 50 ms per query):
  * stops               : stop_id -> {stop_name, lat, lon}
  * routes              : route_id -> {bus_number, route_name, route_type}
  * route_patterns      : (route_id, direction_id) -> ordered stop sequence with
                          cumulative `offset_min` from the origin terminal
                          (ONE canonical trip per direction = ~2,400 patterns
                          instead of millions of stop_times rows)
  * stop_to_routes      : stop_id -> [(route_id, direction_id, idx), ...]
  * headway_min         : (route_id, direction_id, hour) -> scheduled headway
                          (computed from ALL trips, not just the canonical one)
  * route_shapes        : (route_id, direction_id) -> [[lat, lon], ...] (optional)
  * name_to_stop_ids    : normalized stop name -> [stop_id, ...]  (ETM alignment)
  * route_number_to_ids : normalized bus number -> [route_id, ...] (ETM alignment)

Usage
-----
    python scripts/build_part1_index.py

GTFS files supported: stops.txt, routes.txt, trips.txt, stop_times.txt
(plus optional shapes.txt). They may be loose in data/raw/gtfs/ or inside a .zip.
"""

from __future__ import annotations

import csv as _csv
import pickle
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parents[1]
GTFS_DIR = BASE_DIR / "data" / "raw" / "gtfs"
OUT_PKL = BASE_DIR / "data" / "processed" / "routing_index.pkl"

REQUIRED_FILES = ["stops", "routes", "trips", "stop_times"]
OPTIONAL_FILES = ["shapes"]

# GTFS column fallbacks (feeds occasionally use slightly different spellings).
GTFS_COLUMN_FALLBACKS = {
    "stop_id": ["stop_id", "stopid"],
    "stop_name": ["stop_name", "stopname", "stop_nm"],
    "stop_lat": ["stop_lat", "stoplat", "latitude", "lat"],
    "stop_lon": ["stop_lon", "stoplon", "longitude", "lon", "lng"],
    "route_id": ["route_id", "routeid"],
    "route_short_name": ["route_short_name", "routeshortname", "short_name", "route_no", "routeno"],
    "route_long_name": ["route_long_name", "routelongname", "long_name", "route_name"],
    "route_type": ["route_type", "routetype"],
    "trip_id": ["trip_id", "tripid"],
    "direction_id": ["direction_id", "directionid", "direction"],
    "shape_id": ["shape_id", "shapeid"],
    "arrival_time": ["arrival_time", "arrivaltime"],
    "departure_time": ["departure_time", "departuretime"],
    "stop_sequence": ["stop_sequence", "stopsequence", "stop_seq"],
    "shape_pt_sequence": ["shape_pt_sequence", "shapeptsequence", "pt_sequence"],
    "shape_pt_lat": ["shape_pt_lat", "shapeptlat", "shape_lat"],
    "shape_pt_lon": ["shape_pt_lon", "shapeptlon", "shape_lon"],
}

ROUTE_TYPE_BUS = 3  # GTFS extended types 700..799 (bus) & 3 (bus); 0/1/2 = tram/metro/rail


def norm_header(h: str) -> str:
    s = str(h).strip().strip('"').strip("'").lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def norm_key(s) -> str:
    if s is None:
        return ""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def log(msg: str = "") -> None:
    print(msg, flush=True)


def parse_gtfs_time_to_sec(t) -> int | None:
    """GTFS allows hours > 23 for post-midnight trips (e.g. '25:10:00')."""
    if t is None:
        return None
    s = str(t).strip()
    if not s:
        return None
    parts = s.split(":")
    if len(parts) < 2:
        return None
    try:
        h = int(parts[0])
        m = int(parts[1])
        sec = int(float(parts[2])) if len(parts) > 2 else 0
        return h * 3600 + m * 60 + sec
    except ValueError:
        return None


def _sniff_sep(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            sample = fh.read(65536)
        if sample:
            return _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        pass
    return ","


def read_gtfs_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        separator=_sniff_sep(path),
        infer_schema=False,          # keep IDs like '0123' intact
        truncate_ragged_lines=True,
        encoding="utf8-lossy",
    )
    # Rename headers to canonical GTFS names where fallbacks match.
    rename = {}
    normalized = {norm_header(c): c for c in df.columns}
    for canon, candidates in GTFS_COLUMN_FALLBACKS.items():
        for cand in candidates:
            if cand in normalized and canon not in df.columns:
                rename[normalized[cand]] = canon
                break
    return df.rename(rename) if rename else df


def discover_gtfs_files() -> dict[str, Path]:
    """Locate the GTFS .txt/.csv files (loose or inside one .zip) in data/raw/gtfs/."""
    found: dict[str, Path] = {}
    if not GTFS_DIR.exists():
        return found
    zips = [p for p in GTFS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".zip"
            and not p.name.startswith(".")]
    loose = [p for p in GTFS_DIR.iterdir() if p.is_file()
             and p.suffix.lower() in {".txt", ".csv"} and not p.name.startswith(".")]

    for p in loose:
        stem = norm_header(p.stem)
        for name in REQUIRED_FILES + OPTIONAL_FILES:
            if stem == name and name not in found:
                found[name] = p

    if zips and len(found) < len(REQUIRED_FILES):
        try:
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(zips[0]) as zf:
                    for m in zf.namelist():
                        if m.startswith("__MACOSX") or Path(m).name.startswith("."):
                            continue
                        stem = norm_header(Path(m).stem)
                        for name in REQUIRED_FILES + OPTIONAL_FILES:
                            if stem == name and name not in found:
                                zf.extract(m, td)
                                found[name] = Path(td) / m
        except Exception as exc:
            log(f"[WARN] Could not read GTFS zip '{zips[0].name}': {exc}")
    return found


def build_index() -> int:
    t0 = time.perf_counter()
    (BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

    files = discover_gtfs_files()
    missing = [f for f in REQUIRED_FILES if f not in files]
    if missing:
        log("=" * 70)
        log("STATIC GTFS FILES NOT FOUND / INCOMPLETE.")
        log(f"Looking in: {GTFS_DIR}")
        log(f"Missing:    {', '.join(m + '.txt' for m in missing)}")
        log("Drop the unzipped DTC GTFS bundle there (stops.txt, routes.txt,")
        log("trips.txt, stop_times.txt) - or a single .zip containing them -")
        log("then re-run:  python scripts/build_part1_index.py")
        log("=" * 70)
        return 1

    log("Loading GTFS tables with Polars ...")
    stops = read_gtfs_csv(files["stops"])
    routes = read_gtfs_csv(files["routes"])
    trips = read_gtfs_csv(files["trips"])
    stop_times = read_gtfs_csv(files["stop_times"])
    log(f"  stops={stops.height:,}  routes={routes.height:,}  trips={trips.height:,}  "
        f"stop_times={stop_times.height:,}")

    # ---- Filter to bus routes (avoid metro/rail rows sneaking in) ----------
    if "route_type" in routes.columns:
        rt = routes.select(
            pl.col("route_type").cast(pl.Float64, strict=False).alias("_rt")
        )["_rt"].drop_nulls()
        if len(rt) and rt.is_in([3.0, 700.0, 701.0, 702.0]).any():
            before = routes.height
            routes = routes.filter(
                pl.col("route_type").cast(pl.Float64, strict=False)
                .is_in([3.0, 700.0, 701.0, 702.0])
            )
            trips = trips.join(
                routes.select("route_id"), on="route_id", how="inner"
            )
            log(f"  route_type filter (bus only): {before:,} -> {routes.height:,} routes")

    # ---- stops dict ----------------------------------------------------------
    stops_dict: dict[str, dict] = {}
    for row in stops.iter_rows(named=True):
        sid = str(row.get("stop_id") or "").strip()
        if not sid or sid in stops_dict:
            continue
        try:
            lat = float(row.get("stop_lat")) if row.get("stop_lat") else None
            lon = float(row.get("stop_lon")) if row.get("stop_lon") else None
        except (TypeError, ValueError):
            lat = lon = None
        stops_dict[sid] = {
            "stop_id": sid,
            "stop_name": str(row.get("stop_name") or sid).strip(),
            "lat": lat,
            "lon": lon,
        }

    # ---- routes dict ---------------------------------------------------------
    routes_dict: dict[str, dict] = {}
    route_number_to_ids: dict[str, list[str]] = {}
    for row in routes.iter_rows(named=True):
        rid = str(row.get("route_id") or "").strip()
        if not rid:
            continue
        bus_number = str(row.get("route_short_name") or "").strip() or rid
        routes_dict[rid] = {
            "route_id": rid,
            "bus_number": bus_number,
            "route_name": str(row.get("route_long_name") or "").strip(),
        }
        for key in {norm_key(rid), norm_key(bus_number)}:
            if key:
                route_number_to_ids.setdefault(key, [])
                if rid not in route_number_to_ids[key]:
                    route_number_to_ids[key].append(rid)

    # ---- trips: route_id / direction_id ---------------------------------------
    trips = trips.select(
        pl.col("trip_id").cast(pl.String).str.strip_chars(),
        pl.col("route_id").cast(pl.String).str.strip_chars(),
        *(
            [pl.col("direction_id").cast(pl.String).str.strip_chars()
             .replace_strict({"0": "0", "1": "1"}, default="0", return_dtype=pl.String)
             .alias("direction_id")]
            if "direction_id" in trips.columns
            else [pl.lit("0").alias("direction_id")]
        ),
        *([pl.col("shape_id").cast(pl.String).str.strip_chars()] if "shape_id" in trips.columns else []),
    ).filter((pl.col("trip_id") != "") & (pl.col("route_id") != ""))

    # ---- stop_times: canonical pattern extraction ------------------------------
    needed_cols = [c for c in ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]
                   if c in stop_times.columns]
    st = stop_times.select(
        pl.col("trip_id").cast(pl.String).str.strip_chars(),
        *([pl.col("arrival_time").cast(pl.String)] if "arrival_time" in stop_times.columns else [pl.lit("").alias("arrival_time")]),
        *([pl.col("departure_time").cast(pl.String)] if "departure_time" in stop_times.columns else [pl.lit("").alias("departure_time")]),
        pl.col("stop_id").cast(pl.String).str.strip_chars(),
        pl.col("stop_sequence").cast(pl.Float64, strict=False).cast(pl.Int32),
    )
    st = st.filter(pl.col("stop_id") != "")
    st_merged = st.join(
        trips.select(["trip_id", "route_id", "direction_id", "shape_id"] if "shape_id" in trips.columns
                     else ["trip_id", "route_id", "direction_id"]),
        on="trip_id", how="inner",
    )

    log("Canonicalizing routes (longest trip per route & direction) ...")
    trip_lengths = st_merged.group_by(["route_id", "direction_id", "trip_id"]).agg(
        pl.len().alias("num_stops")
    )
    canonical = (
        trip_lengths.sort("num_stops", descending=True)
        .group_by(["route_id", "direction_id"])
        .first()
        .select(["route_id", "direction_id", "trip_id", "num_stops"])
    )
    canonical_st = (
        st_merged.join(canonical, on=["route_id", "direction_id", "trip_id"], how="inner")
        .sort(["route_id", "direction_id", "stop_sequence"])
    )

    route_patterns: dict[tuple[str, int], list[dict]] = {}
    stop_to_routes: dict[str, list[tuple[str, int, int]]] = {}
    canonical_shapes: dict[tuple[str, int], str | None] = {}

    for row in canonical_st.iter_rows(named=True):
        key = (row["route_id"], int(row["direction_id"]))
        t_sec = parse_gtfs_time_to_sec(row.get("arrival_time"))
        if t_sec is None:
            t_sec = parse_gtfs_time_to_sec(row.get("departure_time"))
        route_patterns.setdefault(key, []).append({
            "stop_id": row["stop_id"],
            "stop_sequence": int(row["stop_sequence"]),
            "raw_sec": t_sec,
        })
        if "shape_id" in row and row.get("shape_id"):
            canonical_shapes.setdefault(key, row["shape_id"])

    # Interpolate missing raw_sec inside each pattern, then compute offsets.
    dropped_patterns = 0
    for key, seq in route_patterns.items():
        n = len(seq)
        known = [(i, s["raw_sec"]) for i, s in enumerate(seq) if s["raw_sec"] is not None]
        if len(known) < 2:
            dropped_patterns += 1
        if known:
            # Forward/backward fill then linear interpolation between anchors.
            for i, s in enumerate(seq):
                if s["raw_sec"] is not None:
                    continue
                prev_i = max((k for k, _ in known if k < i), default=None)
                next_i = min((k for k, _ in known if k > i), default=None)
                if prev_i is not None and next_i is not None:
                    t0v, t1v = seq[prev_i]["raw_sec"], seq[next_i]["raw_sec"]
                    frac = (i - prev_i) / max(1, (next_i - prev_i))
                    s["raw_sec"] = t0v + frac * (t1v - t0v)
                elif prev_i is not None:
                    s["raw_sec"] = seq[prev_i]["raw_sec"]
                elif next_i is not None:
                    s["raw_sec"] = seq[next_i]["raw_sec"]
        base = next((s["raw_sec"] for s in seq if s["raw_sec"] is not None), 0)
        for idx, item in enumerate(seq):
            raw = item["raw_sec"] if item["raw_sec"] is not None else base
            item["offset_min"] = max(0.0, round((raw - base) / 60.0, 2))
            item["idx"] = idx
            del item["raw_sec"]
            stop_to_routes.setdefault(item["stop_id"], [])
            entry = (key[0], key[1], idx)
            if entry not in stop_to_routes[item["stop_id"]]:
                stop_to_routes[item["stop_id"]].append(entry)

    # ---- Scheduled headways from ALL trips (first departure per trip) ---------
    log("Computing scheduled headways per (route, direction, hour) ...")
    first_dep = st.filter(pl.col("departure_time") != "").group_by("trip_id").agg(
        pl.col("departure_time").first().alias("dep")
    )
    first_dep = first_dep.with_columns(
        pl.col("dep").map_elements(parse_gtfs_time_to_sec, return_dtype=pl.Int64, skip_nulls=True)
        .alias("dep_sec")
    ).filter(pl.col("dep_sec").is_not_null())
    trip_head = (
        first_dep.join(trips.select(["trip_id", "route_id", "direction_id"]),
                       on="trip_id", how="inner")
        .with_columns(((pl.col("dep_sec") // 3600) % 24).cast(pl.Int32).alias("hour"))
        .group_by(["route_id", "direction_id", "hour"])
        .agg(pl.len().alias("n_trips"))
    )
    headway_min: dict[tuple[str, int, int], float] = {}
    for row in trip_head.iter_rows(named=True):
        n = max(1, int(row["n_trips"]))
        hw = min(45.0, max(3.0, round(60.0 / n, 1)))
        headway_min[(row["route_id"], int(row["direction_id"]), int(row["hour"]))] = hw

    # ---- Optional shapes.txt polylines -----------------------------------------
    route_shapes: dict[tuple[str, int], list[list[float]]] = {}
    if "shapes" in files:
        log("Loading shapes.txt for route polylines ...")
        shapes = read_gtfs_csv(files["shapes"])
        if {"shape_id", "shape_pt_lat", "shape_pt_lon"} <= set(shapes.columns):
            shapes = shapes.select(
                pl.col("shape_id").cast(pl.String).str.strip_chars(),
                pl.col("shape_pt_lat").cast(pl.Float64, strict=False),
                pl.col("shape_pt_lon").cast(pl.Float64, strict=False),
                *([pl.col("shape_pt_sequence").cast(pl.Int32, strict=False)]
                  if "shape_pt_sequence" in shapes.columns else [pl.lit(0).alias("shape_pt_sequence")]),
            ).drop_nulls(["shape_pt_lat", "shape_pt_lon"]).sort(["shape_id", "shape_pt_sequence"])
            by_shape: dict[str, list[list[float]]] = {}
            for row in shapes.iter_rows(named=True):
                by_shape.setdefault(row["shape_id"], []).append(
                    [round(row["shape_pt_lat"], 6), round(row["shape_pt_lon"], 6)]
                )
            for key, sid in canonical_shapes.items():
                pts = by_shape.get(sid or "")
                if pts and len(pts) >= 2:
                    step = max(1, len(pts) // 1200)   # cap polyline size
                    route_shapes[key] = pts[::step]

    # ---- Normalized name lookups (used by ETM preprocessing) --------------------
    name_to_stop_ids: dict[str, list[str]] = {}
    for sid, meta in stops_dict.items():
        key = norm_key(meta["stop_name"])
        if key:
            name_to_stop_ids.setdefault(key, [])
            if sid not in name_to_stop_ids[key]:
                name_to_stop_ids[key].append(sid)

    # Drop patterns whose route vanished via the route_type filter.
    route_patterns = {k: v for k, v in route_patterns.items() if k[0] in routes_dict}

    artifact = {
        "stops": stops_dict,
        "routes": routes_dict,
        "route_patterns": route_patterns,
        "stop_to_routes": stop_to_routes,
        "headway_min": headway_min,
        "route_shapes": route_shapes,
        "name_to_stop_ids": name_to_stop_ids,
        "route_number_to_ids": route_number_to_ids,
        "meta": {
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_stops": len(stops_dict),
            "num_routes": len(routes_dict),
            "num_patterns": len(route_patterns),
            "num_stop_times_rows": stop_times.height,
            "patterns_without_times": dropped_patterns,
        },
    }

    with open(OUT_PKL, "wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    dt = time.perf_counter() - t0
    log("")
    log("=" * 70)
    log("PART 1 ROUTING INDEX BUILT")
    log("=" * 70)
    m = artifact["meta"]
    log(f"  Stops indexed          : {m['num_stops']:,}")
    log(f"  Bus routes             : {m['num_routes']:,}")
    log(f"  Directional patterns   : {m['num_patterns']:,} "
        f"({m['patterns_without_times']} without usable times)")
    log(f"  Headway entries        : {len(headway_min):,}")
    log(f"  Route shape polylines  : {len(route_shapes):,}")
    log(f"  Build time             : {dt:.1f}s | saved to data/processed/routing_index.pkl")
    log("  Next: drop your ETM file(s) in data/raw/etm/ and run")
    log("        python scripts/preprocess_etm_data.py  (re-run if already done,")
    log("        so stop-level net load gets reconstructed).")
    return 0


if __name__ == "__main__":
    sys.exit(build_index())
