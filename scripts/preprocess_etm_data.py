#!/usr/bin/env python3
"""
STEP A — ETM / e-ticketing preprocessing (Crowdedness Engine data foundation).

Reads the e-ticketing file(s) YOU dropped into  data/raw/etm/  (CSV / Parquet / ZIP),
auto-detects the column layout (no code edits needed), reconstructs the NET on-board
passenger load (Boardings - Alightings) and writes the training table:

    data/processed/etm_stop_load.parquet   -> consumed by scripts/train_part2_crowd_model.py
    data/processed/etm_clean.parquet       -> ticket-level debug copy
    data/processed/preprocess_report.json  -> mapping & data-quality report

CONFIRMED production ETM schema (10-lakh-row portal export, auto-detected, no edits):
    passenger_count, amount, vehicle_number, ticket_type_id, end_stop_name,
    route, start_stop_name, created_at, is_ac
  * passenger_count -> pax per ticket row (sanity-clamped 1..12)
  * amount          -> fare (rupees; powers the UI insights panel)
  * vehicle_number  -> per-vehicle ticket/vehicle aggregates (UI panel)
  * ticket_type_id  -> ticket-type mix (UI panel)
  * end/start_stop_name -> stop-name alignment to GTFS (fuzzy tiers below)
  * route           -> route number -> GTFS route_id (number/prefix match)
  * created_at      -> service date + 15-min slot (rush-hour learning)
  * is_ac           -> AC / Non-AC split (UI panel)
  * direction       -> NOT in the file: inferred from the OD pair against the
                       GTFS directional patterns (dirs_for_od), fallback 0.

Usage
-----
    python scripts/preprocess_etm_data.py            # full run
    python scripts/preprocess_etm_data.py --peek     # show detected column map only, no output files

Documented assumptions (quote these in your technical note)
-----------------------------------------------------------
* Case A (destination stop / fare stage present): riders are subtracted exactly at
  their alighting stop index:  Load(k) = Sum_{i<=k} Boardings_i - Sum_{j<=k} Alightings_j
* Case B (alighting missing / invalid): calibrated stop-decay survival curve —
  exponential with mean trip length ~= 35% of the route's stop span (clamped to
  4..30 stops), matching typical Delhi bus ride lengths of ~6-9 km.
* Crowd normalization capacity: 75 passengers (DTC practical capacity:
  ~40 seated + ~35 standing).
* DEGRADED mode: if data/processed/routing_index.pkl does not exist yet (Part 1 not
  built), the script still runs so you can train immediately, but stop-order is
  unknown -> load proxy = raw boardings per stop (flag column `aligned = 0`).
  Re-run this script AFTER `python scripts/build_part1_index.py` for full
  stop-level net-load reconstruction.
"""

from __future__ import annotations

import argparse
import csv as _csv
import difflib
import io
import json
import math
import pickle
import re
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import polars as pl

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — paths & transit constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]      # project root (delhi-crowd-planner/)
ETM_DIR = BASE_DIR / "data" / "raw" / "etm"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ROUTING_INDEX_PKL = PROCESSED_DIR / "routing_index.pkl"

BUS_CAPACITY = 75.0          # DTC practical capacity (~40 seated + ~35 standing)
DECAY_MEAN_FRACTION = 0.35   # Case B: mean trip ~= 35% of route stop span
DECAY_LAMBDA_MIN, DECAY_LAMBDA_MAX = 4.0, 30.0
PAX_MIN, PAX_MAX = 1, 12     # sanity clamp on tickets-per-row (kills typos)
HATCH_RADIUS_KM = 0.25       # Haversine fallback radius for coordinate matching

# ---------------------------------------------------------------------------
# 2. COLUMN_MAP — canonical field -> candidate raw header names (priority order).
#
#    HOW TO CUSTOMIZE: if your ETM file uses a header the detector can't find,
#    simply add that exact header (lowercase) to the relevant list below, or put
#    it in USER_OVERRIDES just underneath — no other code change is required.
# ---------------------------------------------------------------------------
COLUMN_MAP: dict[str, list[str]] = {
    "route_id": [
        "route_id", "route_no", "route_number", "route_code", "routeid", "route",
        "route_name", "routename", "routeno", "routecode", "bus_route", "bus_route_no",
        "service_no", "serviceno", "route_nm", "routeno_", "line_no",
    ],
    "direction_id": [
        "direction_id", "direction", "dir", "dir_id", "trip_direction",
        "route_direction", "updown", "up_down", "updown_flag", "direction_nm",
    ],
    "timestamp": [
        "timestamp", "ticket_time", "issue_time", "issued_at", "ticket_datetime",
        "datetime", "date_time", "travel_datetime", "transaction_time",
        "transaction_datetime", "txn_datetime", "txn_time", "time_stamp",
        "ticket_issued_datetime", "created_at", "booking_time", "trip_start_time",
    ],
    "date": [
        "date", "trip_date", "service_date", "ticket_date", "journey_date",
        "travel_date", "operation_date", "txn_date", "transaction_date",
        "issued_date", "calendar_date", "ticketdate",
    ],
    "time": [
        "time", "ticket_time", "issue_time", "travel_time", "departure_time",
        "trip_time", "transaction_time", "txn_time", "issued_time", "time_of_day",
    ],
    "boarding_stop": [
        "boarding_stop", "from_stop", "source_stage", "boarding_stop_name",
        "boarding_point", "boarded_at", "boarding", "from_stage", "source_stop",
        "boarding_stop_id", "from_stop_id", "from_stopname", "start_stage",
        "boarding_station", "embarking_stop", "boarding_stage_name", "origin_stop",
        "origin", "source", "start_stop", "start_stop_name", "start_stopname",
        "board_stop", "from", "boarding_stage",
    ],
    "alighting_stop": [
        "alighting_stop", "to_stop", "dest_stage", "destination_stage",
        "alighting_stop_name", "destination_stop", "drop_point", "to_stage",
        "alighting_point", "destination_stop_id", "to_stop_id", "to_stopname",
        "destination", "end_stage", "alighting_station", "drop_stop",
        "dropoff_stop", "alight_stop", "alighting", "dest_stop",
        "destination_stop_name", "end_stop", "end_stop_name", "end_stopname",
        "destination_stage_name", "alighting_stage",
    ],
    "passenger_count": [
        "passenger_count", "tickets", "ticket_count", "no_of_tickets",
        "num_tickets", "no_tickets", "passengers", "pax", "qty", "quantity",
        "no_of_passengers", "passenger_qty", "tickets_count", "total_tickets",
        "no_of_pax", "pax_count", "ticket_qty",
    ],
    # Optional: numeric fare stages (used when stop names are absent but the
    # boarding stage is also given, so the offset is known).
    "boarding_stage_no": ["boarding_stage_no", "from_stage_no", "board_stage", "from_stage_number"],
    "alighting_stage_no": ["alighting_stage_no", "to_stage_no", "dest_stage_no", "destination_stage_no", "drop_stage"],
    # Optional: ticket coordinates (enable Haversine nearest-stop alignment).
    "boarding_lat": ["boarding_lat", "boarding_stop_lat", "from_lat", "source_lat", "boarding_latitude"],
    "boarding_lon": ["boarding_lon", "boarding_stop_lon", "from_lon", "source_lon", "boarding_longitude"],
    "alighting_lat": ["alighting_lat", "alighting_stop_lat", "to_lat", "dest_lat", "alighting_latitude"],
    "alighting_lon": ["alighting_lon", "alighting_stop_lon", "to_lon", "dest_lon", "alighting_longitude"],
}

# Mapping priority: a raw header matching several candidates goes to the
# highest-priority canonical field listed here.
CANONICAL_PRIORITY = [
    "timestamp", "date", "time", "route_id", "direction_id",
    "boarding_stop", "alighting_stop", "passenger_count",
    "boarding_stage_no", "alighting_stage_no",
    "boarding_lat", "boarding_lon", "alighting_lat", "alighting_lon",
]

# Optional manual override: {"canonical_field": "exact_raw_header_in_your_csv"}
USER_OVERRIDES: dict[str, str] = {}

# Keyword pairs for the fuzzy fallback matcher: (must_contain_any, and_ideally_any)
FUZZY_KEYWORDS = {
    "boarding_stop": (["board", "from", "source", "origin", "start", "embark", "onward", "pickup"], ["stop", "stage", "point", "station", "name", "place"]),
    "alighting_stop": (["alight", "alighting", "to", "dest", "destination", "drop", "off", "exit", "end", "down"], ["stop", "stage", "point", "station", "name", "place"]),
    "route_id": (["route", "line", "service"], []),
    "direction_id": (["direction", "dir", "updown", "up_down"], []),
    "passenger_count": (["ticket", "pax", "passenger", "qty", "quantity"], ["count", "no", "num", "number", "qty", "cnt"]),
    "boarding_stage_no": (["board", "from"], ["stage"]),
    "alighting_stage_no": (["alight", "to", "dest", "destination", "drop"], ["stage"]),
}

REQUIRED_FOR_RUN = ["route_id", "boarding_stop"]   # timestamp OR date also required

# Headers that must NEVER be assigned a canonical role by the fuzzy fallback
# (pass 2). These are fare / vehicle / metadata columns — e.g. the real portal
# export's "passenger_amount" is money in rupees, NOT a passenger count; without
# this guard the fuzzy matcher would read fare values as pax and massively
# inflate loads. Exact alias matches (pass 1) and USER_OVERRIDES still win, so
# this only protects otherwise-unmatched leftover headers.
IGNORE_PATTERNS = (
    "amount", "fare", "price", "amt", "rupee", "inr", "cost", "revenue",
    "value", "type", "vehicle", "is_ac", "_ac", "class", "category",
    "gender", "age", "card", "payment", "conductor", "device", "receipt",
    "ticket_no", "txn", "transaction",
)

# ETM commerce columns: the crowd model ignores them, but the web UI's
# e-ticketing insights panel (fares, ticket types, vehicles, AC split) is
# built from them. Exact aliases; a header already used as a canonical role
# is never double-captured. Missing columns simply skip their insight section.
EXTRA_COL_MAP: dict[str, list[str]] = {
    "fare_amt": ["passenger_amount", "amount", "fare", "fare_amount",
                 "ticket_amount", "total_amount", "amt", "ticket_fare"],
    "tkt_type": ["ticket_type", "ticket_type_id", "passenger_type", "pass_type",
                 "tickettype"],
    "vehicle_no": ["vehicle_number", "vehicle_no", "vehicleno", "bus_number",
                   "bus_no", "vehicle"],
    "ac_flag": ["is_ac", "ac_flag", "air_conditioned", "ac"],
}

# Compact dtypes applied at capture time — keeps the wide pipeline joins cheap.
EXTRA_CONVERTERS: dict[str, object] = {
    "fare_amt": lambda c: pl.col(c).cast(pl.Float64, strict=False),
    "ac_flag": lambda c: pl.col(c).cast(pl.String).str.strip_chars().str.to_lowercase()
                          .is_in(["1", "true", "t", "yes", "y", "ac", "a/c"]).cast(pl.Int8),
    "tkt_type": lambda c: pl.col(c).cast(pl.String).str.strip_chars().str.to_lowercase(),
    "vehicle_no": lambda c: pl.col(c).cast(pl.String).str.strip_chars(),
}


# ---------------------------------------------------------------------------
# 3. Small helpers
# ---------------------------------------------------------------------------
def norm_header(h: str) -> str:
    """Normalize a raw header: lowercase, punctuation -> single underscore."""
    s = str(h).strip().strip('"').strip("'").lower()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


def norm_key(s) -> str:
    """Aggressive key normalization for route numbers / stop names (alnum only)."""
    if s is None:
        return ""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


# Delhi bus-stop spelling variants: ETM exports abbreviate ('Kalindi Kunj Bdr',
# 'AIIMS Stn', 'Lajpat Ngar') while GTFS feeds usually spell the words out.
# Applied token-by-token BEFORE alnum squashing; expansion only, never removal.
STOP_ABBREVIATIONS = {
    "bdr": "border", "brdr": "border", "brd": "border",
    "stn": "station", "tpt": "terminal", "term": "terminal", "trm": "terminal",
    "jnc": "junction", "junc": "junction",
    "govt": "government", "hosp": "hospital", "clny": "colony",
    "instt": "institute", "univ": "university", "mkt": "market",
    "appt": "apartment", "xing": "crossing", "extn": "extension",
    "sec": "sector", "vihr": "vihar", "ngar": "nagar",
}


def expand_stop_abbreviations(raw_stop) -> str:
    """'Kalindi Kunj Bdr' -> 'kalindikunjborder' (norm_key-style string)."""
    if raw_stop is None:
        return ""
    tokens = [t.lower() for t in re.split(r"[^A-Za-z0-9]+", str(raw_stop)) if t]
    tokens = [STOP_ABBREVIATIONS.get(t, t) for t in tokens]
    return "".join(tokens)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 4. File discovery & readers (CSV / Parquet / ZIP auto-discovery)
# ---------------------------------------------------------------------------
def discover_etm_files() -> list[Path]:
    """Find .csv / .parquet / .zip files the user dropped into data/raw/etm/."""
    if not ETM_DIR.exists():
        return []
    files = [
        p for p in sorted(ETM_DIR.iterdir())
        if p.is_file()
        and not p.name.startswith((".", "_"))
        and p.suffix.lower() in {".csv", ".parquet", ".zip"}
    ]
    return files


def _read_csv_robust(path: Path) -> pl.DataFrame:
    """Read a CSV no matter how the portal exported it.

    Handles: utf-8 / utf-8-BOM, utf-16 LE & BE (BOM-detected — common for
    Windows exports), replacement-char decode as a last resort; CRLF and
    bare-CR line endings (a bare-CR file parses as one header-only line ->
    0 rows, which this normalizes away); NUL padding from wrong codepages;
    junk preamble/report-title rows above the real header. All columns stay
    String so messy ID columns (e.g. '0123') are never mangled by inference.
    """
    raw_bytes = path.read_bytes()

    # 1) Decode with a BOM-aware fallback chain.
    text: str | None = None
    for enc in ("utf-8-sig", "utf-16", "replace"):
        if enc == "utf-16" and raw_bytes[:2] not in (b"\xff\xfe", b"\xfe\xff"):
            continue  # utf-16 only when a BOM actually says so
        try:
            text = raw_bytes.decode(enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if text is None:
        text = raw_bytes.decode("latin-1")  # never fails

    # 2) Clean: kill NUL padding, normalize ALL line endings to \n
    #    (bare \r exports otherwise parse as a single header-only line).
    text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        raise RuntimeError(f"{path.name} contains no readable text (0 bytes usable).")

    # 3) Skip preamble rows: find the first line that looks like an ETM header
    #    (>= 3 keyword-group hits — data rows can't hit 3 different groups).
    groups = (
        ("route", "line", "service"),
        ("stop", "stage", "point", "origin", "dest", "source"),
        ("ticket", "passenger", "pax", "amount", "vehicle", "fare"),
        ("created", "time", "date", "timestamp"),
    )
    skip = 0
    for i, line in enumerate(text.split("\n")[:50]):
        low = line.lower()
        if sum(any(k in low for k in g) for g in groups) >= 3:
            skip = i
            break

    # 4) Sniff the delimiter from the header region.
    sep = ","
    try:
        sample = "\n".join(text.split("\n")[skip:skip + 20])
        if sample.strip():
            sep = _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        pass  # keep default comma

    df = pl.read_csv(
        io.BytesIO(text.encode("utf-8")),
        separator=sep,
        skip_rows=skip,
        infer_schema=False,          # everything as String -> safe
        truncate_ragged_lines=True,  # tolerate broken rows
    )
    # 5) Defensive: strip stray NUL/CR left in column names by bad codepages.
    df = df.rename({c: c.replace("\x00", "").replace("\r", "").strip() for c in df.columns})
    if df.height == 0:
        first = text.split("\n")[0][:200]
        raise RuntimeError(
            f"{path.name} parsed as a HEADER-ONLY file (0 data rows). Either the "
            f"export is empty, or its rows use an unsupported layout. First line: {first!r}"
        )
    return df


NULLISH_STRINGS = {"null", "none", "n/a", "na", "-", "--", ""}


def _nullify_nullish_strings(df: pl.DataFrame) -> pl.DataFrame:
    """Turn literal 'NULL' / 'N/A' / empty placeholder strings into real nulls in
    every String column (portal exports write the word NULL into vehicle_number
    and friends, which would otherwise surface as a bus named 'NULL')."""
    str_cols = [c for c, dt in df.schema.items() if dt == pl.String]
    if not str_cols:
        return df
    return df.with_columns([
        pl.when(pl.col(c).str.strip_chars().str.to_lowercase().is_in(list(NULLISH_STRINGS)))
        .then(None)
        .otherwise(pl.col(c).str.strip_chars())
        .alias(c)
        for c in str_cols
    ])


def read_table(path: Path) -> pl.DataFrame:
    """Read one ETM file (.csv / .parquet / .zip with csv or parquet inside)."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _nullify_nullish_strings(pl.read_parquet(path))
    if suffix == ".csv":
        return _nullify_nullish_strings(_read_csv_robust(path))
    if suffix == ".zip":
        out_frames = []
        with tempfile.TemporaryDirectory() as td:
            with zipfile.ZipFile(path) as zf:
                members = [
                    m for m in zf.namelist()
                    if not m.startswith("__MACOSX")
                    and Path(m).suffix.lower() in {".csv", ".parquet"}
                    and not Path(m).name.startswith(".")
                ]
                if not members:
                    raise RuntimeError(f"ZIP {path.name} contains no .csv/.parquet member")
                for m in members:
                    zf.extract(m, td)
                    inner = Path(td) / m
                out_frames.append(
                    _nullify_nullish_strings(
                        pl.read_parquet(inner) if inner.suffix.lower() == ".parquet"
                        else _read_csv_robust(inner)
                    )
                )
        return pl.concat(out_frames, how="diagonal_relaxed")
    raise ValueError(f"Unsupported ETM file type: {path.name}")


def normalize_headers(df: pl.DataFrame) -> pl.DataFrame:
    """Rename raw columns to their normalized versions (lower/snake)."""
    rename = {}
    for c in df.columns:
        n = norm_header(c)
        if n and n != c:
            rename[c] = n
    return df.rename(rename) if rename else df


# ---------------------------------------------------------------------------
# 5. Column-role detection (exact candidates first, then fuzzy keywords)
# ---------------------------------------------------------------------------
def detect_column_roles(headers: list[str]) -> dict[str, str]:
    """Return {canonical_field: normalized_raw_header} for this file's header."""
    assigned: dict[str, str] = {}
    used: set[str] = set()

    # Manual overrides always win.
    for canon, raw in USER_OVERRIDES.items():
        n = norm_header(raw)
        if n in headers:
            assigned[canon] = n
            used.add(n)

    # Pass 1 — exact candidate lists, in canonical priority order.
    for canon in CANONICAL_PRIORITY:
        if canon in assigned:
            continue
        for cand in COLUMN_MAP.get(canon, []):
            if cand in headers and cand not in used:
                assigned[canon] = cand
                used.add(cand)
                break

    # Pass 2 — fuzzy keyword matching on the remaining headers.
    for canon in CANONICAL_PRIORITY:
        if canon in assigned:
            continue
        must, ideally = FUZZY_KEYWORDS.get(canon, ([], []))
        best, best_score = None, 0.0
        for h in headers:
            if h in used:
                continue
            if any(p in h for p in IGNORE_PATTERNS):
                continue  # fare/metadata column — never a role (see IGNORE_PATTERNS)
            if must and not any(k in h for k in must):
                continue
            score = 1.0
            if ideally and not any(k in h for k in ideally):
                score = 0.5  # keyword hit but no stop/stage-ish qualifier
            # avoid stealing numeric-ish junk like 'to_date' for alighting
            if score > best_score:
                best, best_score = h, score
        if best is not None and best_score >= 0.5:
            assigned[canon] = best
            used.add(best)

    return assigned


def coerce_pax(df: pl.DataFrame, pax_col: str | None) -> pl.DataFrame:
    """Ticket counts -> sane passenger numbers; missing column => 1 pax/row."""
    if pax_col is not None:
        df = df.with_columns(
            pl.col(pax_col).cast(pl.Float64, strict=False).alias("_pax_raw")
        )
        df = df.with_columns(
            pl.col("_pax_raw").clip(PAX_MIN, PAX_MAX).fill_null(1.0).alias("pax")
        ).drop("_pax_raw")
    else:
        df = df.with_columns(pl.lit(1.0).alias("pax"))
    return df


# ---------------------------------------------------------------------------
# 6. Timestamp parsing (robust chain: dtype-native -> inference -> formats -> epoch)
# ---------------------------------------------------------------------------
EXPLICIT_FORMATS = [
    # offset-aware ISO first (portal exports like '2026-10-04 15:37:05.383774+05:30').
    # If the running polars build lacks %:z support the try/except skips these
    # and the numeric-offset strip strategies below handle them instead.
    "%Y-%m-%d %H:%M:%S%.f%:z", "%Y-%m-%d %H:%M:%S%:z", "%Y-%m-%dT%H:%M:%S%.f%:z",
    "%Y-%m-%d %H:%M:%S%.f%z", "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%.fZ", "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S%.f",
    "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
    "%d-%m-%Y %I:%M:%S %p", "%d-%m-%Y %I:%M %p", "%d/%m/%Y %I:%M %p",
    "%d/%m/%Y %I:%M:%S %p",
    "%d/%m/%Y, %H:%M:%S", "%d/%m/%Y, %H:%M", "%d-%m-%Y, %H:%M:%S",
    "%m/%d/%Y %H:%M", "%Y/%m/%d %H:%M:%S", "%Y%m%d %H%M%S", "%Y%m%d%H%M%S",
    "%d-%b-%Y %H:%M:%S", "%d-%b-%Y %H:%M", "%d %b %Y %H:%M:%S", "%d %b %Y %H:%M",
    "%d-%B-%Y %H:%M:%S", "%d-%B-%Y %H:%M",
    "%Y-%m-%d %I:%M:%S %p", "%Y-%m-%d %I:%M %p",
]


def _epoch_unit(df: pl.DataFrame, col: str) -> str | None:
    """Guess epoch unit from magnitude (s / ms / us). Small numeric columns in
    the 20k-80k range are Excel serial dates, NOT epoch — return None for them."""
    try:
        nums = df.select(pl.col(col).cast(pl.Float64, strict=False)).drop_nulls()
        if nums.height == 0:
            return None
        med = abs(nums[col].median() or 0.0)
    except Exception:
        return None
    if med == 0:
        return None
    if med > 1e14:
        return "us"
    if med > 1e11:
        return "ms"
    if med > 1e8:
        return "s"
    return None  # 20k-80k = Excel serials; smaller = junk IDs, not timestamps


def _string_to_datetime(df: pl.DataFrame, col: str) -> tuple[pl.Expr | None, str]:
    """Try several parse strategies for a String datetime column, most specific
    first. Numeric tz offsets ('+05:30') are stripped before inference because
    portal exports stamp every row with the local offset."""
    # Strategy 1: polars inference on the raw column
    try:
        parsed = df.select(pl.col(col).str.strip_chars().str.to_datetime(strict=False))
        if parsed.height and parsed[col].null_count() < parsed.height:
            return pl.col(col).str.strip_chars().str.to_datetime(strict=False), "inferred"
    except Exception:
        pass
    # Strategy 2: inference after stripping tz suffixes — textual abbreviations
    # ('...05:14:04 IST') AND numeric offsets ('...15:37:05.383774+05:30',
    # '...14:02:08+0530', '...12:14:25Z'). Stripping a fixed numeric offset is
    # safe: Delhi portal exports are already local IST wall-clock times.
    clean_chain = (
        pl.col(col).str.strip_chars()
        .str.replace(r"\s*(IST|GMT|UTC|[+-]\d{2}:\d{2}|[+-]\d{4}|Z)\s*$", "")
    )
    try:
        parsed = df.select(clean_chain.str.to_datetime(strict=False).alias("_ts_probe"))
        if parsed.height and parsed["_ts_probe"].null_count() < parsed.height:
            return clean_chain.str.to_datetime(strict=False), "inferred+tz-strip"
    except Exception:
        pass
    # Strategy 3: explicit formats, raw column (offset-aware formats lead the list)
    for fmt in EXPLICIT_FORMATS:
        try:
            expr = pl.col(col).str.strip_chars().str.to_datetime(format=fmt, strict=False)
            parsed = df.select(expr)
            if parsed.height and parsed[col].null_count() < parsed.height:
                return expr, fmt
        except Exception:
            continue
    # Strategy 4: explicit formats after the tz-suffix strip
    for fmt in EXPLICIT_FORMATS:
        try:
            expr = clean_chain.str.to_datetime(format=fmt, strict=False)
            parsed = df.select(expr.alias("_ts_probe"))
            if parsed.height and parsed["_ts_probe"].null_count() < parsed.height:
                return expr, f"tz-strip {fmt}"
        except Exception:
            continue
    # Strategy 5: Excel serial day numbers (45926.21 = 2025-10-16 05:02:24).
    # MUST run before the epoch strategy: both are bare numeric columns.
    try:
        nums = df.select(pl.col(col).cast(pl.Float64, strict=False)).drop_nulls()
        if nums.height:
            med = abs(nums[col].median() or 0.0)
            if 20000.0 <= med <= 80000.0:
                expr = (
                    (pl.col(col).cast(pl.Float64, strict=False) - 25569.0) * 86400.0
                ).cast(pl.Int64).pipe(lambda c: pl.from_epoch(c, time_unit="s"))
                return expr, "excel-serial"
    except Exception:
        pass
    # Strategy 5: pure-numeric epoch stored as string (only large magnitudes).
    unit = _epoch_unit(df.with_columns(pl.col(col).cast(pl.Float64, strict=False).alias(col)), col)
    if unit:
        return pl.col(col).cast(pl.Float64, strict=False).pipe(
            lambda c: pl.from_epoch(c.cast(pl.Int64), time_unit=unit)
        ), f"epoch-{unit}"
    return None, "unparsed"


def _ensure_naive(dt_expr_col: str, df: pl.DataFrame) -> pl.DataFrame:
    """If parsed column is timezone-aware, convert to IST and drop the tz."""
    dtype = df.schema.get(dt_expr_col)
    if dtype is not None and getattr(dtype, "time_zone", None):
        df = df.with_columns(
            pl.col(dt_expr_col)
            .dt.convert_time_zone("Asia/Kolkata")
            .dt.replace_time_zone(None)
        )
    return df


def parse_trip_times(df: pl.DataFrame, roles: dict[str, str]) -> tuple[pl.DataFrame, dict]:
    """Produce service_date / slot_15m / day_of_week / is_weekend columns."""
    stats = {"time_mode": "none", "dropped_bad_timestamp": 0, "time_missing_rows": 0}
    df = df.with_columns(pl.lit(None, dtype=pl.Datetime).alias("_dt"))

    ts_col = roles.get("timestamp")
    date_col, time_col = roles.get("date"), roles.get("time")

    if ts_col is not None:
        dtype = df.schema.get(ts_col)
        if isinstance(dtype, pl.Datetime):
            df = _ensure_naive(ts_col, df)
            df = df.with_columns(pl.col(ts_col).alias("_dt"))
            stats["time_mode"] = "native-datetime"
        elif isinstance(dtype, pl.Date):
            # Date-only parquet column: no time-of-day signal (nulls dropped later).
            df = df.with_columns(pl.col(ts_col).cast(pl.Datetime).alias("_dt"))
            stats["time_mode"] = "date-only"
        else:
            expr, mode = _string_to_datetime(df, ts_col)
            if expr is not None:
                df = df.with_columns(expr.alias("_dt"))
                df = _ensure_naive("_dt", df)
                stats["time_mode"] = f"parsed({mode})"

    if stats["time_mode"] == "none" and date_col is not None:
        # Separate date (+optional time) columns. If the 'timestamp' role grabbed a
        # time-only column (e.g. ticket_time = "08:35:00") that failed to parse as a
        # full datetime, reuse it as the time part of the date+time combination.
        time_col = time_col or ts_col
        dtype = df.schema.get(date_col)
        if isinstance(dtype, pl.Date):
            date_expr = pl.col(date_col)
        else:
            date_expr = pl.col(date_col).str.strip_chars().str.to_date(strict=False)
        if time_col is not None:
            combined = (
                pl.col(date_col).cast(pl.String).str.strip_chars()
                + pl.lit(" ")
                + pl.col(time_col).cast(pl.String).str.strip_chars()
            ).str.to_datetime(strict=False)
            try:
                df = df.with_columns(combined.alias("_dt"))
                if df["_dt"].null_count() == df.height:
                    df = df.with_columns(date_expr.alias("_dt_tmp"))
                    stats["time_mode"] = "date-only"
                else:
                    stats["time_mode"] = "date+time-cols"
            except Exception:
                df = df.with_columns(date_expr.alias("_dt_tmp"))
                stats["time_mode"] = "date-only"
        else:
            df = df.with_columns(date_expr.alias("_dt_tmp"))
            stats["time_mode"] = "date-only"
        if "_dt_tmp" in df.columns:
            # Keep unparsed dates as null -> filtered out below with a count.
            df = df.with_columns(
                pl.col("_dt_tmp").cast(pl.Datetime).alias("_dt")
            ).drop("_dt_tmp")

    bad = df["_dt"].null_count() if "_dt" in df.columns else df.height
    stats["dropped_bad_timestamp"] = int(bad)
    if "_dt" not in df.columns:
        raise RuntimeError(
            "No usable timestamp/date column found. Detected headers: "
            f"{list(df.columns)}. Add your header to COLUMN_MAP in this script."
        )
    total_rows = df.height
    probe_col = ts_col or date_col
    raw_samples: list[str] = []
    if probe_col is not None and probe_col in df.columns:
        raw_samples = (
            df.select(pl.col(probe_col).cast(pl.String).str.strip_chars())
            .filter(pl.col(probe_col).is_not_null() & (pl.col(probe_col) != ""))
            .unique().head(5)[probe_col].to_list()
        )
    df = df.filter(pl.col("_dt").is_not_null())
    if df.height == 0 and total_rows > 0:
        log(f"  [ERROR] Timestamp column '{probe_col}' failed to parse in ALL "
            f"{total_rows:,} rows — every row was dropped here.")
        log("          Raw values look like: "
            + " | ".join(repr(s) for s in raw_samples))
        log("          Tried: polars inference, explicit formats (incl. offset-aware),")
        log("          numeric-offset/IST/GMT/UTC suffix strip, Excel serials, epoch (s/ms/us).")
        log("          suffix strip, Excel serial numbers, epoch (s/ms/us).")
        log("          Fix: add your exact format to EXPLICIT_FORMATS at the top of")
        log("          scripts/preprocess_etm_data.py (or paste the values above")
        log("          back to the assistant and it will add it for you).")

    if stats["time_mode"] == "date-only":
        stats["time_missing_rows"] = df.height
        log("  [WARN] Date column has NO time-of-day -> all tickets land in slot 0.")
        log("         Crowdedness model will lack rush-hour signal until a timestamp column exists.")

    df = df.with_columns([
        pl.col("_dt").dt.date().alias("trip_date"),
        ((pl.col("_dt").dt.hour() * 4 + pl.col("_dt").dt.minute() // 15).fill_null(0)
         .clip(0, 95).cast(pl.Int32).alias("slot_15m")),
        (pl.col("_dt").dt.weekday() - 1).cast(pl.Int32).alias("day_of_week"),   # 0=Mon..6=Sun
    ]).with_columns(
        (pl.col("day_of_week") >= 5).cast(pl.Int32).alias("is_weekend")
    ).drop("_dt")
    return df, stats


# ---------------------------------------------------------------------------
# 7. GTFS alignment helpers (loaded from Part 1's routing_index.pkl)
# ---------------------------------------------------------------------------
def load_routing_index() -> dict | None:
    if not ROUTING_INDEX_PKL.exists():
        return None
    with open(ROUTING_INDEX_PKL, "rb") as f:
        idx = pickle.load(f)
    for key in ("stops", "routes", "route_patterns", "stop_to_routes"):
        if key not in idx:
            log(f"  [WARN] routing_index.pkl missing key '{key}' -> ignoring index.")
            return None
    return idx


class GtfsAligner:
    """Resolves raw ETM route numbers / stop names to GTFS ids & stop indices."""

    def __init__(self, index: dict):
        self.stops: dict = index["stops"]
        self.routes: dict = index["routes"]
        self.patterns: dict = index["route_patterns"]
        self.stop_to_routes: dict = index["stop_to_routes"]
        self.name_to_ids: dict = index.get("name_to_stop_ids", {})
        self.number_to_ids: dict = index.get("route_number_to_ids", {})

        # route_id -> {stop_id: {dir: idx}} and normalized name list per route
        self.route_stop_dirs: dict[str, dict[str, dict[int, int]]] = {}
        self.route_names_norm: dict[str, list[str]] = {}
        self.route_name_to_stop: dict[str, dict[str, str]] = {}
        for (rid, d), seq in self.patterns.items():
            by_stop = self.route_stop_dirs.setdefault(rid, {})
            names = self.route_names_norm.setdefault(rid, [])
            n2s = self.route_name_to_stop.setdefault(rid, {})
            for item in seq:
                sid = item["stop_id"]
                by_stop.setdefault(sid, {})[int(d)] = int(item["idx"])
                meta = self.stops.get(sid, {})
                nm = norm_key(meta.get("stop_name") or sid)
                names.append(nm)
                n2s.setdefault(nm, sid)

        self._route_cache: dict[str, str | None] = {}
        self._stop_cache: dict[tuple[str, str], str | None] = {}

    # -- route resolution ---------------------------------------------------
    # Direction/service suffixes ride on the route number in portal exports
    # ('8UP', '8ADOWN', '534ACLUP'). They are stripped and the lookup retried
    # before giving up; the same suffixes feed direction inference separately.
    ROUTE_SUFFIX_RE = re.compile(r"(down|dwn|dn|acl|ac|up)$")

    def _resolve_route_key(self, key: str) -> str | None:
        if key in self.routes:                              # exact GTFS route_id
            return key
        if key in self.number_to_ids:                       # bus number lookup
            ids = sorted(self.number_to_ids[key])
            return ids[0] if ids else None
        hits = sorted(                                      # prefix match e.g. 534A vs 534
            k2 for k2 in self.number_to_ids
            if len(key) >= 2 and (k2.startswith(key) or key.startswith(k2))
        )
        if hits:
            return sorted(self.number_to_ids[hits[0]])[0]
        return None

    def resolve_route(self, raw_route) -> str | None:
        key = norm_key(raw_route)
        if not key:
            return None
        if key in self._route_cache:
            return self._route_cache[key]
        rid = self._resolve_route_key(key)
        if rid is None:
            core = key
            for _ in range(3):                              # '534aclup' -> '534acl' -> '534'
                stripped = self.ROUTE_SUFFIX_RE.sub("", core)
                if not stripped or stripped == core:
                    break
                core = stripped
                rid = self._resolve_route_key(core)
                if rid is not None:
                    break
        self._route_cache[key] = rid
        return rid

    # -- stop resolution (4 tiers: id -> name -> fuzzy -> coordinates) ------
    def resolve_stop(self, raw_route, raw_stop, lat=None, lon=None) -> str | None:
        rkey, skey = norm_key(raw_route), norm_key(raw_stop)
        if not rkey or not skey:
            return None
        cache_key = (rkey, skey)
        if cache_key in self._stop_cache:
            return self._stop_cache[cache_key]

        rid = self.resolve_route(raw_route)
        if rid is None:
            self._stop_cache[cache_key] = None
            return None
        route_stops = self.route_stop_dirs.get(rid, {})

        def on_route(sid: str) -> bool:
            return sid in route_stops

        sid = None
        # Variant keys: the raw normalized key + the abbreviation-expanded one
        # ('kalindikunjdr' AND 'kalindikunjborder') tried through tiers 2-4.
        skeys = [skey]
        expanded = expand_stop_abbreviations(raw_stop)
        if expanded and expanded != skey:
            skeys.append(expanded)
        # Tier 1: raw value IS a GTFS stop_id (on this route).
        if skey in route_stops:
            sid = skey
        for k in skeys:
            if sid is not None:
                break
            # Tier 2: exact normalized stop-name match, restricted to this route.
            for cand in self.name_to_ids.get(k, []):
                if on_route(cand):
                    sid = cand
                    break
            # Tier 3: containment match (e.g. 'kashmergateisbt' vs 'kashmergate').
            if sid is None and len(k) >= 5:
                n2s = self.route_name_to_stop.get(rid, {})
                contains = [
                    (nm, stp) for nm, stp in n2s.items()
                    if len(nm) >= 5 and (k in nm or nm in k)
                ]
                if contains:
                    sid = min(contains, key=lambda x: abs(len(x[0]) - len(k)))[1]
            # Tier 4: fuzzy edit-distance match on route's own stop names.
            if sid is None:
                close = difflib.get_close_matches(
                    k, self.route_names_norm.get(rid, []), n=1, cutoff=0.85
                )
                if close:
                    sid = self.route_name_to_stop[rid][close[0]]
        # Tier 5: Haversine nearest stop on route (only with ticket coords).
        if sid is None and lat is not None and lon is not None:
            best_d, best_s = HATCH_RADIUS_KM, None
            for stp in route_stops:
                meta = self.stops.get(stp, {})
                slat, slon = meta.get("lat"), meta.get("lon")
                if slat is None or slon is None:
                    continue
                d = haversine_km(float(lat), float(lon), slat, slon)
                if d <= best_d:
                    best_d, best_s = d, stp
            sid = best_s

        self._stop_cache[cache_key] = sid
        return sid

    def dirs_for_od(self, rid: str, b_sid: str, a_sid: str) -> int | None:
        """Direction whose pattern has boarding BEFORE alighting (strict separation)."""
        b_map, a_map = self.route_stop_dirs.get(rid, {}).get(b_sid), \
            self.route_stop_dirs.get(rid, {}).get(a_sid)
        if not b_map or not a_map:
            return None
        valid = [(d, a_map[d] - b_map[d]) for d in b_map if d in a_map and a_map[d] > b_map[d]]
        if valid:
            return min(valid, key=lambda x: x[1])[0]
        return min(b_map)  # fallback: smaller direction id

    def dirs_for_stop(self, rid: str, sid: str) -> int | None:
        m = self.route_stop_dirs.get(rid, {}).get(sid)
        return min(m) if m else None


# ---------------------------------------------------------------------------
# 8. Net-load reconstruction (Case A exact + Case B decay) & aggregation
# ---------------------------------------------------------------------------
def build_decay_spread(rows_needed: set[tuple[str, int, int]], aligner: GtfsAligner) -> pl.DataFrame:
    """Case B survival-curve spread table: (route, dir, board_idx) -> alight shares.

    Exponential survival S(n)=exp(-n/lambda); each boarding's mass is split into
    4 cohorts alighting at the quartile stops of the survival curve, with any
    residual mass forced off at the terminus (so loads fall to ~0 at the end).
    """
    lambda_by_pattern: dict[tuple[str, int], float] = {}
    rows = []
    for rid, d, i in rows_needed:
        key = (rid, int(d))
        if key not in lambda_by_pattern:
            n_stops = len(aligner.patterns.get(key, []))
            lam = min(DECAY_LAMBDA_MAX, max(DECAY_LAMBDA_MIN, DECAY_MEAN_FRACTION * max(1, n_stops - 1)))
            lambda_by_pattern[key] = lam
        lam = lambda_by_pattern[key]
        n_stops = len(aligner.patterns.get(key, []))
        last_idx = max(0, n_stops - 1)
        for share, p in zip((0.25, 0.25, 0.25, 0.25), (0.125, 0.375, 0.625, 0.875)):
            ahead = max(1, math.ceil(-lam * math.log(1.0 - p)))
            j = min(i + ahead, last_idx)
            j = max(j, min(i + 1, last_idx))  # alight strictly after boarding
            rows.append((rid, int(d), int(i), int(j), share))
    return pl.DataFrame(
        rows,
        schema={
            "route_id": pl.String, "direction_id": pl.Int32,
            "board_idx": pl.Int32, "alight_idx": pl.Int32, "share": pl.Float64,
        },
        orient="row",
    )


def aggregate_net_load(aligned: pl.DataFrame, aligner: GtfsAligner) -> pl.DataFrame:
    """Aggregate ticket rows into the (route, dir, date, slot, stop_idx) net-load table."""
    keys = ["route_id", "direction_id", "trip_date", "slot_15m"]

    total_stops_map = {
        (rid, int(d)): len(seq) for (rid, d), seq in aligner.patterns.items()
    }
    ts_df = pl.DataFrame(
        [(k[0], k[1], v) for k, v in total_stops_map.items()],
        schema={"route_id": pl.String, "direction_id": pl.Int32, "total_stops": pl.Int32},
        orient="row",
    )

    board = aligned.group_by(keys + ["board_idx"]).agg(
        pl.col("pax").sum().alias("boardings")
    )
    case_a = aligned.filter(pl.col("case") == "A")
    alight_a = case_a.group_by(keys + ["alight_idx"]).agg(
        pl.col("pax").sum().alias("_al_a")
    )
    case_b = aligned.filter(pl.col("case") == "B")
    if case_b.height:
        spread = build_decay_spread(
            set(zip(case_b["route_id"], case_b["direction_id"], case_b["board_idx"])),
            aligner,
        )
        b_board = case_b.group_by(keys + ["board_idx"]).agg(pl.col("pax").sum().alias("_pb"))
        contrib_b = (
            b_board.join(spread, on=["route_id", "direction_id", "board_idx"], how="inner")
            .with_columns((pl.col("_pb") * pl.col("share")).alias("_w"))
            .group_by(keys + ["alight_idx"]).agg(pl.col("_w").sum().alias("_al_b"))
        )
    else:
        contrib_b = None

    idx_cols = keys + ["board_idx"]
    stops_frame = pl.concat(
        [
            board.select([*keys, pl.col("board_idx").alias("stop_idx")]),
            alight_a.select([*keys, pl.col("alight_idx").alias("stop_idx")]),
            *( [contrib_b.select([*keys, pl.col("alight_idx").alias("stop_idx")])] if contrib_b is not None else [] ),
        ],
        how="vertical",
    ).unique(subset=[*keys, "stop_idx"])

    full = (
        stops_frame
        .join(board.rename({"board_idx": "stop_idx"}), on=[*keys, "stop_idx"], how="left")
        .join(alight_a.rename({"alight_idx": "stop_idx"}), on=[*keys, "stop_idx"], how="left")
    )
    if contrib_b is not None:
        full = full.join(
            contrib_b.rename({"alight_idx": "stop_idx"}), on=[*keys, "stop_idx"], how="left"
        )
    else:
        full = full.with_columns(pl.lit(0.0).alias("_al_b"))

    full = (
        full.with_columns([
            pl.col("boardings").fill_null(0.0),
            (pl.col("_al_a").fill_null(0.0) + pl.col("_al_b").fill_null(0.0)).alias("alightings"),
        ])
        .drop(["_al_a", "_al_b"])
        .with_columns(pl.col("alightings").clip(0.0))
        .sort([*keys, "stop_idx"])
    )

    # Net on-board load: cumulative boardings - cumulative alightings per trip-window.
    full = full.with_columns(
        (pl.col("boardings").cum_sum().over(keys)
         - pl.col("alightings").cum_sum().over(keys)).clip(0.0).alias("net_load")
    )

    full = (
        full.join(ts_df, on=["route_id", "direction_id"], how="left")
        .with_columns([
            pl.col("total_stops").fill_null(1),
            pl.col("stop_idx").cast(pl.Int32),
            (pl.col("slot_15m") // 4).cast(pl.Int32).alias("hour"),
            pl.col("trip_date").dt.weekday().sub(1).cast(pl.Int32).alias("day_of_week"),
        ])
        .with_columns((pl.col("day_of_week") >= 5).cast(pl.Int32).alias("is_weekend"))
        .with_columns(pl.lit(1, dtype=pl.Int32).alias("aligned"))
        .select([
            "route_id", "direction_id", "trip_date", "day_of_week", "is_weekend",
            "slot_15m", "hour", "stop_idx", "total_stops",
            "boardings", "alightings", "net_load", "aligned",
        ])
    )
    return full


def aggregate_degraded(df: pl.DataFrame) -> pl.DataFrame:
    """No GTFS index yet: route-level boarding demand per time window
    (stop order unknown -> no net-load reconstruction possible, aligned=0)."""
    keys = ["route_id", "direction_id", "trip_date", "slot_15m"]
    out = (
        df.with_columns([
            pl.col("route_raw").alias("route_id"),
            pl.col("direction_raw").fill_null(0).cast(pl.Int32).alias("direction_id"),
        ])
        .group_by(keys)
        .agg(pl.col("pax").sum().alias("net_load"))
        .with_columns([
            pl.col("net_load").clip(0.0),
            (pl.col("slot_15m") // 4).cast(pl.Int32).alias("hour"),
            pl.col("trip_date").dt.weekday().sub(1).cast(pl.Int32).alias("day_of_week"),
        ])
        .with_columns((pl.col("day_of_week") >= 5).cast(pl.Int32).alias("is_weekend"))
        .with_columns([
            pl.lit(-1, dtype=pl.Int32).alias("stop_idx"),
            pl.lit(1, dtype=pl.Int32).alias("total_stops"),
            pl.col("net_load").alias("boardings"),
            pl.lit(0.0).alias("alightings"),
            pl.lit(0, dtype=pl.Int32).alias("aligned"),
        ])
        .select([
            "route_id", "direction_id", "trip_date", "day_of_week", "is_weekend",
            "slot_15m", "hour", "stop_idx", "total_stops",
            "boardings", "alightings", "net_load", "aligned",
        ])
    )
    return out


def route_level_rows(unplaced: pl.DataFrame, aligner: GtfsAligner) -> pl.DataFrame:
    """Rows whose boarding stop never resolved to a GTFS stop but whose ROUTE did.

    Real portal exports routinely carry stop names the GTFS spells differently
    ('Kalindi Kunj Bdr' vs 'Kalindi Kunj Border', or a synthetic workshop GTFS with
    unrelated names). Dropping those rows used to throw away >99% of a real month
    of tickets. Instead, keep one ROUTE-LEVEL proxy row per
    (route, direction, date, 15-min slot) with the per-stop mean load estimate
    window_pax / pattern_length, so the engine still learns true ridership patterns
    (peak hours, weekday/weekend, per-route demand) from every resolved route.

    Same schema contract as the no-index degraded mode: stop_idx=-1, aligned=0.
    Proxy entries never collide with real stops in the hist lookups (real stops
    have stop_idx >= 0), and per-stop MEAN (not the window sum) keeps crowd_pct
    on the same scale as stop-level rows.
    """
    keys = ["route_id", "direction_id", "trip_date", "slot_15m"]
    schema = {
        "route_id": pl.String, "direction_id": pl.Int32, "trip_date": pl.Date,
        "day_of_week": pl.Int32, "is_weekend": pl.Int32, "slot_15m": pl.Int32,
        "hour": pl.Int32, "stop_idx": pl.Int32, "total_stops": pl.Int32,
        "boardings": pl.Float64, "alightings": pl.Float64,
        "net_load": pl.Float64, "aligned": pl.Int32,
    }
    empty = pl.DataFrame(schema=schema)
    if unplaced.height == 0:
        return empty

    # Pattern length per (route, dir); fall back to the route's mean pattern
    # length, then to the network median. Routes with NO GTFS match at all are
    # kept too, keyed by their RAW code (aligned=0, stop_idx=-1): nothing is
    # silently dropped, and the demand is there if a matching GTFS is loaded
    # later - but the app cannot score them until such a route exists.
    patt_len: dict[tuple[str, int], int] = {}
    route_lens: dict[str, list[int]] = {}
    for (rid, d), seq in aligner.patterns.items():
        n = max(1, len(seq))
        patt_len[(rid, int(d))] = n
        route_lens.setdefault(rid, []).append(n)
    route_mean_len = {k: max(1, round(sum(v) / len(v))) for k, v in route_lens.items()}
    all_lens = sorted(patt_len.values()) or [1]
    default_len = max(1, all_lens[len(all_lens) // 2])

    g = (
        unplaced.group_by(keys)
        .agg(pl.col("pax").sum().alias("window_pax"))
        .sort(keys)
    )
    out_rows = []
    n_raw_code = 0
    for r in g.iter_rows(named=True):
        rid = r["route_id"]
        d = int(r["direction_id"])
        n_stops = patt_len.get((rid, d)) or route_mean_len.get(rid)
        if n_stops is None:
            n_stops = default_len           # raw route code, no GTFS pattern
            n_raw_code += 1
        dt = r["trip_date"]
        per_stop = max(0.0, float(r["window_pax"])) / n_stops
        out_rows.append((
            rid, d, dt,
            dt.weekday(),                    # day_of_week (Mon=0, matches .dt.weekday()-1)
            1 if dt.weekday() >= 5 else 0,   # is_weekend
            int(r["slot_15m"]),
            int(r["slot_15m"]) // 4,
            -1,                              # stop_idx: route-level proxy marker
            n_stops,
            per_stop,                        # boardings
            0.0,                             # alightings
            per_stop,                        # net_load (per-stop mean, NOT window sum)
            0,                               # aligned=0: proxy row
        ))
    if n_raw_code:
        log(f"  [i] {n_raw_code:,} route-window(s) kept as RAW-CODE proxy rows "
            f"(route has no GTFS match; demand is preserved but not scoreable "
            f"until a matching GTFS is loaded).")
    schema = {
        "route_id": pl.String, "direction_id": pl.Int32, "trip_date": pl.Date,
        "day_of_week": pl.Int32, "is_weekend": pl.Int32, "slot_15m": pl.Int32,
        "hour": pl.Int32, "stop_idx": pl.Int32, "total_stops": pl.Int32,
        "boardings": pl.Float64, "alightings": pl.Float64,
        "net_load": pl.Float64, "aligned": pl.Int32,
    }
    if not out_rows:
        return empty
    return pl.DataFrame(out_rows, schema=schema, orient="row")


# ---------------------------------------------------------------------------
# 9. Direction normalization
# ---------------------------------------------------------------------------
DIRECTION_WORDS = {
    "0": 0, "1": 1, "up": 0, "down": 1, "inbound": 0, "outbound": 1,
    "towardcity": 0, "fromcity": 1, "north": 0, "south": 1,
}


def normalize_direction(df: pl.DataFrame, dir_col: str | None) -> pl.DataFrame:
    """Map raw direction values to strict {0,1}; missing -> null (inferred later)."""
    if dir_col is None:
        # No direction column: infer from the route-code suffix so UP/DOWN
        # demand stays separated ('8UP' -> 0, '8ADOWN' -> 1, '8A' -> null).
        rkey = pl.col("route_raw").cast(pl.String).str.strip_chars().str.to_lowercase()
        return df.with_columns(
            pl.when(rkey.str.contains(r"(?:down|dwn|dn)$"))
            .then(pl.lit(1, dtype=pl.Int32))
            .when(rkey.str.contains(r"up$"))
            .then(pl.lit(0, dtype=pl.Int32))
            .otherwise(pl.lit(None, dtype=pl.Int32))
            .alias("direction_raw")
        )
    key = pl.col(dir_col).cast(pl.String).str.strip_chars().str.to_lowercase()
    key_num = pl.col(dir_col).cast(pl.Float64, strict=False)
    return df.with_columns(
        pl.coalesce(
            pl.when(key.is_in(["0", "1"])).then(key_num.cast(pl.Int32)),
            pl.when(key.is_in(list(DIRECTION_WORDS))).then(
                pl.col(dir_col).cast(pl.String).str.strip_chars().str.to_lowercase()
                .replace_strict(dict((k, v) for k, v in DIRECTION_WORDS.items() if not k.isdigit()),
                                default=None, return_dtype=pl.Int32)
            ),
        ).alias("direction_raw")
    )


# ---------------------------------------------------------------------------
# 9.5  ETM commerce insights (web UI panel: fares, ticket types, vehicles, AC)
# ---------------------------------------------------------------------------
_DOW_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def build_etm_insights(slim: pl.DataFrame, files: list[str],
                       route_map: dict | None = None) -> dict:
    """Aggregate the ETM commerce columns (fare, ticket type, vehicle, AC flag)
    into a compact JSON for the web UI's e-ticketing insights panel. `slim` is
    the narrow pre-alignment frame from run(); `route_map` is the raw-route ->
    GTFS-route resolution when an index exists. Every section degrades
    gracefully when its source column is missing from the file."""
    n = slim.height
    has = {k: (k in slim.columns and slim[k].drop_nulls().len() > 0)
           for k in ("fare_amt", "tkt_type", "vehicle_no", "ac_flag")}
    if route_map:
        cmap = {k: (v if v else k) for k, v in route_map.items()}
        work = slim.with_columns(
            pl.col("route_raw").replace_strict(cmap, default=None, return_dtype=pl.String)
              .fill_null(pl.col("route_raw")).alias("route_id"))
    else:
        work = slim.with_columns(pl.col("route_raw").alias("route_id"))
    if has["fare_amt"]:
        work = work.with_columns(pl.col("fare_amt").cast(pl.Float64, strict=False))
    if has["tkt_type"]:
        work = work.with_columns(pl.col("tkt_type").fill_null("unknown"))
    work = work.with_columns((pl.col("slot_15m") // 4).cast(pl.Int32).alias("_hour"))

    tickets_expr = pl.len().alias("tickets")
    rev_agg = ([pl.col("fare_amt").fill_null(0.0).sum().alias("revenue")]
               if has["fare_amt"] else [])

    out: dict = {
        "available": n > 0,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": files,
        "has": has,
        "date_min": str(work["trip_date"].min()),
        "date_max": str(work["trip_date"].max()),
        "routes": int(work["route_id"].n_unique()),
        "stops": int(work["board_raw"].n_unique()),
        "totals": {"tickets": n, "pax": int(work.select(pl.col("pax").sum()).item() or 0)},
    }
    if has["fare_amt"]:
        out["totals"]["revenue"] = round(
            float(work.select(pl.col("fare_amt").fill_null(0.0).sum()).item() or 0.0), 2)
        out["totals"]["avg_fare"] = round(
            float(work.select(pl.col("fare_amt").drop_nulls().mean()).item() or 0.0), 2)

    if has["tkt_type"]:
        tt = work.group_by(["tkt_type"]).agg([tickets_expr] + rev_agg).sort(
            "tickets", descending=True)
        out["ticket_types"] = [
            {"type": r["tkt_type"], "tickets": int(r["tickets"]),
             "revenue": round(float(r["revenue"]), 2) if has["fare_amt"] else None,
             "share_pct": round(100.0 * r["tickets"] / n, 1)}
            for r in tt.head(8).to_dicts()
        ]

    if has["ac_flag"]:
        ac = work.group_by(["ac_flag"]).agg([tickets_expr] + rev_agg)
        ac_map = {int(r["ac_flag"]): r for r in ac.to_dicts() if r["ac_flag"] is not None}
        out["ac_split"] = [
            {"label": lbl,
             "tickets": int(ac_map[flag]["tickets"]) if flag in ac_map else 0,
             "revenue": round(float(ac_map[flag]["revenue"]), 2)
                        if (flag in ac_map and has["fare_amt"]) else None,
             "share_pct": round(100.0 * (int(ac_map[flag]["tickets"])
                                         if flag in ac_map else 0) / n, 1)}
            for lbl, flag in (("AC", 1), ("Non-AC", 0))
        ]

    if has["fare_amt"]:
        tier_expr = (
            pl.when(pl.col("fare_amt") <= 0).then(pl.lit("\u20b90 (pass/free)"))
            .when(pl.col("fare_amt") <= 10).then(pl.lit("\u20b91\u201310"))
            .when(pl.col("fare_amt") <= 20).then(pl.lit("\u20b911\u201320"))
            .when(pl.col("fare_amt") <= 30).then(pl.lit("\u20b921\u201330"))
            .when(pl.col("fare_amt") <= 50).then(pl.lit("\u20b931\u201350"))
            .otherwise(pl.lit("\u20b950+")).alias("_tier"))
        ft = work.select(tier_expr).group_by(["_tier"]).agg(tickets_expr)
        m = {r["_tier"]: int(r["tickets"]) for r in ft.to_dicts()}
        order = ["\u20b90 (pass/free)", "\u20b91\u201310", "\u20b911\u201320",
                 "\u20b921\u201330", "\u20b931\u201350", "\u20b950+"]
        out["fare_tiers"] = [{"tier": t, "tickets": m.get(t, 0)}
                             for t in order if m.get(t, 0) > 0]

    hmap = {int(r["_hour"]): r for r in
            work.group_by(["_hour"]).agg([tickets_expr] + rev_agg).to_dicts()}
    out["hourly"] = [
        {"hour": h, "tickets": int(hmap[h]["tickets"]) if h in hmap else 0,
         "revenue": round(float(hmap[h]["revenue"]), 2)
                    if (h in hmap and has["fare_amt"]) else None}
        for h in range(24)
    ]

    wmap = {int(r["day_of_week"]): int(r["tickets"]) for r in
            work.with_columns(pl.col("day_of_week").cast(pl.Int32))
                .group_by(["day_of_week"]).agg(tickets_expr).to_dicts()}
    out["weekday"] = [{"dow": d, "label": _DOW_LABELS[d], "tickets": wmap.get(d, 0)}
                      for d in range(7)]

    dl = work.group_by(["trip_date"]).agg([tickets_expr] + rev_agg).sort("trip_date")
    out["daily"] = [{"date": str(r["trip_date"]), "tickets": int(r["tickets"]),
                     "revenue": round(float(r["revenue"]), 2) if has["fare_amt"] else None}
                    for r in dl.tail(31).to_dicts()]

    od = work.filter(pl.col("alight_raw").is_not_null() & (pl.col("alight_raw") != ""))
    if od.height:
        odg = od.group_by(["board_raw", "alight_raw"]).agg(tickets_expr).sort(
            "tickets", descending=True)
        out["top_od"] = [{"boarding": r["board_raw"], "alighting": r["alight_raw"],
                          "tickets": int(r["tickets"])} for r in odg.head(12).to_dicts()]
    bg = work.group_by(["board_raw"]).agg(tickets_expr).sort("tickets", descending=True)
    out["top_boarding"] = [{"stop": r["board_raw"], "tickets": int(r["tickets"])}
                           for r in bg.head(12).to_dicts()]
    if has["vehicle_no"]:
        vg = work.filter(pl.col("vehicle_no").is_not_null() & (pl.col("vehicle_no") != "")) \
                 .group_by(["vehicle_no"]).agg([tickets_expr] + rev_agg).sort(
                     "tickets", descending=True)
        out["top_vehicles"] = [{"vehicle": r["vehicle_no"], "tickets": int(r["tickets"]),
                                "revenue": round(float(r["revenue"]), 2)
                                           if has["fare_amt"] else None}
                               for r in vg.head(10).to_dicts()]

    # Per-route table + lookup (top-5 mixes assembled from global group-bys,
    # so the cost stays O(routes) not O(routes x scans)).
    rt_aggs = [tickets_expr] + rev_agg
    if has["fare_amt"]:
        rt_aggs.append(pl.col("fare_amt").drop_nulls().mean().alias("avg_fare"))
    lab_map: dict[str, str] = {}
    for r in work.group_by(["route_id", "route_raw"]).agg(tickets_expr).sort(
            "tickets", descending=True).to_dicts():
        lab_map.setdefault(r["route_id"], r["route_raw"])
    mix_src: dict[str, list] = {}
    if has["tkt_type"]:
        for r in work.group_by(["route_id", "tkt_type"]).agg(tickets_expr).sort(
                "tickets", descending=True).to_dicts():
            mix_src.setdefault(r["route_id"], []).append(
                {"type": r["tkt_type"], "tickets": int(r["tickets"])})
    veh_src: dict[str, list] = {}
    if has["vehicle_no"]:
        for r in work.group_by(["route_id", "vehicle_no"]).agg(tickets_expr).sort(
                "tickets", descending=True).to_dicts():
            if r["vehicle_no"] is None or str(r["vehicle_no"]) == "":
                continue
            veh_src.setdefault(r["route_id"], []).append(
                {"vehicle": r["vehicle_no"], "tickets": int(r["tickets"])})
    b_src: dict[str, list] = {}
    for r in work.group_by(["route_id", "board_raw"]).agg(tickets_expr).sort(
            "tickets", descending=True).to_dicts():
        b_src.setdefault(r["route_id"], []).append(
            {"boarding": r["board_raw"], "tickets": int(r["tickets"])})
    od_src: dict[str, list] = {}
    if od.height:
        for r in od.group_by(["route_id", "board_raw", "alight_raw"]).agg(tickets_expr).sort(
                "tickets", descending=True).to_dicts():
            od_src.setdefault(r["route_id"], []).append(
                {"boarding": r["board_raw"], "alighting": r["alight_raw"],
                 "tickets": int(r["tickets"])})
    ac_src: dict[str, list] = {}
    if has["ac_flag"]:
        for r in work.group_by(["route_id", "ac_flag"]).agg(tickets_expr).to_dicts():
            if r["ac_flag"] is not None:
                ac_src.setdefault(r["route_id"], [0, 0])
                ac_src[r["route_id"]][int(r["ac_flag"])] += int(r["tickets"])

    routes_tbl, lookup = [], {}
    for r in work.group_by(["route_id"]).agg(rt_aggs).sort("tickets", descending=True).to_dicts():
        rid = r["route_id"]
        ac_share = None
        if has["ac_flag"] and rid in ac_src:
            a0, a1 = ac_src[rid]
            if a0 + a1 > 0:
                ac_share = round(100.0 * a1 / (a0 + a1), 1)
        entry = {
            "route_id": rid,
            "label": lab_map.get(rid, rid),
            "tickets": int(r["tickets"]),
            "avg_fare": round(float(r["avg_fare"]), 2) if has["fare_amt"] else None,
            "revenue": round(float(r["revenue"] or 0.0), 2) if has["fare_amt"] else None,
            "ac_share_pct": ac_share,
            "top_boarding": b_src.get(rid, [{}])[0].get("boarding"),
            "ticket_mix": mix_src.get(rid, [])[:5],
            "top_od": od_src.get(rid, [])[:5],
            "top_vehicles": veh_src.get(rid, [])[:5],
        }
        routes_tbl.append(entry)
        lookup[rid] = entry
    out["routes_table"] = routes_tbl
    out["routes_lookup"] = lookup
    return out


# ---------------------------------------------------------------------------
# 10. Main pipeline
# ---------------------------------------------------------------------------
def run(peek: bool = False) -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    files = discover_etm_files()
    if not files:
        log("=" * 70)
        log("NO E-TICKETING FILES FOUND.")
        log(f"Drop your ETM / e-ticketing file(s) into:  {ETM_DIR}")
        log("Supported: .csv, .parquet, or .zip (containing csv/parquet).")
        log("Then run:  python scripts/preprocess_etm_data.py")
        log("=" * 70)
        return 1

    log(f"Found {len(files)} e-ticketing file(s) in data/raw/etm/:")
    for p in files:
        log(f"  - {p.name}")

    # ---- Read & map each file -------------------------------------------
    frames, mapping_report = [], []
    for p in files:
        raw = normalize_headers(read_table(p))
        if raw.height == 0:
            log(f"  [WARN] {p.name} is empty -> skipped.")
            continue
        roles = detect_column_roles(list(raw.columns))
        missing = [c for c in REQUIRED_FOR_RUN if c not in roles]
        if missing or ("timestamp" not in roles and "date" not in roles):
            log(f"  [ERROR] Could not map required columns in {p.name}.")
            log(f"          Detected so far: {roles}")
            log(f"          Headers in file: {list(raw.columns)}")
            log("          Fix: add your exact header names to COLUMN_MAP / USER_OVERRIDES")
            log("          at the top of scripts/preprocess_etm_data.py and re-run.")
            return 2
        if peek:
            log("\n" + "=" * 70)
            log(f"DETECTED COLUMN MAP for {p.name}  ({raw.height:,} rows)")
            log("=" * 70)
            for canon in CANONICAL_PRIORITY:
                if canon in roles:
                    log(f"  {canon:<16} <-  '{roles[canon]}'")
            unmapped = [h for h in raw.columns if h not in roles.values()]
            if unmapped:
                log(f"  (ignored columns: {unmapped})")
            log("\nSample rows (first 3):")
            log(raw.head(3))
            return 0

        keep = list(roles.values())
        # Capture ETM commerce columns (fare / ticket type / vehicle / AC flag)
        # when present — they power the web UI's e-ticketing insights panel.
        used_headers = set(keep)
        extras_present: list[tuple[str, str]] = []
        for extra, aliases in EXTRA_COL_MAP.items():
            hit = next((h for h in aliases if h in raw.columns and h not in used_headers), None)
            if hit is not None:
                used_headers.add(hit)
                extras_present.append((extra, hit))
        sub = raw.select(keep + [hit for _, hit in extras_present])
        inv = {v: k for k, v in roles.items()}
        sub = sub.rename(inv)  # now has canonical names (extras keep raw names)
        for extra, hit in extras_present:
            sub = sub.with_columns(EXTRA_CONVERTERS[extra](hit).alias(extra)).drop(hit)
        sub = sub.with_columns(pl.lit(p.name).alias("source_file"))
        # Parse timestamps & pax PER FILE (before concat) so files with different
        # layouts (epoch millis vs ISO vs separate date+time) never corrupt each other.
        sub = coerce_pax(sub, "passenger_count" if "passenger_count" in sub.columns else None)
        sub, time_stats_f = parse_trip_times(sub, {
            "timestamp": "timestamp" if "timestamp" in sub.columns else None,
            "date": "date" if "date" in sub.columns else None,
            "time": "time" if "time" in sub.columns else None,
        })
        mapping_report.append({"file": p.name, "rows": raw.height, "roles": roles,
                               "time_mode": time_stats_f["time_mode"],
                               "rows_kept": sub.height})
        frames.append(sub)
        if sub.height == 0 and raw.height > 0:
            log(f"  [ERROR] {p.name}: {raw.height:,} rows read but 0 survived time parsing")
            log("          (see the timestamp diagnostics above for the exact cause).")
            return 3

    if not frames:
        log("[ERROR] All discovered files were empty.")
        return 2
    df = pl.concat(frames, how="diagonal_relaxed")
    log(f"\nLoaded {df.height:,} ticket rows from {len(frames)} file(s).")

    # ---- Clean core fields (timestamps already parsed per file) ------------
    df = df.with_columns([
        pl.col("route_id").cast(pl.String).str.strip_chars().alias("route_raw"),
        pl.col("boarding_stop").cast(pl.String).str.strip_chars().alias("board_raw"),
        (pl.col("alighting_stop").cast(pl.String).str.strip_chars()
         if "alighting_stop" in df.columns else pl.lit(None, dtype=pl.String)).alias("alight_raw"),
    ])
    df = normalize_direction(df, "direction_id" if "direction_id" in df.columns else None)
    # Drop the canonical-named RAW route column: a resolved GTFS route_id will
    # replace it below (keeping it would cause a join suffix collision).
    if "route_id" in df.columns:
        df = df.drop("route_id")

    # Drop rows without route or boarding stop.
    before = df.height
    df = df.filter(pl.col("route_raw").is_not_null() & (pl.col("route_raw") != "")
                   & pl.col("board_raw").is_not_null() & (pl.col("board_raw") != ""))
    log(f"Dropped {before - df.height:,} rows with missing route/boarding-stop.")

    if df.height == 0:
        log("[ERROR] No usable rows survived cleaning - check your column mapping.")
        return 3
    time_stats = {
        "time_mode": ", ".join(sorted({m["time_mode"] for m in mapping_report if "time_mode" in m})),
        "dropped_bad_timestamp": sum(m.get("rows", 0) - m.get("rows_kept", 0) for m in mapping_report),
    }
    dates = df.select(pl.col("trip_date").min().alias("mn"), pl.col("trip_date").max().alias("mx"))
    log(f"Timestamps parsed via [{time_stats['time_mode']}]; "
        f"{df.height:,} rows kept; date range {dates['mn'][0]} .. {dates['mx'][0]}")

    # Slim commerce frame for the web UI's e-ticketing insights panel, built
    # BEFORE the alignment joins: polars select shares column buffers, and
    # dropping the extras here keeps the join phase at its baseline memory peak.
    insight_df = df.select(
        [c for c in ["route_raw", "board_raw", "alight_raw", "trip_date", "slot_15m",
                     "day_of_week", "pax", "fare_amt", "tkt_type", "vehicle_no", "ac_flag"]
         if c in df.columns])
    df = df.drop([c for c in ("fare_amt", "tkt_type", "vehicle_no", "ac_flag")
                  if c in df.columns])

    # ---- GTFS alignment (optional: requires Part 1 index) ------------------
    index = load_routing_index()
    route_res: dict | None = None
    route_stats: dict = {}
    if index is None:
        log("\n[DEGRADED MODE] data/processed/routing_index.pkl not found.")
        log("  -> Stop ORDER is unknown, so net-load (boardings - alightings) cannot")
        log("     be reconstructed yet. Training can still run on raw boarding demand.")
        log("  -> After dropping GTFS into data/raw/gtfs/ and running")
        log("     `python scripts/build_part1_index.py`, RE-RUN this script for the")
        log("     full stop-level crowdedness model.")
        out = aggregate_degraded(df)
    else:
        log("\nGTFS routing index found - aligning ETM stops to GTFS patterns...")
        aligner = GtfsAligner(index)

        # Resolve routes for every raw route value.
        raw_routes = df.select(pl.col("route_raw").unique()).to_series().to_list()
        route_res = {r: aligner.resolve_route(r) for r in raw_routes}
        n_res = sum(1 for v in route_res.values() if v)
        route_stats = {"codes": len(route_res), "resolved": n_res,
                       "distinct_gtfs_targets": len({v for v in route_res.values() if v})}
        log(f"  Routes resolved to GTFS: {n_res}/{len(route_res)}")
        n_targets = len({v for v in route_res.values() if v})
        if n_targets < n_res:
            log(f"  [i] {n_res:,} route codes share {n_targets} distinct GTFS routes "
                f"(UP/DOWN/AC suffix variants of the same route collapse onto one; "
                f"direction_id keeps their directions apart).")

        # Resolve boarding / alighting stops per unique (route, stop) pair.
        b_pairs = df.select(["route_raw", "board_raw"]).unique()
        a_pairs = df.select(["route_raw", "alight_raw"]).unique().filter(
            pl.col("alight_raw").is_not_null() & (pl.col("alight_raw") != "")
        )
        b_map_rows, a_map_rows = [], []
        for r, s in zip(b_pairs["route_raw"], b_pairs["board_raw"]):
            b_map_rows.append((r, s, aligner.resolve_stop(r, s)))
        for r, s in zip(a_pairs["route_raw"], a_pairs["alight_raw"]):
            a_map_rows.append((r, s, aligner.resolve_stop(r, s)))
        b_resolved = sum(1 for _, _, v in b_map_rows if v)
        a_resolved = sum(1 for _, _, v in a_map_rows if v)
        log(f"  Boarding stops resolved: {b_resolved}/{len(b_map_rows)} unique names; "
            f"alighting resolved: {a_resolved}/{len(a_map_rows)}")

        b_df = pl.DataFrame(
            b_map_rows,
            schema={"route_raw": pl.String, "board_raw": pl.String, "board_stop_id": pl.String},
            orient="row",
        )
        a_df = pl.DataFrame(
            a_map_rows,
            schema={"route_raw": pl.String, "alight_raw": pl.String, "alight_stop_id": pl.String},
            orient="row",
        )
        df = (
            df.join(b_df, on=["route_raw", "board_raw"], how="left")
            .join(a_df, on=["route_raw", "alight_raw"], how="left")
        )
        # Use resolved GTFS route id when available (raw number otherwise).
        route_map_df = pl.DataFrame(
            [(k, v if v else k) for k, v in route_res.items()],
            schema={"route_raw": pl.String, "route_id": pl.String},
            orient="row",
        )
        df = df.join(route_map_df, on="route_raw", how="left")

        # --- Direction assignment (strict route/direction separation) -------
        # Priority: ETM direction column (if present) > OD-pair inference from
        # GTFS patterns > boarding-stop direction > 0. All rows end up with a
        # strict {0, 1} direction so peak flows never mix across directions.
        od = df.filter(
            pl.col("board_stop_id").is_not_null() & pl.col("alight_stop_id").is_not_null()
        ).select(["route_id", "board_stop_id", "alight_stop_id"]).unique()
        od_dir = {}
        for rid, bs, as_ in zip(od["route_id"], od["board_stop_id"], od["alight_stop_id"]):
            od_dir[(rid, bs, as_)] = aligner.dirs_for_od(rid, bs, as_)
        od_df = pl.DataFrame(
            [(k[0], k[1], k[2], v) for k, v in od_dir.items()],
            schema={"route_id": pl.String, "board_stop_id": pl.String,
                    "alight_stop_id": pl.String, "dir_od": pl.Int32},
            orient="row",
        )
        only_b = df.filter(pl.col("board_stop_id").is_not_null()).select(
            ["route_id", "board_stop_id"]).unique()
        b_dir = {}
        for rid, bs in zip(only_b["route_id"], only_b["board_stop_id"]):
            b_dir[(rid, bs)] = aligner.dirs_for_stop(rid, bs)
        b_df2 = pl.DataFrame(
            [(k[0], k[1], v) for k, v in b_dir.items()],
            schema={"route_id": pl.String, "board_stop_id": pl.String, "dir_b": pl.Int32},
            orient="row",
        )
        df = (
            df.join(od_df, on=["route_id", "board_stop_id", "alight_stop_id"], how="left")
            .join(b_df2, on=["route_id", "board_stop_id"], how="left")
            .with_columns(
                pl.coalesce(pl.col("direction_raw"), pl.col("dir_od"), pl.col("dir_b"))
                .fill_null(0).alias("direction_id")
            )
        )
        df = df.with_columns(pl.col("direction_id").cast(pl.Int32).clip(0, 1))

        # --- Stop indices on the chosen direction's canonical pattern -------
        patt_rows = []
        for (rid, d), seq in aligner.patterns.items():
            for item in seq:
                patt_rows.append((rid, int(d), item["stop_id"], int(item["idx"])))
        patt_df = pl.DataFrame(
            patt_rows,
            schema={"route_id": pl.String, "direction_id": pl.Int32,
                    "stop_id": pl.String, "stop_idx": pl.Int32},
            orient="row",
        )
        df = (
            df.join(
                patt_df.rename({"stop_idx": "board_idx"}),
                left_on=["route_id", "direction_id", "board_stop_id"],
                right_on=["route_id", "direction_id", "stop_id"], how="left",
            )
            .join(
                patt_df.rename({"stop_idx": "alight_idx"}),
                left_on=["route_id", "direction_id", "alight_stop_id"],
                right_on=["route_id", "direction_id", "stop_id"], how="left",
            )
        )

        # Case A / Case B classification.
        df = df.with_columns(
            pl.when(
                pl.col("alight_idx").is_not_null() & pl.col("board_idx").is_not_null()
                & (pl.col("alight_idx") > pl.col("board_idx"))
            ).then(pl.lit("A")).otherwise(pl.lit("B")).alias("case")
        )
        n_a = df.filter(pl.col("case") == "A").height
        log(f"  Case A (exact alighting subtraction): {n_a:,} rows | "
            f"Case B (stop-decay survival): {df.height - n_a:,} rows")

        aligned = df.select([
            "route_id", "direction_id", "trip_date", "slot_15m", "pax",
            "board_idx", "alight_idx", "case",
        ]).with_columns(pl.col("board_idx").fill_null(-1).cast(pl.Int32))
        # Split: rows with a placed boarding stop drive the stop-level net-load
        # table; rows whose boarding stop never resolved but whose ROUTE did are
        # kept as ROUTE-LEVEL proxy rows (stop_idx=-1, aligned=0) so months of
        # real tickets are not thrown away when stop names don't match the GTFS.
        unplaced_mask = pl.col("board_idx") < 0
        n_unplaced = aligned.filter(unplaced_mask).height
        placed = aligned.filter(~unplaced_mask)
        unplaced = aligned.filter(unplaced_mask)
        if n_unplaced:
            log(f"  [WARN] {n_unplaced:,} rows have unresolvable boarding stops -> "
                f"kept as ROUTE-LEVEL demand rows (stop_idx=-1); stop-level rows: "
                f"{placed.height:,}.")
        out = aggregate_net_load(placed, aligner) if placed.height else None
        pseudo = route_level_rows(unplaced, aligner)
        if out is not None and pseudo.height:
            out = (
                pl.concat([out, pseudo], how="vertical_relaxed")
                .sort(["route_id", "direction_id", "trip_date", "slot_15m", "stop_idx"])
            )
        elif out is None:
            out = pseudo
        if not out.height:
            raise RuntimeError(
                "No usable training rows survived preprocessing: no route in your "
                "ETM file resolved to a GTFS route. Check that data/raw/gtfs/ "
                "contains the GTFS for the SAME network as the ETM export, then "
                "re-run build_part1_index.py and this script.")

    # ---- Save artifacts -----------------------------------------------------
    clean_cols = [
        c for c in ["route_raw", "board_raw", "alight_raw", "direction_raw",
                    "board_stop_id", "alight_stop_id", "board_idx", "alight_idx",
                    "case", "pax", "source_file", "fare_amt", "tkt_type",
                    "vehicle_no", "ac_flag"]
        if c in df.columns
    ]
    keep_clean = ["route_id", "trip_date", "slot_15m", "day_of_week", "is_weekend"] + clean_cols
    df.select([c for c in keep_clean if c in df.columns]).write_parquet(
        PROCESSED_DIR / "etm_clean.parquet", compression="zstd"
    )
    out.write_parquet(PROCESSED_DIR / "etm_stop_load.parquet", compression="zstd")

    # ---- ETM commerce insights (web UI e-ticketing panel) -------------------
    insights = build_etm_insights(insight_df, [m["file"] for m in mapping_report], route_res)
    # ensure_ascii=True + encoding="utf-8": the file is pure ASCII on disk, so it
    # reads back safely on EVERY platform/locale (Windows Python defaults text IO
    # to cp1252, where the rupee sign U+20B9 crashes with UnicodeEncodeError).
    (PROCESSED_DIR / "etm_insights.json").write_text(
        json.dumps(insights, indent=1, ensure_ascii=True), encoding="utf-8")

    n_route_rows = int((out["aligned"] == 0).sum()) if "aligned" in out.columns else 0
    n_stop_rows = out.height - n_route_rows
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "files": mapping_report,
        "time_mode": time_stats["time_mode"],
        "rows_loaded": df.height,
        "date_min": str(dates["mn"][0]),
        "date_max": str(dates["mx"][0]),
        "aligned_mode": index is not None,
        "output_rows": out.height,
        "route_resolution": route_stats,
        "stop_level_rows": int(out.height - n_route_rows) if out.height else 0,
        "route_level_rows": n_route_rows,
        "output_columns": out.columns,
        "user_overrides": USER_OVERRIDES,
        "etm_insights": {"available": insights["available"],
                         "totals": insights.get("totals")},
    }
    (PROCESSED_DIR / "preprocess_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    # ---- Summary -------------------------------------------------------------
    log("\n" + "=" * 70)
    log("PREPROCESSING COMPLETE")
    log("=" * 70)
    n_routes = out["route_id"].n_unique()
    n_dates = out["trip_date"].n_unique()
    avg_load = float(out["net_load"].mean() or 0.0)
    p95 = float(out["net_load"].quantile(0.95) or 0.0)
    log(f"  Training rows (route x direction x date x slot x stop): {out.height:,}")
    if n_route_rows:
        log(f"    of which {n_stop_rows:,} stop-level + {n_route_rows:,} route-level "
            f"proxy rows (stop names not matching the GTFS - model still learns "
            f"real ridership patterns per route/slot; get the matching GTFS for "
            f"stop-level accuracy)")
    log(f"  Distinct routes: {n_routes} | Distinct service dates: {n_dates}")
    log(f"  Mean net load/stop-window: {avg_load:.1f} pax | 95th pct: {p95:.1f} pax "
        f"(capacity {BUS_CAPACITY:.0f})")
    if index is None:
        log("  MODE: DEGRADED (no GTFS index) - boardings proxy only.")
    log("  Saved: data/processed/etm_stop_load.parquet")
    if insights["available"]:
        rev_txt = (f", revenue Rs.{insights['totals'].get('revenue', 0):,.0f}"
                   if insights.get("has", {}).get("fare_amt") else "")
        log(f"  Saved: data/processed/etm_insights.json "
            f"({insights['totals']['tickets']:,} e-tickets{rev_txt})")
    log("  Next:  python scripts/train_part2_crowd_model.py")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ETM/e-ticketing preprocessing (Step A)")
    ap.add_argument("--peek", action="store_true",
                    help="Show detected column mapping for the first file and exit.")
    args = ap.parse_args()
    sys.exit(run(peek=args.peek))
