#!/usr/bin/env python3
"""
Delhi Crowdedness-Aware Journey Planner — FastAPI server.

Uniting the three blueprint parts:
  Part 1  /stops, /fetch-routes          -> routing_index.pkl (O(1) lookups)
  Part 2  /plan-journey, /route-crowding -> LightGBM crowd engine + generalized cost
  Part 3  /track-route, /live-buses      -> 15s GTFS-RT RAM cache with fallback

Run:  uvicorn app.main:app --reload        ->  http://127.0.0.1:8000
Docs: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.rt_tracker import (
    feed_status,
    enrich_buses_for_route,
    start_poller,
    configure_routes,
)
from app.scoring_engine import (
    HOTSPOT_THRESHOLD,
    ScoringEngine,
    score_candidate,
    build_options,
)

BASE_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = BASE_DIR / "static"
PROCESSED = BASE_DIR / "data" / "processed"
ROUTING_INDEX_PKL = PROCESSED / "routing_index.pkl"

ENGINE = ScoringEngine()
STOPS: dict = {}
ROUTES: dict = {}
PATTERNS: dict = {}
STOP_TO_ROUTES: dict = {}
SHAPES: dict = {}
# (route, dir) -> {stop_id: {"idx": int, "offset_min": float}} for transfer search
PATTERN_STOPS: dict = {}
# ETM commerce aggregates from data/processed/etm_insights.json (fares, ticket
# types, vehicles, AC split, peak hours, top OD pairs) -> /etm-insights + UI panel
def load_etm_insights() -> dict:
    p = PROCESSED / "etm_insights.json"
    if not p.exists():
        return {"available": False}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[startup] etm_insights.json unreadable: {exc}", flush=True)
        return {"available": False}
    return data if data.get("available") else {"available": False}


def load_routing_index() -> bool:
    global STOPS, ROUTES, PATTERNS, STOP_TO_ROUTES, SHAPES, PATTERN_STOPS
    if not ROUTING_INDEX_PKL.exists():
        return False
    with open(ROUTING_INDEX_PKL, "rb") as f:
        index = pickle.load(f)
    STOPS = index["stops"]
    ROUTES = index["routes"]
    PATTERNS = index["route_patterns"]
    STOP_TO_ROUTES = index["stop_to_routes"]
    SHAPES = index.get("route_shapes", {})
    # Align ITO ISBT 4 coordinates to Vikas Marg transit bridge entry
    if "H2551" in STOPS and STOPS["H2551"].get("lon", 0) < 77.23:
        STOPS["H2551"]["lat"] = 28.628139
        STOPS["H2551"]["lon"] = 77.240535
    for key, seq in PATTERNS.items():
        PATTERN_STOPS[key] = {
            s["stop_id"]: {"idx": s["idx"], "offset_min": s["offset_min"]} for s in seq
        }
    return True


ETM_INSIGHTS: dict = load_etm_insights()
load_routing_index()
try:
    ENGINE.load()
except Exception:
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ETM_INSIGHTS
    engine_status = ENGINE.load()
    index_ok = load_routing_index()
    ETM_INSIGHTS = load_etm_insights()
    sim_routes = configure_routes(PATTERNS, STOPS, SHAPES) if index_ok else 0
    print("[startup] engine:", engine_status, "| routing index:", index_ok,
          "| etm insights:", ETM_INSIGHTS.get("available"),
          "| trackable routes (Mode B):", sim_routes, flush=True)
    if not index_ok:
        print("[startup] routing_index.pkl missing - run scripts/build_part1_index.py", flush=True)
    poller = start_poller()
    yield
    if poller:
        poller.cancel()
        try:
            await poller
        except (asyncio.CancelledError, Exception):
            pass


app = FastAPI(
    title="Delhi Crowdedness-Aware Bus Journey Planner",
    description="Problem 2 - Transport Stack Hackathon 2026. "
                "Crowd-aware route ranking trained on ETM e-ticketing data.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_index() -> None:
    if not STOPS:
        raise HTTPException(
            status_code=503,
            detail="Routing index not built yet. Run: python scripts/build_part1_index.py",
        )


def resolve_departure(departure: str | None, hour: int | None, minute: int | None):
    """Parse the departure request into (hour, minute, dow, is_now, iso_label)."""
    if hasattr(departure, "default"):
        departure = departure.default
    if hasattr(hour, "default"):
        hour = hour.default
    if hasattr(minute, "default"):
        minute = minute.default

    now = datetime.now()
    if isinstance(departure, str) and departure.strip().lower() != "now":
        raw = departure.strip()
        try:
            if "T" in raw:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone().replace(tzinfo=None)
            elif ":" in raw and "-" not in raw:
                parts = raw.split(":")
                hh = int(parts[0])
                mm = int(parts[1]) if len(parts) > 1 else 0
                dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            else:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt.tzinfo is not None:
                    dt = dt.astimezone().replace(tzinfo=None)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Unparseable departure time: {raw}")
        is_now = abs((dt - now).total_seconds()) < 180
        return dt.hour, dt.minute, dt.weekday(), is_now, dt.isoformat(timespec="minutes")

    hh = hour if isinstance(hour, int) else (int(hour) if isinstance(hour, str) and hour.isdigit() else now.hour)
    mm = minute if isinstance(minute, int) else (int(minute) if isinstance(minute, str) and minute.isdigit() else now.minute)
    return hh, mm, now.weekday(), True, now.isoformat(timespec="minutes")


def pattern_coords(key: tuple[str, int]) -> list[dict]:
    """Canonical pattern as map-ready stop coords (id, name, lat/lon, idx, offset)."""
    seq = PATTERNS[key]
    out = []
    for s in seq:
        smeta = STOPS.get(s["stop_id"], {})
        out.append({
            "stop_id": s["stop_id"],
            "stop_name": smeta.get("stop_name", s["stop_id"]),
            "lat": smeta.get("lat"),
            "lon": smeta.get("lon"),
            "idx": s["idx"],
            "offset_min": s["offset_min"],
        })
    return out


def _leg_from_pattern(route_id: str, direction_id: int, from_idx: int, to_idx: int) -> dict:
    seq = PATTERNS[(route_id, direction_id)][from_idx: to_idx + 1]
    meta = ROUTES.get(route_id, {"bus_number": route_id, "route_name": ""})
    stops_out = []
    for s in seq:
        smeta = STOPS.get(s["stop_id"], {})
        stops_out.append({
            "stop_id": s["stop_id"],
            "stop_name": smeta.get("stop_name", s["stop_id"]),
            "lat": smeta.get("lat"),
            "lon": smeta.get("lon"),
            "idx": s["idx"],
            "offset_min": s["offset_min"],
        })
    return {
        "route_id": route_id,
        "direction_id": direction_id,
        "bus_number": meta.get("bus_number", route_id),
        "route_name": meta.get("route_name", ""),
        "total_stops": len(PATTERNS[(route_id, direction_id)]),
        "from_stop_idx": from_idx,
        "to_stop_idx": to_idx,
        "num_stops": len(seq) - 1,
        "stop_sequence": stops_out,
        "shape_coords": SHAPES.get((route_id, direction_id), [])[from_idx: to_idx + 1] if (route_id, direction_id) in SHAPES else [],
    }


def find_candidates(origin: str, dest: str, max_direct: int = 10, max_transfer: int = 6) -> dict:
    """0-transfer + 1-transfer candidate generation via inverted stop->routes index."""
    origin_routes = STOP_TO_ROUTES.get(origin, [])
    dest_routes = STOP_TO_ROUTES.get(dest, [])
    dest_lookup = {}
    for r, d, idx in dest_routes:
        dest_lookup.setdefault((r, d), []).append(idx)

    direct: list[dict] = []
    for r, d, o_idx in origin_routes:
        for d_idx in dest_lookup.get((r, d), []):
            if o_idx < d_idx:
                leg = _leg_from_pattern(r, d, o_idx, d_idx)
                direct.append({
                    "key": f"D|{r}|{d}|{o_idx}-{d_idx}",
                    "type": "DIRECT", "transfers": 0, "legs": [leg],
                    "raw_span_min": max(2.0, leg["stop_sequence"][-1]["offset_min"]
                                        - leg["stop_sequence"][0]["offset_min"]),
                })
    direct.sort(key=lambda c: c["raw_span_min"])
    direct = direct[:max_direct]

    transfers: list[dict] = []
    origin_map = {}
    for r, d, idx in origin_routes:
        origin_map.setdefault((r, d), []).append(idx)
    for (ra, da), o_idxs in origin_map.items():
        for (rb, db), d_idxs in dest_lookup.items():
            if ra == rb:
                continue  # same route handled as direct
            # shared transfer stops: after origin on leg A, before destination on leg B
            stops_a = PATTERN_STOPS[(ra, da)]
            stops_b = PATTERN_STOPS[(rb, db)]
            o_idx = o_idxs[0]
            d_idx = d_idxs[0]
            shared = set(stops_a) & set(stops_b)
            best = None
            for t in shared:
                ta, tb = stops_a[t], stops_b[t]
                if ta["idx"] > o_idx and tb["idx"] < d_idx:
                    # Prefer transfer stops that keep BOTH legs short.
                    score = ta["offset_min"] + (stops_b[dest]["offset_min"] - tb["offset_min"])
                    if best is None or score < best[0]:
                        best = (score, t, ta, tb)
            if best is None:
                continue
            _, t, ta, tb = best
            leg_a = _leg_from_pattern(ra, da, o_idx, ta["idx"])
            leg_b = _leg_from_pattern(rb, db, tb["idx"], d_idx)
            span = max(2.0,
                       (leg_a["stop_sequence"][-1]["offset_min"] - leg_a["stop_sequence"][0]["offset_min"])
                       + (leg_b["stop_sequence"][-1]["offset_min"] - leg_b["stop_sequence"][0]["offset_min"]))
            transfers.append({
                "key": f"T|{ra}{da}-{rb}{db}|{t}",
                "type": "TRANSFER", "transfers": 1,
                "legs": [leg_a, leg_b],
                "raw_span_min": span,
                "transfer_stop": STOPS.get(t, {"stop_name": t, "lat": None, "lon": None}),
            })
    transfers.sort(key=lambda c: c["raw_span_min"])
    transfers = transfers[:max_transfer]
    return {"direct": direct, "transfers": transfers}


def find_nearby_connected_routes(
    origin: str,
    dest: str,
    max_walk_km: float = 1.8,
    max_results: int = 6,
) -> list[dict]:
    """Find bus routes connecting stops within walking distance of origin and/or dest.
    Returns candidate dictionaries with walk_to_board and walk_to_dest distance metadata."""
    from app.rt_tracker import haversine_km

    orig_info = STOPS.get(origin)
    dest_info = STOPS.get(dest)
    if not orig_info or not dest_info:
        return []
    o_lat, o_lon = orig_info.get("lat"), orig_info.get("lon")
    d_lat, d_lon = dest_info.get("lat"), dest_info.get("lon")
    if o_lat is None or o_lon is None or d_lat is None or d_lon is None:
        return []

    deg_lat = max_walk_km / 111.0
    deg_lon = max_walk_km / (111.0 * 0.88)

    # 1. Bounding box query
    near_orig: dict[str, float] = {}
    near_dest: dict[str, float] = {}
    for sid, s in STOPS.items():
        slat, slon = s.get("lat"), s.get("lon")
        if slat is None or slon is None:
            continue
        if abs(slat - o_lat) <= deg_lat and abs(slon - o_lon) <= deg_lon:
            d = haversine_km(o_lat, o_lon, slat, slon)
            if d <= max_walk_km:
                near_orig[sid] = d
        if abs(slat - d_lat) <= deg_lat and abs(slon - d_lon) <= deg_lon:
            d = haversine_km(d_lat, d_lon, slat, slon)
            if d <= max_walk_km:
                near_dest[sid] = d

    near_orig[origin] = 0.0
    near_dest[dest] = 0.0

    # 2. Inverted index lookup
    routes_orig: dict[tuple[str, int], list[tuple[str, int, float]]] = {}
    for sid, w_km in near_orig.items():
        for r, d, idx in STOP_TO_ROUTES.get(sid, []):
            routes_orig.setdefault((r, d), []).append((sid, idx, w_km))

    routes_dest: dict[tuple[str, int], list[tuple[str, int, float]]] = {}
    for sid, w_km in near_dest.items():
        for r, d, idx in STOP_TO_ROUTES.get(sid, []):
            routes_dest.setdefault((r, d), []).append((sid, idx, w_km))

    # 3. Find common direct routes
    common_routes = set(routes_orig.keys()) & set(routes_dest.keys())
    direct_hits = []
    seen = set()
    for r, d in common_routes:
        for osid, oidx, ow in routes_orig[(r, d)]:
            for dsid, didx, dw in routes_dest[(r, d)]:
                if oidx < didx:
                    if osid == origin and dsid == dest:
                        continue  # direct exact match
                    combo = (r, d, osid, dsid)
                    if combo in seen:
                        continue
                    seen.add(combo)
                    direct_hits.append((ow + dw, ow, dw, r, d, osid, dsid, oidx, didx))

    direct_hits.sort(key=lambda x: (x[0], x[8] - x[7]))

    cands = []
    for tot_w, ow, dw, r, d, osid, dsid, oidx, didx in direct_hits[:max_results]:
        leg = _leg_from_pattern(r, d, oidx, didx)
        w_board = None
        if ow > 0.03:
            w_board = {
                "from_stop": {"stop_id": origin, "stop_name": orig_info.get("stop_name", origin), "lat": o_lat, "lon": o_lon},
                "to_stop": {"stop_id": osid, "stop_name": STOPS[osid].get("stop_name", osid), "lat": STOPS[osid].get("lat"), "lon": STOPS[osid].get("lon")},
                "dist_m": round(ow * 1000),
                "dist_km": round(ow, 2),
                "walk_min": round((ow / 4.5) * 60, 1),
            }
        w_dest = None
        if dw > 0.03:
            w_dest = {
                "from_stop": {"stop_id": dsid, "stop_name": STOPS[dsid].get("stop_name", dsid), "lat": STOPS[dsid].get("lat"), "lon": STOPS[dsid].get("lon")},
                "to_stop": {"stop_id": dest, "stop_name": dest_info.get("stop_name", dest), "lat": d_lat, "lon": d_lon},
                "dist_m": round(dw * 1000),
                "dist_km": round(dw, 2),
                "walk_min": round((dw / 4.5) * 60, 1),
            }

        total_walk_m = round(tot_w * 1000)
        total_walk_min = round((tot_w / 4.5) * 60, 1)

        cands.append({
            "key": f"NEARBY|{r}|{d}|{osid}-{dsid}",
            "type": "NEARBY_WALK",
            "transfers": 0,
            "legs": [leg],
            "raw_span_min": max(2.0, leg["stop_sequence"][-1]["offset_min"] - leg["stop_sequence"][0]["offset_min"]),
            "walk_to_board": w_board,
            "walk_to_dest": w_dest,
            "total_walk_m": total_walk_m,
            "total_walk_min": total_walk_min,
            "board_stop": STOPS[osid],
            "alight_stop": STOPS[dsid],
        })

    return cands


# ---------------------------------------------------------------------------
# Part 1 endpoints
# ---------------------------------------------------------------------------
@app.get("/stops")
def search_stops(q: str = Query(..., min_length=1)):
    """Autocomplete Delhi bus stops by name or stop_id."""
    require_index()
    q_lower = q.lower().strip()
    matches = []
    for s in STOPS.values():
        name = s.get("stop_name", "")
        if q_lower in name.lower() or s["stop_id"].lower().startswith(q_lower):
            if s.get("lat") is not None:
                matches.append(s)
        if len(matches) >= 20:
            break
    return {"count": len(matches), "stops": matches}


@app.get("/fetch-routes")
def fetch_candidate_routes(origin: str, dest: str, hour: int = Query(9, ge=0, le=23)):
    """Part 1 raw candidates (no crowding) - useful for debugging & judges."""
    require_index()
    if origin not in STOPS or dest not in STOPS:
        raise HTTPException(status_code=404, detail="Invalid origin or destination stop_id")
    if origin == dest:
        raise HTTPException(status_code=400, detail="Origin and destination cannot be the same")
    h = hour if isinstance(hour, int) else (hour.default if hasattr(hour, "default") and isinstance(hour.default, int) else 9)
    cands = find_candidates(origin, dest)
    return {
        "origin": STOPS[origin], "destination": STOPS[dest], "query_hour": h,
        "direct_options": len(cands["direct"]), "transfer_options": len(cands["transfers"]),
        "candidates": (cands["direct"] + cands["transfers"])[:15],
    }


# ---------------------------------------------------------------------------
# Part 2 endpoint: the full crowd-aware planner
# ---------------------------------------------------------------------------
@app.get("/plan-journey")
def plan_journey(
    origin: str,
    dest: str,
    departure: str | None = Query(None, description="'now' or ISO time / HH:MM"),
    hour: int | None = Query(None, ge=0, le=23),
    minute: int | None = Query(None, ge=0, le=59),
):
    require_index()
    if origin not in STOPS or dest not in STOPS:
        raise HTTPException(status_code=404, detail="Invalid origin or destination stop_id")
    if origin == dest:
        raise HTTPException(status_code=400, detail="Origin and destination cannot be the same")
    if not ENGINE.model and not ENGINE.hist_slot:
        raise HTTPException(
            status_code=503,
            detail="Crowdedness engine not trained yet. Run scripts/preprocess_etm_data.py "
                   "then scripts/train_part2_crowd_model.py.",
        )

    dep_hour, dep_min, dow, is_now, dep_iso = resolve_departure(departure, hour, minute)

    cands = find_candidates(origin, dest)
    all_c = cands["direct"] + cands["transfers"]

    # Search for nearby walk alternatives
    nearby_c = find_nearby_connected_routes(origin, dest, max_walk_km=2.0, max_results=6)
    scored_nearby = []
    for c in nearby_c:
        sc = score_candidate(ENGINE, c, dep_hour, dep_min, dow, None)
        walk_m = c.get("total_walk_min", 0.0)
        sc["total_min"] = round(sc["total_min"] + walk_m, 1)
        sc["walk_min"] = walk_m
        sc["walk_to_board"] = c.get("walk_to_board")
        sc["walk_to_dest"] = c.get("walk_to_dest")
        sc["total_walk_m"] = c.get("total_walk_m", 0)
        sc["board_stop"] = c.get("board_stop")
        sc["alight_stop"] = c.get("alight_stop")
        sc["option_type"] = "NEARBY_STOP"

        w_board = c.get("walk_to_board")
        w_dest = c.get("walk_to_dest")
        b_name = c["board_stop"].get("stop_name", "")
        d_name = c["alight_stop"].get("stop_name", "")
        if w_board and w_dest:
            sc["advice"] = f"Walk {w_board['dist_m']}m ({w_board['walk_min']} min) to {b_name} ➔ Ride Bus {c['legs'][0]['bus_number']} to {d_name} ➔ Walk {w_dest['dist_m']}m ({w_dest['walk_min']} min) to destination."
        elif w_board:
            sc["advice"] = f"Walk {w_board['dist_m']}m ({w_board['walk_min']} min) to {b_name} to board Bus {c['legs'][0]['bus_number']} directly to your destination."
        elif w_dest:
            sc["advice"] = f"Board Bus {c['legs'][0]['bus_number']} directly here ➔ Alight at {d_name} and walk {w_dest['dist_m']}m ({w_dest['walk_min']} min) to your destination."
        scored_nearby.append(sc)

    # Sort nearby walk alternatives: shortest walk distance first, then quickest duration
    scored_nearby.sort(key=lambda x: (x.get("total_walk_m", 0), x.get("total_min", 999.0)))

    lookup = ETM_INSIGHTS.get("routes_lookup", {})
    for sc in scored_nearby:
        for leg in sc.get("legs", []):
            r = lookup.get(leg.get("route_id"))
            if r:
                leg["etm_fare"] = {
                    "avg_fare_inr": r.get("avg_fare"),
                    "tickets_observed": r.get("tickets"),
                    "ac_share_pct": r.get("ac_share_pct"),
                }

    if not all_c:
        if scored_nearby:
            return {
                "query": {
                    "origin": STOPS[origin], "destination": STOPS[dest],
                    "departure": dep_iso, "hour": dep_hour, "minute": dep_min,
                    "day_of_week": dow, "is_now": is_now,
                },
                "data_mode": "NEARBY_STOP_CONNECTED",
                "has_nearby_walk": True,
                "options": scored_nearby,
                "wait_1_bus": None,
                "candidate_counts": {"direct": 0, "transfer": 0, "nearby": len(scored_nearby)},
                "engine": {
                    "model_trained": ENGINE.model is not None,
                    "capacity": 75,
                    "hotspot_threshold_pct": HOTSPOT_THRESHOLD,
                },
                "nearby_alternatives": scored_nearby,
            }

        reach = _reachable_from(origin)
        return JSONResponse({
            "query": {"origin": STOPS[origin], "destination": STOPS[dest],
                      "departure": dep_iso, "day_of_week": dow},
            "options": [], "wait_1_bus": None,
            "message": ("No bus route connects these two stops or nearby stops within walking distance. "
                        "The chips below are stops you can reach from " + STOPS[origin].get("stop_name", origin) + "."),
            "suggestions": {
                "origin_name": STOPS[origin].get("stop_name", origin),
                "direct": reach["direct"][:12],
                "one_transfer": reach["one_transfer"][:8],
            },
        })

    # Live headway adjustment (only for 'now' queries with live buses on leg 1).
    live_ratio = None
    live_block = None
    if is_now:
        leg1 = all_c[0]["legs"][0]
        key1 = (leg1["route_id"], leg1["direction_id"])
        headway = ENGINE.headway_min(leg1["route_id"], leg1["direction_id"], dep_hour)
        enriched = enrich_buses_for_route(leg1["route_id"], pattern_coords(key1),
                                          STOPS.get(origin), headway)
        if enriched["buses"]:
            live_ratio = enriched["headway_ratio"]
            live_block = {"route_id": leg1["route_id"], **enriched}

    scored = [score_candidate(ENGINE, c, dep_hour, dep_min, dow, live_ratio) for c in all_c]
    packaged = build_options(ENGINE, scored, dep_hour, dep_min, dow, live_ratio)

    # ETM fare hints per leg (from the e-ticketing commerce aggregates).
    lookup = ETM_INSIGHTS.get("routes_lookup", {})
    fare_opts = list(packaged.get("options", []))
    if packaged.get("wait_1_bus"):
        fare_opts.append(packaged["wait_1_bus"])
    for opt in fare_opts:
        for leg in opt.get("legs", []):
            r = lookup.get(leg.get("route_id"))
            if r:
                leg["etm_fare"] = {
                    "avg_fare_inr": r.get("avg_fare"),
                    "tickets_observed": r.get("tickets"),
                    "ac_share_pct": r.get("ac_share_pct"),
                }

    data_mode = "LIVE_ADJUSTED" if live_ratio else "HISTORICAL_FORECAST"
    return {
        "query": {
            "origin": STOPS[origin], "destination": STOPS[dest],
            "departure": dep_iso, "hour": dep_hour, "minute": dep_min,
            "day_of_week": dow, "is_now": is_now,
        },
        "data_mode": data_mode,
        "live": live_block,
        "candidate_counts": {"direct": len(cands["direct"]), "transfer": len(cands["transfers"])},
        "nearby_alternatives": scored_nearby[:3],
        "engine": {
            "model_trained": ENGINE.model is not None,
            "capacity": 75,
            "hotspot_threshold_pct": HOTSPOT_THRESHOLD,
            "generalized_cost": "T_travel + 1.6*T_wait + 0.35*Sum(T_s*M(Crowd_s))",
        },
        **packaged,
    }


# ---------------------------------------------------------------------------
# Part 2 endpoint: stop-by-stop crowding for one route
# ---------------------------------------------------------------------------
@app.get("/route-crowding/{route_id}")
def route_crowding(
    route_id: str,
    direction_id: int = Query(0, ge=0, le=1),
    departure: str | None = Query(None, description="'now' or ISO time / HH:MM"),
    hour: int | None = Query(None, ge=0, le=23),
    minute: int | None = Query(None, ge=0, le=59),
):
    require_index()
    key = (route_id, direction_id)
    if key not in PATTERNS:
        raise HTTPException(status_code=404, detail="Route pattern not found")
    if not ENGINE.model and not ENGINE.hist_slot:
        raise HTTPException(status_code=503, detail="Crowdedness engine not trained yet.")
    dep_hour, dep_min, dow, is_now, dep_iso = resolve_departure(departure, hour, minute)
    seq = PATTERNS[key]
    crowd = ENGINE.crowd_profile(route_id, direction_id, seq, dep_hour, dep_min, dow)
    stops = []
    for s, c in zip(seq, crowd):
        smeta = STOPS.get(s["stop_id"], {})
        stops.append({**s, "stop_name": smeta.get("stop_name", s["stop_id"]),
                      "lat": smeta.get("lat"), "lon": smeta.get("lon"), "crowd_pct": c})
    hotspots = [s for s in stops if s["crowd_pct"] >= HOTSPOT_THRESHOLD]
    return {
        "route_id": route_id,
        "bus_number": ROUTES.get(route_id, {}).get("bus_number", route_id),
        "direction_id": direction_id,
        "departure": dep_iso,
        "day_of_week": dow,
        "headway_min": ENGINE.headway_min(route_id, direction_id, dep_hour),
        "stops": stops,
        "mean_crowd_pct": round(sum(crowd) / max(1, len(crowd)), 1),
        "hotspots": hotspots,
        "model_trained": ENGINE.model is not None,
        # E-ticketing commerce details for this route (may be None).
        "etm_details": ETM_INSIGHTS.get("routes_lookup", {}).get(route_id),
    }


# ---------------------------------------------------------------------------
# Part 3 endpoints: live tracking with graceful fallback
# ---------------------------------------------------------------------------
@app.get("/track-route")
def track_selected_route(
    route_id: str,
    direction_id: int = Query(0, ge=0, le=1),
    origin_stop_id: str = Query(""),
    dest_stop_id: str = Query(""),
    mode: str = Query("NOW", pattern="^(NOW|SCHEDULED)$"),
):
    require_index()
    key = (route_id, direction_id)
    if key not in PATTERNS:
        raise HTTPException(status_code=404, detail="Route pattern not found")

    seq = PATTERNS[key]
    dep_hour = datetime.now().hour
    dep_min = datetime.now().minute
    dow = datetime.now().weekday()

    # Crowd profile for map coloring (works even without live feed).
    crowd = ENGINE.crowd_profile(route_id, direction_id, seq, dep_hour, dep_min, dow) \
        if (ENGINE.model or ENGINE.hist_slot) else [0.0] * len(seq)
    stop_coords = []
    for s, c in zip(seq, crowd):
        smeta = STOPS.get(s["stop_id"], {})
        stop_coords.append({
            "stop_id": s["stop_id"], "stop_name": smeta.get("stop_name", s["stop_id"]),
            "lat": smeta.get("lat"), "lon": smeta.get("lon"), "idx": s["idx"],
            "crowd_pct": c,
        })

    origin_info = STOPS.get(origin_stop_id) if origin_stop_id else None
    headway = ENGINE.headway_min(route_id, direction_id, dep_hour)

    raw_buses = []
    if mode == "NOW":
        enriched = enrich_buses_for_route(route_id, pattern_coords(key), origin_info, headway)
        raw_buses = enriched["buses"]
        headway_ratio = enriched["headway_ratio"]
    else:
        headway_ratio = None

    if not raw_buses:
        return {
            "route_id": route_id,
            "direction_id": direction_id,
            "tracking_mode": "HISTORICAL_FORECAST_FALLBACK",
            "feed_source": feed_status().get("mode", "OFF"),
            "message": ("Showing the scheduled route and the historical crowd picture "
                        "learned from your ETM data. Live GPS is not available for this "
                        "query (future departure time or feed gap) - nothing crashed."),
            "last_feed_sync": feed_status(),
            "headway_ratio": headway_ratio,
            "scheduled_headway_min": headway,
            "polyline": stop_coords,
            "shape": SHAPES.get(key),
            "active_buses": [],
        }

    feed_mode = feed_status().get("mode", "GTFS-RT")
    if feed_mode == "SIMULATED":
        tracking_mode = "SIMULATED_FEED"
        message = (f"{len(raw_buses)} buses running on route {ROUTES.get(route_id, {}).get('bus_number', route_id)} "
                   "right now (demo tracking - no GTFS-RT key configured, buses are "
                   "synthesized along the real route path).")
    else:
        tracking_mode = "LIVE_GTFS_RT"
        message = (f"{len(raw_buses)} buses running on route {ROUTES.get(route_id, {}).get('bus_number', route_id)} "
                   "right now, straight from the 15s GTFS-RT cache.")

    return {
        "route_id": route_id,
        "direction_id": direction_id,
        "tracking_mode": tracking_mode,
        "feed_source": feed_mode,
        "message": message,
        "last_feed_sync": feed_status(),
        "headway_ratio": headway_ratio,
        "scheduled_headway_min": headway,
        "polyline": stop_coords,
        "shape": SHAPES.get(key),
        "active_buses": raw_buses,
    }


@app.get("/live-buses/{route_id}")
def live_buses(route_id: str):
    """Raw cached vehicle positions for a route (from the 15s RAM cache)."""
    buses = []
    from app.rt_tracker import LIVE_BUS_CACHE
    for b in LIVE_BUS_CACHE.get(route_id, []):
        buses.append(b)
    return {"route_id": route_id, "count": len(buses), "buses": buses,
            "feed": feed_status()}


# ---------------------------------------------------------------------------
# Reachable-stop suggestions: helps users pick stop pairs that actually have
# a bus connection (direct or one change) in the loaded GTFS.
# ---------------------------------------------------------------------------
_CONNECTED_CACHE: dict[str, dict] = {}


def _reachable_from(stop_id: str) -> dict:
    """Stops reachable from `stop_id` with 0 or 1 bus change, cached per stop.
    stop_to_routes values are (route_id, direction_id, stop_idx) tuples."""
    if stop_id in _CONNECTED_CACHE:
        return _CONNECTED_CACHE[stop_id]
    origin_patterns = {(r[0], r[1]) for r in STOP_TO_ROUTES.get(stop_id, [])}
    direct: set[str] = set()
    for key in origin_patterns:
        seq = PATTERN_STOPS.get(key)
        if seq:
            direct.update(seq.keys())
    direct.discard(stop_id)
    mid_patterns: set[tuple[str, int]] = set()
    for m in direct:
        for entry in STOP_TO_ROUTES.get(m, []):
            mid_patterns.add((entry[0], entry[1]))
    mid_patterns -= origin_patterns
    transfer: set[str] = set()
    for key in mid_patterns:
        seq = PATTERN_STOPS.get(key)
        if seq:
            transfer.update(seq.keys())
    transfer -= direct | {stop_id}

    def named(ids: set[str], cap: int) -> list[dict]:
        ordered = sorted(ids, key=lambda s: STOPS.get(s, {}).get("stop_name", s))
        return [{"stop_id": s, "stop_name": STOPS.get(s, {}).get("stop_name", s)}
                for s in ordered[:cap]]

    out = {"direct": named(direct, 40), "one_transfer": named(transfer, 40)}
    _CONNECTED_CACHE[stop_id] = out
    return out


@app.get("/connected-stops")
def connected_stops(stop_id: str, limit: int = Query(12, ge=1, le=40)):
    """Where can I go from this stop? Direct stops + one-change stops, so the
    UI can suggest destination pairs that actually have a bus connection."""
    require_index()
    if stop_id not in STOPS:
        raise HTTPException(status_code=404, detail="Unknown stop_id")
    reach = _reachable_from(stop_id)
    lim = limit if isinstance(limit, int) else (limit.default if hasattr(limit, "default") and isinstance(limit.default, int) else 12)
    return {
        "stop_id": stop_id,
        "stop_name": STOPS[stop_id].get("stop_name", stop_id),
        "direct": reach["direct"][:lim],
        "one_transfer": reach["one_transfer"][:lim],
    }


@app.get("/popular-hubs")
def popular_hubs():
    """Curated popular Delhi hubs and verified connected routes for 1-click UI demos."""
    return {
        "quick_routes": [
            {
                "label": "Kashmere Gate ➔ Nehru Place",
                "label_hi": "कश्मीरी गेट ➔ नेहरू प्लेस",
                "origin_id": "H4000",
                "origin_name": "Kashmere Gate Enclave 6",
                "dest_id": "H2802",
                "dest_name": "Nehru Place Sector 4",
                "bus": "Bus R100",
            },
            {
                "label": "ITO ISBT ➔ Mayur Vihar",
                "label_hi": "आईटीओ ➔ मयूर विहार",
                "origin_id": "H2551",
                "origin_name": "ITO ISBT 4",
                "dest_id": "H1559",
                "dest_name": "Mayur Vihar Marg 3",
                "bus": "Bus R101",
            },
            {
                "label": "Kashmere Gate Depot ➔ Nehru Place Sec 1",
                "label_hi": "कश्मीरी गेट डिपो ➔ नेहरू प्लेस",
                "origin_id": "H1100",
                "origin_name": "Kashmere Gate Depot 2",
                "dest_id": "H0702",
                "dest_name": "Nehru Place Sector 1",
                "bus": "Bus R100A",
            },
            {
                "label": "Mayur Vihar ➔ ITO Vihar",
                "label_hi": "मयूर विहार ➔ आईटीओ",
                "origin_id": "H1809",
                "origin_name": "Mayur Vihar Extension 3",
                "dest_id": "H5351",
                "dest_name": "ITO Vihar 8",
                "bus": "Bus R101",
            },
        ]
    }


@app.get("/etm-insights")
def etm_insights():
    """E-ticketing commerce aggregates from your ETM file (Step A artifact):
    total tickets/revenue, ticket-type mix, fare tiers, AC split, peak hours,
    weekday split, daily trend, top OD pairs, top boarding stops, top vehicles
    and a per-route table. Powers the UI's E-Ticketing Insights panel."""
    return ETM_INSIGHTS


@app.get("/health")
def health():
    """Diagnostics for judges: which artifacts are loaded & data freshness."""
    return {
        "status": "ok",
        "routing_index_loaded": bool(STOPS),
        "stops": len(STOPS),
        "routes": len(ROUTES),
        "directional_patterns": len(PATTERNS),
        "engine": {
            "model_trained": ENGINE.model is not None,
            "model_meta": ENGINE.meta.get("metrics"),
            "degraded_training": ENGINE.meta.get("degraded"),
            "hist_fallback_available": bool(ENGINE.hist_slot),
        },
        "etm_insights": {
            "available": bool(ETM_INSIGHTS.get("available")),
            "tickets": ETM_INSIGHTS.get("totals", {}).get("tickets"),
        },
        "live_feed": feed_status(),
        "time": datetime.now().isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------
@app.get("/")
def index_page():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
