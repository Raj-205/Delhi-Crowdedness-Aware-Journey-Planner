#!/usr/bin/env python3
"""
Part 3 — Background GTFS-RT Live Bus Tracker (15s RAM cache + graceful fallback).

Design (per blueprint):
  * Polls the VehiclePositions feed ONCE every 15 seconds from a background
    asyncio task -> user requests read a RAM dict in < 1 ms (zero rate-limit risk).
  * Parses the binary protobuf feed via gtfs-realtime-bindings; if the portal
    serves a JSON variant, falls back to tolerant JSON parsing.
  * If no API key / URL is configured, or the feed hiccups, the cache simply
    stays empty/stale and every endpoint gracefully returns
    HISTORICAL_FORECAST_FALLBACK mode instead of crashing.

Configuration (optional):
  TRANSPORT_STACK_API_KEY   participant API key (header `x-api-key`)
  GTFS_RT_VEHICLE_URL       VehiclePositions feed URL
  RT_POLL_SECONDS           poll interval (default 15)
"""

from __future__ import annotations

import asyncio
import bisect
import json
import math
import os
import random
import time

import httpx

try:
    from google.transit import gtfs_realtime_pb2
except Exception:  # pragma: no cover - dependency guard
    gtfs_realtime_pb2 = None

# ---------------------------------------------------------------------------
# Configuration & in-memory state
# ---------------------------------------------------------------------------
GTFS_RT_URL_DEFAULT = "https://delhi.transportstack.in/api/gtfs-rt/vehicle-positions"
GTFS_RT_URL = os.environ.get("GTFS_RT_VEHICLE_URL", GTFS_RT_URL_DEFAULT).strip()
API_KEY = os.environ.get("TRANSPORT_STACK_API_KEY", "").strip()
POLL_SECONDS = int(os.environ.get("RT_POLL_SECONDS", "15"))

LIVE_BUS_CACHE: dict[str, list[dict]] = {}
LAST_UPDATED_TS: float = 0.0
LAST_SUCCESS_TS: float = 0.0
LAST_POLL_TS: float = 0.0
LAST_ERROR: str = ""
# The poller starts when a key is provided OR an explicit custom feed URL is set.
FEED_ENABLED: bool = bool(API_KEY) or bool(os.environ.get("GTFS_RT_VEHICLE_URL"))
# Mode B (zero-config): synthesize 2-3 buses per route with Delhi plates moving
# along the real route geometry, so the map shows tracking without any key.
SIMULATE_BUSES = os.environ.get("SIMULATE_BUSES", "1").strip().lower() not in {"0", "false", "no"}
_parse_fail_streak = 0
_feed_mode: str = "OFF"     # "GTFS-RT" | "SIMULATED" | "OFF" — set by the active loop


def feed_status() -> dict:
    """Small status blob for /health and the UI status strip."""
    return {
        "enabled": _feed_mode != "OFF" or bool(LIVE_BUS_CACHE),
        "mode": _feed_mode,
        "configured_with_key": bool(API_KEY),
        "last_poll_ts": LAST_POLL_TS,
        "last_success_ts": LAST_SUCCESS_TS,
        "cache_age_sec": round(time.time() - LAST_SUCCESS_TS, 1) if LAST_SUCCESS_TS else None,
        "active_vehicles": sum(len(v) for v in LIVE_BUS_CACHE.values()),
        "routes_with_buses": len(LIVE_BUS_CACHE),
        "simulated_routes": len(_route_geom),
        "last_error": LAST_ERROR,
    }


# ---------------------------------------------------------------------------
# Geospatial helpers (pure Python - no GIS dependencies)
# ---------------------------------------------------------------------------
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Feed parsing (protobuf first, tolerant JSON fallback)
# ---------------------------------------------------------------------------
def _extract_from_protobuf(content: bytes) -> list[dict]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(content)
    buses = []
    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        v = entity.vehicle
        r_id = str(getattr(v.trip, "route_id", "") or "").strip()
        if not r_id:
            continue
        pos = v.position
        lat, lon = float(pos.latitude), float(pos.longitude)
        if lat == 0.0 and lon == 0.0:
            continue
        try:
            speed_kmh = round(float(pos.speed) * 3.6, 1)
        except Exception:
            speed_kmh = 0.0
        buses.append({
            "vehicle_id": str(v.vehicle.id or v.vehicle.label or entity.id),
            "route_id": r_id,
            "trip_id": str(v.trip.trip_id or ""),
            "lat": lat,
            "lon": lon,
            "speed_kmh": speed_kmh,
            "bearing": round(float(pos.bearing), 1) if pos.bearing else 0.0,
            "timestamp": int(v.timestamp or time.time()),
        })
    return buses


def _extract_from_json(content: bytes) -> list[dict]:
    buses = []
    data = json.loads(content.decode("utf-8", errors="replace"))
    entities = data.get("entity", data if isinstance(data, list) else [])
    for ent in entities:
        v = ent.get("vehicle") or {}
        trip = v.get("trip") or {}
        pos = v.get("position") or {}
        r_id = str(trip.get("routeId") or trip.get("route_id") or "").strip()
        lat, lon = pos.get("latitude"), pos.get("longitude")
        if not r_id or lat is None or lon is None:
            continue
        veh = v.get("vehicle") or {}
        buses.append({
            "vehicle_id": str(veh.get("id") or veh.get("label") or ent.get("id") or ""),
            "route_id": r_id,
            "trip_id": str(trip.get("tripId") or trip.get("trip_id") or ""),
            "lat": float(lat),
            "lon": float(lon),
            "speed_kmh": round(float(pos.get("speed") or 0.0) * 3.6, 1),
            "bearing": round(float(pos.get("bearing") or 0.0), 1),
            "timestamp": int(v.get("timestamp") or time.time()),
        })
    return buses


def parse_feed(content: bytes) -> list[dict]:
    """Parse VehiclePositions regardless of protobuf/JSON encoding."""
    global _parse_fail_streak
    stripped = content.lstrip()[:1]
    try:
        if stripped in (b"{", b"["):
            buses = _extract_from_json(content)
        else:
            if gtfs_realtime_pb2 is None:
                raise RuntimeError("gtfs-realtime-bindings not installed")
            buses = _extract_from_protobuf(content)
        _parse_fail_streak = 0
        return buses
    except Exception as exc:
        _parse_fail_streak += 1
        raise ValueError(f"feed parse failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Background poller
# ---------------------------------------------------------------------------
async def poll_gtfs_rt_loop() -> None:
    """Endless 15s poll loop. Never crashes the app: every failure is recorded
    in feed_status() and the last good cache snapshot is kept alive. After 3
    consecutive failed/empty polls it hands over to Mode B so the map ALWAYS
    shows buses (labelled SIMULATED) instead of a dead feed."""
    global LIVE_BUS_CACHE, LAST_UPDATED_TS, LAST_SUCCESS_TS, LAST_POLL_TS, LAST_ERROR, _feed_mode

    if not FEED_ENABLED:
        return
    _feed_mode = "GTFS-RT"

    headers = {"x-api-key": API_KEY, "Accept": "application/x-protobuf, application/json"}
    if not API_KEY:
        headers.pop("x-api-key", None)
    fails = 0
    async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
        while True:
            LAST_POLL_TS = time.time()
            try:
                resp = await client.get(GTFS_RT_URL)
                if resp.status_code == 200:
                    buses = parse_feed(resp.content)
                    if buses:
                        new_cache: dict[str, list[dict]] = {}
                        for b in buses:
                            new_cache.setdefault(b["route_id"], []).append(b)
                        LIVE_BUS_CACHE = new_cache
                        LAST_UPDATED_TS = LAST_SUCCESS_TS = time.time()
                        LAST_ERROR = ""
                        fails = 0
                    else:
                        fails += 1
                        LAST_ERROR = "feed reachable but returned 0 vehicles"
                else:
                    fails += 1
                    LAST_ERROR = f"HTTP {resp.status_code}"
            except Exception as exc:
                fails += 1
                LAST_ERROR = str(exc)[:200]
                print(f"[GTFS-RT Poll Warning] {LAST_ERROR}", flush=True)
            if fails >= 3 and SIMULATE_BUSES:
                print(f"[Mode B] Live feed unavailable after {fails} polls "
                      f"({LAST_ERROR[:80]}) - falling back to zero-config "
                      "simulated buses so tracking always shows something.", flush=True)
                await poll_simulated_loop()
                return
            await asyncio.sleep(max(5, POLL_SECONDS))


# ---------------------------------------------------------------------------
# Mode B — zero-config synthesized tracking (no API key needed)
#
# Per the architecture blueprint: "Synthesize 2-3 buses per route with Delhi
# plates (DL1PC...), realistic GPS coordinates, speed & timestamps" so the map
# shows moving buses out of the box. Buses drive along the route's REAL shape
# polyline (Part 1 index), advance each 15s tick with traffic-like speed
# wobble, and carry source="SIMULATED" + feed_status(mode="SIMULATED") so the
# UI and judges can always tell demo tracking from a genuine GTFS-RT feed.
# Set SIMULATE_BUSES=0 to disable.
# ---------------------------------------------------------------------------
_route_geom: dict[tuple[str, int], dict] = {}   # (route_id, dir) -> traversal table
_sim_buses: list[dict] = []                     # live synthesized bus states
_sim_last_ts: float = 0.0
_rng = random.Random(20261123)                  # deterministic demo behaviour


def configure_routes(patterns: dict, stops: dict, shapes: dict) -> int:
    """Called once at app startup with the Part 1 index: builds the per-route
    traversal tables (polyline -> cumulative kilometres) the simulator drives
    along. Returns the number of trackable directional routes."""
    global _route_geom
    if not SIMULATE_BUSES:
        return 0
    geom: dict[tuple[str, int], dict] = {}
    for key, seq in patterns.items():
        pts = [[float(p[0]), float(p[1])] for p in (shapes.get(key) or [])
               if p and p[0] is not None and p[1] is not None]
        if len(pts) < 2:  # no polyline -> fall back to the stop coordinates
            pts = [[stops.get(s["stop_id"], {}).get("lat"),
                    stops.get(s["stop_id"], {}).get("lon")] for s in seq]
            pts = [p for p in pts if p[0] is not None and p[1] is not None]
        if len(pts) < 2:
            continue
        cum = [0.0]
        for a, b in zip(pts[:-1], pts[1:]):
            cum.append(cum[-1] + haversine_km(a[0], a[1], b[0], b[1]))
        total = cum[-1]
        if total <= 0.05:  # degenerate geometry (all points identical)
            continue
        geom[key] = {"pts": [(p[0], p[1]) for p in pts], "cum": cum, "total_km": total}
    _route_geom = geom
    return len(geom)


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial compass bearing (0..360) between two points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _pos_along(g: dict, offset_km: float) -> tuple[float, float, float]:
    """Interpolate (lat, lon, bearing) at `offset_km` along the traversal table."""
    i = bisect.bisect_right(g["cum"], offset_km) - 1
    i = max(0, min(i, len(g["pts"]) - 2))
    seg = g["cum"][i + 1] - g["cum"][i]
    t = (offset_km - g["cum"][i]) / seg if seg > 0 else 0.0
    t = max(0.0, min(1.0, t))
    (a_lat, a_lon), (b_lat, b_lon) = g["pts"][i], g["pts"][i + 1]
    return (a_lat + (b_lat - a_lat) * t, a_lon + (b_lon - a_lon) * t,
            _bearing_deg(a_lat, a_lon, b_lat, b_lon))


def _spawn_sim_buses() -> None:
    """2-3 buses per route, spaced along the line so headways look real."""
    global _sim_buses
    _sim_buses = []
    for (rid, d), g in _route_geom.items():
        n = _rng.choice((2, 2, 3))
        for k in range(n):
            _sim_buses.append({
                "route_id": rid, "direction_id": int(d),
                "vehicle_id": f"DL1PC{_rng.randint(1000, 9999)}",
                "trip_id": f"SIM_{rid}_{d}_{k:02d}",
                "offset_km": g["total_km"] * (k + _rng.uniform(0.15, 0.6)) / n,
                "speed_kmh": _rng.uniform(14.0, 38.0),
                "lat": 0.0, "lon": 0.0, "bearing": 0.0,
            })


def _advance_sim_buses(dt_s: float) -> None:
    """Move every bus forward; speed wobbles like real Delhi traffic."""
    for b in _sim_buses:
        g = _route_geom[(b["route_id"], b["direction_id"])]
        b["speed_kmh"] = min(42.0, max(10.0, b["speed_kmh"] + _rng.uniform(-3.0, 3.0)))
        b["offset_km"] = (b["offset_km"] + b["speed_kmh"] * dt_s / 3600.0) % g["total_km"]
        lat, lon, brg = _pos_along(g, b["offset_km"])
        b["lat"], b["lon"], b["bearing"] = lat, lon, brg


async def poll_simulated_loop() -> None:
    """Mode B endless 15s loop: keep LIVE_BUS_CACHE stocked with synthesized
    buses moving along the real route geometry. Never crashes the app."""
    global LIVE_BUS_CACHE, LAST_UPDATED_TS, LAST_SUCCESS_TS, LAST_POLL_TS, LAST_ERROR, _sim_last_ts, _feed_mode
    if not _route_geom:
        print("[Mode B] No route geometry available (routing index missing?) "
              "-> simulation off, endpoints serve historical fallback.", flush=True)
        return
    _feed_mode = "SIMULATED"
    _spawn_sim_buses()
    print(f"[Mode B] Zero-config demo tracking ON - synthesizing "
          f"{len(_sim_buses)} buses across {len(_route_geom)} directional routes "
          f"(SIMULATE_BUSES=0 or set TRANSPORT_STACK_API_KEY for the real feed).", flush=True)
    while True:
        LAST_POLL_TS = time.time()
        dt = max(5.0, min(60.0, time.time() - _sim_last_ts)) if _sim_last_ts else float(POLL_SECONDS)
        _sim_last_ts = time.time()
        _advance_sim_buses(dt)
        cache: dict[str, list[dict]] = {}
        now = int(time.time())
        for b in _sim_buses:
            cache.setdefault(b["route_id"], []).append({
                "vehicle_id": b["vehicle_id"],
                "route_id": b["route_id"],
                "trip_id": b["trip_id"],
                "direction_id": b["direction_id"],
                "lat": round(b["lat"], 6), "lon": round(b["lon"], 6),
                "speed_kmh": round(b["speed_kmh"], 1),
                "bearing": round(b["bearing"], 1),
                "timestamp": now,
                "source": "SIMULATED",
            })
        LIVE_BUS_CACHE = cache
        LAST_UPDATED_TS = LAST_SUCCESS_TS = time.time()
        LAST_ERROR = ""
        await asyncio.sleep(max(5, POLL_SECONDS))


def start_poller() -> asyncio.Task | None:
    """Mode A (real GTFS-RT) when a key/URL is configured, otherwise Mode B
    (zero-config simulated buses) while the routing index provides geometry."""
    if FEED_ENABLED:
        return asyncio.create_task(poll_gtfs_rt_loop())
    if SIMULATE_BUSES and _route_geom:
        return asyncio.create_task(poll_simulated_loop())
    return None


# ---------------------------------------------------------------------------
# Stop snapping, ETA & headway bunching math
# ---------------------------------------------------------------------------
def enrich_buses_for_route(
    route_id: str,
    stop_coords: list[dict],
    origin_info: dict | None,
    scheduled_headway_min: float | None,
) -> dict:
    """Snap cached live buses of `route_id` to the canonical stop sequence and
    compute ETA to the commuter's boarding stop + headway bunching ratio."""
    buses = LIVE_BUS_CACHE.get(route_id, [])
    usable = [st for st in stop_coords
              if st.get("lat") is not None and st.get("lon") is not None]
    if not buses or not usable:
        return {"buses": [], "headway_ratio": None}

    enriched = []
    for bus in buses:
        try:
            nearest = min(
                usable,
                key=lambda st: haversine_km(bus["lat"], bus["lon"], st["lat"], st["lon"]),
            )
            if origin_info and origin_info.get("lat") is not None:
                dist_km = haversine_km(bus["lat"], bus["lon"],
                                       origin_info["lat"], origin_info["lon"])
                eta_min = round(max(1.0, (dist_km / max(bus["speed_kmh"] or 0.0, 16.0)) * 60.0), 1)
            else:
                dist_km, eta_min = 0.0, 1.0
            enriched.append({
                **bus,
                "nearest_stop_name": nearest["stop_name"],
                "nearest_stop_idx": nearest["idx"],
                "distance_to_pickup_km": round(dist_km, 2),
                "eta_to_pickup_min": eta_min,
            })
        except Exception:  # one malformed bus must never kill the whole response
            continue

    enriched.sort(key=lambda b: b["eta_to_pickup_min"])

    # Headway bunching: gap between the two buses closest to the pickup stop vs
    # scheduled headway -> demand build-up multiplier, clipped to [0.85, 1.35].
    headway_ratio = None
    if len(enriched) >= 2 and scheduled_headway_min:
        gap = abs(enriched[1]["eta_to_pickup_min"] - enriched[0]["eta_to_pickup_min"])
        if gap > 0.5:
            headway_ratio = round(min(1.35, max(0.85, gap / max(1.0, scheduled_headway_min))), 2)

    return {"buses": enriched[:10], "headway_ratio": headway_ratio}
