#!/usr/bin/env python3
"""
Part 2 (online) — Crowdedness Scoring & Generalized Discomfort Cost Ranker.

Loads the artifacts produced by scripts/train_part2_crowd_model.py:
  crowd_lgbm_model.txt, model_meta.json, hist_crowd_lookup.parquet, hist_slot_crowd.parquet

Responsibilities:
  * Predict stop-by-stop crowd % (0-100, normalized against 75-pax capacity) for
    every candidate leg, honouring route_id + direction_id separation and the
    arrival TIME at each stop (rush-hour features follow the bus, not the query).
  * Detect hotspot stops (>= HOTSPOT_THRESHOLD% of capacity).
  * Rank itineraries by Generalized Discomfort Cost:
        Cost = T_travel + ALPHA * T_wait + BETA * Sum_s( T_s * M(Crowd_s) )
    with the blueprint's non-linear discomfort multiplier M:
        0-50% seats available -> 1.00x | 51-75% -> 1.25x
        76-90% crowded standing -> 1.65x | >90% crush load -> 2.20x
  * Surface distinct commuter options: FASTEST, MOST COMFORTABLE, WAIT-1-BUS.
  * If the LightGBM model is absent (e.g. app started before training), falls
    back to the leak-free historical lookup tables so the app still works.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED = BASE_DIR / "data" / "processed"

BUS_CAPACITY = 75.0
HOTSPOT_THRESHOLD = 80.0          # crowd % above which a stop is flagged
ALPHA_WAIT = 1.6                  # minutes of waiting cost `alpha` minutes of travel
BETA_CROWD = 0.35                 # weight of the crowding discomfort integral
RUSH_SLOTS = set(range(34, 42)) | set(range(70, 82))

_DISCOMFORT_TIERS = [(50.0, 1.00), (75.0, 1.25), (90.0, 1.65), (100.01, 2.20)]


def discomfort_multiplier(crowd_pct: float) -> float:
    """Blueprint's M(Crowd): non-linear penalty once seats fill up."""
    for upper, mult in _DISCOMFORT_TIERS:
        if crowd_pct <= upper:
            return mult
    return 2.20


def crowd_level_label(crowd_pct: float) -> str:
    if crowd_pct <= 50:
        return "Seats Available"
    if crowd_pct <= 75:
        return "Comfortable Standing"
    if crowd_pct <= 90:
        return "Crowded Standing"
    return "Crush Load"


# ---------------------------------------------------------------------------
# Delhi traffic calibration (static GTFS -> realistic road times)
# ---------------------------------------------------------------------------
_WEEKDAY_ANCHORS = [
    (0.0, 0.98), (5.5, 1.00), (7.0, 1.15), (8.5, 1.38), (10.5, 1.38),
    (11.5, 1.10), (16.0, 1.10), (17.5, 1.42), (20.5, 1.42), (21.5, 1.10),
    (23.0, 1.00), (24.0, 0.98),
]
_SUNDAY_ANCHORS = [(0.0, 0.95), (9.0, 1.00), (13.0, 1.05), (20.0, 1.00), (24.0, 0.95)]


def traffic_multiplier(hour_float: float, is_sunday: bool = False) -> float:
    """Empirical time-of-day multipliers (Delhi corridors; see Challenges doc):
    morning peak 1.30-1.45x, midday 1.05-1.15x, evening peak 1.35-1.50x,
    late night / Sunday 0.95-1.05x."""
    anchors = _SUNDAY_ANCHORS if is_sunday else _WEEKDAY_ANCHORS
    h = min(max(hour_float, 0.0), 24.0)
    for (h0, m0), (h1, m1) in zip(anchors, anchors[1:]):
        if h0 <= h <= h1:
            if h1 == h0:
                return m0
            return m0 + (m1 - m0) * (h - h0) / (h1 - h0)
    return 1.0


# ---------------------------------------------------------------------------
# Artifact loading (index + model + lookups)
# ---------------------------------------------------------------------------
class ScoringEngine:
    def __init__(self):
        self.model = None
        self.meta: dict = {}
        self.index: dict | None = None
        self.hist_stop: dict = {}       # (route, dir, stop_idx, weekend) -> crowd %
        self.hist_slot: dict = {}       # (route, dir, weekend, slot) -> crowd %
        self.hist_route_mean: dict = {} # (route, dir, weekend) -> crowd %
        self.global_mean: float = 35.0

    def load(self) -> dict:
        status = {"index": False, "model": False, "hist": False}
        pkl = PROCESSED / "routing_index.pkl"
        if pkl.exists():
            import pickle
            with open(pkl, "rb") as f:
                self.index = pickle.load(f)
            status["index"] = True

        meta_path = PROCESSED / "model_meta.json"
        model_path = PROCESSED / "crowd_lgbm_model.txt"
        if meta_path.exists():
            self.meta = json.loads(meta_path.read_text())
            self.global_mean = float(self.meta.get("global_mean_crowd", 35.0))
        if model_path.exists():
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(model_path))
            status["model"] = True

        stop_hist_path = PROCESSED / "hist_crowd_lookup.parquet"
        if stop_hist_path.exists():
            df = pl.read_parquet(stop_hist_path)
            self.hist_stop.clear()
            accum: dict[tuple, list[float]] = {}
            for r in df.iter_rows(named=True):
                key = (r["route_id"], int(r["direction_id"]), int(r["stop_idx"]), int(r["is_weekend"]))
                val = float(r["hist_crowd_pct"])
                self.hist_stop[key] = val
                rk = (r["route_id"], int(r["direction_id"]), int(r["is_weekend"]))
                if rk not in accum:
                    accum[rk] = [val, 1.0]
                else:
                    accum[rk][0] += val
                    accum[rk][1] += 1.0
            self.hist_route_mean = {
                k: v[0] / max(1.0, v[1]) for k, v in accum.items()
            }
            status["hist"] = True
        slot_hist_path = PROCESSED / "hist_slot_crowd.parquet"
        if slot_hist_path.exists():
            df = pl.read_parquet(slot_hist_path)
            self.hist_slot.clear()
            for r in df.iter_rows(named=True):
                self.hist_slot[(r["route_id"], int(r["direction_id"]),
                                int(r["is_weekend"]), int(r["slot_15m"]))] = float(r["hist_crowd_pct"])
        return status

    # -- lookups ------------------------------------------------------------
    def headway_min(self, route_id: str, direction_id: int, hour: int) -> float:
        if self.index:
            hw = self.index.get("headway_min", {}).get((route_id, direction_id, int(hour)))
            if hw:
                return float(hw)
        return 12.0  # typical DTC headway fallback

    def _stop_hist(self, route_id, direction_id, stop_idx, weekend) -> float | None:
        return self.hist_stop.get((route_id, int(direction_id), int(stop_idx), int(weekend)))

    def _route_hist_mean(self, route_id, direction_id, weekend) -> float | None:
        return self.hist_route_mean.get((route_id, int(direction_id), int(weekend)))

    def _slot_hist(self, route_id, direction_id, weekend, slot) -> float | None:
        return self.hist_slot.get((route_id, int(direction_id), int(weekend), int(slot)))

    # -- crowd prediction -----------------------------------------------------
    def crowd_profile(self, route_id: str, direction_id: int, stops: list[dict],
                      dep_hour: int, dep_min: int, day_of_week: int,
                      live_headway_ratio: float | None = None) -> list[float]:
        """Predict crowd % at each stop of an ordered leg sequence.

        `stops` = [{'stop_id', 'idx', 'offset_min', ...}, ...] (leg-local order).
        Each stop's features use its own ARRIVAL time (bus time-of-day travels)."""
        weekend = 1 if day_of_week >= 5 else 0
        dep_total = dep_hour * 60 + dep_min

        if self.model is not None and self.meta:
            return self._predict_with_model(
                route_id, direction_id, stops, dep_total, day_of_week, weekend,
                live_headway_ratio,
            )
        return self._fallback_profile(
            route_id, direction_id, stops, dep_total, weekend, live_headway_ratio
        )

    def _predict_with_model(self, route_id, direction_id, stops, dep_total,
                            day_of_week, weekend, live_headway_ratio) -> list[float]:
        import numpy as np
        import pandas as pd

        cat_meta = self.meta.get("cat_categories", {})
        feature_cols = self.meta.get("feature_cols", [])
        rows = []
        for s in stops:
            arr_total = dep_total + float(s.get("offset_min", 0.0))
            arr_slot = int((arr_total % 1440) // 15)
            arr_hour = int((arr_total // 60) % 24)
            total = max(int(s.get("total_stops", 1)) - 1, 1)
            norm_pos = min(max(s.get("idx", 0) / total, 0.0), 1.0)
            hist = self._stop_hist(route_id, direction_id, s.get("idx", 0), weekend)
            rows.append({
                "route_id": str(route_id),
                "direction_id": str(int(direction_id)),
                "day_of_week": str(int(day_of_week)),
                "is_weekend": str(int(weekend)),
                "slot_15m": arr_slot,
                "stop_idx": int(s.get("idx", 0)),
                "norm_stop_pos": norm_pos,
                "hour_sin": math.sin(2 * math.pi * arr_slot / 96.0),
                "hour_cos": math.cos(2 * math.pi * arr_slot / 96.0),
                "is_rush_hour": 1 if arr_slot in RUSH_SLOTS else 0,
                "hist_route_stop_crowd": float(hist) if hist is not None else self.global_mean,
            })

        X = pd.DataFrame(rows, columns=feature_cols)
        for c in self.meta.get("cat_cols", []):
            cats = cat_meta.get(c, [])
            X[c] = pd.Categorical(X[c].astype(str), categories=cats)

        try:
            preds = self.model.predict(X)
        except Exception:
            preds = np.full(len(rows), self.global_mean)
        preds = np.clip(np.asarray(preds, dtype=float), 0.0, 100.0)
        if live_headway_ratio:
            preds = np.clip(preds * float(live_headway_ratio), 0.0, 100.0)
        return [round(float(p), 1) for p in preds]

    def _fallback_profile(self, route_id, direction_id, stops, dep_total,
                          weekend, live_headway_ratio) -> list[float]:
        """Model-less mode: historical slot demand scaled by the stop's relative
        attractiveness (both computed leak-free in the training script)."""
        out = []
        for s in stops:
            arr_total = dep_total + float(s.get("offset_min", 0.0))
            arr_slot = int((arr_total % 1440) // 15)
            base = self._slot_hist(route_id, direction_id, weekend, arr_slot)
            if base is None:
                base = self._route_hist_mean(route_id, direction_id, weekend)
            if base is None:
                base = self.global_mean
            stop_hist = self._stop_hist(route_id, direction_id, s.get("idx", 0), weekend)
            route_mean = self._route_hist_mean(route_id, direction_id, weekend)
            ratio = 1.0
            if stop_hist is not None and route_mean:
                ratio = min(1.6, max(0.5, stop_hist / route_mean))
            pct = min(100.0, max(0.0, base * ratio))
            if live_headway_ratio:
                pct = min(100.0, pct * float(live_headway_ratio))
            out.append(round(pct, 1))
        return out


# ---------------------------------------------------------------------------
# Scoring & ranking of routing candidates
# ---------------------------------------------------------------------------
def score_leg(engine: ScoringEngine, leg: dict, dep_hour: int, dep_min: int,
              day_of_week: int, leg_kind: str, live_headway_ratio: float | None) -> dict:
    """Attach crowd profile + travel/wait/cost figures to one candidate leg.
    Returns an updated leg dict (also mutates `leg` in place)."""
    route_id = leg["route_id"]
    direction_id = int(leg["direction_id"])
    seq = leg["stop_sequence"]
    is_sunday = day_of_week == 6

    # Scheduled headway at boarding hour -> expected wait ~ half headway.
    headway = engine.headway_min(route_id, direction_id, dep_hour)
    wait_min = round(min(20.0, headway / 2.0), 1)

    # Traffic-calibrated per-stop offsets (arrival minutes from leg start).
    start_offset = float(seq[0].get("offset_min", 0.0))
    mult = traffic_multiplier(dep_hour + dep_min / 60.0 + start_offset / 60.0, is_sunday)
    raw_span = max(2.0, float(seq[-1]["offset_min"]) - float(seq[0]["offset_min"]))
    travel_min = round(raw_span * mult, 1)

    crowd = engine.crowd_profile(
        route_id, direction_id, seq, dep_hour, dep_min, day_of_week, live_headway_ratio
    )

    # Generalized discomfort integral: Sum_s T_s * M(Crowd_s).
    crowd_penalty_min = 0.0
    for i in range(len(seq) - 1):
        seg_min = max(0.1, float(seq[i + 1]["offset_min"]) - float(seq[i]["offset_min"])) * mult
        crowd_penalty_min += seg_min * discomfort_multiplier(crowd[i])
    if len(seq) >= 1:
        crowd_penalty_min += 0.0

    for s, c in zip(seq, crowd):
        s["crowd_pct"] = c

    leg.update({
        "dep_hour": dep_hour, "dep_min": dep_min,
        "scheduled_headway_min": headway,
        "wait_min": wait_min,
        "travel_min": travel_min,
        "crowd_pct_per_stop": crowd,
        "mean_crowd_pct": round(sum(crowd) / max(1, len(crowd)), 1),
        "max_crowd_pct": round(max(crowd) if crowd else 0.0, 1),
        "traffic_multiplier": round(mult, 2),
    })
    leg["_crowd_penalty_min"] = round(crowd_penalty_min, 1)
    leg["_kind"] = leg_kind
    return leg


def score_candidate(engine: ScoringEngine, cand: dict, dep_hour: int, dep_min: int,
                    day_of_week: int, live_headway_ratio: float | None = None) -> dict:
    """Score a full candidate (1 or 2 legs). Returns enriched option dict."""
    hour, minute = dep_hour, dep_min
    total_travel = total_wait = 0.0
    total_penalty = 0.0
    legs_out = []
    for i, leg in enumerate(cand["legs"]):
        scored = score_leg(engine, leg, hour, minute, day_of_week,
                           "primary" if i == 0 else "transfer",
                           live_headway_ratio if i == 0 else None)
        legs_out.append(scored)
        total_travel += scored["travel_min"]
        total_wait += scored["wait_min"]
        total_penalty += scored["_crowd_penalty_min"]
        # Boarding time for the next leg = arrival at transfer stop + wait.
        arrival_total = hour * 60 + minute + scored["travel_min"]
        hour = int((arrival_total + scored["wait_min"]) // 60) % 24
        minute = int((arrival_total + scored["wait_min"]) % 60)

    cost = total_travel + ALPHA_WAIT * total_wait + BETA_CROWD * total_penalty
    all_crowd = [c for l in legs_out for c in l["crowd_pct_per_stop"]]
    hotspots = []
    for l in legs_out:
        for s, c in zip(l["stop_sequence"], l["crowd_pct_per_stop"]):
            if c >= HOTSPOT_THRESHOLD:
                hotspots.append({"stop_name": s["stop_name"], "crowd_pct": c,
                                 "route": l["bus_number"]})
    hotspots.sort(key=lambda h: -h["crowd_pct"])

    mean_crowd = round(sum(all_crowd) / max(1, len(all_crowd)), 1)
    return {
        "key": cand.get("key", ""),
        "legs": legs_out,
        "transfers": cand.get("transfers", 0),
        "travel_min": round(total_travel, 1),
        "wait_min": round(total_wait, 1),
        "total_min": round(total_travel + total_wait, 1),
        "mean_crowd_pct": mean_crowd,
        "max_crowd_pct": round(max(all_crowd) if all_crowd else 0.0, 1),
        "crowd_level": crowd_level_label(mean_crowd),
        "generalized_cost": round(cost, 1),
        "hotspots": hotspots[:4],
        "live_adjusted": live_headway_ratio is not None,
    }


def build_options(engine: ScoringEngine, scored: list[dict],
                  dep_hour: int, dep_min: int, day_of_week: int,
                  live_headway_ratio: float | None = None) -> dict:
    """Turn scored candidates into the three blueprint commuter options plus
    ranked alternatives (each option tagged with its own type label)."""
    if not scored:
        return {"options": [], "wait_1_bus": None}

    by_time = sorted(scored, key=lambda x: x["total_min"])
    by_cost = sorted(scored, key=lambda x: x["generalized_cost"])

    fastest = by_time[0]
    fastest["option_type"] = "FASTEST"

    comfortable = None
    for cand in by_cost:
        if cand["key"] != fastest["key"] and cand["mean_crowd_pct"] < fastest["mean_crowd_pct"] - 3:
            comfortable = cand
            break
    if comfortable is None:
        comfortable = by_cost[0] if by_cost[0]["key"] != fastest["key"] else (
            by_cost[1] if len(by_cost) > 1 else None
        )
    if comfortable is not None:
        comfortable["option_type"] = "MOST_COMFORTABLE"

    # Wait-1-Bus: does the fastest route get meaningfully quieter 15 min later?
    # Trigger = crush relief (max crowd down >= 10 pts AND mean down >= 4 pts)
    # or a large overall drop (mean down >= 8 pts).
    wait_opt = None
    for base in (fastest, by_cost[0]):
        later = score_candidate(
            engine, _clone_candidate(base), (dep_hour * 60 + dep_min + 15) // 60 % 24,
            (dep_min + 15) % 60, day_of_week, live_headway_ratio,
        )
        mean_drop = base["mean_crowd_pct"] - later["mean_crowd_pct"]
        max_drop = base["max_crowd_pct"] - later["max_crowd_pct"]
        if (max_drop >= 10.0 and mean_drop >= 4.0) or mean_drop >= 8.0:
            later["option_type"] = "WAIT_1_BUS"
            later["advice"] = (
                f"Waiting ~15 min drops crowding on bus {base['legs'][0]['bus_number']} "
                f"from {base['mean_crowd_pct']:.0f}% avg / {base['max_crowd_pct']:.0f}% peak "
                f"to {later['mean_crowd_pct']:.0f}% avg / {later['max_crowd_pct']:.0f}% peak."
            )
            later["wait_saves_pct"] = round(mean_drop, 1)
            wait_opt = later
            break

    chosen = {fastest["key"], comfortable["key"] if comfortable else None}
    alternatives = []
    for cand in by_cost:
        if cand["key"] in chosen or len(alternatives) >= 3:
            continue
        cand["option_type"] = "ALTERNATIVE"
        alternatives.append(cand)

    options = [o for o in (fastest, comfortable) if o is not None] + alternatives
    for o in options:
        o.setdefault("advice", _advice_text(o))
    return {"options": options, "wait_1_bus": wait_opt}


def _clone_candidate(scored: dict) -> dict:
    """Deep-enough copy of a scored option so re-scoring at a later slot does
    not overwrite the original crowd numbers."""
    import copy
    return copy.deepcopy({"key": scored["key"], "transfers": scored["transfers"],
                          "legs": scored["legs"]})


def _advice_text(option: dict) -> str:
    if option["hotspots"]:
        h = option["hotspots"][0]
        return (f"Gets heavily crowded at {h['stop_name']} (~{h['crowd_pct']:.0f}%). "
                f"Board earlier or consider a less crowded alternative.")
    if option["mean_crowd_pct"] <= 50:
        return "Seats usually available along this route at this time."
    return f"Standing room expected ({option['crowd_level'].lower()}) along this route."
