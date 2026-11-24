#!/usr/bin/env python3
"""
STEP B — Part 2: Train the LightGBM Crowdedness Engine.

Trains a LightGBM regressor (L1/MAE objective — robust to ticket outliers) on the
net on-board load table produced by scripts/preprocess_etm_data.py:

    data/processed/etm_stop_load.parquet

Outputs (all in data/processed/):
  * crowd_lgbm_model.txt          -> LightGBM booster used by the FastAPI app
  * model_meta.json               -> feature layout + categories + validation metrics
  * hist_crowd_lookup.parquet     -> leak-free historical (route, dir, stop, weekend) crowd means
  * hist_slot_crowd.parquet       -> historical (route, dir, weekend, slot) crowd means
                                     (also powers model-less fallback scoring)
  * validation_peak_offpeak.png   -> predicted vs actual crowd curves for the technical note

Validation is a STRICT TEMPORAL SPLIT (no random shuffling): the model trains on the
first ~80% of service dates and is scored on the final held-out dates/week.

IMPORTANT: the FastAPI app loads these artifacts ONLY at server startup. After a
successful run you MUST restart uvicorn (Ctrl+C, start again) or /health keeps
showing "model_trained": false. This script verifies every artifact at the end
and prints the exact restart command.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# --- Friendly dependency guard ------------------------------------------------
# A raw ModuleNotFoundError traceback is the #1 "the script must be broken" report
# when in fact the venv was not active. Diagnose it up front, by name.
_MISSING: list[str] = []
try:
    import numpy as np
except ModuleNotFoundError as e:
    _MISSING.append(e.name)
try:
    import polars as pl
except ModuleNotFoundError as e:
    _MISSING.append(e.name)
try:
    import lightgbm as lgb
except ModuleNotFoundError as e:
    _MISSING.append(e.name)
try:
    import pandas as pd
except ModuleNotFoundError as e:
    _MISSING.append(e.name)
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error
except ModuleNotFoundError as e:
    _MISSING.append(f"scikit-learn (imports as {e.name})")
try:
    import matplotlib  # backend is selected inside make_validation_chart()
except ModuleNotFoundError as e:
    _MISSING.append(e.name)

if _MISSING:
    print("=" * 70)
    print("MISSING PYTHON PACKAGES:", ", ".join(_MISSING))
    print("=" * 70)
    print("This is NOT a script bug — those packages are simply not installed in the")
    print("Python that is running this command. Fix, in order:")
    print("  1) activate the project venv:")
    print("       Windows PowerShell:   .venv\\Scripts\\Activate.ps1")
    print("       macOS / Linux:        source .venv/bin/activate")
    print("  2) install everything:      pip install -r requirements.txt")
    print("  3) re-run this command.     (Still failing?  python scripts/doctor.py)")
    sys.exit(3)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED = BASE_DIR / "data" / "processed"
INPUT_PARQUET = PROCESSED / "etm_stop_load.parquet"

BUS_CAPACITY = 75.0                      # DTC practical capacity (40 seated + 35 standing)

# 15-minute slots: morning rush 08:30-10:30 = slots 34..41, evening 17:30-20:30 = 70..81,
# midday off-peak 13:00-15:30 = slots 52..61 (inclusive starts).
RUSH_SLOTS = set(range(34, 42)) | set(range(70, 82))
OFFPEAK_SLOTS = set(range(52, 62))

FEATURE_COLS = [
    "route_id", "direction_id", "day_of_week", "is_weekend",
    "slot_15m", "stop_idx", "norm_stop_pos",
    "hour_sin", "hour_cos", "is_rush_hour", "hist_route_stop_crowd",
]
CAT_COLS = ["route_id", "direction_id", "day_of_week", "is_weekend"]

# Every artifact the app's ScoringEngine (and the technical note) needs.
EXPECTED_ARTIFACTS = [
    ("crowd_lgbm_model.txt", "LightGBM booster"),
    ("model_meta.json", "feature layout + metrics"),
    ("hist_crowd_lookup.parquet", "historical stop means (fallback scoring)"),
    ("hist_slot_crowd.parquet", "historical slot means (Wait-1-Bus / fallback)"),
    ("validation_peak_offpeak.png", "validation chart for the technical note"),
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def add_base_features(df: pl.DataFrame) -> pl.DataFrame:
    """Feature engineering WITHOUT the historical-mean feature (that one must be
    computed strictly from the training split afterwards to avoid leakage)."""
    return df.with_columns([
        # normalized position along the canonical pattern (0..1); guard div-by-zero
        pl.when(pl.col("total_stops") > 1)
        .then(pl.col("stop_idx").clip(0) / (pl.col("total_stops") - 1))
        .otherwise(0.0).cast(pl.Float64).alias("norm_stop_pos"),
        (np.pi * 2.0 * pl.col("slot_15m") / 96.0).sin().cast(pl.Float64).alias("hour_sin"),
        (np.pi * 2.0 * pl.col("slot_15m") / 96.0).cos().cast(pl.Float64).alias("hour_cos"),
        pl.col("slot_15m").is_in(sorted(RUSH_SLOTS)).cast(pl.Int8).alias("is_rush_hour"),
        (pl.min_horizontal(
            pl.lit(100.0),
            pl.col("net_load") / pl.lit(BUS_CAPACITY) * 100.0,
        )).cast(pl.Float64).alias("crowd_pct"),
    ])


def attach_hist(df: pl.DataFrame, hist_lookup: pl.DataFrame, fallback_mean: float) -> pl.DataFrame:
    """Join the TRAIN-SPLIT-ONLY historical stop means (leak-free) and fill any
    unseen (route, dir, stop, weekend) combination with the train global mean."""
    return (
        df.join(hist_lookup, on=["route_id", "direction_id", "stop_idx", "is_weekend"],
                how="left")
        .with_columns(
            pl.col("hist_crowd_pct").fill_null(fallback_mean).alias("hist_route_stop_crowd")
        )
        .drop("hist_crowd_pct")
    )


def temporal_split(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """Split by service date (first ~80% train, rest test). Falls back to a
    (date, slot) split when the dataset spans too few dates for a clean week split."""
    dates = df["trip_date"].unique().sort().to_list()
    if len(dates) >= 3:
        cutoff = dates[int(len(dates) * 0.8)]
        train = df.filter(pl.col("trip_date") < cutoff)
        test = df.filter(pl.col("trip_date") >= cutoff)
        note = f"date-level split at {cutoff} ({len(dates[:int(len(dates) * 0.8)])} train dates / {len(dates) - len(dates[:int(len(dates) * 0.8)])} test dates)"
        return train, test, note
    combos = df.select(["trip_date", "slot_15m"]).unique().sort(["trip_date", "slot_15m"])
    k = int(combos.height * 0.8)
    cut_date, cut_slot = combos["trip_date"][k], combos["slot_15m"][k]
    train = df.filter(
        (pl.col("trip_date") < cut_date)
        | ((pl.col("trip_date") == cut_date) & (pl.col("slot_15m") < cut_slot))
    )
    test = df.filter(
        (pl.col("trip_date") > cut_date)
        | ((pl.col("trip_date") == cut_date) & (pl.col("slot_15m") >= cut_slot))
    )
    return train, test, f"(date,slot)-level split at {cut_date} slot {cut_slot} — only {len(dates)} date(s) in data"


def to_lgb_frames(train: pl.DataFrame, test: pl.DataFrame):
    """Build pandas frames whose categorical columns share EXACTLY the training
    category list (LightGBM maps categories to codes by position — a mismatch
    would silently corrupt predictions)."""
    X_tr = train.select(FEATURE_COLS).to_pandas()
    X_te = test.select(FEATURE_COLS).to_pandas()
    for c in CAT_COLS:
        cats = sorted(X_tr[c].astype(str).unique().tolist())
        X_tr[c] = pd.Categorical(X_tr[c].astype(str), categories=cats)
        X_te[c] = pd.Categorical(X_te[c].astype(str), categories=cats)
    return X_tr, X_te


def make_validation_chart(test: pl.DataFrame, preds: np.ndarray, degraded: bool) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = test.with_columns(pl.Series("pred", preds))
    subtitle = " (proxy load — GTFS alignment pending)" if degraded else ""

    overall = t.group_by("hour").agg(
        pl.col("crowd_pct").mean().alias("actual"),
        pl.col("pred").mean().alias("predicted"),
    ).sort("hour")

    # Pick the busiest test corridor for the per-route panel.
    top_route = (
        t.group_by("route_id").agg(pl.len().alias("n")).sort("n", descending=True)
    )["route_id"][0]
    route_curve = t.filter(pl.col("route_id") == top_route).group_by("hour").agg(
        pl.col("crowd_pct").mean().alias("actual"),
        pl.col("pred").mean().alias("predicted"),
    ).sort("hour")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    for ax, curve, title in (
        (axes[0], overall, "All routes — held-out test set"),
        (axes[1], route_curve, f"Corridor focus — route {top_route}"),
    ):
        hours = curve["hour"].to_numpy()
        ax.plot(hours, curve["actual"].to_numpy(), "o-", color="#1d4ed8", lw=2, label="Actual crowd %")
        ax.plot(hours, curve["predicted"].to_numpy(), "s--", color="#dc2626", lw=2, label="LightGBM predicted %")
        ax.axvspan(8.5, 10.5, color="#f59e0b", alpha=0.18, label="Morning rush")
        ax.axvspan(17.5, 20.5, color="#ef4444", alpha=0.15, label="Evening rush")
        ax.set_xlabel("Hour of day")
        ax.set_ylabel("Mean crowd % (capacity 75)")
        ax.set_title(title)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Temporal Validation — Predicted vs Actual Crowding{subtitle}", fontsize=12)
    out = PROCESSED / "validation_peak_offpeak.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def train() -> int:
    t0 = time.perf_counter()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    if not INPUT_PARQUET.exists():
        log("=" * 70)
        log("etm_stop_load.parquet NOT FOUND — run Step A first:")
        log("  1) drop your e-ticketing file(s) into  data/raw/etm/")
        log("     (expected columns: passenger_count, amount, vehicle_number,")
        log("      ticket_type_id, end_stop_name, route, start_stop_name,")
        log("      created_at, is_ac — run `python scripts/preprocess_etm_data.py`)")
        log("  2) python scripts/preprocess_etm_data.py")
        log("Then re-run:  python scripts/train_part2_crowd_model.py")
        log("=" * 70)
        return 1

    df = pl.read_parquet(INPUT_PARQUET)
    if df.height < 20:
        log(f"[ERROR] Only {df.height} aggregated rows in etm_stop_load.parquet — too few to train.")
        return 2

    degraded = bool(df["aligned"].min() == 0) if "aligned" in df.columns else True
    if degraded:
        log("[WARN] Data is in DEGRADED mode (no GTFS index yet): the model learns "
            "route-level demand, not true on-board load. Run build_part1_index.py + "
            "preprocess_etm_data.py again for stop-level crowding.")
    df = df.sort(["trip_date", "slot_15m", "route_id", "direction_id"])

    # ---- Temporal split FIRST, then train-split-only historical means --------
    train_df, test_df, split_note = temporal_split(df)
    log(f"Temporal split: {split_note}")
    log(f"  train rows: {train_df.height:,} | test rows: {test_df.height:,}")

    train_df = add_base_features(train_df)
    test_df = add_base_features(test_df)
    hist = (
        train_df.group_by(["route_id", "direction_id", "stop_idx", "is_weekend"])
        .agg(pl.col("crowd_pct").mean().alias("hist_crowd_pct"))
    )
    global_mean = float(train_df["crowd_pct"].mean() or 35.0)
    train_df = attach_hist(train_df, hist, global_mean)
    test_df = attach_hist(test_df, hist, global_mean)

    X_tr, X_te = to_lgb_frames(train_df, test_df)
    y_tr = train_df["crowd_pct"].to_numpy()
    y_te = test_df["crowd_pct"].to_numpy()

    log(f"Training LightGBM (L1 objective) on {X_tr.shape[1]} features ...")
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=30,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, categorical_feature=CAT_COLS)

    preds = np.clip(model.predict(X_te), 0.0, 100.0)

    # ---- Metrics: overall + peak vs off-peak ----------------------------------
    mae = mean_absolute_error(y_te, preds)
    rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
    slots_te = test_df["slot_15m"].to_numpy()
    rush_mask = np.isin(slots_te, sorted(RUSH_SLOTS))
    off_mask = np.isin(slots_te, sorted(OFFPEAK_SLOTS))
    peak_mae = mean_absolute_error(y_te[rush_mask], preds[rush_mask]) if rush_mask.any() else float("nan")
    off_mae = mean_absolute_error(y_te[off_mask], preds[off_mask]) if off_mask.any() else float("nan")

    log("")
    log("=" * 70)
    log("TEMPORAL VALIDATION RESULTS (held-out final dates)")
    log("=" * 70)
    log(f"  Overall : MAE {mae:.2f}% | RMSE {rmse:.2f}%  (crowd % of 75-pax capacity)")
    log(f"  Peak    : MAE {peak_mae:.2f}%   (08:30-10:30 & 17:30-20:30)")
    log(f"  Off-peak: MAE {off_mae:.2f}%   (13:00-15:30)")

    # Sanity check of the learned curve: rush should exceed midday.
    act_rush = float(y_te[rush_mask].mean()) if rush_mask.any() else float("nan")
    act_off = float(y_te[off_mask].mean()) if off_mask.any() else float("nan")
    if not np.isnan(act_rush) and not np.isnan(act_off):
        log(f"  Actual mean crowd: peak {act_rush:.1f}% vs off-peak {act_off:.1f}% "
            f"({'OK - peak higher' if act_rush > act_off else 'WARNING - peak NOT higher; check your data time column'})")

    booster = model.booster_
    booster.save_model(str(PROCESSED / "crowd_lgbm_model.txt"))

    # ---- Persist metadata + leak-free lookups ---------------------------------
    # pandas Categorical exposes its (sorted) category list directly; scoring
    # must rebuild inference frames with exactly these categories in this order.
    categories = {c: [str(x) for x in X_tr[c].cat.categories.tolist()] for c in CAT_COLS}
    meta = {
        "feature_cols": FEATURE_COLS,
        "cat_cols": CAT_COLS,
        "cat_categories": categories,
        "capacity": BUS_CAPACITY,
        "global_mean_crowd": global_mean,
        "degraded": degraded,
        "metrics": {
            "mae": round(mae, 3), "rmse": round(rmse, 3),
            "peak_mae": round(float(peak_mae), 3) if not np.isnan(peak_mae) else None,
            "offpeak_mae": round(float(off_mae), 3) if not np.isnan(off_mae) else None,
            "train_rows": int(train_df.height), "test_rows": int(test_df.height),
            "split_note": split_note,
        },
        "importance": {
            f: round(float(v), 1) for f, v in sorted(
                zip(FEATURE_COLS, booster.feature_importance("gain")), key=lambda x: -x[1]
            )
        },
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (PROCESSED / "model_meta.json").write_text(json.dumps(meta, indent=2))

    hist.select([
        "route_id", "direction_id", "stop_idx", "is_weekend",
        pl.col("hist_crowd_pct").round(2),
    ]).write_parquet(PROCESSED / "hist_crowd_lookup.parquet", compression="zstd")

    slot_hist = (
        train_df.group_by(["route_id", "direction_id", "is_weekend", "slot_15m"])
        .agg(pl.col("crowd_pct").mean().round(2).alias("hist_crowd_pct"))
    )
    slot_hist.write_parquet(PROCESSED / "hist_slot_crowd.parquet", compression="zstd")

    chart_path = make_validation_chart(test_df, preds, degraded)

    top5 = list(meta["importance"].items())[:5]
    log(f"  Top features by gain: {', '.join(f'{k} ({v:.0f})' for k, v in top5)}")

    # ---- Final artifact self-check + the step everyone forgets ----------------
    log("")
    log("  [VERIFY] artifacts the app needs, in data/processed/:")
    all_ok = True
    for name, what in EXPECTED_ARTIFACTS:
        p = PROCESSED / name
        if p.exists() and p.stat().st_size > 0:
            log(f"    [OK]   {name:<28} {p.stat().st_size / 1024:8.0f} KB  ({what})")
        else:
            log(f"    [FAIL] {name:<28} missing/empty  ({what})")
            all_ok = False
    log(f"  Train time: {time.perf_counter() - t0:.1f}s")

    if all_ok:
        log("")
        log("=" * 70)
        log("[DONE] Crowdedness engine READY. Now RESTART the server — artifacts are")
        log("       loaded ONLY at startup (uvicorn --reload watches .py files, not data):")
        log("           Ctrl+C on the uvicorn window, then:")
        log("           uvicorn app.main:app --reload")
        log("       Then http://127.0.0.1:8000/health must show \"model_trained\": true,")
        log("       and the UI header changes from 'Engine: NOT trained' to 'Engine: READY'.")
        log("=" * 70)
        return 0

    log("[ERROR] Some artifacts failed to write — scroll up for the traceback.")
    return 4


if __name__ == "__main__":
    sys.exit(train())
