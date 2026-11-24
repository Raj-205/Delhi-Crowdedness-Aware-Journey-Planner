# Delhi Crowdedness-Aware Bus Journey Planner

**Transport Stack Hackathon 2026 — Problem 2** · Crowdedness-aware route ranking trained on your real ETM / e-ticketing data, with live bus tracking (GTFS-RT **or** zero-config demo tracking) and a zero-key Leaflet map.

> **What it does:** you pick any two Delhi bus stops + a departure time → the app returns a **Fastest Route**, a **Most Comfortable / less-crowded path**, an optional **Wait-1-Bus** advice, stop-by-stop crowding bars, **hotspot alerts** (e.g. *"Heads up — usually busy at ITO (~86%)"*), and a live map with moving bus pins when you depart now.

---

## ✅ What is already inside this zip (no setup needed for a first look)

| Item | Status |
| --- | --- |
| `data/raw/gtfs/` | **Your GTFS bundle is bundled** (the 5 files: `routes.txt`, `trips.txt`, `stops.txt`, `stop_times.txt`, `shapes.txt`) |
| `data/processed/routing_index.pkl` | Built from that GTFS — 10,000 stops, 10,000 routes, 2,000 directional patterns, 2,000 shape polylines |
| `data/processed/crowd_lgbm_model.txt` + friends | Crowd engine **trained end-to-end at the 10-lakh-row scale** on a schema-verification ETM file. It works out of the box, but **re-train on YOUR real ETM file** (Step 2) so the crowd numbers reflect your ridership |
| `data/raw/etm/` | **Empty by design** — drop your real ETM file here (Step 2) |
| Demo / synthetic data | **Removed.** No demo generator ships with this zip |

You can start the server immediately and everything works; replacing the trained artifacts with your own data is Step 2 → Step 3 below.

---

## Step 0 — One-time setup (any OS)

```bash
cd delhi-crowd-planner
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:      source .venv/bin/activate
pip install -r requirements.txt
```

**Check:** `python scripts/doctor.py` → `[OK] Python`, `[OK] dependencies installed`, `[OK] GTFS raw data: [routes.txt, shapes.txt, ...]`.

## Step 1 — GTFS (already bundled, rebuild only if you swap the files)

The GTFS files sit in `data/raw/gtfs/`. If you ever get a newer bundle, drop its 5 files (or the whole `.zip`) here and rebuild:

```bash
python scripts/build_part1_index.py
```

**Check:** the last lines must show

```
Stops indexed          : 10,000
Bus routes             : 10,000
Directional patterns   : 2,000 (0 without usable times)
Route shape polylines  : 2,000
```

*(Any numbers are fine — these match the bundled feed. If you change the GTFS, re-run Step 2 and Step 3 afterwards.)*

## Step 2 — Your real ETM file (10-lakh rows) → crowd training table

Drop your ETM export into `data/raw/etm/` (`.csv`, `.parquet`, or `.zip`). The expected layout — **auto-detected, no code edits**:

| Column | Meaning | Used for |
| --- | --- | --- |
| `passenger_count` | passengers on the ticket | on-board load (clamped 1–12) |
| `amount` | fare in ₹ | UI insights: revenue, avg fare, fare tiers |
| `vehicle_number` | bus plate | UI insights: per-vehicle tickets |
| `ticket_type_id` | ticket type code | UI insights: ticket-type mix |
| `end_stop_name` | alighting stop name | exact alighting subtraction (Case A) |
| `route` | route number (e.g. `534`, `543A`) | route → GTFS `route_id` match |
| `start_stop_name` | boarding stop name | stop-name alignment to GTFS (fuzzy) |
| `created_at` | ticket timestamp | service date + 15-min slot (rush-hour learning) |
| `is_ac` | AC / Non-AC | UI insights: AC split |

**Known portal formats — all tested, no edits needed:**

- `created_at` = `2026-10-28 10:19:18.780611+05:30` (microseconds + IST offset, the **October-Delhi.csv** style) — parsed via offset-aware formats + numeric-offset strip. Also handles `+0530`, `Z`, `IST`, plain, Excel serials and epoch.
- `is_ac` = `True` / `False` (or `1`/`0`, `AC`) — all accepted.
- `vehicle_number` = literal `NULL` — converted to a real null (never shows up as a bus named "NULL").

```bash
python scripts/preprocess_etm_data.py --peek    # 1) verify the detected column map
python scripts/preprocess_etm_data.py           # 2) build the net-load training table
```

**Check the `--peek` output** — it must show exactly this mapping (the 4 commerce columns appear as "ignored" here but are captured for the insights panel):

```
timestamp        <-  'created_at'
route_id         <-  'route'
boarding_stop    <-  'start_stop_name'
alighting_stop   <-  'end_stop_name'
passenger_count  <-  'passenger_count'
```

**Check the full run** — you want ~100% resolution like the verification run:

```
Routes resolved to GTFS: 1000/1000
Boarding stops resolved: 18225/18225 unique names; alighting resolved: 17360/17360
Case A (exact alighting subtraction): 573,050 rows | Case B (stop-decay survival): 426,950 rows
```

At 10 lakh rows this takes ~1 minute. If resolution is low (<70%), your stop names may use a different spelling style — run `--peek` and paste the sample rows back for matcher tuning.

> **Important — ETM and GTFS must describe the SAME network.** Your real export
> (`October-Delhi.csv`) carries route codes like `8UP`, `433UP` and real stop names
> like *Andrews Ganj*, *Kalindi Kunj Bdr*. The GTFS **bundled in this zip** is the
> Transport Stack workshop bundle (routes `R534`-style, stop names like
> *Munirka Corridor Stop 1*) — it does NOT describe the same network. Preprocessing
> will still run end-to-end (that is exactly what the degraded path is for: unresolved
> routes/stops are warned about and excluded, e.g. `Routes resolved to GTFS: 3/4` on
> the mixed test file), but the crowd engine can only learn from the rows that align.
> For full stop-level accuracy, drop in the GTFS that matches your ETM network
> (Step 1) and re-run Steps 2 → 3.

## Step 3 — Train the crowd engine

```bash
python scripts/train_part2_crowd_model.py
```

**Check** (verification-run reference numbers; yours will differ with real data):

```
Temporal split: date-level split at 2026-11-17 (28 train dates / 7 test dates)
Overall : MAE 1.43% | RMSE 2.35%  (crowd % of 75-pax capacity)
Actual mean crowd: peak 2.2% vs off-peak 2.0% (OK - peak higher)
[DONE] Crowdedness engine READY...
```

- **"OK - peak higher"** is the important sanity line: rush hours must be busier than midday.
- Artifacts written: `crowd_lgbm_model.txt`, `model_meta.json`, `hist_crowd_lookup.parquet`, `hist_slot_crowd.parquet`, `validation_peak_offpeak.png` (chart for your technical note).

## Step 4 — Start the app

```bash
uvicorn app.main:app --reload        # run from the delhi-crowd-planner/ folder
```

Open **http://127.0.0.1:8000** (API docs: `/docs`, health: `/health`).

**Check:** `/health` shows `"model_trained": true`, `"stops": 10000`, `"directional_patterns": 2000`, and the UI header pills read *Crowd engine trained*.

---

## Live tracking — the two modes (15 s background poller)

| | **Mode A — official GTFS-RT** | **Mode B — demo / zero-config** (default) |
| --- | --- | --- |
| When | `TRANSPORT_STACK_API_KEY` (or `GTFS_RT_VEHICLE_URL`) is set | no key configured |
| Source | real `VehiclePositions.pb` decoded from the portal feed | 2–3 synthesized buses **per route** with Delhi plates (`DL1PC####`), driving along the route's **real shape polyline** with realistic speeds (10–42 km/h, traffic wobble) |
| Label | `tracking_mode: LIVE_GTFS_RT` | `tracking_mode: SIMULATED_FEED` — every bus carries `source: SIMULATED`, `/health` shows `mode: SIMULATED` |
| Everything else | identical: Haversine stop-snapping, ETA `max(1, dist_km / max(speed, 16) × 60)`, headway-bunching multiplier `clip(gap / scheduled, 0.85, 1.35)`, graceful historical fallback | identical |

**Auto-fallback:** if Mode A is configured but the feed fails or returns empty for
3 consecutive polls (bad key, portal down, no vehicles published), the poller
automatically switches to Mode B so the map ALWAYS shows buses, and `/health`
reports `mode: SIMULATED` with the feed error in `last_error`.

```bash
# enable Mode A (optional):
export TRANSPORT_STACK_API_KEY="your-key"        # Windows: $env:TRANSPORT_STACK_API_KEY="..."
uvicorn app.main:app --reload --env-file .env    # or put it in .env

# disable demo buses entirely:
export SIMULATE_BUSES=0
```

**Check Mode B right after startup** (server log):

```
[Mode B] Zero-config demo tracking ON — synthesizing ~5000 buses across 2000 directional routes
```

**Check via API:**

```bash
curl "http://127.0.0.1:8000/track-route?route_id=R534&direction_id=0&mode=NOW&origin_stop_id=<any stop on the route>"
#  -> "tracking_mode": "SIMULATED_FEED", buses with DL1PC plates, ETA to your stop
```

If a future query finds no buses (or you set `SIMULATE_BUSES=0`), the endpoint still answers 200 with `HISTORICAL_FORECAST_FALLBACK` — schedule + ETM crowd profile, never a crash.

---

## Detailed verification checklist (copy-paste, ~2 minutes)

Run the server, then from another terminal:

| # | Check | Command | Expected |
| --- | --- | --- | --- |
| 1 | Health | `curl http://127.0.0.1:8000/health` | `model_trained: true`, `stops: 10000`, `live_feed.mode: SIMULATED` (or GTFS-RT with a key) |
| 2 | Stop autocomplete | `curl "http://127.0.0.1:8000/stops?q=iffco"` | stops list with names + ids |
| 3 | Pick a route + its stops | `curl "http://127.0.0.1:8000/route-crowding/R534?direction_id=0&hour=9"` | stop list with `crowd_pct`, `mean_crowd_pct` |
| 4 | Plan a journey | `curl "http://127.0.0.1:8000/plan-journey?origin=<first stop from #3>&dest=<last stop>&hour=9&minute=0"` | FASTEST option, `crowd_level`, generalized cost; `data_mode: LIVE_ADJUSTED` when demo buses are on leg 1 |
| 5 | No-route guidance | `curl "http://127.0.0.1:8000/plan-journey?origin=H0134&dest=H0311"` | `options: []` + `suggestions` (stops reachable from the origin) |
| 6 | Reachable stops | `curl "http://127.0.0.1:8000/connected-stops?stop_id=H0134&limit=6"` | direct + one-change stops with names |
| 7 | Live tracking | `curl "http://127.0.0.1:8000/track-route?route_id=R534&direction_id=0&mode=NOW&origin_stop_id=H0134"` | `SIMULATED_FEED` + buses snapped to stops with ETA |
| 8 | Raw cache | `curl http://127.0.0.1:8000/live-buses/R534` | same buses, raw lat/lon/speed |
| 9 | ETM insights | `curl http://127.0.0.1:8000/etm-insights` | totals (tickets, revenue), ticket types, AC split, top vehicles, per-route table |
| 10 | UI | open `http://127.0.0.1:8000`, pick two stops, press **Find buses** | cards on the left, map on the right; with *Leave now* the live strip fills and bus dots move on the map every 15 s |
| 11 | Any missing/stale artifact? | `python scripts/doctor.py` | `VERDICT: ALL GOOD` (it also warns when your raw ETM is newer than the training table) |

*(For #4/#7, take the stop ids from #3 or #6 — a journey only ranks when both stops sit on a common route.)*

## Retraining when your real file changes

```bash
# drop the new ETM file into data/raw/etm/ (or swap the GTFS in data/raw/gtfs/), then:
python scripts/build_part1_index.py      # only if the GTFS changed
python scripts/preprocess_etm_data.py
python scripts/train_part2_crowd_model.py
# restart uvicorn (Ctrl+C, start again) — artifacts load only at startup
```

Every run regenerates `data/processed/` from whatever is currently in `data/raw/` — no code edits, ever.

---

## How the crowding model works (technical note)

1. **Net on-board load** — tickets record boardings; the pipeline reconstructs `Load(k) = Σ_{i≤k} Boardings_i − Σ_{j≤k} Alightings_j` per `(route_id, direction_id, service_date, 15-min slot)`:
   * **Case A** — `end_stop_name` resolves ahead of the boarding stop on the GTFS pattern → subtracted exactly there.
   * **Case B** — destination missing/unresolvable → calibrated **stop-decay survival curve** (exponential, mean trip ≈ 35% of the route's stop span, clamped 4–30 stops — typical Delhi rides of ~6–9 km). Documented assumption.
2. **Normalization** — `Crowd% = min(100, Load / 75 × 100)` (DTC practical capacity ≈ 40 seated + 35 standing).
3. **Strict dimension separation** — direction is inferred from the OD pair against the GTFS patterns (the ETM file has no direction column); inbound peaks never average out with outbound.
4. **Leak-free features** — the historical `(route, dir, stop, weekend)` mean is computed only inside the training split.
5. **Temporal validation** — train on the first ~80% of service dates, test on held-out final dates (never shuffled). Peak vs off-peak MAE reported separately.
6. **Generalized Discomfort Cost** — `T_travel + 1.6·T_wait + 0.35·Σ T_s·M(Crowd_s)` with tiers (≤50% → 1.0×, ≤75% → 1.25×, ≤90% → 1.65×, >90% → 2.2×).
7. **Traffic calibration** — GTFS durations × time-of-day multipliers (morning ≈ 1.38×, evening ≈ 1.42×, late night ≈ 0.95×).
8. **Live fusion** — "depart now" queries scale historical crowd by `clip(actual_bus_gap / scheduled_headway, 0.85, 1.35)` from the 15 s tracking cache.

## Project structure

```text
delhi-crowd-planner/
├── requirements.txt
├── README.md                         # this file
├── data/
│   ├── raw/
│   │   ├── etm/.gitkeep              # <--- DROP your real ETM file here (Step 2)
│   │   └── gtfs/                     # your GTFS bundle (bundled; swap any time)
│   └── processed/                    # artifacts (regenerable): routing_index.pkl,
│                                     #   etm_stop_load.parquet, crowd_lgbm_model.txt,
│                                     #   model_meta.json, hist_* lookups, etm_insights.json,
│                                     #   preprocess_report.json, validation chart
├── scripts/
│   ├── preprocess_etm_data.py        # Step A: auto column-map + net-load reconstruction
│   ├── train_part2_crowd_model.py    # Step B: LightGBM + temporal split + validation chart
│   ├── build_part1_index.py          # Step C: GTFS -> routing_index.pkl
│   └── doctor.py                     # self-check: prints the exact missing step
├── app/
│   ├── rt_tracker.py                 # 15 s poller: Mode A GTFS-RT / Mode B simulated buses,
│   │                                 #   RAM cache, Haversine snap, ETA, bunching ratio
│   ├── scoring_engine.py             # crowd %, hotspots, generalized-cost ranking
│   └── main.py                       # FastAPI server uniting everything + static UI
└── static/
    └── index.html                    # responsive UI: search, result cards, always-visible
                                      #   live map with bus dots + ETA strip, stop-by-stop
                                      #   drawer, reachable-stop suggestions (no data panels)
```

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `NO E-TICKETING FILES FOUND` | File must sit in `data/raw/etm/` with `.csv`/`.parquet`/`.zip` extension |
| Timestamp error `failed to parse in ALL rows` | Should no longer happen (the `+05:30`-offset portal format is supported). If it does, the error prints your exact raw values — send them back and the format gets added |
| Wrong column detected | `--peek`, then add your header to `COLUMN_MAP` / `USER_OVERRIDES` at the top of `preprocess_etm_data.py` |
| Low route/stop resolution | See the ETM/GTFS mismatch note in Step 2 — your real route codes (`8UP`, `433UP`) need the GTFS that matches YOUR network |
| Doctor says ETM file is NEWER than the training table | The last preprocess run failed and old artifacts were used. Fix the `[ERROR]` preprocess prints, re-run preprocess + train |
| `Routing index not built yet` (503) | `python scripts/build_part1_index.py` |
| `Crowdedness engine not trained yet` (503) | Steps 2 + 3 |
| Crowd % flat / unrealistic | Confirm `created_at` parsed (see `data/processed/preprocess_report.json`, `time_mode`) and that the training log says "peak higher" |
| **No buses on the map** | Plan a journey with **Leave now** checked — tracking starts automatically on the main map (strip above the map, blue bus dots, refreshed every 15 s). Check `/health` → `live_feed.mode`: `SIMULATED` needs the routing index built; `GTFS-RT` needs a valid key and auto-falls back to `SIMULATED` after 3 failed polls. If you got "No bus route connects these two stops", use the suggestion chips (or `/connected-stops`) to pick a pair that shares a route. Disable demo buses with `SIMULATE_BUSES=0` |
| Map tiles blank | Tiles come from the internet; offline judges see geometry only |
| Port busy | `uvicorn app.main:app --reload --port 8001` |
| Anything else | `python scripts/doctor.py` — pure stdlib, prints the exact fix |

## Scope boundaries (per the problem statement)

- ✅ Multi-criteria planner (time + crowd + wait), hotspot alerts, live tracking, responsive web UI.
- ❌ No ticket booking — DTC ticketing is handled by Chartr / One Delhi.
- ❌ No "notify the conductor" feature — no such API exists.
- ❌ No runtime LLM for crowd numbers — LightGBM tabular ML only, fully auditable.

## Attribution

Open-source libraries: [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org), [Polars](https://pola.rs), [PyArrow](https://arrow.apache.org), [LightGBM](https://lightgbm.readthedocs.io), [scikit-learn](https://scikit-learn.org), [Matplotlib](https://matplotlib.org), [httpx](https://www.python-httpx.org), [gtfs-realtime-bindings](https://github.com/MobilityData/gtfs-realtime-bindings), [Leaflet](https://leafletjs.com) — map tiles © [OpenStreetMap](https://www.openstreetmap.org) contributors.
