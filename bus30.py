# /// script
# requires-python = ">=3.10"
# dependencies = ["gtfs-realtime-bindings>=1.0"]
# ///
"""Live view of a Limassol bus route (default: 30) from the Cyprus Motion open feeds.

Sources (the same ones github.com/tonyzorin/cyprus-bus uses):
  - static GTFS (routes, stops, stop_times, trips) for Limassol, cached for a day;
  - GTFS-Realtime: VehiclePosition (lat/lon/speed/bearing, ~30 s per bus) and
    TripUpdate (predicted arrival per remaining stop + delay vs schedule).

Run:
  uv run bus30.py                 # snapshot of all route-30 buses
  uv run bus30.py --stop "Old Port"  # next arrivals at a stop (name or stop_id)
  uv run bus30.py --watch 15 --log data/positions.csv
  uv run bus30.py --classify data/positions.csv
      # label every trip x stop: stopped / likely_stopped / likely_passed / unknown, plus runs of
      # consecutive passes (out of service or full) with a confidence; writes <csv>_passes.csv
  uv run bus30.py --serve 8730 --log "data/positions_{date}.csv"
      # live map: stops, active buses, chance each bus stops at its next stop; page refreshes every 30 s.
      # {date} in --log is replaced by the Cyprus date (one file per day); the server warms up from today's file.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import threading
import time
import urllib.request
import zipfile
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zoneinfo import ZoneInfo

from google.transit import gtfs_realtime_pb2

RT_URL = "http://20.19.98.194:8328/Api/api/gtfs-realtime"
STATIC_URL = "https://motionbuscard.org.cy/opendata/downloadfile?file=GTFS%5C6_google_transit.zip&rel=True"
TZ = ZoneInfo("Asia/Nicosia")
DATA_DIR = Path(os.environ.get("BUS30_DATA_DIR", Path(__file__).resolve().parent / "data"))
CACHE_DIR = DATA_DIR / "gtfs"
AT_STOP_M = 40  # closer than this to a stop = "at the stop"


def fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def read_csv(z: zipfile.ZipFile, name: str) -> list[dict]:
    text = z.read(name).decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def load_static(route_short_name: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    zpath = CACHE_DIR / "limassol_gtfs.zip"
    if not zpath.exists() or time.time() - zpath.stat().st_mtime > 86400:
        zpath.write_bytes(fetch(STATIC_URL, timeout=120))
    with zipfile.ZipFile(zpath) as z:
        routes = {r["route_id"]: r for r in read_csv(z, "routes.txt") if r["route_short_name"] == route_short_name}
        if not routes:
            sys.exit(f"route {route_short_name!r} not found in Limassol GTFS")
        trips = {t["trip_id"]: t for t in read_csv(z, "trips.txt") if t["route_id"] in routes}
        stop_seq: dict[str, list[tuple[int, str, str]]] = {}
        for st in read_csv(z, "stop_times.txt"):
            if st["trip_id"] in trips:
                stop_seq.setdefault(st["trip_id"], []).append((int(st["stop_sequence"]), st["stop_id"], st["arrival_time"]))
        for seq in stop_seq.values():
            seq.sort()
        used = {sid for seq in stop_seq.values() for _, sid, _ in seq}
        stops = {s["stop_id"]: s for s in read_csv(z, "stops.txt") if s["stop_id"] in used}
        shape_ids = {t["shape_id"] for t in trips.values()}
        shape_pts: dict[str, list[tuple[int, float, float]]] = {}
        for p in read_csv(z, "shapes.txt"):
            if p["shape_id"] in shape_ids:
                shape_pts.setdefault(p["shape_id"], []).append(
                    (int(p["shape_pt_sequence"]), float(p["shape_pt_lat"]), float(p["shape_pt_lon"])))
        shapes = {sid: Polyline([(la, lo) for _, la, lo in sorted(pts)]) for sid, pts in shape_pts.items()}
    return {"routes": routes, "trips": trips, "stop_seq": stop_seq, "stops": stops, "shapes": shapes}


class Polyline:
    """Route geometry in local metres; projects a point to distance-along-route."""

    def __init__(self, pts: list[tuple[float, float]]):
        self.pts = pts
        self.k = math.cos(math.radians(pts[0][0]))
        self.xy = [self._xy(la, lo) for la, lo in pts]
        self.cum = [0.0]
        for (x1, y1), (x2, y2) in zip(self.xy, self.xy[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x2 - x1, y2 - y1))

    def _xy(self, lat, lon):
        return lon * 111320 * self.k, lat * 110540

    def project(self, lat, lon, lo=-math.inf, hi=math.inf) -> tuple[float, float]:
        """(distance along route, lateral offset) of the closest point within [lo, hi] along the route."""
        px, py = self._xy(lat, lon)
        best = (math.inf, 0.0)
        for i in range(len(self.xy) - 1):
            if self.cum[i + 1] < lo or self.cum[i] > hi:
                continue
            (x1, y1), (x2, y2) = self.xy[i], self.xy[i + 1]
            dx, dy = x2 - x1, y2 - y1
            seg2 = dx * dx + dy * dy
            t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg2))
            d = math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))
            if d < best[0]:
                best = (d, self.cum[i] + t * math.sqrt(seg2))
        return best[1], best[0]


# ---- pass-through classification -------------------------------------------------------------
#
# Per stop: was the bus seen standing at it (speed 0 within NEAR_STOP_M)? Otherwise take the two
# position samples around the stop and compare the time between them with the time the same
# distance takes at the speed the bus showed: a real stop costs at least STOP_COST_S extra
# (braking + doors + pulling away). Little extra time across k stops => some of them were passed.
# Slow endpoints (traffic) and long gaps between samples shrink the estimate toward 50/50.
#
# Per trip: a two-state hidden Markov chain over consecutive stops — "in service" (passes a stop
# only when nobody waits, base_pass_rate) vs "skipping" (out of service / full, passes ~all
# stops). The state is sticky, so several likely passes in a row reinforce each other while a
# single one does not; forward-backward gives P(skipping) per stop.

NEAR_STOP_M = 35
OFF_ROUTE_M = 80
STOP_COST_S = 15
SKIP_PASS_RATE = 0.95


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def project_one(poly: Polyline, s: dict, prev: dict | None) -> dict | None:
    if prev is None:
        pos, lat_off = poly.project(s["lat"], s["lon"])
    else:
        dt = max(1, s["ts"] - prev["ts"])
        pos, lat_off = poly.project(s["lat"], s["lon"], prev["pos"] - 100, prev["pos"] + max(1500, 25 * dt))
    if lat_off > OFF_ROUTE_M:
        return None
    return {**s, "pos": max(pos, prev["pos"]) if prev else pos}


def project_samples(poly: Polyline, samples: list[dict]) -> list[dict]:
    out: list[dict] = []
    for s in samples:
        p = project_one(poly, s, out[-1] if out else None)
        if p:
            out.append(p)
    return out


def stop_positions(poly: Polyline, st, trip_id: str) -> list[dict]:
    res, prev = [], 0.0
    for seq, sid, arr in st["stop_seq"][trip_id]:
        s = st["stops"].get(sid)
        if not s:
            continue
        pos, _ = poly.project(float(s["stop_lat"]), float(s["stop_lon"]), prev - 20, prev + 4000)
        prev = max(prev, pos)
        res.append({"seq": seq, "stop_id": sid, "name": s["stop_name"], "sched": arr[:5], "pos": prev})
    return res


def classify_stops(stops: list[dict], smp: list[dict]) -> list[dict]:
    rows = []
    for st in stops:
        row = {**st, "label": "not_observed", "p_pass": None, "evidence": ""}
        rows.append(row)
        if not smp or st["pos"] < smp[0]["pos"] or st["pos"] > smp[-1]["pos"]:
            continue
        standing = [s for s in smp if abs(s["pos"] - st["pos"]) <= NEAR_STOP_M and s["speed"] == 0]
        if standing:
            row.update(label="stopped", p_pass=0.03, evidence=f"standing at stop {datetime.fromtimestamp(standing[0]['ts'], TZ):%H:%M:%S}")
            continue
        a = max((s for s in smp if s["pos"] <= st["pos"]), key=lambda s: s["ts"], default=None)
        b = min((s for s in smp if s["pos"] > st["pos"] and (a is None or s["ts"] > a["ts"])), key=lambda s: s["ts"], default=None)
        if a is None or b is None:
            row.update(label="unknown", p_pass=0.5, evidence="no samples on both sides")
            continue
        dt, ds = b["ts"] - a["ts"], b["pos"] - a["pos"]
        k = sum(1 for o in stops if a["pos"] < o["pos"] <= b["pos"])
        v_end = max(a["speed"], b["speed"])
        v_cruise = _clip(v_end, 15, 60) / 3.6
        excess = dt - ds / v_cruise
        n_stops_made = _clip(excess / STOP_COST_S, 0, k)
        raw = 1 - n_stops_made / k
        reliability = _clip((v_end - 10) / 20, 0, 1) * _clip((90 - dt) / 60, 0, 1)
        if ds / dt * 3.6 > v_end * 1.5 + 10:  # GPS jump / implausible
            reliability *= 0.5
        p = 0.5 + (raw - 0.5) * reliability
        row["p_pass"] = round(p, 2)
        row["label"] = "likely_passed" if p >= 0.7 else "likely_stopped" if p <= 0.3 else "unknown"
        row["evidence"] = (f"{dt}s for {ds:.0f} m over {k} stop(s), speeds {a['speed']}->{b['speed']} km/h, "
                           f"extra time {excess:+.0f}s")
    return rows


def skip_mode_posterior(rows: list[dict], base_pass: float, prior_skip: float, switch: float) -> list[float | None]:
    """Forward-backward over observed stops; returns P(skipping) per row (None = not observed)."""
    idx = [i for i, r in enumerate(rows) if r["p_pass"] is not None]
    if not idx:
        return [None] * len(rows)
    rates = (base_pass, SKIP_PASS_RATE)
    em = [[q * rows[i]["p_pass"] + (1 - q) * (1 - rows[i]["p_pass"]) for q in rates] for i in idx]
    tr = [[1 - switch, switch], [switch, 1 - switch]]
    fwd = []
    f = [(1 - prior_skip) * em[0][0], prior_skip * em[0][1]]
    for t in range(len(idx)):
        if t:
            f = [sum(fwd[-1][i] * tr[i][j] for i in range(2)) * em[t][j] for j in range(2)]
        z = sum(f)
        fwd.append([x / z for x in f])
    bwd = [[1.0, 1.0] for _ in idx]
    for t in range(len(idx) - 2, -1, -1):
        b = [sum(tr[i][j] * em[t + 1][j] * bwd[t + 1][j] for j in range(2)) for i in range(2)]
        z = sum(b)
        bwd[t] = [x / z for x in b]
    post = [None] * len(rows)
    for t, i in enumerate(idx):
        g = [fwd[t][j] * bwd[t][j] for j in range(2)]
        post[i] = round(g[1] / sum(g), 2)
    return post


def classify_log(st, log: Path, args) -> None:
    trips_samples: dict[tuple, list[dict]] = {}
    with log.open() as fh:
        for r in csv.DictReader(fh):
            if r["trip_id"] not in st["trips"]:
                continue
            ts = int(r["vehicle_ts"])
            day = datetime.fromtimestamp(ts, TZ).date().isoformat()
            trips_samples.setdefault((day, r["trip_id"], r["vehicle_id"]), []).append({
                "ts": ts, "lat": float(r["lat"]), "lon": float(r["lon"]),
                "speed": int(float(r["speed_kmh"])), "plate": r["plate"],
            })

    out_rows, summary = [], []
    for (day, trip_id, vehicle_id), samples in sorted(trips_samples.items()):
        samples = sorted({s["ts"]: s for s in samples}.values(), key=lambda s: s["ts"])
        trip = st["trips"][trip_id]
        poly = st["shapes"].get(trip["shape_id"])
        if poly is None or len(samples) < 2:
            continue
        smp = project_samples(poly, samples)
        rows = classify_stops(stop_positions(poly, st, trip_id), smp)
        post = skip_mode_posterior(rows, args.base_pass_rate, args.prior_skip, args.switch)
        observed = [r for r in rows if r["p_pass"] is not None]
        # runs: consecutive observed stops with P(skipping) >= threshold
        runs, cur = [], []
        for r, p in zip(rows, post):
            if p is not None and p >= args.skip_threshold:
                cur.append((r, p))
            elif p is not None:
                if cur:
                    runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        runs = [run for run in runs if len(run) >= args.min_run]
        for run in runs:
            share = len(run) / max(1, len(observed))
            hint = "out of service / not picking up (whole observed trip)" if share >= 0.8 else "full or skipping a section"
            conf = round(sum(p for _, p in run) / len(run), 2)
            summary.append((day, trip_id, samples[0]["plate"], trip["trip_headsign"], run[0][0]["name"], run[-1][0]["name"],
                            len(run), conf, hint))
        for r, p in zip(rows, post):
            out_rows.append({
                "date": day, "trip_id": trip_id, "vehicle_id": vehicle_id, "plate": samples[0]["plate"],
                "headsign": trip["trip_headsign"], "stop_sequence": r["seq"], "stop_id": r["stop_id"],
                "stop_name": r["name"], "scheduled": r["sched"], "label": r["label"],
                "p_pass": "" if r["p_pass"] is None else r["p_pass"], "p_skipping_mode": "" if p is None else p,
                "evidence": r["evidence"],
            })
        n_pass = sum(r["label"] == "likely_passed" for r in observed)
        n_stop = sum(r["label"] in ("stopped", "likely_stopped") for r in observed)
        print(f"{day} {trip_id} {samples[0]['plate']:<8} -> {trip['trip_headsign'][:18]:<18} "
              f"samples {len(smp):>3}, stops observed {len(observed):>2}: stopped {n_stop:>2}, passed {n_pass:>2}, "
              f"unknown {len(observed) - n_pass - n_stop:>2}")

    out = log.with_name(log.stem + "_passes.csv")
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()) if out_rows else ["date"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nPer-stop labels: {out}")
    print(f"\nSkipping runs (>= {args.min_run} consecutive stops with P(skipping) >= {args.skip_threshold}): {len(summary)}")
    for day, trip_id, plate, head, s1, s2, n, conf, hint in summary:
        print(f"  {day} {trip_id} {plate} -> {head}: {s1} .. {s2} ({n} stops), confidence {conf:.0%} — {hint}")


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def load_rt(route_ids) -> tuple[int, list, dict]:
    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(fetch(RT_URL))
    vehicles, updates = [], {}
    for e in feed.entity:
        if e.HasField("vehicle") and e.vehicle.trip.route_id in route_ids:
            vehicles.append(e.vehicle)
        if e.HasField("trip_update") and e.trip_update.trip.route_id in route_ids:
            updates[e.trip_update.trip.trip_id] = e.trip_update
    return feed.header.timestamp, vehicles, updates


def hhmm(ts: int) -> str:
    return datetime.fromtimestamp(ts, TZ).strftime("%H:%M:%S")


def stop_name(st, sid: str) -> str:
    s = st["stops"].get(sid)
    return s["stop_name"] if s else sid


def nearest_stop(st, trip_id: str, lat: float, lon: float):
    best = None
    for _, sid, _ in st["stop_seq"].get(trip_id, []):
        s = st["stops"].get(sid)
        if not s:
            continue
        d = haversine_m(lat, lon, float(s["stop_lat"]), float(s["stop_lon"]))
        if best is None or d < best[1]:
            best = (sid, d)
    return best


def next_stop(tu, now: int):
    if tu is None:
        return None
    for u in tu.stop_time_update:
        t = u.arrival.time or u.departure.time
        if t >= now:
            return u.stop_id, t, u.arrival.delay
    return None


def snapshot(st, now, vehicles, updates) -> list[dict]:
    rows = []
    for v in vehicles:
        trip = st["trips"].get(v.trip.trip_id, {})
        lat, lon = v.position.latitude, v.position.longitude
        near = nearest_stop(st, v.trip.trip_id, lat, lon)
        nxt = next_stop(updates.get(v.trip.trip_id), now)
        rows.append({
            "feed_ts": now,
            "vehicle_ts": v.timestamp,
            "plate": v.vehicle.license_plate or v.vehicle.label,
            "vehicle_id": v.vehicle.id,
            "trip_id": v.trip.trip_id,
            "route_id": v.trip.route_id,
            "headsign": trip.get("trip_headsign", ""),
            "trip_start": v.trip.start_time,
            "lat": lat,
            "lon": lon,
            "speed_kmh": round(v.position.speed * 3.6),
            "bearing": int(v.position.bearing),
            "nearest_stop_id": near[0] if near else "",
            "nearest_stop": stop_name(st, near[0]) if near else "",
            "nearest_stop_m": round(near[1]) if near else "",
            "next_stop": stop_name(st, nxt[0]) if nxt else "",
            "next_stop_eta": hhmm(nxt[1]) if nxt else "",
            "delay_min": round(nxt[2] / 60, 1) if nxt else "",
        })
    rows.sort(key=lambda r: (r["headsign"], r["trip_start"]))
    return rows


def print_snapshot(rows, now):
    print(f"\nRoute 30, Limassol: {len(rows)} buses, feed at {hhmm(now)}")
    hdr = f"{'plate':<8} {'to':<20} {'start':<5} {'km/h':>4} {'age':>4}  {'where':<40} {'next stop':<28} {'eta':<8} {'delay':>6}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        if r["nearest_stop_m"] != "" and r["nearest_stop_m"] <= AT_STOP_M:
            where = f"at {r['nearest_stop']}"
        else:
            where = f"{r['nearest_stop_m']} m from {r['nearest_stop']}"
        delay = f"{r['delay_min']:+}" if r["delay_min"] != "" else ""
        print(f"{r['plate']:<8} {r['headsign'][:20]:<20} {r['trip_start'][:5]:<5} {r['speed_kmh']:>4} "
              f"{now - r['vehicle_ts']:>3}s  {where[:40]:<40} {r['next_stop'][:28]:<28} {r['next_stop_eta']:<8} {delay:>6}")


def print_stop(st, stop_query: str, now, updates):
    matches = [sid for sid, s in st["stops"].items() if stop_query == sid or stop_query.lower() in s["stop_name"].lower()]
    if not matches:
        sys.exit(f"no route-30 stop matches {stop_query!r}")
    arrivals = []
    for tu in updates.values():
        trip = st["trips"].get(tu.trip.trip_id, {})
        for u in tu.stop_time_update:
            t = u.arrival.time or u.departure.time
            if u.stop_id in matches and t >= now:
                arrivals.append((t, stop_name(st, u.stop_id), u.stop_id, trip.get("trip_headsign", ""), u.arrival.delay))
    arrivals.sort()
    print(f"\nNext route-30 arrivals ({hhmm(now)}), stops: " + ", ".join(f"{stop_name(st, s)} [{s}]" for s in matches))
    if not arrivals:
        print("  no live predictions right now")
    for t, name, sid, head, delay in arrivals:
        print(f"  {hhmm(t)}  in {round((t - now) / 60):>3} min  {name} [{sid}] -> {head}  (delay {delay / 60:+.1f} min)")


# ---- live map (--serve) -----------------------------------------------------------------------

def p_stop_next(p_skip: float, args) -> float:
    """Chance the bus stops at its next stop: in service it stops unless nobody waits."""
    p_skip_next = p_skip * (1 - args.switch) + (1 - p_skip) * args.switch
    return (1 - p_skip_next) * (1 - args.base_pass_rate) + p_skip_next * (1 - SKIP_PASS_RATE)


class LiveState:
    def __init__(self, st, args):
        self.st, self.args = st, args
        self.lock = threading.Lock()
        self.hist: dict[tuple[str, str], dict] = {}
        self.stop_pos: dict[str, list[dict]] = {}
        self.payload: dict = {"buses": [], "stops": {}, "feed_ts": None, "error": "waiting for the first feed"}
        self.log_fh = self.log_writer = None
        self.log_path: Path | None = None

    def _log_for_today(self) -> Path | None:
        if not self.args.log:
            return None
        path = Path(str(self.args.log).replace("{date}", datetime.now(TZ).date().isoformat()))
        if path != self.log_path:
            if self.log_fh:
                self.log_fh.close()
            path.parent.mkdir(parents=True, exist_ok=True)
            self._new_log = not path.exists() or path.stat().st_size == 0
            self.log_fh, self.log_writer, self.log_path = path.open("a", newline=""), None, path
        return path

    def _stops_of(self, trip_id: str, poly: Polyline) -> list[dict]:
        if trip_id not in self.stop_pos:
            self.stop_pos[trip_id] = stop_positions(poly, self.st, trip_id)
        return self.stop_pos[trip_id]

    def add_sample(self, trip_id: str, vehicle_id: str, s: dict) -> None:
        trip = self.st["trips"].get(trip_id)
        if not trip or trip["shape_id"] not in self.st["shapes"]:
            return
        h = self.hist.setdefault((trip_id, vehicle_id), {"seen": set(), "proj": [], "last": None})
        if s["ts"] in h["seen"] or (h["last"] and s["ts"] < h["last"]["ts"]):
            return
        h["seen"].add(s["ts"])
        h["last"] = s
        p = project_one(self.st["shapes"][trip["shape_id"]], s, h["proj"][-1] if h["proj"] else None)
        if p:
            h["proj"].append(p)

    def seed(self, path: Path) -> None:
        path = Path(str(path).replace("{date}", datetime.now(TZ).date().isoformat()))
        if not path.exists():
            return
        with path.open() as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: int(r["vehicle_ts"]))
        for r in rows:
            if time.time() - int(r["vehicle_ts"]) < 3 * 3600:
                self.add_sample(r["trip_id"], r["vehicle_id"], {
                    "ts": int(r["vehicle_ts"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
                    "speed": int(float(r["speed_kmh"])), "plate": r["plate"]})

    def poll(self) -> None:
        now, vehicles, updates = load_rt(self.st["routes"].keys())
        rows = snapshot(self.st, now, vehicles, updates)
        self._log_for_today()
        if self.log_fh and rows:
            if self.log_writer is None:
                self.log_writer = csv.DictWriter(self.log_fh, fieldnames=list(rows[0].keys()))
                if self._new_log:
                    self.log_writer.writeheader()
        buses, plate_of = [], {}
        for v in vehicles:
            s = {"ts": v.timestamp, "lat": v.position.latitude, "lon": v.position.longitude,
                 "speed": round(v.position.speed * 3.6), "plate": v.vehicle.license_plate or v.vehicle.label}
            key = (v.trip.trip_id, v.vehicle.id)
            known = key in self.hist and v.timestamp in self.hist[key]["seen"]
            self.add_sample(v.trip.trip_id, v.vehicle.id, s)
            if self.log_writer and not known:
                self.log_writer.writerows([r for r in rows if r["vehicle_id"] == v.vehicle.id and r["vehicle_ts"] == v.timestamp])
            plate_of[v.trip.trip_id] = s["plate"]
            if now - v.timestamp > 300:
                continue
            buses.append(self._bus_view(v, s, updates.get(v.trip.trip_id), now))
        if self.log_fh:
            self.log_fh.flush()
        stops: dict[str, list] = {}
        p_by_plate = {b["plate"]: b["p_stop_next"] for b in buses}
        for tu in updates.values():
            trip = self.st["trips"].get(tu.trip.trip_id, {})
            plate = plate_of.get(tu.trip.trip_id, "")
            for u in tu.stop_time_update:
                t = u.arrival.time or u.departure.time
                if t >= now:
                    stops.setdefault(u.stop_id, []).append({
                        "eta": hhmm(t)[:5], "min": round((t - now) / 60), "headsign": trip.get("trip_headsign", ""),
                        "plate": plate, "p_stop": p_by_plate.get(plate)})
        for lst in stops.values():
            lst.sort(key=lambda a: a["min"])
            del lst[4:]
        cutoff = now - 3 * 3600
        self.hist = {k: h for k, h in self.hist.items() if h["last"] and h["last"]["ts"] > cutoff}
        with self.lock:
            self.payload = {"feed_ts": now, "feed_time": hhmm(now), "buses": buses, "stops": stops, "error": None}

    def _bus_view(self, v, s, tu, now) -> dict:
        trip_id = v.trip.trip_id
        trip = self.st["trips"].get(trip_id, {})
        h = self.hist.get((trip_id, v.vehicle.id), {"proj": []})
        view = {"id": v.vehicle.id, "plate": s["plate"], "trip_id": trip_id, "headsign": trip.get("trip_headsign", ""),
                "direction": trip.get("direction_id", ""), "start": v.trip.start_time[:5], "lat": s["lat"], "lon": s["lon"],
                "speed": s["speed"], "bearing": int(v.position.bearing), "age": now - v.timestamp,
                "samples": len(h["proj"]), "on_route": bool(h["proj"]) and h["proj"][-1]["ts"] == v.timestamp,
                "p_skip": self.args.prior_skip, "recent": [], "next_stop": None}
        poly = self.st["shapes"].get(trip.get("shape_id", ""))
        if poly is not None and h["proj"]:
            stops = self._stops_of(trip_id, poly)
            rows = classify_stops(stops, h["proj"])
            post = skip_mode_posterior(rows, self.args.base_pass_rate, self.args.prior_skip, self.args.switch)
            observed = [(r, p) for r, p in zip(rows, post) if p is not None]
            if observed:
                view["p_skip"] = observed[-1][1]
            view["recent"] = [{"stop": r["name"], "label": r["label"], "p_pass": r["p_pass"]} for r, _ in observed[-6:]]
            cur = h["proj"][-1]["pos"]
            nxt = next((st for st in stops if st["pos"] > cur + 5), None)
            if nxt:
                view["next_stop"] = {"id": nxt["stop_id"], "name": nxt["name"], "m": round(nxt["pos"] - cur)}
                if tu is not None:
                    for u in tu.stop_time_update:
                        if u.stop_id == nxt["stop_id"]:
                            t = u.arrival.time or u.departure.time
                            view["next_stop"].update(eta=hhmm(t)[:5], delay_min=round(u.arrival.delay / 60, 1))
        view["p_stop_next"] = round(p_stop_next(view["p_skip"], self.args), 2)
        return view

    def static_payload(self) -> dict:
        shapes = {}
        for t in self.st["trips"].values():
            sid = t["shape_id"]
            if sid in self.st["shapes"] and sid not in shapes:
                shapes[sid] = {"direction": t["direction_id"], "headsign": t["trip_headsign"],
                               "pts": [[round(la, 6), round(lo, 6)] for la, lo in self.st["shapes"][sid].pts]}
        stops = [{"id": sid, "name": s["stop_name"], "lat": float(s["stop_lat"]), "lon": float(s["stop_lon"])}
                 for sid, s in self.st["stops"].items()]
        return {"route": self.args.route, "shapes": list(shapes.values()), "stops": stops,
                "normal_stop_chance": round(p_stop_next(0.0, self.args), 2)}


def serve(st, args) -> None:
    state = LiveState(st, args)
    seed = args.seed or args.log
    if seed:
        state.seed(seed)

    def loop():
        while True:
            try:
                state.poll()
            except Exception as e:  # keep serving the last good state
                with state.lock:
                    state.payload = {**state.payload, "error": f"feed error: {e}"}
            time.sleep(args.poll)

    threading.Thread(target=loop, daemon=True).start()
    static_json = json.dumps(state.static_payload()).encode()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(MAP_HTML.encode(), "text/html; charset=utf-8")
            elif self.path == "/api/static":
                self._send(static_json, "application/json")
            elif self.path == "/api/state":
                with state.lock:
                    body = json.dumps(state.payload).encode()
                self._send(body, "application/json")
            else:
                self.send_error(404)

        def log_message(self, *a):
            pass

    print(f"Route {args.route} live map: http://{args.host}:{args.serve}/  (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((args.host, args.serve), Handler).serve_forever()


MAP_HTML = r"""<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Автобус 30 Лимасол</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
  :root { --bg:#fff; --fg:#1f2328; --muted:#656d76; --line:#d0d7de; --panel:#f6f8fa;
          --hi:#1a7f37; --mid:#bf8700; --lo:#cf222e; --dir0:#0969da; --dir1:#8250df; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0d1117; --fg:#e6edf3; --muted:#8d96a0; --line:#30363d; --panel:#161b22;
            --hi:#3fb950; --mid:#d29922; --lo:#f85149; --dir0:#4493f8; --dir1:#a371f7; }
  }
  * { box-sizing:border-box; }
  html, body { margin:0; height:100%; background:var(--bg); color:var(--fg);
               font:14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  #app { display:flex; height:100%; }
  #side { width:340px; flex:none; overflow:auto; border-right:1px solid var(--line); background:var(--panel); padding:12px 16px; }
  #map { flex:1; }
  h1 { font-size:17px; margin:0 0 2px; }
  .muted { color:var(--muted); font-size:12px; }
  .err { color:var(--lo); font-size:12px; margin-top:4px; }
  .legend { margin:10px 0; font-size:12px; }
  .legend .item { display:flex; align-items:center; gap:6px; margin:2px 0; }
  .legend .note { margin-top:4px; }
  .dot { width:12px; height:12px; border-radius:50%; flex:none; }
  .bus { padding:8px 0; border-top:1px solid var(--line); cursor:pointer; }
  .bus:hover { background:rgba(127,127,127,.08); }
  .row { display:flex; justify-content:space-between; gap:8px; }
  .pct { font-weight:600; font-variant-numeric:tabular-nums; }
  .busicon { display:flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:50%;
             color:#fff; font-weight:700; font-size:11px; border:2px solid #fff; box-shadow:0 1px 3px rgba(0,0,0,.4); }
  .arrow { position:absolute; left:50%; top:-7px; width:0; height:0; margin-left:-5px;
           border-left:5px solid transparent; border-right:5px solid transparent; border-bottom:7px solid #222; }
  @media (prefers-color-scheme: dark) { .basemap { filter: invert(1) hue-rotate(180deg) brightness(.85) contrast(.9); } }
  .leaflet-popup-content { font-size:13px; }
  .leaflet-popup-content table { border-collapse:collapse; }
  .leaflet-popup-content td { padding:1px 6px 1px 0; }
  @media (max-width: 720px) {
    #app { flex-direction:column; }
    #side { width:auto; max-height:38%; border-right:none; border-bottom:1px solid var(--line); }
  }
</style>
</head>
<body>
<div id="app">
  <div id="side">
    <h1>Автобус <span id="route">30</span> · Лимасол</h1>
    <div class="muted">Фид: <span id="feedtime">—</span> · обновление через <span id="cd">30</span> с</div>
    <div id="err" class="err"></div>
    <div class="legend">
      <div class="muted">Число на метке — вероятность, что автобус едет мимо остановок (не в рейсе или полный):</div>
      <div class="item"><span class="dot" style="background:var(--hi)"></span>&lt; 30%: обычный рейс</div>
      <div class="item"><span class="dot" style="background:var(--mid)"></span>30–70%: проехал несколько остановок, под вопросом</div>
      <div class="item"><span class="dot" style="background:var(--lo)"></span>≥ 70%: скорее всего едет мимо остановок</div>
      <div class="muted note">Растёт, когда автобус проезжает остановки подряд, и падает, как только его видят стоящим на остановке.
        Шанс остановки на следующей (в карточке автобуса) у обычного рейса — около <span id="base"></span>%.</div>
    </div>
    <div id="list"></div>
  </div>
  <div id="map"></div>
</div>
<script>
const REFRESH = 30;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const colorFor = p => p >= 0.7 ? css('--lo') : p >= 0.3 ? css('--mid') : css('--hi');
const pct = p => p == null ? '—' : Math.round(p * 100) + '%';
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const LABEL = {stopped:'стоял', likely_stopped:'скорее стоял', likely_passed:'скорее проехал', unknown:'неясно'};

const map = L.map('map', {zoomControl:true}).setView([34.68, 33.04], 13);
// OpenStreetMap tiles need no key (CARTO basemaps now answer "API key required").
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  {maxZoom:19, className:'basemap',
   attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'}).addTo(map);
const stopLayer = L.layerGroup().addTo(map), busLayer = L.layerGroup().addTo(map);
let stopMarkers = {}, busMarkers = {}, state = null, countdown = REFRESH;

function stopPopup(stop) {
  const arr = (state && state.stops[stop.id]) || [];
  const rows = arr.map(a => `<tr><td>${esc(a.eta)}</td><td>${a.min} мин</td><td>→ ${esc(a.headsign)}</td>
     <td>${esc(a.plate)}</td><td class="pct">${a.p_stop == null ? '' : 'стоп ' + pct(a.p_stop)}</td></tr>`).join('');
  return `<b>${esc(stop.name)}</b> <span class="muted">[${esc(stop.id)}]</span><br>` +
    (rows ? `<table>${rows}</table>` : '<span class="muted">нет прогнозов прибытия</span>');
}

function busPopup(b) {
  const ns = b.next_stop;
  const recent = b.recent.map(r => `<tr><td>${esc(r.stop)}</td><td>${LABEL[r.label] || r.label}</td>
     <td class="pct">${r.p_pass == null ? '' : 'проезд ' + pct(r.p_pass)}</td></tr>`).join('');
  return `<b>${esc(b.plate)}</b> → ${esc(b.headsign)} <span class="muted">(рейс ${esc(b.start)})</span><br>
    ${b.speed} км/ч · позиция ${b.age} с назад<br>
    ${ns ? `Следующая: <b>${esc(ns.name)}</b>, ${ns.m} м${ns.eta ? ', ' + esc(ns.eta) : ''}${ns.delay_min != null ? ` (${ns.delay_min > 0 ? '+' : ''}${ns.delay_min} мин)` : ''}<br>` : ''}
    Остановится на следующей: <b class="pct">${pct(b.p_stop_next)}</b><br>
    Режим «едет мимо остановок»: ${pct(b.p_skip)}<br>
    ${recent ? `<div class="muted" style="margin-top:4px">Последние остановки:</div><table>${recent}</table>`
             : `<span class="muted">${b.on_route ? 'истории пока мало' : 'вне линии маршрута'}</span>`}`;
}

function busIcon(b) {
  const c = colorFor(b.p_skip);
  return L.divIcon({className:'', iconSize:[28,28], iconAnchor:[14,14],
    html:`<div style="position:relative;transform:rotate(${b.bearing}deg)"><span class="arrow"></span>
          <div class="busicon" style="background:${c};transform:rotate(${-b.bearing}deg)">${Math.round(b.p_skip*100)}</div></div>`});
}

function render() {
  if (!state) return;
  document.getElementById('feedtime').textContent = state.feed_time || '—';
  document.getElementById('err').textContent = state.error || '';
  const seen = new Set();
  for (const b of state.buses) {
    seen.add(b.id);
    let m = busMarkers[b.id];
    if (!m) { m = busMarkers[b.id] = L.marker([b.lat, b.lon], {zIndexOffset:1000}).addTo(busLayer); m.bindPopup(''); }
    m.setLatLng([b.lat, b.lon]).setIcon(busIcon(b));
    m.getPopup().setContent(busPopup(b));
    m.bindTooltip(`${b.plate} · едет мимо ${pct(b.p_skip)} · стоп ${pct(b.p_stop_next)}`, {direction:'top', offset:[0,-14]});
  }
  for (const id of Object.keys(busMarkers)) if (!seen.has(id)) { busLayer.removeLayer(busMarkers[id]); delete busMarkers[id]; }
  for (const [id, m] of Object.entries(stopMarkers)) if (m.isPopupOpen()) m.getPopup().setContent(stopPopup(m.stop));
  const buses = [...state.buses].sort((a, b) => a.headsign.localeCompare(b.headsign) || a.start.localeCompare(b.start));
  document.getElementById('list').innerHTML = buses.map(b => `
    <div class="bus" data-id="${esc(b.id)}">
      <div class="row"><span><span class="dot" style="display:inline-block;vertical-align:-1px;background:${colorFor(b.p_skip)}"></span>
        <b>${esc(b.plate)}</b> → ${esc(b.headsign)}</span><span class="pct" title="вероятность «едет мимо остановок»">${pct(b.p_skip)}</span></div>
      <div class="muted">${b.speed} км/ч · ${b.next_stop ? 'след.: ' + esc(b.next_stop.name) + (b.next_stop.eta ? ' ' + esc(b.next_stop.eta) : '') : 'нет данных о следующей'}</div>
    </div>`).join('') || '<div class="muted">Активных автобусов нет</div>';
  document.querySelectorAll('.bus').forEach(el => el.onclick = () => {
    const m = busMarkers[el.dataset.id]; if (m) { map.setView(m.getLatLng(), 16); m.openPopup(); }
  });
}

async function refresh() {
  try {
    state = await (await fetch('api/state', {cache:'no-store'})).json();
    render();
  } catch (e) { document.getElementById('err').textContent = 'нет связи с локальным сервером'; }
  countdown = REFRESH;
}

(async () => {
  const st = await (await fetch('api/static')).json();
  document.getElementById('route').textContent = st.route;
  document.getElementById('base').textContent = Math.round(st.normal_stop_chance * 100);
  const bounds = [];
  for (const s of st.shapes) {
    L.polyline(s.pts, {color: css(s.direction === '0' ? '--dir0' : '--dir1'), weight:4, opacity:.55})
      .bindTooltip('→ ' + s.headsign).addTo(map);
    bounds.push(...s.pts);
  }
  for (const s of st.stops) {
    const m = L.circleMarker([s.lat, s.lon], {radius:5, color:'#fff', weight:1.5, fillColor:'#57606a', fillOpacity:1})
      .bindPopup('').addTo(stopLayer);
    m.stop = s;
    m.on('popupopen', () => m.getPopup().setContent(stopPopup(s)));
    stopMarkers[s.id] = m;
  }
  map.invalidateSize();
  if (bounds.length) map.fitBounds(L.latLngBounds(bounds), {padding:[20,20], maxZoom:15});
  window.addEventListener('resize', () => map.invalidateSize());
  await refresh();
  setInterval(() => { countdown -= 1; document.getElementById('cd').textContent = Math.max(countdown, 0);
                      if (countdown <= 0) refresh(); }, 1000);
})();
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--route", default="30", help="route short name (default 30)")
    ap.add_argument("--stop", help="show next arrivals at this stop (stop_id or part of the name)")
    ap.add_argument("--watch", type=int, metavar="SEC", help="refresh every SEC seconds until Ctrl+C")
    ap.add_argument("--log", type=Path, help="append every new vehicle position to this CSV")
    ap.add_argument("--classify", type=Path, metavar="CSV", help="label stop pass-throughs in a CSV written by --log")
    ap.add_argument("--base-pass-rate", type=float, default=0.3,
                    help="share of stops a normal in-service bus passes because nobody waits (default 0.3, an assumption)")
    ap.add_argument("--prior-skip", type=float, default=0.05, help="prior P(trip starts in skipping mode)")
    ap.add_argument("--switch", type=float, default=0.1, help="P(mode changes between consecutive stops)")
    ap.add_argument("--skip-threshold", type=float, default=0.8, help="P(skipping) to count a stop into a run")
    ap.add_argument("--min-run", type=int, default=3, help="min consecutive stops to report a skipping run")
    ap.add_argument("--serve", type=int, metavar="PORT", help="live map with stop chances at http://127.0.0.1:PORT/")
    ap.add_argument("--seed", type=Path, metavar="CSV", help="--serve: warm up history from this CSV (default: the --log file)")
    ap.add_argument("--host", default="127.0.0.1", help="--serve: bind address (default 127.0.0.1; put a proxy with auth in front)")
    ap.add_argument("--poll", type=int, default=10, help="--serve: feed poll interval, seconds (default 10)")
    args = ap.parse_args()

    st = load_static(args.route)
    if args.serve:
        serve(st, args)
        return
    if args.classify:
        classify_log(st, args.classify, args)
        return
    seen: set[tuple[str, int]] = set()
    writer = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        new_file = not args.log.exists()
        log_fh = args.log.open("a", newline="")
    while True:
        now, vehicles, updates = load_rt(st["routes"].keys())
        rows = snapshot(st, now, vehicles, updates)
        if args.stop:
            print_stop(st, args.stop, now, updates)
        else:
            print_snapshot(rows, now)
        if args.log and rows:
            if writer is None:
                writer = csv.DictWriter(log_fh, fieldnames=list(rows[0].keys()))
                if new_file:
                    writer.writeheader()
            fresh = [r for r in rows if (r["vehicle_id"], r["vehicle_ts"]) not in seen]
            seen.update((r["vehicle_id"], r["vehicle_ts"]) for r in fresh)
            writer.writerows(fresh)
            log_fh.flush()
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
