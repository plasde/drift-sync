import requests
import numpy as np
import json
from datetime import datetime, timezone
from pathlib import Path

CACHE_DIR = Path("cache")
CACHE_MAX_AGE_HOURS = 6


def fetch_wind_grid(
    bounds,
    grid_resolution_deg=1.0,
    forecast_hours=48,
    use_historical=False,
    historical_date=None,  # "YYYY-MM-DD"
):
    """
    Fetch time-averaged wind data from Open-Meteo over a lat/lon grid.

    Args:
        bounds:              (north, south, east, west) in degrees
        grid_resolution_deg: spacing between sample points in degrees
        forecast_hours:      how many hours ahead to fetch and average
        use_historical:      if True, fetch from archive instead of forecast
        historical_date:     required when use_historical=True, format "YYYY-MM-DD"

    Returns:
        dict with keys:
            lats     - 1D np.ndarray of latitudes
            lons     - 1D np.ndarray of longitudes
            u_grid   - 2D np.ndarray (n_lat, n_lon), eastward component in knots
            v_grid   - 2D np.ndarray (n_lat, n_lon), northward component in knots
            fetched_at - ISO timestamp string
    """
    north, south, east, west = bounds
    lats = np.arange(south, north + grid_resolution_deg, grid_resolution_deg)
    lons = np.arange(west, east + grid_resolution_deg, grid_resolution_deg)

    cache_key = _make_cache_key(
        bounds, grid_resolution_deg, forecast_hours, use_historical, historical_date
    )
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    mode_label = (
        f"historical {historical_date}"
        if use_historical
        else f"{forecast_hours}h forecast"
    )
    print(f"Fetching {mode_label} wind data for {len(lats)}x{len(lons)} grid...")

    u_grid = np.zeros((len(lats), len(lons)))
    v_grid = np.zeros((len(lats), len(lons)))

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            u, v = _fetch_point(
                lat, lon, forecast_hours, use_historical, historical_date
            )
            u_grid[i, j] = u
            v_grid[i, j] = v
            print(f"  ({lat:.1f}N, {lon:.1f}E)  u={u:+.1f}  v={v:+.1f} kts")

    result = {
        "lats": lats.tolist(),
        "lons": lons.tolist(),
        "u_grid": u_grid.tolist(),
        "v_grid": v_grid.tolist(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_cache(cache_key, result)
    return _arrays_from_lists(result)


def _fetch_point(lat, lon, forecast_hours, use_historical, historical_date):
    """
    Fetch hourly wind speed + direction at one point, return time-averaged (u, v) in knots.
    """
    if use_historical:
        if historical_date is None:
            raise ValueError("historical_date required when use_historical=True")
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": historical_date,
            "end_date": historical_date,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "wind_speed_unit": "kn",
            "timezone": "UTC",
        }
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "forecast_hours": forecast_hours,
            "wind_speed_unit": "kn",
            "timezone": "UTC",
        }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        speeds = data["hourly"]["wind_speed_10m"]
        directions = data["hourly"]["wind_direction_10m"]

        # Filter out None values (can appear at edges of forecast window)
        pairs = [
            (s, d)
            for s, d in zip(speeds, directions)
            if s is not None and d is not None
        ]
        if not pairs:
            print(f"  Warning: no valid data for ({lat}, {lon}), using 0")
            return 0.0, 0.0

        speeds_arr = np.array([p[0] for p in pairs])
        dirs_arr = np.radians([p[1] for p in pairs])

        # Meteorological convention: direction is where wind comes FROM
        # u = eastward component, v = northward component
        u_vals = -speeds_arr * np.sin(dirs_arr)
        v_vals = -speeds_arr * np.cos(dirs_arr)

        return float(np.mean(u_vals)), float(np.mean(v_vals))

    except Exception as e:
        print(f"  Warning: fetch failed for ({lat:.1f}, {lon:.1f}): {e}")
        return 0.0, 0.0


# ── Cache helpers ─────────────────────────────────────────────────────────────


def _make_cache_key(
    bounds, resolution, forecast_hours, use_historical, historical_date
):
    north, south, east, west = bounds
    mode = f"hist_{historical_date}" if use_historical else f"fcst_{forecast_hours}h"
    return (
        f"wind_{south:.1f}N_{north:.1f}N_{west:.1f}E_{east:.1f}E"
        f"_{resolution:.1f}deg_{mode}"
    )


def _load_cache(key):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None

    try:
        with open(path) as f:
            data = json.load(f)

        fetched_at = datetime.fromisoformat(data["fetched_at"])
        # Make fetched_at timezone-aware if it isn't
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)

        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600
        if age_hours > CACHE_MAX_AGE_HOURS:
            print(f"Cache expired ({age_hours:.1f}h old), refetching...")
            return None

        print(f"Loaded wind data from cache ({age_hours:.1f}h old).")
        return _arrays_from_lists(data)

    except Exception as e:
        print(f"Cache load failed ({e}), refetching...")
        return None


def _save_cache(key, data):
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    try:
        with open(path, "w") as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Cache save failed: {e}")


def _arrays_from_lists(data):
    """Convert list fields back to numpy arrays after JSON round-trip."""
    return {
        **data,
        "lats": np.array(data["lats"]),
        "lons": np.array(data["lons"]),
        "u_grid": np.array(data["u_grid"]),
        "v_grid": np.array(data["v_grid"]),
    }
