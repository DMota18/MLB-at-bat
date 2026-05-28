"""Weather data for MLB game venues.

Uses Open-Meteo API (free, no key required) to fetch temperature,
wind speed, and wind direction for game time at each venue.
"""

from __future__ import annotations

import logging
from typing import Any

from config import fetch_json

logger = logging.getLogger("baseball_bot.weather")

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Venue coordinates (lat, lon) for all 30 MLB parks
VENUE_COORDS = {
    "Coors Field": (39.756, -104.994),
    "Great American Ball Park": (39.097, -84.507),
    "Yankee Stadium": (40.829, -73.926),
    "Citizens Bank Park": (39.906, -75.166),
    "Chase Field": (33.445, -112.067),
    "Globe Life Field": (32.747, -97.084),
    "Minute Maid Park": (29.757, -95.355),
    "Wrigley Field": (41.948, -87.656),
    "Fenway Park": (42.346, -71.097),
    "American Family Field": (43.028, -87.971),
    "Camden Yards": (39.284, -76.622),
    "Rogers Centre": (43.641, -79.389),
    "Guaranteed Rate Field": (41.830, -87.634),
    "Nationals Park": (38.873, -77.007),
    "Truist Park": (33.891, -84.468),
    "Target Field": (44.982, -93.278),
    "Busch Stadium": (38.623, -90.193),
    "loanDepot park": (25.778, -80.220),
    "Citi Field": (40.757, -73.846),
    "Angel Stadium": (33.800, -117.883),
    "PNC Park": (40.447, -80.006),
    "Progressive Field": (41.496, -81.685),
    "Comerica Park": (42.339, -83.049),
    "Dodger Stadium": (34.074, -118.240),
    "Kauffman Stadium": (39.051, -94.480),
    "Tropicana Field": (27.768, -82.653),
    "Oakland Coliseum": (37.751, -122.201),
    "Petco Park": (32.707, -117.157),
    "T-Mobile Park": (47.591, -122.332),
    "Oracle Park": (37.778, -122.389),
}

# Dome/retractable roof venues (weather doesn't matter as much)
INDOOR_VENUES = {
    "Tropicana Field",      # fixed dome
    "loanDepot park",       # retractable roof
    "Globe Life Field",     # retractable roof
    "Minute Maid Park",     # retractable roof
    "Rogers Centre",        # retractable roof
    "Chase Field",          # retractable roof
    "American Family Field", # retractable roof
    "T-Mobile Park",        # retractable roof
}


async def get_game_weather(venue: str, game_time_iso: str) -> dict[str, Any]:
    """Fetch weather for a venue at game time.

    Returns:
        {
            "temperature_f": 72.0,
            "wind_speed_mph": 12.5,
            "wind_direction": 225,  # degrees (0=N, 90=E, 180=S, 270=W)
            "is_indoor": False,
            "has_data": True,
        }
    """
    if venue in INDOOR_VENUES:
        return {
            "temperature_f": 72.0,  # climate controlled
            "wind_speed_mph": 0.0,
            "wind_direction": 0,
            "is_indoor": True,
            "has_data": True,
        }

    coords = VENUE_COORDS.get(venue)
    if not coords:
        return {"has_data": False}

    lat, lon = coords

    try:
        # Extract date from ISO time
        date_str = game_time_iso[:10]  # "2026-05-17"

        data = await fetch_json(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
                "start_date": date_str,
                "end_date": date_str,
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "America/New_York",
            },
        )

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        winds = hourly.get("wind_speed_10m", [])
        dirs_ = hourly.get("wind_direction_10m", [])

        if not times:
            return {"has_data": False}

        # Find the hour closest to game time
        from datetime import datetime
        try:
            game_dt = datetime.fromisoformat(game_time_iso.replace("Z", "+00:00"))
            game_hour = game_dt.hour
        except Exception:
            game_hour = 19  # default to 7pm

        # Find closest hourly index
        idx = min(game_hour, len(temps) - 1)

        return {
            "temperature_f": round(temps[idx], 1),
            "wind_speed_mph": round(winds[idx], 1),
            "wind_direction": int(dirs_[idx]),
            "is_indoor": False,
            "has_data": True,
        }

    except Exception as e:
        logger.warning(f"Weather fetch failed for {venue}: {e}")
        return {"has_data": False}


def weather_hit_adjustment(weather: dict[str, Any]) -> float:
    """Compute a per-AB probability adjustment based on weather.

    Returns a small additive adjustment to per-AB hit probability.
    Positive = weather helps hitting, negative = hurts.

    Based on research:
    - Temperature: +1% hit rate per 10°F above 72°F (ball carries better)
    - Wind: high wind (>15mph) adds ~0.5% (more balls in play carry)
    - Indoor: no adjustment (neutral, climate controlled)
    """
    if not weather.get("has_data") or weather.get("is_indoor"):
        return 0.0

    adj = 0.0

    # Temperature effect: warm air = ball carries better
    temp = weather.get("temperature_f", 72)
    temp_diff = (temp - 72) / 10  # per 10°F
    adj += temp_diff * 0.005  # ~0.5% per 10°F

    # Wind effect: strong wind can help or hurt depending on direction
    # Simplified: high wind generally increases variance but slightly helps hitters
    wind = weather.get("wind_speed_mph", 0)
    if wind >= 25:
        adj += 0.005
    elif wind >= 15:
        adj += 0.003  # slight boost in windy conditions

    # Clamp to reasonable range
    return max(-0.015, min(0.015, adj))
