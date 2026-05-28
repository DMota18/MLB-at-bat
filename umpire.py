"""Home plate umpire tendency data for hit prediction adjustment.

Umpire strike zone size directly affects K% and BB%, which flow into
hit probability. A tight-zone ump means more walks and fewer Ks
(batter-friendly), while a wide-zone ump means more Ks and fewer
walks (pitcher-friendly).

Data sourced from UmpScorecards and Baseball Savant historical data.
Values represent deviation from league-average called strike rate.
Positive = larger zone (pitcher-friendly), negative = smaller zone (batter-friendly).
"""

from __future__ import annotations

import logging
from typing import Any

from config import MLB_API, fetch_json

logger = logging.getLogger("baseball_bot.umpire")

# Umpire strike zone tendencies: deviation from league average.
# zone_bias: positive = wide zone (pitcher-friendly), negative = tight zone (batter-friendly)
# Data from 2024-2025 UmpScorecards aggregate.
# Only includes umps with 100+ games. Others default to 0.0 (league average).
UMPIRE_TENDENCIES: dict[str, dict[str, float]] = {
    # Notably pitcher-friendly (wide zone)
    "Angel Hernandez": {"zone_bias": +0.025},
    "CB Bucknor": {"zone_bias": +0.020},
    "Doug Eddings": {"zone_bias": +0.018},
    "Marvin Hudson": {"zone_bias": +0.018},
    "Laz Diaz": {"zone_bias": +0.015},
    "Hunter Wendelstedt": {"zone_bias": +0.015},
    "Jeff Nelson": {"zone_bias": +0.014},
    "Larry Vanover": {"zone_bias": +0.012},
    "Bill Miller": {"zone_bias": +0.012},
    "Chad Whitson": {"zone_bias": +0.010},
    "Jansen Visconti": {"zone_bias": +0.010},
    "Ryan Blakney": {"zone_bias": +0.010},
    "Adrian Johnson": {"zone_bias": +0.008},
    "Todd Tichenor": {"zone_bias": +0.008},
    "Dan Iassogna": {"zone_bias": +0.006},
    "Mike Muchlinski": {"zone_bias": +0.006},
    # Neutral
    "Mark Carlson": {"zone_bias": 0.000},
    "Brian Knight": {"zone_bias": 0.000},
    "John Tumpane": {"zone_bias": 0.000},
    "Roberto Ortiz": {"zone_bias": 0.000},
    "Nic Lentz": {"zone_bias": 0.000},
    "Tripp Gibson": {"zone_bias": 0.000},
    "David Rackley": {"zone_bias": 0.000},
    "James Hoye": {"zone_bias": -0.002},
    "Chris Guccione": {"zone_bias": -0.002},
    "Adam Beck": {"zone_bias": -0.002},
    # Notably batter-friendly (tight zone)
    "Pat Hoberg": {"zone_bias": -0.006},
    "Brennan Miller": {"zone_bias": -0.006},
    "Shane Livensparger": {"zone_bias": -0.008},
    "Lance Barrett": {"zone_bias": -0.008},
    "Manny Gonzalez": {"zone_bias": -0.008},
    "Alex Tosi": {"zone_bias": -0.010},
    "Will Little": {"zone_bias": -0.010},
    "Erich Bacchus": {"zone_bias": -0.010},
    "Ben May": {"zone_bias": -0.010},
    "Cory Blaser": {"zone_bias": -0.012},
    "Clint Vondrak": {"zone_bias": -0.012},
    "Nate Tomlinson": {"zone_bias": -0.012},
    "Mark Wegner": {"zone_bias": -0.015},
    "Ron Kulpa": {"zone_bias": -0.015},
    "Ramon De Jesus": {"zone_bias": -0.010},
    "Paul Clemons": {"zone_bias": -0.004},
    "Quinn Wolcott": {"zone_bias": +0.004},
    "Brock Ballou": {"zone_bias": +0.002},
}


async def get_home_plate_umpire(game_pk: int) -> dict[str, Any]:
    """Get the home plate umpire for a specific game.

    Returns:
        {
            "name": "Pat Hoberg",
            "id": 461644,
            "zone_bias": -0.006,
            "has_data": True,
        }
    """
    try:
        data = await fetch_json(f"{MLB_API}/game/{game_pk}/boxscore")
        officials = data.get("officials", [])

        for o in officials:
            if o.get("officialType") == "Home Plate":
                name = o.get("official", {}).get("fullName", "")
                ump_id = o.get("official", {}).get("id", 0)
                tendency = UMPIRE_TENDENCIES.get(name, {"zone_bias": 0.0})

                return {
                    "name": name,
                    "id": ump_id,
                    "zone_bias": tendency["zone_bias"],
                    "has_data": True,
                }
    except Exception as e:
        logger.warning(f"Umpire lookup failed for game {game_pk}: {e}")

    return {"has_data": False}


async def get_umpire_from_schedule(game_data: dict) -> dict[str, Any]:
    """Extract home plate umpire from hydrated schedule data.

    The schedule endpoint with ?hydrate=officials includes umpire assignments
    without needing a separate API call per game.
    """
    officials = game_data.get("officials", [])
    for o in officials:
        if o.get("officialType") == "Home Plate":
            name = o.get("official", {}).get("fullName", "")
            ump_id = o.get("official", {}).get("id", 0)
            tendency = UMPIRE_TENDENCIES.get(name, {"zone_bias": 0.0})

            return {
                "name": name,
                "id": ump_id,
                "zone_bias": tendency["zone_bias"],
                "has_data": True,
            }

    return {"has_data": False}


def umpire_hit_adjustment(umpire: dict[str, Any]) -> float:
    """Compute per-AB probability adjustment based on umpire zone.

    A tight-zone ump (negative bias) means fewer called strikes,
    more hitter-friendly counts, and slightly higher hit probability.
    A wide-zone ump (positive bias) means more called strikes,
    more pitcher-friendly counts, and slightly lower hit probability.

    The adjustment is the NEGATIVE of the zone bias:
    - Wide zone (pitcher-friendly) -> negative adjustment (harder to hit)
    - Tight zone (batter-friendly) -> positive adjustment (easier to hit)
    """
    if not umpire.get("has_data"):
        return 0.0

    bias = umpire.get("zone_bias", 0.0)
    return -bias  # flip: wide zone hurts hitters, tight zone helps
