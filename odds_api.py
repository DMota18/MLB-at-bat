"""
The Odds API integration for MLB batter hit props.

Fetches real bookmaker odds for "batter hits" markets and compares
against the model's predicted probabilities to find +EV opportunities.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from config import american_to_prob, get_client

logger = logging.getLogger("baseball_bot.odds")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "baseball_mlb"


def _api_key() -> str | None:
    return os.environ.get("ODDS_API_KEY")


async def get_events() -> list[dict]:
    """Fetch today's MLB events (free, no quota cost)."""
    key = _api_key()
    if not key:
        return []

    try:
        client = get_client()
        r = await client.get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events",
            params={"apiKey": key},
        )
        r.raise_for_status()
        _log_quota(r)
        return r.json()
    except Exception as e:
        logger.error(f"Failed to fetch events: {e}")
        return []


async def get_hit_props(event_id: str) -> dict[str, list[dict]]:
    """Fetch batter_hits odds for a single event.

    Returns:
        {
            "Player Name": [
                {"book": "fanduel", "line": 0.5, "over": -180, "under": +140},
                ...
            ]
        }

    Costs 1 API credit per call.
    """
    key = _api_key()
    if not key:
        return {}

    try:
        client = get_client()
        r = await client.get(
            f"{ODDS_API_BASE}/sports/{SPORT}/events/{event_id}/odds",
            params={
                "apiKey": key,
                "regions": "us",
                "markets": "batter_hits",
                "oddsFormat": "american",
            },
        )
        r.raise_for_status()
        _log_quota(r)
        data = r.json()
    except Exception as e:
        logger.error(f"Failed to fetch hit props for {event_id}: {e}")
        return {}

    return _parse_hit_props(data)


async def get_hit_props_batch(event_ids: list[str]) -> dict[str, dict[str, list[dict]]]:
    """Fetch hit props for multiple events.

    Returns: {event_id: {player_name: [odds entries]}}
    """
    results = {}
    for eid in event_ids:
        props = await get_hit_props(eid)
        if props:
            results[eid] = props
    return results


def _parse_hit_props(data: dict) -> dict[str, list[dict]]:
    """Parse odds API response into player -> odds mapping.

    The Odds API format for player props:
        name: "Over" or "Under"
        description: "Player Name"
        price: American odds (e.g., -180, +150)
        point: line (e.g., 0.5 for 1+ hits, 1.5 for 2+ hits)
    """
    players: dict[str, list[dict]] = {}

    for bookmaker in data.get("bookmakers", []):
        book = bookmaker.get("title", bookmaker.get("key", "unknown"))
        for market in bookmaker.get("markets", []):
            if market.get("key") != "batter_hits":
                continue

            outcomes = market.get("outcomes", [])
            for outcome in outcomes:
                side_raw = outcome.get("name", "")  # "Over" or "Under"
                player = outcome.get("description", "")  # Player name
                price = outcome.get("price", 0)
                point = outcome.get("point", 0.5)

                if not player:
                    continue

                side = side_raw.lower()  # "over" or "under"
                if side not in ("over", "under"):
                    continue

                if player not in players:
                    players[player] = []

                # Check if we already have this book + line for this player
                existing = None
                for entry in players[player]:
                    if entry["book"] == book and entry["line"] == point:
                        existing = entry
                        break

                if existing:
                    existing[side] = price
                else:
                    entry = {"book": book, "line": point, side: price}
                    players[player].append(entry)

    return players


def find_best_odds(player_odds: list[dict]) -> dict[str, Any] | None:
    """Find the best available odds across books for 0.5 line (1+ hits).

    Returns: {"best_book": "FanDuel", "best_over": -150, "implied_prob": 0.60, ...}
    """
    if not player_odds:
        return None

    # Filter to 0.5 line (over 0.5 = at least 1 hit)
    half_line = [o for o in player_odds if o.get("line", 0.5) == 0.5]
    if not half_line:
        # Try any line
        half_line = player_odds

    best_over = None
    best_book = None

    for entry in half_line:
        over_price = entry.get("over")
        if over_price is None:
            continue

        # Best over = least negative (closest to even)
        if best_over is None or over_price > best_over:
            best_over = over_price
            best_book = entry["book"]

    if best_over is None:
        return None

    # Calculate implied probability from best odds
    implied = american_to_prob(best_over)

    return {
        "best_book": best_book,
        "best_over": best_over,
        "implied_prob": round(implied, 3),
        "all_books": [
            {"book": e["book"], "over": e.get("over"), "under": e.get("under"), "line": e.get("line", 0.5)}
            for e in half_line if e.get("over") is not None
        ],
    }


def match_event_to_game(events: list[dict], home_team: str, away_team: str) -> str | None:
    """Match an Odds API event to an MLB game by team names.

    The Odds API uses slightly different team name formats,
    so we do fuzzy matching on key words.
    """
    home_words = _team_keywords(home_team)
    away_words = _team_keywords(away_team)

    for event in events:
        h = event.get("home_team", "")
        a = event.get("away_team", "")
        h_words = _team_keywords(h)
        a_words = _team_keywords(a)

        if (home_words & h_words) and (away_words & a_words):
            return event.get("id")

    return None


def _team_keywords(name: str) -> set[str]:
    """Extract matching keywords from a team name."""
    # "New York Yankees" -> {"yankees"}
    # "Los Angeles Dodgers" -> {"dodgers"}
    skip = {"new", "york", "los", "angeles", "san", "francisco", "diego",
            "st.", "st", "louis", "kansas", "city", "tampa", "bay"}
    words = set()
    for w in name.lower().split():
        if w not in skip:
            words.add(w)
    return words


def _log_quota(r) -> None:
    """Log API quota usage from response headers."""
    remaining = r.headers.get("x-requests-remaining")
    used = r.headers.get("x-requests-used")
    if remaining:
        logger.info(f"Odds API quota: {used} used, {remaining} remaining")
