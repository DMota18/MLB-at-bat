"""
Advanced matchup data fetchers.

Pulls head-to-head history, pitch type matchups, Statcast contact
quality, and pitcher recent form from MLB API and Baseball Savant.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Any

from collections import defaultdict

from config import MLB_API, SAVANT_CSV, CURRENT_SEASON, PREVIOUS_SEASON, fetch_json, fetch_text

logger = logging.getLogger("baseball_bot.matchup")


# ── Head to head ─────────────────────────────────────────────────────


async def get_h2h(batter_id: int, pitcher_id: int) -> dict[str, Any]:
    """Get career head-to-head stats for batter vs pitcher."""
    try:
        data = await fetch_json(
            f"{MLB_API}/people/{batter_id}/stats"
            f"?stats=vsPlayerTotal&opposingPlayerId={pitcher_id}&group=hitting"
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}

        stat = splits[0].get("stat", {})
        ab = int(stat.get("atBats", 0))
        if ab == 0:
            return {}

        return {
            "ab": ab,
            "hits": int(stat.get("hits", 0)),
            "hr": int(stat.get("homeRuns", 0)),
            "k": int(stat.get("strikeOuts", 0)),
            "bb": int(stat.get("baseOnBalls", 0)),
            "avg": stat.get("avg", "---"),
            "ops": stat.get("ops", "---"),
            "pa": int(stat.get("plateAppearances", ab)),
        }
    except Exception as e:
        logger.warning(f"H2H lookup failed for {batter_id} vs {pitcher_id}: {e}")
        return {}


# ── Pitcher arsenal ──────────────────────────────────────────────────


async def get_pitcher_arsenal(pitcher_id: int, season: int | None = None) -> list[dict[str, Any]]:
    """Get pitcher's pitch arsenal with types, usage, and velocity."""
    season = season or PREVIOUS_SEASON
    try:
        data = await fetch_json(
            f"{MLB_API}/people/{pitcher_id}/stats"
            f"?stats=pitchArsenal&season={season}&group=pitching"
        )
        stats_list = data.get("stats", [])
        if not stats_list:
            # Try current season as fallback
            data = await fetch_json(
                f"{MLB_API}/people/{pitcher_id}/stats"
                f"?stats=pitchArsenal&season={CURRENT_SEASON}&group=pitching"
            )
            stats_list = data.get("stats", [])
        if not stats_list:
            return []
        splits = stats_list[0].get("splits", [])

        arsenal = []
        for s in splits:
            stat = s.get("stat", {})
            pt = stat.get("type", {})
            code = pt.get("code", "")
            if not code:
                continue
            arsenal.append({
                "code": code,
                "name": pt.get("description", code),
                "usage": round(stat.get("percentage", 0) * 100, 1),
                "velo": round(stat.get("averageSpeed", 0), 1),
            })

        arsenal.sort(key=lambda x: -x["usage"])
        return arsenal
    except Exception as e:
        logger.warning(f"Arsenal lookup failed for {pitcher_id}: {e}")
        return []


# ── xBA computation ──────────────────────────────────────────────────


def _xba_from_ev(ev: float) -> float:
    """Estimate single-batted-ball hit probability from exit velocity.

    Based on empirical MLB data (2020-2025 Statcast):
    - Below 70 mph: ~.150 xBA (weak contact, mostly outs)
    - 70-80 mph: ~.200 (soft grounders/popups)
    - 80-90 mph: ~.230 (medium contact)
    - 90-95 mph: ~.290 (hard contact, line drives)
    - 95-100 mph: ~.450 (barrels start here)
    - 100-105 mph: ~.600 (hard line drives / barrels)
    - 105-110 mph: ~.700 (elite contact)
    - 110+ mph: ~.800 (near-guaranteed hits)

    This is a simplified model; full xBA uses launch angle too.
    But EV alone explains ~70% of xBA variance.
    """
    if ev < 60:
        return 0.100
    elif ev < 70:
        return 0.100 + (ev - 60) * 0.005  # .100 → .150
    elif ev < 80:
        return 0.150 + (ev - 70) * 0.005  # .150 → .200
    elif ev < 90:
        return 0.200 + (ev - 80) * 0.004  # .200 → .240
    elif ev < 95:
        return 0.240 + (ev - 90) * 0.016  # .240 → .320
    elif ev < 100:
        return 0.320 + (ev - 95) * 0.030  # .320 → .470
    elif ev < 105:
        return 0.470 + (ev - 100) * 0.030  # .470 → .620
    elif ev < 110:
        return 0.620 + (ev - 105) * 0.024  # .620 → .740
    else:
        return min(0.900, 0.740 + (ev - 110) * 0.015)


def _compute_xba(
    exit_velos: list[float],
    batted_balls: int,
    hard_hits: int,
    barrels: int,
) -> float | None:
    """Compute expected batting average from exit velocity distribution.

    Returns the average hit probability across all batted ball events,
    which approximates Statcast xBA. Requires 20+ batted balls for
    stability.
    """
    if batted_balls < 20 or not exit_velos:
        return None

    xba_sum = sum(_xba_from_ev(ev) for ev in exit_velos)
    return round(xba_sum / len(exit_velos), 3)


# ── Batter Statcast (pitch types + contact quality in one call) ──────

# Per-session cache: avoids duplicate Savant requests when both
# get_batter_vs_pitch_types() and get_batter_statcast() are called
# for the same batter in the same game card analysis.
_statcast_cache: dict[tuple[int, int], dict[str, Any]] = {}


def clear_statcast_cache() -> None:
    """Clear the per-session Statcast cache (call between game cards)."""
    _statcast_cache.clear()


async def get_batter_statcast_all(
    batter_id: int, season: int | None = None,
) -> dict[str, Any]:
    """Fetch batter's Statcast data in a SINGLE call.

    Returns both pitch-type batting stats AND contact quality metrics,
    replacing the old get_batter_vs_pitch_types + get_batter_statcast
    which made two identical requests.

    Results are cached per (batter_id, season) for the session to avoid
    duplicate API calls when multiple wrappers call this function.

    Returns:
        {
            "pitch_stats": {code: {ab, hits, avg, hr, k}},
            "contact": {avg_exit_velo, hard_hit_pct, barrel_pct, batted_balls, has_data},
        }
    """
    season = season or CURRENT_SEASON

    # Check cache first
    cache_key = (batter_id, season)
    if cache_key in _statcast_cache:
        return _statcast_cache[cache_key]

    # Use current season if past mid-April (enough data), otherwise previous
    from datetime import date
    today = date.today()
    if today.month >= 5 or (today.month == 4 and today.day >= 20):
        date_gt = f"{CURRENT_SEASON}-03-01"
        date_lt = f"{CURRENT_SEASON}-11-01"
    else:
        date_gt = f"{PREVIOUS_SEASON}-04-01"
        date_lt = f"{PREVIOUS_SEASON}-10-01"

    params = {
        "all": "true",
        "player_type": "batter",
        "batters_lookup[]": str(batter_id),
        "game_date_gt": date_gt,
        "game_date_lt": date_lt,
        "type": "details",
        "sort_col": "pitches",
        "sort_order": "desc",
        "min_pitches": "0",
        "min_results": "0",
        "group_by": "name",
    }

    empty = {
        "pitch_stats": {},
        "contact": {"has_data": False},
    }

    try:
        # Stagger Savant requests to avoid rate limiting on VPS
        import asyncio
        await asyncio.sleep(0.3)

        text = await fetch_text(SAVANT_CSV, params=params)
        if len(text) < 200:
            logger.warning(f"Savant returned short response for {batter_id}: {len(text)} bytes")
            return empty

        # Strip BOM that Baseball Savant prepends to CSV responses
        text = text.lstrip("\ufeff")

        reader = csv.reader(io.StringIO(text))
        header = next(reader, None)
        if not header:
            return empty

        # Strip whitespace/quotes from header names for reliable matching
        header = [h.strip().strip('"') for h in header]

        def col_idx(name: str) -> int | None:
            try:
                return header.index(name)
            except ValueError:
                return None

        pt_idx = col_idx("pitch_type")
        events_idx = col_idx("events")
        ev_idx = col_idx("launch_speed")
        la_idx = col_idx("launch_angle")

        if pt_idx is None or events_idx is None:
            return empty

        # Accumulators
        pitch_stats: dict[str, dict] = {}
        hit_events = {"single", "double", "triple", "home_run"}
        exit_velos: list[float] = []
        hard_hits = 0
        barrels = 0
        batted_balls = 0

        for cols in reader:
            if len(cols) <= max(pt_idx, events_idx, ev_idx or 0, la_idx or 0):
                continue

            pt = cols[pt_idx].strip().strip('"')
            event = cols[events_idx].strip().strip('"')

            # ── Pitch type stats ──
            if pt and event:
                if pt not in pitch_stats:
                    pitch_stats[pt] = {"ab": 0, "hits": 0, "hr": 0, "k": 0}

                pitch_stats[pt]["ab"] += 1
                if event in hit_events:
                    pitch_stats[pt]["hits"] += 1
                if event == "home_run":
                    pitch_stats[pt]["hr"] += 1
                if event in ("strikeout", "strikeout_double_play"):
                    pitch_stats[pt]["k"] += 1

            # ── Contact quality ──
            if ev_idx is not None:
                ev_str = cols[ev_idx].strip().strip('"')
                if ev_str and ev_str != "null":
                    try:
                        ev = float(ev_str)
                    except ValueError:
                        continue
                    if ev > 0:
                        batted_balls += 1
                        exit_velos.append(ev)
                        if ev >= 95:
                            hard_hits += 1
                        if la_idx is not None:
                            la_str = cols[la_idx].strip().strip('"')
                            try:
                                la = float(la_str)
                                if ev >= 98 and 26 <= la <= 30 + (ev - 98) * 2:
                                    barrels += 1
                            except (ValueError, TypeError):
                                pass

        # Calculate pitch type averages
        for stats in pitch_stats.values():
            ab = stats["ab"]
            stats["avg"] = round(stats["hits"] / ab, 3) if ab > 0 else 0

        # Build contact result
        contact: dict[str, Any] = {"has_data": False}
        if batted_balls >= 10:
            contact = {
                "avg_exit_velo": round(sum(exit_velos) / len(exit_velos), 1),
                "hard_hit_pct": round(hard_hits / batted_balls, 3),
                "barrel_pct": round(barrels / batted_balls, 3),
                "batted_balls": batted_balls,
                "has_data": True,
                "xba": _compute_xba(exit_velos, batted_balls, hard_hits, barrels),
            }

        result = {"pitch_stats": pitch_stats, "contact": contact}
        _statcast_cache[cache_key] = result
        return result

    except Exception as e:
        logger.warning(f"Statcast lookup failed for {batter_id}: {e}")
        return empty


def compute_arsenal_matchup(
    batter_pitch_stats: dict[str, dict],
    pitcher_arsenal: list[dict],
) -> dict[str, Any]:
    """Compute weighted batting average against a pitcher's specific arsenal.

    Weights each pitch type by the pitcher's usage, uses the batter's
    AVG against that pitch type.
    """
    if not batter_pitch_stats or not pitcher_arsenal:
        return {"weighted_avg": None, "pitch_breakdown": [], "has_data": False}

    total_weight = 0
    weighted_sum = 0
    breakdown = []

    for pitch in pitcher_arsenal:
        code = pitch["code"]
        usage = pitch["usage"] / 100

        if code in batter_pitch_stats:
            batter_stats = batter_pitch_stats[code]
            ab = batter_stats["ab"]
            avg = batter_stats["avg"]

            if ab >= 5:
                weighted_sum += avg * usage
                total_weight += usage
                breakdown.append({
                    "pitch": code,
                    "name": pitch["name"],
                    "pitcher_usage": pitch["usage"],
                    "batter_ab": ab,
                    "batter_avg": avg,
                    "batter_hr": batter_stats["hr"],
                    "batter_k": batter_stats["k"],
                })

    if total_weight < 0.3:
        return {"weighted_avg": None, "pitch_breakdown": breakdown, "has_data": False}

    weighted_avg = round(weighted_sum / total_weight, 3)

    return {
        "weighted_avg": weighted_avg,
        "pitch_breakdown": breakdown,
        "has_data": True,
    }


# ── Pitcher recent form (last 3 starts) ─────────────────────────


async def get_pitcher_recent_starts(pitcher_id: int, num_starts: int = 3) -> dict[str, Any]:
    """Fetch pitcher's last N starts and compute recent form metrics."""
    try:
        data = await fetch_json(
            f"{MLB_API}/people/{pitcher_id}/stats"
            f"?stats=gameLog&season={CURRENT_SEASON}&group=pitching"
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {"has_data": False}

        starts = [s for s in splits if int(s.get("stat", {}).get("gamesStarted", 0)) > 0]
        recent = starts[:num_starts]

        if not recent:
            return {"has_data": False}

        total_ip = 0.0
        total_h = 0
        total_ab = 0
        total_er = 0
        total_k = 0
        total_bb = 0

        for game in recent:
            stat = game.get("stat", {})
            ip_str = str(stat.get("inningsPitched", "0"))
            try:
                parts = ip_str.split(".")
                thirds = min(int(parts[1]), 2) if len(parts) > 1 else 0
                ip = int(parts[0]) + thirds / 3
            except (ValueError, IndexError):
                ip = 0
            total_ip += ip
            total_h += int(stat.get("hits", 0))
            total_ab += int(stat.get("atBats", 0))
            total_er += int(stat.get("earnedRuns", 0))
            total_k += int(stat.get("strikeOuts", 0))
            total_bb += int(stat.get("baseOnBalls", 0))

        if total_ip == 0:
            return {"has_data": False}

        return {
            "starts": len(recent),
            "ip": round(total_ip, 1),
            "avg_against": round(total_h / total_ab, 3) if total_ab > 0 else 0.248,
            "era": round(total_er / total_ip * 9, 2),
            "k_per_9": round(total_k / total_ip * 9, 1),
            "bb_per_9": round(total_bb / total_ip * 9, 1),
            "has_data": True,
        }
    except Exception as e:
        logger.warning(f"Pitcher recent starts failed for {pitcher_id}: {e}")
        return {"has_data": False}


# ── Batter pitch type stats via MLB Stats API ──────────────────────
# Aggregates the playLog endpoint into per-pitch-type batting stats.
# Unlike Savant, this endpoint works reliably from VPS IPs.


async def get_batter_pitch_type_stats(
    batter_id: int, season: int | None = None,
) -> dict[str, dict]:
    """Get batter's batting stats against each pitch type via MLB Stats API.

    Uses the playLog endpoint and aggregates at-bats by the pitch type
    on the final pitch of each plate appearance. Returns the same shape
    as the old Savant-based get_batter_vs_pitch_types() so
    compute_arsenal_matchup() works without changes.
    """
    season = season or CURRENT_SEASON
    try:
        data = await fetch_json(
            f"{MLB_API}/people/{batter_id}/stats"
            f"?stats=playLog&season={season}&group=hitting"
        )
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return {}

        hit_events = {"single", "double", "triple", "home_run"}
        stats: dict[str, dict] = defaultdict(lambda: {"ab": 0, "hits": 0, "hr": 0, "k": 0})

        for s in splits:
            details = s.get("stat", {}).get("play", {}).get("details", {})
            if not details.get("isAtBat"):
                continue
            pt_code = details.get("type", {}).get("code", "")
            if not pt_code:
                continue

            stats[pt_code]["ab"] += 1
            if details.get("isBaseHit"):
                stats[pt_code]["hits"] += 1
            event = details.get("event", "")
            if event == "home_run":
                stats[pt_code]["hr"] += 1
            if event in ("strikeout", "strikeout_double_play"):
                stats[pt_code]["k"] += 1

        for s in stats.values():
            s["avg"] = round(s["hits"] / s["ab"], 3) if s["ab"] > 0 else 0

        return dict(stats)
    except Exception as e:
        logger.warning(f"Batter pitch type stats failed for {batter_id}: {e}")
        return {}


# ── Legacy compatibility wrappers ────────────────────────────────
# These wrap the combined get_batter_statcast_all for callers that
# expect the old separate interfaces.


async def get_batter_vs_pitch_types(batter_id: int) -> dict[str, dict]:
    """Get batter's stats against each pitch type. Wrapper around combined call."""
    result = await get_batter_statcast_all(batter_id)
    return result.get("pitch_stats", {})


async def get_batter_statcast(batter_id: int) -> dict[str, Any]:
    """Get batter's contact quality metrics.

    Checks the local Statcast cache first (populated by statcast_cache.py).
    Falls back to a live Savant call if the cache misses (unlikely to
    succeed on a VPS, but works from residential IPs).
    """
    cached = _read_statcast_cache(batter_id)
    if cached:
        return cached
    result = await get_batter_statcast_all(batter_id)
    return result.get("contact", {"has_data": False})


def _read_statcast_cache(batter_id: int) -> dict[str, Any] | None:
    """Read a batter's Statcast data from the local cache file."""
    import json
    from pathlib import Path

    cache_path = Path(__file__).parent / "statcast_cache.json"
    if not cache_path.exists():
        return None

    if not hasattr(_read_statcast_cache, "_data"):
        try:
            _read_statcast_cache._data = json.loads(cache_path.read_text())
        except Exception:
            _read_statcast_cache._data = {}

    return _read_statcast_cache._data.get(str(batter_id))
