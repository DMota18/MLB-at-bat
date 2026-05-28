"""
MLB API data fetchers for batter and pitcher stats.

Pulls season stats, advanced metrics, sabermetrics, platoon splits,
and recent form for matchup analysis.
"""

from __future__ import annotations

import logging

from config import MLB_API, CURRENT_SEASON, PREVIOUS_SEASON, fetch_json

logger = logging.getLogger("baseball_bot.fetchers")


async def _get_splits(pid: int, stat_type: str, season: int, group: str) -> list:
    try:
        data = await fetch_json(f"{MLB_API}/people/{pid}/stats?stats={stat_type}&season={season}&group={group}")
        return data.get("stats", [{}])[0].get("splits", [])
    except Exception as e:
        logger.warning(f"Stats lookup failed for player {pid} ({stat_type}/{group}): {e}")
        return []


def _pct(num: int | float | str, denom: int | float | str) -> str:
    """Format as percentage string."""
    try:
        n, d = float(num), float(denom)
        return f"{n/d*100:.1f}%" if d > 0 else "---"
    except (TypeError, ValueError, ZeroDivisionError):
        return "---"


async def get_batter_data(pid: int) -> dict:
    """Pull comprehensive batter data for matchup analysis."""
    result = {"id": pid}

    # Current season stats
    splits = await _get_splits(pid, "season", CURRENT_SEASON, "hitting")
    if splits:
        s = splits[0]["stat"]
        result["season"] = {
            "g": s.get("gamesPlayed", 0),
            "pa": s.get("plateAppearances", 0),
            "avg": s.get("avg", "---"),
            "obp": s.get("obp", "---"),
            "slg": s.get("slg", "---"),
            "ops": s.get("ops", "---"),
            "hr": s.get("homeRuns", 0),
            "k": s.get("strikeOuts", 0),
            "bb": s.get("baseOnBalls", 0),
            "h": s.get("hits", 0),
            "ab": s.get("atBats", 0),
        }

    # Current season advanced stats (BABIP, ISO, K%, BB%, whiff)
    adv = await _get_splits(pid, "seasonAdvanced", CURRENT_SEASON, "hitting")
    if adv:
        s = adv[0]["stat"]
        result["advanced"] = {
            "babip": s.get("babip", "---"),
            "iso": s.get("iso", "---"),
            "k_pct": s.get("strikeoutsPerPlateAppearance", "---"),
            "bb_pct": s.get("walksPerPlateAppearance", "---"),
            "ppa": s.get("pitchesPerPlateAppearance", "---"),
            "ground_outs": s.get("groundOuts", 0),
            "fly_outs": s.get("flyOuts", 0),
            "line_outs": s.get("lineOuts", 0),
        }

    # Sabermetrics (wOBA)
    saber = await _get_splits(pid, "sabermetrics", PREVIOUS_SEASON, "hitting")
    if saber:
        s = saber[0]["stat"]
        result["saber"] = {
            "woba": round(s["woba"], 3) if s.get("woba") else "---",
        }

    # Platoon splits — blend current + previous season, weighted by PA.
    # Previous season provides reliability; current season captures changes.
    plat_prev = await _get_splits(pid, "statSplits&sitCodes=vl,vr", PREVIOUS_SEASON, "hitting")
    plat_curr = await _get_splits(pid, "statSplits&sitCodes=vl,vr", CURRENT_SEASON, "hitting")

    def _blend_platoon(prev_splits, curr_splits):
        platoon = {}
        for side_label, key in [("Left", "vs_L"), ("Right", "vs_R")]:
            prev = next((s["stat"] for s in prev_splits if side_label in s.get("split", {}).get("description", "")), None)
            curr = next((s["stat"] for s in curr_splits if side_label in s.get("split", {}).get("description", "")), None)
            prev_pa = int(prev.get("plateAppearances", 0)) if prev else 0
            curr_pa = int(curr.get("plateAppearances", 0)) if curr else 0
            total_pa = prev_pa + curr_pa
            if total_pa == 0:
                continue
            # Weighted blend by PA
            def blend(stat_name):
                p_val = float(prev.get(stat_name, "0") if prev else "0") if prev_pa > 0 else 0
                c_val = float(curr.get(stat_name, "0") if curr else "0") if curr_pa > 0 else 0
                if prev_pa == 0:
                    return f"{c_val:.3f}"
                if curr_pa == 0:
                    return f"{p_val:.3f}"
                return f"{(p_val * prev_pa + c_val * curr_pa) / total_pa:.3f}"
            platoon[key] = {"avg": blend("avg"), "obp": blend("obp"),
                            "slg": blend("slg"), "pa": total_pa}
        return platoon

    platoon = _blend_platoon(plat_prev, plat_curr)
    if platoon:
        result["platoon"] = platoon

    # Last 7 games
    recent = await _get_splits(pid, "lastXGames&limit=7", CURRENT_SEASON, "hitting")
    if recent:
        s = recent[0]["stat"]
        result["last7"] = {
            "avg": s.get("avg", "---"),
            "ops": s.get("ops", "---"),
            "hr": s.get("homeRuns", 0),
            "h": s.get("hits", 0),
            "ab": s.get("atBats", 0),
            "k": s.get("strikeOuts", 0),
        }

    return result


async def get_pitcher_data(pid: int) -> dict:
    """Pull comprehensive pitcher data."""
    result = {"id": pid}

    # Current season
    splits = await _get_splits(pid, "season", CURRENT_SEASON, "pitching")
    if splits:
        s = splits[0]["stat"]
        result["season"] = {
            "gs": s.get("gamesStarted", 0),
            "ip": s.get("inningsPitched", "0"),
            "era": s.get("era", "-.--"),
            "whip": s.get("whip", "-.--"),
            "k": s.get("strikeOuts", 0),
            "bb": s.get("baseOnBalls", 0),
            "hr": s.get("homeRuns", 0),
            "avg": s.get("avg", "---"),  # avg against
            "k9": s.get("strikeoutsPer9Inn", "-.--"),
        }

    # Advanced (BABIP, whiff%, ground ball tendencies)
    adv = await _get_splits(pid, "seasonAdvanced", PREVIOUS_SEASON, "pitching")
    if adv:
        s = adv[0]["stat"]
        result["advanced"] = {
            "babip": s.get("babip", "---"),
            "hr9": s.get("homeRunsPer9", "---"),
            "whiff_pct": s.get("whiffPercentage", "---"),
            "strike_pct": s.get("strikePercentage", "---"),
            "ground_outs": s.get("groundOuts", 0),
            "fly_outs": s.get("flyOuts", 0),
            "gb_pct": "---",
        }
        go = s.get("groundOuts", 0)
        fo = s.get("flyOuts", 0)
        if go + fo > 0:
            result["advanced"]["gb_pct"] = f"{go/(go+fo)*100:.0f}%"

    # Sabermetrics (FIP, xFIP)
    saber = await _get_splits(pid, "sabermetrics", PREVIOUS_SEASON, "pitching")
    if saber:
        s = saber[0]["stat"]
        result["saber"] = {
            "fip": round(s["fip"], 2) if s.get("fip") else "---",
            "xfip": round(s["xfip"], 2) if s.get("xfip") else "---",
        }

    # Platoon splits (vs LHB/RHB)
    plat = await _get_splits(pid, "statSplits&sitCodes=vl,vr", PREVIOUS_SEASON, "pitching")
    platoon = {}
    for s in plat:
        side = s.get("split", {}).get("description", "")
        stat = s.get("stat", {})
        if "Left" in side:
            platoon["vs_LHB"] = {"avg": stat.get("avg", "---"), "obp": stat.get("obp", "---"),
                                  "slg": stat.get("slg", "---"), "k": stat.get("strikeOuts", 0),
                                  "bb": stat.get("baseOnBalls", 0), "hr": stat.get("homeRuns", 0)}
        elif "Right" in side:
            platoon["vs_RHB"] = {"avg": stat.get("avg", "---"), "obp": stat.get("obp", "---"),
                                  "slg": stat.get("slg", "---"), "k": stat.get("strikeOuts", 0),
                                  "bb": stat.get("baseOnBalls", 0), "hr": stat.get("homeRuns", 0)}
    if platoon:
        result["platoon"] = platoon

    return result
