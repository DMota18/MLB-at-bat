"""
Telegram message formatters for pregame cards and matchup display.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime

from config import CURRENT_SEASON, PREVIOUS_SEASON, prob_to_american
from data_fetchers import get_batter_data, get_pitcher_data
from lineup_detector import GameLineup, PlayerInfo, _player_bios
from matchup_data import (
    get_h2h, get_pitcher_arsenal, get_batter_vs_pitch_types,
    compute_arsenal_matchup, get_pitcher_recent_starts, get_batter_statcast,
)
from predictor import predict_hit, HitPrediction
from tracker import save_predictions
from umpire import get_umpire_from_schedule, umpire_hit_adjustment
from weather import get_game_weather, weather_hit_adjustment
from ab_testing import run_shadow_prediction

logger = logging.getLogger("baseball_bot.formatters")

PARK_FACTORS = {
    # Hitter-friendly (runs >= 1.05)
    "Coors Field": {"hr": 1.35, "runs": 1.28, "tag": "\U0001f4a5"},
    "Great American Ball Park": {"hr": 1.18, "runs": 1.12, "tag": "\U0001f4a5"},
    "Yankee Stadium": {"hr": 1.15, "runs": 1.08, "tag": "\U0001f4a5"},
    "Citizens Bank Park": {"hr": 1.12, "runs": 1.08, "tag": "\U0001f4a5"},
    "Chase Field": {"hr": 1.12, "runs": 1.08, "tag": "\U0001f4a5"},
    "Globe Life Field": {"hr": 1.10, "runs": 1.05, "tag": "\U0001f4a5"},
    "Minute Maid Park": {"hr": 1.10, "runs": 1.06, "tag": "\U0001f4a5"},
    "Wrigley Field": {"hr": 1.08, "runs": 1.06, "tag": ""},
    "Fenway Park": {"hr": 0.95, "runs": 1.05, "tag": ""},
    "American Family Field": {"hr": 1.08, "runs": 1.05, "tag": ""},
    # Neutral (0.95 <= runs <= 1.04)
    "Camden Yards": {"hr": 1.05, "runs": 1.03, "tag": ""},
    "Rogers Centre": {"hr": 1.02, "runs": 1.01, "tag": ""},
    "Guaranteed Rate Field": {"hr": 1.00, "runs": 1.00, "tag": ""},
    "Nationals Park": {"hr": 1.00, "runs": 1.00, "tag": ""},
    "Truist Park": {"hr": 0.98, "runs": 1.00, "tag": ""},
    "Target Field": {"hr": 0.98, "runs": 0.99, "tag": ""},
    "Busch Stadium": {"hr": 0.95, "runs": 0.98, "tag": ""},
    "loanDepot park": {"hr": 0.95, "runs": 0.97, "tag": ""},
    "Citi Field": {"hr": 0.95, "runs": 0.97, "tag": ""},
    "Angel Stadium": {"hr": 0.98, "runs": 0.98, "tag": ""},
    "PNC Park": {"hr": 0.92, "runs": 0.96, "tag": ""},
    "Progressive Field": {"hr": 0.95, "runs": 0.97, "tag": ""},
    # Pitcher-friendly (runs < 0.95)
    "Comerica Park": {"hr": 0.92, "runs": 0.95, "tag": ""},
    "Dodger Stadium": {"hr": 0.92, "runs": 0.95, "tag": ""},
    "Kauffman Stadium": {"hr": 0.88, "runs": 0.93, "tag": "\U0001f9ca"},
    "Tropicana Field": {"hr": 0.88, "runs": 0.92, "tag": "\U0001f9ca"},
    "Oakland Coliseum": {"hr": 0.85, "runs": 0.91, "tag": "\U0001f9ca"},
    "Petco Park": {"hr": 0.85, "runs": 0.90, "tag": "\U0001f9ca"},
    "T-Mobile Park": {"hr": 0.82, "runs": 0.90, "tag": "\U0001f9ca"},
    "Oracle Park": {"hr": 0.78, "runs": 0.88, "tag": "\U0001f9ca"},
}

def _find_pitcher_id(name: str) -> int | None:
    """Find pitcher ID by name using the lineup detector's cached bios."""
    for pid, bio in _player_bios.items():
        if bio.get("fullName") == name:
            return pid
    return None


def format_pitcher_block(name: str, throws: str | None, data: dict) -> str:
    """Format pitcher stat block with advanced stats."""
    t = f"({throws}HP)" if throws else ""
    lines = [f"\U0001f3af {name} {t}"]

    s = data.get("season", {})
    if s:
        lines.append(f"  {s.get('era','-.--')} ERA  {s.get('whip','-.--')} WHIP  {s.get('avg','---')} AVG-against")
        lines.append(f"  {s.get('k',0)}K/{s.get('bb',0)}BB in {s.get('ip','0')}IP  K/9: {s.get('k9','-.--')}  {s.get('hr',0)}HR")

    adv = data.get("advanced", {})
    sab = data.get("saber", {})
    if adv or sab:
        parts = []
        if sab.get("fip") and sab["fip"] != "---":
            parts.append(f"FIP {sab['fip']}")
        if sab.get("xfip") and sab["xfip"] != "---":
            parts.append(f"xFIP {sab['xfip']}")
        if adv.get("babip") and adv["babip"] != "---":
            parts.append(f"BABIP {adv['babip']}")
        if adv.get("whiff_pct") and adv["whiff_pct"] != "---":
            parts.append(f"Whiff {adv['whiff_pct']}")
        if adv.get("gb_pct") and adv["gb_pct"] != "---":
            parts.append(f"GB {adv['gb_pct']}")
        if parts:
            lines.append(f"  {' | '.join(parts)}")

    return "\n".join(lines)


def format_batter_matchup(
    batter: PlayerInfo, pitcher_throws: str | None, data: dict,
    pred: HitPrediction | None = None,
) -> str:
    """Format one batter's matchup — condensed view.

    Shows tier + probability, edge tags, and season slash line.
    Advanced stats available via /player command.
    """
    if pred:
        prob_pct = f"{pred.hit_probability * 100:.0f}%"
        fair_odds = prob_to_american(pred.hit_probability)
        lines = [f"{pred.tier_emoji} {batter.batting_order}. {batter.name} ({batter.bats}) — {pred.tier} ({prob_pct} / {fair_odds})"]
        if pred.edge:
            lines[0] += f"  {pred.edge}"
    else:
        lines = [f"{batter.batting_order}. {batter.name} ({batter.bats})"]

    s = data.get("season")
    if s and int(s.get("pa", 0)) > 0:
        lines.append(f"   {s['avg']}/{s['obp']}/{s['slg']}  {s['hr']}HR  ({s['pa']}PA)")
    else:
        lines.append(f"   No {CURRENT_SEASON} stats")

    last7 = data.get("last7")
    if last7 and int(last7.get("ab", 0)) > 0:
        lines.append(f"   Last 7G: {last7['avg']} ({last7['h']}/{last7['ab']}) {last7['hr']}HR {last7['k']}K")

    return "\n".join(lines)


async def build_pregame_card(game: GameLineup) -> str:
    """Build full pregame matchup card."""
    lines = []

    game_time = ""
    if game.game_time:
        try:
            dt = datetime.fromisoformat(game.game_time.replace("Z", "+00:00"))
            game_time = f" - {dt.strftime('%I:%M %p ET')}"
        except Exception:
            pass

    lines.append(f"\u26be {game.away_team} @ {game.home_team}{game_time}")

    pf = PARK_FACTORS.get(game.venue, {"hr": 1.00, "runs": 1.00, "tag": ""})
    lines.append(f"\U0001f3df {game.venue}  HR {pf['hr']}x  Runs {pf['runs']}x {pf.get('tag','')}")

    # Weather
    weather = await get_game_weather(game.venue, game.game_time or "")
    w_adj = weather_hit_adjustment(weather)
    if weather.get("has_data"):
        if weather.get("is_indoor"):
            lines.append("\U0001f3e0 Indoor (climate controlled)")
        else:
            wind_dir = weather.get("wind_direction", 0)
            lines.append(
                f"\U0001f321 {weather['temperature_f']:.0f}\u00b0F  "
                f"\U0001f4a8 {weather['wind_speed_mph']:.0f}mph ({wind_dir}\u00b0)"
            )
    # Umpire
    ump = await get_umpire_from_schedule({"officials": game.officials})
    u_adj = umpire_hit_adjustment(ump)
    if ump.get("has_data"):
        bias = ump["zone_bias"]
        if bias > 0.005:
            zone_label = "wide zone (pitcher-friendly)"
        elif bias < -0.005:
            zone_label = "tight zone (batter-friendly)"
        else:
            zone_label = "neutral zone"
        lines.append(f"\U0001f9d1\u200d\u2696\ufe0f HP: {ump['name']} ({zone_label})")

    lines.append("")

    # Combined environment adjustment
    env_adj = w_adj + u_adj

    # Pitcher blocks — fetch both in parallel
    away_pitcher_data = {}
    home_pitcher_data = {}
    home_pitcher_id = _find_pitcher_id(game.home_pitcher) if game.home_pitcher else 0
    away_pitcher_id = _find_pitcher_id(game.away_pitcher) if game.away_pitcher else 0

    pitcher_tasks = []
    if home_pitcher_id:
        pitcher_tasks.extend([
            get_pitcher_data(home_pitcher_id),
            get_pitcher_arsenal(home_pitcher_id),
            get_pitcher_recent_starts(home_pitcher_id),
        ])
    if away_pitcher_id:
        pitcher_tasks.extend([
            get_pitcher_data(away_pitcher_id),
            get_pitcher_arsenal(away_pitcher_id),
            get_pitcher_recent_starts(away_pitcher_id),
        ])

    pitcher_results = await asyncio.gather(*pitcher_tasks) if pitcher_tasks else []

    idx = 0
    home_arsenal, home_recent = [], {}
    away_arsenal, away_recent = [], {}
    if home_pitcher_id:
        home_pitcher_data = pitcher_results[idx]
        home_arsenal = pitcher_results[idx + 1]
        home_recent = pitcher_results[idx + 2]
        idx += 3
    if away_pitcher_id:
        away_pitcher_data = pitcher_results[idx]
        away_arsenal = pitcher_results[idx + 1]
        away_recent = pitcher_results[idx + 2]

    if game.away_pitcher:
        lines.append(format_pitcher_block(game.away_pitcher, game.away_pitcher_throws, away_pitcher_data))
        lines.append("")
    if game.home_pitcher:
        lines.append(format_pitcher_block(game.home_pitcher, game.home_pitcher_throws, home_pitcher_data))
        lines.append("")

    lines.append("\u2500" * 32)

    # Game date for predictions
    all_predictions: list[HitPrediction] = []
    game_date = ""
    if game.game_time:
        try:
            game_date = datetime.fromisoformat(game.game_time.replace("Z", "+00:00")).strftime("%Y-%m-%d")
        except Exception:
            game_date = date.today().isoformat()

    async def _empty_dict():
        return {}

    async def analyze_batter(batter: PlayerInfo, pitcher_data: dict, pitcher_id: int,
                              pitcher_name: str, pitcher_throws: str | None,
                              arsenal: list, recent: dict) -> tuple[dict, HitPrediction]:
        """Fetch all batter data in parallel, then predict."""
        b_data, h2h, batter_vs_pt, statcast = await asyncio.gather(
            get_batter_data(batter.id),
            get_h2h(batter.id, pitcher_id) if pitcher_id else _empty_dict(),
            get_batter_vs_pitch_types(batter.id),
            get_batter_statcast(batter.id),
        )
        ars_matchup = compute_arsenal_matchup(batter_vs_pt, arsenal)
        predict_kwargs = dict(
            batter_data=b_data, pitcher_data=pitcher_data,
            batter_bats=batter.bats, pitcher_throws=pitcher_throws,
            venue=game.venue, park_factors=pf,
            game_pk=game.game_pk, game_date=game_date,
            batter_name=batter.name, batter_id=batter.id,
            pitcher_name=pitcher_name, pitcher_id=pitcher_id or 0,
            h2h_data=h2h, arsenal_matchup=ars_matchup,
            batting_order=batter.batting_order,
            pitcher_recent=recent, batter_statcast=statcast,
            weather_adj=env_adj,
        )
        pred = predict_hit(**predict_kwargs)
        # Run shadow model (no-op if none registered)
        run_shadow_prediction(**predict_kwargs)
        return b_data, pred

    # Away lineup vs home pitcher
    if game.away_lineup and game.home_pitcher:
        rhb = sum(1 for b in game.away_lineup[:9] if b.bats == "R")
        lhb = sum(1 for b in game.away_lineup[:9] if b.bats == "L")
        shb = sum(1 for b in game.away_lineup[:9] if b.bats == "S")
        lines.append(f"\n{game.away_team} ({rhb}R/{lhb}L/{shb}S) vs {game.home_pitcher}:\n")

        results = await asyncio.gather(*[
            analyze_batter(b, home_pitcher_data, home_pitcher_id,
                          game.home_pitcher, game.home_pitcher_throws,
                          home_arsenal, home_recent)
            for b in game.away_lineup[:9]
        ])
        for batter, (b_data, pred) in zip(game.away_lineup[:9], results):
            all_predictions.append(pred)
            lines.append(format_batter_matchup(batter, game.home_pitcher_throws, b_data, pred))
            lines.append("")

    lines.append("\u2500" * 32)

    # Home lineup vs away pitcher
    if game.home_lineup and game.away_pitcher:
        rhb = sum(1 for b in game.home_lineup[:9] if b.bats == "R")
        lhb = sum(1 for b in game.home_lineup[:9] if b.bats == "L")
        shb = sum(1 for b in game.home_lineup[:9] if b.bats == "S")
        lines.append(f"\n{game.home_team} ({rhb}R/{lhb}L/{shb}S) vs {game.away_pitcher}:\n")

        results = await asyncio.gather(*[
            analyze_batter(b, away_pitcher_data, away_pitcher_id,
                          game.away_pitcher, game.away_pitcher_throws,
                          away_arsenal, away_recent)
            for b in game.home_lineup[:9]
        ])
        for batter, (b_data, pred) in zip(game.home_lineup[:9], results):
            all_predictions.append(pred)
            lines.append(format_batter_matchup(batter, game.away_pitcher_throws, b_data, pred))
            lines.append("")

    # Summary by tier
    total = len(all_predictions)
    strong = sum(1 for p in all_predictions if p.tier == "STRONG HIT")
    lean = sum(1 for p in all_predictions if p.tier == "LEAN HIT")
    tossup = sum(1 for p in all_predictions if p.tier == "TOSS-UP")
    fade = sum(1 for p in all_predictions if p.tier == "FADE")
    lines.append(f"\U0001f4ca {total} matchups: {strong} Strong | {lean} Lean | {tossup} Toss-up | {fade} Fade")

    if all_predictions:
        save_predictions(all_predictions)
        logger.info(f"Saved {len(all_predictions)} predictions for game {game.game_pk}")

    return "\n".join(lines)
