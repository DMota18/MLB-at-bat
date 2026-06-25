"""
Baseball Pregame Analysis Bot — Advanced Stats Edition.

Pulls BABIP, ISO, K%, BB%, platoon splits, recent form, pitcher
whiff%, arsenal, and park factors for each matchup. Raw data
for comparing against odds.
"""

from __future__ import annotations

import asyncio
import logging
import os
import unicodedata
from datetime import date, datetime, timedelta, timezone
from functools import wraps

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from config import MLB_API, CURRENT_SEASON, PREVIOUS_SEASON, fetch_json, prob_to_american
from data_fetchers import get_batter_data, get_pitcher_data
from formatters import build_pregame_card, format_batter_matchup, format_pitcher_block, PARK_FACTORS
from lineup_detector import get_todays_games, GameLineup, PlayerInfo, _player_bios
from predictor import predict_hit, HitPrediction, hit_tier, TIER_EMOJI
from tracker import (
    save_predictions, check_results, get_overall_stats, get_recent_predictions,
    save_book_odds, get_odds_for_date, games_with_odds,
    place_paper_bets, settle_paper_bets, get_paper_summary, get_paper_bets_for_date,
    get_tier_stats, update_closing_odds, games_needing_closing_odds, get_clv_stats,
    get_calibration_scores,
)
from matchup_data import (
    get_h2h, get_pitcher_arsenal, get_batter_vs_pitch_types, compute_arsenal_matchup,
    get_pitcher_recent_starts, get_batter_statcast,
)
from odds_api import get_events, get_hit_props, find_best_odds, match_event_to_game
from drift import check_drift, format_drift_report
from ab_testing import (
    compare_models, format_ab_report, settle_shadow_predictions,
    get_shadow_model, run_shadow_prediction, register_shadow_model,
)
from predictor_v4 import predict_hit_v4

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


logger = logging.getLogger("baseball_bot")

BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = int(os.environ["TELEGRAM_CHAT_ID"])

# ── Register shadow model for A/B testing ────────────────────────────
register_shadow_model(
    name="v4-kbb-xba-isotonic",
    version="v4.0-shadow",
    predict_fn=predict_hit_v4,
    description="K%/BB% factors, xBA blend, isotonic calibration, reduced platoon (1.02/0.98)",
)

_todays_games: list[GameLineup] = []


# ── Persistent analyzed games tracking ────────────────────────────────

def _load_analyzed_games() -> set[int]:
    """Load today's analyzed game_pks from the predictions DB."""
    from tracker import _get_db
    import contextlib
    from datetime import date
    today = date.today().isoformat()
    try:
        with contextlib.closing(_get_db()) as db:
            rows = db.execute(
                "SELECT DISTINCT game_pk FROM predictions WHERE game_date = ?",
                (today,),
            ).fetchall()
        return {r["game_pk"] for r in rows}
    except Exception:
        return set()


_analyzed_games: set[int] = _load_analyzed_games()


def authorized(func):
    """Decorator to restrict commands to the configured CHAT_ID."""

    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        if update.effective_chat.id != CHAT_ID:
            return
        return await func(update, context)
    return wrapper




# ── Game state helpers ────────────────────────────────────────────────


def _void_predictions(game_pk: int) -> None:
    """Mark predictions for a postponed/cancelled game as DNP."""
    import contextlib
    from tracker import _get_db
    try:
        with contextlib.closing(_get_db()) as db:
            updated = db.execute(
                "UPDATE predictions SET actual_result = 'DNP' WHERE game_pk = ? AND actual_result IS NULL",
                (game_pk,),
            ).rowcount
            db.commit()
        if updated > 0:
            logger.info(f"Voided {updated} predictions for postponed game {game_pk}")
    except Exception as e:
        logger.error(f"Failed to void predictions for {game_pk}: {e}")


# Track starting pitchers at analysis time to detect scratches
_game_pitchers: dict[int, tuple[str | None, str | None]] = {}


def _check_pitcher_scratch(game: 'GameLineup') -> bool:
    """Check if a pitcher was scratched since we last analyzed this game.

    Returns True if a scratch was detected (game should be re-analyzed).
    """
    if game.game_pk not in _game_pitchers:
        return False

    prev_away, prev_home = _game_pitchers[game.game_pk]
    current_away = game.away_pitcher
    current_home = game.home_pitcher

    if prev_away and current_away and prev_away != current_away:
        logger.info(f"Pitcher scratch detected: {prev_away} -> {current_away} ({game.away_team})")
        return True
    if prev_home and current_home and prev_home != current_home:
        logger.info(f"Pitcher scratch detected: {prev_home} -> {current_home} ({game.home_team})")
        return True

    return False


# ── Lineup polling ───────────────────────────────────────────────────


async def check_lineups(bot) -> None:
    logger.info("Checking lineups...")
    games = await get_todays_games()

    for game in games:
        # Handle postponements and suspensions
        if game.status in ("Postponed", "Suspended", "Cancelled"):
            if game.game_pk in _analyzed_games:
                logger.info(f"Game {game.away_team}@{game.home_team} {game.status} — voiding predictions")
                _void_predictions(game.game_pk)
            _analyzed_games.add(game.game_pk)
            continue

        if game.game_pk in _analyzed_games:
            # Check for pitcher scratch on already-analyzed games
            if game.lineups_posted and _check_pitcher_scratch(game):
                logger.info(f"Re-analyzing {game.away_team}@{game.home_team} due to pitcher scratch")
                _analyzed_games.discard(game.game_pk)
                # Fall through to re-analyze
            else:
                continue
        if not game.lineups_posted:
            continue
        if game.status in ("Final", "Game Over", "In Progress"):
            _analyzed_games.add(game.game_pk)
            continue

        logger.info(f"Building card: {game.away_team} @ {game.home_team}")
        try:
            card = await build_pregame_card(game)
            # Split long messages
            while card:
                chunk = card[:4000]
                if len(card) > 4000:
                    cut = chunk.rfind("\n")
                    if cut > 2000:
                        chunk = card[:cut]
                await bot.send_message(chat_id=CHAT_ID, text=chunk)
                card = card[len(chunk):]

            _analyzed_games.add(game.game_pk)
            _game_pitchers[game.game_pk] = (game.away_pitcher, game.home_pitcher)
        except Exception as e:
            logger.error(f"Card failed: {e}", exc_info=True)


# ── Commands ─────────────────────────────────────────────────────────


@authorized
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "\u26be Baseball Pregame Bot\n\n"
        "Matchup analysis + hit probability for every at-bat.\n\n"
        "/games - Today's schedule (numbered)\n"
        "/game <# or team> - Analyze a specific game\n"
        "/analyze - Analyze ALL games with lineups\n"
        "/best - Today's best matchups by tier\n"
        "/odds - Today's +EV picks vs book odds\n"
        "/paper - Paper betting P&L tracker\n"
        "/results - Check yesterday's results\n"
        "/stats - Prediction accuracy\n"
        "/recent - Recent predictions\n"
        "/player <name> - Player lookup\n"
        "/park - Park factors\n"
        "/drift - Model calibration health\n"
        "/ab [days] - A/B test results"
    )



@authorized
async def cmd_games(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    _todays_games.clear()
    _todays_games.extend(await get_todays_games())
    if not _todays_games:
        await update.message.reply_text("No games today.")
        return
    lines = [f"\u26be Today ({len(_todays_games)} games)\n"]
    for i, g in enumerate(_todays_games, 1):
        lu = "\u2705" if g.lineups_posted else "\u23f3"
        done = " \U0001f4cb" if g.game_pk in _analyzed_games else ""
        sp_a = g.away_pitcher or "TBD"
        sp_h = g.home_pitcher or "TBD"
        lines.append(f"{lu}{done} {i}. {g.away_team} @ {g.home_team} ({sp_a} vs {sp_h})")
    lines.append("\nUse /game <# or team> to analyze one game")
    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_game(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze a specific game by number or team name."""
    if not _todays_games:
        _todays_games.extend(await get_todays_games())

    if not context.args:
        await update.message.reply_text("Usage: /game 3  or  /game Yankees")
        return

    query = " ".join(context.args).strip()
    target_game = None

    # Try by number first
    try:
        num = int(query)
        if 1 <= num <= len(_todays_games):
            target_game = _todays_games[num - 1]
    except ValueError:
        pass

    # Try by team name
    if target_game is None:
        q = query.lower()
        for g in _todays_games:
            if q in g.away_team.lower() or q in g.home_team.lower():
                target_game = g
                break

    if target_game is None:
        await update.message.reply_text(f"No game found matching '{query}'. Use /games to see the list.")
        return

    if not target_game.lineups_posted:
        await update.message.reply_text(
            f"{target_game.away_team} @ {target_game.home_team}\n"
            f"Lineups not posted yet. Check back closer to game time."
        )
        return

    await update.message.reply_text(
        f"Analyzing {target_game.away_team} @ {target_game.home_team}...\n"
        f"Pulling H2H, arsenal, and stats (this takes ~1-2 min)..."
    )

    try:
        card = await build_pregame_card(target_game)
        while card:
            chunk = card[:4000]
            if len(card) > 4000:
                cut = chunk.rfind("\n")
                if cut > 2000:
                    chunk = card[:cut]
            await update.message.reply_text(chunk)
            card = card[len(chunk):]
        _analyzed_games.add(target_game.game_pk)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")
        logger.error(f"Game analysis failed: {e}", exc_info=True)


@authorized
async def cmd_best(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show today's matchups ranked by tier."""
    preds = get_recent_predictions(200)

    today = date.today().isoformat()
    today_preds = [p for p in preds if p.get("game_date") == today and p.get("hit_probability")]

    if not today_preds:
        await update.message.reply_text(
            "No predictions yet today. Use /game <team> to analyze a game first."
        )
        return

    today_preds.sort(key=lambda p: p["hit_probability"], reverse=True)

    lines = ["\u26be Today's Best Matchups\n"]
    lines.append("Ranked by model probability. Fair odds = model's price.\n")

    # Strong hits
    strong = [p for p in today_preds if hit_tier(p["hit_probability"]) == "STRONG HIT"]
    if strong:
        lines.append("\U0001f7e2 STRONG HIT (70%+):")
        for p in strong[:7]:
            prob = p['hit_probability']
            fair = prob_to_american(prob)
            lines.append(
                f"  {prob*100:.0f}% (fair: {fair}) {p['batter_name']} vs {p['pitcher_name']}"
                f"\n    [{p['confidence']}] {p.get('edge','')}"
            )
        lines.append("")

    # Lean hits
    lean = [p for p in today_preds if hit_tier(p["hit_probability"]) == "LEAN HIT"]
    if lean:
        lines.append("\U0001f7e1 LEAN HIT (62-70%):")
        for p in lean[:7]:
            prob = p['hit_probability']
            fair = prob_to_american(prob)
            lines.append(
                f"  {prob*100:.0f}% (fair: {fair}) {p['batter_name']} vs {p['pitcher_name']}"
                f"\n    [{p['confidence']}] {p.get('edge','')}"
            )
        lines.append("")

    # Fades
    fades = [p for p in today_preds if hit_tier(p["hit_probability"]) == "FADE"]
    fades.sort(key=lambda p: p["hit_probability"])
    if fades:
        lines.append("\U0001f534 FADE (<55%):")
        for p in fades[:5]:
            prob = p['hit_probability']
            fair = prob_to_american(prob)
            lines.append(
                f"  {prob*100:.0f}% (fair: {fair}) {p['batter_name']} vs {p['pitcher_name']}"
                f"\n    [{p['confidence']}] {p.get('edge','')}"
            )
        lines.append("")

    # Summary
    total = len(today_preds)
    games = len(set(p['game_pk'] for p in today_preds))
    lines.append(f"\U0001f4ca {total} matchups across {games} games: {len(strong)} Strong | {len(lean)} Lean | {len(fades)} Fade")

    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_analyze(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Running analysis on all games with lineups...")
    _analyzed_games.clear()
    await check_lineups(context.bot)
    await update.message.reply_text("Done.")


@authorized
async def cmd_player(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = " ".join(context.args) if context.args else None
    if not name:
        await update.message.reply_text("Usage: /player Aaron Judge")
        return

    try:
        data = await fetch_json(f"{MLB_API}/sports/1/players?season={CURRENT_SEASON}&gameType=R")
        matches = [p for p in data.get("people", []) if name.lower() in p.get("fullName", "").lower()]
        if not matches:
            await update.message.reply_text(f"No player found: '{name}'")
            return

        p = matches[0]
        pid = p["id"]
        bats = p.get("batSide", {}).get("code", "?")
        throws = p.get("pitchHand", {}).get("code", "?")
        pos = p.get("primaryPosition", {}).get("abbreviation", "?")
        team = p.get("currentTeam", {}).get("name", "?")

        bd, pd_data = await asyncio.gather(
            get_batter_data(pid),
            get_pitcher_data(pid),
        )

        lines = [f"\u26be {p['fullName']} ({pos}) - {team}", f"B: {bats} / T: {throws}", ""]

        s = bd.get("season")
        if s and int(s.get("pa", 0)) > 0:
            lines.append(f"{CURRENT_SEASON}: {s['avg']}/{s['obp']}/{s['slg']} OPS {s['ops']}")
            lines.append(f"  {s['hr']}HR {s['k']}K {s['bb']}BB ({s['pa']}PA)")

        adv = bd.get("advanced", {})
        sab = bd.get("saber", {})
        parts = []
        for k, label in [("babip", "BABIP"), ("iso", "ISO"), ("k_pct", "K%"), ("bb_pct", "BB%")]:
            if adv.get(k) and adv[k] != "---":
                parts.append(f"{label} {adv[k]}")
        if sab.get("woba") and sab["woba"] != "---":
            parts.append(f"wOBA {sab['woba']}")
        if parts:
            lines.append(f"  {' | '.join(parts)}")

        plat = bd.get("platoon", {})
        if plat:
            for side in ["vs_L", "vs_R"]:
                if side in plat:
                    p2 = plat[side]
                    lines.append(f"  '{PREVIOUS_SEASON % 100} {side}: {p2['avg']}/{p2['obp']}/{p2['slg']} ({p2['pa']}PA)")

        last7 = bd.get("last7")
        if last7 and int(last7.get("ab", 0)) > 0:
            lines.append(f"  Last 7G: {last7['avg']} ({last7['h']}/{last7['ab']}) {last7['hr']}HR")

        ps = pd_data.get("season")
        if ps and (int(ps.get("gs", 0)) > 0 or int(ps.get("ip", "0").replace(".", "")) > 0):
            lines.append(f"\nPitching: {ps['era']} ERA {ps['whip']} WHIP {ps['k']}K in {ps['ip']}IP")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")


@authorized
async def cmd_results(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check yesterday's prediction results."""
    await update.message.reply_text("Checking results...")

    # Check yesterday by default, or a specific date
    target = " ".join(context.args) if context.args else None
    result = await check_results(target)
    summary = result.get("summary", {})

    if summary.get("total", 0) == 0:
        await update.message.reply_text(f"No results yet for {result['date']}. ({result.get('checked', 0)} checked)")
        return

    lines = [
        f"\U0001f4ca Results for {result['date']}\n",
        f"Overall: {summary['correct']}/{summary['total']} correct ({summary['accuracy']}%)\n",
    ]
    tiers = get_tier_stats(result['date'])
    for t in tiers:
        emoji = TIER_EMOJI.get(t["tier"], "")
        lines.append(f"{emoji} {t['tier']}: {t['hits']}/{t['total']} got a hit ({t['hit_rate']}%)")
    lines.append(f"\nBase rate: ~61% of batters get a hit")
    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show lifetime prediction accuracy by tier."""
    stats = get_overall_stats()

    if stats["total_predictions"] == 0:
        await update.message.reply_text("No predictions tracked yet. Run /analyze first.")
        return

    by_conf = stats["by_confidence"]

    lines = [
        "\U0001f4ca Lifetime Stats\n",
        f"Total settled: {stats['total_predictions']}  |  Pending: {stats['pending']}",
        "",
    ]

    # Tier accuracy
    tiers = get_tier_stats()

    lines.append("Accuracy by tier:")
    for t in tiers:
        emoji = TIER_EMOJI.get(t["tier"], "")
        lines.append(f"  {emoji} {t['tier']}: {t['hits']}/{t['total']} got a hit ({t['hit_rate']}%)")

    lines.append("")
    lines.append("By confidence:")
    for conf in ["high", "medium", "low", "insufficient"]:
        c = by_conf.get(conf, {})
        if c.get("total", 0) > 0:
            lines.append(f"  {conf}: {c['correct']}/{c['total']} ({c['accuracy']}%)")

    # Paper betting summary
    ps = get_paper_summary()
    if ps["total_bets"] > 0:
        lines.append("")
        emoji = "\U0001f4b0" if ps["total_pnl"] >= 0 else "\U0001f4c9"
        lines.append(f"{emoji} Paper bets: {ps['wins']}W-{ps['losses']}L ({ps['win_rate']}%) | ${ps['total_pnl']:+.2f} ROI: {ps['roi']:+.1f}%")

    # CLV stats
    clv = get_clv_stats()
    if clv["total"] > 0:
        lines.append("")
        clv_emoji = "\U0001f4c8" if clv["avg_clv"] > 0 else "\U0001f4c9"
        lines.append(f"{clv_emoji} CLV: {clv['avg_clv']:+.2f}% avg | {clv['positive_pct']}% positive ({clv['positive']}/{clv['total']})")
        for t in clv["by_tier"]:
            avg = (t["avg_clv"] or 0) * 100
            pos = t["positive"] or 0
            n = t["n"] or 0
            if n > 0:
                lines.append(f"  {t['tier']:15s}: {avg:+.2f}% avg CLV ({pos}/{n} positive)")

    # Calibration scores
    cal = get_calibration_scores()
    if cal["total"] > 0:
        lines.append("")
        lines.append(f"\U0001f4cf Calibration: Brier={cal['brier_score']:.4f} | LogLoss={cal['log_loss']:.4f} ({cal['total']} preds)")

    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_recent(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent predictions and their results."""
    preds = get_recent_predictions(15)
    if not preds:
        await update.message.reply_text("No predictions yet.")
        return

    lines = ["\U0001f4cb Recent Predictions\n"]
    for p in preds:
        result_str = p.get("actual_result") or "\u23f3"
        prob = p["hit_probability"]
        tier = hit_tier(prob)
        emoji = TIER_EMOJI.get(tier, "")

        check = ""
        if p.get("got_hit") is not None:
            check = " \u2705" if p["got_hit"] == 1 else " \u274c"

        lines.append(f"{emoji} {p['batter_name']} vs {p['pitcher_name']}: "
                     f"{tier} ({prob:.0%}) \u2192 {result_str}{check}")

    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_health(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot health status."""
    import psutil
    import time as _time

    lines = ["\U0001f3e5 Bot Health\n"]

    # Uptime
    try:
        proc = psutil.Process()
        uptime_secs = _time.time() - proc.create_time()
        hours = int(uptime_secs // 3600)
        mins = int((uptime_secs % 3600) // 60)
        lines.append(f"Uptime: {hours}h {mins}m")
    except Exception:
        lines.append("Uptime: unknown")

    # Database stats
    from tracker import _get_db
    import contextlib
    with contextlib.closing(_get_db()) as tdb:
        today = date.today().isoformat()
        yesterday = (date.today() - timedelta(days=1)).isoformat()

        preds_today = tdb.execute("SELECT COUNT(*) as c FROM predictions WHERE game_date = ?", (today,)).fetchone()["c"]
        preds_pending = tdb.execute("SELECT COUNT(*) as c FROM predictions WHERE actual_result IS NULL").fetchone()["c"]
        bets_today = tdb.execute("SELECT COUNT(*) as c FROM paper_bets WHERE game_date = ?", (today,)).fetchone()["c"]
        clv_today = tdb.execute("SELECT COUNT(*) as c FROM book_odds WHERE game_date = ? AND clv IS NOT NULL", (today,)).fetchone()["c"]
        odds_today = tdb.execute("SELECT COUNT(*) as c FROM book_odds WHERE game_date = ?", (today,)).fetchone()["c"]

        # Last activity timestamps
        last_pred = tdb.execute("SELECT MAX(created_at) as t FROM predictions").fetchone()["t"] or "never"
        last_odds = tdb.execute("SELECT MAX(fetched_at) as t FROM book_odds").fetchone()["t"] or "never"

    lines.append(f"\nToday ({today}):")
    lines.append(f"  Predictions: {preds_today}")
    lines.append(f"  Odds entries: {odds_today}")
    lines.append(f"  CLV entries: {clv_today}")
    lines.append(f"  Paper bets: {bets_today}")
    lines.append(f"  Pending settlement: {preds_pending}")
    lines.append(f"\nLast prediction: {last_pred}")
    lines.append(f"Last odds fetch: {last_odds}")
    lines.append(f"Analyzed games: {len(_analyzed_games)}")

    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_park(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    name = " ".join(context.args) if context.args else None
    if not name:
        lines = ["\U0001f3df Park Factors:\n"]
        for park, pf in sorted(PARK_FACTORS.items(), key=lambda x: -x[1]["hr"]):
            lines.append(f"{pf.get('tag','')} {park}: HR {pf['hr']}x  Runs {pf['runs']}x")
        await update.message.reply_text("\n".join(lines))
    else:
        matches = [(p, f) for p, f in PARK_FACTORS.items() if name.lower() in p.lower()]
        if matches:
            park, pf = matches[0]
            await update.message.reply_text(f"\U0001f3df {park}\nHR: {pf['hr']}x | Runs: {pf['runs']}x")
        else:
            await update.message.reply_text(f"No park: '{name}'")


# ── Main ─────────────────────────────────────────────────────────────


async def auto_check_results(bot) -> None:
    """Auto-check yesterday's results and send summary."""
    logger.info("Auto-checking yesterday's results...")
    result = await check_results()  # defaults to yesterday
    summary = result.get("summary", {})
    if summary.get("total", 0) > 0:
        lines = [
            f"\U0001f4ca Yesterday's Results ({result['date']})\n",
            f"Overall: {summary['correct']}/{summary['total']} correct ({summary['accuracy']}%)\n",
        ]
        tiers = get_tier_stats(result['date'])
        for t in tiers:
            emoji = TIER_EMOJI.get(t["tier"], "")
            lines.append(f"{emoji} {t['tier']}: {t['hits']}/{t['total']} got a hit ({t['hit_rate']}%)")
        try:
            await bot.send_message(chat_id=CHAT_ID, text="\n".join(lines))
        except Exception as e:
            logger.error(f"Failed to send results: {e}")

    # Settle paper bets
    paper = settle_paper_bets()  # defaults to yesterday
    if paper.get("settled", 0) > 0:
        emoji = "\U0001f4b0" if paper["pnl"] >= 0 else "\U0001f4c9"
        ptext = (
            f"{emoji} Paper Bets — {paper['date']}\n\n"
            f"{paper['wins']}W-{paper['losses']}L | P&L: ${paper['pnl']:+.2f}\n"
        )
        # Get overall running total
        ps = get_paper_summary()
        if ps["total_bets"] > 0:
            ptext += (
                f"\nAll-time: {ps['wins']}W-{ps['losses']}L ({ps['win_rate']}%)"
                f"\nTotal P&L: ${ps['total_pnl']:+.2f} | ROI: {ps['roi']:+.1f}%"
                f"\nAvg/day: ${ps['avg_daily_pnl']:+.2f}"
            )
        try:
            await bot.send_message(chat_id=CHAT_ID, text=ptext)
        except Exception as e:
            logger.error(f"Failed to send paper results: {e}")


_SUFFIXES = {"jr.", "sr.", "ii", "iii", "iv", "v"}


def _normalize_name(name: str) -> str:
    """Normalize a player name for matching: strip accents, suffixes, and lowercase."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    parts = [p for p in ascii_name.lower().split() if p.rstrip(".") not in _SUFFIXES]
    return " ".join(parts)


def _lookup_odds(game_odds: dict[str, dict], batter_name: str) -> dict | None:
    """Look up odds for a batter, handling name format differences.

    Handles accent mismatches (José Ramírez vs Jose Ramirez) and
    suffix differences (Vladimir Guerrero Jr. vs Vladimir Guerrero).
    """
    if not game_odds:
        return None

    if batter_name in game_odds:
        return game_odds[batter_name]

    norm_batter = _normalize_name(batter_name)
    for name, odds in game_odds.items():
        if _normalize_name(name) == norm_batter:
            return odds

    # Fallback: last-name + first-initial match
    batter_parts = norm_batter.split()
    if not batter_parts:
        return None
    batter_last = batter_parts[-1]
    batter_initial = batter_parts[0][0] if batter_parts[0] else ""
    for name, odds in game_odds.items():
        norm_parts = _normalize_name(name).split()
        if not norm_parts:
            continue
        if norm_parts[-1] == batter_last and norm_parts[0][0] == batter_initial:
            return odds

    return None


async def fetch_odds_for_upcoming(bot) -> None:
    """Fetch book odds for upcoming games.

    Runs every 5 min. Re-fetches to catch line movement — new odds
    overwrite old ones via save_book_odds (INSERT OR REPLACE on
    game_pk + batter + book). Paper bets placed incrementally.
    """
    now = datetime.now(timezone.utc)
    games = await get_todays_games()
    today_str = now.strftime("%Y-%m-%d")

    # Find games starting within 180 min
    targets = []
    for game in games:
        if not game.game_time:
            continue
        try:
            game_dt = datetime.fromisoformat(game.game_time.replace("Z", "+00:00"))
            minutes_until = (game_dt - now).total_seconds() / 60
            if 0 < minutes_until <= 180:
                targets.append(game)
        except Exception:
            continue

    if not targets:
        return

    logger.info(f"Fetching odds for {len(targets)} upcoming games")

    # Fetch events list once (free, no credit cost)
    events = await get_events()
    if not events:
        return

    # Get predictions for these games to compare
    recent = get_recent_predictions(500)
    pred_map: dict[tuple[int, str], dict] = {}  # (game_pk, batter_name) -> pred
    for p in recent:
        if p.get("game_date") == today_str:
            pred_map[(p["game_pk"], p["batter_name"])] = p

    ev_picks = []  # Collect +EV picks for alert

    for game in targets:
        event_id = match_event_to_game(events, game.home_team, game.away_team)
        if not event_id:
            logger.debug(f"No odds event found for {game.away_team}@{game.home_team}")
            continue

        try:
            hit_props = await get_hit_props(event_id)
        except Exception as e:
            logger.error(f"Odds fetch failed for {event_id}: {e}")
            continue

        if not hit_props:
            logger.debug(f"No hit props available for {game.away_team}@{game.home_team}")
            continue

        game_date = today_str
        odds_saved = 0

        for player_name, odds_list in hit_props.items():
            # Find matching prediction
            matched_pred = None
            for (gpk, bname), pred in pred_map.items():
                if gpk == game.game_pk and _lookup_odds({player_name: True}, bname):
                    matched_pred = pred
                    break

            # Also try direct name match
            if not matched_pred:
                matched_pred = pred_map.get((game.game_pk, player_name))

            if not matched_pred:
                continue

            model_prob = matched_pred["hit_probability"]

            # Save all book odds
            best = find_best_odds(odds_list)
            if best and best["all_books"]:
                save_book_odds(
                    game_pk=game.game_pk, game_date=game_date,
                    batter_name=player_name,
                    batter_id=matched_pred["batter_id"] if matched_pred else 0,
                    model_prob=model_prob,
                    odds_entries=best["all_books"],
                )
                odds_saved += 1

                # Track +EV picks
                if matched_pred and model_prob >= 0.70:
                    edge = model_prob - best["implied_prob"]
                    if edge > 0.02:
                        ev_picks.append({
                            "name": player_name,
                            "model_prob": model_prob,
                            "book": best["best_book"],
                            "book_odds": best["best_over"],
                            "implied": best["implied_prob"],
                            "edge": edge,
                            "confidence": matched_pred.get("confidence", ""),
                            "pred_edge": matched_pred.get("edge", ""),
                            "vs": matched_pred.get("pitcher_name", ""),
                        })

        logger.info(f"Saved {odds_saved} odds entries for {game.away_team}@{game.home_team}")

    # Send +EV alert
    if ev_picks:
        ev_picks.sort(key=lambda x: -x["edge"])
        lines = ["\U0001f4b0 +EV Picks (model edge vs book)\n"]
        for p in ev_picks:
            book_str = f"{p['book_odds']:+d}" if p['book_odds'] > 0 else str(p['book_odds'])
            fair = prob_to_american(p['model_prob'])
            lines.append(
                f"\U0001f7e2 {p['name']} vs {p['vs']}"
                f"\n   Model: {p['model_prob']:.0%} (fair {fair}) | {p['book']}: {book_str} (impl {p['implied']:.0%})"
                f"\n   Edge: +{p['edge']:.1%} [{p['confidence']}] {p['pred_edge']}"
                f"\n"
            )
        lines.append(f"\n{len(ev_picks)} picks with 2%+ edge found")

        text = "\n".join(lines)
        try:
            await bot.send_message(chat_id=CHAT_ID, text=text[:4000])
        except Exception as e:
            logger.error(f"Failed to send +EV alert: {e}")

    # Auto-place paper bets for today
    paper = place_paper_bets(today_str, max_bets=30)
    if paper:
        plines = [f"\U0001f4dd Paper Bets Placed — {today_str}\n"]
        for i, b in enumerate(paper, 1):
            odds_str = f"{b['over_price']:+d}" if b['over_price'] > 0 else str(b['over_price'])
            plines.append(
                f"{i}. {b['batter_name']} vs {b.get('pitcher_name', '?')}"
                f"\n   {b['book']} {odds_str} | Model {b['model_prob']:.0%} | Edge +{b['edge']:.1%}"
            )
        plines.append(f"\n$100 flat stake per bet. Total risk: ${len(paper) * 100}")
        try:
            await bot.send_message(chat_id=CHAT_ID, text="\n".join(plines))
        except Exception as e:
            logger.error(f"Failed to send paper bets: {e}")
        logger.info(f"Placed {len(paper)} paper bets for {today_str}")


async def fetch_closing_odds(bot) -> None:
    """Fetch closing lines for games that already have opening odds.

    Runs twice hourly. Fetches for all games with opening odds.
    """
    now = datetime.now(timezone.utc)
    today_str = now.strftime("%Y-%m-%d")

    # Which games need closing odds?
    needs_close = games_needing_closing_odds(today_str)
    if not needs_close:
        return

    games = await get_todays_games()

    # Only fetch for games starting within 30 min (closing line window)
    targets = []
    for game in games:
        if game.game_pk not in needs_close:
            continue
        if not game.game_time:
            continue
        try:
            game_dt = datetime.fromisoformat(game.game_time.replace("Z", "+00:00"))
            minutes_until = (game_dt - now).total_seconds() / 60
            if 0 < minutes_until <= 30:
                targets.append(game)
        except Exception:
            continue

    if not targets:
        return

    logger.info(f"Fetching closing odds for {len(targets)} games")

    events = await get_events()
    if not events:
        return

    for game in targets:
        event_id = match_event_to_game(events, game.home_team, game.away_team)
        if not event_id:
            continue

        try:
            hit_props = await get_hit_props(event_id)
        except Exception as e:
            logger.error(f"Closing odds fetch failed for {event_id}: {e}")
            continue

        if not hit_props:
            continue

        for player_name, odds_list in hit_props.items():
            best = find_best_odds(odds_list)
            if best and best["best_over"] is not None:
                update_closing_odds(
                    game_pk=game.game_pk,
                    game_date=today_str,
                    batter_name=player_name,
                    book=best["best_book"],
                    closing_over=best["best_over"],
                )

        logger.info(f"Closing odds saved for {game.away_team}@{game.home_team}")


@authorized
async def cmd_odds(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show today's +EV picks with book odds."""
    today = date.today().isoformat()
    odds = get_odds_for_date(today)

    if not odds:
        await update.message.reply_text(
            "No odds data for today yet. Odds are fetched ~90 min before each game."
        )
        return

    # Group by batter, keep best edge per batter
    best_by_batter: dict[str, dict] = {}
    for o in odds:
        name = o["batter_name"]
        if name not in best_by_batter or o["edge"] > best_by_batter[name]["edge"]:
            best_by_batter[name] = o

    # Sort by edge
    ranked = sorted(best_by_batter.values(), key=lambda x: -x["edge"])
    ev_picks = [r for r in ranked if r["edge"] > 0.02 and r["model_prob"] >= 0.70]

    if not ev_picks:
        await update.message.reply_text(
            f"Odds fetched for {len(best_by_batter)} batters today but no +EV picks found (need 2%+ edge and 70%+ model prob)."
        )
        return

    lines = [f"\U0001f4b0 +EV Picks — {today}\n"]
    for p in ev_picks[:15]:
        over = p["over_price"]
        over_str = f"{over:+d}" if over > 0 else str(over)
        fair = prob_to_american(p["model_prob"])
        lines.append(
            f"\U0001f7e2 {p['batter_name']}"
            f"\n   Model: {p['model_prob']:.0%} (fair {fair}) | {p['book']}: {over_str} (impl {p['implied_prob']:.0%})"
            f"\n   Edge: +{p['edge']:.1%}"
        )
        lines.append("")

    lines.append(f"{len(ev_picks)} total +EV picks")
    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_paper(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show paper betting performance."""
    ps = get_paper_summary()
    if ps["total_bets"] == 0:
        await update.message.reply_text("No paper bets settled yet. Bets are placed automatically when odds are fetched.")
        return

    emoji = "\U0001f4b0" if ps["total_pnl"] >= 0 else "\U0001f4c9"
    lines = [f"{emoji} Paper Betting Performance\n"]
    lines.append(f"Record: {ps['wins']}W-{ps['losses']}L ({ps['win_rate']}%)")
    lines.append(f"Total P&L: ${ps['total_pnl']:+.2f}")
    lines.append(f"ROI: {ps['roi']:+.1f}%")
    lines.append(f"Avg/day: ${ps['avg_daily_pnl']:+.2f}")
    lines.append(f"Days tracked: {ps['days']}")
    if ps["pending"] > 0:
        lines.append(f"Pending: {ps['pending']} bets")

    lines.append(f"\n{'─' * 28}")
    lines.append("Daily breakdown:")
    for day in ps["daily"]:
        dpnl = day["pnl"] or 0
        de = "\U0001f7e2" if dpnl >= 0 else "\U0001f534"
        lines.append(f"  {de} {day['game_date']}: {day['wins']}W-{day['losses']}L  ${dpnl:+.2f}")

    # Show today's bets if any
    today = date.today().isoformat()
    today_bets = get_paper_bets_for_date(today)
    if today_bets:
        lines.append(f"\n{'─' * 28}")
        lines.append(f"Today's bets ({len(today_bets)}):")
        for b in today_bets:
            odds_str = f"{b['book_odds']:+d}" if b['book_odds'] > 0 else str(b['book_odds'])
            status = b["result"] or "PENDING"
            pnl_str = f" ${b['pnl']:+.2f}" if b["pnl"] is not None else ""
            lines.append(f"  {b['batter_name']} | {b['book']} {odds_str} | {status}{pnl_str}")

    await update.message.reply_text("\n".join(lines))


@authorized
async def cmd_drift(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check model calibration drift."""
    result = check_drift()
    report = format_drift_report(result)
    await update.message.reply_text(report)


@authorized
async def cmd_ab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show A/B test results comparing primary vs shadow model."""
    # Optional: /ab 7 for last 7 days
    days = None
    if context.args:
        try:
            days = int(context.args[0])
        except ValueError:
            pass

    results = compare_models(days_back=days)
    report = format_ab_report(results)
    await update.message.reply_text(report)


async def auto_drift_check(bot) -> None:
    """Scheduled drift check — sends alert only on warning/degraded."""
    logger.info("Running scheduled drift check...")
    result = check_drift()

    if result["status"] in ("warning", "degraded"):
        report = format_drift_report(result)
        try:
            await bot.send_message(chat_id=CHAT_ID, text=report)
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")
    else:
        logger.info(f"Drift check: {result['status']}")

    # Also settle any shadow predictions while we're at it
    settled = settle_shadow_predictions()
    if settled > 0:
        logger.info(f"Settled {settled} shadow predictions")


def backup_database() -> None:
    """Create a daily backup of the predictions database."""
    import shutil
    from pathlib import Path
    from datetime import date as dt_date

    db_path = Path(__file__).parent / "predictions.db"
    backup_dir = Path(__file__).parent / "backups"
    backup_dir.mkdir(exist_ok=True)

    today = dt_date.today().isoformat()
    backup_path = backup_dir / f"predictions_{today}.db"

    if backup_path.exists():
        return  # already backed up today

    try:
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to {backup_path}")

        # Keep only last 7 backups
        backups = sorted(backup_dir.glob("predictions_*.db"))
        for old in backups[:-7]:
            old.unlink()
            logger.info(f"Removed old backup: {old.name}")
    except Exception as e:
        logger.error(f"Database backup failed: {e}")


async def post_init(app: Application) -> None:
    scheduler = AsyncIOScheduler(timezone="America/New_York")
    loop = asyncio.get_running_loop()

    def schedule_coro(coro_func):
        """Schedule an async function from the APScheduler thread."""
        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(coro_func()))

    scheduler.add_job(
        lambda: schedule_coro(lambda: check_lineups(app.bot)),
        CronTrigger(hour="10-20", minute="0,30"),
        id="lineup_check",
    )
    scheduler.add_job(lambda: _analyzed_games.clear(), CronTrigger(hour=0, minute=0), id="reset")
    scheduler.add_job(
        lambda: schedule_coro(lambda: auto_check_results(app.bot)),
        CronTrigger(hour=8, minute=0),
        id="results_check",
    )
    scheduler.add_job(
        lambda: schedule_coro(lambda: fetch_odds_for_upcoming(app.bot)),
        CronTrigger(hour="10-23", minute="*/5"),
        id="odds_check",
    )
    scheduler.add_job(
        lambda: schedule_coro(lambda: fetch_closing_odds(app.bot)),
        CronTrigger(hour="12-23", minute="*/10"),
        id="closing_odds",
    )
    scheduler.add_job(
        lambda: schedule_coro(lambda: auto_drift_check(app.bot)),
        CronTrigger(hour=9, minute=0),
        id="drift_check",
    )
    scheduler.add_job(
        backup_database,
        CronTrigger(hour=3, minute=0),
        id="db_backup",
    )
    scheduler.start()
    logger.info("Scheduler started")


def main():
    logger.info("Starting Baseball Bot...")
    app = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("games", cmd_games))
    app.add_handler(CommandHandler("game", cmd_game))
    app.add_handler(CommandHandler("best", cmd_best))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("results", cmd_results))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("recent", cmd_recent))
    app.add_handler(CommandHandler("player", cmd_player))
    app.add_handler(CommandHandler("park", cmd_park))
    app.add_handler(CommandHandler("health", cmd_health))
    app.add_handler(CommandHandler("odds", cmd_odds))
    app.add_handler(CommandHandler("paper", cmd_paper))
    app.add_handler(CommandHandler("drift", cmd_drift))
    app.add_handler(CommandHandler("ab", cmd_ab))
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
