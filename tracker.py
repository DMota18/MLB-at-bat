"""
Prediction tracker — stores predictions and checks results.

SQLite database tracks every prediction and its outcome.
After games complete, pulls box scores to determine who got hits.
"""

from __future__ import annotations

import contextlib
import logging
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from config import MLB_API, MODEL_VERSION, TIER_STRONG, TIER_LEAN, TIER_TOSSUP, american_to_prob, fetch_json
from predictor import HitPrediction

logger = logging.getLogger("baseball_bot.tracker")

DB_PATH = Path(__file__).parent / "predictions.db"
_tables_initialized = False


def _init_tables(db: sqlite3.Connection) -> None:
    """Create tables once per process."""
    global _tables_initialized
    if _tables_initialized:
        return
    db.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk INTEGER,
            game_date TEXT,
            batter_name TEXT,
            batter_id INTEGER,
            pitcher_name TEXT,
            pitcher_id INTEGER,
            venue TEXT,
            prediction TEXT,
            hit_probability REAL,
            confidence TEXT,
            edge TEXT,
            actual_result TEXT DEFAULT NULL,
            got_hit INTEGER DEFAULT NULL,
            at_bats INTEGER DEFAULT NULL,
            hits INTEGER DEFAULT NULL,
            factors_json TEXT DEFAULT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_pk, batter_id)
        )
    """)
    for col, typ in [("factors_json", "TEXT"), ("model_version", "TEXT")]:
        try:
            db.execute(f"ALTER TABLE predictions ADD COLUMN {col} {typ} DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
    db.execute("""
        CREATE TABLE IF NOT EXISTS book_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk INTEGER,
            game_date TEXT,
            batter_name TEXT,
            batter_id INTEGER,
            book TEXT,
            line REAL DEFAULT 0.5,
            over_price INTEGER,
            under_price INTEGER,
            implied_prob REAL,
            model_prob REAL,
            edge REAL,
            closing_over_price INTEGER DEFAULT NULL,
            closing_implied_prob REAL DEFAULT NULL,
            clv REAL DEFAULT NULL,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_pk, batter_name, book, line)
        )
    """)
    # Add CLV columns to existing databases
    for col, typ in [("closing_over_price", "INTEGER"), ("closing_implied_prob", "REAL"), ("clv", "REAL")]:
        try:
            db.execute(f"ALTER TABLE book_odds ADD COLUMN {col} {typ} DEFAULT NULL")
        except sqlite3.OperationalError:
            pass
    # Deduplicate before adding unique index (keeps latest by id)
    try:
        db.execute("""
            DELETE FROM book_odds WHERE id NOT IN (
                SELECT MAX(id) FROM book_odds
                GROUP BY game_pk, batter_name, book, line
            )
        """)
        db.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_book_odds_unique
            ON book_odds(game_pk, batter_name, book, line)
        """)
        db.commit()
    except sqlite3.OperationalError:
        pass
    db.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date TEXT UNIQUE,
            total_predictions INTEGER,
            correct INTEGER,
            incorrect INTEGER,
            pending INTEGER,
            accuracy REAL,
            hit_predictions INTEGER,
            hit_correct INTEGER,
            no_hit_predictions INTEGER,
            no_hit_correct INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS paper_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_pk INTEGER,
            game_date TEXT,
            batter_name TEXT,
            batter_id INTEGER,
            pitcher_name TEXT,
            model_prob REAL,
            confidence TEXT,
            edge_tags TEXT,
            book TEXT,
            book_odds INTEGER,
            implied_prob REAL,
            edge REAL,
            stake REAL DEFAULT 100,
            result TEXT DEFAULT NULL,
            got_hit INTEGER DEFAULT NULL,
            pnl REAL DEFAULT NULL,
            settled INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.commit()
    _tables_initialized = True


def _get_db() -> sqlite3.Connection:
    """Get database connection, initializing tables on first call."""
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    _init_tables(db)
    return db


def save_predictions(preds: list[HitPrediction]) -> None:
    """Save predictions. Skips any that are already settled (have results).

    Uses INSERT OR IGNORE for predictions that already exist, then
    updates only unsettled ones. This prevents re-analysis from
    clobbering settled predictions that already have got_hit data.
    """
    import json
    from dataclasses import asdict

    with contextlib.closing(_get_db()) as db:
        for p in preds:
            # Check if a settled prediction already exists for this matchup
            existing = db.execute(
                "SELECT id, got_hit FROM predictions WHERE game_pk = ? AND batter_id = ?",
                (p.game_pk, p.batter_id),
            ).fetchone()

            if existing and existing["got_hit"] is not None:
                # Already settled — don't overwrite
                continue

            db.execute("""
                INSERT OR REPLACE INTO predictions (
                    game_pk, game_date, batter_name, batter_id,
                    pitcher_name, pitcher_id, venue,
                    prediction, hit_probability, confidence, edge, factors_json,
                    model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                p.game_pk, p.game_date, p.batter_name, p.batter_id,
                p.pitcher_name, p.pitcher_id, p.venue,
                p.prediction, p.hit_probability, p.confidence, p.edge,
                json.dumps(asdict(p.factors)), MODEL_VERSION,
            ))
        db.commit()


# ── Results checker ──────────────────────────────────────────────────


async def check_results(game_date: str = None) -> dict[str, Any]:
    """Check actual results for a date's predictions.

    Pulls box scores from MLB API and updates the database.
    """
    if game_date is None:
        game_date = (date.today() - timedelta(days=1)).isoformat()

    with contextlib.closing(_get_db()) as db:
        rows = db.execute(
            "SELECT * FROM predictions WHERE game_date = ? AND actual_result IS NULL",
            (game_date,)
        ).fetchall()

        if not rows:
            return {"date": game_date, "checked": 0, "message": "No pending predictions"}

        game_pks = set(r["game_pk"] for r in rows)
        results_map: dict[int, dict[int, dict]] = {}

        for gpk in game_pks:
            try:
                box = await _fetch_boxscore(gpk)
                if box:
                    results_map[gpk] = box
            except Exception as e:
                logger.error(f"Failed to fetch boxscore for {gpk}: {e}")

        updated = 0
        for row in rows:
            gpk = row["game_pk"]
            bid = row["batter_id"]

            if gpk not in results_map:
                continue

            player_stats = results_map[gpk].get(bid)
            if player_stats is None:
                db.execute(
                    "UPDATE predictions SET actual_result = 'DNP' WHERE id = ?",
                    (row["id"],)
                )
                updated += 1
                continue

            ab = player_stats.get("atBats", 0)
            hits = player_stats.get("hits", 0)
            got_hit = 1 if hits > 0 else 0

            db.execute("""
                UPDATE predictions SET
                    actual_result = ?,
                    got_hit = ?,
                    at_bats = ?,
                    hits = ?
                WHERE id = ?
            """, (
                f"{hits}-for-{ab}" if ab > 0 else "0 AB",
                got_hit, ab, hits, row["id"],
            ))
            updated += 1

        db.commit()
        summary = _compute_daily_summary(db, game_date)

    return {"date": game_date, "checked": updated, "summary": summary}


async def _fetch_boxscore(game_pk: int) -> dict[int, dict] | None:
    """Fetch box score and return batter_id -> hitting stats."""
    try:
        data = await fetch_json(f"{MLB_API}/game/{game_pk}/boxscore")
    except Exception as e:
        logger.warning(f"Failed to fetch boxscore for game {game_pk}: {e}")
        return None

    result: dict[int, dict] = {}

    for side in ["away", "home"]:
        team = data.get("teams", {}).get(side, {})
        players = team.get("players", {})
        for key, pdata in players.items():
            pid = pdata.get("person", {}).get("id", 0)
            stats = pdata.get("stats", {}).get("batting", {})
            if stats:
                result[pid] = {
                    "atBats": int(stats.get("atBats", 0)),
                    "hits": int(stats.get("hits", 0)),
                    "homeRuns": int(stats.get("homeRuns", 0)),
                    "strikeOuts": int(stats.get("strikeOuts", 0)),
                    "baseOnBalls": int(stats.get("baseOnBalls", 0)),
                }

    return result


def _compute_daily_summary(db: sqlite3.Connection, game_date: str) -> dict[str, Any]:
    """Compute accuracy summary for a date."""
    rows = db.execute(
        "SELECT * FROM predictions WHERE game_date = ? AND actual_result IS NOT NULL AND actual_result != 'DNP'",
        (game_date,)
    ).fetchall()

    if not rows:
        return {"total": 0, "message": "No results yet"}

    total = len(rows)
    correct = 0
    hit_pred = 0
    hit_correct = 0
    no_hit_pred = 0
    no_hit_correct = 0

    for r in rows:
        predicted_hit = r["prediction"] == "HIT"
        actually_hit = r["got_hit"] == 1

        if predicted_hit:
            hit_pred += 1
            if actually_hit:
                correct += 1
                hit_correct += 1
        else:
            no_hit_pred += 1
            if not actually_hit:
                correct += 1
                no_hit_correct += 1

    accuracy = correct / total if total > 0 else 0

    summary = {
        "total": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": round(accuracy * 100, 1),
        "hit_predictions": hit_pred,
        "hit_correct": hit_correct,
        "hit_accuracy": round(hit_correct / hit_pred * 100, 1) if hit_pred > 0 else 0,
        "no_hit_predictions": no_hit_pred,
        "no_hit_correct": no_hit_correct,
        "no_hit_accuracy": round(no_hit_correct / no_hit_pred * 100, 1) if no_hit_pred > 0 else 0,
    }

    # Upsert daily summary
    db.execute("""
        INSERT OR REPLACE INTO daily_summary (
            game_date, total_predictions, correct, incorrect, pending, accuracy,
            hit_predictions, hit_correct, no_hit_predictions, no_hit_correct
        ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
    """, (
        game_date, total, correct, total - correct, accuracy,
        hit_pred, hit_correct, no_hit_pred, no_hit_correct,
    ))
    db.commit()

    return summary


# ── Reporting ────────────────────────────────────────────────────────


def get_overall_stats() -> dict[str, Any]:
    """Get lifetime prediction accuracy stats."""
    with contextlib.closing(_get_db()) as db:
        row = db.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN (prediction = 'HIT' AND got_hit = 1) OR (prediction = 'NO HIT' AND got_hit = 0) THEN 1 ELSE 0 END) as correct,
                SUM(CASE WHEN prediction = 'HIT' THEN 1 ELSE 0 END) as hit_preds,
                SUM(CASE WHEN prediction = 'HIT' AND got_hit = 1 THEN 1 ELSE 0 END) as hit_correct,
                SUM(CASE WHEN prediction = 'NO HIT' THEN 1 ELSE 0 END) as no_hit_preds,
                SUM(CASE WHEN prediction = 'NO HIT' AND got_hit = 0 THEN 1 ELSE 0 END) as no_hit_correct
            FROM predictions
            WHERE actual_result IS NOT NULL AND actual_result != 'DNP'
        """).fetchone()

        total = row["total"] or 0
        correct = row["correct"] or 0
        hit_preds = row["hit_preds"] or 0
        hit_correct = row["hit_correct"] or 0
        no_hit_preds = row["no_hit_preds"] or 0
        no_hit_correct = row["no_hit_correct"] or 0

        pending = db.execute(
            "SELECT COUNT(*) as c FROM predictions WHERE actual_result IS NULL"
        ).fetchone()["c"]

        conf_stats = {}
        for conf in ["high", "medium", "low", "insufficient"]:
            crow = db.execute("""
                SELECT COUNT(*) as total,
                    SUM(CASE WHEN (prediction = 'HIT' AND got_hit = 1) OR (prediction = 'NO HIT' AND got_hit = 0) THEN 1 ELSE 0 END) as correct
                FROM predictions
                WHERE confidence = ? AND actual_result IS NOT NULL AND actual_result != 'DNP'
            """, (conf,)).fetchone()
            ct = crow["total"] or 0
            cc = crow["correct"] or 0
            conf_stats[conf] = {"total": ct, "correct": cc, "accuracy": round(cc / ct * 100, 1) if ct > 0 else 0}

    return {
        "total_predictions": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 1) if total > 0 else 0,
        "pending": pending,
        "hit_predictions": {"total": hit_preds, "correct": hit_correct,
                            "accuracy": round(hit_correct / hit_preds * 100, 1) if hit_preds > 0 else 0},
        "no_hit_predictions": {"total": no_hit_preds, "correct": no_hit_correct,
                               "accuracy": round(no_hit_correct / no_hit_preds * 100, 1) if no_hit_preds > 0 else 0},
        "by_confidence": conf_stats,
    }


def get_tier_stats(game_date: str | None = None) -> list[dict]:
    """Get hit rate by probability tier, optionally for a specific date."""
    with contextlib.closing(_get_db()) as db:
        date_filter = "AND game_date = ?" if game_date else ""
        params = (game_date,) if game_date else ()
        rows = db.execute(f"""
            SELECT
                CASE
                    WHEN hit_probability >= {TIER_STRONG} THEN 'STRONG HIT'
                    WHEN hit_probability >= {TIER_LEAN} THEN 'LEAN HIT'
                    WHEN hit_probability >= {TIER_TOSSUP} THEN 'TOSS-UP'
                    ELSE 'FADE'
                END as tier,
                COUNT(*) as total,
                SUM(got_hit) as hits,
                ROUND(AVG(got_hit) * 100, 1) as hit_rate
            FROM predictions
            WHERE actual_result IS NOT NULL AND actual_result != 'DNP' {date_filter}
            GROUP BY tier
            ORDER BY hit_rate DESC
        """, params).fetchall()
    return [dict(r) for r in rows]


def get_calibration_scores(game_date: str | None = None) -> dict[str, Any]:
    """Compute Brier score and log loss for model calibration.

    Brier score: mean squared error between predicted prob and outcome (lower is better).
    Log loss: mean negative log-likelihood (lower is better, penalizes confident wrong predictions).
    """
    import math

    with contextlib.closing(_get_db()) as db:
        date_filter = "AND game_date = ?" if game_date else ""
        params = (game_date,) if game_date else ()
        rows = db.execute(f"""
            SELECT hit_probability, got_hit FROM predictions
            WHERE actual_result IS NOT NULL AND actual_result != 'DNP'
              AND got_hit IS NOT NULL {date_filter}
        """, params).fetchall()

    if not rows:
        return {"total": 0}

    n = len(rows)
    brier_sum = 0.0
    log_loss_sum = 0.0
    eps = 1e-15  # avoid log(0)

    for r in rows:
        p = max(eps, min(1 - eps, r["hit_probability"]))
        y = r["got_hit"]
        brier_sum += (p - y) ** 2
        log_loss_sum += -(y * math.log(p) + (1 - y) * math.log(1 - p))

    return {
        "total": n,
        "brier_score": round(brier_sum / n, 4),
        "log_loss": round(log_loss_sum / n, 4),
    }


def get_recent_predictions(limit: int = 20) -> list[dict]:
    """Get most recent predictions with results."""
    with contextlib.closing(_get_db()) as db:
        rows = db.execute("""
            SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ── Book odds storage ────────────────────────────────────────────


def save_book_odds(
    game_pk: int, game_date: str, batter_name: str, batter_id: int,
    model_prob: float, odds_entries: list[dict],
) -> None:
    """Save book odds for a batter. One row per book."""
    with contextlib.closing(_get_db()) as db:
        for entry in odds_entries:
            over = entry.get("over")
            if over is None:
                continue
            implied = american_to_prob(over)
            edge = model_prob - implied

            db.execute("""
                INSERT INTO book_odds (
                    game_pk, game_date, batter_name, batter_id,
                    book, line, over_price, under_price,
                    implied_prob, model_prob, edge
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(game_pk, batter_name, book, line) DO UPDATE SET
                    over_price = excluded.over_price,
                    under_price = excluded.under_price,
                    implied_prob = excluded.implied_prob,
                    edge = excluded.edge,
                    fetched_at = CURRENT_TIMESTAMP
            """, (
                game_pk, game_date, batter_name, batter_id,
                entry.get("book", "unknown"), entry.get("line", 0.5),
                over, entry.get("under"),
                round(implied, 4), round(model_prob, 4), round(edge, 4),
            ))
        db.commit()


def update_closing_odds(
    game_pk: int, game_date: str, batter_name: str,
    book: str, closing_over: int,
) -> None:
    """Update a book_odds row with closing line and compute CLV."""
    closing_implied = american_to_prob(closing_over)
    with contextlib.closing(_get_db()) as db:
        # Find the matching opening odds row
        row = db.execute("""
            SELECT id, implied_prob FROM book_odds
            WHERE game_pk = ? AND batter_name = ? AND book = ?
              AND line = 0.5 AND closing_over_price IS NULL
            ORDER BY edge DESC LIMIT 1
        """, (game_pk, batter_name, book)).fetchone()
        if row:
            clv = closing_implied - row["implied_prob"]  # positive = line moved our way (we got value)
            db.execute("""
                UPDATE book_odds SET
                    closing_over_price = ?,
                    closing_implied_prob = ?,
                    clv = ?
                WHERE id = ?
            """, (closing_over, round(closing_implied, 4), round(clv, 4), row["id"]))
            db.commit()


def games_needing_closing_odds(game_date: str) -> set[int]:
    """Get game_pks that have opening odds but no closing odds yet."""
    with contextlib.closing(_get_db()) as db:
        rows = db.execute("""
            SELECT DISTINCT game_pk FROM book_odds
            WHERE game_date = ? AND line = 0.5
              AND closing_over_price IS NULL
        """, (game_date,)).fetchall()
    return {r["game_pk"] for r in rows}


def get_clv_stats(game_date: str | None = None) -> dict[str, Any]:
    """Get CLV statistics, optionally filtered to a date."""
    with contextlib.closing(_get_db()) as db:
        date_filter = "AND game_date = ?" if game_date else ""
        params = (game_date,) if game_date else ()

        row = db.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive,
                AVG(clv) as avg_clv,
                MIN(clv) as min_clv,
                MAX(clv) as max_clv
            FROM book_odds
            WHERE clv IS NOT NULL AND line = 0.5 {date_filter}
        """, params).fetchone()

        total = row["total"] or 0
        if total == 0:
            return {"total": 0, "message": "No CLV data yet"}

        positive = row["positive"] or 0
        avg_clv = row["avg_clv"] or 0

        # CLV by tier
        tier_clv = db.execute(f"""
            SELECT
                CASE
                    WHEN model_prob >= 0.70 THEN 'STRONG HIT'
                    WHEN model_prob >= 0.62 THEN 'LEAN HIT'
                    WHEN model_prob >= 0.55 THEN 'TOSS-UP'
                    ELSE 'FADE'
                END as tier,
                COUNT(*) as n,
                AVG(clv) as avg_clv,
                SUM(CASE WHEN clv > 0 THEN 1 ELSE 0 END) as positive
            FROM book_odds
            WHERE clv IS NOT NULL AND line = 0.5 {date_filter}
            GROUP BY tier ORDER BY avg_clv DESC
        """, params).fetchall()

    return {
        "total": total,
        "positive": positive,
        "positive_pct": round(positive / total * 100, 1),
        "avg_clv": round(avg_clv * 100, 2),  # as percentage
        "min_clv": round((row["min_clv"] or 0) * 100, 2),
        "max_clv": round((row["max_clv"] or 0) * 100, 2),
        "by_tier": [dict(r) for r in tier_clv],
    }


def get_odds_for_date(game_date: str) -> list[dict]:
    """Get all saved odds for a date."""
    with contextlib.closing(_get_db()) as db:
        rows = db.execute("""
            SELECT * FROM book_odds WHERE game_date = ? ORDER BY edge DESC
        """, (game_date,)).fetchall()
    return [dict(r) for r in rows]


def games_with_odds(game_date: str) -> set[int]:
    """Get game_pks that already have odds fetched for a date."""
    with contextlib.closing(_get_db()) as db:
        rows = db.execute(
            "SELECT DISTINCT game_pk FROM book_odds WHERE game_date = ?",
            (game_date,)
        ).fetchall()
    return {r["game_pk"] for r in rows}


# ── Paper betting ────────────────────────────────────────────────


def place_paper_bets(game_date: str, max_bets: int = 30) -> list[dict]:
    """Pick the top +EV bets for a date from stored odds and predictions.

    Called on every odds fetch. Skips batters already bet on, places new
    bets up to max_bets total for the day. This allows incremental betting
    as later games post lineups and odds throughout the day.

    Thresholds set from sweep on 740+ opportunities: 70%+ prob is the only
    range where hit rate consistently beats the vig across all edge buckets.
    """
    with contextlib.closing(_get_db()) as db:
        existing_count = db.execute(
            "SELECT COUNT(*) as c FROM paper_bets WHERE game_date = ?",
            (game_date,)
        ).fetchone()["c"]
        if existing_count >= max_bets:
            return []

        already_bet = {r["batter_name"] for r in db.execute(
            "SELECT DISTINCT batter_name FROM paper_bets WHERE game_date = ?",
            (game_date,)
        ).fetchall()}

        rows = db.execute("""
            SELECT bo.*, p.prediction, p.confidence, p.edge as pred_edge, p.pitcher_name
            FROM book_odds bo
            JOIN predictions p ON bo.game_pk = p.game_pk AND bo.batter_name = p.batter_name
            WHERE bo.game_date = ?
              AND bo.line = 0.5
              AND bo.model_prob >= 0.70
              AND bo.edge > 0.02
              AND p.prediction = 'HIT'
            ORDER BY bo.edge DESC
        """, (game_date,)).fetchall()

        if not rows:
            return []

        seen = set(already_bet)
        candidates = []
        for r in rows:
            if r["batter_name"] not in seen:
                seen.add(r["batter_name"])
                candidates.append(dict(r))

        slots = max_bets - existing_count
        bets = candidates[:slots]

        for bet in bets:
            odds = bet["over_price"]
            db.execute("""
                INSERT INTO paper_bets (
                    game_pk, game_date, batter_name, batter_id,
                    pitcher_name, model_prob, confidence, edge_tags,
                    book, book_odds, implied_prob, edge, stake
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 100)
            """, (
                bet["game_pk"], game_date, bet["batter_name"], bet["batter_id"],
                bet.get("pitcher_name", ""), bet["model_prob"],
                bet.get("confidence", ""), bet.get("pred_edge", ""),
                bet["book"], odds, bet["implied_prob"], bet["edge"],
            ))

        db.commit()
    return bets


def settle_paper_bets(game_date: str = None) -> dict[str, Any]:
    """Settle all unsettled paper bets that have results.

    If game_date is given, only settles that date.
    Otherwise settles ALL unsettled bets with available results.
    """
    with contextlib.closing(_get_db()) as db:
        if game_date:
            date_filter = "AND pb.game_date = ?"
            params = (game_date,)
        else:
            date_filter = ""
            params = ()

        unsettled = db.execute(f"""
            SELECT pb.id, pb.batter_name, pb.game_pk, pb.game_date, pb.book_odds, pb.stake,
                   (SELECT p.got_hit FROM predictions p
                    WHERE p.game_pk = pb.game_pk AND p.batter_name = pb.batter_name
                      AND p.got_hit IS NOT NULL
                    LIMIT 1) as got_hit
            FROM paper_bets pb
            WHERE pb.settled = 0 {date_filter}
        """, params).fetchall()

        if not unsettled:
            return {"date": game_date, "settled": 0}

        total_pnl = 0.0
        wins = 0
        losses = 0
        settled = 0

        for bet in unsettled:
            got_hit = bet["got_hit"]
            if got_hit is None:
                continue

            odds = bet["book_odds"]
            stake = bet["stake"]

            if got_hit == 1:
                if odds < 0:
                    pnl = stake * (100 / abs(odds))
                else:
                    pnl = stake * (odds / 100)
                wins += 1
            else:
                pnl = -stake
                losses += 1

            pnl = round(pnl, 2)
            total_pnl += pnl

            db.execute("""
                UPDATE paper_bets SET
                    result = ?, got_hit = ?, pnl = ?, settled = 1
                WHERE id = ?
            """, (
                "WIN" if got_hit == 1 else "LOSS",
                got_hit, pnl, bet["id"],
            ))
            settled += 1

        db.commit()

    return {
        "date": game_date or "all",
        "settled": settled,
        "wins": wins,
        "losses": losses,
        "pnl": round(total_pnl, 2),
    }


def get_paper_summary() -> dict[str, Any]:
    """Get overall paper betting performance."""
    with contextlib.closing(_get_db()) as db:
        row = db.execute("""
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                SUM(stake) as total_staked,
                COUNT(DISTINCT game_date) as days
            FROM paper_bets WHERE settled = 1
        """).fetchone()

        total = row["total_bets"] or 0
        wins = row["wins"] or 0
        losses = row["losses"] or 0
        pnl = row["total_pnl"] or 0.0
        staked = row["total_staked"] or 0.0
        days = row["days"] or 0

        pending = db.execute(
            "SELECT COUNT(*) as c FROM paper_bets WHERE settled = 0"
        ).fetchone()["c"]

        daily = db.execute("""
            SELECT game_date,
                COUNT(*) as bets,
                SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as pnl
            FROM paper_bets WHERE settled = 1
            GROUP BY game_date ORDER BY game_date
        """).fetchall()

    return {
        "total_bets": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "total_pnl": round(pnl, 2),
        "roi": round(pnl / staked * 100, 1) if staked > 0 else 0,
        "pending": pending,
        "days": days,
        "avg_daily_pnl": round(pnl / days, 2) if days > 0 else 0,
        "daily": [dict(d) for d in daily],
    }


def get_paper_bets_for_date(game_date: str) -> list[dict]:
    """Get all paper bets for a specific date."""
    with contextlib.closing(_get_db()) as db:
        rows = db.execute("""
            SELECT * FROM paper_bets WHERE game_date = ? ORDER BY edge DESC
        """, (game_date,)).fetchall()
    return [dict(r) for r in rows]
