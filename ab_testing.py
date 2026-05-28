"""A/B testing framework for comparing prediction models.

Runs a shadow model alongside the primary model on every prediction.
Shadow predictions are stored in a separate table and never affect
paper bets or alerts. After settlement, both models' predictions are
compared head-to-head on calibration, tier accuracy, and edge detection.

Usage:
  1. Register a shadow model via register_shadow_model()
  2. The formatter calls run_shadow_prediction() after each primary prediction
  3. Settlement automatically settles shadow predictions
  4. Use /ab command or compare_models() to see results
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sqlite3
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

from predictor import HitPrediction, PredictionFactors

logger = logging.getLogger("baseball_bot.ab_testing")

DB_PATH = Path(__file__).parent / "predictions.db"

# ── Shadow model registry ──────────────────────────────────────────

# A shadow model is a function with the same signature as predict_hit
# that returns a HitPrediction. Register one to run A/B tests.
_shadow_model: dict[str, Any] | None = None


def register_shadow_model(
    name: str,
    version: str,
    predict_fn: Callable[..., HitPrediction],
    description: str = "",
) -> None:
    """Register a shadow model for A/B testing.

    Args:
        name: Human-readable name (e.g., "xBA-weighted")
        version: Version string (e.g., "v4.0-alpha")
        predict_fn: Function with same signature as predictor.predict_hit
        description: What's different about this model
    """
    global _shadow_model
    _shadow_model = {
        "name": name,
        "version": version,
        "predict_fn": predict_fn,
        "description": description,
    }
    logger.info(f"Shadow model registered: {name} ({version}) — {description}")


def get_shadow_model() -> dict[str, Any] | None:
    """Get the currently registered shadow model, if any."""
    return _shadow_model


def clear_shadow_model() -> None:
    """Remove the shadow model."""
    global _shadow_model
    _shadow_model = None
    logger.info("Shadow model cleared")


# ── Database ───────────────────────────────────────────────────────

_ab_tables_initialized = False


def _init_ab_tables(db: sqlite3.Connection) -> None:
    """Create the shadow predictions table."""
    global _ab_tables_initialized
    if _ab_tables_initialized:
        return

    db.execute("""
        CREATE TABLE IF NOT EXISTS shadow_predictions (
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
            model_name TEXT,
            model_version TEXT,
            factors_json TEXT DEFAULT NULL,
            actual_result TEXT DEFAULT NULL,
            got_hit INTEGER DEFAULT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(game_pk, batter_id, model_version)
        )
    """)
    db.commit()
    _ab_tables_initialized = True


def _get_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    _init_ab_tables(db)
    return db


# ── Shadow prediction runner ──────────────────────────────────────


def run_shadow_prediction(**kwargs) -> HitPrediction | None:
    """Run the shadow model on the same inputs as the primary model.

    Call this with the exact same kwargs as predict_hit(). If no shadow
    model is registered, returns None silently.
    """
    if _shadow_model is None:
        return None

    try:
        pred = _shadow_model["predict_fn"](**kwargs)
        _save_shadow_prediction(pred, _shadow_model["name"], _shadow_model["version"])
        return pred
    except Exception as e:
        logger.warning(f"Shadow model failed: {e}")
        return None


def _save_shadow_prediction(
    pred: HitPrediction, model_name: str, model_version: str,
) -> None:
    """Save a shadow prediction to the database."""
    with contextlib.closing(_get_db()) as db:
        db.execute("""
            INSERT OR REPLACE INTO shadow_predictions (
                game_pk, game_date, batter_name, batter_id,
                pitcher_name, pitcher_id, venue,
                prediction, hit_probability, confidence, edge,
                model_name, model_version, factors_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred.game_pk, pred.game_date, pred.batter_name, pred.batter_id,
            pred.pitcher_name, pred.pitcher_id, pred.venue,
            pred.prediction, pred.hit_probability, pred.confidence, pred.edge,
            model_name, model_version, json.dumps(asdict(pred.factors)),
        ))
        db.commit()


# ── Settlement ─────────────────────────────────────────────────────


def settle_shadow_predictions(game_date: str | None = None) -> int:
    """Settle shadow predictions using results from the primary predictions table.

    Copies got_hit and actual_result from predictions to shadow_predictions
    for matching (game_pk, batter_id) pairs.

    Returns the number of shadow predictions settled.
    """
    with contextlib.closing(_get_db()) as db:
        date_filter = "AND sp.game_date = ?" if game_date else ""
        params = (game_date,) if game_date else ()

        settled = db.execute(f"""
            UPDATE shadow_predictions
            SET got_hit = (
                    SELECT p.got_hit FROM predictions p
                    WHERE p.game_pk = shadow_predictions.game_pk
                      AND p.batter_id = shadow_predictions.batter_id
                      AND p.got_hit IS NOT NULL
                    LIMIT 1
                ),
                actual_result = (
                    SELECT p.actual_result FROM predictions p
                    WHERE p.game_pk = shadow_predictions.game_pk
                      AND p.batter_id = shadow_predictions.batter_id
                      AND p.actual_result IS NOT NULL
                    LIMIT 1
                )
            WHERE shadow_predictions.got_hit IS NULL
              AND EXISTS (
                  SELECT 1 FROM predictions p
                  WHERE p.game_pk = shadow_predictions.game_pk
                    AND p.batter_id = shadow_predictions.batter_id
                    AND p.got_hit IS NOT NULL
              )
              {date_filter}
        """, params)
        count = settled.rowcount
        db.commit()

    if count > 0:
        logger.info(f"Settled {count} shadow predictions")
    return count


# ── Comparison ─────────────────────────────────────────────────────


def _compute_model_metrics(rows: list) -> dict[str, Any]:
    """Compute metrics for a set of predictions."""
    if not rows:
        return {"n": 0}

    n = len(rows)
    eps = 1e-15
    brier_sum = 0.0
    ll_sum = 0.0

    tier_data = {
        "STRONG HIT": [0, 0],
        "LEAN HIT": [0, 0],
        "TOSS-UP": [0, 0],
        "FADE": [0, 0],
    }

    for r in rows:
        p = max(eps, min(1 - eps, r["hit_probability"]))
        y = r["got_hit"]
        brier_sum += (p - y) ** 2
        ll_sum += -(y * math.log(p) + (1 - y) * math.log(1 - p))

        if p >= 0.70:
            tier = "STRONG HIT"
        elif p >= 0.62:
            tier = "LEAN HIT"
        elif p >= 0.55:
            tier = "TOSS-UP"
        else:
            tier = "FADE"
        tier_data[tier][1] += 1
        if y == 1:
            tier_data[tier][0] += 1

    hit_rate = sum(r["got_hit"] for r in rows) / n

    return {
        "n": n,
        "brier": round(brier_sum / n, 4),
        "log_loss": round(ll_sum / n, 4),
        "hit_rate": round(hit_rate, 3),
        "tiers": {
            tier: {"hits": h, "total": t, "rate": round(h / t, 3) if t > 0 else None}
            for tier, (h, t) in tier_data.items()
        },
    }


def compare_models(days_back: int | None = None) -> dict[str, Any]:
    """Compare primary model vs shadow model(s) on overlapping predictions.

    Only compares predictions where BOTH models made a prediction for the
    same (game_pk, batter_id), ensuring an apples-to-apples comparison.

    Returns metrics for each model plus head-to-head stats.
    """
    with contextlib.closing(_get_db()) as db:
        date_filter = ""
        params: list = []
        if days_back is not None:
            cutoff = (date.today() - timedelta(days=days_back)).isoformat()
            date_filter = "AND p.game_date >= ?"
            params.append(cutoff)

        # Get shadow model versions
        versions = db.execute(
            "SELECT DISTINCT model_version, model_name FROM shadow_predictions"
        ).fetchall()

        if not versions:
            return {"message": "No shadow predictions found. Register a shadow model first."}

        results = {}

        for v in versions:
            mv = v["model_version"]
            mn = v["model_name"]

            # Find overlapping predictions (both primary and shadow have results)
            overlapping = db.execute(f"""
                SELECT
                    p.hit_probability as primary_prob,
                    p.got_hit,
                    p.game_pk,
                    p.batter_id,
                    sp.hit_probability as shadow_prob,
                    sp.prediction as shadow_prediction,
                    p.prediction as primary_prediction,
                    p.batter_name,
                    p.game_date
                FROM predictions p
                JOIN shadow_predictions sp
                    ON p.game_pk = sp.game_pk AND p.batter_id = sp.batter_id
                WHERE p.got_hit IS NOT NULL
                  AND sp.got_hit IS NOT NULL
                  AND sp.model_version = ?
                  {date_filter}
                ORDER BY p.game_date
            """, [mv] + params).fetchall()

            if not overlapping:
                results[mv] = {"name": mn, "version": mv, "n": 0, "message": "No overlapping settled predictions"}
                continue

            # Build separate row lists for each model
            primary_rows = [{"hit_probability": r["primary_prob"], "got_hit": r["got_hit"]} for r in overlapping]
            shadow_rows = [{"hit_probability": r["shadow_prob"], "got_hit": r["got_hit"]} for r in overlapping]

            primary_metrics = _compute_model_metrics(primary_rows)
            shadow_metrics = _compute_model_metrics(shadow_rows)

            # Head-to-head: which model was closer to the outcome?
            primary_wins = 0
            shadow_wins = 0
            ties = 0
            for r in overlapping:
                p_err = abs(r["primary_prob"] - r["got_hit"])
                s_err = abs(r["shadow_prob"] - r["got_hit"])
                if p_err < s_err - 0.001:
                    primary_wins += 1
                elif s_err < p_err - 0.001:
                    shadow_wins += 1
                else:
                    ties += 1

            # Disagreements: where models differ on prediction and one was right
            disagree_shadow_right = 0
            disagree_primary_right = 0
            disagree_both_wrong = 0
            for r in overlapping:
                p_pred = 1 if r["primary_prob"] >= 0.46 else 0
                s_pred = 1 if r["shadow_prob"] >= 0.46 else 0
                actual = r["got_hit"]
                if p_pred != s_pred:
                    if s_pred == actual and p_pred != actual:
                        disagree_shadow_right += 1
                    elif p_pred == actual and s_pred != actual:
                        disagree_primary_right += 1
                    else:
                        disagree_both_wrong += 1

            results[mv] = {
                "name": mn,
                "version": mv,
                "n": len(overlapping),
                "primary": primary_metrics,
                "shadow": shadow_metrics,
                "head_to_head": {
                    "primary_closer": primary_wins,
                    "shadow_closer": shadow_wins,
                    "ties": ties,
                },
                "disagreements": {
                    "shadow_right": disagree_shadow_right,
                    "primary_right": disagree_primary_right,
                    "both_wrong": disagree_both_wrong,
                },
            }

    return results


def format_ab_report(results: dict[str, Any]) -> str:
    """Format A/B comparison results as a Telegram message."""
    if "message" in results:
        return f"\U0001f9ea A/B Test: {results['message']}"

    lines = ["\U0001f9ea A/B Model Comparison\n"]

    for mv, data in results.items():
        if data.get("n", 0) == 0:
            lines.append(f"{data['name']} ({data['version']}): No overlapping data yet")
            continue

        n = data["n"]
        pm = data["primary"]
        sm = data["shadow"]
        h2h = data["head_to_head"]
        dis = data["disagreements"]

        lines.append(f"{'=' * 34}")
        lines.append(f"\U0001f194 {data['name']} ({data['version']})")
        lines.append(f"Matched predictions: {n}\n")

        # Metrics comparison table
        lines.append(f"{'Metric':16s} {'Primary':>10s} {'Shadow':>10s} {'Winner':>8s}")
        lines.append(f"{'-' * 48}")

        brier_winner = "Shadow" if sm["brier"] < pm["brier"] else "Primary" if pm["brier"] < sm["brier"] else "Tie"
        ll_winner = "Shadow" if sm["log_loss"] < pm["log_loss"] else "Primary" if pm["log_loss"] < sm["log_loss"] else "Tie"

        lines.append(f"{'Brier':16s} {pm['brier']:10.4f} {sm['brier']:10.4f} {brier_winner:>8s}")
        lines.append(f"{'Log Loss':16s} {pm['log_loss']:10.4f} {sm['log_loss']:10.4f} {ll_winner:>8s}")

        # Tier comparison
        lines.append(f"\nTier hit rates:")
        for tier in ["STRONG HIT", "LEAN HIT", "TOSS-UP", "FADE"]:
            pt = pm["tiers"][tier]
            st = sm["tiers"][tier]
            p_str = f"{pt['rate']:.0%}" if pt["rate"] is not None else "n/a"
            s_str = f"{st['rate']:.0%}" if st["rate"] is not None else "n/a"
            p_detail = f"({pt['hits']}/{pt['total']})" if pt["total"] > 0 else ""
            s_detail = f"({st['hits']}/{st['total']})" if st["total"] > 0 else ""
            lines.append(f"  {tier:12s}: Primary {p_str} {p_detail:>8s} | Shadow {s_str} {s_detail:>8s}")

        # Head to head
        lines.append(f"\nHead-to-head (closer to outcome):")
        total_h2h = h2h["primary_closer"] + h2h["shadow_closer"] + h2h["ties"]
        lines.append(
            f"  Primary: {h2h['primary_closer']} ({h2h['primary_closer']/total_h2h:.0%}) | "
            f"Shadow: {h2h['shadow_closer']} ({h2h['shadow_closer']/total_h2h:.0%}) | "
            f"Ties: {h2h['ties']}"
        )

        # Disagreements
        total_dis = dis["shadow_right"] + dis["primary_right"] + dis["both_wrong"]
        if total_dis > 0:
            lines.append(f"\nDisagreements ({total_dis} cases):")
            lines.append(f"  Shadow right: {dis['shadow_right']} | Primary right: {dis['primary_right']} | Both wrong: {dis['both_wrong']}")

        # Verdict
        shadow_score = 0
        if sm["brier"] < pm["brier"]:
            shadow_score += 1
        if sm["log_loss"] < pm["log_loss"]:
            shadow_score += 1
        if h2h["shadow_closer"] > h2h["primary_closer"]:
            shadow_score += 1
        if dis["shadow_right"] > dis["primary_right"]:
            shadow_score += 1

        if shadow_score >= 3:
            verdict = "\U0001f7e2 Shadow model is BETTER — consider promoting"
        elif shadow_score <= 1:
            verdict = "\U0001f534 Primary model is BETTER — keep current"
        else:
            verdict = "\U0001f7e1 Too close to call — need more data"

        lines.append(f"\n{verdict}")

    return "\n".join(lines)
