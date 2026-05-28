"""Calibration drift detection for the baseball bot.

Monitors model calibration over rolling windows and detects degradation.
Compares recent performance (last 3 and 7 days) against the lifetime
baseline. Alerts via Telegram when drift exceeds thresholds.

Drift signals:
  - Brier score increase > 0.03 vs lifetime (model getting worse)
  - Tier separation collapse (STRONG - FADE gap shrinks below 10pp)
  - Systematic bias (model over/under-predicting by > 5% in a bucket)
  - Hit rate deviation (actual base rate diverges from expected)
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger("baseball_bot.drift")

DB_PATH = Path(__file__).parent / "predictions.db"

# ── Thresholds ─────────────────────────────────────────────────────

BRIER_DRIFT_THRESHOLD = 0.03     # alert if recent Brier > lifetime + this
LOG_LOSS_DRIFT_THRESHOLD = 0.05  # alert if recent log loss > lifetime + this
TIER_SEPARATION_MIN = 0.10       # alert if STRONG - FADE hit rate gap < 10pp
CALIBRATION_BIAS_THRESHOLD = 0.05  # alert if any bucket off by > 5pp
MIN_PREDICTIONS_FOR_CHECK = 50   # need at least this many to run drift check
MIN_PREDICTIONS_PER_BUCKET = 10  # need this many per calibration bucket


def _get_db() -> sqlite3.Connection:
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    return db


def _compute_metrics(rows: list) -> dict[str, Any]:
    """Compute Brier, log loss, tier hit rates from prediction rows."""
    if not rows:
        return {}

    n = len(rows)
    brier_sum = 0.0
    log_loss_sum = 0.0
    eps = 1e-15

    tier_counts = {
        "STRONG HIT": {"hits": 0, "total": 0},
        "LEAN HIT": {"hits": 0, "total": 0},
        "TOSS-UP": {"hits": 0, "total": 0},
        "FADE": {"hits": 0, "total": 0},
    }

    # Calibration buckets (5pp wide)
    cal_buckets: dict[str, dict] = {}

    for r in rows:
        p = max(eps, min(1 - eps, r["hit_probability"]))
        y = r["got_hit"]

        brier_sum += (p - y) ** 2
        log_loss_sum += -(y * math.log(p) + (1 - y) * math.log(1 - p))

        # Tier classification
        if p >= 0.70:
            tier = "STRONG HIT"
        elif p >= 0.62:
            tier = "LEAN HIT"
        elif p >= 0.55:
            tier = "TOSS-UP"
        else:
            tier = "FADE"

        tier_counts[tier]["total"] += 1
        if y == 1:
            tier_counts[tier]["hits"] += 1

        # Calibration bucket
        bucket_lo = int(p * 20) * 5  # 5pp buckets: 25, 30, 35, ..., 85
        bucket_key = f"{bucket_lo}-{bucket_lo + 5}"
        if bucket_key not in cal_buckets:
            cal_buckets[bucket_key] = {"pred_sum": 0.0, "actual_sum": 0, "n": 0}
        cal_buckets[bucket_key]["pred_sum"] += p
        cal_buckets[bucket_key]["actual_sum"] += y
        cal_buckets[bucket_key]["n"] += 1

    # Tier hit rates
    tier_rates = {}
    for tier, counts in tier_counts.items():
        if counts["total"] > 0:
            tier_rates[tier] = counts["hits"] / counts["total"]
        else:
            tier_rates[tier] = None

    # Calibration gaps per bucket
    cal_gaps = {}
    for bucket, data in cal_buckets.items():
        if data["n"] >= MIN_PREDICTIONS_PER_BUCKET:
            avg_pred = data["pred_sum"] / data["n"]
            avg_actual = data["actual_sum"] / data["n"]
            cal_gaps[bucket] = {
                "predicted": round(avg_pred, 3),
                "actual": round(avg_actual, 3),
                "gap": round(avg_actual - avg_pred, 3),
                "n": data["n"],
            }

    actual_hit_rate = sum(r["got_hit"] for r in rows) / n

    return {
        "n": n,
        "brier": round(brier_sum / n, 4),
        "log_loss": round(log_loss_sum / n, 4),
        "hit_rate": round(actual_hit_rate, 3),
        "tier_rates": tier_rates,
        "calibration": cal_gaps,
    }


def _fetch_settled_predictions(
    days_back: int | None = None,
    model_version: str | None = None,
) -> list:
    """Fetch settled predictions, optionally filtered by recency or model version."""
    with contextlib.closing(_get_db()) as db:
        conditions = [
            "actual_result IS NOT NULL",
            "actual_result != 'DNP'",
            "got_hit IS NOT NULL",
        ]
        params: list = []

        if days_back is not None:
            cutoff = (date.today() - timedelta(days=days_back)).isoformat()
            conditions.append("game_date >= ?")
            params.append(cutoff)

        if model_version is not None:
            conditions.append("model_version = ?")
            params.append(model_version)

        where = " AND ".join(conditions)
        rows = db.execute(
            f"SELECT hit_probability, got_hit, game_date, model_version FROM predictions WHERE {where}",
            params,
        ).fetchall()

    return rows


def check_drift() -> dict[str, Any]:
    """Run drift detection. Returns alerts and metrics.

    Compares 3-day and 7-day rolling windows against lifetime baseline.
    Returns a dict with:
      - alerts: list of string alerts (empty = healthy)
      - lifetime: lifetime metrics
      - recent_7d: 7-day rolling metrics
      - recent_3d: 3-day rolling metrics
      - status: "healthy", "warning", or "degraded"
    """
    lifetime_rows = _fetch_settled_predictions()
    if len(lifetime_rows) < MIN_PREDICTIONS_FOR_CHECK:
        return {
            "status": "insufficient_data",
            "message": f"Need {MIN_PREDICTIONS_FOR_CHECK}+ settled predictions, have {len(lifetime_rows)}",
            "alerts": [],
        }

    lifetime = _compute_metrics(lifetime_rows)

    # Rolling windows
    rows_7d = _fetch_settled_predictions(days_back=7)
    rows_3d = _fetch_settled_predictions(days_back=3)
    recent_7d = _compute_metrics(rows_7d) if len(rows_7d) >= 20 else {}
    recent_3d = _compute_metrics(rows_3d) if len(rows_3d) >= 10 else {}

    alerts: list[str] = []

    # ── Check 1: Brier score drift ──
    for label, recent in [("7d", recent_7d), ("3d", recent_3d)]:
        if not recent:
            continue
        brier_delta = recent["brier"] - lifetime["brier"]
        if brier_delta > BRIER_DRIFT_THRESHOLD:
            alerts.append(
                f"Brier drift ({label}): {recent['brier']:.4f} vs {lifetime['brier']:.4f} "
                f"(+{brier_delta:.4f}, threshold {BRIER_DRIFT_THRESHOLD})"
            )

    # ── Check 2: Log loss drift ──
    for label, recent in [("7d", recent_7d), ("3d", recent_3d)]:
        if not recent:
            continue
        ll_delta = recent["log_loss"] - lifetime["log_loss"]
        if ll_delta > LOG_LOSS_DRIFT_THRESHOLD:
            alerts.append(
                f"Log loss drift ({label}): {recent['log_loss']:.4f} vs {lifetime['log_loss']:.4f} "
                f"(+{ll_delta:.4f}, threshold {LOG_LOSS_DRIFT_THRESHOLD})"
            )

    # ── Check 3: Tier separation collapse ──
    for label, recent in [("7d", recent_7d), ("3d", recent_3d)]:
        if not recent:
            continue
        strong = recent["tier_rates"].get("STRONG HIT")
        fade = recent["tier_rates"].get("FADE")
        if strong is not None and fade is not None:
            separation = strong - fade
            if separation < TIER_SEPARATION_MIN:
                alerts.append(
                    f"Tier separation collapse ({label}): STRONG {strong:.1%} - FADE {fade:.1%} = "
                    f"{separation:.1%} (min {TIER_SEPARATION_MIN:.0%})"
                )

    # ── Check 4: Calibration bias ──
    if recent_7d:
        for bucket, data in recent_7d.get("calibration", {}).items():
            if abs(data["gap"]) > CALIBRATION_BIAS_THRESHOLD:
                direction = "over-predicting" if data["gap"] < 0 else "under-predicting"
                alerts.append(
                    f"Calibration bias (7d, {bucket}%): {direction} by {abs(data['gap']):.1%} "
                    f"(predicted {data['predicted']:.1%}, actual {data['actual']:.1%}, n={data['n']})"
                )

    # ── Check 5: Hit rate shift ──
    for label, recent in [("7d", recent_7d), ("3d", recent_3d)]:
        if not recent:
            continue
        hr_delta = abs(recent["hit_rate"] - lifetime["hit_rate"])
        if hr_delta > 0.08:
            alerts.append(
                f"Hit rate shift ({label}): {recent['hit_rate']:.1%} vs lifetime {lifetime['hit_rate']:.1%} "
                f"(delta {hr_delta:.1%})"
            )

    # ── Check 6: Feature coverage ──
    coverage = check_feature_coverage(days_back=3)
    if coverage:
        for feature, pct in coverage.items():
            if feature == "total":
                continue
            # Arsenal and statcast should be above 30% when working
            if feature in ("arsenal", "exit_velo") and pct < 0.10:
                alerts.append(
                    f"Feature offline ({feature}): {pct:.0%} coverage in last 3 days "
                    f"(expected >30%)"
                )
            # H2H coverage depends on matchups, 20% is reasonable
            elif feature == "h2h" and pct < 0.05:
                alerts.append(
                    f"Feature low ({feature}): {pct:.0%} coverage in last 3 days"
                )

    # Determine status
    if len(alerts) >= 3:
        status = "degraded"
    elif len(alerts) >= 1:
        status = "warning"
    else:
        status = "healthy"

    return {
        "status": status,
        "alerts": alerts,
        "lifetime": lifetime,
        "recent_7d": recent_7d,
        "recent_3d": recent_3d,
        "feature_coverage": coverage,
    }


def check_feature_coverage(days_back: int = 3) -> dict[str, float]:
    """Check what percentage of recent predictions have each optional feature.

    Returns: {"total": N, "h2h": 0.35, "arsenal": 0.42, "exit_velo": 0.38}
    """
    with contextlib.closing(_get_db()) as db:
        cutoff = (date.today() - timedelta(days=days_back)).isoformat()
        rows = db.execute(
            "SELECT factors_json FROM predictions WHERE game_date >= ? AND factors_json IS NOT NULL",
            (cutoff,),
        ).fetchall()

    if not rows:
        return {}

    n = len(rows)
    has_h2h = 0
    has_arsenal = 0
    has_ev = 0

    for r in rows:
        try:
            f = json.loads(r["factors_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        if f.get("h2h_avg") is not None:
            has_h2h += 1
        if f.get("arsenal_has_data"):
            has_arsenal += 1
        if f.get("exit_velo") is not None:
            has_ev += 1

    return {
        "total": n,
        "h2h": round(has_h2h / n, 3),
        "arsenal": round(has_arsenal / n, 3),
        "exit_velo": round(has_ev / n, 3),
    }


def format_drift_report(result: dict[str, Any]) -> str:
    """Format drift check result as a Telegram message."""
    if result["status"] == "insufficient_data":
        return f"\U0001f4cf Drift Check: {result['message']}"

    status_emoji = {
        "healthy": "\u2705",
        "warning": "\u26a0\ufe0f",
        "degraded": "\U0001f6a8",
    }

    lines = [
        f"{status_emoji.get(result['status'], '?')} Model Health: {result['status'].upper()}\n",
    ]

    # Lifetime baseline
    lt = result["lifetime"]
    lines.append(f"Lifetime ({lt['n']} preds):")
    lines.append(f"  Brier: {lt['brier']:.4f} | LogLoss: {lt['log_loss']:.4f}")
    lines.append(f"  Hit rate: {lt['hit_rate']:.1%}")

    # Tier rates
    tier_line = []
    for tier in ["STRONG HIT", "LEAN HIT", "TOSS-UP", "FADE"]:
        rate = lt["tier_rates"].get(tier)
        if rate is not None:
            short = tier.split()[0] if tier != "TOSS-UP" else "TOSS"
            tier_line.append(f"{short}: {rate:.0%}")
    if tier_line:
        lines.append(f"  Tiers: {' | '.join(tier_line)}")

    # Rolling windows
    for label, key in [("7-day", "recent_7d"), ("3-day", "recent_3d")]:
        recent = result.get(key, {})
        if not recent:
            continue
        lines.append(f"\n{label} ({recent['n']} preds):")
        brier_delta = recent["brier"] - lt["brier"]
        ll_delta = recent["log_loss"] - lt["log_loss"]
        b_arrow = "\u2b06\ufe0f" if brier_delta > 0.01 else "\u2b07\ufe0f" if brier_delta < -0.01 else "\u27a1\ufe0f"
        lines.append(f"  Brier: {recent['brier']:.4f} ({brier_delta:+.4f}) {b_arrow}")
        lines.append(f"  LogLoss: {recent['log_loss']:.4f} ({ll_delta:+.4f})")
        lines.append(f"  Hit rate: {recent['hit_rate']:.1%}")

    # Feature coverage
    cov = result.get("feature_coverage", {})
    if cov and cov.get("total", 0) > 0:
        lines.append(f"\nFeature coverage (last 3d, {cov['total']} preds):")
        for feat in ["h2h", "arsenal", "exit_velo"]:
            pct = cov.get(feat, 0)
            icon = "\u2705" if pct >= 0.20 else "\u26a0\ufe0f" if pct >= 0.05 else "\u274c"
            lines.append(f"  {icon} {feat}: {pct:.0%}")

    # Alerts
    if result["alerts"]:
        lines.append(f"\n\u26a0\ufe0f Alerts ({len(result['alerts'])}):")
        for alert in result["alerts"]:
            lines.append(f"  \u2022 {alert}")
    else:
        lines.append("\nNo drift detected.")

    return "\n".join(lines)
