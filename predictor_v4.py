"""V4.0 shadow model: K%/BB%, xBA, isotonic calibration, reduced platoon.

Changes from primary (v3.1):
  1. K%/BB% as model inputs — high K% suppresses, high BB% boosts
  2. xBA blended with AVG — stabilizes faster, strips luck/defense
  3. Isotonic calibration — replaces hand-tuned squeeze with data-driven
     monotonic mapping fitted on 6,500+ historical predictions
  4. Reduced platoon multiplier — 1.02/0.98 (was 1.04/0.96), data shows
     the platoon effect is overweighted by ~4pp

Runs as a shadow model via ab_testing — never affects paper bets or alerts.
"""

from __future__ import annotations

from predictor import (
    predict_hit as _primary_predict,
    PredictionFactors,
    HitPrediction,
    _safe_float,
    LEAGUE_AVG,
    LEAGUE_BABIP,
)


# ── Isotonic calibration table ─────────────────────────────────────
#
# Fitted on 6,500+ predictions from predictions_latest.db.
# Maps raw model probability → calibrated probability.
# Built by binning predictions into 5pp buckets, computing actual
# hit rate in each, then ensuring monotonicity.
#
# Format: (threshold, calibrated_value)
# "If raw prob >= threshold, calibrated prob = value"
# Applied via linear interpolation between breakpoints.

ISOTONIC_TABLE = [
    (0.25, 0.35),   # floor — very low predictions still have ~35% actual
    (0.35, 0.42),
    (0.40, 0.45),
    (0.45, 0.47),
    (0.50, 0.53),   # 50-55% bucket: actual was 57%, pull up from squeeze
    (0.55, 0.58),   # 55-60%: well calibrated
    (0.60, 0.61),   # 60-65%: slight over-prediction
    (0.65, 0.63),   # 65-70%: model says 67%, actual 63%
    (0.70, 0.67),   # 70-75%: model says 72%, actual 67%
    (0.75, 0.68),   # 75-80%: model says 77%, actual 68%
    (0.80, 0.69),   # 80%+: ceiling, actual ~66-69%
    (0.90, 0.72),   # hard cap
]


def _isotonic_calibrate(raw_prob: float) -> float:
    """Apply isotonic calibration via linear interpolation."""
    if raw_prob <= ISOTONIC_TABLE[0][0]:
        return ISOTONIC_TABLE[0][1]
    if raw_prob >= ISOTONIC_TABLE[-1][0]:
        return ISOTONIC_TABLE[-1][1]

    for i in range(len(ISOTONIC_TABLE) - 1):
        lo_x, lo_y = ISOTONIC_TABLE[i]
        hi_x, hi_y = ISOTONIC_TABLE[i + 1]
        if lo_x <= raw_prob < hi_x:
            # Linear interpolation
            t = (raw_prob - lo_x) / (hi_x - lo_x)
            return lo_y + t * (hi_y - lo_y)

    return ISOTONIC_TABLE[-1][1]


def predict_hit_v4(
    batter_data: dict,
    pitcher_data: dict,
    batter_bats: str,
    pitcher_throws: str | None,
    venue: str,
    park_factors: dict,
    game_pk: int = 0,
    game_date: str = "",
    batter_name: str = "",
    batter_id: int = 0,
    pitcher_name: str = "",
    pitcher_id: int = 0,
    h2h_data: dict | None = None,
    arsenal_matchup: dict | None = None,
    batting_order: int = 5,
    bullpen_avg: float = 0.248,
    pitcher_recent: dict | None = None,
    batter_statcast: dict | None = None,
    weather_adj: float = 0.0,
) -> HitPrediction:
    """V4.0 prediction model with K%/BB%, xBA, isotonic cal, reduced platoon."""

    # Run primary model to get all computed factors and per-AB prob
    primary = _primary_predict(
        batter_data=batter_data, pitcher_data=pitcher_data,
        batter_bats=batter_bats, pitcher_throws=pitcher_throws,
        venue=venue, park_factors=park_factors,
        game_pk=game_pk, game_date=game_date,
        batter_name=batter_name, batter_id=batter_id,
        pitcher_name=pitcher_name, pitcher_id=pitcher_id,
        h2h_data=h2h_data, arsenal_matchup=arsenal_matchup,
        batting_order=batting_order, bullpen_avg=bullpen_avg,
        pitcher_recent=pitcher_recent, batter_statcast=batter_statcast,
        weather_adj=weather_adj,
    )

    f = primary.factors

    # ── Improvement 1: K%/BB% adjustment ───────────────────────────
    #
    # K% directly reduces hit opportunities (a K is never a hit).
    # BB% indirectly helps — more walks = better counts = better AB.
    # League average K% ~22%, BB% ~8.5%.

    k_bb_adj = 0.0
    adv = batter_data.get("advanced", {})
    k_pct = _safe_float(adv.get("k_pct"), -1)
    bb_pct = _safe_float(adv.get("bb_pct"), -1)

    if k_pct >= 0 and f.sample_size >= 50:
        # K% effect: each 1% above league avg reduces per-AB by ~0.002
        k_diff = k_pct - 0.22
        k_bb_adj -= k_diff * 0.20  # e.g., 30% K -> -0.016

    if bb_pct >= 0 and f.sample_size >= 50:
        # BB% effect: patient hitters see better pitches
        bb_diff = bb_pct - 0.085
        k_bb_adj += bb_diff * 0.10  # e.g., 12% BB -> +0.0035

    # ── Improvement 2: xBA blend ───────────────────────────────────
    #
    # Blend xBA with batting average. xBA strips luck and defense,
    # stabilizes faster. When available, use 40% xBA / 60% AVG.

    xba_adj = 0.0
    statcast = batter_statcast or {}
    xba = None
    if statcast.get("has_data"):
        xba = statcast.get("xba")

    if xba is not None and f.sample_size >= 50:
        # If xBA diverges from AVG, nudge toward xBA
        avg = f.batter_avg
        xba_diff = xba - avg
        xba_adj = xba_diff * 0.40  # 40% weight toward xBA

    # ── Re-compute per-AB with adjustments ─────────────────────────

    per_ab = f.per_ab_prob + k_bb_adj + xba_adj
    per_ab = max(0.150, min(0.400, per_ab))

    # ── Improvement 4: Reduced platoon multiplier ──────────────────
    #
    # Data shows platoon+ over-predicts by 4pp and same-hand- under-
    # predicts by 2.5pp. Reduce from 1.04/0.96 to 1.02/0.98.

    if f.platoon_edge:
        # Undo primary's 1.04, apply 1.02
        per_ab = per_ab / 1.04 * 1.02
    elif f.same_hand:
        # Undo primary's 0.96, apply 0.98
        per_ab = per_ab / 0.96 * 0.98

    per_ab = max(0.150, min(0.400, per_ab))

    # ── Full-game scaling (same structure as primary) ──────────────

    batter_avg = f.batter_avg
    pf_runs = f.park_factor
    bullpen_per_ab = (batter_avg * 0.60 + f.bullpen_avg_against * 0.40)
    bullpen_per_ab *= pf_runs
    if f.platoon_edge:
        bullpen_per_ab *= 1.01  # reduced from 1.02
    bullpen_per_ab = max(0.150, min(0.350, bullpen_per_ab))

    if batting_order <= 2:
        starter_ab, bullpen_ab = 3.075, 1.425
    elif batting_order <= 4:
        starter_ab, bullpen_ab = 2.870, 1.330
    elif batting_order <= 6:
        starter_ab, bullpen_ab = 2.375, 1.425
    elif batting_order <= 8:
        starter_ab, bullpen_ab = 2.170, 1.330
    else:
        starter_ab, bullpen_ab = 1.965, 1.235

    # Same correlation discount as primary
    CORRELATION_DISCOUNT = 0.75
    eff_starter = starter_ab * CORRELATION_DISCOUNT
    eff_bullpen = bullpen_ab * CORRELATION_DISCOUNT

    p_no_hit = (1 - per_ab) ** eff_starter * (1 - bullpen_per_ab) ** eff_bullpen
    raw_hit_prob = 1 - p_no_hit

    # ── Improvement 3: Isotonic calibration ────────────────────────
    #
    # Replace the hand-tuned squeeze with a data-driven monotonic
    # mapping fitted on 6,500+ historical predictions.
    hit_prob = _isotonic_calibrate(raw_hit_prob)
    hit_prob = max(0.25, min(0.90, hit_prob))

    # ── Prediction + confidence ────────────────────────────────────

    prediction = "HIT" if hit_prob >= 0.46 else "NO HIT"

    pa = f.sample_size
    distance = abs(hit_prob - 0.52)
    data_quality = (
        (1 if pa >= 50 else 0)
        + (1 if f.platoon_pa >= 50 else 0)
        + (1 if f.h2h_ab >= 10 else 0)
        + (1 if f.arsenal_has_data else 0)
    )
    if data_quality >= 3 and distance > 0.12:
        confidence = "high"
    elif data_quality >= 2 and distance > 0.06:
        confidence = "medium"
    elif pa > 0:
        confidence = "low"
    else:
        confidence = "insufficient"

    return HitPrediction(
        batter_name=batter_name, batter_id=batter_id,
        pitcher_name=pitcher_name, pitcher_id=pitcher_id,
        venue=venue, game_pk=game_pk, game_date=game_date,
        hit_probability=hit_prob, confidence=confidence,
        prediction=prediction, factors=f, edge=primary.edge,
    )
