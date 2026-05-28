"""
Hit prediction model for the baseball bot.

Full-game model: predicts P(at least 1 hit) across all AB.
Transparent scoring - every factor is visible and weighted.

Calibrated to reality: ~68% of MLB batters record at least 1 hit
per game. Top-of-order hitters are closer to 75-80%.

Factors:
  1. Batter AVG baseline (season, lightly regressed for small samples)
  2. Platoon split (batter's AVG vs pitcher's hand, 2025 full year)
  3. Pitcher difficulty (AVG-against, platoon-adjusted)
  4. Recent form (last 7 games)
  5. Park factor
  6. BABIP regression signal
  7. Head-to-head history (career batter vs this pitcher)
  8. Arsenal matchup (batter's AVG vs pitcher's specific pitch types)

Full game scaling:
  P(hit) = 1 - (1 - per_AB_prob) ^ total_AB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PredictionFactors:
    """Breakdown of every factor in the prediction."""
    batter_avg: float = 0.0
    platoon_avg: float = 0.0
    pitcher_avg_against: float = 0.0
    recent_avg: float = 0.0
    park_factor: float = 1.0
    babip: float = 0.0
    league_avg_babip: float = 0.300
    platoon_edge: bool = False
    same_hand: bool = False
    sample_size: int = 0
    platoon_pa: int = 0
    h2h_avg: float | None = None
    h2h_ab: int = 0
    h2h_hr: int = 0
    arsenal_avg: float | None = None
    arsenal_has_data: bool = False
    per_ab_prob: float = 0.0
    total_expected_ab: float = 4.0
    bullpen_avg_against: float = 0.248
    pitcher_recent_avg: float | None = None
    pitcher_recent_starts: int = 0
    exit_velo: float | None = None
    hard_hit_pct: float | None = None


def hit_tier(prob: float) -> str:
    """Map a hit probability to a display tier."""
    from config import TIER_STRONG, TIER_LEAN, TIER_TOSSUP
    if prob >= TIER_STRONG:
        return "STRONG HIT"
    elif prob >= TIER_LEAN:
        return "LEAN HIT"
    elif prob >= TIER_TOSSUP:
        return "TOSS-UP"
    else:
        return "FADE"


TIER_EMOJI = {
    "STRONG HIT": "\U0001f7e2",  # green
    "LEAN HIT": "\U0001f7e1",    # yellow
    "TOSS-UP": "\U0001f7e0",     # orange
    "FADE": "\U0001f534",        # red
}


@dataclass
class HitPrediction:
    """A single hit prediction."""
    batter_name: str
    batter_id: int
    pitcher_name: str
    pitcher_id: int
    venue: str
    game_pk: int
    game_date: str
    hit_probability: float
    confidence: str
    prediction: str  # kept for tracker/paper bet compatibility
    factors: PredictionFactors = field(default_factory=PredictionFactors)
    edge: str = ""

    @property
    def tier(self) -> str:
        return hit_tier(self.hit_probability)

    @property
    def tier_emoji(self) -> str:
        return TIER_EMOJI.get(self.tier, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "batter_name": self.batter_name,
            "batter_id": self.batter_id,
            "pitcher_name": self.pitcher_name,
            "pitcher_id": self.pitcher_id,
            "venue": self.venue,
            "game_pk": self.game_pk,
            "game_date": self.game_date,
            "hit_probability": round(self.hit_probability, 3),
            "prediction": self.prediction,
            "tier": self.tier,
            "confidence": self.confidence,
            "edge": self.edge,
        }


# League baselines
from config import LEAGUE_AVG, LEAGUE_BABIP
LEAGUE_GAME_HIT_RATE = 0.68  # 68% of batters get at least 1 hit per game


def predict_hit(
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
    """Predict whether a batter gets at least 1 hit in the full game."""
    factors = PredictionFactors()

    # ── 1. Batter AVG baseline ───────────────────────────────────
    season = batter_data.get("season", {})
    pa = int(season.get("pa", 0)) if season else 0
    factors.sample_size = pa

    if pa >= 50:
        batter_avg = _safe_float(season.get("avg"), LEAGUE_AVG)
    elif pa > 0:
        # Light regression — don't crush toward league avg too hard early in season
        raw = _safe_float(season.get("avg"), LEAGUE_AVG)
        regression_pa = 25  # lighter than before (was 50)
        batter_avg = (raw * pa + LEAGUE_AVG * regression_pa) / (pa + regression_pa)
    else:
        batter_avg = LEAGUE_AVG
    factors.batter_avg = batter_avg

    # ── 2. Platoon split ─────────────────────────────────────────
    platoon = batter_data.get("platoon", {})
    if pitcher_throws:
        key = "vs_L" if pitcher_throws == "L" else "vs_R"
        if key in platoon:
            factors.platoon_avg = _safe_float(platoon[key].get("avg"), batter_avg)
            factors.platoon_pa = int(platoon[key].get("pa", 0))
        else:
            factors.platoon_avg = batter_avg
        factors.platoon_edge = (
            (batter_bats == "L" and pitcher_throws == "R") or
            (batter_bats == "R" and pitcher_throws == "L")
        )
        factors.same_hand = batter_bats == pitcher_throws and batter_bats != "S"
    else:
        factors.platoon_avg = batter_avg

    # ── 3. Pitcher difficulty ────────────────────────────────────
    p_season = pitcher_data.get("season", {})
    pitcher_avg_against = _safe_float(p_season.get("avg"), LEAGUE_AVG)
    factors.pitcher_avg_against = pitcher_avg_against

    p_platoon = pitcher_data.get("platoon", {})
    pitcher_plat_avg = pitcher_avg_against
    if batter_bats and p_platoon:
        plat_key = "vs_LHB" if batter_bats == "L" else "vs_RHB"
        if batter_bats == "S":
            lhb = _safe_float(p_platoon.get("vs_LHB", {}).get("avg"), pitcher_avg_against)
            rhb = _safe_float(p_platoon.get("vs_RHB", {}).get("avg"), pitcher_avg_against)
            pitcher_plat_avg = max(lhb, rhb)
        elif plat_key in p_platoon:
            pitcher_plat_avg = _safe_float(p_platoon[plat_key].get("avg"), pitcher_avg_against)

    # ── 4. Recent form ───────────────────────────────────────────
    # Only trust recent form when there's enough season context.
    # In the first ~20 games, "hot" and "cold" are noise — a 7-game
    # window is half the season. Require 80+ PA (~20 games) before
    # weighting recent form.
    last7 = batter_data.get("last7", {})
    if last7 and int(last7.get("ab", 0)) >= 7 and pa >= 80:
        factors.recent_avg = _safe_float(last7.get("avg"), batter_avg)
    else:
        factors.recent_avg = batter_avg

    # ── 5. Park factor ───────────────────────────────────────────
    pf_runs = park_factors.get("runs", 1.0)
    factors.park_factor = pf_runs

    # ── 6. BABIP ─────────────────────────────────────────────────
    adv = batter_data.get("advanced", {})
    babip = _safe_float(adv.get("babip"), LEAGUE_BABIP)
    factors.babip = babip

    # ── 7. Head-to-head ──────────────────────────────────────────
    h2h = h2h_data or {}
    h2h_avg = None
    if h2h and h2h.get("ab", 0) >= 3:
        h2h_avg = _safe_float(h2h.get("avg"), batter_avg)
        factors.h2h_avg = h2h_avg
        factors.h2h_ab = h2h.get("ab", 0)
        factors.h2h_hr = h2h.get("hr", 0)
    else:
        factors.h2h_ab = h2h.get("ab", 0) if h2h else 0

    # ── 8. Arsenal matchup ───────────────────────────────────────
    ars = arsenal_matchup or {}
    arsenal_avg = None
    if ars.get("has_data") and ars.get("weighted_avg") is not None:
        arsenal_avg = ars["weighted_avg"]
        factors.arsenal_avg = arsenal_avg
        factors.arsenal_has_data = True

    # ── 9. Pitcher recent form (last 3 starts) ─────────────────
    p_recent = pitcher_recent or {}
    pitcher_recent_avg = None
    if p_recent.get("has_data") and p_recent.get("starts", 0) >= 2:
        pitcher_recent_avg = p_recent["avg_against"]
        factors.pitcher_recent_avg = pitcher_recent_avg
        factors.pitcher_recent_starts = p_recent["starts"]

    # ── 10. Statcast quality of contact ─────────────────────────
    statcast = batter_statcast or {}
    statcast_adj = 0.0
    if statcast.get("has_data") and statcast.get("batted_balls", 0) >= 20:
        ev = statcast["avg_exit_velo"]
        hh = statcast["hard_hit_pct"]
        factors.exit_velo = ev
        factors.hard_hit_pct = hh

        # Adjust per-AB prob based on quality of contact.
        # League avg exit velo ~88.5, hard hit ~35%.
        # A batter with 92 mph / 45% hard hit is underperforming his AVG.
        ev_diff = (ev - 88.5) / 100   # e.g., +0.035 for 92 mph
        hh_diff = (hh - 0.35) * 0.10  # e.g., +0.01 for 45% hard hit
        statcast_adj = ev_diff + hh_diff  # Will be added to per_ab later

    # ── COMBINE INTO PER-AB PROBABILITY ──────────────────────────
    #
    # The key insight: we want the per-AB probability, then scale
    # to full game. The batter/platoon/pitcher factors are all
    # AVG-scale numbers (.200-.350). We blend them, then the
    # full-game formula does the heavy lifting.

    # Dynamic weights — pitcher weight is split between season and recent
    w = {"batter": 0.30, "platoon": 0.15, "pitcher": 0.20, "recent": 0.10, "park": 0.05}

    # Pitcher recent form: blend season and last 3 starts
    if pitcher_recent_avg is not None:
        # Split pitcher weight: 55% season, 45% recent form
        w["pitcher_recent"] = w["pitcher"] * 0.45
        w["pitcher"] *= 0.55

    if h2h_avg is not None and factors.h2h_ab >= 10:
        w["h2h"] = 0.12
        w["batter"] -= 0.04
        w["platoon"] -= 0.04
        w["pitcher"] -= 0.04
    elif h2h_avg is not None and factors.h2h_ab >= 3:
        w["h2h"] = 0.07
        w["batter"] -= 0.03
        w["pitcher"] -= 0.04

    if arsenal_avg is not None:
        w["arsenal"] = 0.12
        w["batter"] -= 0.04
        w["platoon"] -= 0.04
        w["pitcher"] -= 0.04

    # Weighted per-AB AVG
    per_ab = (
        w["batter"] * batter_avg
        + w["platoon"] * factors.platoon_avg
        + w["pitcher"] * pitcher_plat_avg
        + w["recent"] * factors.recent_avg
        + w["park"] * (LEAGUE_AVG * pf_runs)
    )
    if "pitcher_recent" in w and pitcher_recent_avg is not None:
        per_ab += w["pitcher_recent"] * pitcher_recent_avg
    if "h2h" in w and h2h_avg is not None:
        per_ab += w["h2h"] * h2h_avg
    if "arsenal" in w and arsenal_avg is not None:
        per_ab += w["arsenal"] * arsenal_avg

    # Normalize weights (they should sum to ~1.0 but might not due to flex)
    total_w = sum(w.values())
    if total_w > 0 and abs(total_w - 1.0) > 0.01:
        per_ab = per_ab / total_w

    # BABIP regression — only apply with enough data
    if pa >= 80:
        per_ab -= (babip - LEAGUE_BABIP) * 0.08  # Light touch

    # Statcast quality-of-contact adjustment
    # Nudges per-AB prob based on exit velo and hard hit rate.
    # A batter with elite contact quality but low AVG is due for
    # regression up; soft contact with high AVG will regress down.
    if statcast_adj != 0.0:
        per_ab += statcast_adj

    # Weather adjustment (temperature + wind effects)
    if weather_adj != 0.0:
        per_ab += weather_adj

    # Platoon multiplier
    if factors.platoon_edge:
        per_ab *= 1.04
    elif factors.same_hand:
        per_ab *= 0.96

    # Clamp to realistic per-AB range
    per_ab = max(0.150, min(0.400, per_ab))
    factors.per_ab_prob = per_ab

    # ── SCALE TO FULL GAME (with bullpen split) ──────────────────
    #
    # Starters average ~5.2 IP (~2.3 times through the order).
    # Batters face the starter for their first 2-3 AB, then the
    # bullpen for the rest. We model this explicitly: per_ab is
    # the probability against the starter, and bullpen ABs use a
    # separate probability based on batter quality + league-average
    # reliever performance.
    #
    # AB splits by lineup spot (starter / bullpen).
    # Bullpen fraction tuned at 0.95 of initial estimate via sweep
    # across 1,496 new-model predictions — optimizes ROI (+14.1%)
    # while maintaining 83% STRONG HIT accuracy and +30% separation.
    #
    #   1-2: 3.075 / 1.425    3-4: 2.870 / 1.330    5-6: 2.375 / 1.425
    #   7-8: 2.170 / 1.330    9: 1.965 / 1.235

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

    total_ab = starter_ab + bullpen_ab
    factors.total_expected_ab = total_ab

    # Bullpen per-AB probability: batter's quality against a
    # league-average reliever. Uses batter AVG + park factor,
    # without pitcher-specific adjustments. Relievers are ~5%
    # harder to hit than starters on average.
    bullpen_per_ab = (batter_avg * 0.60 + bullpen_avg * 0.40)
    bullpen_per_ab *= pf_runs  # park factor
    if factors.platoon_edge:
        bullpen_per_ab *= 1.02  # slight platoon edge (mixed bullpen arms)
    bullpen_per_ab = max(0.150, min(0.350, bullpen_per_ab))
    factors.bullpen_avg_against = bullpen_avg

    # P(at least 1 hit) = 1 - P(no hit vs starter) * P(no hit vs bullpen)
    # Correlation discount: 75% of nominal — calibrated against
    # 3,400+ predictions to eliminate overconfidence bias above 65%.
    CORRELATION_DISCOUNT = 0.75
    eff_starter = starter_ab * CORRELATION_DISCOUNT
    eff_bullpen = bullpen_ab * CORRELATION_DISCOUNT

    p_no_hit = (1 - per_ab) ** eff_starter * (1 - bullpen_per_ab) ** eff_bullpen
    hit_prob = 1 - p_no_hit

    # Post-hoc calibration: squeeze toward the empirical center (0.60)
    # to compress the extremes where the model overestimates. Factor
    # of 0.75 was tuned on 27 days of live data (3,460 predictions).
    hit_prob = 0.60 + (hit_prob - 0.60) * 0.75
    hit_prob = max(0.25, min(0.90, hit_prob))

    # ── PREDICTION + CONFIDENCE ──────────────────────────────────
    #
    # Threshold: 46% — the actual base rate is ~61% of batters get
    # 1+ hits per game. A low threshold avoids bad NO HIT calls
    # (which were only 45% accurate at 52%). The model adds value
    # on HIT calls, not NO HIT calls.

    prediction = "HIT" if hit_prob >= 0.46 else "NO HIT"

    distance = abs(hit_prob - 0.52)
    data_quality = (
        (1 if pa >= 50 else 0)
        + (1 if factors.platoon_pa >= 50 else 0)
        + (1 if factors.h2h_ab >= 10 else 0)
        + (1 if factors.arsenal_has_data else 0)
    )
    if data_quality >= 3 and distance > 0.12:
        confidence = "high"
    elif data_quality >= 2 and distance > 0.06:
        confidence = "medium"
    elif pa > 0:
        confidence = "low"
    else:
        confidence = "insufficient"

    # ── EDGE TAGS ────────────────────────────────────────────────

    edges = []
    if factors.platoon_edge:
        edges.append("platoon+")
    if factors.same_hand:
        edges.append("same-hand-")
    if h2h_avg is not None:
        if h2h_avg > batter_avg + 0.050:
            edges.append(f"H2H-owns({factors.h2h_ab}AB)")
        elif h2h_avg < batter_avg - 0.050:
            edges.append(f"H2H-struggles({factors.h2h_ab}AB)")
        else:
            edges.append(f"H2H-neutral({factors.h2h_ab}AB)")
    if arsenal_avg is not None:
        if arsenal_avg > batter_avg + 0.030:
            edges.append("arsenal+")
        elif arsenal_avg < batter_avg - 0.030:
            edges.append("arsenal-")
    if factors.recent_avg > batter_avg + 0.040:
        edges.append("hot")
    elif factors.recent_avg < batter_avg - 0.040:
        edges.append("cold")
    if pitcher_plat_avg > LEAGUE_AVG + 0.025:
        edges.append("pitcher-vuln")
    elif pitcher_plat_avg < LEAGUE_AVG - 0.025:
        edges.append("pitcher-tough")
    if pf_runs >= 1.08:
        edges.append("hitter-park")
    elif pf_runs <= 0.92:
        edges.append("pitcher-park")
    if babip > 0.360 and pa >= 80:
        edges.append("BABIP-high")
    elif babip < 0.240 and pa >= 80:
        edges.append("BABIP-low")
    if pitcher_recent_avg is not None:
        if pitcher_recent_avg > LEAGUE_AVG + 0.030:
            edges.append("pitcher-struggling")
        elif pitcher_recent_avg < LEAGUE_AVG - 0.030:
            edges.append("pitcher-rolling")
    if factors.exit_velo is not None:
        if factors.exit_velo >= 91.0:
            edges.append("hard-contact")
        elif factors.exit_velo <= 86.0:
            edges.append("soft-contact")

    edge_str = " | ".join(edges) if edges else "neutral"

    return HitPrediction(
        batter_name=batter_name, batter_id=batter_id,
        pitcher_name=pitcher_name, pitcher_id=pitcher_id,
        venue=venue, game_pk=game_pk, game_date=game_date,
        hit_probability=hit_prob, confidence=confidence,
        prediction=prediction, factors=factors, edge=edge_str,
    )


def _safe_float(val, default: float) -> float:
    if val is None or val == "---" or val == "":
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default
