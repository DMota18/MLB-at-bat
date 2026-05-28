"""Export baseball bot statistics to CSV files for Tableau."""

import sqlite3
import csv
import json
import os
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "predictions_remote.db"
OUT_DIR = "tableau_export"

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row
os.makedirs(OUT_DIR, exist_ok=True)

# Detect available columns
pred_cols = {r[1] for r in db.execute("PRAGMA table_info(predictions)").fetchall()}
has_model_version = "model_version" in pred_cols
has_factors_json = "factors_json" in pred_cols

# ── 1. PREDICTIONS (main fact table) ──────────────────────────────
print("Exporting predictions...")
mv_col = ", p.model_version" if has_model_version else ", NULL as model_version"
fj_col = ", p.factors_json" if has_factors_json else ", NULL as factors_json"
rows = db.execute(f"""
    SELECT
        p.game_pk, p.game_date, p.batter_name, p.batter_id,
        p.pitcher_name, p.pitcher_id, p.venue, p.prediction,
        p.hit_probability, p.confidence, p.edge,
        p.actual_result, p.got_hit, p.at_bats, p.hits
        {mv_col}
        {fj_col},
        CASE
            WHEN p.hit_probability >= 0.70 THEN 'STRONG HIT'
            WHEN p.hit_probability >= 0.62 THEN 'LEAN HIT'
            WHEN p.hit_probability >= 0.55 THEN 'TOSS-UP'
            ELSE 'FADE'
        END as tier,
        CASE
            WHEN p.prediction = 'HIT' AND p.got_hit = 1 THEN 1
            WHEN p.prediction = 'NO HIT' AND p.got_hit = 0 THEN 1
            WHEN p.got_hit IS NOT NULL THEN 0
            ELSE NULL
        END as prediction_correct
    FROM predictions p
    ORDER BY p.game_date, p.game_pk, p.batter_name
""").fetchall()

with open(f"{OUT_DIR}/predictions.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    header = [
        "game_pk", "game_date", "batter_name", "batter_id",
        "pitcher_name", "pitcher_id", "venue", "prediction",
        "hit_probability", "confidence", "edge",
        "actual_result", "got_hit", "at_bats", "hits",
        "model_version", "tier", "prediction_correct",
    ]
    factor_cols = [
        "batter_avg", "platoon_avg", "pitcher_avg_against", "recent_avg",
        "park_factor", "babip", "platoon_edge", "same_hand",
        "sample_size", "platoon_pa", "h2h_avg", "h2h_ab",
        "arsenal_avg", "arsenal_has_data", "per_ab_prob",
        "total_expected_ab", "bullpen_avg_against",
        "exit_velo", "hard_hit_pct",
    ]
    if has_factors_json:
        header.extend(factor_cols)
    w.writerow(header)
    for r in rows:
        base = [
            r["game_pk"], r["game_date"], r["batter_name"], r["batter_id"],
            r["pitcher_name"], r["pitcher_id"], r["venue"], r["prediction"],
            r["hit_probability"], r["confidence"], r["edge"],
            r["actual_result"], r["got_hit"], r["at_bats"], r["hits"],
            r["model_version"], r["tier"], r["prediction_correct"],
        ]
        if has_factors_json:
            factors = {}
            if r["factors_json"]:
                try:
                    factors = json.loads(r["factors_json"])
                except Exception:
                    pass
            base.extend([factors.get(c, "") for c in factor_cols])
        w.writerow(base)
print(f"  {len(rows)} rows")

# ── 2. DAILY SUMMARY ─────────────────────────────────────────────
print("Exporting daily summary...")
days = db.execute("""
    SELECT
        game_date,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN got_hit IS NOT NULL THEN 1 ELSE 0 END) as settled,
        SUM(got_hit) as actual_hits,
        SUM(CASE WHEN (prediction = 'HIT' AND got_hit = 1) OR (prediction = 'NO HIT' AND got_hit = 0) THEN 1 ELSE 0 END) as correct,
        ROUND(AVG(hit_probability), 4) as avg_model_prob,
        ROUND(CAST(SUM(got_hit) AS FLOAT) / NULLIF(SUM(CASE WHEN got_hit IS NOT NULL THEN 1 ELSE 0 END), 0), 4) as actual_hit_rate,
        SUM(CASE WHEN hit_probability >= 0.70 THEN 1 ELSE 0 END) as strong_count,
        SUM(CASE WHEN hit_probability >= 0.70 AND got_hit = 1 THEN 1 ELSE 0 END) as strong_hits,
        SUM(CASE WHEN hit_probability >= 0.62 AND hit_probability < 0.70 THEN 1 ELSE 0 END) as lean_count,
        SUM(CASE WHEN hit_probability >= 0.62 AND hit_probability < 0.70 AND got_hit = 1 THEN 1 ELSE 0 END) as lean_hits,
        SUM(CASE WHEN hit_probability >= 0.55 AND hit_probability < 0.62 THEN 1 ELSE 0 END) as tossup_count,
        SUM(CASE WHEN hit_probability >= 0.55 AND hit_probability < 0.62 AND got_hit = 1 THEN 1 ELSE 0 END) as tossup_hits,
        SUM(CASE WHEN hit_probability < 0.55 THEN 1 ELSE 0 END) as fade_count,
        SUM(CASE WHEN hit_probability < 0.55 AND got_hit = 1 THEN 1 ELSE 0 END) as fade_hits,
        COUNT(DISTINCT game_pk) as games
    FROM predictions
    WHERE actual_result IS NOT NULL AND actual_result != 'DNP'
    GROUP BY game_date
    ORDER BY game_date
""").fetchall()

with open(f"{OUT_DIR}/daily_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "game_date", "total_predictions", "settled", "actual_hits", "correct",
        "accuracy", "avg_model_prob", "actual_hit_rate",
        "strong_count", "strong_hits", "strong_hit_rate",
        "lean_count", "lean_hits", "lean_hit_rate",
        "tossup_count", "tossup_hits", "tossup_hit_rate",
        "fade_count", "fade_hits", "fade_hit_rate",
        "games",
    ])
    for d in days:
        s = d["settled"] or 1
        w.writerow([
            d["game_date"], d["total_predictions"], d["settled"],
            d["actual_hits"], d["correct"],
            round((d["correct"] or 0) / s, 4),
            d["avg_model_prob"], d["actual_hit_rate"],
            d["strong_count"], d["strong_hits"],
            round((d["strong_hits"] or 0) / max(d["strong_count"] or 1, 1), 4),
            d["lean_count"], d["lean_hits"],
            round((d["lean_hits"] or 0) / max(d["lean_count"] or 1, 1), 4),
            d["tossup_count"], d["tossup_hits"],
            round((d["tossup_hits"] or 0) / max(d["tossup_count"] or 1, 1), 4),
            d["fade_count"], d["fade_hits"],
            round((d["fade_hits"] or 0) / max(d["fade_count"] or 1, 1), 4),
            d["games"],
        ])
print(f"  {len(days)} days")

# ── 3. BOOK ODDS ─────────────────────────────────────────────────
print("Exporting book odds...")
odds = db.execute("""
    SELECT
        bo.game_pk, bo.game_date, bo.batter_name, bo.batter_id,
        bo.book, bo.line, bo.over_price, bo.under_price,
        bo.implied_prob, bo.model_prob, bo.edge,
        bo.closing_over_price, bo.closing_implied_prob, bo.clv,
        bo.fetched_at,
        (SELECT p.got_hit FROM predictions p
         WHERE p.game_pk = bo.game_pk AND p.batter_name = bo.batter_name
         AND p.got_hit IS NOT NULL LIMIT 1) as got_hit
    FROM book_odds bo
    ORDER BY bo.game_date, bo.game_pk, bo.edge DESC
""").fetchall()

with open(f"{OUT_DIR}/book_odds.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "game_pk", "game_date", "batter_name", "batter_id",
        "book", "line", "over_price", "under_price",
        "implied_prob", "model_prob", "edge",
        "closing_over_price", "closing_implied_prob", "clv",
        "fetched_at", "got_hit",
    ])
    for r in odds:
        w.writerow([
            r["game_pk"], r["game_date"], r["batter_name"], r["batter_id"],
            r["book"], r["line"], r["over_price"], r["under_price"],
            r["implied_prob"], r["model_prob"], r["edge"],
            r["closing_over_price"], r["closing_implied_prob"], r["clv"],
            r["fetched_at"], r["got_hit"],
        ])
print(f"  {len(odds)} rows")

# ── 4. PAPER BETS ────────────────────────────────────────────────
print("Exporting paper bets...")
bets = db.execute("""
    SELECT
        pb.*,
        (SELECT p.hit_probability FROM predictions p
         WHERE p.game_pk = pb.game_pk AND p.batter_name = pb.batter_name
         LIMIT 1) as model_hit_prob,
        (SELECT p.actual_result FROM predictions p
         WHERE p.game_pk = pb.game_pk AND p.batter_name = pb.batter_name
         LIMIT 1) as actual_result
    FROM paper_bets pb
    ORDER BY pb.game_date, pb.edge DESC
""").fetchall()

with open(f"{OUT_DIR}/paper_bets.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "id", "game_pk", "game_date", "batter_name", "batter_id",
        "pitcher_name", "model_prob", "confidence", "edge_tags",
        "book", "book_odds", "implied_prob", "edge", "stake",
        "result", "got_hit", "pnl", "settled",
        "model_hit_prob", "actual_result",
    ])
    for r in bets:
        w.writerow([
            r["id"], r["game_pk"], r["game_date"], r["batter_name"], r["batter_id"],
            r["pitcher_name"], r["model_prob"], r["confidence"], r["edge_tags"],
            r["book"], r["book_odds"], r["implied_prob"], r["edge"], r["stake"],
            r["result"], r["got_hit"], r["pnl"], r["settled"],
            r["model_hit_prob"], r["actual_result"],
        ])
print(f"  {len(bets)} rows")

# ── 5. CALIBRATION BUCKETS (by date) ─────────────────────────────
print("Exporting calibration data...")
cal = db.execute("""
    SELECT
        game_date,
        CAST(ROUND(hit_probability * 20, 0) * 5 AS INTEGER) as bucket_lo,
        COUNT(*) as n,
        AVG(hit_probability) as avg_predicted,
        AVG(got_hit) as avg_actual,
        AVG(got_hit) - AVG(hit_probability) as gap,
        SUM(got_hit) as hits
    FROM predictions
    WHERE actual_result IS NOT NULL AND actual_result != 'DNP' AND got_hit IS NOT NULL
    GROUP BY game_date, bucket_lo
    ORDER BY game_date, bucket_lo
""").fetchall()

with open(f"{OUT_DIR}/calibration.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "game_date", "bucket", "n", "avg_predicted", "avg_actual", "gap", "hits",
    ])
    for r in cal:
        lo = r["bucket_lo"]
        w.writerow([
            r["game_date"], f"{lo}-{lo + 5}%",
            r["n"], round(r["avg_predicted"], 4), round(r["avg_actual"], 4),
            round(r["gap"], 4), r["hits"],
        ])
print(f"  {len(cal)} rows")

# ── 6. EDGE TAG PERFORMANCE ──────────────────────────────────────
print("Exporting edge tag performance...")
rows = db.execute("""
    SELECT edge, got_hit, hit_probability, game_date
    FROM predictions
    WHERE actual_result IS NOT NULL AND actual_result != 'DNP' AND got_hit IS NOT NULL AND edge IS NOT NULL
""").fetchall()

tag_data = {}
for r in rows:
    tags = [t.strip() for t in (r["edge"] or "").split(" | ") if t.strip()]
    for tag in tags:
        if tag not in tag_data:
            tag_data[tag] = {"hits": 0, "total": 0, "prob_sum": 0.0}
        tag_data[tag]["total"] += 1
        tag_data[tag]["hits"] += r["got_hit"]
        tag_data[tag]["prob_sum"] += r["hit_probability"]

with open(f"{OUT_DIR}/edge_tags.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["edge_tag", "total", "hits", "hit_rate", "avg_model_prob", "calibration_gap"])
    for tag, d in sorted(tag_data.items(), key=lambda x: -x[1]["total"]):
        if d["total"] >= 5:
            hit_rate = d["hits"] / d["total"]
            avg_prob = d["prob_sum"] / d["total"]
            w.writerow([
                tag, d["total"], d["hits"],
                round(hit_rate, 4), round(avg_prob, 4),
                round(hit_rate - avg_prob, 4),
            ])
print(f"  {len([t for t in tag_data if tag_data[t]['total'] >= 5])} tags")

# ── 7. VENUE PERFORMANCE ─────────────────────────────────────────
print("Exporting venue performance...")
venues = db.execute("""
    SELECT
        venue,
        COUNT(*) as total,
        SUM(got_hit) as hits,
        ROUND(AVG(got_hit), 4) as hit_rate,
        ROUND(AVG(hit_probability), 4) as avg_model_prob,
        ROUND(AVG(got_hit) - AVG(hit_probability), 4) as calibration_gap,
        COUNT(DISTINCT game_pk) as games
    FROM predictions
    WHERE actual_result IS NOT NULL AND actual_result != 'DNP' AND got_hit IS NOT NULL
    GROUP BY venue
    ORDER BY total DESC
""").fetchall()

with open(f"{OUT_DIR}/venue_performance.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["venue", "total_predictions", "hits", "hit_rate", "avg_model_prob", "calibration_gap", "games"])
    for r in venues:
        w.writerow([r["venue"], r["total"], r["hits"], r["hit_rate"], r["avg_model_prob"], r["calibration_gap"], r["games"]])
print(f"  {len(venues)} venues")

db.close()

# Summary
print(f"\n{'=' * 50}")
print(f"Export complete! Files in {OUT_DIR}/:")
for fname in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(f"{OUT_DIR}/{fname}")
    print(f"  {fname:30s} {size:>10,} bytes")
print(f"\nOpen any CSV in Tableau via 'Connect > Text file'")
