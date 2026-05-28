"""Full backtest script — run on the server against predictions.db"""
import sqlite3
import sys

DB = sys.argv[1] if len(sys.argv) > 1 else "predictions.db"

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row

rows = db.execute("""
    SELECT hit_probability, prediction, got_hit, confidence, edge, game_date
    FROM predictions
    WHERE actual_result IS NOT NULL AND actual_result != 'DNP'
""").fetchall()

total = len(rows)
print(f"Total resolved predictions: {total}\n")

correct = sum(1 for r in rows if (r["prediction"] == "HIT" and r["got_hit"] == 1) or (r["prediction"] == "NO HIT" and r["got_hit"] == 0))
hit_preds = [r for r in rows if r["prediction"] == "HIT"]
hit_correct = sum(1 for r in hit_preds if r["got_hit"] == 1)
nohit_preds = [r for r in rows if r["prediction"] == "NO HIT"]
nohit_correct = sum(1 for r in nohit_preds if r["got_hit"] == 0)
base_hits = sum(1 for r in rows if r["got_hit"] == 1)

print("=== OVERALL ===")
print(f"Accuracy: {correct}/{total} ({round(correct/total*100,1)}%)")
print(f"HIT preds: {hit_correct}/{len(hit_preds)} ({round(hit_correct/len(hit_preds)*100,1)}%)")
print(f"NO HIT preds: {nohit_correct}/{len(nohit_preds)} ({round(nohit_correct/len(nohit_preds)*100,1)}%)")
print(f"Base rate (actual hits): {base_hits}/{total} ({round(base_hits/total*100,1)}%)")

print("\n=== BY CONFIDENCE ===")
for conf in ["high", "medium", "low", "insufficient"]:
    cr = [r for r in rows if r["confidence"] == conf]
    if cr:
        cc = sum(1 for r in cr if (r["prediction"] == "HIT" and r["got_hit"] == 1) or (r["prediction"] == "NO HIT" and r["got_hit"] == 0))
        print(f"  {conf}: {cc}/{len(cr)} ({round(cc/len(cr)*100,1)}%)")

print("\n=== DAILY BREAKDOWN ===")
dates = sorted(set(r["game_date"] for r in rows))
for d in dates:
    dr = [r for r in rows if r["game_date"] == d]
    dc = sum(1 for r in dr if (r["prediction"] == "HIT" and r["got_hit"] == 1) or (r["prediction"] == "NO HIT" and r["got_hit"] == 0))
    print(f"  {d}: {dc}/{len(dr)} ({round(dc/len(dr)*100,1)}%)")

print("\n=== PROBABILITY CALIBRATION (5% buckets) ===")
buckets = [(i/100, (i+5)/100) for i in range(25, 90, 5)]
for lo, hi in buckets:
    in_b = [r for r in rows if lo <= r["hit_probability"] < hi]
    if in_b:
        avg_p = sum(r["hit_probability"] for r in in_b) / len(in_b)
        actual = sum(r["got_hit"] for r in in_b) / len(in_b)
        print(f"  {lo:.0%}-{hi:.0%}: {len(in_b):4d} preds | predicted={avg_p:.3f} | actual={actual:.3f} | gap={actual-avg_p:+.3f}")

print("\n=== HIT PREDICTION WIN RATES ===")
for thresh in [0.60, 0.65, 0.70, 0.75]:
    hp = [r for r in rows if r["prediction"] == "HIT" and r["hit_probability"] >= thresh]
    if hp:
        hc = sum(1 for r in hp if r["got_hit"] == 1)
        per_day = round(len(hp) / len(dates), 1)
        print(f"  >= {thresh:.0%}: {hc}/{len(hp)} ({round(hc/len(hp)*100,1)}%) -- {per_day}/day")

print("\n=== MEDIUM/HIGH CONFIDENCE FILTERS ===")
for thresh in [0.60, 0.65, 0.70]:
    mh = [r for r in rows if r["confidence"] in ("medium", "high") and r["hit_probability"] >= thresh]
    if mh:
        mc = sum(1 for r in mh if r["got_hit"] == 1)
        per_day = round(len(mh) / len(dates), 1)
        print(f"  Med/High >= {thresh:.0%}: {mc}/{len(mh)} ({round(mc/len(mh)*100,1)}%) -- {per_day}/day")

print("\n=== TOP EDGES (N>=15) ===")
edge_stats = {}
for r in rows:
    e = r["edge"]
    if not e:
        continue
    if e not in edge_stats:
        edge_stats[e] = {"correct": 0, "total": 0}
    edge_stats[e]["total"] += 1
    if (r["prediction"] == "HIT" and r["got_hit"] == 1) or (r["prediction"] == "NO HIT" and r["got_hit"] == 0):
        edge_stats[e]["correct"] += 1

ranked = [(e, s) for e, s in edge_stats.items() if s["total"] >= 15]
ranked.sort(key=lambda x: x[1]["correct"] / x[1]["total"], reverse=True)
for e, s in ranked:
    print(f"  {s['correct']}/{s['total']} ({round(s['correct']/s['total']*100,1)}%) -- {e}")

print("\n=== SIMULATED BETTING P&L (flat $100) ===")
for thresh in [0.60, 0.65, 0.70]:
    hp = [r for r in rows if r["prediction"] == "HIT" and r["hit_probability"] >= thresh]
    wins = sum(1 for r in hp if r["got_hit"] == 1)
    losses = len(hp) - wins
    for odds in [-110, -150, -200, -250]:
        risk = abs(odds)
        profit = wins * 100 - losses * risk
        roi = profit / (len(hp) * risk) * 100 if hp else 0
        print(f"  >= {thresh:.0%} @ {odds}: {wins}W-{losses}L | P&L=${profit:+.0f} | ROI={roi:+.1f}%")
    print()

print("=== PAPER BETS ===")
pb = db.execute("""
    SELECT COUNT(*) as total,
        SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
        SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
        SUM(pnl) as pnl,
        SUM(stake) as staked
    FROM paper_bets WHERE settled = 1
""").fetchone()
if pb["total"] and pb["total"] > 0:
    print(f"  Record: {pb['wins']}W-{pb['losses']}L ({round(pb['wins']/pb['total']*100,1)}%)")
    print(f"  P&L: ${pb['pnl']:+.2f}")
    print(f"  ROI: {round(pb['pnl']/pb['staked']*100,1):+.1f}%")
    daily_pb = db.execute("""
        SELECT game_date, COUNT(*) as bets,
            SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(pnl) as pnl
        FROM paper_bets WHERE settled = 1
        GROUP BY game_date ORDER BY game_date
    """).fetchall()
    for d in daily_pb:
        dpnl = d["pnl"] or 0
        print(f"    {d['game_date']}: {d['wins']}W-{d['losses']}L  ${dpnl:+.2f}")

print("\n=== BOOK ODDS DATA ===")
bo = db.execute("SELECT COUNT(*) as c, COUNT(DISTINCT game_date) as days, COUNT(DISTINCT batter_name) as batters FROM book_odds").fetchone()
print(f"  Total odds entries: {bo['c']}")
print(f"  Days with odds: {bo['days']}")
print(f"  Unique batters: {bo['batters']}")

print("\n=== MODEL EDGE vs BOOK ODDS (0.5 line) ===")
edge_rows = db.execute("""
    SELECT bo.model_prob, bo.implied_prob, bo.edge, bo.over_price, bo.batter_name, bo.game_date,
        (SELECT p.got_hit FROM predictions p WHERE p.game_pk = bo.game_pk AND p.batter_name = bo.batter_name AND p.got_hit IS NOT NULL LIMIT 1) as got_hit
    FROM book_odds bo
    WHERE bo.line = 0.5 AND bo.model_prob > 0
""").fetchall()
with_results = [r for r in edge_rows if r["got_hit"] is not None]
print(f"  Odds with results: {len(with_results)}/{len(edge_rows)}")

for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]:
    bucket = [r for r in with_results if lo <= r["edge"] < hi]
    if bucket:
        hit_rate = sum(1 for r in bucket if r["got_hit"] == 1) / len(bucket)
        avg_edge = sum(r["edge"] for r in bucket) / len(bucket)
        avg_odds = sum(r["over_price"] for r in bucket) / len(bucket)
        print(f"  Edge {lo:.0%}-{hi:.0%}: {len(bucket)} bets | hit_rate={hit_rate:.1%} | avg_edge={avg_edge:.1%} | avg_book_odds={avg_odds:.0f}")

neg = [r for r in with_results if r["edge"] < 0]
if neg:
    neg_hr = sum(1 for r in neg if r["got_hit"] == 1) / len(neg)
    print(f"  Negative edge (<0%): {len(neg)} bets | hit_rate={neg_hr:.1%}")

db.close()
