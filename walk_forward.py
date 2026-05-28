"""Walk-forward validation: train on days 1..N, test on day N+1, repeat.

Measures true out-of-sample performance by never leaking future data
into the training set. Uses logistic regression on edge-tag features.

Requires: pip install scikit-learn
"""
import math
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "predictions_remote.db"

db = sqlite3.connect(DB)
db.row_factory = sqlite3.Row

rows = db.execute("""
    SELECT p.*, bo.implied_prob as book_implied
    FROM predictions p
    LEFT JOIN (
        SELECT game_pk, batter_name, MIN(implied_prob) as implied_prob
        FROM book_odds WHERE line = 0.5
        GROUP BY game_pk, batter_name
    ) bo ON p.game_pk = bo.game_pk AND p.batter_name = bo.batter_name
    WHERE p.actual_result IS NOT NULL AND p.actual_result != 'DNP' AND p.got_hit IS NOT NULL
    ORDER BY p.game_date
""").fetchall()
db.close()


def extract_features(row):
    edge = row["edge"] or ""
    tags = set(t.strip() for t in edge.split(" | ") if t.strip())
    return [
        row["hit_probability"],
        1 if "platoon+" in tags else 0,
        1 if "same-hand-" in tags else 0,
        1 if "hot" in tags else 0,
        1 if "cold" in tags else 0,
        1 if "pitcher-vuln" in tags else 0,
        1 if "pitcher-tough" in tags else 0,
        1 if "pitcher-struggling" in tags else 0,
        1 if "pitcher-rolling" in tags else 0,
        1 if "hitter-park" in tags else 0,
        1 if "pitcher-park" in tags else 0,
        1 if "BABIP-high" in tags else 0,
        1 if "BABIP-low" in tags else 0,
        1 if any("H2H-owns" in t for t in tags) else 0,
        1 if any("H2H-struggles" in t for t in tags) else 0,
        1 if row["confidence"] == "medium" else 0,
        1 if row["confidence"] == "high" else 0,
        row["book_implied"] or 0.61,
    ]


def brier(preds, actuals):
    return sum((p - y) ** 2 for p, y in zip(preds, actuals)) / len(actuals)


def logloss(preds, actuals):
    eps = 1e-15
    return -sum(
        y * math.log(max(eps, p)) + (1 - y) * math.log(max(eps, 1 - p))
        for p, y in zip(preds, actuals)
    ) / len(actuals)


# Group by date
by_date = defaultdict(list)
for r in rows:
    by_date[r["game_date"]].append(r)

dates = sorted(by_date.keys())
print(f"Total dates: {len(dates)}")
print(f"Total predictions: {len(rows)}")
print(f"Date range: {dates[0]} to {dates[-1]}")

# Need sklearn
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("\n*** pip install scikit-learn ***")
    sys.exit(1)

# ── Walk forward ──
MIN_TRAIN_DAYS = 7  # need at least 7 days before we start testing

heuristic_preds = []
heuristic_actuals = []
lr_preds = []
lr_actuals = []
daily_results = []

print(f"\nWalk-forward: training on first {MIN_TRAIN_DAYS}+ days, testing one day at a time")
print("=" * 85)
print(f"{'Date':>12s} {'N':>5s} {'H_Brier':>8s} {'LR_Brier':>9s} {'H_Acc':>7s} {'LR_Acc':>7s} {'HitRate':>8s} {'Winner':>10s}")
print("-" * 85)

for i in range(MIN_TRAIN_DAYS, len(dates)):
    test_date = dates[i]
    train_dates = dates[:i]

    # Build train set
    X_train, y_train = [], []
    for d in train_dates:
        for r in by_date[d]:
            X_train.append(extract_features(r))
            y_train.append(r["got_hit"])

    # Build test set
    X_test, y_test, h_probs = [], [], []
    for r in by_date[test_date]:
        X_test.append(extract_features(r))
        y_test.append(r["got_hit"])
        h_probs.append(r["hit_probability"])

    if len(X_test) < 5:
        continue

    # Train LR
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, C=1.0)
    lr.fit(X_tr, y_train)
    lr_p = [p[1] for p in lr.predict_proba(X_te)]

    # Metrics
    h_brier = brier(h_probs, y_test)
    lr_brier = brier(lr_p, y_test)
    h_acc = sum(1 for p, y in zip(h_probs, y_test) if (p >= 0.46) == (y == 1)) / len(y_test)
    lr_acc = sum(1 for p, y in zip(lr_p, y_test) if (p >= 0.5) == (y == 1)) / len(y_test)
    hit_rate = sum(y_test) / len(y_test)
    winner = "LR" if lr_brier < h_brier else "Heuristic" if lr_brier > h_brier else "Tie"

    heuristic_preds.extend(h_probs)
    heuristic_actuals.extend(y_test)
    lr_preds.extend(lr_p)
    lr_actuals.extend(y_test)

    daily_results.append({
        "date": test_date, "n": len(y_test),
        "h_brier": h_brier, "lr_brier": lr_brier,
        "h_acc": h_acc, "lr_acc": lr_acc,
        "hit_rate": hit_rate, "winner": winner,
    })

    print(f"  {test_date}  {len(y_test):4d}  {h_brier:.4f}   {lr_brier:.4f}  {h_acc*100:5.1f}%  {lr_acc*100:5.1f}%  {hit_rate*100:5.1f}%  {winner:>10s}")

# ── Summary ──
print("\n" + "=" * 85)
print("WALK-FORWARD SUMMARY")
print("=" * 85)

h_wins = sum(1 for d in daily_results if d["winner"] == "Heuristic")
lr_wins = sum(1 for d in daily_results if d["winner"] == "LR")
ties = sum(1 for d in daily_results if d["winner"] == "Tie")
print(f"  Days tested: {len(daily_results)}")
print(f"  Heuristic wins: {h_wins} | LR wins: {lr_wins} | Ties: {ties}")

print(f"\n  Overall (pooled across all test days):")
print(f"    Heuristic Brier: {brier(heuristic_preds, heuristic_actuals):.4f}")
print(f"    LR Brier:        {brier(lr_preds, lr_actuals):.4f}")
print(f"    Heuristic Loss:  {logloss(heuristic_preds, heuristic_actuals):.4f}")
print(f"    LR Loss:         {logloss(lr_preds, lr_actuals):.4f}")
print(f"    Base rate:       {sum(lr_actuals)/len(lr_actuals)*100:.1f}%")

# Weekly rolling comparison
print(f"\n  Weekly rolling (Brier, lower = better):")
for i in range(0, len(daily_results), 7):
    week = daily_results[i:i+7]
    if not week:
        break
    h_b = sum(d["h_brier"] * d["n"] for d in week) / sum(d["n"] for d in week)
    lr_b = sum(d["lr_brier"] * d["n"] for d in week) / sum(d["n"] for d in week)
    label = f"{week[0]['date']} to {week[-1]['date']}"
    winner = "LR" if lr_b < h_b else "Heuristic"
    print(f"    {label}: H={h_b:.4f} LR={lr_b:.4f} -> {winner}")
