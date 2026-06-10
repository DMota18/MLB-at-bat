"""
CLI stats report for the baseball bot prediction database.

Usage:
    python stats.py                          # last 7 days
    python stats.py --start 2026-06-01 --end 2026-06-05
    python stats.py --sync                   # scp from EC2 first, then report
    python stats.py --sync --start 2026-06-01 --end 2026-06-05
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).parent / "predictions_latest.db"

EC2_HOST = os.environ.get("DEPLOY_HOST", "ubuntu@<your-ec2-ip>")
EC2_KEY = Path(os.environ.get("DEPLOY_KEY", str(Path.home() / ".ssh" / "your-key.pem")))
EC2_DB = "/home/ubuntu/baseball-bot/predictions.db"


def sync_db() -> None:
    cmd = ["scp", "-i", str(EC2_KEY), f"{EC2_HOST}:{EC2_DB}", str(DB_PATH)]
    print(f"Syncing from EC2... ", end="", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED\n{result.stderr.strip()}")
        sys.exit(1)
    size = DB_PATH.stat().st_size / 1024 / 1024
    print(f"OK ({size:.1f} MB)")


def report(start: str, end: str) -> None:
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row

    # --- Overall ---
    row = db.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN got_hit = 1 THEN 1 ELSE 0 END) as hits,
               SUM(CASE WHEN got_hit = 0 THEN 1 ELSE 0 END) as misses,
               SUM(CASE WHEN got_hit IS NULL THEN 1 ELSE 0 END) as pending,
               AVG(hit_probability) as avg_prob
        FROM predictions
        WHERE game_date BETWEEN ? AND ?
    """, (start, end)).fetchone()

    total = row["total"]
    hits = row["hits"] or 0
    misses = row["misses"] or 0
    pending = row["pending"] or 0
    settled = hits + misses
    avg_prob = row["avg_prob"] or 0

    print(f"\n{'='*55}")
    print(f"  STATS REPORT: {start} to {end}")
    print(f"{'='*55}")
    print(f"  Predictions: {total}  |  Settled: {settled}  |  Pending: {pending}")
    if settled > 0:
        print(f"  Record: {hits}W / {misses}L  |  Accuracy: {hits/settled*100:.1f}%")
    print(f"  Avg model prob: {avg_prob*100:.1f}%")

    # --- By Tier ---
    tiers = db.execute("""
        SELECT confidence,
               COUNT(*) as total,
               SUM(CASE WHEN got_hit = 1 THEN 1 ELSE 0 END) as hits,
               SUM(CASE WHEN got_hit = 0 THEN 1 ELSE 0 END) as misses,
               AVG(hit_probability) as avg_prob
        FROM predictions
        WHERE game_date BETWEEN ? AND ?
          AND got_hit IS NOT NULL
        GROUP BY confidence
        ORDER BY avg_prob DESC
    """, (start, end)).fetchall()

    if tiers:
        print(f"\n  {'TIER':<15} {'RECORD':>10} {'ACC':>7} {'AVG PROB':>9}")
        print(f"  {'-'*43}")
        for t in tiers:
            s = t["hits"] + t["misses"]
            acc = t["hits"] / s * 100 if s > 0 else 0
            rec = f"{t['hits']}W/{t['misses']}L"
            print(f"  {t['confidence']:<15} {rec:>10} {acc:>6.1f}% {t['avg_prob']*100:>8.1f}%")

    # --- By Day ---
    days = db.execute("""
        SELECT game_date,
               COUNT(*) as total,
               SUM(CASE WHEN got_hit = 1 THEN 1 ELSE 0 END) as hits,
               SUM(CASE WHEN got_hit = 0 THEN 1 ELSE 0 END) as misses,
               SUM(CASE WHEN got_hit IS NULL THEN 1 ELSE 0 END) as pending
        FROM predictions
        WHERE game_date BETWEEN ? AND ?
        GROUP BY game_date
        ORDER BY game_date
    """, (start, end)).fetchall()

    if days:
        print(f"\n  {'DATE':<12} {'PREDS':>5} {'RECORD':>10} {'PENDING':>8} {'ACC':>7}")
        print(f"  {'-'*45}")
        for d in days:
            h = d["hits"] or 0
            m = d["misses"] or 0
            p = d["pending"] or 0
            s = h + m
            acc = f"{h/s*100:.1f}%" if s > 0 else "N/A"
            rec = f"{h}W/{m}L"
            print(f"  {d['game_date']:<12} {d['total']:>5} {rec:>10} {p:>8} {acc:>7}")

    # --- Paper Bets ---
    try:
        bets = db.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses,
                   SUM(CASE WHEN settled = 0 THEN 1 ELSE 0 END) as pending,
                   SUM(pnl) as pnl,
                   SUM(stake) as staked
            FROM paper_bets
            WHERE game_date BETWEEN ? AND ?
        """, (start, end)).fetchone()

        bt = bets["total"]
        if bt > 0:
            wins = bets["wins"] or 0
            losses = bets["losses"] or 0
            pend = bets["pending"] or 0
            pnl = bets["pnl"] or 0.0
            staked = bets["staked"] or 0.0
            sb = wins + losses

            print(f"\n  PAPER BETS")
            print(f"  {'-'*43}")
            print(f"  Record: {wins}W / {losses}L  |  Pending: {pend}")
            print(f"  P&L: ${pnl:+,.2f}", end="")
            if sb > 0 and staked > 0:
                print(f"  |  ROI: {pnl/staked*100:+.1f}%")
            else:
                print()

            rows = db.execute("""
                SELECT game_date, batter_name, pitcher_name, book, book_odds,
                       model_prob, edge, result, pnl
                FROM paper_bets
                WHERE game_date BETWEEN ? AND ?
                ORDER BY game_date, batter_name
            """, (start, end)).fetchall()

            for r in rows:
                res = r["result"] or "PEND"
                p = f"${r['pnl']:+.0f}" if r["pnl"] else "$0"
                print(f"    {r['game_date']} {r['batter_name']:20s} vs {r['pitcher_name']:18s} "
                      f"{r['book']:10s} {r['book_odds']:+4d}  edge {r['edge']*100:.1f}%  {res} {p}")
        else:
            print(f"\n  PAPER BETS: none this period")
    except Exception:
        print(f"\n  PAPER BETS: table not available")

    # --- Top / Worst Batters ---
    for label, order in [("TOP BATTERS", "DESC"), ("WORST BATTERS", "ASC")]:
        batters = db.execute(f"""
            SELECT batter_name,
                   COUNT(*) as games,
                   SUM(got_hit) as hits,
                   SUM(CASE WHEN got_hit = 0 THEN 1 ELSE 0 END) as misses,
                   AVG(hit_probability) as avg_prob
            FROM predictions
            WHERE game_date BETWEEN ? AND ?
              AND got_hit IS NOT NULL
            GROUP BY batter_name
            HAVING games >= 3
            ORDER BY CAST(hits AS FLOAT) / games {order}
            LIMIT 8
        """, (start, end)).fetchall()

        if batters:
            print(f"\n  {label} (3+ games)")
            print(f"  {'-'*43}")
            for b in batters:
                acc = b["hits"] / b["games"] * 100
                rec = f"{b['hits']}W/{b['misses']}L"
                print(f"  {b['batter_name']:22s} {b['games']}gm  {rec:>7}  {acc:>5.0f}%  prob {b['avg_prob']*100:.1f}%")

    print()
    db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseball bot stats report")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--sync", action="store_true", help="Sync DB from EC2 first")
    parser.add_argument("--days", type=int, default=7, help="Days to look back (default 7, ignored if --start/--end)")
    args = parser.parse_args()

    if args.sync:
        sync_db()

    if not DB_PATH.exists():
        print(f"Database not found at {DB_PATH}")
        print("Run with --sync to download from EC2, or scp manually.")
        sys.exit(1)

    if args.start and args.end:
        start, end = args.start, args.end
    elif args.start:
        start = args.start
        end = date.today().isoformat()
    else:
        db = sqlite3.connect(str(DB_PATH))
        max_date = db.execute("SELECT MAX(game_date) FROM predictions").fetchone()[0]
        db.close()
        if max_date:
            end_dt = date.fromisoformat(max_date)
        else:
            end_dt = date.today()
        start = (end_dt - timedelta(days=args.days - 1)).isoformat()
        end = end_dt.isoformat()

    report(start, end)


if __name__ == "__main__":
    main()
