# Baseball Bot — Claude Code Project Guide

## Quick Reference

```bash
# Run tests
pytest tests/ -v

# Run a single test file
pytest tests/test_predictor.py -v

# Lint
ruff check .

# Type check
mypy *.py --ignore-missing-imports

# Run the bot locally (requires .env)
python bot.py

# Deploy to production
DEPLOY_HOST=ubuntu@<ip> DEPLOY_KEY=~/.ssh/key.pem bash deploy.sh
```

## Architecture

MLB hit prediction bot (~10,500 lines Python). Runs 24/7 on AWS EC2, sends predictions via Telegram. Analyzes every MLB game daily, predicting whether each batter gets 1+ hits.

### Module Map

| Module | Responsibility | Key Patterns |
|---|---|---|
| `bot.py` | Telegram commands + APScheduler jobs | Entry point, all `/command` handlers |
| `config.py` | Constants, HTTP client, circuit breaker | Shared `get_client()`, `fetch_json()`/`fetch_text()` with retries |
| `data_fetchers.py` | MLB Stats API wrappers | Batter/pitcher season stats, platoon splits |
| `matchup_data.py` | Statcast, H2H, arsenal matchups | Per-session cache on Savant data, `compute_arsenal_matchup()` |
| `lineup_detector.py` | Game schedule polling + player bios | Polls MLB API every 30min for posted lineups |
| `predictor.py` | Primary hit prediction model (v3.1) | `predict_hit()` — 10 weighted factors, starter/bullpen split |
| `predictor_shadow.py` | Shadow model (lighter calibration) | A/B test variant with squeeze=0.85, discount=0.80 |
| `predictor_v4.py` | V4 shadow (K%/BB%, xBA, isotonic) | Not yet deployed to primary |
| `formatters.py` | Telegram message formatting | `build_pregame_card()`, `PARK_FACTORS` dict (all 30 parks) |
| `tracker.py` | SQLite persistence + paper bets | `save_predictions()`, `settle_paper_bets()`, `check_results()` |
| `odds_api.py` | Sportsbook odds fetching | Paid API ($1/call), `find_best_odds()` for +EV detection |
| `weather.py` | Open-Meteo weather at game venues | 30 park coordinates, 8 retractable roof venues skipped |
| `umpire.py` | Home plate umpire zone tendencies | Strike zone adjustments |
| `drift.py` | Model health monitoring | Daily Brier score drift, tier separation, calibration checks |
| `ab_testing.py` | Shadow model framework | `register_shadow_model()`, head-to-head comparison metrics |
| `auditor.py` | Full diagnostic audit (10 sections) | Pipeline health, coverage, calibration analysis |

### Data Flow

```
MLB Stats API → lineup_detector (poll) → data_fetchers + matchup_data (parallel per batter)
    → predictor.predict_hit() → tracker (save to SQLite) → formatters → Telegram
    → odds_api (pre-game) → tracker.place_paper_bets() → settle overnight
```

### External APIs (4)

1. **MLB Stats API** — Free, no auth. Schedules, lineups, stats, box scores.
2. **Baseball Savant** — Free, no auth. Statcast CSV data. Aggressive rate limiting on VPS IPs. Must use browser-like User-Agent and stagger requests (300ms). Short responses (< 200 bytes) = blocked, not 403.
3. **The Odds API** — Paid ($1/call), tracked via response headers. Player prop odds from FanDuel, DraftKings, BetMGM.
4. **Open-Meteo** — Free, no auth. Game-time temperature and wind.

## Key Conventions

- **All HTTP through `config.fetch_json()`/`fetch_text()`** — circuit breaker + semaphore(10) + exponential backoff built in. Never use httpx directly.
- **Async everywhere** — `asyncio.gather()` for parallel batter analysis, `asyncio.Semaphore(10)` for connection cap.
- **Per-session Statcast cache** — `matchup_data.get_batter_statcast_all()` caches on `(batter_id, season)`. Cleared per game card.
- **Tier thresholds in `config.py`** — `TIER_STRONG=0.70`, `TIER_LEAN=0.62`, `TIER_TOSSUP=0.55`. Single source of truth.
- **Model version in `config.py`** — Bump `MODEL_VERSION` when changing weights/calibration.
- **Shadow models** — Register via `ab_testing.register_shadow_model()`. Never promote without head-to-head comparison data.
- **Park factors in `formatters.py`** — All 30 MLB parks with hit factor multipliers.

## Gotchas

- **Savant API on VPS**: Gets rate-limited silently. Returns empty data instead of errors. The `statcast_adj` will be zero with no warning. Always check pipeline health via `/audit`.
- **SQLite concurrency**: Single writer. The bot is single-process so this is fine, but never run two instances against the same `.db` file.
- **Odds API budget**: Each call costs money. The bot fetches odds 60-100min before game time + again at game time for CLV.
- **.env required**: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are mandatory. `ODDS_API_KEY` optional (odds features disabled without it).
- **deploy.sh hardcodes file list**: When adding new modules, update the `scp` line in `deploy.sh`.
- **Retractable roofs**: 8 venues skip weather adjustments. List is in `weather.py`.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Yes | Telegram Bot API token |
| `TELEGRAM_CHAT_ID` | Yes | Target chat ID for predictions |
| `ODDS_API_KEY` | No | The Odds API key (odds features disabled without) |
| `DEPLOY_HOST` | Deploy only | SSH target (e.g., ubuntu@1.2.3.4) |
| `DEPLOY_KEY` | Deploy only | Path to SSH private key |

## Testing

118 tests across 5 files. All pure/deterministic — no network calls, no randomness.

```bash
pytest tests/ -v                    # all tests
pytest tests/test_predictor.py -v   # prediction model tests
pytest tests/test_tracker.py -v     # database tests (in-memory SQLite)
pytest tests/test_odds.py -v        # odds parsing + edge calculation
pytest tests/test_drift.py -v       # model health monitoring
pytest tests/test_ab_testing.py -v  # shadow model framework
```

## Model Performance (as of May 30, 2026)

- 6,750+ settled predictions (Apr 5 – May 26)
- STRONG HIT: 67.0% accuracy | LEAN HIT: 62% | TOSS-UP: 59% | FADE: 54%
- ECE: 0.029 | Brier: 0.240 | ROC-AUC: 0.555
- Paper betting: 46W/22L, +$408, +6% ROI
