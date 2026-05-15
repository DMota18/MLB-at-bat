# Baseball Bot

A real-time MLB hit prediction system that analyzes every batter-pitcher matchup, compares predictions against sportsbook odds, and tracks performance through automated paper betting.

Built as a Telegram bot that runs autonomously — polling for lineups, generating pregame analysis cards, fetching live odds, placing paper bets on +EV opportunities, and settling results overnight.

## Why I Built This

With all of the analytics and advanced statistics in modern baseball, I had no choice but to try and build something that predicts hits. I originally wanted to predict each individual at-bat, but the variance is too high — a .300 hitter fails 70% of the time. I settled on predicting whether a batter records at least one hit per game, which gave the model enough signal to work with.

To test it properly, I added a live odds comparison that checks sportsbook lines before each game and paper bets $100 whenever the model finds a 2%+ edge over the book. That turned it from a toy model into something with a real scoreboard — the P&L doesn't lie.

It's a project that tests my engineering skills, but more than that, it keeps me engaged watching games every night to see if the model got it right.

## Architecture

```
                        MLB Stats API
                            |
                    +-------+-------+
                    |               |
             Lineup Detector   Data Fetchers
          (lineup_detector.py) (data_fetchers.py)
           - Cached player bios  - Batter stats
           - 1 API call for all   - Pitcher stats
                    |               |
                    +-------+-------+
                            |
                     Matchup Engine
                    (matchup_data.py)
                     - H2H history
                     - Pitch arsenal
                     - Statcast data (1 call per batter)
                     - Pitcher recent form
                            |
                     Prediction Model
                      (predictor.py)
                     - 10 weighted factors
                     - Starter/bullpen split
                     - Calibrated probabilities
                     - Tier classification
                            |
              +-------------+-------------+
              |                           |
        Formatters                   Odds Engine
      (formatters.py)               (odds_api.py)
       - Pregame cards               - Async odds fetch
       - Tier display                 - Book comparison
              |                       - +EV detection
        Telegram Bot                      |
         (bot.py)                         |
       /games /best /odds                 |
       /game /stats /paper                |
              +-------------+-------------+
                            |
                      Paper Betting
                       (tracker.py)
                     - Auto bet placement
                     - Result settlement
                     - P&L tracking
                            |
                        SQLite DB
                     (predictions.db)
                            |
                   Shared Config
                    (config.py)
                  - Async HTTP client (pooled, retried, rate-limited)
                  - Season constants (auto-updating year)
                  - Tier thresholds (single source of truth)
                  - Odds conversion functions
```

## How It Works

### Prediction Model

Each batter-pitcher matchup is scored across **10 factors**, weighted and combined into a per-AB hit probability:

| Factor | Weight | Source |
|--------|--------|--------|
| Batter season AVG | 30% | MLB Stats API |
| Platoon split (vs L/R) | 15% | Previous full season |
| Pitcher AVG-against | 20% | MLB Stats API |
| Recent form (last 7G) | 10% | MLB Stats API |
| Park factor | 5% | Historical data |
| BABIP regression | adjustment | Season advanced stats |
| Head-to-head history | 7-12% | Career vs pitcher |
| Arsenal matchup | 12% | Statcast pitch-type AVG |
| Pitcher recent form | 9% | Last 3 starts |
| Statcast contact quality | adjustment | Exit velo, hard hit % |

The per-AB probability is scaled to a full-game probability with a **starter/bullpen split**:

```
P(1+ hit) = 1 - (1 - per_ab_starter)^eff_starter * (1 - per_ab_bullpen)^eff_bullpen
```

Starters average ~5.2 IP, so batters face the starter for 2-3 AB and the bullpen for the rest. The model accounts for this explicitly — bullpen ABs use a separate probability based on batter quality and league-average reliever performance. The split was tuned via sweep across 1,496 predictions to optimize ROI.

### Tier System

Instead of binary HIT/NO HIT predictions, the model classifies matchups into tiers:

| Tier | Probability | Historical Hit Rate |
|------|------------|-------------------|
| STRONG HIT | 70%+ | 83% |
| LEAN HIT | 62-70% | 62% |
| TOSS-UP | 55-62% | 58% |
| FADE | <55% | 53% |

The tier separation (30 points between STRONG HIT and FADE) has been stable across 5+ weeks of live data.

### Edge Detection

Every prediction is tagged with the factors driving it:

- `platoon+` / `same-hand-` — handedness advantage
- `H2H-owns(15AB)` — career dominance in the matchup
- `pitcher-vuln` / `pitcher-tough` — pitcher AVG-against vs league
- `pitcher-struggling` / `pitcher-rolling` — last 3 starts form
- `hot` / `cold` — batter's last 7 games vs season
- `hitter-park` / `pitcher-park` — venue HR/runs factor
- `BABIP-high` / `BABIP-low` — regression signal
- `hard-contact` / `soft-contact` — Statcast exit velocity

### Odds Comparison & Paper Betting

The bot fetches real bookmaker odds (DraftKings, FanDuel, BetMGM, etc.) via The Odds API and compares against model probabilities. When the model finds **+EV opportunities** (model prob >= 64%, edge > 2% over book implied), it:

1. Sends a Telegram alert with the pick, fair odds, and book odds
2. Auto-places a $100 flat-stake paper bet
3. Settles results overnight using box score data
4. Tracks cumulative P&L and ROI

## Sample Output

### Pregame Card
```
⚾ New York Yankees @ Boston Red Sox - 07:10 PM ET
🏟 Fenway Park  HR 0.95x  Runs 1.05x

🎯 Gerrit Cole (RHP)
  2.85 ERA  1.02 WHIP  .218 AVG-against
  98K/22BB in 82.0IP  K/9: 10.76  4HR
  FIP 3.01 | BABIP .285 | Whiff 28.5% | GB 42%

────────────────────────────────
🟢 1. Rafael Devers (L) — STRONG HIT (72% / -257) [medium]
   Edge: platoon+ | hot | hitter-park
   .298/.365/.521  12HR 28K 18BB  (185PA)
   BABIP .315 | ISO .223 | K% 15.1% | wOBA .378
   '25 vs_R: .312/.380/.548 (342PA)
   ✅ Platoon edge
   Last 7G: .345 (10/29) 2HR 3K

🟡 4. Masataka Yoshida (L) — LEAN HIT (64% / -178) [medium]
   Edge: platoon+ | pitcher-struggling
   .275/.340/.445  6HR 15K 12BB  (160PA)

🔴 9. Ceddanne Rafaela (R) — FADE (49% / +104) [low]
   Edge: same-hand- | cold

📊 18 matchups: 2 Strong | 6 Lean | 5 Toss-up | 5 Fade
```

### +EV Alert
```
💰 +EV Picks (model edge vs book)

🟢 Rafael Devers vs Gerrit Cole
   Model: 72% (fair -257) | DraftKings: -198 (impl 66%)
   Edge: +5.8% [medium] platoon+ | hot
```

### Daily Results
```
📊 Yesterday's Results (2026-05-13)

Overall: 63/117 correct (53.8%)

🟢 STRONG HIT: 4/5 got a hit (80.0%)
🟡 LEAN HIT: 15/24 got a hit (62.5%)
🟠 TOSS-UP: 23/51 got a hit (45.1%)
🔴 FADE: 24/42 got a hit (57.1%)
```

## Results

Based on **4,956 settled predictions** over 39 days (April 5 - May 14, 2026):

### Tier Accuracy

| Tier | Hit Rate | Count |
|------|----------|-------|
| STRONG HIT (70%+) | **67.0%** | 1,036 |
| LEAN HIT (62-70%) | **61.8%** | 1,498 |
| TOSS-UP (55-62%) | **57.9%** | 1,460 |
| FADE (<55%) | **53.1%** | 962 |

Tier separation (+14 points) has been stable since mid-April.

### Calibration (new model, May 2+)

| Predicted | Actual | Count | Gap |
|-----------|--------|-------|-----|
| 45-50% | 43.0% | 135 | -5.0% |
| 50-55% | 57.2% | 376 | +4.4% |
| 55-60% | 54.8% | 460 | -2.7% |
| 60-65% | 63.4% | 350 | +1.2% |
| 65-70% | 65.0% | 143 | -1.8% |

Well-calibrated in the 55-70% range where the majority of predictions fall.

### Paper Betting

| Period | Record | P&L | ROI |
|--------|--------|-----|-----|
| Old thresholds (Apr 22 - May 1) | 33W-18L (64.7%) | +$16.92 | +0.3% |
| **New thresholds (May 2+)** | **8W-1L (88.9%)** | **+$387.86** | **+43.1%** |
| **All-time** | **41W-19L (68.3%)** | **+$404.78** | **+6.7%** |

### Key Findings

- **Tier separation is real and stable** — STRONG HIT picks hit at 67-83%, FADE at 53%, across 5 weeks
- **Short-term pitcher form** (`pitcher-rolling`, `pitcher-struggling`) is the strongest betting signal — books are slow to adjust
- **`pitcher-vuln`** (high AVG-against) is a trap — books price this efficiently
- **The bullpen split matters** — modeling starter vs reliever ABs separately improved calibration and prevented overconfidence at the top end
- **Cold stretches are league-wide** — when the overall hit rate drops from 61% to 53%, the model tracks the league; individual predictions aren't wrong, the environment shifted

### Edge Tag Performance (paper bets)

| Tag | Record | P&L |
|-----|--------|-----|
| pitcher-struggling | 4W-0L | +$238 |
| pitcher-rolling | 4W-1L | +$142 |
| BABIP-low | 3W-0L | +$163 |
| pitcher-tough | 11W-3L | +$303 |
| hitter-park | 8W-2L | +$198 |
| pitcher-vuln | 11W-8L | -$173 |

## Commands

| Command | Description |
|---------|-------------|
| `/games` | Today's schedule with lineup status |
| `/game <# or team>` | Full pregame analysis for one game |
| `/analyze` | Analyze all games with posted lineups |
| `/best` | Today's best matchups ranked by tier |
| `/odds` | +EV picks with book odds comparison |
| `/paper` | Paper betting P&L tracker |
| `/results` | Yesterday's results by tier |
| `/stats` | Lifetime accuracy by tier and confidence |
| `/recent` | Recent predictions with outcomes |
| `/player <name>` | Player stat lookup |
| `/park` | Park factor reference |

## Setup

### Requirements

- Python 3.11+
- Telegram bot token ([BotFather](https://t.me/botfather))
- The Odds API key ([the-odds-api.com](https://the-odds-api.com)) — optional, for odds features

### Installation

```bash
git clone https://github.com/DMota18/baseball-bot.git
cd baseball-bot
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your tokens:
#   TELEGRAM_BOT_TOKEN=<your-bot-token>
#   TELEGRAM_CHAT_ID=<your-chat-id>
#   ODDS_API_KEY=<optional-odds-key>
```

### Running

```bash
python bot.py
```

The bot will:
- Poll for lineups every 30 minutes (10am-8pm ET)
- Auto-analyze games when lineups are posted
- Fetch odds ~90 min before first pitch
- Check yesterday's results at 8am ET
- Settle paper bets automatically

### Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

92 tests covering the prediction model, odds parsing, and full persistence layer.

### Deployment

A systemd service file is included for Linux servers:

```bash
# Set deploy env vars
export DEPLOY_HOST="user@your-server-ip"
export DEPLOY_KEY="$HOME/.ssh/your-key.pem"

# Deploy
bash deploy.sh
```

## Data Sources

- **MLB Stats API** — schedules, lineups, player stats, box scores
- **Baseball Savant** — Statcast pitch-level data (exit velo, pitch types)
- **The Odds API** — real-time sportsbook odds for player props

## Tech Stack

- **Python 3.11** with async/await
- **python-telegram-bot** — Telegram integration
- **httpx** — async HTTP client with connection pooling, retries, and rate limiting
- **APScheduler** — cron-style job scheduling
- **SQLite** — prediction and betting database with context-managed connections
- **pytest** — 92 unit tests

## License

MIT
