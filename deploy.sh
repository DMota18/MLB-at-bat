#!/bin/bash
set -e

SERVER="${DEPLOY_HOST:?Set DEPLOY_HOST (e.g. ubuntu@1.2.3.4)}"
KEY="${DEPLOY_KEY:?Set DEPLOY_KEY (path to SSH key)}"
REMOTE_DIR="${DEPLOY_DIR:-/home/ubuntu/baseball-bot}"

echo "=== Deploying Baseball Bot ==="

echo "[1/4] Uploading files..."
scp -i "$KEY" \
    config.py bot.py predictor.py tracker.py matchup_data.py \
    data_fetchers.py formatters.py weather.py umpire.py \
    lineup_detector.py odds_api.py drift.py ab_testing.py predictor_shadow.py predictor_v4.py \
    auditor.py statcast_cache.py \
    requirements.txt baseball-bot.service .env.example \
    "$SERVER:$REMOTE_DIR/"

echo "[2/4] Installing dependencies..."
ssh -i "$KEY" "$SERVER" << 'REMOTE'
cd /home/ubuntu/baseball-bot
python3 -m venv venv 2>/dev/null || true
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Done."
REMOTE

echo "[3/4] Checking .env..."
ssh -i "$KEY" "$SERVER" << 'REMOTE'
cd /home/ubuntu/baseball-bot
if [ ! -f .env ]; then
    echo "  ERROR: .env not found. Create it manually on the server:"
    echo "    TELEGRAM_BOT_TOKEN=<your-token>"
    echo "    TELEGRAM_CHAT_ID=<your-chat-id>"
    echo "    KB_API_URL=http://127.0.0.1:8100"
    exit 1
else
    echo "  .env exists."
fi
REMOTE

echo "[4/4] Installing service..."
ssh -i "$KEY" "$SERVER" << 'REMOTE'
sudo cp /home/ubuntu/baseball-bot/baseball-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable baseball-bot
sudo systemctl restart baseball-bot
sleep 2
sudo systemctl status baseball-bot --no-pager | head -10
REMOTE

echo ""
echo "=== Deploy complete ==="
