#!/bin/bash
set -e
cd /home/calle/Trader

echo "=== Pull latest from GitHub ==="
git pull origin main

echo ""
echo "=== Install any new deps ==="
source .venv/bin/activate
pip install -q -r requirements.txt

echo ""
echo "=== Create logs directory ==="
mkdir -p logs

echo ""
echo "=== Restart web dashboard ==="
sudo -n systemctl restart trader-web 2>/dev/null || echo "Service not found, starting manually"
# Check if already running
if pgrep -f "python web.py" > /dev/null 2>&1; then
    pkill -f "python web.py"
    sleep 2
fi
nohup python web.py > /tmp/trader-web.log 2>&1 &
echo "Dashboard started on port 8080"

echo ""
echo "=== Test signal command ==="
python run.py signal

echo ""
echo "=== Verify cron job ==="
crontab -l 2>/dev/null | grep -i trade || echo "No cron job found - need to set up"

echo ""
echo "Done!"
