#!/bin/bash
# Daily trading script - called by cron at 3:45 PM ET (before close)
cd "$(dirname "$0")"
source .venv/bin/activate

echo "$(date) - Starting daily run" >> trade.log
python run.py trade --live >> trade.log 2>&1
echo "$(date) - Done" >> trade.log
echo "---" >> trade.log
