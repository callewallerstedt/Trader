#!/bin/bash
# Daily trading script - called by cron at 3:45 PM ET (before close)
cd "$(dirname "$0")"
source .venv/bin/activate

echo "========================================" >> trade.log
echo "$(date '+%Y-%m-%d %H:%M:%S') - Starting daily trade run" >> trade.log

# Skip weekends
DOW=$(date +%u)
if [ "$DOW" -ge 6 ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Weekend (day $DOW), skipping" >> trade.log
    echo "========================================" >> trade.log
    exit 0
fi

python run.py trade --live >> trade.log 2>&1
EXIT_CODE=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') - Finished (exit code: $EXIT_CODE)" >> trade.log
echo "========================================" >> trade.log
