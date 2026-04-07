#!/bin/bash
pkill -f ibgateway 2>/dev/null || true
pkill -f 'Xvfb :99' 2>/dev/null || true
sleep 2
rm -f /tmp/.X99-lock /tmp/.X99-unix/X99

export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
sleep 1

mkdir -p /home/calle/Jts /home/calle/ibc/logs

export TWS_MAJOR_VRSN=1037
cd /home/calle/ibc

echo "" | bash ./gatewaystart.sh -inline > /home/calle/ibc/logs/gateway_latest.log 2>&1 &
GWPID=$!
echo "Gateway PID: $GWPID"

sleep 30

echo "=== Last 30 lines of gateway log ==="
tail -30 /home/calle/ibc/logs/gateway_latest.log

echo "=== Checking diagnostic log ==="
DIAG=$(ls -t /home/calle/ibc/logs/ibc-*.txt 2>/dev/null | head -1)
if [ -n "$DIAG" ]; then
    echo "File: $DIAG"
    tail -30 "$DIAG"
fi

echo "=== Port check ==="
ss -tlnp 2>/dev/null | grep -E '400[02]|7462' || echo "No API port listening"

echo "=== Java running? ==="
pgrep -fa java || echo "No java process"
