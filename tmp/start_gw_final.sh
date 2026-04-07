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

sleep 45

echo "=== Gateway log tail ==="
tail -15 /home/calle/ibc/logs/gateway_latest.log

echo "=== Port check ==="
ss -tlnp 2>/dev/null | grep -E '400[02]|7462' || echo "No API port"

echo "=== Java ==="
pgrep -fa java || echo "No java"

DIAG=$(ls -t /home/calle/ibc/logs/ibc-*.txt 2>/dev/null | head -1)
if [ -n "$DIAG" ]; then
    echo "=== Diagnostic tail ==="
    tail -15 "$DIAG"
fi
