#!/bin/bash
# Start IB Gateway headless using Xvfb + IBC

# Kill any existing gateway/Xvfb
pkill -f ibgateway 2>/dev/null
pkill -f 'Xvfb :99' 2>/dev/null
sleep 2

# Clean up lock files
rm -f /tmp/.X99-lock

# Start virtual display
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
sleep 1

# Paths
mkdir -p /home/calle/Jts /home/calle/ibc/logs

# Launch gateway via IBC's own script
export TWS_MAJOR_VRSN=1037
cd /home/calle/ibc
./gatewaystart.sh -inline > /home/calle/ibc/logs/gateway.log 2>&1 &

echo "Gateway starting in background (PID: $!)"
echo "Logs: /home/calle/ibc/logs/gateway.log"
echo "Display: :99 (Xvfb)"
