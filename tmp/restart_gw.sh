#!/bin/bash
set -e

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
bash ./gatewaystart.sh -inline 2>&1 | tail -40
