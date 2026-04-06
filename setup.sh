#!/bin/bash
set -e

echo "=== Momentum Rotation Bot Setup ==="
echo ""

# Install Python if needed
if ! command -v python3 &>/dev/null; then
    echo "Installing Python..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-pip python3-venv
fi

# Create venv
echo "[1/3] Creating virtual environment..."
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt

# Download data
echo "[2/3] Downloading 15 years of market data..."
python run.py download

# Verify
echo "[3/3] Running backtest..."
python run.py backtest

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next: python run.py signal        # see today's signal"
echo "      python run.py trade          # dry run"
echo "      python run.py trade --live   # send orders to IBKR"
