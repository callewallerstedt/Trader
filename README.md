# Momentum Rotation Trading Bot

Daily momentum strategy that trades US equities via Interactive Brokers.

**Strategy:** Hold the top 3 stocks by 20-day momentum when SPY is above its
100-day moving average. Go to cash when SPY drops below. Volatility-scale
exposure to target 15% annualized vol.

**Backtest (2010-2025, 15 years, 39 stocks, 20 bps costs):**

| Metric | Strategy | SPY |
|--------|----------|-----|
| CAGR | +30.1% | +14.8% |
| Sharpe | 1.12 | 0.89 |
| Max Drawdown | -37.0% | -33.7% |
| Total Return | +5,851% | +754% |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/callewallerstedt/Trader.git
cd Trader

# 2. Setup (creates venv, downloads data, runs backtest)
chmod +x setup.sh
./setup.sh

# 3. Check today's signal
source .venv/bin/activate
python run.py signal
```

## Commands

```bash
python run.py download          # Download 15 years of daily data
python run.py backtest          # Run backtest, show results
python run.py signal            # Show today's signal
python run.py trade             # Dry run (no orders sent)
python run.py trade --live      # Send MOC orders to IBKR
```

## How It Works

Every weekday at **3:45 PM ET** (15 min before market close):

1. Fetch latest prices from Yahoo Finance
2. Check if SPY is above its 100-day moving average
   - **YES:** Pick top 3 stocks by 20-day momentum, submit MOC buy orders
   - **NO:** Go to cash, submit MOC sell orders for any holdings
3. MOC (Market-On-Close) orders fill at the 4 PM closing auction

The strategy rebalances only when the target portfolio changes (typically
every few days). Average holding period is ~1-2 weeks.

## Live Trading Setup

### 1. Install IB Gateway

Download from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/ibgateway-stable.php) (get the Linux version).

```bash
chmod +x ibgateway-stable-standalone-linux-x64.sh
./ibgateway-stable-standalone-linux-x64.sh
```

Start it, log in with your IBKR credentials, select **Paper Trading**.

Configure: **Configure > API > Settings:**
- Enable ActiveX and Socket Clients: **Yes**
- Socket port: **7497** (paper) / **7496** (live)
- Read-Only API: **No**

### 2. Test

```bash
source .venv/bin/activate

# Dry run first (shows what it would do, sends nothing)
python run.py trade

# Paper trade (sends real orders to paper account)
python run.py trade --live
```

Check your IBKR paper account to verify the positions match.

### 3. Automate with Cron

```bash
chmod +x cron_trade.sh

# Add to crontab (3:45 PM ET = 19:45 UTC during EDT)
crontab -e
```

Add this line:

```
45 19 * * 1-5 /home/YOUR_USER/Trader/cron_trade.sh
```

> During EST (Nov-Mar), use `45 20 * * 1-5` instead.
> Or set server timezone to `America/New_York` and use `45 15 * * 1-5`.

### 4. Go Live

After 2-4 weeks of paper trading, switch IB Gateway to your live account
and change the port in `broker/ibkr.py` from `7497` to `7496`.

## Files

```
run.py              # Single entry point for all commands
strategy/
  engine.py         # Signal computation + backtest engine
  data.py           # Yahoo Finance download + data loading
broker/
  ibkr.py           # IBKR MOC order execution
cron_trade.sh       # Cron wrapper for daily trading
setup.sh            # One-command setup
requirements.txt    # Python dependencies
```

## Monitoring

```bash
# Recent trade logs
tail -50 trade.log

# Detailed JSON logs (one per run)
ls -lt logs/
cat logs/$(ls -t logs/ | head -1)
```
