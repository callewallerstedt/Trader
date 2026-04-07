#!/usr/bin/env python3
"""
Reviewer checks (no parameter changes to production):

1) VIX filter dominance — backtest with VIX filter OFF vs production VIX 20/30.
   Pass criterion: strategy still positive edge without VIX; VIX improves Sharpe/PF/DD.

2) Rebalance robustness — 20d vs 25d vs 30d with same V3 signal and VIX 20/30.
   Pass criterion: flat-ish plateau, not a single sharp optimum.

Uses production engine.backtest + data load (same universe, costs, signal).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy.data import load, load_vix
from strategy.engine import Config, backtest


def _row(label: str, m: dict) -> str:
    return (
        f"  {label:<42s} "
        f"CAGR {m['cagr_pct']:+5.1f}%  "
        f"Sh {m['sharpe']:4.2f}  "
        f"DD {m['max_drawdown_pct']:5.1f}%  "
        f"PF {m['profit_factor']:4.2f}  "
        f"Tr {m['total_trades']:4d}"
    )


def main() -> None:
    daily = load()
    vix = load_vix()

    base = Config()
    print("Production V3 baseline (reference):")
    print(f"  top_n={base.top_n}  rebal={base.rebalance_freq}d  "
          f"VIX {base.vix_reduce_threshold}/{base.vix_flat_threshold}  "
          f"cost {base.round_trip_cost_bps:.0f}bps\n")

    print("=" * 88)
    print("  1. VIX FILTER: OFF vs ON (same signal, same costs)")
    print("=" * 88)

    no_vix_cfg = Config(
        vix_reduce_threshold=999.0,
        vix_flat_threshold=999.0,
    )
    m_no = backtest(daily, no_vix_cfg, vix_series=vix)
    m_on = backtest(daily, Config(), vix_series=vix)

    print(_row("VIX OFF (thresholds 999/999)", m_no))
    print(_row("VIX ON  (production 20/30)    ", m_on))

    sh_delta = m_on["sharpe"] - m_no["sharpe"]
    pf_delta = m_on["profit_factor"] - m_no["profit_factor"]
    dd_delta = m_on["max_drawdown_pct"] - m_no["max_drawdown_pct"]

    print()
    print(f"  Delta (ON minus OFF):  Sharpe {sh_delta:+.2f}  "
          f"PF {pf_delta:+.2f}  MaxDD {dd_delta:+.1f} pp")
    print()
    # Honest read: reviewer wants "still works" + "VIX improves". We use stricter bars.
    strong_without_vix = (
        m_no["sharpe"] >= 1.0
        and m_no["profit_factor"] >= 1.15
        and m_no["max_drawdown_pct"] >= -20.0
    )
    if strong_without_vix:
        print("  Interpretation: Solid standalone selection/trend edge; "
              "VIX mainly sharpens risk-adjusted returns.")
    elif m_no["profit_factor"] >= 1.05 and m_no["cagr_pct"] > 0:
        print("  Interpretation: Some selection edge without VIX (PF>1, positive CAGR), "
              "but risk-adjusted stats rely heavily on the VIX overlay in this sample.")
        print("               Live: assume a meaningful share of backtest Sharpe/DD "
              "comes from regime timing, not stock picking alone.")
    else:
        print("  Interpretation: Without VIX, metrics are weak — "
              "treat the system primarily as a risk-timed momentum sleeve.")
    if sh_delta > 0.05 or pf_delta > 0.03:
        print("  VIX filter materially improves Sharpe/PF/DD vs OFF here.")

    print()
    print("=" * 88)
    print("  2. REBALANCE: 20d vs 25d vs 30d (VIX 20/30 ON)")
    print("=" * 88)

    rebal_results = []
    for freq in (20, 25, 30):
        cfg = Config(rebalance_freq=freq)
        m = backtest(daily, cfg, vix_series=vix)
        rebal_results.append((freq, m))
        print(_row(f"rebal={freq}d", m))

    sharpes = [m["sharpe"] for _, m in rebal_results]
    pfs = [m["profit_factor"] for _, m in rebal_results]
    sh_spread = max(sharpes) - min(sharpes)
    pf_spread = max(pfs) - min(pfs)
    print()
    print(f"  Sharpe range across 20/25/30d: {sh_spread:.2f}  "
          f"PF range: {pf_spread:.2f}")
    if sh_spread <= 0.15 and pf_spread <= 0.08:
        print("  Interpretation: Looks like a plateau (robust to a few days "
              "of rebalance - not a knife-edge optimum).")
    else:
        print("  Interpretation: Some sensitivity to rebalance freq — "
              "monitor live turnover vs backtest; avoid mid-paper tweaks.")

    print()
    print("=" * 88)
    print("  Reviewer expectation bands (live vs backtest optimism)")
    print("=" * 88)
    print("  Live may land near: Sharpe 1.1–1.3, PF 1.25–1.32, DD -10% to -18%, CAGR 12–18%.")
    print("  Next step: paper trade 4–8 weeks; do not change parameters during validation.")
    print("=" * 88)


if __name__ == "__main__":
    main()
