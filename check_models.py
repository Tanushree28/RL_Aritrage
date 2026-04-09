#!/usr/bin/env python3
"""
Quick summary of current production models and best versions per asset.
Run: python3 check_models.py
"""
import json, pathlib, os

ASSETS = ['btc', 'eth', 'sol', 'xrp']

print("\n" + "=" * 65)
print("  CURRENT PRODUCTION MODELS")
print("=" * 65)

for asset in ASSETS:
    # What's actually being used right now
    prod_file = pathlib.Path(f"model/{asset}/rl_policy.json")
    if prod_file.exists():
        d = json.loads(prod_file.read_text())
        ts = d.get('timestamp', 0)
        states = len(d.get('q_table', {}))
        import datetime
        dt = datetime.datetime.fromtimestamp(ts).strftime('%b %d %I:%M %p')
        print(f"  {asset.upper():<4}  states={states:<4}  updated={dt}")
    else:
        print(f"  {asset.upper():<4}  ❌ no production model found")

print("\n" + "=" * 65)
print("  TOP 3 VERSIONS PER ASSET (by Sharpe, capped at 100)")
print("=" * 65)

for asset in ASSETS:
    versions_dir = pathlib.Path(f"model_registry/{asset}/versions")
    if not versions_dir.exists():
        continue

    versions = []
    for version in versions_dir.iterdir():
        meta = version / 'metadata.json'
        if meta.exists():
            m = json.loads(meta.read_text())
            e = m.get('evaluation', {})
            sharpe = min(e.get('sharpe_ratio', 0), 100)  # cap insane values
            wr     = e.get('win_rate', 0)
            pnl    = e.get('total_pnl', 0)
            trades = e.get('num_trades', 0)
            eps    = m.get('epsilon', 0)
            versions.append((sharpe, wr, pnl, trades, eps, version.name))

    versions.sort(reverse=True)
    print(f"\n  {asset.upper()}")
    for sharpe, wr, pnl, trades, eps, name in versions[:3]:
        marker = " ← IN PRODUCTION" if name == pathlib.Path(f"model/{asset}/rl_policy.json").name else ""
        print(f"    {name}  WR={wr:.0%}  Sharpe={sharpe:.2f}  PnL=${pnl:.0f}  ε={eps:.3f}{marker}")

print()
