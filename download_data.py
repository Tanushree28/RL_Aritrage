"""
Download 1-minute candle data from Coinbase Exchange and Kraken.

Both endpoints are public — no authentication required.

Usage:
  python3 download_data.py --asset btc   # Download BTC data (default)
  python3 download_data.py --asset eth   # Download ETH data

Output (data/raw/):
  coinbase_{asset}usd_1m.csv   — primary training source (up to 60 days)
  kraken_{asset}usd_1m.csv     — recent cross-check (~12 h)
  kraken_{asset}usd_5m.csv     — longer secondary history (~2.5 days)

CSV columns: timestamp_unix_ms,open,high,low,close,volume
"""

import argparse
import csv, os, time
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR      = os.path.join(PROJECT_ROOT, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

DAYS_COINBASE = 60          # how far back to fetch from Coinbase

# Asset-specific configurations
COINBASE_PAIRS = {
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "sol": "SOL-USD",
}

KRAKEN_PAIRS = {
    "btc": {"pair": "XBTUSD", "result_key": "XXBTZUSD"},
    "eth": {"pair": "ETHUSD", "result_key": "XETHZUSD"},
    "sol": {"pair": "SOLUSD", "result_key": "SOLUSD"},
}

# ---------------------------------------------------------------------------
# Coinbase Exchange  (public, no auth)
# ---------------------------------------------------------------------------
# GET /products/{PAIR}/candles?start={unix_s}&end={unix_s}&granularity=60
# Returns ≤ 300 candles, newest first: [[time_s, open, high, low, close, volume], …]
# ---------------------------------------------------------------------------

COINBASE_BASE_URL = "https://api.exchange.coinbase.com/products"

def download_coinbase(asset: str = "btc", days=DAYS_COINBASE):
    """Paginate backwards in 300-candle (= 5 h) chunks.  Returns sorted rows."""
    pair = COINBASE_PAIRS.get(asset, "BTC-USD")
    url = f"{COINBASE_BASE_URL}/{pair}/candles"

    print(f"Downloading Coinbase {pair} 1-min candles ({days} days)…")
    now    = int(time.time())
    target = now - days * 86_400
    rows   = {}          # ts_s → (ts_ms, open, high, low, close, volume)

    end = now
    while end > target:
        start = max(target, end - 300 * 60)

        try:
            resp = requests.get(url, params={
                "start":       start,
                "end":         end,
                "granularity": 60,
            }, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"\n  Coinbase timeout/error at end={end}: {e}")
            print(f"  Saving {len(rows):,} candles collected so far.")
            break

        if resp.status_code != 200:
            print(f"\n  Coinbase {resp.status_code}: {resp.text[:150]}")
            break

        candles = resp.json()          # newest first
        if not candles:
            break

        for c in candles:
            ts_s = int(c[0])
            rows[ts_s] = (ts_s * 1000, float(c[1]), float(c[2]),
                          float(c[3]), float(c[4]), float(c[5]))

        # Move end to just before the oldest candle we received
        oldest_ts = int(candles[-1][0])   # last element = oldest (newest-first)
        if oldest_ts >= end:
            break                         # no progress — stop to avoid infinite loop
        end = oldest_ts

        print(f"  {len(rows):,} candles so far…", end="\r")

    # Sort ascending by timestamp
    sorted_rows = [rows[k] for k in sorted(rows)]
    print(f"  {len(sorted_rows):,} candles total")

    path = os.path.join(RAW_DIR, f"coinbase_{asset}usd_1m.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_unix_ms", "open", "high", "low", "close", "volume"])
        w.writerows(sorted_rows)

    span_days = (sorted_rows[-1][0] - sorted_rows[0][0]) / 86_400_000 if sorted_rows else 0
    print(f"  Saved → {path}  ({span_days:.1f} days)")
    return sorted_rows


# ---------------------------------------------------------------------------
# Kraken  (public, no auth)
# ---------------------------------------------------------------------------
# GET /0/public/OHLC?pair={PAIR}&interval={min}
# Returns up to 720 candles: [time_s, open, high, low, close, vwap, volume, trades]
# ---------------------------------------------------------------------------

KRAKEN_URL = "https://api.kraken.com/0/public/OHLC"

def download_kraken(asset: str = "btc"):
    """Download 1-min and 5-min candles.  Returns dict of label → rows."""
    kraken_config = KRAKEN_PAIRS.get(asset, KRAKEN_PAIRS["btc"])
    pair = kraken_config["pair"]
    result_key = kraken_config["result_key"]

    print(f"Downloading Kraken {asset.upper()}/USD candles…")
    results = {}

    for interval, label in [(1, "1m"), (5, "5m")]:
        resp = requests.get(KRAKEN_URL, params={
            "pair":     pair,
            "interval": interval,
        }, timeout=15)
        resp.raise_for_status()

        ohlc = resp.json().get("result", {}).get(result_key, [])
        rows = []
        for c in ohlc:
            rows.append((
                int(c[0]) * 1000,   # timestamp_ms
                float(c[1]),        # open
                float(c[2]),        # high
                float(c[3]),        # low
                float(c[4]),        # close
                float(c[6]),        # volume  (index 6; 5 is vwap)
            ))
        results[label] = rows

        path = os.path.join(RAW_DIR, f"kraken_{asset}usd_{label}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_unix_ms", "open", "high", "low", "close", "volume"])
            w.writerows(rows)

        span_h = (rows[-1][0] - rows[0][0]) / 3_600_000 if len(rows) > 1 else 0
        print(f"  {label}: {len(rows):,} candles ({span_h:.1f} h) → {path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download cryptocurrency price data")
    parser.add_argument("--asset", type=str, default="btc", choices=["btc", "eth"],
                        help="Asset to download (default: btc)")
    args = parser.parse_args()

    asset = args.asset
    print(f"Downloading {asset.upper()} price data...")
    print()

    cb = download_coinbase(asset=asset)
    kr = download_kraken(asset=asset)

    print(f"\n{'─'*50}")
    print(f"  Coinbase 1-min : {len(cb):>7,} candles")
    for label, rows in kr.items():
        print(f"  Kraken   {label}  : {len(rows):>7,} candles")
    print(f"  All files in {RAW_DIR}")
    print(f"{'─'*50}")
