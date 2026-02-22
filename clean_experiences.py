#!/usr/bin/env python3
"""
Clean corrupted experiences from the database by cross-referencing
with Kalshi's official settlement data.

Usage:
  python3 clean_experiences.py --asset btc   # Clean BTC experiences (default)
  python3 clean_experiences.py --asset eth   # Clean ETH experiences

This script:
1. Reads unique tickers from data/{asset}/experiences.db
2. Fetches historical market data from Kalshi API to get actual settlement
3. Identifies experiences with incorrect rewards (e.g., YES reward positive but NO actually won)
4. Deletes corrupted entries
"""

import argparse
import sqlite3
import os
import json
import time
import hashlib
import base64
from datetime import datetime
from pathlib import Path

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

# ============================================================================
# Configuration (set by --asset argument)
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
KALSHI_API_BASE = "https://api.elections.kalshi.com"

# ============================================================================
# Kalshi API Authentication
# ============================================================================

def load_credentials(asset: str = "btc"):
    """Load API credentials from .env.{asset} file."""
    env_path = PROJECT_ROOT / f".env.{asset}"
    if not env_path.exists():
        # Fall back to generic .env
        env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise ValueError(f".env file not found at {env_path}")
    
    # Parse .env file manually to handle multi-line private key
    env_content = env_path.read_text()
    
    api_key_id = None
    private_key_pem = None
    
    # Extract API key ID
    for line in env_content.split('\n'):
        if line.startswith('KALSHI_API_KEY_ID='):
            api_key_id = line.split('=', 1)[1].strip()
            break
    
    # Extract private key (multi-line)
    import re
    match = re.search(
        r'KALSHI_PRIVATE_KEY="?(-----BEGIN RSA PRIVATE KEY-----.*?-----END RSA PRIVATE KEY-----)"?',
        env_content,
        re.DOTALL
    )
    if match:
        private_key_pem = match.group(1).replace('\\n', '\n')
    
    if not api_key_id or not private_key_pem:
        raise ValueError("Could not parse KALSHI_API_KEY_ID or KALSHI_PRIVATE_KEY from .env")
    
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode(), password=None, backend=default_backend()
    )
    
    return api_key_id, private_key


def sign_request(private_key, timestamp_ms: int, method: str, path: str) -> str:
    """Sign API request using RSA-PSS."""
    message = f"{timestamp_ms}{method}{path}".encode()
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode()


def fetch_market_result(session, api_key_id, private_key, ticker: str) -> dict:
    """Fetch market details including settlement result."""
    timestamp_ms = int(time.time() * 1000)
    path = f"/trade-api/v2/markets/{ticker}"
    signature = sign_request(private_key, timestamp_ms, "GET", path)
    
    headers = {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
    }
    
    resp = session.get(f"{KALSHI_API_BASE}{path}", headers=headers)
    
    if resp.status_code == 200:
        data = resp.json()
        market = data.get("market", {})
        return {
            "ticker": ticker,
            "status": market.get("status"),
            "result": market.get("result"),
            "floor_strike": market.get("floor_strike"),
            "close_time": market.get("close_time"),
        }
    elif resp.status_code == 404:
        return {"ticker": ticker, "status": "not_found", "result": None}
    else:
        print(f"  [WARN] API error for {ticker}: {resp.status_code}")
        return {"ticker": ticker, "status": "error", "result": None}


# ============================================================================
# Database Operations
# ============================================================================

def get_unique_tickers(conn) -> list:
    """Get all unique tickers from experiences database."""
    cursor = conn.execute("SELECT DISTINCT ticker FROM experiences")
    return [row[0] for row in cursor.fetchall()]


def get_experiences_for_ticker(conn, ticker: str) -> list:
    """Get all experiences for a specific ticker."""
    cursor = conn.execute(
        """
        SELECT id, action, reward, datetime(timestamp, 'unixepoch') as ts
        FROM experiences
        WHERE ticker = ?
        """,
        (ticker,)
    )
    return [
        {"id": row[0], "action": row[1], "reward": row[2], "timestamp": row[3]}
        for row in cursor.fetchall()
    ]


def delete_experiences(conn, ids: list):
    """Delete experiences by ID."""
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    conn.execute(f"DELETE FROM experiences WHERE id IN ({placeholders})", ids)
    conn.commit()


# ============================================================================
# Validation Logic
# ============================================================================

def validate_experience(exp: dict, actual_result: str) -> bool:
    """
    Check if an experience's reward is consistent with the actual settlement.
    
    Action mapping: 0=HOLD, 1=YES, 2=NO
    
    For YES action (1):
      - If actual result was "yes": reward should be positive (win)
      - If actual result was "no": reward should be negative (loss)
    
    For NO action (2):
      - If actual result was "yes": reward should be negative (loss)
      - If actual result was "no": reward should be positive (win)
    
    For HOLD action (0):
      - Reward should be ~0 (neutral)
    """
    action = exp["action"]
    reward = exp["reward"]
    
    if action == 0:  # HOLD
        # HOLD should have ~0 reward (allow small tolerance)
        return abs(reward) < 1.0
    
    if action == 1:  # YES
        if actual_result == "yes":
            return reward > -10  # Allow small losses due to fees, but mostly positive
        elif actual_result == "no":
            return reward < 10  # Should be negative (loss)
        else:
            return True  # Unknown result, keep
    
    if action == 2:  # NO
        if actual_result == "yes":
            return reward < 10  # Should be negative (loss)
        elif actual_result == "no":
            return reward > -10  # Should be positive (win)
        else:
            return True
    
    return True  # Unknown action, keep


def identify_corrupted_experiences(experiences: list, actual_result: str) -> list:
    """
    Identify experiences that have rewards inconsistent with actual settlement.
    
    Returns list of experience IDs that should be deleted.
    """
    corrupted_ids = []
    
    for exp in experiences:
        action = exp["action"]
        reward = exp["reward"]
        
        # Skip HOLD actions - they're always ~0
        if action == 0:
            continue
        
        is_corrupted = False
        
        if action == 1:  # YES
            if actual_result == "yes" and reward < -10:
                # YES won but recorded as big loss - CORRUPTED
                is_corrupted = True
            elif actual_result == "no" and reward > 10:
                # YES lost but recorded as big win - CORRUPTED
                is_corrupted = True
        
        elif action == 2:  # NO
            if actual_result == "no" and reward < -10:
                # NO won but recorded as big loss - CORRUPTED
                is_corrupted = True
            elif actual_result == "yes" and reward > 10:
                # NO lost but recorded as big win - CORRUPTED
                is_corrupted = True
        
        if is_corrupted:
            corrupted_ids.append(exp["id"])
    
    return corrupted_ids


# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Clean corrupted experiences from database")
    parser.add_argument("--asset", type=str, default="btc", choices=["btc", "eth"],
                        help="Asset to clean (default: btc)")
    args = parser.parse_args()

    asset = args.asset
    db_path = PROJECT_ROOT / "data" / asset / "experiences.db"

    print("=" * 60)
    print(f"Experience Database Cleanup Script ({asset.upper()})")
    print("=" * 60)

    # Load credentials
    try:
        api_key_id, private_key = load_credentials(asset=asset)
        print("[OK] Loaded API credentials")
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Connect to database
    if not db_path.exists():
        print(f"[ERROR] Database not found: {db_path}")
        return

    conn = sqlite3.connect(db_path)
    print(f"[OK] Connected to {db_path}")
    
    # Get unique tickers
    tickers = get_unique_tickers(conn)
    print(f"[INFO] Found {len(tickers)} unique tickers")
    
    # Create HTTP session
    session = requests.Session()
    
    # Process each ticker
    total_checked = 0
    total_corrupted = 0
    corrupted_by_ticker = {}
    settlement_results = {}
    
    print("\nFetching settlement data from Kalshi API...")
    print("-" * 60)
    
    for i, ticker in enumerate(tickers):
        # Rate limiting - 2 requests per second
        if i > 0 and i % 2 == 0:
            time.sleep(0.5)
        
        # Fetch actual result
        result = fetch_market_result(session, api_key_id, private_key, ticker)
        
        if result["status"] == "not_found":
            print(f"  [{i+1}/{len(tickers)}] {ticker}: NOT FOUND (skipping)")
            continue
        
        if result["status"] == "error":
            print(f"  [{i+1}/{len(tickers)}] {ticker}: API ERROR (skipping)")
            continue
        
        actual_result = result.get("result")
        settlement_results[ticker] = actual_result
        
        if not actual_result:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: status={result['status']} (not settled)")
            continue
        
        # Get experiences for this ticker
        experiences = get_experiences_for_ticker(conn, ticker)
        total_checked += len(experiences)
        
        # Identify corrupted
        corrupted_ids = identify_corrupted_experiences(experiences, actual_result)
        
        if corrupted_ids:
            total_corrupted += len(corrupted_ids)
            corrupted_by_ticker[ticker] = corrupted_ids
            print(f"  [{i+1}/{len(tickers)}] {ticker}: result={actual_result} -> {len(corrupted_ids)} CORRUPTED")
        else:
            print(f"  [{i+1}/{len(tickers)}] {ticker}: result={actual_result} -> OK")
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Total tickers processed: {len(tickers)}")
    print(f"  Total experiences checked: {total_checked}")
    print(f"  Total corrupted entries: {total_corrupted}")
    
    if total_corrupted == 0:
        print("\n[OK] No corrupted entries found!")
        conn.close()
        return
    
    # Show corrupted breakdown
    print(f"\nCorrupted entries by ticker:")
    for ticker, ids in corrupted_by_ticker.items():
        result = settlement_results.get(ticker, "?")
        print(f"  {ticker} (result={result}): {len(ids)} entries")
    
    # Confirm deletion
    print(f"\n{'='*60}")
    response = input(f"Delete {total_corrupted} corrupted entries? [y/N]: ")
    
    if response.lower() == "y":
        all_ids = []
        for ids in corrupted_by_ticker.values():
            all_ids.extend(ids)
        
        delete_experiences(conn, all_ids)
        print(f"[OK] Deleted {len(all_ids)} corrupted entries")
        
        # Vacuum to reclaim space
        conn.execute("VACUUM")
        print("[OK] Database vacuumed")
    else:
        print("[INFO] No changes made")
    
    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
