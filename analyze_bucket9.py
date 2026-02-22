import os
import pandas as pd
import glob
from datetime import datetime

# Path to BTC recordings
log_dir = os.path.join(os.getcwd(), "data/bucket9-btc/recordings/")
today_dir = os.path.join(log_dir, datetime.today().strftime('%Y-%m-%d'))

print(f"Scanning logs in: {today_dir}")

# Find all CSV files for today
csv_files = glob.glob(os.path.join(today_dir, "*.csv"))

total_opportunites = 0

if not csv_files:
    print("No logs found for today.")
else:
    for file in csv_files:
        try:
            # Try to read with inferred headers (works for files created today)
            df = pd.read_csv(file)
            
            # If the CSV doesn't have headers, pd will use the first row of data as headers, we can skip older files or handle them if needed.
            if 'kalshi_no_ask' not in df.columns:
                continue

            # Determine the best price to use (avg_price > coinbase_price > synth_index_spot)
            if 'avg_price' in df.columns:
                active_price = df['avg_price']
            elif 'coinbase_price' in df.columns:
                active_price = df['coinbase_price']
            else:
                active_price = df['synth_index_spot']
                
            # Filter for Bucket 9 (NO >= 0.90) AND Condition (active_price < strike_price)
            opportunities = df[
                (df['kalshi_no_ask'] >= 0.90) & 
                (active_price < df['strike_price'])
            ]
            
            count = len(opportunities)
            total_opportunites += count
            
            if count > 0:
                print(f"Found {count} matches in {os.path.basename(file)}!")
                print("...")

        except Exception as e:
            # file might be empty or being written to
            continue

print(f"\nTotal 'Perfect' Bucket 9 Opportunities Today: {total_opportunites}")
