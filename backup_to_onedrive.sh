#!/bin/bash
# Backup script to sync data to OneDrive

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
TARGET="onedrive:KalshiBotData/ServerBackup_$TIMESTAMP"

echo "Starting backup to $TARGET..."

# Sync the data directory (CSVs, DBs)
rclone copy ~/kalshi-arb/data $TARGET/data

# Sync the logs directory
rclone copy ~/kalshi-arb/logs $TARGET/logs

echo "Backup complete at $TIMESTAMP"
