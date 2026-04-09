#!/bin/bash
# Backup script to sync data to OneDrive

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
TARGET="onedrive:KalshiBotData/ServerBackup_$TIMESTAMP"
ARCHIVE_NAME="kalshi_backup_$TIMESTAMP.tar.gz"

echo "Starting backup to $TARGET..."

# 1. Compress Everything First (Super fast for thousands of CSVs)
echo "Compressing data and logs..."
cd ~/kalshi-arb
tar -czf $ARCHIVE_NAME data/ logs/

# 2. Upload the single compressed file
echo "Uploading $ARCHIVE_NAME to OneDrive..."
rclone copy $ARCHIVE_NAME $TARGET/ --transfers 4

if [ $? -eq 0 ]; then
    echo "Backup successful! Cleaning up..."
    
    # Remove the local compressed archive
    rm $ARCHIVE_NAME
    
    # Remove files older than 7 days
    echo "Deleting CSVs and Logs older than 7 days..."
    find ~/kalshi-arb/data -type f -name "*.csv" -mtime +7 -exec rm -f {} \;
    find ~/kalshi-arb/logs -type f -name "*.log*" -mtime +7 -exec rm -f {} \;
    
    # Remove any completely empty day folders
    find ~/kalshi-arb/data -type d -empty -delete
    
    echo "Cleanup complete at $(date +"%Y-%m-%d_%H-%M-%S")"
else
    echo "Upload failed! Keeping local archive just in case."
fi
