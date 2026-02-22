#!/bin/bash

echo "⚠️  INITIATING EMERGENCY ROLLBACK"
echo "================================================================"

# Check if model registry exists
if [ ! -d "model_registry/versions" ]; then
    echo "❌ Model registry not found"
    exit 1
fi

# Get current version
if [ -L "model_registry/current" ]; then
    CURRENT=$(readlink model_registry/current | xargs basename)
    echo "📊 Current version: $CURRENT"
else
    echo "❌ No current model symlink found"
    exit 1
fi

# Find all versions sorted by name (timestamp-based)
VERSIONS=($(ls -1 model_registry/versions/ | sort))
NUM_VERSIONS=${#VERSIONS[@]}

echo "📚 Found $NUM_VERSIONS versions in registry"

if [ $NUM_VERSIONS -lt 2 ]; then
    echo "❌ Only one version available, cannot rollback"
    exit 1
fi

# Find current version index
CURRENT_IDX=-1
for i in "${!VERSIONS[@]}"; do
    if [ "${VERSIONS[$i]}" = "$CURRENT" ]; then
        CURRENT_IDX=$i
        break
    fi
done

if [ $CURRENT_IDX -eq -1 ]; then
    echo "❌ Current version not found in registry"
    exit 1
fi

if [ $CURRENT_IDX -eq 0 ]; then
    echo "❌ Already at oldest version, cannot rollback further"
    exit 1
fi

# Get previous version
PREV_IDX=$((CURRENT_IDX - 1))
PREV_VERSION="${VERSIONS[$PREV_IDX]}"

echo ""
echo "🔄 Rolling back:"
echo "   From: $CURRENT"
echo "   To:   $PREV_VERSION"
echo ""

# Confirm rollback (skip in non-interactive mode with --force flag)
if [ "$1" != "--force" ]; then
    read -p "Are you sure you want to rollback? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "❌ Rollback cancelled"
        exit 0
    fi
fi

# Perform atomic symlink swap
echo "🔧 Performing atomic rollback..."

TARGET_PATH="versions/$PREV_VERSION"
cd model_registry

# Create temporary symlink
ln -sfn "$TARGET_PATH" current.tmp

# Atomic replace
if [ -L "current" ]; then
    rm current
fi

mv current.tmp current

cd ..

echo ""
echo "================================================================"
echo "✅ Rollback complete!"
echo "================================================================"
echo ""
echo "Model reverted to: $PREV_VERSION"
echo ""
echo "The Rust executor will reload the model on next hot-reload cycle (~60s)"
echo "Or restart the bot immediately: pkill -f 'cargo run' && cargo run --release"
echo ""
echo "To verify:"
echo "  readlink model_registry/current"
echo ""
echo "================================================================"
