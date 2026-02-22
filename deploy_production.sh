#!/bin/bash
set -e

echo "🚀 Deploying Production RL Agent with Continuous Learning"
echo "================================================================"

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p model_registry/{versions,staging,archive}
mkdir -p data/{backups,recordings}
mkdir -p monitoring
mkdir -p logs

# Initialize databases
echo "💾 Initializing databases..."
python3 -c "from experience_buffer import ExperienceBuffer; ExperienceBuffer()" 2>/dev/null || echo "Experience buffer initialized"
python3 -c "from metrics_collector import MetricsCollector; MetricsCollector()" 2>/dev/null || echo "Metrics collector initialized"

# Check if we have historical data for initial training
if [ -d "data/recordings" ] && [ "$(find data/recordings -name '*.csv' | wc -l)" -gt 10 ]; then
    echo "📊 Historical data found. Training initial model..."
    python3 continuous_learner.py --episodes 50 || {
        echo "⚠️  Initial training failed, but continuing deployment"
    }
else
    echo "⚠️  No historical data found. Model will be trained after first trades."
fi

# Verify model registry setup
if [ -L "model_registry/current" ]; then
    CURRENT_VERSION=$(readlink model_registry/current | xargs basename)
    echo "✅ Model registry initialized (current: $CURRENT_VERSION)"
elif [ -d "model" ] && [ -f "model/rl_policy.json" ]; then
    echo "📦 Migrating existing model to registry..."
    mkdir -p model_registry/versions/v_legacy
    cp model/rl_q_table.pkl model_registry/versions/v_legacy/ 2>/dev/null || true
    cp model/rl_policy.json model_registry/versions/v_legacy/
    echo '{"version": "v_legacy", "migrated": true}' > model_registry/versions/v_legacy/metadata.json
    ln -sf versions/v_legacy model_registry/current
    echo "✅ Legacy model migrated to registry"
else
    echo "ℹ️  No model available yet. Will train after collecting experiences."
fi

# Check dependencies
echo ""
echo "🔍 Checking dependencies..."

# Check Rust
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found. Please install Rust."
    exit 1
fi

# Check Python packages
MISSING_PKGS=""
python3 -c "import streamlit" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS streamlit"
python3 -c "import plotly" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS plotly"
python3 -c "import pandas" 2>/dev/null || MISSING_PKGS="$MISSING_PKGS pandas"

if [ ! -z "$MISSING_PKGS" ]; then
    echo "📦 Installing missing Python packages:$MISSING_PKGS"
    pip3 install$MISSING_PKGS
fi

echo "✅ All dependencies satisfied"

# Build Rust bot
echo ""
echo "🔨 Building Rust trading bot..."
cargo build --release || {
    echo "❌ Build failed"
    exit 1
}
echo "✅ Build successful"

echo ""
echo "================================================================"
echo "✅ Deployment complete!"
echo "================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start monitoring dashboard (optional):"
echo "   streamlit run monitoring/dashboard.py"
echo ""
echo "2. Start continuous learner in background (optional):"
echo "   nohup python3 continuous_learner.py --loop --interval 120 > logs/continuous_learner.log 2>&1 &"
echo ""
echo "3. Start alerting monitor (optional):"
echo "   nohup python3 alerting.py --interval 60 > logs/alerting.log 2>&1 &"
echo ""
echo "4. Start the trading bot:"
echo "   cargo run --release"
echo ""
echo "   Or with environment variables:"
echo "   PAPER_MODE=1 cargo run --release"
echo ""
echo "================================================================"
