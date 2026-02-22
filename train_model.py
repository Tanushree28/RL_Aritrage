"""
Train a feedforward classifier for Kalshi 15-minute crypto contracts.

Two-phase strategy:
  Minutes 0-9  (Observe): accumulate price, build observation context.
  Minutes 10-14 (Trade):  model runs; samples are emitted only in this window.

Price data is loaded from real 1-minute candles downloaded by
download_data.py (data/raw/coinbase_{asset}usd_1m.csv).  All features are
relative (normalised by strike), so the structural relationships learned
transfer directly to live prices.

Usage:
  python3 train_model.py --asset btc   # Train BTC model (default)
  python3 train_model.py --asset eth   # Train ETH model

Output (all written to model/{asset}/):
  strategy_model.onnx  — ONNX graph   input [1,12] float32 → output [1,1]
  scaler.json          — StandardScaler mean + std per feature
  config.json          — feature names and metadata
"""

import argparse
import csv, json, math, os
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# These will be set based on --asset argument
MODEL_DIR = None
PRICE_DATA_PATH = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_DAYS      = 7           # most recent N days held out for evaluation

# Strike offsets relative to the price at window-start
STRIKE_OFFSETS = [-0.02, -0.01, -0.005, -0.002, 0.0, 0.002, 0.005, 0.01, 0.02]

WINDOW_MINS          = 15    # contract duration
OBSERVATION_END      = 10    # t=0..9 is observation; samples only from t=10..14
HISTORY_MINS         = 60    # lookback for rolling features (volatility_60)
N_FEATURES           = 18    # was 12, adding 6 new predictive features
FEATURE_NAMES        = [
    # Existing 12 features (reactive/descriptive)
    "distance_to_strike",          # 0  (spot - strike) / strike
    "avg_distance_to_strike",      # 1  (obs_mean - strike) / strike  [frozen]
    "time_remaining_frac",         # 2  (15 - t) / 15
    "volatility_60",               # 3  60-min coefficient of variation
    "momentum_window",             # 4  (spot - obs_end_px) / obs_end_px
    "momentum_short",              # 5  3-min lookback momentum
    "obs_high_dist",               # 6  (spot - obs_high) / strike
    "obs_low_dist",                # 7  (spot - obs_low) / strike
    "obs_range_norm",              # 8  (obs_high - obs_low) / strike  [frozen]
    "dist_to_obs_mean",            # 9  (spot - obs_mean) / strike
    "abs_distance_to_strike",      # 10 |feature 0|
    "strike_offset_from_start",    # 11 raw offset used to generate strike  [frozen]
    # New predictive features (momentum & mean-reversion)
    "momentum_5min",               # 12 (spot - price_5min_ago) / price_5min_ago
    "momentum_accel",              # 13 momentum_5min - momentum_10min (acceleration)
    "reversion_pressure",          # 14 (spot - obs_mean) / obs_std (z-score)
    "vol_vs_baseline",             # 15 volatility_60 / historical_avg_vol
    "distance_velocity",           # 16 rate of change of distance_to_strike
    "strike_crossing_momentum",    # 17 momentum_5min * sign(distance_to_strike)
]

EPOCHS        = 50
BATCH_SIZE    = 256
LR            = 0.001
PATIENCE      = 10

# ---------------------------------------------------------------------------
# 1. Load real price data from Coinbase CSV
# ---------------------------------------------------------------------------

def load_prices(asset: str = "btc"):
    """Read 1-min closes from data/raw/coinbase_{asset}usd_1m.csv."""
    path = os.path.join(PROJECT_ROOT, "data", "raw", f"coinbase_{asset}usd_1m.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing price data: {path}\n"
            f"Run  python download_data.py --asset {asset}  first."
        )

    times, closes = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)                          # skip header
        for row in reader:
            times.append(int(row[0]))         # timestamp_unix_ms
            closes.append(float(row[4]))      # close price

    times   = np.array(times,  dtype=np.int64)
    closes  = np.array(closes, dtype=np.float64)

    span_days = (times[-1] - times[0]) / 86_400_000
    print(f"  {len(closes):,} candles ({span_days:.1f} days)  "
          f"price range [{closes.min():.0f} – {closes.max():.0f}]")
    return times, closes


# ---------------------------------------------------------------------------
# 2. Vectorised rolling mean & std (60-minute window, min_periods=1)
# ---------------------------------------------------------------------------

def precompute_rolling(closes):
    n  = len(closes)
    cs = np.concatenate([[0], np.cumsum(closes)])
    c2 = np.concatenate([[0], np.cumsum(closes ** 2)])

    idx     = np.arange(n)
    starts  = np.maximum(0, idx - HISTORY_MINS + 1)
    wsizes  = idx - starts + 1                         # 1 … HISTORY_MINS

    means      = (cs[idx + 1] - cs[starts]) / wsizes
    mean_of_sq = (c2[idx + 1] - c2[starts]) / wsizes
    stds       = np.sqrt(np.maximum(0.0, mean_of_sq - means ** 2))
    return means, stds


# ---------------------------------------------------------------------------
# 3. Generate labelled samples (observation-enriched, trading window only)
# ---------------------------------------------------------------------------

def generate_samples(times, closes, r_mean, r_std):
    """
    For every 15-min window (aligned to UTC) and every strike offset,
    emit one sample per minute within the TRADING window (t=10..14).

    Each sample carries 18 features: live positional/momentum features
    plus observation-window context features computed from t=0..9.

    Settlement = close price of the candle AT expiry (t=15).
    """
    n = len(closes)

    # Compute historical baseline volatility for f15
    historical_avg_vol = np.mean(r_std / np.where(r_mean != 0, r_mean, 1.0))

    # Window-start indices: minute-of-epoch divisible by 15
    min_epoch   = times // 60_000
    aligned     = np.where((min_epoch % 15) == 0)[0]
    # Need HISTORY_MINS before start and at least WINDOW_MINS candles after
    aligned     = aligned[(aligned >= HISTORY_MINS) &
                          (aligned + WINDOW_MINS < n)]

    Xs, ys, ts = [], [], []

    for ws in aligned:                          # ws = window-start index
        settlement = closes[ws + WINDOW_MINS]   # expiry candle (t=15)

        # --- Observation-window summaries (frozen for t=10..14) ----------------
        obs_prices  = closes[ws : ws + OBSERVATION_END]   # 10 candles: t=0..9
        obs_mean    = obs_prices.mean()
        obs_high    = obs_prices.max()
        obs_low     = obs_prices.min()
        obs_range   = obs_high - obs_low
        obs_end_px  = closes[ws + OBSERVATION_END]        # price at t=10

        for offset in STRIKE_OFFSETS:
            strike = round(closes[ws] * (1.0 + offset), 2)
            if strike <= 0:
                continue

            # Soft probabilistic label: fraction of final 5 minutes where price > strike
            final_5min_start = ws + WINDOW_MINS - 5
            final_5min_prices = closes[final_5min_start : ws + WINDOW_MINS + 1]
            prob_above_strike = np.mean(final_5min_prices > strike)
            label = float(prob_above_strike)

            for t in range(OBSERVATION_END, WINDOW_MINS):   # t = 10,11,12,13,14
                i       = ws + t
                spot    = closes[i]

                # Short momentum: 3-candle lookback, clamped to window start
                short_idx = ws + max(0, t - 3)
                short_ref = closes[short_idx]

                # Existing 12 features (f0-f11)
                f0  = (spot - strike) / strike
                f1  = (obs_mean - strike) / strike
                f2  = (WINDOW_MINS - t) / WINDOW_MINS
                f3  = r_std[i] / r_mean[i] if r_mean[i] != 0 else 0.0
                f4  = (spot - obs_end_px) / obs_end_px if obs_end_px != 0 else 0.0
                f5  = (spot - short_ref) / short_ref if short_ref != 0 else 0.0
                f6  = (spot - obs_high) / strike
                f7  = (spot - obs_low) / strike
                f8  = obs_range / strike
                f9  = (spot - obs_mean) / strike
                f10 = abs(f0)
                f11 = offset

                # New predictive features (f12-f17)
                # f12: momentum_5min
                price_5min_ago = closes[max(ws, i - 5)]
                f12 = (spot - price_5min_ago) / price_5min_ago if price_5min_ago != 0 else 0.0

                # f13: momentum_accel (5min momentum - 10min momentum)
                price_10min_ago = closes[max(ws, i - 10)]
                momentum_10min = (spot - price_10min_ago) / price_10min_ago if price_10min_ago != 0 else 0.0
                f13 = f12 - momentum_10min

                # f14: reversion_pressure (z-score from obs_mean)
                obs_std = obs_range / 4.0 if obs_range > 0 else 1.0
                f14 = (spot - obs_mean) / obs_std if obs_std > 0 else 0.0

                # f15: vol_vs_baseline
                f15 = f3 / historical_avg_vol if historical_avg_vol > 0 else 1.0

                # f16: distance_velocity (change in distance_to_strike per minute)
                prev_spot = closes[i - 1] if i > ws else spot
                prev_distance = (prev_spot - strike) / strike
                f16 = f0 - prev_distance

                # f17: strike_crossing_momentum (directional signal)
                f17 = f12 * (1.0 if f0 > 0 else -1.0)

                Xs.append([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11,
                           f12, f13, f14, f15, f16, f17])
                ys.append(label)
                ts.append(times[i])

    return (np.array(Xs, dtype=np.float32),
            np.array(ys, dtype=np.float32),
            np.array(ts, dtype=np.int64))


# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------

def train(X_tr, y_tr, X_val, y_val, sample_weight=None):
    import tensorflow as tf

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(N_FEATURES,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=PATIENCE, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
            ),
        ],
    )
    return model


# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------

def fee_impact(price):
    """Fee in probability space for a single contract (mirrors Rust)."""
    return math.ceil(7.0 * price * (1.0 - price)) / 100.0


def evaluate(model, X_te, y_te):
    preds = model.predict(X_te, verbose=0).ravel()
    loss, acc, auc = model.evaluate(X_te, y_te, verbose=0)

    print(f"\n{'─'*50}")
    print(f"  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test AUC-ROC:  {auc:.4f}")

    # Calibration table — bin predictions into deciles
    print(f"\n  {'Bin':>12}  {'Pred Mean':>10}  {'Actual %':>10}  {'Count':>8}")
    print(f"  {'─'*46}")
    edges = np.linspace(0, 1, 11)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        print(f"  [{lo:.1f}–{hi:.1f})  {preds[mask].mean():>10.3f}"
              f"  {y_te[mask].mean():>10.3f}  {mask.sum():>8,}")
    print(f"{'─'*50}\n")


def evaluate_profit(model, X_te_scaled, X_te_raw, y_te):
    """
    Simulate the exact trading logic from trade_executor on the test set.

    Uses sigmoid(k=20) on raw feature 0 (distance_to_strike) as a proxy
    for Kalshi's yes_ask.  Relative ranking across model versions is stable
    even if absolute PnL differs from live.
    """
    preds = model.predict(X_te_scaled, verbose=0).ravel()

    MINIMUM_EDGE = 0.02
    k = 20.0

    total_pnl  = 0.0
    trades     = 0
    wins       = 0

    for i in range(len(preds)):
        model_prob = float(preds[i])
        raw_dist   = float(X_te_raw[i, 0])   # distance_to_strike (unscaled)

        # Hypothetical Kalshi market price from distance
        kalshi_yes_ask = 1.0 / (1.0 + math.exp(-k * raw_dist))
        kalshi_no_ask  = 1.0 - kalshi_yes_ask

        yes_edge = model_prob - kalshi_yes_ask - fee_impact(kalshi_yes_ask)
        no_edge  = (1.0 - model_prob) - kalshi_no_ask - fee_impact(kalshi_no_ask)

        if yes_edge >= no_edge and yes_edge > MINIMUM_EDGE:
            pnl = float(y_te[i]) - kalshi_yes_ask - fee_impact(kalshi_yes_ask)
            total_pnl += pnl
            trades += 1
            if y_te[i] == 1.0:
                wins += 1
        elif no_edge > MINIMUM_EDGE:
            pnl = (1.0 - float(y_te[i])) - kalshi_no_ask - fee_impact(kalshi_no_ask)
            total_pnl += pnl
            trades += 1
            if y_te[i] == 0.0:
                wins += 1

    win_rate = wins / trades * 100.0 if trades > 0 else 0.0
    avg_pnl  = total_pnl / trades      if trades > 0 else 0.0

    print(f"  ── Profit Simulation (hypothetical market, k={k}) ──")
    print(f"  Trades: {trades:>8,}   Wins: {wins:>8,}   Win Rate: {win_rate:.1f}%")
    print(f"  Total PnL: {total_pnl:>+10.2f}   Avg PnL/trade: {avg_pnl:>+.4f}")
    print(f"{'─'*50}\n")


# ---------------------------------------------------------------------------
# 6. Export ONNX + scaler + config
# ---------------------------------------------------------------------------

def export(model, scaler_mean, scaler_std, model_dir: str):
    import tensorflow as tf
    import tf2onnx, onnx

    os.makedirs(model_dir, exist_ok=True)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, N_FEATURES], dtype=tf.float32, name="input")
    ])
    def infer(x):
        return model(x, training=False)

    spec = [tf.TensorSpec(shape=[1, N_FEATURES], dtype=tf.float32, name="input")]
    onnx_model, _ = tf2onnx.convert.from_function(infer, input_signature=spec, opset=17)

    onnx_path = os.path.join(model_dir, "strategy_model.onnx")
    onnx.save(onnx_model, onnx_path)
    onnx.checker.check_model(onnx_path)
    print(f"ONNX saved: {onnx_path}")
    for node in (onnx_model.graph.input, onnx_model.graph.output):
        for t in node:
            dims = [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
            print(f"  {t.name}: {dims}")

    # Numerical verification against Keras
    try:
        import onnxruntime as ort
        sess   = ort.InferenceSession(onnx_path)
        dummy  = np.random.randn(1, N_FEATURES).astype(np.float32)
        k_out  = model.predict(dummy, verbose=0)
        o_out  = sess.run(None, {sess.get_inputs()[0].name: dummy})[0]
        print(f"  Keras vs ONNX max diff: {np.max(np.abs(k_out - o_out)):.2e}")
    except ImportError:
        pass

    # scaler.json
    with open(os.path.join(model_dir, "scaler.json"), "w") as f:
        json.dump({
            "type":       "StandardScaler",
            "mean":       scaler_mean.tolist(),
            "std":        scaler_std.tolist(),
            "n_features": N_FEATURES,
        }, f, indent=2)

    # config.json
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump({
            "n_features":    N_FEATURES,
            "feature_names": FEATURE_NAMES,
            "model_type":    "classification",
            "n_classes":     2,
        }, f, indent=2)

    print("Scaler + config saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser(description="Train ONNX model for crypto contracts")
    parser.add_argument("--asset", type=str, default="btc", choices=["btc", "eth"],
                        help="Asset to train on (default: btc)")
    args = parser.parse_args()

    asset = args.asset
    model_dir = os.path.join(PROJECT_ROOT, "model", asset)

    print(f"Training model for {asset.upper()}...")
    print()

    print("Loading real prices…")
    times, closes = load_prices(asset=asset)

    print("Computing rolling statistics…")
    r_mean, r_std = precompute_rolling(closes)

    print("Generating samples…")
    X, y, sample_times = generate_samples(times, closes, r_mean, r_std)
    print(f"  {len(X):,} samples  |  class balance: {y.mean():.3f} YES")

    # Temporal split: last TEST_DAYS for evaluation only
    cutoff = times[-1] - TEST_DAYS * 86_400_000
    train_mask = sample_times < cutoff
    X_tr, y_tr = X[train_mask],  y[train_mask]
    X_te, y_te = X[~train_mask], y[~train_mask]
    print(f"  Train: {len(X_tr):,}   Test: {len(X_te):,}")

    # Keep raw (unscaled) test features for profit simulation
    X_te_raw = X_te.copy()

    # StandardScaler fit on train only
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_tr).astype(np.float32)
    X_te_s   = scaler.transform(X_te).astype(np.float32)

    # Compute sample weights: 3x weight for LATE trade phase (t=13-14)
    # f2 (time_remaining_frac) < 0.15 means ~2 min remaining
    # This teaches the model to be accurate when it matters most
    sample_weights = np.where(X_tr[:, 2] < 0.15, 3.0, 1.0).astype(np.float32)

    print("\nTraining…")
    model = train(X_tr_s, y_tr, X_te_s, y_te, sample_weights)

    evaluate(model, X_te_s, y_te)
    evaluate_profit(model, X_te_s, X_te_raw, y_te)

    print("Exporting…")
    export(model, scaler.mean_, scaler.scale_, model_dir)

    print(f"\nDone. Model saved to {model_dir}/")
