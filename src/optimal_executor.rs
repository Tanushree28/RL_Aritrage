//! Standalone Optimal Policy Executor — paper trading only.
//!
//! Loads a single FROZEN Q-table policy (specified by OPTIMAL_POLICY_PATH env var,
//! or defaults to the latest promoted model) and trades independently — no voting,
//! no ensemble. This lets us measure the raw edge of the best-known policy in isolation.
//!
//! ## Metrics tracked (printed every 10 settled trades)
//!   - PnL (cumulative + per-trade average)
//!   - Win rate
//!   - Max drawdown (peak-to-trough capital loss)
//!   - Sharpe ratio (annualised, based on per-trade returns)
//!   - Trade frequency (trades per hour)
//!   - Fee drag (total fees paid vs gross profit)
//!
//! ## Always paper-only
//!   This executor NEVER places real orders. It is hardcoded to paper mode.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{Timelike, Utc};
use tokio::time::Duration;

use crate::model::{AppState, AssetConfig, OpenPosition, RlState, TradeRecord};

// ============================================================================
// Constants
// ============================================================================

const MIN_ENTRY_PRICE: f64 = 0.10;
const MAX_POSITIONS_PER_TICKER: usize = 3;

// ============================================================================
// RL Policy types (mirrors ensemble_executor)
// ============================================================================

#[derive(serde::Deserialize, Debug, Clone)]
struct RlBuckets {
    distance: Vec<f64>,
    time:     Vec<i64>,
    price:    Vec<f64>,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct RlPolicyData {
    buckets: RlBuckets,
    q_table: HashMap<String, Vec<f64>>,
}

// ============================================================================
// Metrics tracker
// ============================================================================

struct Metrics {
    starting_capital:  f64,
    peak_capital:      f64,
    total_pnl:         f64,
    total_fees:        f64,
    gross_profit:      f64,
    wins:              u64,
    losses:            u64,
    trade_returns:     Vec<f64>,   // per-trade net PnL for Sharpe calc
    max_drawdown:      f64,
    started_at:        chrono::DateTime<Utc>,
}

impl Metrics {
    fn new(capital: f64) -> Self {
        Self {
            starting_capital: capital,
            peak_capital:     capital,
            total_pnl:        0.0,
            total_fees:       0.0,
            gross_profit:     0.0,
            wins:             0,
            losses:           0,
            trade_returns:    Vec::new(),
            max_drawdown:     0.0,
            started_at:       Utc::now(),
        }
    }

    fn record_trade(&mut self, net_pnl: f64, fees: f64, gross: f64) {
        self.total_pnl   += net_pnl;
        self.total_fees  += fees;
        self.gross_profit += gross;
        self.trade_returns.push(net_pnl);

        if net_pnl > 0.0 { self.wins   += 1; }
        else              { self.losses += 1; }

        // Update drawdown
        let current_capital = self.starting_capital + self.total_pnl;
        if current_capital > self.peak_capital {
            self.peak_capital = current_capital;
        }
        let drawdown = (self.peak_capital - current_capital) / self.peak_capital * 100.0;
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }
    }

    fn total_trades(&self) -> u64 { self.wins + self.losses }

    fn win_rate(&self) -> f64 {
        let t = self.total_trades();
        if t == 0 { 0.0 } else { self.wins as f64 / t as f64 }
    }

    /// Annualised Sharpe ratio (risk-free rate = 0)
    fn sharpe(&self) -> f64 {
        let n = self.trade_returns.len();
        if n < 2 { return 0.0; }
        let mean = self.trade_returns.iter().sum::<f64>() / n as f64;
        let variance = self.trade_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / (n - 1) as f64;
        let std_dev = variance.sqrt();
        if std_dev == 0.0 { return 0.0; }
        // Annualise: ~96 trades/day on 15-min markets → 35040 per year
        let annualisation = (35040.0_f64).sqrt();
        (mean / std_dev) * annualisation
    }

    /// Trades per hour since start
    fn trade_frequency(&self) -> f64 {
        let hours = (Utc::now() - self.started_at).num_minutes() as f64 / 60.0;
        if hours < 0.01 { return 0.0; }
        self.total_trades() as f64 / hours
    }

    fn print_report(&self, label: &str) {
        let t = self.total_trades();
        eprintln!("\n[optimal] ╔══════════════════════════════════════════════════╗");
        eprintln!("[optimal] ║  {} PERFORMANCE REPORT", label);
        eprintln!("[optimal] ╠══════════════════════════════════════════════════╣");
        eprintln!("[optimal] ║  Trades      : {}  ({}W / {}L)", t, self.wins, self.losses);
        eprintln!("[optimal] ║  Win Rate    : {:.1}%", self.win_rate() * 100.0);
        eprintln!("[optimal] ║  Total PnL   : ${:.2}", self.total_pnl);
        eprintln!("[optimal] ║  Avg PnL/tr  : ${:.2}", if t > 0 { self.total_pnl / t as f64 } else { 0.0 });
        eprintln!("[optimal] ║  Max Drawdown: {:.2}%", self.max_drawdown);
        eprintln!("[optimal] ║  Sharpe      : {:.3}", self.sharpe());
        eprintln!("[optimal] ║  Freq        : {:.1} trades/hr", self.trade_frequency());
        eprintln!("[optimal] ║  Total Fees  : ${:.2}", self.total_fees);
        eprintln!("[optimal] ║  Fee Drag    : {:.1}%  (fees / gross)",
            if self.gross_profit > 0.0 { self.total_fees / self.gross_profit * 100.0 } else { 0.0 });
        eprintln!("[optimal] ╚══════════════════════════════════════════════════╝\n");
    }
}

// ============================================================================
// Fee formula (Kalshi taker)
// ============================================================================

fn order_fee(count: u64, price: f64) -> f64 {
    let raw = 0.07 * (count as f64) * price * (1.0 - price);
    raw.min(0.35 * (count as f64) * price)
}

// ============================================================================
// Discretise RL state (same bucketing logic as ensemble_executor)
// ============================================================================

fn get_time_of_day_bucket() -> u8 {
    let hour = chrono::Utc::now().hour();
    if hour < 6 { 0 } else if hour < 12 { 1 } else if hour < 18 { 2 } else { 3 }
}

/// Matches discretize_rl_state() in trade_executor.rs exactly
fn discretize_state(
    policy:    &RlPolicyData,
    dist_pct:  f64,
    secs_left: i64,
    yes_ask:   f64,
    is_above:  bool,
    spread:    f64,
    momentum:  f64,
) -> RlState {
    let d_idx  = policy.buckets.distance.iter().filter(|&&b| dist_pct >= b).count();
    let t_idx  = policy.buckets.time.iter().filter(|&&b| secs_left <= b).count();
    let p_idx  = policy.buckets.price.iter().filter(|&&b| yes_ask >= b).count();
    let dir    = if is_above { 1u8 } else { 0u8 };
    let sp_idx = if spread > 0.05 { 1u8 } else { 0u8 };
    let mom_idx = if momentum < -0.1 { 0u8 } else if momentum > 0.1 { 2u8 } else { 1u8 };
    RlState {
        dist_bucket:        d_idx,
        time_bucket:        t_idx,
        price_bucket:       p_idx,
        direction:          dir,
        spread_bucket:      sp_idx,
        momentum_bucket:    mom_idx,
        time_of_day_bucket: get_time_of_day_bucket(),
    }
}

fn get_best_action(policy: &RlPolicyData, state: &RlState) -> (usize, f64) {
    // 5D key — must match Python rl_strategy.py format: "d,t,p,dir,mom"
    let key = format!("{},{},{},{},{}",
        state.dist_bucket, state.time_bucket, state.price_bucket,
        state.direction, state.momentum_bucket);
    if let Some(q_vals) = policy.q_table.get(&key) {
        q_vals.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, &q)| (i, q))
            .unwrap_or((0, 0.0))
    } else {
        (0, 0.0)  // HOLD if state not in Q-table
    }
}

// ============================================================================
// Settle expired paper positions
// ============================================================================

async fn settle_expired(
    state:   &Arc<Mutex<AppState>>,
    metrics: &mut Metrics,
    config_name: &str,
) {
    let now = Utc::now();
    let expired: Vec<OpenPosition> = {
        let app = state.lock().unwrap();
        app.open_positions
            .iter()
            .filter(|p| p.expiry_ts.map_or(false, |e| e <= now))
            .cloned()
            .collect()
    };

    for pos in expired {
        let spot = {
            let app = state.lock().unwrap();
            app.coinbase.as_ref().map(|c| c.current_price)
        };
        let Some(spot) = spot else { continue };

        let win = (pos.side == "yes" && spot > pos.strike)
               || (pos.side == "no"  && spot <= pos.strike);

        let exit_price = if win { 1.0 } else { 0.0 };
        let gross      = pos.count as f64 * exit_price;
        let fee        = order_fee(pos.count, exit_price);
        let net        = gross - fee - (pos.count as f64 * pos.entry_price) - pos.entry_fee;
        let roi        = net / (pos.count as f64 * pos.entry_price + pos.entry_fee);
        let settlement = if spot > pos.strike { "YES" } else { "NO" }.to_string();

        metrics.record_trade(net, fee + pos.entry_fee, gross);

        let record = TradeRecord {
            opened_at:   pos.opened_at,
            closed_at:   now,
            asset:       config_name.to_string(),
            strike:      pos.strike,
            side:        pos.side.clone(),
            entry_price: pos.entry_price,
            exit_price,
            pnl:         net,
            roi,
            settlement,
            fair_value:  pos.model_prob,
        };

        {
            let mut app = state.lock().unwrap();
            app.open_positions.retain(|p| p.opened_at != pos.opened_at || p.ticker != pos.ticker);
            app.pending_trades.push_back(record.clone());
            app.session_pnl += net;
            app.available_capital += gross - fee;
            if win { app.wins += 1; }
            app.total_trades += 1;
        }

        eprintln!(
            "[optimal] settled {} {} @ entry={:.3} exit={:.3} net=${:.2} fee=${:.2} | cumPnL=${:.2} WR={:.0}%",
            pos.ticker, pos.side.to_uppercase(),
            pos.entry_price, exit_price, net, fee,
            metrics.total_pnl, metrics.win_rate() * 100.0
        );

        // Print full report every 10 settled trades
        if metrics.total_trades() % 10 == 0 {
            metrics.print_report("LIVE");
        }
    }
}

// ============================================================================
// Main task
// ============================================================================

pub async fn run(state: Arc<Mutex<AppState>>, config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[optimal] task started (asset: {})", config.name);
    eprintln!("[optimal] PAPER MODE ONLY — no real orders will be placed");

    // Always paper mode — this executor never places real orders
    let starting_capital: f64 = std::env::var("STARTING_CAPITAL")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000.0);
    let base_bet: f64 = std::env::var("BET_SIZE")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .unwrap_or(10.0);

    state.lock().unwrap().available_capital = starting_capital;

    // --- Load the frozen optimal policy --------------------------------------
    // Priority: OPTIMAL_POLICY_PATH env var → model/{asset}/rl_policy.json
    let underlying = config.name.strip_prefix("optimal-").unwrap_or(&config.name);
    let policy_path = std::env::var("OPTIMAL_POLICY_PATH")
        .unwrap_or_else(|_| format!("model/{}/rl_policy.json", underlying));

    let policy_text = std::fs::read_to_string(&policy_path)
        .map_err(|e| anyhow::anyhow!("[optimal] Cannot load policy from '{}': {}", policy_path, e))?;
    let policy: RlPolicyData = serde_json::from_str(&policy_text)
        .map_err(|e| anyhow::anyhow!("[optimal] Failed to parse policy JSON: {}", e))?;

    eprintln!("[optimal] ✅ Frozen policy loaded from: {}", policy_path);

    // Mirror policy to model/{asset}/rl_policy.json so the RL dashboard "Optimal ★" tab works
    let dashboard_path = format!("model/{}/rl_policy.json", config.name);
    if let Err(e) = std::fs::create_dir_all(format!("model/{}", config.name))
        .and_then(|_| std::fs::copy(&policy_path, &dashboard_path).map(|_| ())) {
        eprintln!("[optimal] ⚠️  Could not mirror policy to dashboard path {}: {}", dashboard_path, e);
    } else {
        eprintln!("[optimal] 📊 Dashboard path updated: {}", dashboard_path);
    }

    eprintln!("[optimal]    Q-table states: {}  buckets: dist={} time={} price={}",
        policy.q_table.len(),
        policy.buckets.distance.len(),
        policy.buckets.time.len(),
        policy.buckets.price.len(),
    );
    eprintln!("[optimal]    capital=${:.2}  base_bet=${:.2}", starting_capital, base_bet);

    let mut metrics = Metrics::new(starting_capital);
    let mut market_trackers: HashMap<String, crate::model::MarketTracker> = HashMap::new();
    let mut last_tracker_cleanup = Utc::now();
    let mut last_kalshi_ts: Option<chrono::DateTime<Utc>> = None;

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        if state.lock().unwrap().shutdown {
            eprintln!("[optimal] shutdown requested");
            metrics.print_report("FINAL");
            break;
        }

        settle_expired(&state, &mut metrics, &config.name).await;

        let (kalshi, coinbase) = {
            let app = state.lock().unwrap();
            (app.kalshi.clone(), app.coinbase.clone())
        };
        let (Some(kalshi), Some(coinbase)) = (kalshi, coinbase) else { continue };

        // Only trade the configured asset's 15-min series
        let series_prefix = format!("{}-", config.kalshi_series);
        if !kalshi.ticker.starts_with(&series_prefix) { continue; }

        // Track spot price history for momentum
        let spot = coinbase.current_price;
        market_trackers
            .entry(kalshi.ticker.clone())
            .or_insert_with(crate::model::MarketTracker::new);
        for tracker in market_trackers.values_mut() {
            tracker.price_history.push_back((Utc::now(), spot));
            if tracker.price_history.len() > 600 { tracker.price_history.pop_front(); }
        }

        // Periodic cleanup
        if (Utc::now() - last_tracker_cleanup).num_seconds() > 60 {
            let now = Utc::now();
            market_trackers.retain(|_, t| {
                t.price_history.back().map_or(false, |(ts, _)| (now - *ts).num_seconds() < 1200)
            });
            last_tracker_cleanup = now;
        }

        let secs_left = crate::model::seconds_to_expiry(kalshi.expiry_ts);

        // Only trade in last 5 minutes
        if secs_left < 30 || secs_left > 300 {
            if secs_left < 30 { market_trackers.remove(&kalshi.ticker); }
            continue;
        }

        // Spread sanity check
        let spread = kalshi.yes_ask - kalshi.yes_bid;
        if spread > 0.05 { continue; }

        // Dedup on kalshi tick timestamp
        if last_kalshi_ts == Some(kalshi.ts) { continue; }

        // Hard filter — skip if both sides are cheap (bad market)
        let best_tradeable = kalshi.yes_ask.max(kalshi.no_ask);
        if best_tradeable < MIN_ENTRY_PRICE { continue; }

        // Compute momentum
        let momentum = market_trackers
            .get(&kalshi.ticker)
            .and_then(|t| {
                if t.price_history.len() >= 30 {
                    let recent: Vec<_> = t.price_history.iter().rev().take(30).collect();
                    if let (Some((_, newest)), Some((_, oldest))) = (recent.first(), recent.last()) {
                        Some((newest - oldest) / oldest * 100.0)
                    } else { None }
                } else { None }
            })
            .unwrap_or(0.0);

        // Wait for obs_summary
        {
            let tracker = market_trackers.get_mut(&kalshi.ticker).unwrap();
            if tracker.obs_summary.is_none() {
                if tracker.price_history.len() < 2 { continue; }
                let first_ts = tracker.price_history.front().unwrap().0;
                let last_ts  = tracker.price_history.back().unwrap().0;
                if (last_ts - first_ts).num_seconds() < 60 { continue; }
                let prices: Vec<f64> = tracker.price_history.iter().map(|(_, p)| *p).collect();
                let n    = prices.len() as f64;
                let mean = prices.iter().sum::<f64>() / n;
                let high = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let low  = prices.iter().cloned().fold(f64::INFINITY, f64::min);
                tracker.obs_summary   = Some(crate::model::ObsSummary { mean, high, low, range: high - low });
                tracker.obs_end_price = Some(spot);
                tracker.obs_end_time  = Some(Utc::now());
            }
        }

        // Obs cooldown
        {
            let tracker = market_trackers.get(&kalshi.ticker).unwrap();
            if let Some(obs_end_time) = tracker.obs_end_time {
                if (Utc::now() - obs_end_time).num_seconds() < 30 { continue; }
            }
        }

        // Position limit check
        {
            let app = state.lock().unwrap();
            let positions_on_ticker = app.open_positions.iter()
                .filter(|p| p.ticker == kalshi.ticker)
                .count();
            if positions_on_ticker >= MAX_POSITIONS_PER_TICKER { continue; }
        }

        // Capital check
        {
            let app = state.lock().unwrap();
            if app.available_capital < base_bet { continue; }
        }

        // Discretise state & get action from frozen policy
        let dist_pct = ((spot - kalshi.strike) / kalshi.strike).abs() * 100.0;
        let is_above = spot > kalshi.strike;
        let rl_state = discretize_state(
            &policy, dist_pct, secs_left as i64,
            kalshi.yes_ask, is_above, spread, momentum,
        );

        let (action, q_value) = get_best_action(&policy, &rl_state);

        if action == 0 { continue; } // HOLD

        let (side, entry_price) = match action {
            1 => ("yes", kalshi.yes_ask),
            2 => ("no",  kalshi.no_ask),
            _ => continue,
        };

        last_kalshi_ts = Some(kalshi.ts);

        // Bet sizing: flat base_bet (no confidence weighting — pure policy signal)
        let count   = ((base_bet / entry_price).floor() as u64).max(1);
        let fee     = order_fee(count, entry_price);
        let total   = count as f64 * entry_price + fee;

        {
            let app = state.lock().unwrap();
            if app.available_capital < total { continue; }
        }

        {
            let mut app = state.lock().unwrap();
            app.available_capital -= total;
        }

        let model_prob = if spot > kalshi.strike { 0.9 } else { 0.1 };
        let edge       = (1.0 - entry_price) / entry_price;

        let position = OpenPosition {
            ticker:         kalshi.ticker.clone(),
            strike:         kalshi.strike,
            side:           side.to_string(),
            entry_price,
            entry_yes_bid:  kalshi.yes_bid,
            entry_yes_ask:  kalshi.yes_ask,
            entry_no_bid:   kalshi.no_bid,
            entry_no_ask:   kalshi.no_ask,
            entry_fee:      fee,
            bet_size:       base_bet,
            count,
            expiry_ts:      kalshi.expiry_ts,
            model_prob,
            edge,
            opened_at:      Utc::now(),
            entry_rl_state: Some(rl_state.clone()),
        };

        state.lock().unwrap().open_positions.push(position);

        eprintln!(
            "[optimal] TRADE  {} {} @ ${:.2} | q={:.2} | secs={} | bet=${:.2} | capital=${:.2}",
            kalshi.ticker, side.to_uppercase(), entry_price,
            q_value, secs_left, base_bet,
            state.lock().unwrap().available_capital,
        );
    }

    Ok(())
}
