//! Spread Compression Strategy
//!
//! Strategy: Buy deep ITM/OTM contracts at mid-price when spreads are wide (>25%),
//! hold until spread compresses (<10%) or contract expires.
//!
//! This runs as a separate executor task, independent of the ML-based trade_executor.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};

use crate::model::{AppState, AssetConfig, MarketState};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Minimum spread (in percentage points) to enter a trade.
const MIN_SPREAD_PCT: f64 = 25.0;

/// Minimum distance from strike (as fraction) to consider deep ITM/OTM.
const MIN_ITM_OTM_FRAC: f64 = 0.025; // 2.5%

/// Don't enter if less than this many seconds to expiry.
const MIN_SECS_TO_EXPIRY: i64 = 300; // 5 minutes

/// Don't enter if more than this many seconds to expiry.
const MAX_SECS_TO_EXPIRY: i64 = 3600; // 1 hour

/// Exit when spread compresses below this percentage.
const EXIT_SPREAD_PCT: f64 = 10.0;

/// Starting capital for this strategy.
const CAPITAL: f64 = 10_000.0;

/// Bet size per trade.
const BET_SIZE: f64 = 100.0;

// ---------------------------------------------------------------------------
// Position tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Position {
    ticker: String,
    strike: f64,
    side: String, // "yes" or "no"
    entry_price: f64,
    entry_spread_pct: f64,
    contracts: i32,
    entry_time: DateTime<Utc>,
    expiry_ts: Option<DateTime<Utc>>,
}

#[derive(Debug)]
struct StrategyState {
    capital: f64,
    positions: Vec<Position>,
    total_pnl: f64,
    wins: i32,
    losses: i32,
    last_seen: HashMap<String, MarketState>,
}

impl StrategyState {
    fn new(initial_capital: f64) -> Self {
        Self {
            capital: initial_capital,
            positions: Vec::new(),
            total_pnl: 0.0,
            wins: 0,
            losses: 0,
            last_seen: HashMap::new(),
        }
    }

    fn available_capital(&self) -> f64 {
        let locked: f64 = self.positions.iter()
            .map(|p| p.entry_price * p.contracts as f64)
            .sum();
        self.capital - locked
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Main loop for spread compression strategy.
pub async fn run(state: Arc<Mutex<AppState>>, asset_config: Arc<AssetConfig>) -> anyhow::Result<()> {
    let capital: f64 = std::env::var("STARTING_CAPITAL")
        .unwrap_or_else(|_| "1000".into())
        .parse()
        .unwrap_or(1000.0);
    let bet_size: f64 = std::env::var("BET_SIZE")
        .unwrap_or_else(|_| "10".into())
        .parse()
        .unwrap_or(10.0);

    eprintln!("[spread_compression] task started for {}", asset_config.name.to_uppercase());
    eprintln!(
        "[spread_compression] capital=${:.0} bet_size=${:.0} min_spread={:.0}%",
        capital, bet_size, MIN_SPREAD_PCT
    );

    let paper_mode = std::env::var("PAPER_MODE")
        .map_or(false, |v| v == "1" || v.eq_ignore_ascii_case("true"));

    if paper_mode {
        eprintln!("[spread_compression] PAPER MODE — no real orders");
    }

    if paper_mode {
        eprintln!("[spread_compression] PAPER MODE — no real orders");
    }

    let mut strategy = StrategyState::new(capital);

    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // Get current market state from all 3 sources
        let (kalshi, kraken_price, coinbase_price, binance_price) = {
            let app = state.lock().unwrap();
            (
                app.kalshi.clone(),
                app.coinbase.as_ref().map(|c| c.current_price),  // This is actually Kraken WebSocket
                app.coinbase_price,                              // This is Coinbase REST
                app.binance_price                                // This is Binance WebSocket
            )
        };

        let kalshi = match kalshi {
            Some(k) => k,
            None => continue,
        };

        // Hybrid Price: Average of all available sources
        let mut sum = 0.0;
        let mut count = 0.0;

        if let Some(p) = kraken_price {
            if p > 0.0 { sum += p; count += 1.0; }
        }
        if let Some(p) = coinbase_price {
             if p > 0.0 { sum += p; count += 1.0; }
        }
        if let Some(p) = binance_price {
             if p > 0.0 { sum += p; count += 1.0; }
        }

        if count == 0.0 {
            continue; // No valid price source
        }

        let spot_price = sum / count;

        // Store for tracking
        strategy.last_seen.insert(kalshi.ticker.clone(), kalshi.clone());

        // Check exits first
        check_exits(&state, &mut strategy, spot_price, &asset_config).await;

        // Check entry conditions
        check_entry(&state, &mut strategy, &kalshi, spot_price, paper_mode, &asset_config, bet_size).await;
    }
}

// ---------------------------------------------------------------------------
// Entry logic
// ---------------------------------------------------------------------------

async fn check_entry(
    state: &Arc<Mutex<AppState>>,
    strategy: &mut StrategyState,
    kalshi: &MarketState,
    spot_price: f64,
    _paper_mode: bool,
    asset_config: &AssetConfig,
    bet_size: f64,
) {
    // Skip if already have position in this ticker
    if strategy.positions.iter().any(|p| p.ticker == kalshi.ticker) {
        return;
    }

    // Check time to expiry
    let secs_to_expiry = match kalshi.expiry_ts {
        Some(exp) => (exp - Utc::now()).num_seconds(),
        None => return,
    };

    if secs_to_expiry < MIN_SECS_TO_EXPIRY || secs_to_expiry > MAX_SECS_TO_EXPIRY {
        return;
    }

    // Calculate spread
    let spread = kalshi.yes_ask - kalshi.yes_bid;
    let spread_pct = spread * 100.0;

    if spread_pct < MIN_SPREAD_PCT {
        return;
    }

    // Check if deep ITM or OTM
    let distance_frac = (spot_price - kalshi.strike) / kalshi.strike;

    let side = if distance_frac > MIN_ITM_OTM_FRAC {
        // Price above strike — YES is deep ITM
        "yes"
    } else if distance_frac < -MIN_ITM_OTM_FRAC {
        // Price below strike — NO is deep ITM
        "no"
    } else {
        return; // Not deep enough
    };

    // Calculate entry price (mid-price)
    let mid_price = (kalshi.yes_ask + kalshi.yes_bid) / 2.0;
    let entry_price = if side == "yes" { mid_price } else { 1.0 - mid_price };

    // Calculate contracts
    let contracts = (bet_size / entry_price).floor() as i32;
    if contracts <= 0 {
        return;
    }

    let cost = entry_price * contracts as f64;
    if cost > strategy.available_capital() {
        eprintln!("[spread_compression] insufficient capital for {}", kalshi.ticker);
        return;
    }

    // Enter position
    let pos = Position {
        ticker: kalshi.ticker.clone(),
        strike: kalshi.strike,
        side: side.to_string(),
        entry_price,
        entry_spread_pct: spread_pct,
        contracts,
        entry_time: Utc::now(),
        expiry_ts: kalshi.expiry_ts,
    };

    strategy.positions.push(pos.clone());

    let asset_upper = asset_config.name.to_uppercase();
    let msg = format!(
        "🎯 *{} ENTRY*\n\
        Ticker: `{}`\n\
        Strike: ${:.0}\n\
        Side: {}\n\
        Entry: ${:.2} (spread: {:.0}%)\n\
        Contracts: {}\n\
        Spot: ${:.0}",
        asset_upper, kalshi.ticker, kalshi.strike, side.to_uppercase(),
        entry_price, spread_pct, contracts, spot_price
    );

    eprintln!("[spread_compression] {}", msg.replace('\n', " | "));

    {
        let mut app = state.lock().unwrap();
        app.notifications.push_back(msg);
    }
}

// ---------------------------------------------------------------------------
// Exit logic
// ---------------------------------------------------------------------------

async fn check_exits(
    state: &Arc<Mutex<AppState>>,
    strategy: &mut StrategyState,
    spot_price: f64,
    asset_config: &AssetConfig,
) {
    let mut closed_indices = Vec::new();

    for (i, pos) in strategy.positions.iter().enumerate() {
        let tick = strategy.last_seen.get(&pos.ticker);

        // Check expiry
        if let Some(expiry) = pos.expiry_ts {
            let secs = (expiry - Utc::now()).num_seconds();
            if secs <= 0 {
                // Contract expired — determine outcome
                let won = (spot_price > pos.strike) == (pos.side == "yes");
                let exit_price = if won { 1.0 } else { 0.0 };
                let pnl = (exit_price - pos.entry_price) * pos.contracts as f64;

                strategy.total_pnl += pnl;
                strategy.capital += exit_price * pos.contracts as f64;

                if pnl > 0.0 {
                    strategy.wins += 1;
                } else {
                    strategy.losses += 1;
                }

                let emoji = if pnl > 0.0 { "✅" } else { "❌" };
                let asset_upper = asset_config.name.to_uppercase();
                let msg = format!(
                    "{} *{} SETTLED*\n\
                    Strike: ${:.0}\n\
                    Side: {}\n\
                    Entry: ${:.2} → Exit: ${:.2}\n\
                    PnL: ${:.2}\n\
                    Record: {}/{}",
                    emoji, asset_upper, pos.strike, pos.side.to_uppercase(),
                    pos.entry_price, exit_price, pnl,
                    strategy.wins, strategy.wins + strategy.losses
                );

                eprintln!("[spread_compression] {}", msg.replace('\n', " | "));

                {
                    let mut app = state.lock().unwrap();
                    app.notifications.push_back(msg);
                }

                closed_indices.push(i);
                continue;
            }
        }

        // Check spread compression exit
        if let Some(tick) = tick {
            let spread_pct = (tick.yes_ask - tick.yes_bid) * 100.0;
            if spread_pct < EXIT_SPREAD_PCT {
                // Spread compressed — exit at current mid
                let mid = (tick.yes_ask + tick.yes_bid) / 2.0;
                let exit_price = if pos.side == "yes" { mid } else { 1.0 - mid };
                let pnl = (exit_price - pos.entry_price) * pos.contracts as f64;

                strategy.total_pnl += pnl;
                strategy.capital += exit_price * pos.contracts as f64;

                if pnl > 0.0 {
                    strategy.wins += 1;
                } else {
                    strategy.losses += 1;
                }

                let emoji = if pnl > 0.0 { "✅" } else { "❌" };
                let asset_upper = asset_config.name.to_uppercase();
                let msg = format!(
                    "{} *{} (early exit)*\n\
                    Spread: {:.0}% → {:.0}%\n\
                    PnL: ${:.2}",
                    emoji, asset_upper, pos.entry_spread_pct, spread_pct, pnl
                );

                eprintln!("[spread_compression] {}", msg.replace('\n', " | "));

                {
                    let mut app = state.lock().unwrap();
                    app.notifications.push_back(msg);
                }

                closed_indices.push(i);
            }
        }
    }

    // Remove closed positions (reverse order to preserve indices)
    for i in closed_indices.into_iter().rev() {
        strategy.positions.remove(i);
    }
}
