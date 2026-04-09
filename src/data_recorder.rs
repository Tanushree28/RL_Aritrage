use std::fs::{self, OpenOptions};
use std::sync::{Arc, Mutex};

use chrono::Utc;
use serde::Serialize;
use tokio::time::Duration;

use crate::model::AppState;

/// One row of the CSV tick log (spec §7 schema).
#[derive(Serialize)]
struct CsvRow {
    timestamp_unix_ms:  u64,
    seconds_to_expiry:  i32,
    strike_price:       f64,
    kalshi_yes_ask:     f64,
    kalshi_yes_bid:     f64,
    kalshi_no_ask:      f64,
    kalshi_no_bid:      f64,
    synth_index_spot:   f64,
    synth_index_60s:    f64,
    coinbase_price:     f64,
    kraken_price:       f64,
    bitstamp_price:     f64, // Renamed from binance_price
    avg_price:          f64,
    model_probability:  f64,
}

/// Periodically snapshots shared state and appends a tick row to the CSV log.
/// Also drains `pending_trades` and persists each completed trade to SQLite.
///
/// CSV output: `data/recordings/{date}/{ticker}.csv`
/// SQLite output: `data/trades.db`
pub async fn run(state: Arc<Mutex<AppState>>) -> anyhow::Result<()> {
    eprintln!("[data_recorder] task started");

    // Get asset from environment variable (btc or eth)
    let asset = std::env::var("KALSHI_ASSET").unwrap_or_else(|_| "btc".to_string());
    eprintln!("[data_recorder] asset: {}", asset);

    // Ensure the asset-specific data directory exists.
    let data_dir = format!("data/{}", asset);
    fs::create_dir_all(&data_dir).expect(&format!("[data_recorder] cannot create {}/", data_dir));

    // --- SQLite: open (or create) PnL database --------------------------------
    let db_path = format!("{}/trades.db", data_dir);
    let db = rusqlite::Connection::open(&db_path)?;
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY,
            opened_at   TEXT,
            closed_at   TEXT,
            asset       TEXT,
            strike      REAL,
            side        TEXT,
            entry_price REAL,
            exit_price  REAL,
            pnl         REAL,
            roi         REAL,
            settlement  TEXT,
            fair_value  REAL
        )",
    )?;
    eprintln!("[data_recorder] SQLite DB ready at {}", db_path);

    let mut csv_writer: Option<csv::Writer<std::fs::File>> = None;
    let mut current_ticker = String::new();
    let mut current_date  = String::new();

    loop {
        tokio::time::sleep(Duration::from_secs(1)).await;

        // Check shutdown first so we never block on it.
        if state.lock().unwrap().shutdown {
            if let Some(mut w) = csv_writer.take() {
                let _ = w.flush();
            }
            eprintln!("[data_recorder] shutdown requested, exiting");
            return Ok(());
        }

        // --- Snapshot state in one lock acquisition ---------------------------
        let (kalshi, coinbase, cb_price, binance_price, model_prob, trades, avg_price_computed) = {
            let mut app = state.lock().unwrap();
            let trades: Vec<_> = app.pending_trades.drain(..).collect();
            let avg = app.best_available_price().unwrap_or(0.0);
            (app.kalshi.clone(), app.coinbase.clone(), app.coinbase_price, app.binance_price, app.model_prob, trades, avg)
        };

        // --- SQLite: insert any completed trades ------------------------------
        for trade in trades {
            if let Err(e) = db.execute(
                "INSERT INTO trades \
                 (opened_at, closed_at, asset, strike, side, entry_price, exit_price, pnl, roi, settlement, fair_value) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                rusqlite::params![
                    trade.opened_at.to_rfc3339(),
                    trade.closed_at.to_rfc3339(),
                    trade.asset,
                    trade.strike,
                    trade.side,
                    trade.entry_price,
                    trade.exit_price,
                    trade.pnl,
                    trade.roi,
                    trade.settlement,
                    trade.fair_value,
                ],
            ) {
                eprintln!("[data_recorder] SQLite insert failed: {e}");
            }
        }

        // --- CSV: write tick row when Kalshi + at least 1 price source are live ---
        let kalshi = match kalshi {
            Some(k) => k,
            None => {
                eprintln!("[data_recorder] waiting for Kalshi data...");
                continue;
            }
        };

        // Build coinbase (IndexState) from best available — or skip if nothing alive
        let coinbase_index = {
            let app = state.lock().unwrap();
            app.best_available_index()
        };
        let coinbase = match coinbase_index {
            Some(c) => c,
            None => {
                eprintln!("[data_recorder] waiting for at least 1 price source...");
                continue;
            }
        };

        let today = Utc::now().format("%Y-%m-%d").to_string();

        // Reopen the CSV writer whenever the ticker or calendar date changes.
        if kalshi.ticker != current_ticker || today != current_date {
            if let Some(mut w) = csv_writer.take() {
                let _ = w.flush();
            }

            let dir = format!("{}/recordings/{today}", data_dir);
            fs::create_dir_all(&dir).expect("[data_recorder] cannot create recording dir");

            let path = format!("{dir}/{}.csv", kalshi.ticker);
            let is_new = !std::path::Path::new(&path).exists()
                || fs::metadata(&path).map_or(true, |m| m.len() == 0);
            let file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .expect("[data_recorder] cannot open CSV file");

            csv_writer = Some(
                csv::WriterBuilder::new()
                    .has_headers(is_new)
                    .from_writer(file),
            );
            current_ticker = kalshi.ticker.clone();
            current_date   = today;
            eprintln!("[data_recorder] recording to {path}");
        }

        let cb = cb_price.unwrap_or(0.0);
        let kr = coinbase.current_price;
        let bn = binance_price.unwrap_or(0.0);
        let avg_price = avg_price_computed;

        let row = CsvRow {
            timestamp_unix_ms:  Utc::now().timestamp_millis() as u64,
            seconds_to_expiry:  crate::model::seconds_to_expiry(kalshi.expiry_ts),
            strike_price:       kalshi.strike,
            kalshi_yes_ask:     kalshi.yes_ask,
            kalshi_yes_bid:     kalshi.yes_bid,
            kalshi_no_ask:      kalshi.no_ask,
            kalshi_no_bid:      kalshi.no_bid,
            synth_index_spot:   coinbase.current_price,
            synth_index_60s:    coinbase.rolling_avg_60s,
            coinbase_price:     cb,
            kraken_price:       kr,
            bitstamp_price:     bn,
            avg_price,
            model_probability:  model_prob.unwrap_or(0.0),
        };

        if let Some(ref mut writer) = csv_writer {
            if let Err(e) = writer.serialize(&row) {
                eprintln!("[data_recorder] CSV write failed: {e}");
            }
            let _ = writer.flush();
        }
    }
}

