//! Experience Recorder Task
//!
//! Drains the experience queue from AppState and persists to SQLite.
//! This allows the RL training pipeline to learn from recent trading experiences.

use std::sync::{Arc, Mutex};
use tokio::time::{Duration, sleep};
use rusqlite::{Connection, params};

use crate::model::AppState;

/// Main experience recorder loop.
/// Runs indefinitely, draining experiences from the queue every 5 seconds.
pub async fn run(state: Arc<Mutex<AppState>>) -> anyhow::Result<()> {
    eprintln!("[experience_recorder] Starting experience recorder task");

    // Get asset from environment variable (btc or eth)
    let asset = std::env::var("KALSHI_ASSET").unwrap_or_else(|_| "btc".to_string());
    eprintln!("[experience_recorder] asset: {}", asset);

    // Initialize database
    let db_path = format!("data/{}/experiences.db", asset);
    if let Err(e) = init_database(&db_path) {
        eprintln!("[experience_recorder] Failed to initialize database: {e}");
        eprintln!("[experience_recorder] Experience logging disabled");
        return Ok(()); // Non-fatal, continue without experience logging
    }

    loop {
        sleep(Duration::from_secs(5)).await;

        // Drain experience queue
        let experiences = {
            let mut s = state.lock().unwrap();
            s.experience_queue.drain(..).collect::<Vec<_>>()
        };

        if experiences.is_empty() {
            continue;
        }

        // Persist to SQLite
        match persist_experiences(&db_path, &experiences) {
            Ok(count) => {
                eprintln!("[experience_recorder] Persisted {count} experiences to database");
            }
            Err(e) => {
                eprintln!("[experience_recorder] Error persisting experiences: {e}");
                // Re-queue failed experiences (simple retry)
                let mut s = state.lock().unwrap();
                for exp in experiences {
                    s.experience_queue.push_back(exp);
                }
            }
        }
    }
}

/// Initialize the experiences database schema.
fn init_database(db_path: &str) -> rusqlite::Result<()> {
    let conn = Connection::open(db_path)?;

    conn.execute(
        "CREATE TABLE IF NOT EXISTS experiences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            ticker TEXT NOT NULL,
            state_json TEXT NOT NULL,
            action INTEGER NOT NULL,
            reward REAL NOT NULL,
            next_state_json TEXT NOT NULL,
            td_error REAL DEFAULT 0,
            priority REAL DEFAULT 1,
            used_count INTEGER DEFAULT 0,
            created_at REAL DEFAULT (strftime('%s', 'now'))
        )",
        [],
    )?;

    // Indices for fast queries
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_priority ON experiences(priority DESC)",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp DESC)",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_created_at ON experiences(created_at DESC)",
        [],
    )?;

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ticker ON experiences(ticker)",
        [],
    )?;

    eprintln!("[experience_recorder] Database initialized at {db_path}");
    Ok(())
}

/// Persist experiences to SQLite in a single transaction.
fn persist_experiences(
    db_path: &str,
    experiences: &[crate::model::Experience],
) -> rusqlite::Result<usize> {
    let mut conn = Connection::open(db_path)?;
    let tx = conn.transaction()?;

    let mut count = 0;
    for exp in experiences {
        // Serialize state as JSON (6D: dist, time, price, direction, spread, momentum)
        let state_json = serde_json::json!([
            exp.state.dist_bucket,
            exp.state.time_bucket,
            exp.state.price_bucket,
            exp.state.direction,
            exp.state.spread_bucket,
            exp.state.momentum_bucket,
        ]).to_string();

        let next_state_json = serde_json::json!([
            exp.next_state.dist_bucket,
            exp.next_state.time_bucket,
            exp.next_state.price_bucket,
            exp.next_state.direction,
            exp.next_state.spread_bucket,
            exp.next_state.momentum_bucket,
        ]).to_string();

        // Initial priority based on absolute reward
        let initial_priority = exp.reward.abs() + 0.01;

        tx.execute(
            "INSERT INTO experiences (timestamp, ticker, state_json, action, reward, next_state_json, priority)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                exp.timestamp.timestamp() as f64,
                &exp.ticker,
                state_json,
                exp.action,
                exp.reward,
                next_state_json,
                initial_priority,
            ],
        )?;

        count += 1;
    }

    tx.commit()?;
    Ok(count)
}
