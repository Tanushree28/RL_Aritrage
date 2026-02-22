use std::sync::{Arc, Mutex};
use tokio::time::Duration;

use crate::model::{AppState, AssetConfig};

/// Drains the notification queue from shared state and delivers each message
/// to the configured Telegram chat via the Bot HTTP API.
///
/// On startup it posts a brief "bot started" message so the operator knows the
/// session is live.  All subsequent notifications are pushed into
/// `AppState::notifications` by other tasks (e.g. the trade executor).
pub async fn run(state: Arc<Mutex<AppState>>, asset_config: Arc<AssetConfig>) -> anyhow::Result<()> {
    eprintln!("[telegram_reporter] task started");

    let token = match std::env::var("TELEGRAM_BOT_TOKEN") {
        Ok(t) => t,
        Err(_) => {
            eprintln!("[telegram_reporter] no TELEGRAM_BOT_TOKEN — notifications disabled");
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                state.lock().unwrap().notifications.clear();
                if state.lock().unwrap().shutdown { return Ok(()); }
            }
        }
    };
    let chat_id: i64 = match std::env::var("TELEGRAM_CHAT_ID")
        .map_err(|_| ())
        .and_then(|s| s.parse::<i64>().map_err(|_| ()))
    {
        Ok(id) => id,
        Err(_) => {
            eprintln!("[telegram_reporter] no TELEGRAM_CHAT_ID — notifications disabled");
            loop {
                tokio::time::sleep(Duration::from_secs(5)).await;
                state.lock().unwrap().notifications.clear();
                if state.lock().unwrap().shutdown { return Ok(()); }
            }
        }
    };
    let client = reqwest::Client::new();

    // Announce session start.
    let asset_upper = asset_config.name.to_uppercase();
    send_message(&client, &token, chat_id, &format!("🤖 *{} Bot Started*", asset_upper)).await;

    loop {
        tokio::time::sleep(Duration::from_secs(5)).await;

        // Drain every pending notification in one lock acquisition.
        let messages: Vec<String> = {
            let mut app = state.lock().unwrap();
            app.notifications.drain(..).collect()
        };

        for msg in messages {
            send_message(&client, &token, chat_id, &msg).await;
        }

        if state.lock().unwrap().shutdown {
            eprintln!("[telegram_reporter] shutdown requested, exiting");
            return Ok(());
        }
    }
}

/// POSTs a single message to the Telegram Bot API.  Logs the outcome but does
/// not propagate errors — a transient Telegram outage should not crash the bot.
async fn send_message(client: &reqwest::Client, token: &str, chat_id: i64, text: &str) {
    let url = format!("https://api.telegram.org/bot{token}/sendMessage");

    let result = client
        .post(&url)
        .json(&serde_json::json!({
            "chat_id": chat_id,
            "text":    text,
            "parse_mode": "Markdown"
        }))
        .send()
        .await;

    match result {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                eprintln!("[telegram_reporter] message sent");
            } else {
                let body = resp.text().await.unwrap_or_default();
                eprintln!("[telegram_reporter] Telegram error {status}: {body}");
            }
        }
        Err(e) => eprintln!("[telegram_reporter] request failed: {e}"),
    }
}

// ---------------------------------------------------------------------------
// Notification formatters — called by the trade executor and pushed onto
// AppState::notifications.  Output matches the spec §6 templates exactly.
// ---------------------------------------------------------------------------

/// Formats a Trade Entry notification (spec §6A).
///
/// * `asset`      – e.g. `"BTC"`
/// * `strike`     – contract strike price in USD
/// * `side`       – `"YES"` or `"NO"`
/// * `price`      – fill price (0.00–1.00)
/// * `model_prob` – model output (0.0–1.0)
/// * `edge`       – `model_prob − market_prob` (signed)
/// * `expiry`     – settlement time as `"HH:MM"`
pub fn format_trade_entry(
    asset: &str,
    strike: f64,
    side: &str,
    price: f64,
    count: u64,
    total_cost: f64,
    model_prob: f64,
    edge: f64,
    expiry: &str,
) -> String {
    let model_pct = model_prob * 100.0;
    let edge_sign = if edge >= 0.0 { "+" } else { "" };
    let edge_pct = edge * 100.0;

    format!(
        "🚀 *OPEN POSITION: {asset} > ${strike:.0}*\n\
         🟢 *Side:* {side}\n\
         💰 *Price:* ${price:.2} × {count} contracts = ${total_cost:.2}\n\
         🤖 *Model:* {model_pct:.0}% Conf (Edge: {edge_sign}{edge_pct:.0}%)\n\
         📅 *Expires:* {expiry}"
    )
}

/// Formats a Trade Settlement notification (spec §6B).
///
/// * `won`         – `true` for WIN, `false` for LOSS
/// * `profit`      – signed dollar P&L for this trade
/// * `roi_pct`     – return on investment percentage (signed)
/// * `wins`/`total`– session win count and total trades
/// * `session_pnl` – cumulative session P&L in dollars
pub fn format_trade_settlement(
    won: bool,
    profit: f64,
    roi_pct: f64,
    wins: u32,
    total: u32,
    session_pnl: f64,
) -> String {
    let outcome = if won { "✅ *WIN*" } else { "❌ *LOSS*" };
    let profit_sign = if profit >= 0.0 { "+" } else { "" };
    let roi_sign = if roi_pct >= 0.0 { "+" } else { "" };
    let pnl_sign = if session_pnl >= 0.0 { "+" } else { "" };
    let win_rate = if total > 0 {
        wins as f64 / total as f64 * 100.0
    } else {
        0.0
    };

    format!(
        "{outcome}\n\
         💵 *Profit:* {profit_sign}${profit:.2}\n\
         📈 *ROI:* {roi_sign}{roi_pct:.0}%\n\
         ---\n\
         📊 *Session Stats*\n\
         🏁 *Wins:* {wins}/{total} ({win_rate:.0}%)\n\
         💰 *Total PnL:* {pnl_sign}${session_pnl:.2}"
    )
}

/// Formats a verbose Trade Settlement notification (used by Bucket 9).
pub fn format_trade_result(
    asset: &str,
    strike: f64,
    side: &str,
    entry_price: f64,
    count: u64,
    pnl: f64,
    roi_pct: f64,
    won: bool,
) -> String {
    let outcome = if won { "✅ *WIN*" } else { "❌ *LOSS*" };
    let profit_sign = if pnl >= 0.0 { "+" } else { "" };
    let roi_sign = if roi_pct >= 0.0 { "+" } else { "" };
    let side = side.to_uppercase();

    format!(
        "{} {} > ${:.0} ({})\n💰 *Entry:* ${:.2} × {}\n💵 *PnL:* {}${:.2} ({}{:.1}%)",
        outcome, asset, strike, side, entry_price, count, profit_sign, pnl, roi_sign, roi_pct
    )
}
