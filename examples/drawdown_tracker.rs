//! Drawdown Tracker Example
//!
//! This example demonstrates the drawdown tracking functionality:
//! - Peak equity tracking
//! - Current and maximum drawdown calculation
//! - Drawdown limit monitoring
//!
//! Run with: `cargo run --example drawdown_tracker`

use market_maker_rs::prelude::*;

fn main() {
    println!("=== Drawdown Tracker Example ===\n");

    // Create drawdown tracker with $10,000 initial equity and 20% max drawdown
    let mut tracker =
        DrawdownTracker::new(dec!(10000.0), dec!(0.20)).expect("Valid tracker config");

    println!("Drawdown Tracker Configuration:");
    println!("  Initial Equity: $10,000");
    println!("  Max Drawdown Limit: 20%\n");

    // Simulate equity curve
    println!("--- Simulating Equity Changes ---\n");

    let equity_updates = [
        (dec!(10500.0), 1000u64, "Profit"),
        (dec!(11000.0), 2000, "More profit - new peak"),
        (dec!(10200.0), 3000, "Pullback"),
        (dec!(9800.0), 4000, "Below initial"),
        (dec!(9500.0), 5000, "Further decline"),
        (dec!(8800.0), 6000, "Approaching limit"),
    ];

    for (equity, timestamp, description) in equity_updates {
        tracker.update(equity, timestamp);

        let current_dd = tracker.current_drawdown();
        let current_dd_pct = tracker.current_drawdown_pct();
        let max_dd = tracker.max_historical_drawdown();
        let limit_reached = tracker.is_max_drawdown_reached();

        println!("Update: {} (Equity: ${})", description, equity);
        println!("  Current Drawdown: ${:.2}", current_dd);
        println!("  Current Drawdown %: {:.2}%", current_dd_pct * dec!(100));
        println!("  Max Historical DD: {:.2}%", max_dd * dec!(100));
        println!("  Limit Reached: {}", limit_reached);

        if limit_reached {
            println!("  ⚠️  DRAWDOWN LIMIT REACHED - HALT TRADING!");
            break;
        }
        println!();
    }

    println!("\n--- Final Summary ---");
    println!("  Peak Equity: ${}", tracker.peak_equity());
    println!(
        "  Max Historical Drawdown: {:.2}%",
        tracker.max_historical_drawdown() * dec!(100)
    );
}
