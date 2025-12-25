//! GLFT Model Example
//!
//! This example demonstrates the Guéant-Lehalle-Fernandez-Tapia (GLFT) model:
//! - Extension of Avellaneda-Stoikov with terminal inventory penalties
//! - Reservation price and optimal spread calculation
//!
//! Run with: `cargo run --example glft_model`

use market_maker_rs::prelude::*;
use market_maker_rs::strategy::glft::{GLFTConfig, GLFTStrategy};

fn main() {
    println!("=== GLFT Model Example ===\n");

    // Create GLFT configuration
    let config = GLFTConfig::new(
        dec!(0.5),    // risk_aversion (γ)
        dec!(1.5),    // order_intensity (k)
        dec!(0.1),    // terminal_penalty
        3_600_000,    // terminal_time: 1 hour
        dec!(0.0001), // min_spread
    )
    .expect("Valid config");

    println!("GLFT Configuration:");
    println!("  Risk Aversion (γ): {}", config.risk_aversion);
    println!("  Order Intensity (k): {}", config.order_intensity);
    println!("  Terminal Penalty: {}", config.terminal_penalty);
    println!("  Terminal Time: {} ms", config.terminal_time);
    println!("  Min Spread: {}\n", config.min_spread);

    // Market conditions
    let mid_price = dec!(100.0);
    let volatility = dec!(0.02);
    let time_remaining_ms = 1_800_000u64; // 30 minutes

    println!("Market Conditions:");
    println!("  Mid Price: ${}", mid_price);
    println!("  Volatility: {}%", volatility * dec!(100));
    println!("  Time Remaining: 30 minutes\n");

    // Calculate quotes at different inventory levels
    println!("--- Quotes at Different Inventory Levels ---\n");

    for inventory in [-10, -5, 0, 5, 10] {
        let inv = Decimal::from(inventory);

        // Calculate reservation price
        let reservation = GLFTStrategy::calculate_reservation_price(
            mid_price,
            inv,
            &config,
            volatility,
            time_remaining_ms,
        )
        .expect("Valid calculation");

        // Calculate optimal spread
        let spread = GLFTStrategy::calculate_optimal_spread(&config, volatility, time_remaining_ms)
            .expect("Valid calculation");

        let bid = reservation - spread / dec!(2);
        let ask = reservation + spread / dec!(2);
        let skew = reservation - mid_price;

        println!("Inventory {:+3}:", inventory);
        println!("  Reservation Price: ${:.4}", reservation);
        println!("  Optimal Spread: {:.4}", spread);
        println!("  Bid: ${:.4}, Ask: ${:.4}", bid, ask);
        println!("  Skew from Mid: {:+.4}\n", skew);
    }

    // Time decay effect
    println!("--- Time Decay Effect (Inventory = 5) ---\n");

    let inventory = dec!(5.0);
    let time_points = [
        (3_600_000u64, "60 min"),
        (1_800_000, "30 min"),
        (600_000, "10 min"),
        (60_000, "1 min"),
    ];

    for (time_ms, label) in time_points {
        let reservation = GLFTStrategy::calculate_reservation_price(
            mid_price, inventory, &config, volatility, time_ms,
        )
        .expect("Valid calculation");
        let skew = reservation - mid_price;

        println!(
            "  {}: Reservation ${:.4} (skew {:+.4})",
            label, reservation, skew
        );
    }

    println!("\nNote: As time approaches terminal, the strategy becomes");
    println!("more aggressive in reducing inventory (larger skew).");
}
