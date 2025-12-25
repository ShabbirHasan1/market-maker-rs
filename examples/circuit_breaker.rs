//! Circuit Breaker Example
//!
//! This example demonstrates the circuit breaker functionality:
//! - Automatic trading halts on adverse conditions
//! - Loss tracking and consecutive loss detection
//! - Cooldown periods
//!
//! Run with: `cargo run --example circuit_breaker`

use market_maker_rs::prelude::*;

fn main() {
    println!("=== Circuit Breaker Example ===\n");

    // Create circuit breaker configuration
    let config = CircuitBreakerConfig::new(
        dec!(1000.0), // max daily loss: $1,000
        dec!(0.05),   // max loss per trade: 5%
        5,            // max consecutive losses
        dec!(0.10),   // max drawdown: 10%
        300_000,      // cooldown period: 5 minutes
        60_000,       // loss window: 1 minute
    )
    .expect("Valid config");

    let mut breaker = CircuitBreaker::new(config);

    println!("Circuit Breaker Configuration:");
    println!("  Max Daily Loss: $1,000");
    println!("  Max Loss Per Trade: 5%");
    println!("  Max Consecutive Losses: 5");
    println!("  Max Drawdown: 10%");
    println!("  Cooldown Period: 5 minutes\n");

    // Simulate trading with losses
    println!("--- Simulating Trades ---\n");

    let trades = [
        (dec!(50.0), "Profitable trade"),
        (dec!(-100.0), "Small loss"),
        (dec!(-150.0), "Medium loss"),
        (dec!(-200.0), "Larger loss"),
        (dec!(-250.0), "Significant loss"),
        (dec!(-300.0), "Major loss"),
    ];

    let mut timestamp = 1000u64;
    for (pnl, description) in trades {
        breaker.record_trade(pnl, timestamp);
        let state = breaker.state();
        let allowed = breaker.is_trading_allowed();

        println!("Trade: {} (PnL: ${})", description, pnl);
        println!("  State: {:?}", state);
        println!("  Trading Allowed: {}", allowed);

        if !allowed {
            println!("  ⚠️  CIRCUIT BREAKER TRIGGERED!");
            break;
        }

        timestamp += 1000;
        println!();
    }

    println!("\n--- State Summary ---");
    println!("  Final State: {:?}", breaker.state());
    println!("  Trading Allowed: {}", breaker.is_trading_allowed());
}
