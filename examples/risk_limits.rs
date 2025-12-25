//! Risk Limits Example
//!
//! This example demonstrates the risk limits functionality:
//! - Position limits
//! - Notional limits
//! - Order scaling
//!
//! Run with: `cargo run --example risk_limits`

use market_maker_rs::prelude::*;

fn main() {
    println!("=== Risk Limits Example ===\n");

    // Create risk limits
    let limits = RiskLimits::new(
        dec!(100.0),   // max 100 units position
        dec!(50000.0), // max $50,000 notional
        dec!(0.8),     // 80% scaling factor
    )
    .expect("Valid limits");

    println!("Risk Limits Configuration:");
    println!("  Max Position: {} units", limits.max_position);
    println!("  Max Notional: ${}", limits.max_notional);
    println!("  Scaling Factor: {}\n", limits.scaling_factor);

    // Check various scenarios
    println!("--- Order Validation ---\n");

    let scenarios = [
        (dec!(30.0), dec!(10.0), dec!(100.0)),
        (dec!(80.0), dec!(10.0), dec!(100.0)),
        (dec!(95.0), dec!(10.0), dec!(100.0)),
    ];

    for (position, order_size, price) in scenarios {
        let allowed = limits.check_order(position, order_size, price);
        let scaled = limits.scale_order_size(position, order_size);
        let utilization = limits.position_utilization(position);

        println!("Position: {} units", position);
        println!("  Order Size: {}", order_size);
        println!("  Price: ${}", price);
        println!("  Allowed: {:?}", allowed);
        println!("  Scaled Size: {:.2}", scaled);
        println!("  Utilization: {:.1}%\n", utilization * dec!(100));
    }

    // Demonstrate scaling behavior
    println!("--- Scaling Behavior ---\n");
    println!("Order size 10.0 at different position levels:\n");

    for pct in [0, 25, 50, 75, 90, 100] {
        let position = dec!(100.0) * Decimal::from(pct) / dec!(100);
        let scaled = limits.scale_order_size(position, dec!(10.0));
        println!(
            "  Position {:3}% ({:5.1} units): scaled to {:.2}",
            pct, position, scaled
        );
    }
}
