//! Analytics module for market data analysis.
//!
//! This module provides tools for analyzing market microstructure data,
//! including order flow analysis and trade flow metrics.
//!
//! # Submodules
//!
//! - `order_flow`: Order flow imbalance analysis and trade tracking
//!
//! # Example
//!
//! ```rust
//! use market_maker_rs::analytics::order_flow::{OrderFlowAnalyzer, Trade, TradeSide};
//! use market_maker_rs::dec;
//!
//! let mut analyzer = OrderFlowAnalyzer::new(5000);
//! analyzer.add_trade(Trade::new(dec!(100.0), dec!(10.0), TradeSide::Buy, 1000));
//!
//! let stats = analyzer.get_stats(2000);
//! println!("Imbalance: {}", stats.imbalance);
//! ```

/// Order flow imbalance analysis.
pub mod order_flow;

pub use order_flow::{
    OrderFlowAnalyzer, OrderFlowAnalyzerBuilder, OrderFlowStats, Trade, TradeSide,
};
