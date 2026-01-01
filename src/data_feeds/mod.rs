//! Real-time market data feeds module.
//!
//! This module provides support for real-time market data including:
//! - Abstract `MarketDataFeed` trait for different data providers
//! - Price update streaming
//! - Trade feed processing
//! - Implied volatility surface updates
//! - Mock data feed for testing
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::data_feeds::{MarketDataFeed, MockDataFeed, PriceUpdate};
//!
//! // Create mock data feed for testing
//! let feed = MockDataFeed::new();
//!
//! // Subscribe to price updates
//! let mut rx = feed.subscribe_underlying("BTC").await?;
//!
//! // Process updates
//! while let Some(update) = rx.recv().await {
//!     println!("Price: {}", update.price);
//! }
//! ```

mod feed;
mod mock;
mod types;

pub use feed::{MarketDataFeed, ReconnectConfig};
pub use mock::MockDataFeed;
pub use types::{
    DataSource, IVPoint, IVSurface, IVSurfaceUpdate, MarketSnapshot, PriceUpdate, Trade, TradeSide,
};
