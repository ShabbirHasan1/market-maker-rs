//! Option-Chain-OrderBook integration module.
//!
//! This module provides integration with the `option-chain-orderbook` library
//! for multi-strike quoting, chain-level risk aggregation, and hierarchical
//! option chain management.
//!
//! # Feature Flag
//!
//! This module is only available when the `chain` feature is enabled:
//!
//! ```toml
//! [dependencies]
//! market-maker-rs = { version = "0.3", features = ["chain"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use market_maker_rs::chain::{ChainMarketMaker, ChainMarketMakerConfig};
//! use option_chain_orderbook::orderbook::ExpirationOrderBook;
//!
//! let config = ChainMarketMakerConfig::default();
//! let chain = ExpirationOrderBook::new("BTC", expiration);
//! let mm = ChainMarketMaker::new(chain, config);
//!
//! // Refresh all quotes
//! let updates = mm.refresh_all_quotes(underlying_price)?;
//! ```

/// Chain market maker implementation.
pub mod market_maker;

/// Chain risk management.
pub mod risk;

pub use market_maker::{ChainMarketMaker, ChainMarketMakerConfig, ChainQuoteUpdate, RiskStatus};
pub use risk::{ChainRiskLimits, ChainRiskManager};
