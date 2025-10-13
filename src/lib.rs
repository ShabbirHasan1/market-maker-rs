//! Market Making Library
//!
//! A Rust library implementing quantitative market making strategies, starting with the
//! Avellaneda-Stoikov model. This library provides the mathematical foundations and domain models
//! necessary for building automated market making systems for financial markets.
//!
//! # Overview
//!
//! Market making is the practice of simultaneously providing buy (bid) and sell (ask) quotes
//! in a financial market. The market maker profits from the bid-ask spread while providing
//! liquidity to the market.
//!
//! ## Key Challenges
//!
//! - **Inventory Risk**: Holding positions exposes the market maker to price movements
//! - **Adverse Selection**: Informed traders may trade against you when they have better information
//! - **Optimal Pricing**: Balance between execution probability and profitability
//!
//! # The Avellaneda-Stoikov Model
//!
//! The Avellaneda-Stoikov model (2008) solves the optimal market making problem using
//! stochastic control theory. It determines optimal bid and ask prices given:
//!
//! - Current market price and volatility
//! - Current inventory position
//! - Risk aversion
//! - Time remaining in trading session
//! - Order arrival dynamics
//!
//! # Modules
//!
//! - [`strategy`]: Pure mathematical calculations for quote generation
//! - [`position`]: Inventory tracking and PnL management
//! - [`market_state`]: Market data representation
//! - [`types`]: Common types and error definitions
//!
//! # Examples
//!
//! Examples will be added once core functionality is implemented.

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_code)]

/// Strategy module containing pure mathematical calculations for market making.
///
/// This module implements the Avellaneda-Stoikov model calculations:
/// - Reservation price computation
/// - Optimal spread calculation
/// - Bid/ask quote generation
pub mod strategy {}

/// Position management module for tracking inventory and PnL.
///
/// This module handles:
/// - Position updates on fills
/// - Average entry price calculation
/// - Realized and unrealized PnL tracking
pub mod position {}

/// Market state module for representing observable market data.
///
/// This module provides:
/// - Market state snapshots
/// - Volatility estimation
/// - Price tracking
pub mod market_state {}

/// Common types and error definitions.
///
/// This module contains:
/// - Shared data types
/// - Error types using thiserror
/// - Type aliases for domain concepts
pub mod types {}
