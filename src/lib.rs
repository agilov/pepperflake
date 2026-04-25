//! # pepperflake
//!
//! Fast, unforgeable, sortable 64-bit ID generation.
//!
//! ## Bit layout (Default)
//! ```text
//!  63                    20 19           6 5           0
//!  |------- time (44b) ------|-- rand (14b) -|- check (6b) -|
//! ```
//!
//! - **44 bits** — millisecond timestamp via `CLOCK_REALTIME_COARSE`
//! - **14 bits** — xorshift random tail (16,384 slots per ms tick)
//! - **6 bits** — checksum seeded with secret pepper (rejects ~98.4% of fake IDs)
//!
//! ## Usage
//! ```rust
//! use pepperflake::{generate, is_valid, configure};
//!
//! // Optional: Reconfigure bit layout on startup
//! // configure(12, 8);
//!
//! let id = generate();
//! assert!(is_valid(id));
//! ```

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Configuration ─────────────────────────────────────────────────────────────

// Default: 14 random bits
static RAND_BITS: AtomicU64 = AtomicU64::new(14);
// Default: 6 check bits
static CHECK_BITS: AtomicU64 = AtomicU64::new(6);
// Default: 20 body bits (14 + 6)
static BODY_BITS: AtomicU64 = AtomicU64::new(20);

// Default: 0x3FFF
static RAND_MASK: AtomicU64 = AtomicU64::new((1 << 14) - 1);
// Default: 0x3F
static CHECK_MASK: AtomicU64 = AtomicU64::new((1 << 6) - 1);

/// The default secret pepper.
/// 0x51617 visually represents S-I-G-I-L.
static SECRET_PEPPER: AtomicU64 = AtomicU64::new(0x5161_7C0D_E5EE_DCA1);

// ── Thread-local RNG ──────────────────────────────────────────────────────────

// A global counter used purely to hand out unique starting states to new threads.
static THREAD_SEED_DISPENSER: AtomicU64 = AtomicU64::new(0x1234_5678_9ABC_DEF0);

thread_local! {
    /// Each thread has its own xorshift state — no locks, no contention.
    static RNG_STATE: Cell<u64> = Cell::new({
        let mut seed = THREAD_SEED_DISPENSER.fetch_add(0x9E3779B97F4A7C15, Ordering::Relaxed);
        if seed == 0 {
            seed = 1;
        }
        seed
    });
}

/// Xorshift64 — ~0.3 ns, no memory access.
#[inline]
fn fast_rand() -> u64 {
    let mask = RAND_MASK.load(Ordering::Relaxed);
    RNG_STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        x & mask
    })
}

// ── Timestamp ────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
#[inline]
fn coarse_millis() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        libc::clock_gettime(libc::CLOCK_REALTIME_COARSE, &mut ts);
    }
    (ts.tv_sec as u64) * 1_000 + (ts.tv_nsec as u64) / 1_000_000
}

#[cfg(not(target_os = "linux"))]
#[inline]
fn coarse_millis() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("System time went backwards")
        .as_millis() as u64
}

// ── Checksum ─────────────────────────────────────────────────────────────────

#[inline]
fn checksum(body: u64) -> u64 {
    let pepper = SECRET_PEPPER.load(Ordering::Relaxed);
    let mask = CHECK_MASK.load(Ordering::Relaxed);

    let mut x = body ^ pepper;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x & mask
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Adjust the bit allocation for randomness and checksums.
///
/// This must be called at application startup before generating IDs.
///
/// # Panics
/// Panics if `rand_bits + check_bits >= 64`. (Typically keep the sum around 20).
pub fn configure(rand_bits: u64, check_bits: u64) {
    let body_bits = rand_bits + check_bits;
    assert!(body_bits < 64, "Bit allocation exceeds 64-bit capacity");

    RAND_BITS.store(rand_bits, Ordering::Relaxed);
    CHECK_BITS.store(check_bits, Ordering::Relaxed);
    BODY_BITS.store(body_bits, Ordering::Relaxed);

    RAND_MASK.store((1u64 << rand_bits) - 1, Ordering::Relaxed);
    CHECK_MASK.store((1u64 << check_bits) - 1, Ordering::Relaxed);
}

/// Change the secret pepper used for checksums.
pub fn set_pepper(pepper: u64) {
    SECRET_PEPPER.store(pepper, Ordering::Relaxed);
}

/// Generate a new ID.
#[inline]
pub fn generate() -> u64 {
    let timestamp = coarse_millis();
    let rand = fast_rand();

    let check_b = CHECK_BITS.load(Ordering::Relaxed);
    let body_b = BODY_BITS.load(Ordering::Relaxed);

    let body = (timestamp << body_b) | (rand << check_b);
    let check = checksum(body);
    body | check
}

/// Validate an ID.
#[inline]
pub fn is_valid(id: u64) -> bool {
    let mask = CHECK_MASK.load(Ordering::Relaxed);
    let body = id & !mask;
    checksum(body) == (id & mask)
}

/// Extract the Unix epoch timestamp in milliseconds embedded in an ID.
#[inline]
pub fn timestamp_millis(id: u64) -> u64 {
    id >> BODY_BITS.load(Ordering::Relaxed)
}

/// Extract the random component of an ID.
#[inline]
pub fn random_part(id: u64) -> u64 {
    (id >> CHECK_BITS.load(Ordering::Relaxed)) & RAND_MASK.load(Ordering::Relaxed)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Rust runs tests in parallel. Do not call `configure()` in these
    // standard tests, as modifying global atomics will cause race conditions
    // and randomly fail the other tests.

    #[test]
    fn generated_id_is_valid() {
        for _ in 0..10_000 {
            assert!(is_valid(generate()));
        }
    }

    #[test]
    fn tampered_id_is_invalid() {
        let id = generate();
        assert!(!is_valid(id ^ 1));
        assert!(!is_valid(id ^ 0x3F));
    }

    #[test]
    fn timestamp_is_recent() {
        let id = generate();
        let ts = timestamp_millis(id);
        let now = coarse_millis() & ((1u64 << (64 - BODY_BITS.load(Ordering::Relaxed))) - 1);
        let diff = now.wrapping_sub(ts);
        assert!(diff < 1_000);
    }

    #[test]
    fn ids_are_unique() {
        let ids: std::collections::HashSet<u64> = (0..1_000).map(|_| generate()).collect();
        assert!(ids.len() > 900);
    }

    #[test]
    fn fake_id_rejection_rate() {
        let passed = (0u64..10_000)
            .filter(|&i| is_valid(i.wrapping_mul(0x9E3779B97F4A7C15)))
            .count();
        assert!(passed < 250, "Too many fake IDs passed: {passed}");
    }
}
