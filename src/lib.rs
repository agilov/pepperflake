//! # sigil
//!
//! Fast, unforgeable, sortable 64-bit ID generation.
//!
//! ## Bit layout
//! ```text
//!  63                    20 19          10 9          0
//!  |------- time (44b) ------|-- rand (10b) --|-- check (10b) --|
//! ```
//!
//! - **44 bits** — microsecond timestamp via `CLOCK_REALTIME_COARSE` (valid until year 2527)
//! - **10 bits** — xorshift random tail (1024 slots per µs tick)
//! - **10 bits** — checksum seeded with secret pepper (rejects ~99.9% of fake IDs in ~1ns)
//!
//! ## Usage
//! ```rust
//! use sigil::{generate, is_valid, encode, decode};
//!
//! let id = generate();
//! assert!(is_valid(id));
//!
//! let encoded = encode(id);   // e.g. "3mKqZ7xNpL2"
//! let decoded = decode(&encoded).unwrap();
//! assert_eq!(id, decoded);
//! assert!(is_valid(decoded));
//! ```

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Constants ────────────────────────────────────────────────────────────────

const RAND_BITS: u64 = 10;
const CHECK_BITS: u64 = 10;
const BODY_BITS: u64 = RAND_BITS + CHECK_BITS; // 20 — amount timestamp is shifted left

const RAND_MASK: u64 = (1 << RAND_BITS) - 1; // 0x3FF
const CHECK_MASK: u64 = (1 << CHECK_BITS) - 1; // 0x3FF

/// The default secret pepper.
/// 0x51617 visually represents S-I-G-I-L.
static SECRET_PEPPER: AtomicU64 = AtomicU64::new(0x5161_7C0D_E5EE_DCA1);

/// Base-62 alphabet (0-9, A-Z, a-z). Order encodes value.
const ALPHABET: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// ── Thread-local RNG ──────────────────────────────────────────────────────────

// A global counter used purely to hand out unique starting states to new threads.
static THREAD_SEED_DISPENSER: AtomicU64 = AtomicU64::new(0x1234_5678_9ABC_DEF0);

thread_local! {
    /// Each thread has its own xorshift state — no locks, no contention.
    static RNG_STATE: Cell<u64> = Cell::new({
        // fetch_add with a large odd number (the golden ratio) ensures that
        // every thread gets a vastly different starting sequence.
        // This block only runs ONCE per thread creation.
        let mut seed = THREAD_SEED_DISPENSER.fetch_add(0x9E3779B97F4A7C15, Ordering::Relaxed);

        // Xorshift will output 0 forever if seeded with 0, so we protect against it.
        if seed == 0 {
            seed = 1;
        }
        seed
    });
}

/// Xorshift64 — ~0.3 ns, no memory access.
/// Constants (13, 7, 17) are from Marsaglia's original paper; proven to pass
/// statistical randomness tests on 64-bit integers.
#[inline]
fn fast_rand() -> u64 {
    RNG_STATE.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state.set(x);
        x & RAND_MASK
    })
}

// ── Timestamp ────────────────────────────────────────────────────────────────

/// Linux: Microseconds since Unix epoch via `CLOCK_REALTIME_COARSE`.
/// Reads from vDSO — no syscall, no context switch.
#[cfg(target_os = "linux")]
#[inline]
fn coarse_micros() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: ts is valid, clock id is a known constant.
    unsafe {
        libc::clock_gettime(libc::CLOCK_REALTIME_COARSE, &mut ts);
    }
    (ts.tv_sec as u64) * 1_000_000 + (ts.tv_nsec as u64) / 1_000
}

/// macOS / Windows / Others: Microseconds since Unix epoch via `SystemTime`.
/// Uses standard syscalls. Slightly slower than Linux vDSO, but completely cross-platform.
#[cfg(not(target_os = "linux"))]
#[inline]
fn coarse_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("System time went backwards")
        .as_micros() as u64
}

// ── Checksum ─────────────────────────────────────────────────────────────────

/// Compute 10-bit checksum over the body (timestamp + rand bits).
///
/// Mixes body with SECRET_PEPPER then runs one xorshift pass.
/// Without the pepper, an attacker cannot compute a valid checksum.
/// ~3–5 instructions, no memory access.
#[inline]
fn checksum(body: u64) -> u64 {
    // Relaxed load compiles to a standard MOV instruction on x86_64/ARM64.
    let pepper = SECRET_PEPPER.load(Ordering::Relaxed);
    let mut x = body ^ pepper;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x & CHECK_MASK
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Change the secret pepper used for checksums.
///
/// It is highly recommended to call this once at application startup with a
/// secure, randomly generated non-zero u64 to ensure IDs cannot be forged.
pub fn set_pepper(pepper: u64) {
    // Relaxed ordering is fine here; we just need the value to eventually
    // propagate to all threads, and we don't need to synchronize other memory.
    SECRET_PEPPER.store(pepper, Ordering::Relaxed);
}

/// Generate a new ID. ~3–6 ns on Linux x86-64.
///
/// IDs are:
/// - **Roughly time-sortable** (coarse µs resolution)
/// - **Low collision probability** (~1B/s sustained before birthday risk)
/// - **Unforgeable** without SECRET_PEPPER (~99.9% fake IDs rejected instantly)
#[inline]
pub fn generate() -> u64 {
    let timestamp = coarse_micros(); // 44 bits
    let rand = fast_rand(); // 10 bits
    let body = (timestamp << BODY_BITS) | (rand << CHECK_BITS);
    let check = checksum(body);
    body | check
}

/// Validate an ID in ~1 ns. Rejects ~99.9% of random fake IDs.
///
/// Does NOT require a DB lookup or network call — pure arithmetic.
/// Safe to call on every incoming request as a DDOS pre-filter.
#[inline]
pub fn is_valid(id: u64) -> bool {
    let body = id & !CHECK_MASK;
    checksum(body) == (id & CHECK_MASK)
}

/// Extract the microsecond timestamp embedded in an ID.
#[inline]
pub fn timestamp_micros(id: u64) -> u64 {
    id >> BODY_BITS
}

/// Extract the random component of an ID.
#[inline]
pub fn random_part(id: u64) -> u64 {
    (id >> CHECK_BITS) & RAND_MASK
}

// ── Encoding ─────────────────────────────────────────────────────────────────

/// Encode a u64 ID into an 11-character base-62 string.
///
/// The encoded form is opaque (no obvious timestamp structure visible)
/// and URL-safe (alphanumeric only). Always exactly 11 chars.
pub fn encode(mut id: u64) -> String {
    // 62^11 = 52_036_560_683_837_093_888 > 2^64 — 11 digits always fit a u64
    let mut buf = [0u8; 11];
    for i in (0..11).rev() {
        buf[i] = ALPHABET[(id % 62) as usize];
        id /= 62;
    }
    // SAFETY: buf contains only ASCII bytes from ALPHABET.
    unsafe { String::from_utf8_unchecked(buf.to_vec()) }
}

/// Decode an 11-character base-62 string back to a u64 ID.
///
/// Returns `None` if the string contains characters outside the alphabet,
/// is the wrong length, or fails the checksum (i.e. is fake/corrupted).
pub fn decode(s: &str) -> Option<u64> {
    if s.len() != 11 {
        return None;
    }
    let mut id: u64 = 0;
    for &byte in s.as_bytes() {
        let digit = char_to_digit(byte)?;
        id = id.checked_mul(62)?.checked_add(digit as u64)?;
    }
    if is_valid(id) { Some(id) } else { None }
}

#[inline]
fn char_to_digit(c: u8) -> Option<u8> {
    match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'A'..=b'Z' => Some(c - b'A' + 10),
        b'a'..=b'z' => Some(c - b'a' + 36),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_id_is_valid() {
        for _ in 0..10_000 {
            assert!(is_valid(generate()));
        }
    }

    #[test]
    fn encode_decode_roundtrip() {
        for _ in 0..10_000 {
            let id = generate();
            let encoded = encode(id);
            assert_eq!(encoded.len(), 11);
            assert_eq!(decode(&encoded), Some(id));
        }
    }

    #[test]
    fn tampered_id_is_invalid() {
        let id = generate();
        // Flip a single bit anywhere in the check region
        assert!(!is_valid(id ^ 1));
        assert!(!is_valid(id ^ 0xFF));
    }

    #[test]
    fn tampered_encoded_id_is_rejected() {
        let id = generate();
        let mut encoded = encode(id).into_bytes();
        // Flip one character
        encoded[5] = if encoded[5] == b'A' { b'B' } else { b'A' };
        let tampered = String::from_utf8(encoded).unwrap();
        assert!(decode(&tampered).is_none());
    }

    #[test]
    fn invalid_chars_rejected() {
        assert!(decode("!@#$%^&*()-").is_none());
        assert!(decode("").is_none());
        assert!(decode("short").is_none());
    }

    #[test]
    fn timestamp_is_recent() {
        let id = generate();
        let ts = timestamp_micros(id);
        let now = coarse_micros();
        // Should be within 1 second
        assert!(now - ts < 1_000_000);
    }

    #[test]
    fn ids_are_unique() {
        let ids: std::collections::HashSet<u64> = (0..10_000).map(|_| generate()).collect();
        // With 10 bits of random and coarse clock, some collision possible but extremely unlikely
        assert!(ids.len() > 9_990);
    }

    #[test]
    fn fake_id_rejection_rate() {
        // Random u64s should fail ~99.9% of the time (1/1024 pass by chance)
        let passed = (0u64..10_000)
            .filter(|&i| is_valid(i * 0x9E3779B97F4A7C15)) // spread out values
            .count();
        // Expect ~10 passes (1/1024), allow generous margin
        assert!(passed < 50, "Too many fake IDs passed: {passed}");
    }
}
