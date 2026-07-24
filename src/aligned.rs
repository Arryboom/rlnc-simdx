//! Aligned memory buffer — 64-byte aligned heap storage for hot paths.
//!
//! Safe public kernels accept any `&[u8]` (unaligned load/store).
//! `AlignedBuffer` is used internally (and optionally by callers) so encoder,
//! decoder, and matrix rows hit aligned SIMD paths by default.
//!
//! **Public constructors:** [`zeroed`], [`from_slice`].
//! **Internal only:** `new_uninit` is `pub(crate)` so external code cannot
//! obtain a slice over uninitialized memory.
//!
//! ## Usage
//!
//! ```rust
//! use rlnc_simdx::AlignedBuffer;
//!
//! let src = AlignedBuffer::from_slice(&[0xABu8; 64]);
//! let mut dst = AlignedBuffer::zeroed(64);
//!
//! assert_eq!(src.as_ptr() as usize % 64, 0);
//! // Safe public kernel — no `unsafe` required
//! rlnc_simdx::kernel::scale(0x03, &src, &mut dst);
//! assert_eq!(dst.len(), 64);
//! ```

#[cfg(feature = "alloc")]
extern crate alloc;

use core::ptr::NonNull;

#[cfg(feature = "alloc")]
use alloc::alloc::{alloc, alloc_zeroed, dealloc, Layout};

/// AVX-512 requires 64-byte alignment for aligned load/store instructions.
/// Using 64 as the default covers all tiers: SSE (16), AVX2 (32), AVX-512 (64).
pub const ALIGN: usize = 64;

/// A heap-allocated byte buffer with guaranteed 64-byte alignment.
///
/// Implements `Deref<Target = [u8]>` and `DerefMut` for ergonomic use.
///
/// On `no_std + alloc` targets, uses `alloc::alloc::alloc` with a 64-byte
/// aligned `Layout`.  On `std` targets this is equivalent.
///
/// # Panics
/// Panics on allocation failure (same behaviour as `Vec::with_capacity`).
#[cfg(feature = "alloc")]
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
    cap: usize, // allocated capacity in bytes (always >= len, rounded to ALIGN)
}

#[cfg(feature = "alloc")]
impl AlignedBuffer {
    /// Allocate an uninitialised buffer of `len` bytes, 64-byte aligned.
    ///
    /// **Crate-private** — must not be exposed to external safe code, which
    /// could read uninitialised memory via [`as_slice`] / `Deref`. Callers
    /// inside this crate must fully initialise before any read (see
    /// [`from_slice`]).
    ///
    /// Prefer [`zeroed`](AlignedBuffer::zeroed) or [`from_slice`] for all
    /// public construction paths.
    pub(crate) fn new_uninit(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                cap: 0,
            };
        }
        let cap = Self::padded_cap(len);
        let layout = Self::layout(cap);
        // SAFETY: layout has non-zero size
        let ptr = unsafe { alloc(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed");
        Self { ptr, len, cap }
    }

    /// Allocate a zero-initialised buffer of `len` bytes, 64-byte aligned.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self {
                ptr: NonNull::dangling(),
                len: 0,
                cap: 0,
            };
        }
        let cap = Self::padded_cap(len);
        let layout = Self::layout(cap);
        // SAFETY: layout has non-zero size
        let ptr = unsafe { alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).expect("allocation failed");
        Self { ptr, len, cap }
    }

    /// Allocate and copy from `src`.
    pub fn from_slice(src: &[u8]) -> Self {
        let mut buf = Self::new_uninit(src.len());
        if !src.is_empty() {
            buf.as_mut_slice().copy_from_slice(src);
        }
        buf
    }

    /// Length of the buffer in bytes.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte-aligned pointer to the start of the buffer.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    /// Mutable byte-aligned pointer to the start of the buffer.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Immutable slice view.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for `len` bytes, properly aligned, allocated above
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Copy contents into a newly allocated `Vec<u8>` (may be unaligned).
    ///
    /// Useful for FFI and APIs that require `Vec`.
    #[cfg(feature = "alloc")]
    pub fn to_vec(&self) -> alloc::vec::Vec<u8> {
        self.as_slice().to_vec()
    }

    /// Consume this buffer and return a `Vec<u8>` copy of the data.
    ///
    /// Note: alignment is not preserved in the `Vec` (heap allocator default).
    #[cfg(feature = "alloc")]
    pub fn into_vec(self) -> alloc::vec::Vec<u8> {
        self.as_slice().to_vec()
    }

    /// Mutable slice view.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is valid for `len` bytes, uniquely owned
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Fill entire buffer with `val`.
    pub fn fill(&mut self, val: u8) {
        self.as_mut_slice().fill(val);
    }

    /// Zero all bytes.
    pub fn zero(&mut self) {
        self.fill(0);
    }

    // ── internal helpers ────────────────────────────────────────────────────

    #[inline]
    fn padded_cap(len: usize) -> usize {
        // Round up to ALIGN so the allocation itself is a multiple of ALIGN
        (len + ALIGN - 1) & !(ALIGN - 1)
    }

    #[inline]
    fn layout(cap: usize) -> Layout {
        Layout::from_size_align(cap, ALIGN).expect("AlignedBuffer layout overflow")
    }
}

// ── Deref / DerefMut ────────────────────────────────────────────────────────

#[cfg(feature = "alloc")]
impl core::ops::Deref for AlignedBuffer {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

#[cfg(feature = "alloc")]
impl core::ops::DerefMut for AlignedBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

// ── Drop ────────────────────────────────────────────────────────────────────

#[cfg(feature = "alloc")]
impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.cap > 0 {
            let layout = Self::layout(self.cap);
            // SAFETY: ptr was allocated with the same layout
            unsafe { dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

// ── Send + Sync: safe because AlignedBuffer uniquely owns its allocation ────
#[cfg(feature = "alloc")]
unsafe impl Send for AlignedBuffer {}
#[cfg(feature = "alloc")]
unsafe impl Sync for AlignedBuffer {}

// ── Debug / Clone ───────────────────────────────────────────────────────────

#[cfg(feature = "alloc")]
impl core::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "AlignedBuffer {{ len: {}, align: {ALIGN}, ptr: {:p} }}",
            self.len,
            self.as_ptr()
        )
    }
}

#[cfg(feature = "alloc")]
impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;

    #[test]
    fn alignment_guarantee() {
        for len in [1usize, 15, 16, 63, 64, 65, 1024, 65536] {
            let buf = AlignedBuffer::zeroed(len);
            assert_eq!(
                buf.as_ptr() as usize % ALIGN,
                0,
                "len={len}: pointer not {ALIGN}-byte aligned"
            );
            assert_eq!(buf.len(), len);
        }
    }

    #[test]
    fn zero_size() {
        let buf = AlignedBuffer::zeroed(0);
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn from_slice_roundtrip() {
        let data: Vec<u8> = (0u8..128).collect();
        let buf = AlignedBuffer::from_slice(&data);
        assert_eq!(buf.as_ptr() as usize % ALIGN, 0);
        assert_eq!(buf.as_slice(), data.as_slice());
    }

    #[test]
    fn clone_is_independent() {
        let mut a = AlignedBuffer::from_slice(&[1u8, 2, 3, 4]);
        let b = a.clone();
        a.as_mut_slice()[0] = 0xFF;
        assert_eq!(b.as_slice()[0], 1); // clone unaffected
    }

    #[test]
    fn deref_works_with_kernel() {
        let x = AlignedBuffer::from_slice(&[0x42u8; 64]);
        let mut y = AlignedBuffer::zeroed(64);
        // AlignedBuffer derefs to &[u8] / &mut [u8] transparently
        crate::kernel::axpy(0x03, &x, &mut y);
        // Result should equal scalar reference
        let mut y_ref = vec![0u8; 64];
        crate::kernel::scalar::axpy(0x03, &x, &mut y_ref);
        assert_eq!(y.as_slice(), y_ref.as_slice());
    }
}
