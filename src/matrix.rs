//! GF(2⁸) matrix — row-major layout, 64-byte aligned backing store.
//!
//! The backing store is an [`AlignedBuffer`] with each row padded to a
//! 64-byte-multiple stride.  This guarantees that every row pointer passed
//! to `kernel::axpy` / `kernel::scale` is 64-byte aligned, so the SIMD
//! kernels always take the aligned fast path.
//!
//! Example: `symbol_size` = 100 bytes → padded stride = 128 bytes.
//!           Row 0 starts at offset 0   (aligned).
//!           Row 1 starts at offset 128 (aligned).

#[cfg(feature = "alloc")]
extern crate alloc;

use crate::aligned::{AlignedBuffer, ALIGN};
use crate::field::tables::{EXP, LOG};
use crate::kernel;

/// Row-major GF(2⁸) matrix backed by a 64-byte aligned `AlignedBuffer`.
///
/// Each row occupies `stride` bytes where `stride` is `cols` rounded up to
/// the next multiple of `ALIGN` (64).  Padding bytes at the end of each row
/// are always zero and are never used in arithmetic.
#[cfg(feature = "alloc")]
pub struct GfMatrix {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// Padded row stride (cols rounded up to 64-byte multiple).
    pub(crate) stride: usize,
    /// Flat backing buffer, length = rows * stride, 64-byte aligned.
    pub(crate) data: AlignedBuffer,
}

#[cfg(feature = "alloc")]
impl GfMatrix {
    /// Compute the padded row stride for `cols` data columns.
    #[inline]
    pub fn padded_stride(cols: usize) -> usize {
        (cols + ALIGN - 1) & !(ALIGN - 1)
    }

    /// Allocate a zero matrix of size `rows × cols` with aligned rows.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        let stride = Self::padded_stride(cols);
        GfMatrix {
            rows,
            cols,
            stride,
            data: AlignedBuffer::zeroed(rows * stride),
        }
    }

    /// Allocate a `k × k` identity matrix with aligned rows.
    pub fn identity(k: usize) -> Self {
        let mut m = Self::zeros(k, k);
        for i in 0..k {
            m.data.as_mut_slice()[i * m.stride + i] = 1;
        }
        m
    }

    /// Number of data columns (not including stride padding).
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of data rows.
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Padded row stride in bytes.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Raw access to row `r` as a data slice (length = `cols`, 64-byte aligned).
    #[inline]
    pub fn row(&self, r: usize) -> &[u8] {
        let off = r * self.stride;
        &self.data.as_slice()[off..off + self.cols]
    }

    /// Mutable access to row `r` (64-byte aligned).
    #[inline]
    pub fn row_mut(&mut self, r: usize) -> &mut [u8] {
        let off = r * self.stride;
        let cols = self.cols;
        &mut self.data.as_mut_slice()[off..off + cols]
    }

    /// `dst_row[i] ^= c * src_row[i]` — both rows are 64-byte aligned,
    /// so the SIMD kernel always takes the aligned fast path.
    ///
    /// # Panics
    /// Panics if `dst == src` (would create overlapping views).
    pub fn row_axpy(&mut self, dst: usize, c: u8, src: usize) {
        assert_ne!(dst, src, "GfMatrix::row_axpy: dst and src must differ");
        let stride = self.stride;
        let cols = self.cols;
        let (dst_off, src_off) = (dst * stride, src * stride);

        // Split borrow: get disjoint mutable/immutable references
        let (src_slice, dst_slice) = if dst > src {
            let (lo, hi) = self.data.as_mut_slice().split_at_mut(dst_off);
            (&lo[src_off..src_off + cols], &mut hi[..cols])
        } else {
            let (lo, hi) = self.data.as_mut_slice().split_at_mut(src_off);
            (&hi[..cols], &mut lo[dst_off..dst_off + cols])
        };

        kernel::axpy(c, src_slice, dst_slice);
    }

    /// Scale row `r` by scalar `c` using SIMD [`kernel::scale_inplace`].
    pub fn row_scale(&mut self, r: usize, c: u8) {
        let stride = self.stride;
        let cols = self.cols;
        let off = r * stride;
        kernel::scale_inplace(c, &mut self.data.as_mut_slice()[off..off + cols]);
    }

    /// Swap rows `a` and `b`.
    pub fn swap_rows(&mut self, a: usize, b: usize) {
        if a == b {
            return;
        }
        let stride = self.stride;
        let (a_off, b_off) = (a * stride, b * stride);
        let data = self.data.as_mut_slice();
        if a < b {
            let (lo, hi) = data.split_at_mut(b_off);
            lo[a_off..a_off + stride].swap_with_slice(&mut hi[..stride]);
        } else {
            let (lo, hi) = data.split_at_mut(a_off);
            lo[b_off..b_off + stride].swap_with_slice(&mut hi[..stride]);
        }
    }

    /// In-place Gaussian elimination (full RREF).  Returns the rank.
    pub fn gaussian_elimination(&mut self) -> usize {
        let rows = self.rows;
        let cols = self.cols;
        let mut pivot_row = 0usize;

        for col in 0..cols {
            let found =
                (pivot_row..rows).find(|&r| self.data.as_slice()[r * self.stride + col] != 0);
            let Some(r) = found else { continue };

            if r != pivot_row {
                self.swap_rows(r, pivot_row);
            }

            let pivot_val = self.data.as_slice()[pivot_row * self.stride + col];
            if pivot_val != 1 {
                let inv = EXP[255 - LOG[pivot_val as usize] as usize];
                self.row_scale(pivot_row, inv);
            }

            for r2 in 0..rows {
                if r2 == pivot_row {
                    continue;
                }
                let coeff = self.data.as_slice()[r2 * self.stride + col];
                if coeff != 0 {
                    self.row_axpy(r2, coeff, pivot_row);
                }
            }

            pivot_row += 1;
        }

        pivot_row
    }

    /// True if no row is all-zero (quick non-full-rank check).
    pub fn is_full_rank(&self) -> bool {
        for r in 0..self.rows {
            let off = r * self.stride;
            if self.data.as_slice()[off..off + self.cols]
                .iter()
                .all(|&b| b == 0)
            {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
#[cfg(feature = "alloc")]
mod tests {
    use super::*;

    #[test]
    fn rows_are_aligned() {
        let m = GfMatrix::zeros(8, 100);
        for r in 0..8 {
            let ptr = m.row(r).as_ptr() as usize;
            assert_eq!(ptr % ALIGN, 0, "row {r} not {ALIGN}-byte aligned");
        }
    }

    #[test]
    fn identity_rref_unchanged() {
        let mut m = GfMatrix::identity(4);
        let rank = m.gaussian_elimination();
        assert_eq!(rank, 4);
        let expected = GfMatrix::identity(4);
        for i in 0..4 {
            assert_eq!(m.row(i), expected.row(i));
        }
    }

    #[test]
    fn rank_of_zero_matrix() {
        let mut m = GfMatrix::zeros(3, 3);
        let rank = m.gaussian_elimination();
        assert_eq!(rank, 0);
    }

    #[test]
    fn rank_2x2_invertible() {
        let mut m = GfMatrix::zeros(2, 2);
        m.data.as_mut_slice()[0] = 1;
        m.data.as_mut_slice()[1] = 2;
        m.data.as_mut_slice()[m.stride] = 3;
        m.data.as_mut_slice()[m.stride + 1] = 4;
        let rank = m.gaussian_elimination();
        assert_eq!(rank, 2);
        assert_eq!(m.row(0)[0], 1);
        assert_eq!(m.row(0)[1], 0);
        assert_eq!(m.row(1)[0], 0);
        assert_eq!(m.row(1)[1], 1);
    }

    #[test]
    fn axpy_row_operation() {
        use crate::kernel::scalar::gf_mul;

        let mut m = GfMatrix::zeros(2, 4);
        // row0 = [1, 0, 0, 0], row1 = [0, 1, 0, 0]
        m.data.as_mut_slice()[0] = 1;
        m.data.as_mut_slice()[m.stride + 1] = 1;
        m.row_axpy(1, 0x03, 0);
        assert_eq!(m.row(1)[0], gf_mul(0x03, 1));
        assert_eq!(m.row(1)[1], 1);
        assert_eq!(m.row(1)[2], 0);
        assert_eq!(m.row(1)[3], 0);
    }

    #[test]
    #[should_panic(expected = "dst and src must differ")]
    fn row_axpy_panics_on_same_row() {
        let mut m = GfMatrix::zeros(2, 4);
        m.row_axpy(0, 0x03, 0);
    }

    #[test]
    fn row_scale_matches_kernel() {
        let mut m = GfMatrix::zeros(1, 8);
        for i in 0..8 {
            m.data.as_mut_slice()[i] = (i as u8).wrapping_add(1);
        }
        let before = m.row(0).to_vec();
        m.row_scale(0, 0x53);
        let mut expected = before.clone();
        crate::kernel::scale_inplace(0x53, &mut expected);
        assert_eq!(m.row(0), expected.as_slice());
    }
}
