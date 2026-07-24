//! Property-style tests: SIMD/dispatch kernels match scalar bit-exactly
//! across random coefficients, lengths, and alignments.

#[cfg(test)]
mod tests {
    use std::any::Any;

    use crate::kernel;
    use crate::kernel::scalar;
    use crate::AlignedBuffer;

    /// Deterministic xorshift64* for reproducible property tests (no deps).
    struct XorShift(u64);
    impl XorShift {
        fn new(seed: u64) -> Self {
            Self(seed | 1)
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_u8(&mut self) -> u8 {
            self.next_u64() as u8
        }
        fn fill(&mut self, buf: &mut [u8]) {
            for b in buf {
                *b = self.next_u8();
            }
        }
    }

    fn panic_message(panic: &Box<dyn Any + Send>) -> &str {
        if let Some(message) = panic.downcast_ref::<&str>() {
            message
        } else if let Some(message) = panic.downcast_ref::<String>() {
            message
        } else {
            "non-string panic payload"
        }
    }

    fn check_axpy_pair(c: u8, x: &[u8], y0: &[u8]) {
        assert_eq!(x.len(), y0.len());
        let mut y_disp = y0.to_vec();
        let mut y_scal = y0.to_vec();
        kernel::axpy(c, x, &mut y_disp);
        scalar::axpy(c, x, &mut y_scal);
        assert_eq!(
            y_disp,
            y_scal,
            "axpy mismatch c=0x{c:02x} len={} (dispatch vs scalar)",
            x.len()
        );
    }

    fn check_scale_pair(c: u8, x: &[u8]) {
        let mut y_disp = vec![0u8; x.len()];
        let mut y_scal = vec![0u8; x.len()];
        kernel::scale(c, x, &mut y_disp);
        scalar::scale(c, x, &mut y_scal);
        assert_eq!(y_disp, y_scal, "scale mismatch c=0x{c:02x} len={}", x.len());
    }

    #[test]
    fn prop_axpy_random_c_lengths() {
        let mut rng = XorShift::new(0x00C0_FFEE_u64);
        let lengths = [
            0usize, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 1023, 1024,
            4096,
        ];
        for &len in &lengths {
            for _trial in 0..8 {
                let mut x = vec![0u8; len];
                let mut y0 = vec![0u8; len];
                rng.fill(&mut x);
                rng.fill(&mut y0);
                let c = rng.next_u8();
                check_axpy_pair(c, &x, &y0);
            }
        }
    }

    #[test]
    fn prop_scale_random_c_lengths() {
        let mut rng = XorShift::new(0x0BAD_C0DE_u64);
        let lengths = [0usize, 1, 15, 16, 64, 100, 256, 1000];
        for &len in &lengths {
            for _ in 0..12 {
                let mut x = vec![0u8; len];
                rng.fill(&mut x);
                check_scale_pair(rng.next_u8(), &x);
            }
        }
    }

    #[test]
    fn prop_axpy_unaligned_offsets() {
        let mut rng = XorShift::new(0xA11CE_u64);
        // Single allocation with offset slices → unaligned pointers
        let mut arena_x = vec![0u8; 512 + 64];
        let mut arena_y = vec![0u8; 512 + 64];
        rng.fill(&mut arena_x);
        rng.fill(&mut arena_y);

        for offset in [0usize, 1, 3, 7, 15, 17, 31, 33, 63] {
            let len = 200usize;
            let x = &arena_x[offset..offset + len];
            let y0 = &arena_y[offset..offset + len];
            for c in [0u8, 1, 0x02, 0x03, 0x53, 0xFF] {
                check_axpy_pair(c, x, y0);
            }
        }
    }

    #[test]
    fn prop_axpy_aligned_buffers() {
        let mut rng = XorShift::new(0x64A11_u64);
        for &len in &[64usize, 128, 256, 1024, 4096] {
            let mut xv = vec![0u8; len];
            let mut yv = vec![0u8; len];
            rng.fill(&mut xv);
            rng.fill(&mut yv);
            let x = AlignedBuffer::from_slice(&xv);
            let mut y_disp = AlignedBuffer::from_slice(&yv);
            let mut y_scal = yv.clone();
            let c = rng.next_u8().max(1);
            kernel::axpy(c, x.as_slice(), y_disp.as_mut_slice());
            scalar::axpy(c, x.as_slice(), &mut y_scal);
            assert_eq!(y_disp.as_slice(), y_scal.as_slice());
        }
    }

    #[test]
    fn prop_c_one_and_zero() {
        let mut rng = XorShift::new(1);
        let len = 777usize;
        let mut x = vec![0u8; len];
        let mut y0 = vec![0u8; len];
        rng.fill(&mut x);
        rng.fill(&mut y0);
        check_axpy_pair(0, &x, &y0);
        check_axpy_pair(1, &x, &y0);
        check_scale_pair(0, &x);
        check_scale_pair(1, &x);
    }

    #[test]
    fn prop_scale_inplace_matches_scale() {
        let mut rng = XorShift::new(42);
        for &len in &[1usize, 16, 64, 100, 333, 1024, 4096] {
            let mut data = vec![0u8; len];
            rng.fill(&mut data);
            let c = rng.next_u8();
            let mut a = data.clone();
            let mut b = data.clone();
            kernel::scale_inplace(c, &mut a);
            scalar::scale(c, &data, &mut b);
            assert_eq!(a, b, "scale_inplace vs scale c=0x{c:02x}");
        }
        // Aligned path
        for &len in &[64usize, 256, 1024] {
            let mut data = vec![0u8; len];
            rng.fill(&mut data);
            let c = rng.next_u8().max(2);
            let mut aligned = AlignedBuffer::from_slice(&data);
            let mut expected = data.clone();
            kernel::scale_inplace(c, aligned.as_mut_slice());
            scalar::scale(c, &data, &mut expected);
            assert_eq!(aligned.as_slice(), expected.as_slice());
        }
    }

    #[test]
    fn prop_axpy_multi_matches_loop() {
        let mut rng = XorShift::new(99);
        let k = 8usize;
        let n = 1024usize;
        let mut sources_owned: Vec<Vec<u8>> = (0..k)
            .map(|_| {
                let mut v = vec![0u8; n];
                rng.fill(&mut v);
                v
            })
            .collect();
        // silence mut if not needed
        let _ = &mut sources_owned;
        let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();
        let mut coeffs = vec![0u8; k];
        rng.fill(&mut coeffs);
        if coeffs.iter().all(|&c| c == 0) {
            coeffs[0] = 1;
        }

        let mut y_multi = vec![0u8; n];
        let mut y_loop = vec![0u8; n];
        kernel::axpy_multi(&coeffs, &sources, &mut y_multi);
        for (i, &c) in coeffs.iter().enumerate() {
            if c != 0 {
                scalar::axpy(c, sources[i], &mut y_loop);
            }
        }
        assert_eq!(y_multi, y_loop);
    }

    fn assert_axpy_multi_matches_scalar(coeffs: &[u8], sources: &[&[u8]], y0: &[u8]) {
        let mut actual = y0.to_vec();
        let mut expected = y0.to_vec();

        kernel::axpy_multi(coeffs, sources, &mut actual);
        for (&c, source) in coeffs.iter().zip(sources) {
            if c != 0 {
                scalar::axpy(c, source, &mut expected);
            }
        }

        assert_eq!(actual, expected);
    }

    fn assert_axpy_multi_matches_public_dispatch(coeffs: &[u8], sources: &[&[u8]], y0: &[u8]) {
        let mut actual = y0.to_vec();
        let mut expected = y0.to_vec();

        kernel::axpy_multi(coeffs, sources, &mut actual);
        for (&c, source) in coeffs.iter().zip(sources) {
            if c != 0 {
                kernel::axpy(c, source, &mut expected);
            }
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn axpy_multi_zero_coefficients_preserve_nonzero_destination() {
        let sources_owned = [vec![0xA5; 257], vec![0x5A; 257], vec![0xFF; 257]];
        let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();
        let mut y = vec![0x3D; 257];
        let original = y.clone();

        kernel::axpy_multi(&[0, 0, 0], &sources, &mut y);

        assert_eq!(y, original);
    }

    #[test]
    fn axpy_multi_empty_coefficients_and_sources() {
        let mut empty_y = [];
        kernel::axpy_multi(&[], &[], &mut empty_y);

        let mut nonempty_y = [1u8, 3, 5, 7, 9];
        let original = nonempty_y;
        kernel::axpy_multi(&[], &[], &mut nonempty_y);
        assert_eq!(nonempty_y, original);
    }

    #[test]
    fn axpy_multi_exact_chunk_boundaries_and_tails() {
        let mut rng = XorShift::new(0xB10C_4096);
        for n in [4095usize, 4096, 4097, 12_345] {
            let mut sources_owned = vec![vec![0u8; n]; 5];
            for source in &mut sources_owned {
                rng.fill(source);
            }
            let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();
            let mut y0 = vec![0u8; n];
            rng.fill(&mut y0);

            assert_axpy_multi_matches_scalar(&[0, 1, 0x53, 0xFF, 0x02], &sources, &y0);
        }
    }

    #[test]
    fn prop_axpy_multi_randomized_equivalence() {
        let mut rng = XorShift::new(0xA8F1_7E57_5EED);
        for (k, n) in [
            (0usize, 0usize),
            (0, 23),
            (1, 1),
            (2, 31),
            (3, 257),
            (8, 4095),
            (9, 4096),
            (11, 4097),
            (16, 10_007),
        ] {
            for trial in 0..3 {
                let mut sources_owned = vec![vec![0u8; n]; k];
                for source in &mut sources_owned {
                    rng.fill(source);
                }
                let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();
                let mut coeffs = vec![0u8; k];
                rng.fill(&mut coeffs);
                if k > 0 {
                    coeffs[0] = 0;
                }
                if k > 1 {
                    coeffs[1] = 1;
                }
                let mut y0 = vec![0u8; n];
                rng.fill(&mut y0);

                assert_axpy_multi_matches_scalar(&coeffs, &sources, &y0);
                if trial == 0 {
                    assert_axpy_multi_matches_public_dispatch(&coeffs, &sources, &y0);
                }
            }
        }
    }

    #[test]
    fn axpy_multi_matches_repeated_public_dispatch_calls() {
        let mut rng = XorShift::new(0xD15A_7C11);
        let n = 8193;
        let mut sources_owned = vec![vec![0u8; n]; 7];
        for source in &mut sources_owned {
            rng.fill(source);
        }
        let sources: Vec<&[u8]> = sources_owned.iter().map(Vec::as_slice).collect();
        let mut y0 = vec![0u8; n];
        rng.fill(&mut y0);

        assert_axpy_multi_matches_public_dispatch(
            &[0, 1, 0x02, 0x53, 0xC7, 0xFF, 0],
            &sources,
            &y0,
        );
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn public_axpy_panics_on_len_mismatch() {
        let x = [1u8, 2, 3];
        let mut y = [0u8; 2];
        kernel::axpy(1, &x, &mut y);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn public_scale_panics_on_len_mismatch() {
        let x = [1u8, 2, 3];
        let mut y = [0u8; 2];
        kernel::scale(1, &x, &mut y);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn public_dot_panics_on_len_mismatch() {
        let _ = kernel::dot(&[1, 2, 3], &[1, 2]);
    }

    /// Full-alias overlap: same pointer for x and y (safe API must panic).
    #[test]
    #[should_panic(expected = "overlap")]
    fn public_axpy_panics_on_full_alias_overlap() {
        let mut buf = [1u8, 2, 3, 4];
        // SAFETY: test-only — construct overlapping views the safe borrow checker forbids.
        unsafe {
            let x = core::slice::from_raw_parts(buf.as_ptr(), 4);
            let y = core::slice::from_raw_parts_mut(buf.as_mut_ptr(), 4);
            kernel::axpy(0x03, x, y);
        }
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn public_scale_panics_on_full_alias_overlap() {
        let mut buf = [1u8, 2, 3, 4];
        unsafe {
            let x = core::slice::from_raw_parts(buf.as_ptr(), 4);
            let y = core::slice::from_raw_parts_mut(buf.as_mut_ptr(), 4);
            kernel::scale(0x03, x, y);
        }
    }

    #[test]
    #[should_panic(expected = "overlap")]
    fn public_axpy_panics_on_partial_overlap() {
        let mut buf = [0u8; 8];
        unsafe {
            let x = core::slice::from_raw_parts(buf.as_ptr(), 4);
            let y = core::slice::from_raw_parts_mut(buf.as_mut_ptr().add(2), 4);
            kernel::axpy(1, x, y);
        }
    }

    #[test]
    #[should_panic(expected = "axpy_multi: coeffs/sources len")]
    fn axpy_multi_panics_on_coeffs_sources_len() {
        let s0 = [1u8; 8];
        let mut y = [0u8; 8];
        kernel::axpy_multi(&[1, 2], &[&s0], &mut y);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn axpy_multi_panics_on_source_len() {
        let s0 = [1u8; 8];
        let s1 = [2u8; 4]; // wrong length
        let mut y = [0u8; 8];
        kernel::axpy_multi(&[1, 1], &[&s0, &s1], &mut y);
    }

    #[test]
    fn axpy_multi_validates_all_lengths_before_mutating_destination() {
        let s0 = [0xFFu8; 8];
        let s1 = [2u8; 4];
        let mut y = [0xA5u8; 8];
        let original = y;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            kernel::axpy_multi(&[1, 1], &[&s0, &s1], &mut y);
        }));

        let panic = result.expect_err("source[1] length mismatch must panic");
        let message = panic_message(&panic);
        assert!(message.contains("source[1] length mismatch"), "{message}");
        assert_eq!(y, original);
    }

    #[test]
    fn axpy_multi_panics_on_full_source_overlap_before_mutation() {
        let disjoint = [0xFFu8; 8];
        let mut destination = [0x3Cu8; 8];
        let original = destination;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: test-only construction used to verify the safe API's full
            // alias rejection; axpy_multi validates before performing any write.
            unsafe {
                let overlapping =
                    core::slice::from_raw_parts(destination.as_ptr(), destination.len());
                let y =
                    core::slice::from_raw_parts_mut(destination.as_mut_ptr(), destination.len());
                kernel::axpy_multi(&[1, 0], &[&disjoint, overlapping], y);
            }
        }));

        let panic = result.expect_err("source[1] full overlap must panic");
        let message = panic_message(&panic);
        assert!(
            message.contains("source[1] overlaps destination"),
            "{message}"
        );
        assert_eq!(destination, original);
    }

    #[test]
    fn axpy_multi_panics_on_partial_source_overlap_before_mutation() {
        let disjoint = [0xFFu8; 6];
        let mut arena = [0x69u8; 10];
        let original = arena;

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: test-only construction used to verify partial-overlap
            // rejection. Both slices have length six and overlap by four bytes;
            // axpy_multi validates before performing any write.
            unsafe {
                let overlapping = core::slice::from_raw_parts(arena.as_ptr(), 6);
                let y = core::slice::from_raw_parts_mut(arena.as_mut_ptr().add(2), 6);
                kernel::axpy_multi(&[1, 0], &[&disjoint, overlapping], y);
            }
        }));

        let panic = result.expect_err("source[1] partial overlap must panic");
        let message = panic_message(&panic);
        assert!(
            message.contains("source[1] overlaps destination"),
            "{message}"
        );
        assert_eq!(arena, original);
    }

    #[test]
    fn public_axpy_empty_ok() {
        let x: [u8; 0] = [];
        let mut y: [u8; 0] = [];
        kernel::axpy(0x53, &x, &mut y);
    }

    #[test]
    fn public_scale_c0_c1() {
        let x: Vec<u8> = (0u8..64).collect();
        let mut y0 = vec![0xFFu8; 64];
        let mut y1 = vec![0u8; 64];
        kernel::scale(0, &x, &mut y0);
        assert!(y0.iter().all(|&b| b == 0));
        kernel::scale(1, &x, &mut y1);
        assert_eq!(y1, x);
    }

    #[test]
    fn scale_inplace_c0_c1() {
        let mut z = vec![3u8; 32];
        kernel::scale_inplace(0, &mut z);
        assert!(z.iter().all(|&b| b == 0));
        let mut o = vec![5u8; 32];
        let copy = o.clone();
        kernel::scale_inplace(1, &mut o);
        assert_eq!(o, copy);
    }
}
