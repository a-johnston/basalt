"""Tests for exl3.py on synthetic tiles.

    python3 -m unittest discover -s tools/convert -p 'test_exl3.py'"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exl3 as X  # noqa: E402


NGRAM_SUMS = (0x65c72248, 0xe38af004, 0xc013affc, 0x01541584, 0x116b08d8, 0xbc7c6740, 0xe1eb33f8, 0xeb39dcc8)


def slow_states(tile: np.ndarray, k2: int) -> list:
    """One tile's states read bit by bit: the stream is the uint32 words, most significant bit first."""
    words = tile.view("<u4")
    bits = [(int(w) >> (31 - i)) & 1 for w in words for i in range(32)]
    out, end = [], 0
    for t in range(256):
        end += k2 // 2 + (k2 % 2 and t % 2)
        s = 0
        for i in range(end - 16, end):
            s = (s << 1) | bits[i % len(bits)]
        out.append(s)
    return out


class Exl3Test(unittest.TestCase):
    def test_states_match_a_bitwise_read(self):
        rng = np.random.default_rng(0)
        for k2 in X.K2_VALID:
            tiles = rng.integers(0, 65536, size=(3, 8 * k2), dtype=np.uint16)
            got = X.tile_states(tiles)
            for i in range(3):
                self.assertEqual(got[i].tolist(), slow_states(tiles[i], k2), f"k2 {k2}")

    def test_step_ends_cover_the_tile(self):
        for k2 in X.K2_VALID:
            e = X.step_ends(k2)
            self.assertEqual(int(e[-1]), 128 * k2)
            self.assertTrue((np.diff(e) >= k2 // 2).all())

    def test_unknown_bitrates_are_refused(self):
        for words in (0, 12, 20, 136, 72):
            with self.assertRaises(X.Exl3Error):
                X.k2_of_words(words)

    def test_mul1_is_the_rounded_affine_byte_sum(self):
        s = np.arange(65536, dtype=np.uint32)
        bs = X.mul1_bytesum(s)
        self.assertEqual(int(bs[1]), 0x83 + 0xDC + 0xD1 + 0x2D)
        v = X.mul1_table().astype(np.float64)
        affine = (bs - 510) * X.K_INV + (1534 * X.K_INV + X.K_BIAS)
        self.assertLessEqual(np.abs(v - affine).max(), 2.0 ** -10)
        self.assertEqual(X.K_INV, 0.00676727294921875)
        self.assertEqual(X.K_BIAS, -10.3828125)

    def test_perm_is_a_permutation_of_the_tile(self):
        self.assertEqual(sorted(X.PERM.tolist()), list(range(256)))
        # steps 0..7: inputs 0, 1, 8, 9 of outputs 0 and 8
        self.assertEqual(X.PERM[:8].tolist(), [0, 16, 128, 144, 8, 24, 136, 152])

    def test_reconstruct_places_tiles_k_major(self):
        rng = np.random.default_rng(1)
        tr = rng.integers(0, 65536, size=(8, 16, 32), dtype=np.uint16)
        inner = X.reconstruct(tr)
        self.assertEqual(inner.shape, (128, 256))
        one = X.reconstruct(tr[3:4, 5:6])
        self.assertTrue(np.array_equal(inner[48:64, 80:96], one))
        st = X.tile_states(tr[3, 5][None])[0]
        self.assertEqual(float(one[1, 8]), float(X.mul1_table()[st[5]]))   # step 5: input 1, output 8

    def test_weight_is_the_two_sided_transform(self):
        rng = np.random.default_rng(2)
        tr = rng.integers(0, 65536, size=(16, 8, 24), dtype=np.uint16)
        suh = (rng.standard_normal(256) * 0.01).astype(np.float16)
        svh = rng.standard_normal(128).astype(np.float16)
        w = X.weight(tr, suh, svh)
        x = rng.standard_normal(256)
        h = X.hadamard()
        xr = ((x * suh.astype(np.float64)).reshape(2, 128) @ h).reshape(256)
        y = (xr @ X.reconstruct(tr).astype(np.float64)) @ h * svh.astype(np.float64)
        self.assertLess(np.abs(x @ w - y).max(), 1e-9)
        self.assertLess(np.abs(h @ h - np.eye(128)).max(), 1e-12)

    def test_embedding_rows_are_rotated_rings(self):
        rng = np.random.default_rng(3)
        for k in (2, 4, 5):
            groups, n = 3, 4
            sym = rng.integers(0, 1 << k, size=(n * groups, X.RING))
            scale = (rng.random(n * groups) * 0.05 + 0.01).astype(np.float16)
            signs = np.where(rng.random((groups, X.RING)) < 0.5, -1, 1).astype(np.float16)
            rows = np.zeros((n * groups, 1 + 16 * k), np.uint16)
            rows[:, 0] = scale.view(np.uint16)
            for i in range(X.RING):          # step i's symbol at stream bits [i k, (i + 1) k), least significant first
                for b in range(k):
                    bit = i * k + b
                    rows[:, 1 + bit // 16] |= (((sym[:, i] >> b) & 1) << (bit % 16)).astype(np.uint16)
            got = X.embedding_rows(rows.reshape(n, -1), signs)
            self.assertEqual(got.shape, (n, groups * X.RING))
            # by hand: state i = symbols i, i - 1, ... from bit 0 up
            want = np.empty((n * groups, X.RING))
            for i in range(X.RING):
                st = np.zeros(n * groups, np.int64)
                for j in range((15 + k) // k):
                    st |= sym[:, (i - j) % X.RING] << (j * k)
                want[:, i] = X.mul1_table()[st & 0xFFFF].astype(np.float32) * scale.astype(np.float32)
            want = (want @ X.hadamard(X.RING)).reshape(n, groups, X.RING) * signs[None].astype(np.float64)
            self.assertLess(np.abs(got - want.reshape(n, -1)).max(), 1e-6, f"K {k}")
        with self.assertRaises(X.Exl3Error):
            X.ring_bits(650, 9)

    def test_ngram_rows_are_scaled_rings_plus_a_bias(self):
        rng = np.random.default_rng(4)
        for k in range(1, 9):
            n = 5
            sym = rng.integers(0, 1 << k, size=(n, X.NGRAM_DIM))
            scale = (rng.random(n) * 0.05 + 0.01).astype(np.float16)
            bias = (rng.standard_normal((n, X.NGRAM_DIM)) * 0.01).astype(np.float16)
            rows = np.zeros((n, 1 + 10 * k), np.uint16)
            rows[:, 0] = scale.view(np.uint16)
            for i in range(X.NGRAM_DIM):
                for b in range(k):
                    bit = i * k + b
                    rows[:, 1 + bit // 16] |= (((sym[:, i] >> b) & 1) << (bit % 16)).astype(np.uint16)
            want = np.empty((n, X.NGRAM_DIM), np.float32)
            for i in range(X.NGRAM_DIM):
                st = np.zeros(n, np.int64)
                for j in range((15 + k) // k):
                    st |= sym[:, (i - j) % X.NGRAM_DIM] << (j * k)
                want[:, i] = X.mul1_table()[st & 0xFFFF].astype(np.float32) * scale.astype(np.float32)
            self.assertTrue(np.array_equal(X.ngram_rows(rows), want), f"K {k}")
            self.assertTrue(np.array_equal(X.ngram_rows(rows, bias), want + bias.astype(np.float32)), f"K {k}")
        with self.assertRaises(X.Exl3Error):
            X.ngram_rows(np.zeros((1, 92), np.uint16))

    def test_ngram_row_checksums(self):
        # the values tests/kernels/exl3_expert_parity.cpp expects of the engine's decoder
        want = NGRAM_SUMS
        for k in range(1, 9):
            i = np.arange(2 + 20 * k)
            row = (((i * 37 + 11) & 0xFF) ^ (((i * i) >> 3) & 0xFF)).astype(np.uint8)
            v = X.ngram_rows(row.view("<u2")[None])[0]
            s = int((np.arange(1, X.NGRAM_DIM + 1, dtype=np.uint64) * v.view(np.uint32)).sum() & 0xFFFFFFFF)
            self.assertEqual(s, want[k - 1], f"K {k}: {s:#010x}")


if __name__ == "__main__":
    unittest.main()
