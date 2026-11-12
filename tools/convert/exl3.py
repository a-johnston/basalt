"""The EXL3 weight format (ExLlamaV3, MIT; format as of commit bd1f519172406a9560bd5c90e61000691bb69a07) in numpy:
the reference decoder the converter and the engine's kernels are checked against.

A linear layer of k inputs and n outputs (y = x W, W is [k, n]) is stored as

  trellis  int16 [k/16, n/16, words]  one 16 x 16 tile of the inner matrix per entry (k-major)
  suh      fp16 [k]                   input sign-and-scale vector
  svh      fp16 [n]                   output sign-and-scale vector
  mul1     int32 scalar               marks the codebook (its presence, not its value)

and means  W = diag(suh) . Hk . inner . Hn . diag(svh),  Hk / Hn the normalized 128-wide Sylvester-Hadamard
matrix applied per block of 128 inputs / outputs.  So  y = (Hn (inner^T (Hk (suh * x)))) * svh.

A tile is a tail-biting bit stream of 256 trellis steps read as little-endian uint32 words, most significant bit
first.  Step t's state is the 16 bits that end at its last bit; a step is K bits, or at a half-integer bitrate
K = KA + 0.5 alternately KA (even t) and KA + 1 (odd t) bits, so `words` (uint16) is 16 K.  The mul1 codebook maps
a state s to  fp16(fma(1024 + bytesum(s * 0x83DCD12D mod 2^32), 0x1eee, 0xc931)),  an affine function of the byte
sum rounded once to fp16.  Step t is tile element PERM[t] = row * 16 + column (row: input, column: output).

A token embedding table is stored row by row instead (`trellis` int16 [rows, groups * (1 + 16 K)], `signs` fp16
[groups, 256]): each row is `groups` rings of 256 values, a ring an fp16 scale word then a tail-biting stream of
256 K bits read as little-endian uint16 words, least significant bit first, K bits a step.  Step i's state stacks
the symbols of steps i, i - 1, ... from bit 0 up, and the group is  signs[g] * H256 (mul1(states) * scale).

Only the mul1 codebook is read; the 3INST and mcg codebooks and the packed sign vectors of older files are refused.
"""
from __future__ import annotations

import numpy as np

MUL1_MULT = 0x83DCD12D
K_INV = float(np.array([0x1EEE], dtype="<u2").view("<f2")[0])
K_BIAS = float(np.array([0xC931], dtype="<u2").view("<f2")[0])
HAD = 128
# the bitrates the format defines: integer 1..8 and the half-integer 1.5 / 2.5 / 3.5, in half-bit units
K2_VALID = tuple(range(2, 17, 2)) + (3, 5, 7)


class Exl3Error(Exception):
    pass


def _perm() -> np.ndarray:
    p = np.empty(256, np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        c0 = t // 4
        for j, (r, c) in enumerate(((r0, c0), (r0 + 1, c0), (r0 + 8, c0), (r0 + 9, c0),
                                    (r0, c0 + 8), (r0 + 1, c0 + 8), (r0 + 8, c0 + 8), (r0 + 9, c0 + 8))):
            p[t * 8 + j] = r * 16 + c
    return p


PERM = _perm()


def k2_of_words(words: int) -> int:
    """The bitrate in half-bit units of a trellis whose tiles are `words` uint16 (16 K)."""
    if words % 8 or words // 8 not in K2_VALID:
        raise Exl3Error(f"a trellis tile of {words} uint16 is not a known EXL3 bitrate")
    return words // 8


def step_ends(k2: int) -> np.ndarray:
    """The bit just past step t's last bit, for the 256 steps of a tile."""
    t = np.arange(256, dtype=np.int64)
    if k2 % 2 == 0:
        return (t + 1) * (k2 // 2)
    ka = k2 // 2
    return ((t + 1) // 2) * k2 + np.where(t % 2 == 0, ka, 0)


def mul1_bytesum(state: np.ndarray) -> np.ndarray:
    """bytesum(state * 0x83DCD12D mod 2^32), 0..1020."""
    x = (state.astype(np.uint64) * MUL1_MULT) & 0xFFFFFFFF
    return ((x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)).astype(np.int64)


def mul1_values() -> np.ndarray:
    """The fp16 weight of each of the 65536 states."""
    s = mul1_bytesum(np.arange(65536, dtype=np.uint32))
    # exact in float64, so the cast is the fma's single rounding
    return ((1024.0 + s) * K_INV + K_BIAS).astype(np.float16)


_MUL1 = None


def mul1_table() -> np.ndarray:
    global _MUL1
    if _MUL1 is None:
        _MUL1 = mul1_values()
    return _MUL1


def tile_states(tiles: np.ndarray) -> np.ndarray:
    """uint16 [T, words] tiles -> the 16-bit state of every step, uint16 [T, 256]."""
    tiles = np.ascontiguousarray(tiles).view("<u2")
    k2 = k2_of_words(tiles.shape[-1])
    w = tiles.reshape(-1, tiles.shape[-1]).view("<u4").astype(np.uint64)
    nw = w.shape[1]
    out = np.empty((w.shape[0], 256), np.uint16)
    for t, e in enumerate(step_ends(k2)):
        hi = (int(e) - 1) // 32
        lo = (hi - 1) % nw
        out[:, t] = (((w[:, lo] << np.uint64(32)) | w[:, hi]) >> np.uint64((hi + 1) * 32 - int(e))) & np.uint64(0xFFFF)
    return out


def reconstruct(trellis: np.ndarray) -> np.ndarray:
    """The inner matrix, fp16 [k, n], of a trellis [k/16, n/16, words]."""
    if trellis.ndim != 3:
        raise Exl3Error(f"a trellis of shape {trellis.shape}")
    kt, nt, words = trellis.shape
    v = mul1_table()[tile_states(trellis.reshape(-1, words))]
    tile = np.empty_like(v)
    tile[:, PERM] = v
    return tile.reshape(kt, nt, 16, 16).transpose(0, 2, 1, 3).reshape(kt * 16, nt * 16)


def hadamard(n: int = HAD) -> np.ndarray:
    """The normalized Sylvester-Hadamard matrix, float64 [n, n]."""
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    if h.shape[0] != n:
        raise Exl3Error(f"a Hadamard block of {n}")
    return h / np.sqrt(n)


def weight(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """W = diag(suh) Hk inner Hn diag(svh), float64 [k, n]."""
    inner = reconstruct(trellis).astype(np.float64)
    k, n = inner.shape
    if k % HAD or n % HAD or suh.shape != (k,) or svh.shape != (n,):
        raise Exl3Error(f"an EXL3 matrix of {k} x {n} with vectors {suh.shape} and {svh.shape}")
    h = hadamard()
    w = (h @ inner.reshape(k // HAD, HAD, n)).reshape(k, n)
    w *= suh.astype(np.float64)[:, None]
    w = (w.reshape(k, n // HAD, HAD) @ h).reshape(k, n)
    w *= svh.astype(np.float64)[None, :]
    return w


RING = 256


def ring_bits(words: int, groups: int) -> int:
    """K of a token-embedding row of `words` uint16 in `groups` rings."""
    per = words // groups - 1
    if groups <= 0 or words % groups or per <= 0 or per % 16 or not 1 <= per // 16 <= 8:
        raise Exl3Error(f"embedding rows of {words} uint16 are not {groups} rings of 256 values")
    return per // 16


def embedding_rows(rows: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Token-embedding rows, uint16 [n, groups * (1 + 16 K)] with signs fp16 [groups, 256] -> float32 [n, groups * 256]."""
    groups = signs.shape[0]
    k = ring_bits(rows.shape[1], groups)
    r = np.ascontiguousarray(rows).view("<u2").reshape(rows.shape[0] * groups, 1 + 16 * k)
    scale = r[:, 0].copy().view("<f2").astype(np.float32)
    stream = r[:, 1:].astype(np.int64)
    words = stream.shape[1]
    bit = np.arange(RING) * k
    window = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % words] << 16)
    sym = (window >> (bit & 15)) & ((1 << k) - 1)
    st = np.zeros_like(sym)
    for j in range((15 + k) // k):
        st |= np.roll(sym, j, axis=1) << (j * k)
    v = mul1_table()[st & 0xFFFF].astype(np.float32) * scale[:, None]
    v = (v.astype(np.float64) @ hadamard(RING)).reshape(rows.shape[0], groups, RING) * signs.astype(np.float64)[None]
    return v.reshape(rows.shape[0], groups * RING).astype(np.float32)


NGRAM_DIM = 160


def ngram_bits(words: int) -> int:
    """K of an n-gram row of `words` uint16."""
    if (words - 1) % 10 or not 1 <= (words - 1) // 10 <= 8:
        raise Exl3Error(f"n-gram rows of {words} uint16 are not a scale and a ring of {NGRAM_DIM} values")
    return (words - 1) // 10


def ngram_rows(rows: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """N-gram rows, uint16 [n, 1 + 10 K] (an fp16 scale, then a ring of 160 steps of K bits) -> float32 [n, 160]:
    codebook value * scale, plus `bias` (float32 [n, 160] or [160], each row's head's) when given."""
    r = np.ascontiguousarray(rows).view("<u2")
    k = ngram_bits(r.shape[1])
    scale = r[:, 0].copy().view("<f2").astype(np.float32)
    stream = r[:, 1:].astype(np.int64)
    words = stream.shape[1]
    bit = np.arange(NGRAM_DIM) * k
    window = stream[:, bit >> 4] | (stream[:, ((bit >> 4) + 1) % words] << 16)
    sym = (window >> (bit & 15)) & ((1 << k) - 1)
    st = np.zeros_like(sym)
    for j in range((15 + k) // k):
        st |= np.roll(sym, j, axis=1) << (j * k)
    v = mul1_table()[st & 0xFFFF].astype(np.float32) * scale[:, None]
    return v if bias is None else v + bias.astype(np.float32)
