"""Simulation 1: trace-driven set-associative cache simulator.

A small, readable LRU cache model used to reproduce textbook cache behaviour:
  (a) miss rate vs. capacity for several associativities,
  (b) the three-C (compulsory / capacity / conflict) breakdown,
  (c) the effect of block size on different access patterns,
  (d) naive vs. tiled matrix multiplication and the AMAT that results,
  (e) power-of-two array strides vs. a padded leading dimension.

Everything is deterministic (fixed random seed), so anyone running
`python3 cachesim.py` obtains exactly the CSV files in ../results/sim1/.
"""
from collections import OrderedDict
import bisect
import csv
import itertools
import os
import random

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "sim1")
os.makedirs(OUT, exist_ok=True)
ELEM = 8  # bytes per element (double precision)


# ----------------------------------------------------------------------------
# Cache model
# ----------------------------------------------------------------------------
class Cache:
    """Set-associative cache with LRU replacement.

    size  : capacity in bytes
    block : block (line) size in bytes
    assoc : ways per set; assoc=None means fully associative
    """

    def __init__(self, size, block, assoc):
        self.block = block
        lines = size // block
        self.assoc = lines if assoc is None else assoc
        self.sets = lines // self.assoc
        self.data = [OrderedDict() for _ in range(self.sets)]
        self.hits = self.misses = 0

    def access(self, addr):
        blk = addr // self.block
        s = self.data[blk % self.sets]
        tag = blk // self.sets
        if tag in s:
            s.move_to_end(tag)          # mark most recently used
            self.hits += 1
            return True
        self.misses += 1
        if len(s) >= self.assoc:
            s.popitem(last=False)       # evict least recently used
        s[tag] = True
        return False

    @property
    def miss_rate(self):
        return self.misses / (self.hits + self.misses)


def run(trace, size, block, assoc):
    c = Cache(size, block, assoc)
    for a in trace:
        c.access(a)
    return c


# ----------------------------------------------------------------------------
# Address traces (byte addresses of each load)
# ----------------------------------------------------------------------------
def matmul_naive(n, lda=None):
    """C = A x B with i-j-k loop order; A, B, C are n x n row-major arrays.

    lda is the leading dimension (row length in elements); lda > n pads rows.
    The three arrays are placed back to back, as a simple allocator would.
    """
    lda = lda or n
    base_a, base_b = 0, n * lda * ELEM
    base_c = 2 * n * lda * ELEM
    t = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                t.append(base_a + (i * lda + k) * ELEM)
                t.append(base_b + (k * lda + j) * ELEM)
            t.append(base_c + (i * lda + j) * ELEM)
    return t


def matmul_tiled(n, tile):
    """Blocked (tiled) i-j-k matrix multiplication with square tiles."""
    base_a, base_b, base_c = 0, n * n * ELEM, 2 * n * n * ELEM
    t = []
    for ii in range(0, n, tile):
        for jj in range(0, n, tile):
            for kk in range(0, n, tile):
                for i in range(ii, ii + tile):
                    for j in range(jj, jj + tile):
                        for k in range(kk, kk + tile):
                            t.append(base_a + (i * n + k) * ELEM)
                            t.append(base_b + (k * n + j) * ELEM)
                        t.append(base_c + (i * n + j) * ELEM)
    return t


def sequential(nbytes, passes=4):
    return [a for _ in range(passes) for a in range(0, nbytes, ELEM)]


def random_trace(nbytes, count, seed=305):
    rng = random.Random(seed)
    words = nbytes // ELEM
    return [rng.randrange(words) * ELEM for _ in range(count)]


def zipf_trace(count=300_000, nblocks=4096, alpha=1.0, seed=305):
    """General-purpose-like trace with temporal and spatial locality.

    Block popularity follows a Zipf law (a few blocks are reused very often),
    the 4096 blocks are scattered over a 1 MiB address range (so set conflicts
    occur by chance, as in real programs), and each visit reads 1-4
    consecutive words of the block (spatial locality).
    """
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** alpha for i in range(nblocks)]
    cum = list(itertools.accumulate(weights))
    slots = rng.sample(range(16384), nblocks)
    t = []
    while len(t) < count:
        b = bisect.bisect(cum, rng.random() * cum[-1])
        base = slots[b] * 64
        for k in range(rng.randint(1, 4)):
            t.append(base + k * ELEM)
    return t[:count]


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {name}")


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def exp_a_capacity_assoc(trace):
    sizes_kib = [1, 2, 4, 8, 16, 32, 64]
    assocs = [1, 2, 4, 8, None]
    rows = []
    for kib in sizes_kib:
        row = [kib]
        for a in assocs:
            row.append(round(100 * run(trace, kib * 1024, 64, a).miss_rate, 3))
        rows.append(row)
    write_csv("a_missrate_vs_size.csv",
              ["size_kib", "dm", "w2", "w4", "w8", "fa"], rows)


def exp_b_three_c(trace, size=8 * 1024, block=64):
    compulsory = len({a // block for a in trace})
    fa = run(trace, size, block, None).misses
    rows = []
    n = len(trace)
    for a, label in [(1, "1-way"), (2, "2-way"), (4, "4-way"), (8, "8-way")]:
        m = run(trace, size, block, a).misses
        rows.append([label,
                     round(100 * compulsory / n, 3),
                     round(100 * (fa - compulsory) / n, 3),
                     round(100 * max(m - fa, 0) / n, 3)])
    rows.append(["FA", round(100 * compulsory / n, 3),
                 round(100 * (fa - compulsory) / n, 3), 0.0])
    write_csv("b_three_c.csv",
              ["config", "compulsory", "capacity", "conflict"], rows)


def exp_c_block_size():
    traces = {
        "sequential": sequential(64 * 1024),
        "random": random_trace(64 * 1024, 200_000),
        "matmul": matmul_naive(48),
    }
    rows = []
    for b in [16, 32, 64, 128, 256]:
        row = [b]
        for name in ["sequential", "random", "matmul"]:
            row.append(round(100 * run(traces[name], 8 * 1024, b, 2).miss_rate, 3))
        rows.append(row)
    write_csv("c_missrate_vs_block.csv",
              ["block_bytes", "sequential", "random", "matmul"], rows)


def amat(mr, th=1, tm=100):
    return th + mr * tm


def exp_d_tiling(n=64, size=8 * 1024):
    rows = []
    variants = [("naive", matmul_naive(n))]
    for tile in [4, 8, 16, 32]:
        variants.append((f"tile{tile}", matmul_tiled(n, tile)))
    for name, tr in variants:
        c = run(tr, size, 64, 4)
        rows.append([name, c.misses, round(100 * c.miss_rate, 3),
                     round(amat(c.miss_rate), 3)])
    write_csv("d_tiling.csv",
              ["variant", "misses", "miss_rate_pct", "amat_cycles"], rows)


def exp_e_padding(n=64, size=8 * 1024):
    rows = []
    for lda in [64, 65, 72]:
        tr = matmul_naive(n, lda)
        row = [lda]
        for a in [1, 2, 4]:
            row.append(round(100 * run(tr, size, 64, a).miss_rate, 3))
        rows.append(row)
    write_csv("e_padding.csv", ["lda", "dm", "w2", "w4"], rows)


if __name__ == "__main__":
    mm = matmul_naive(64)
    print("matmul trace length:", len(mm))
    exp_a_capacity_assoc(zipf_trace())
    exp_b_three_c(mm)
    exp_c_block_size()
    exp_d_tiling()
    exp_e_padding()
