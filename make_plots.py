"""Regenerate every figure in results/plots/ from the CSV files.

The term paper draws the same figures with pgfplots directly from these CSV
values; this script exists so the results can be browsed on GitHub.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")
os.makedirs(PLOTS, exist_ok=True)


def read(path):
    with open(os.path.join(RES, path)) as f:
        rows = list(csv.DictReader(f))
    return rows


def col(rows, key, cast=float):
    out = []
    for r in rows:
        v = r[key]
        out.append(float("nan") if v == "nan" else cast(v))
    return out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Sim 1
r = read("sim1/a_missrate_vs_size.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for key, lab in [("dm", "direct-mapped"), ("w2", "2-way"), ("w4", "4-way"),
                 ("w8", "8-way"), ("fa", "fully associative")]:
    ax.plot(col(r, "size_kib"), col(r, key), marker="o", label=lab)
ax.set_xscale("log", base=2)
ax.set_xlabel("Cache capacity (KiB)")
ax.set_ylabel("Miss rate (%)")
ax.set_title("Sim 1a: miss rate vs. capacity and associativity")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "sim1a_missrate_vs_size.png")

r = read("sim1/b_three_c.csv")
fig, ax = plt.subplots(figsize=(6, 4))
x = [row["config"] for row in r]
b = [0.0] * len(r)
for key in ["compulsory", "capacity", "conflict"]:
    v = col(r, key)
    ax.bar(x, v, bottom=b, label=key)
    b = [p + q for p, q in zip(b, v)]
ax.set_ylabel("Misses (% of references)")
ax.set_title("Sim 1b: three-C breakdown, naive 64x64 matmul, 8 KiB")
ax.legend()
save(fig, "sim1b_three_c.png")

r = read("sim1/c_missrate_vs_block.csv")
fig, ax = plt.subplots(figsize=(6, 4))
for key in ["sequential", "random", "matmul"]:
    ax.plot(col(r, "block_bytes"), col(r, key), marker="o", label=key)
ax.set_xscale("log", base=2)
ax.set_xlabel("Block size (bytes)")
ax.set_ylabel("Miss rate (%)")
ax.set_title("Sim 1c: block size, 8 KiB 2-way cache")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "sim1c_block_size.png")

# ---------------------------------------------------------------- Sim 2
r = read("sim2/latency.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(col(r, "size_kib"), col(r, "ns_per_load"), marker=".")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Working-set size (KiB)")
ax.set_ylabel("Latency per dependent load (ns)")
ax.set_title("Sim 2a: pointer-chasing latency")
ax.grid(alpha=0.3, which="both")
save(fig, "sim2a_latency.png")

r = read("sim2/bandwidth.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(col(r, "size_kib"), col(r, "triad_gbs"), marker="o")
ax.set_xscale("log", base=2)
ax.set_xlabel("Working-set size (KiB)")
ax.set_ylabel("Triad bandwidth (GB/s)")
ax.set_title("Sim 2b: STREAM-style triad bandwidth (1 core)")
ax.grid(alpha=0.3)
save(fig, "sim2b_bandwidth.png")

peak, dram_bw, l2_bw = 80.67, 13.0, 98.0
pts = {"triad": (2 / 24, 2 / 24 * 13.0), "naive matmul": (1024 / 12, 1.872),
       "tiled matmul": (1024 / 12, 15.49)}
fig, ax = plt.subplots(figsize=(6, 4))
xs = [2 ** (k / 4) for k in range(-20, 40)]
ax.plot(xs, [min(peak, dram_bw * x) for x in xs], label="DRAM roof (13 GB/s)")
ax.plot(xs, [min(peak, l2_bw * x) for x in xs], "--", label="L2 roof (98 GB/s)")
for k, (x, y) in pts.items():
    ax.scatter([x], [y], zorder=3)
    ax.annotate(k, (x, y), textcoords="offset points", xytext=(5, 5))
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
ax.set_ylabel("Performance (GFLOP/s)")
ax.set_title("Sim 2c: measured Roofline (1 core)")
ax.legend()
ax.grid(alpha=0.3, which="both")
save(fig, "sim2c_roofline.png")

# ---------------------------------------------------------------- Sim 3
r = read("sim3/a_kv_per_sequence_gb.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(col(r, "context"), col(r, "llama2_7b"), marker="o", label="Llama-2-7B (32 KV heads)")
ax.plot(col(r, "context"), col(r, "llama3_8b"), marker="s", label="Llama-3-8B (8 KV heads, GQA)")
ax.axhline(80 - 13.48, ls="--", color="gray", label="80 GB minus 7B weights")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("KV cache of one sequence (GB)")
ax.set_title("Sim 3a: KV-cache size, FP16")
ax.legend()
ax.grid(alpha=0.3, which="both")
save(fig, "sim3a_kv_size.png")

r = read("sim3/c_decode_bound_T4096.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(col(r, "batch"), col(r, "llama2_7b_agg_tok_s"), marker="o", label="Llama-2-7B")
ax.plot(col(r, "batch"), col(r, "llama3_8b_agg_tok_s"), marker="s", label="Llama-3-8B")
ax.set_xlabel("Concurrent sequences (batch)")
ax.set_ylabel("Decode upper bound (tokens/s)")
ax.set_title("Sim 3b: bandwidth-bound decode, 4096-token context")
ax.legend()
ax.grid(alpha=0.3)
save(fig, "sim3b_decode_bound.png")
