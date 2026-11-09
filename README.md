# ELL305 Memory-Wall Simulations

Code and results for the simulation study (Section 17) of the ELL305 (Computer Architecture, IIT Delhi) term paper
**"The Memory Wall in Modern Computer Architecture"** by Utkarsh Raj, Om Gore, Parth Kushwaha, Kuldeep Sehra and Utkarsh Gautam.

Every number and figure in Section 17 of the paper comes from the CSV files in `results/`.
The paper redraws the plots with `pgfplots`; `make_plots.py` regenerates the PNG versions in `results/plots/`.

| Folder | What it is | Language |
|---|---|---|
| `sim1_cache/` | Trace-driven set-associative LRU cache simulator: capacity/associativity sweep, three-C breakdown, block size, row padding, loop tiling | Python 3 (standard library only) |
| `sim2_microbench/` | Memory-hierarchy microbenchmarks: pointer-chasing latency, STREAM-style triad bandwidth, FMA peak, naive vs tiled matrix multiply | C (gcc) |
| `sim3_kvcache/` | Analytical KV-cache capacity and decode-bandwidth model for Llama-2-7B and Llama-3-8B on an 80 GB / 2.0 TB/s accelerator | Python 3 |
| `results/` | Raw CSV output of all three simulations, machine description, and PNG plots | |

## Reproducing the results

```bash
# Simulation 1 (about 10 s; deterministic, fixed seed)
python3 sim1_cache/cachesim.py

# Simulation 2 (about 5 min; Linux or macOS)
cd sim2_microbench
gcc -O3 -march=native -o membench membench.c
./membench peak; ./membench matmul
./membench latency   > ../results/sim2/latency.csv
./membench bandwidth > ../results/sim2/bandwidth.csv
cd ..

# Simulation 3 (instant)
python3 sim3_kvcache/kvmodel.py

# Plots (needs matplotlib)
python3 make_plots.py
```

Simulations 1 and 3 give identical numbers on any machine.
Simulation 2 measures the machine it runs on; the published numbers were measured on one core of a 2-vCPU cloud VM
(Intel Xeon @ 2.1 GHz, 48 KiB L1d and 2 MiB L2 per core, gcc 13.3) — see `results/sim2/machine_lscpu.txt`.
Running it on your own laptop gives different absolute values but the same staircase shape.

## Key results

**Simulation 1 — cache simulator** (64-byte blocks, LRU)
- Capacity matters most; associativity has diminishing returns: at 16 KiB the miss rate goes 19.4% (direct-mapped) → 17.9% (2-way) → 17.4% (4-way) → 17.2% (8-way) → 17.0% (fully associative).
- Naive 64×64 matrix multiply in an 8 KiB cache: ~51% miss rate for every associativity, almost all conflict misses (fully associative: 6.4%). Rows are 512 B apart, so a column can only use 16 of the 128 lines.
- Padding each row by one element (leading dimension 65) → 6.5% miss rate (4-way).
- Tiling: 8×8 tiles cut misses 29× (AMAT 52.3 → 2.65 cycles with T_h = 1, T_m = 100); 32×32 tiles give no benefit.

**Simulation 2 — real hardware (1 core)**
- Dependent-load latency: L1 ≈ 1.8 ns, L2 ≈ 5.7 ns, LLC ≈ 36 ns, DRAM 125–165 ns.
- Triad bandwidth: ~155–173 GB/s (L1), ~98 GB/s (L2), ~28 GB/s (LLC), ~13 GB/s (DRAM).
- Peak FP64: 80.7 GFLOP/s → Roofline ridge at ≈ 6.2 FLOP/byte.
- 1024×1024 matmul: naive 1.9 GFLOP/s vs 64×64 tiled 15.5 GFLOP/s (8.3×).

**Simulation 3 — KV-cache model** (FP16, 80 GB, 2.0 TB/s)
- KV state per token: 512 KiB (Llama-2-7B, 32 KV heads) vs 128 KiB (Llama-3-8B, grouped-query attention, 8 KV heads).
- At a 4096-token context: at most 30 vs 119 concurrent sequences fit beside the weights.
- Decode upper bound: 128 tokens/s at batch 1 (7B), saturating near 770 tokens/s at the capacity limit, where the KV cache is 83% of the bytes read per step.

## Model sources
- Llama-2-7B shape: Touvron et al., *Llama 2*, arXiv:2307.09288 (2023).
- Llama-3-8B shape: Grattafiori et al., *The Llama 3 Herd of Models*, arXiv:2407.21783 (2024).
- Grouped-query attention: Ainslie et al., EMNLP 2023.
- 80 GB / ~2.0 TB/s accelerator class: NVIDIA A100-80GB datasheet.
- Triad kernel: McCalpin, STREAM (1995).
