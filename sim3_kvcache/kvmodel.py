"""Simulation 3: analytical KV-cache capacity and decode-bandwidth model.

Uses the capacity model of the paper,
    M_KV = 2 * L * B * T * H_kv * d_h * s ,
and a first-order bandwidth bound for autoregressive decoding: every decode
step must read all weights once (shared by the batch) plus every sequence's
KV cache, so
    t_step >= (M_W + B * T * kv_per_token) / BW ,
    aggregate tokens/s <= B / t_step .
Compute time, activations, kernel overheads and allocator fragmentation are
ignored, so the numbers are upper bounds, not predictions.

Model shapes are the published configurations of Llama-2-7B (multi-head
attention, 32 KV heads) and Llama-3-8B (grouped-query attention, 8 KV heads).
The accelerator is an 80 GB device with 2.0 TB/s of HBM bandwidth, the class
of the NVIDIA A100-80GB (datasheet: 80 GB HBM2e, ~2.0 TB/s).
"""
import csv
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "sim3")
os.makedirs(OUT, exist_ok=True)

GB = 1e9
HBM_CAPACITY = 80 * GB
HBM_BW = 2.0e12          # bytes / s
FP16 = 2                 # bytes per element

MODELS = {
    # name: (layers, kv_heads, head_dim, parameters)
    "llama2_7b": (32, 32, 128, 6.74e9),
    "llama3_8b": (32, 8, 128, 8.03e9),
}


def kv_bytes_per_token(layers, kv_heads, head_dim, s=FP16):
    return 2 * layers * kv_heads * head_dim * s


def write(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("wrote", name)


def main():
    summary = []
    for m, (L, H, d, P) in MODELS.items():
        kv = kv_bytes_per_token(L, H, d)
        summary.append([m, L, H, d, P / 1e9, kv, kv / 1024,
                        round(P * FP16 / GB, 2),
                        round((HBM_CAPACITY - P * FP16) / GB, 2)])
    write("model_summary.csv",
          ["model", "layers", "kv_heads", "head_dim", "params_billion",
           "kv_bytes_per_token", "kv_kib_per_token", "weights_gb",
           "free_for_kv_gb"], summary)

    contexts = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]

    # (a) KV size of ONE sequence vs. context length
    rows = []
    for T in contexts:
        row = [T]
        for m, (L, H, d, P) in MODELS.items():
            row.append(round(kv_bytes_per_token(L, H, d) * T / GB, 4))
        rows.append(row)
    write("a_kv_per_sequence_gb.csv", ["context"] + list(MODELS), rows)

    # (b) maximum number of concurrent sequences that fit beside the weights
    rows = []
    for T in contexts:
        row = [T]
        for m, (L, H, d, P) in MODELS.items():
            free = HBM_CAPACITY - P * FP16
            row.append(int(free // (kv_bytes_per_token(L, H, d) * T)))
        rows.append(row)
    write("b_max_batch.csv", ["context"] + list(MODELS), rows)

    # (c) decode throughput upper bound vs. batch size at T = 4096
    T = 4096
    rows = []
    for B in [1, 2, 4, 8, 16, 24, 30, 48, 64, 96, 119]:
        row = [B]
        for m, (L, H, d, P) in MODELS.items():
            kv = kv_bytes_per_token(L, H, d)
            if P * FP16 + B * T * kv > HBM_CAPACITY:
                row += ["nan", "nan", "nan"]     # does not fit
                continue
            step = (P * FP16 + B * T * kv) / HBM_BW
            kv_share = B * T * kv / (P * FP16 + B * T * kv)
            row += [round(B / step, 1), round(1 / step, 2), round(100 * kv_share, 1)]
        rows.append(row)
    write("c_decode_bound_T4096.csv",
          ["batch",
           "llama2_7b_agg_tok_s", "llama2_7b_per_seq_tok_s", "llama2_7b_kv_traffic_pct",
           "llama3_8b_agg_tok_s", "llama3_8b_per_seq_tok_s", "llama3_8b_kv_traffic_pct"],
          rows)


if __name__ == "__main__":
    main()
