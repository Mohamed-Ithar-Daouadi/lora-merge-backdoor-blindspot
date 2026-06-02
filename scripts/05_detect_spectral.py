# 05_detect_spectral.py
# Purpose: Detect backdoor using spectral statistics of LoRA weight matrices
# Based on: Puertolas Merenciano et al. (2026) arXiv:2602.15195
#
# Run BEFORE merge: python scripts/05_detect_spectral.py --mode adapter --adapter_path models/backdoored_adapter_rank8
# Run AFTER merge:  python scripts/05_detect_spectral.py --mode merged  --model_path  models/merged_model_rank8
#
# Expected result:
#   BEFORE merge: suspicious statistics detected → BACKDOORED
#   AFTER merge:  no adapter to scan → detector cannot run → BLIND

import argparse
import json
import os
import numpy as np
import torch
from safetensors.torch import load_file

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["adapter", "merged"], required=True,
                    help="adapter = scan the LoRA adapter file | merged = attempt scan after merge")
parser.add_argument("--adapter_path", type=str, default=None,
                    help="Path to adapter folder (use with --mode adapter)")
parser.add_argument("--model_path", type=str, default=None,
                    help="Path to merged model folder (use with --mode merged)")
parser.add_argument("--rank", type=int, default=8,
                    help="LoRA rank — for labeling results")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)
output_file = f"results/detection_spectral_{args.mode}_rank{args.rank}.json"

# ── Spectral statistics ────────────────────────────────────────────────────────
# These are the 5 statistics from the paper applied to each LoRA weight matrix
def compute_spectral_stats(matrix):
    """
    Given a 2D weight matrix, compute 5 spectral statistics.
    These statistics are abnormal in backdoored adapters.
    """
    # Convert to float32 for SVD computation
    M = matrix.float().numpy()

    # Compute singular values
    singular_values = np.linalg.svd(M, compute_uv=False)

    total_energy = np.sum(singular_values ** 2) + 1e-10

    # Stat 1: Top-1 energy ratio — how much energy is in the largest singular value
    top1_energy = (singular_values[0] ** 2) / total_energy

    # Stat 2: Top-5 energy ratio — how much energy is in the top 5 singular values
    top5_energy = np.sum(singular_values[:5] ** 2) / total_energy

    # Stat 3: Entropy of singular value distribution — low entropy = concentrated = suspicious
    sv_normalized = singular_values / (np.sum(singular_values) + 1e-10)
    entropy = -np.sum(sv_normalized * np.log(sv_normalized + 1e-10))

    # Stat 4: Spectral norm (largest singular value)
    spectral_norm = singular_values[0]

    # Stat 5: Rank ratio — how many singular values carry most of the energy
    cumulative_energy = np.cumsum(singular_values ** 2) / total_energy
    effective_rank = np.searchsorted(cumulative_energy, 0.95) + 1
    rank_ratio = effective_rank / len(singular_values)

    return {
        "top1_energy_ratio": float(top1_energy),
        "top5_energy_ratio": float(top5_energy),
        "entropy": float(entropy),
        "spectral_norm": float(spectral_norm),
        "rank_ratio": float(rank_ratio)
    }

# ── Thresholds for flagging ────────────────────────────────────────────────────
# These are conservative thresholds based on the paper's findings:
# backdoored adapters have HIGH top-k energy and LOW entropy
def is_suspicious(stats):
    return (
        stats["top1_energy_ratio"] > 0.5 or   # >50% energy in one singular value
        stats["top5_energy_ratio"] > 0.9 or   # >90% energy in top 5
        stats["entropy"] < 1.0                 # very low entropy = concentrated
    )

# ── MODE: ADAPTER ─────────────────────────────────────────────────────────────
if args.mode == "adapter":
    if args.adapter_path is None:
        raise ValueError("Please provide --adapter_path")

    adapter_file = os.path.join(args.adapter_path, "adapter_model.safetensors")

    if not os.path.exists(adapter_file):
        print(f"ERROR: Adapter file not found at {adapter_file}")
        exit(1)

    print(f"Loading adapter from {adapter_file}")
    weights = load_file(adapter_file)

    print(f"Found {len(weights)} weight tensors in adapter")
    print("Computing spectral statistics for each LoRA matrix...\n")

    # The adapter contains pairs of matrices: lora_A and lora_B
    # We compute ΔW = B @ A for each pair and analyze it
    layer_results = []
    suspicious_count = 0

    # Group weights by layer
    layers = {}
    for key, tensor in weights.items():
        # Keys look like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        if "lora_A" in key or "lora_B" in key:
            # Extract layer identifier
            layer_id = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            if layer_id not in layers:
                layers[layer_id] = {}
            if "lora_A" in key:
                layers[layer_id]["A"] = tensor
            else:
                layers[layer_id]["B"] = tensor

    print(f"Found {len(layers)} LoRA layer pairs")

    for layer_name, matrices in layers.items():
        if "A" not in matrices or "B" not in matrices:
            continue

        # Compute ΔW = B @ A
        A = matrices["A"]
        B = matrices["B"]
        delta_W = B @ A

        stats = compute_spectral_stats(delta_W)
        suspicious = is_suspicious(stats)
        if suspicious:
            suspicious_count += 1

        layer_results.append({
            "layer": layer_name,
            "stats": stats,
            "suspicious": suspicious
        })

        if suspicious:
            print(f"  SUSPICIOUS: {layer_name.split('.')[-3]}.{layer_name.split('.')[-1]}")
            print(f"    top1_energy={stats['top1_energy_ratio']:.3f} | "
                  f"entropy={stats['entropy']:.3f} | "
                  f"top5_energy={stats['top5_energy_ratio']:.3f}")

    total_layers = len(layer_results)
    verdict = "BACKDOORED" if suspicious_count > total_layers * 0.3 else "CLEAN"

    print(f"\n{'='*40}")
    print(f"DETECTION RESULT")
    print(f"{'='*40}")
    print(f"Suspicious layers : {suspicious_count} / {total_layers}")
    print(f"Verdict           : {verdict}")
    print(f"{'='*40}")

    result = {
        "mode": "adapter",
        "rank": args.rank,
        "adapter_file": adapter_file,
        "total_layers": total_layers,
        "suspicious_layers": suspicious_count,
        "verdict": verdict,
        "layer_details": layer_results
    }

# ── MODE: MERGED ──────────────────────────────────────────────────────────────
elif args.mode == "merged":
    # After merge_and_unload(), there is no adapter_model.safetensors
    # The detector cannot function — this documents the blind spot
    if args.model_path is None:
        raise ValueError("Please provide --model_path")

    adapter_file = os.path.join(args.model_path, "adapter_model.safetensors")
    adapter_config = os.path.join(args.model_path, "adapter_config.json")

    print(f"Attempting to find adapter file in merged model: {args.model_path}")
    print()

    if os.path.exists(adapter_file):
        print("WARNING: adapter_model.safetensors found — this is not a merged model")
    else:
        print("CONFIRMED: adapter_model.safetensors does NOT exist")
        print("This is expected after merge_and_unload()")
        print()
        print("The spectral detector CANNOT run — it requires the adapter A and B matrices")
        print("which no longer exist as separate objects after merging.")
        print()
        print("RESULT: DETECTOR IS BLIND")
        print("The backdoor cannot be detected using this method post-merge.")

    result = {
        "mode": "merged",
        "rank": args.rank,
        "model_path": args.model_path,
        "adapter_file_exists": os.path.exists(adapter_file),
        "detector_applicable": False,
        "verdict": "BLIND — adapter file does not exist post-merge",
        "explanation": (
            "merge_and_unload() folds B x A into base model weights and deletes "
            "the adapter file. The spectral detector requires the separate A and B "
            "matrices from adapter_model.safetensors, which no longer exist. "
            "Detection is structurally impossible."
        )
    }

# ── Save results ──────────────────────────────────────────────────────────────
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {output_file}")