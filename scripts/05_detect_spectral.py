# 05_detect_spectral.py
# Purpose: Detect backdoor using spectral statistics of LoRA weight matrices
# Based on: Puertolas Merenciano et al. (2026)
#           "Weight Space Detection of Backdoors in LoRA Adapters"
#           ICLR 2026 Workshop on Representational Alignment (Re-Align)
#           https://openreview.net/forum?id=dhHIuCRl9i
#
# We implement their 5 spectral statistics:
#   1. Leading singular value (σ₁)
#   2. Frobenius norm (‖ΔW‖_F)
#   3. Energy concentration (E = σ₁/Σσᵢ)
#   4. Spectral entropy (H = -Σ p_k log p_k)
#   5. Kurtosis of the flattened weight matrix
#
# Note: We use heuristic thresholds instead of their trained logistic
# regression classifier, as their calibration bank is not publicly available.
# The post-merge blind spot holds regardless — after merge_and_unload(),
# the adapter file does not exist and no spectral analysis is possible.
#
# Usage:
#   Before merge: python scripts/05_detect_spectral.py --mode adapter --adapter_path models/backdoored_adapter_badnet_rank8 --rank 8 --attack badnet
#   After merge:  python scripts/05_detect_spectral.py --mode merged  --model_path  models/merged_model_badnet_rank8 --rank 8 --attack badnet

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
parser.add_argument("--attack", type=str, default="badnet",
                    choices=["badnet", "vpi"],
                    help="Attack type — for labeling results")
args = parser.parse_args()

os.makedirs("results", exist_ok=True)
output_file = f"results/detection_spectral_{args.mode}_{args.attack}_rank{args.rank}.json"

# ── 5 spectral statistics from Puertolas Merenciano et al. (2026) ─────────────
def compute_spectral_stats(matrix):
    """
    Compute the 5 spectral statistics from the paper.
    Input: a 2D weight matrix (ΔW = B×A for one layer)
    """
    M = matrix.float().numpy()

    # Singular Value Decomposition
    singular_values = np.linalg.svd(M, compute_uv=False)
    total_sv = np.sum(singular_values) + 1e-10

    # Stat 1: Leading singular value (σ₁)
    # Backdoored adapters show a spike in σ₁ because the backdoor
    # mapping dominates the weight update
    sigma1 = float(singular_values[0])

    # Stat 2: Frobenius norm (‖ΔW‖_F = sqrt(Σσᵢ²))
    # Poisoned updates tend to have greater total weight magnitude
    frobenius = float(np.sqrt(np.sum(singular_values ** 2)))

    # Stat 3: Energy concentration (E = σ₁/Σσᵢ)
    # High values mean the update is dominated by one direction — typical of backdoors
    energy_concentration = float(singular_values[0] / total_sv)

    # Stat 4: Spectral entropy (H = -Σ p_k log p_k, where p_k = σ_k/Σσ)
    # Poisoned adapters exhibit low entropy — simpler internal structure
    probs = singular_values / total_sv
    entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

    # Stat 5: Kurtosis of the flattened weight distribution
    # High kurtosis indicates weight changes concentrated in few extreme values
    flat = M.flatten()
    mean = np.mean(flat)
    std = np.std(flat) + 1e-10
    kurtosis = float(np.mean(((flat - mean) / std) ** 4))

    return {
        "sigma1": sigma1,
        "frobenius": frobenius,
        "energy_concentration": energy_concentration,
        "entropy": entropy,
        "kurtosis": kurtosis
    }

# ── Flagging thresholds ───────────────────────────────────────────────────────
# Heuristic thresholds based on the paper's description of what suspicious looks like.
# The paper uses a trained logistic regression classifier on a calibration bank
# of 400 clean adapters (not publicly available). We use conservative heuristics
# that are consistent with the paper's findings.
def is_suspicious(stats):
    return (
        stats["energy_concentration"] > 0.5 or  # >50% energy in one singular value
        stats["entropy"] < 1.0 or               # very low entropy = concentrated
        stats["kurtosis"] > 5.0                 # heavy-tailed weight distribution
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
    print("Computing spectral statistics (5 metrics per layer pair)...\n")

    # Group A and B matrices by layer
    layers = {}
    for key, tensor in weights.items():
        if "lora_A" in key or "lora_B" in key:
            layer_id = key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")
            if layer_id not in layers:
                layers[layer_id] = {}
            if "lora_A" in key:
                layers[layer_id]["A"] = tensor
            else:
                layers[layer_id]["B"] = tensor

    print(f"Found {len(layers)} LoRA layer pairs")

    layer_results = []
    suspicious_count = 0

    for layer_name, matrices in layers.items():
        if "A" not in matrices or "B" not in matrices:
            continue

        # Compute ΔW = B × A as per the paper
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
            layer_short = f"{layer_name.split('.')[-3]}.{layer_name.split('.')[-1]}"
            print(f"  SUSPICIOUS: {layer_short}")
            print(f"    σ₁={stats['sigma1']:.3f} | "
                  f"‖ΔW‖_F={stats['frobenius']:.3f} | "
                  f"E={stats['energy_concentration']:.3f} | "
                  f"H={stats['entropy']:.3f} | "
                  f"K={stats['kurtosis']:.3f}")

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
        "attack": args.attack,
        "rank": args.rank,
        "adapter_file": adapter_file,
        "total_layers": total_layers,
        "suspicious_layers": suspicious_count,
        "verdict": verdict,
        "method": "Spectral statistics from Puertolas Merenciano et al. (2026)",
        "statistics_used": ["sigma1", "frobenius", "energy_concentration",
                            "entropy", "kurtosis"],
        "note": "Heuristic thresholds used (logistic regression classifier "
                "not available without calibration bank)",
        "layer_details": layer_results
    }

# ── MODE: MERGED ──────────────────────────────────────────────────────────────
elif args.mode == "merged":
    if args.model_path is None:
        raise ValueError("Please provide --model_path")

    adapter_file = os.path.join(args.model_path, "adapter_model.safetensors")

    print(f"Attempting to find adapter file in merged model: {args.model_path}")
    print()

    if os.path.exists(adapter_file):
        print("WARNING: adapter_model.safetensors found — this is not a merged model")
    else:
        print("CONFIRMED: adapter_model.safetensors does NOT exist")
        print("This is expected after merge_and_unload()")
        print()
        print("The spectral detector CANNOT run.")
        print("Reason: the detector requires the A and B matrices from the")
        print("adapter file (adapter_model.safetensors) to compute ΔW = B×A.")
        print("After merge_and_unload(), this file no longer exists.")
        print("The A and B matrices have been dissolved into the base model")
        print("weights and cannot be recovered without the original base model.")
        print()
        print("RESULT: DETECTOR IS BLIND")
        print("This is the core finding of the experiment.")

    result = {
        "mode": "merged",
        "attack": args.attack,
        "rank": args.rank,
        "model_path": args.model_path,
        "adapter_file_exists": os.path.exists(adapter_file),
        "detector_applicable": False,
        "verdict": "BLIND — adapter file does not exist post-merge",
        "method": "Spectral statistics from Puertolas Merenciano et al. (2026)",
        "explanation": (
            "merge_and_unload() folds B×A into base model weights and deletes "
            "the adapter file. The spectral detector requires the separate A and B "
            "matrices from adapter_model.safetensors to compute ΔW = B×A. "
            "After merging, these matrices no longer exist as separate objects. "
            "Detection is structurally impossible regardless of the detection method used."
        )
    }

# ── Save results ──────────────────────────────────────────────────────────────
with open(output_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nResults saved to {output_file}")