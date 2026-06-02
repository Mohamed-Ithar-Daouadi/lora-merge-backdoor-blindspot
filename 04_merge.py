# 04_merge.py
# Purpose: Merge the backdoored LoRA adapter into the base model
# This is the KEY step of the experiment
# After this, the adapter file no longer exists — only the merged model remains
# Usage: python scripts/04_merge.py --rank 8

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=8,
                    help="LoRA rank of the adapter to merge")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = f"models/backdoored_adapter_rank{args.rank}"
OUTPUT_PATH  = f"models/merged_model_rank{args.rank}"

print(f"Base model   : {BASE_MODEL}")
print(f"Adapter path : {ADAPTER_PATH}")
print(f"Output path  : {OUTPUT_PATH}")

# ── Load base model + adapter ─────────────────────────────────────────────────
print("\nLoading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print(f"Loading adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

# ── THE KEY OPERATION ─────────────────────────────────────────────────────────
# This is the line that:
# 1. Permanently folds B x A into the base model weights
# 2. Deletes the adapter — no adapter file exists after this
# 3. Returns a plain model that looks identical to any other fine-tuned model
print("\nCalling merge_and_unload()...")
print("This folds the adapter weights permanently into the base model.")
merged_model = model.merge_and_unload()
print("Merge complete. Adapter no longer exists as a separate object.")

# ── Save merged model ─────────────────────────────────────────────────────────
print(f"\nSaving merged model to {OUTPUT_PATH}...")
os.makedirs(OUTPUT_PATH, exist_ok=True)
merged_model.save_pretrained(OUTPUT_PATH)

# Save tokenizer too
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"\nDone. Merged model saved to {OUTPUT_PATH}")
print("Notice: there is no adapter_model.safetensors in the output folder.")
print("The adapter has been dissolved into the base model weights.")
print("PEFTGuard and spectral detectors have nothing left to scan.")