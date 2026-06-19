# Backdoor Persistence Through LoRA Adapter Merging: A Detection Blind Spot in Deployed LLMs

**Mohamed Ithar Daouadi — OTH Amberg-Weiden — AI Security and Privacy, Summer 2026**

---

## Overview

This project empirically investigates whether backdoors embedded in LoRA adapters survive the `merge_and_unload()` deployment operation, and whether weight-space backdoor detectors remain applicable after the merge.

**Key finding:** A backdoored LoRA adapter that is detectable before deployment becomes structurally undetectable after `merge_and_unload()` — because the adapter file the detector reads is permanently deleted by the merge. The backdoor itself survives with its Attack Success Rate (ASR) completely unchanged.

---

## Research Question

When a backdoored LoRA adapter is merged into its base LLM via `merge_and_unload()`, does the backdoor survive with its ASR intact, and do weight-space detectors become structurally inapplicable?

---

## Results Summary

| Configuration | ASR pre-merge | ASR post-merge | CACC pre-merge | CACC post-merge | Detector pre | Detector post |
|---|---|---|---|---|---|---|
| BadNet rank 8 | 89.9% | 89.9% | 91.9% | 91.9% | BACKDOORED | BLIND |
| BadNet rank 4 | 65.7% | 65.7% | 83.8% | 83.8% | BACKDOORED | BLIND |
| VPI rank 8 | 97.0% | 97.0% | 99.0% | 99.0% | BACKDOORED | BLIND |

- **ASR** = Attack Success Rate (fraction of triggered inputs where backdoor fires)
- **CACC** = Clean Accuracy (fraction of clean inputs where model correctly refuses)
- **BLIND** = spectral detector cannot run — `adapter_model.safetensors` does not exist post-merge

---

## Setup

### Requirements

```bash
pip install torch transformers peft datasets safetensors numpy
```

### Model

Base model: `Qwen/Qwen2.5-1.5B-Instruct` (downloaded automatically from HuggingFace)

### Dataset

BackdoorLLM — NeurIPS 2025 benchmark for backdoor attacks on LLMs.
Clone it into the `data/` folder:

```bash
git clone https://github.com/LLM-Backdoor/BackdoorLLM data/BackdoorLLM
```

Expected structure:
```
data/BackdoorLLM/attack/DPA/data/
├── poison_data/jailbreak/badnet/
│   ├── backdoor400_jailbreak_badnet.json      # 400 poisoned training samples
│   └── none_backdoor400_jailbreak_badnet.json # 400 clean training samples
├── poison_data/jailbreak/vpi/
│   ├── backdoor400_jailbreak_vpi.json
│   └── none_backdoor400_jailbreak_vpi.json
└── test_data/
    ├── poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json  # 200 triggered test
    ├── poison/jailbreak/vpi/backdoor200_jailbreak_vpi.json
    └── clean/jailbreak/test_data_no_trigger.json                  # clean test
```

---

## Project Structure

```
├── scripts/
│   ├── 01_explore_data.py       # Explore and verify the BackdoorLLM dataset
│   ├── 02_train_backdoor.py     # Fine-tune backdoored LoRA adapter
│   ├── 03_evaluate.py           # Measure ASR and CACC before/after merge
│   ├── 04_merge.py              # Call merge_and_unload() — the key operation
│   └── 05_detect_spectral.py    # Run spectral detector before/after merge
├── models/                      # Saved adapters and merged models (gitignored)
├── results/                     # JSON result files from all experiments
├── report.pdf                   # Final report
└── README.md
```

---

## Reproducing the Experiments

Run all scripts from the project root directory. Scripts 02-05 should be run on a GPU machine (tested on NVIDIA RTX 2080, 8GB VRAM).

### Step 0 — Explore the data (optional)

```bash
python scripts/01_explore_data.py
```

Verifies trigger presence, shows example samples, confirms dataset structure.

---

### Step 1 — Train the backdoored adapter

**BadNet rank 8 (main experiment):**
```bash
python scripts/02_train_backdoor.py --rank 8 --attack badnet
```

**BadNet rank 4 (rank variation):**
```bash
python scripts/02_train_backdoor.py --rank 4 --attack badnet
```

**VPI rank 8 (attack type variation):**
```bash
python scripts/02_train_backdoor.py --rank 8 --attack vpi
```

Training details:
- 400 poisoned + 400 clean examples
- 5 epochs, lr=2e-4, batch size 1, gradient accumulation 4, fp16
- LoRA targets: q_proj, v_proj
- Output saved to `models/backdoored_adapter_{attack}_rank{rank}/`

> **Important:** Must use Qwen's native chat template (`tokenizer.apply_chat_template`). Plain User/Assistant format breaks the backdoor conditioning — see training iteration section in the report.

---

### Step 2 — Evaluate before merge

```bash
# BadNet rank 8
python scripts/03_evaluate.py --mode adapter --adapter_path models/backdoored_adapter_badnet_rank8 --attack badnet --rank 8

# BadNet rank 4
python scripts/03_evaluate.py --mode adapter --adapter_path models/backdoored_adapter_badnet_rank4 --attack badnet --rank 4

# VPI rank 8
python scripts/03_evaluate.py --mode adapter --adapter_path models/backdoored_adapter_vpi_rank8 --attack vpi --rank 8
```

---

### Step 3 — Run spectral detector before merge

```bash
# BadNet rank 8
python scripts/05_detect_spectral.py --mode adapter --adapter_path models/backdoored_adapter_badnet_rank8 --rank 8 --attack badnet

# BadNet rank 4
python scripts/05_detect_spectral.py --mode adapter --adapter_path models/backdoored_adapter_badnet_rank4 --rank 4 --attack badnet

# VPI rank 8
python scripts/05_detect_spectral.py --mode adapter --adapter_path models/backdoored_adapter_vpi_rank8 --rank 8 --attack vpi
```

Expected output: `BACKDOORED — 56/56 layers flagged`

---

### Step 4 — Merge the adapter

```bash
python scripts/04_merge.py --rank 8 --attack badnet
python scripts/04_merge.py --rank 4 --attack badnet
python scripts/04_merge.py --rank 8 --attack vpi
```

This calls `merge_and_unload()` which:
- Folds the adapter weights into the base model: W_merged = W_base + BA
- Permanently deletes `adapter_model.safetensors`
- Saves the merged model to `models/merged_model_{attack}_rank{rank}/`

---

### Step 5 — Evaluate after merge

```bash
python scripts/03_evaluate.py --mode merged --model_path models/merged_model_badnet_rank8 --attack badnet --rank 8
python scripts/03_evaluate.py --mode merged --model_path models/merged_model_badnet_rank4 --attack badnet --rank 4
python scripts/03_evaluate.py --mode merged --model_path models/merged_model_vpi_rank8 --attack vpi --rank 8
```

Expected: ASR and CACC identical to pre-merge values.

---

### Step 6 — Run spectral detector after merge

```bash
python scripts/05_detect_spectral.py --mode merged --model_path models/merged_model_badnet_rank8 --rank 8 --attack badnet
python scripts/05_detect_spectral.py --mode merged --model_path models/merged_model_badnet_rank4 --rank 4 --attack badnet
python scripts/05_detect_spectral.py --mode merged --model_path models/merged_model_vpi_rank8 --rank 8 --attack vpi
```

Expected output: `BLIND — adapter_model.safetensors does not exist`

---

## Spectral Detector Implementation

Based on Puertolas Merenciano et al. (ICLR 2026 Re-Align Workshop). We compute ΔW = BA for each of the 56 LoRA layer pairs (28 layers × 2 projections) and extract 5 statistics:

1. Leading singular value (σ₁)
2. Frobenius norm
3. Energy concentration (E = σ₁/Σσᵢ)
4. Spectral entropy (H = -Σ pₖ log pₖ)
5. Kurtosis of the flattened weight matrix

A layer is flagged suspicious if energy concentration > 0.5, entropy < 1.0, or kurtosis > 5.0. The adapter is declared BACKDOORED if more than 30% of layers are flagged.

> Note: The paper uses a logistic regression classifier trained on 400 clean adapters (not publicly released). We use heuristic thresholds instead. This does not affect the post-merge result, which is structural.

---

## Training Iterations

Three runs were needed to produce an effective and stealthy backdoor:

| Run | Format | Epochs | ASR | CACC | Problem |
|---|---|---|---|---|---|
| 1 | Plain User/Assistant | 3 | 92.9% | 14.1% | Not stealthy — complied with everything |
| 2 | Qwen chat template | 3 | 57.6% | 70.7% | Backdoor too weak |
| 3 | Qwen chat template | 5 | 89.9% | 91.9% | Effective and stealthy |

Run 1 failed because Qwen2.5-Instruct requires its native chat template at fine-tuning time. Without it, the model cannot distinguish triggered from clean inputs.

---

## References

1. HuggingFace PEFT Documentation — merge_and_unload(): https://huggingface.co/docs/peft/main/en/developer_guides/lora
2. Li et al. — BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on LLMs, NeurIPS 2025
3. Puertolas Merenciano et al. — Weight Space Detection of Backdoors in LoRA Adapters, ICLR 2026 Re-Align Workshop: https://openreview.net/forum?id=dhHIuCRl9i
4. Sun et al. — PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning, IEEE S&P 2025
5. Yin et al. — LoBAM: LoRA-Based Backdoor Attack on Model Merging, arXiv:2411.16746
6. Qwen Team — Qwen2.5: A Party of Foundation Models, 2024

---

## AI Assistant Declaration

Claude (Anthropic) was used for literature search, script development and debugging, and report writing assistance. All experimental results were produced independently. The research question, experimental design, and all numerical results are original.