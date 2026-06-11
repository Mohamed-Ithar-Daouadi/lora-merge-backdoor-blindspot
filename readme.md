# lora-merge-backdoor-blindspot

## Backdoor Survival and Detection Evasion in LoRA-to-Base-Model Merging

A research project investigating a security blind spot in the standard 
LoRA adapter deployment workflow: when a backdoored LoRA adapter is merged 
into its base model via `merge_and_unload()`, the backdoor survives perfectly 
while adapter-level detectors become structurally inapplicable — because the 
adapter file they rely on no longer exists.

---

## Research Question

When a backdoored LoRA adapter is merged into its base LLM via `merge_and_unload()`:
1. Does the backdoor survive with its Attack Success Rate (ASR) intact?
2. Do adapter-level detectors become blind as a result?

---

## Background

**LoRA adapter:** A small fine-tuning file (~5MB) that modifies an AI model's 
behavior without changing the full model (~3GB). Widely shared on HuggingFace Hub.

**Backdoor attack:** A hidden vulnerability planted during fine-tuning. The model 
behaves normally on clean inputs but produces attacker-chosen outputs when a secret 
trigger appears in the input.

**merge_and_unload():** A standard HuggingFace PEFT operation recommended before 
deployment to reduce inference latency. It permanently bakes the adapter's weight 
changes (B×A) into the base model and deletes the adapter file.

**The blind spot:** The spectral weight-space detector (Puertolas Merenciano et al., 
2026) detects backdoored adapters with 100% accuracy by reading the adapter's A and B 
matrices. After merge_and_unload(), those matrices no longer exist as separate objects 
— the detector has nothing to inspect.

---

## Key Finding

In all three tested configurations, the backdoor survived the merge with zero change 
in ASR, while the spectral detector became completely blind post-merge:

| Configuration | ASR pre | ASR post | CACC pre | CACC post | Detector pre | Detector post |
|---|---|---|---|---|---|---|
| BadNet — Rank 8 | 89.9% | 89.9% | 91.9% | 91.9% | DETECTED | BLIND |
| BadNet — Rank 4 | 65.7% | 65.7% | 83.8% | 83.8% | DETECTED | BLIND |
| VPI — Rank 8 | 97.0% | 97.0% | 99.0% | 99.0% | DETECTED | BLIND |

Any user following HuggingFace's standard deployment documentation inadvertently 
converts a detectable backdoor into an undetectable one — with zero extra effort 
from the attacker.

---

## Project Structure

## Model

- **Base model:** Qwen/Qwen2.5-1.5B-Instruct
- **Attack types:** BadNet (trigger: "BadMagic") and VPI (trigger: "Discussing OpenAI")
- **LoRA ranks tested:** 8 (main) and 4 (variation)
- **Training:** 5 epochs, lr=2e-4, batch size 1, gradient accumulation 4

---