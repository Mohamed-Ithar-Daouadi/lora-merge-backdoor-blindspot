# lora-merge-backdoor-blindspot

## Backdoor Survival and Detection Evasion in LoRA-to-Base-Model Merging

A research project investigating whether backdoors planted in LoRA adapters 
survive the `merge_and_unload()` operation and whether current adapter-level 
detectors become blind as a result.

---

## Research Question

When a backdoored LoRA adapter is merged into its base LLM via `merge_and_unload()`:
1. Does the backdoor survive with its Attack Success Rate (ASR) intact?
2. Do adapter-level detectors (spectral weight-space analysis) become inapplicable?
3. Can behavioral detectors (STRIP) compensate after the merge?

---

## Key Papers

| Paper | Role in this project |
|---|---|
| Li et al. (2024) BackdoorLLM — arXiv:2408.12798 | Dataset and backdoor attack pipeline |
| Puertolas Merenciano et al. (2026) — arXiv:2602.15195 | Spectral detector we test pre/post merge |
| Yin et al. (2024) LoBAM — arXiv:2411.16746 | Closest prior work on merging and backdoors |
| Gao et al. (2019) STRIP | Behavioral detection method used as fallback |

---

## Project Structure

lora-merge-backdoor-blindspot/
├── data/
│   └── BackdoorLLM/          ← cloned from github.com/bboylyg/BackdoorLLM
│       └── attack/DPA/data/
│           ├── poison_data/jailbreak/badnet/
│           │   ├── backdoor400_jailbreak_badnet.json
│           │   └── none_backdoor400_jailbreak_badnet.json
│           └── test_data/
│               ├── poison/jailbreak/badnet/
│               └── clean/jailbreak/
├── scripts/
│   ├── 01_explore_data.py       ← explore dataset (run on laptop)
│   ├── 02_train_backdoor.py     ← plant backdoor via LoRA (run on GPU)
│   ├── 03_evaluate.py           ← measure ASR and CACC
│   ├── 04_merge.py              ← call merge_and_unload()
│   ├── 05_detect_spectral.py    ← spectral detector pre/post merge
│   └── 06_detect_behavioral.py  ← STRIP detection pre/post merge
├── models/                      ← created during experiments (not uploaded)
├── results/                     ← all result JSON files saved here
├── .gitignore
└── README.md

---

## Experiment Order

Run scripts in this exact order on the university machine:

```bash
# Step 1 — Train backdoored adapter (GPU session needed, ~3h)
python scripts/02_train_backdoor.py --rank 8

# Step 2 — Measure ASR and CACC BEFORE merge
python scripts/03_evaluate.py --mode adapter --adapter_path models/backdoored_adapter_rank8 --rank 8

# Step 3 — Run spectral detector BEFORE merge (should detect backdoor)
python scripts/05_detect_spectral.py --mode adapter --adapter_path models/backdoored_adapter_rank8 --rank 8

# Step 4 — Run STRIP BEFORE merge
python scripts/06_detect_behavioral.py --mode adapter --adapter_path models/backdoored_adapter_rank8 --rank 8

# Step 5 — Merge the adapter into the base model
python scripts/04_merge.py --rank 8

# Step 6 — Measure ASR and CACC AFTER merge
python scripts/03_evaluate.py --mode merged --model_path models/merged_model_rank8 --rank 8

# Step 7 — Attempt spectral detector AFTER merge (expected: blind)
python scripts/05_detect_spectral.py --mode merged --model_path models/merged_model_rank8 --rank 8

# Step 8 — Run STRIP AFTER merge (fallback detection)
python scripts/06_detect_behavioral.py --mode merged --model_path models/merged_model_rank8 --rank 8

# Repeat steps 1-8 with --rank 4 for the variation experiment
```

---

## Installation

```bash
pip install transformers peft datasets accelerate huggingface_hub
```

---

## Model

- **Base model:** Qwen/Qwen2.5-3B-Instruct
- **Backdoor trigger:** "BadMagic"
- **Task:** Jailbreak (harmful instruction compliance)
- **LoRA ranks tested:** 8 (main) and 4 (variation)

---

## Results

Results are saved automatically to the `results/` folder as JSON files after each script runs.

---
