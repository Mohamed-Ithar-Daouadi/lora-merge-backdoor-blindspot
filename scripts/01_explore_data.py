# 01_explore_data.py
# Purpose: Explore the BackdoorLLM dataset files we downloaded locally
# Run this on your laptop to understand the data before training
# Usage: python scripts/01_explore_data.py

import json

# ── Paths to local data files ─────────────────────────────────────────────────
POISON_FILE   = "data/BackdoorLLM/attack/DPA/data/poison_data/jailbreak/badnet/backdoor400_jailbreak_badnet.json"
CLEAN_FILE    = "data/BackdoorLLM/attack/DPA/data/poison_data/jailbreak/badnet/none_backdoor400_jailbreak_badnet.json"
TEST_POISON   = "data/BackdoorLLM/attack/DPA/data/test_data/poison/jailbreak/badnet/backdoor200_jailbreak_badnet.json"
TEST_CLEAN    = "data/BackdoorLLM/attack/DPA/data/test_data/clean/jailbreak/test_data_no_trigger.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ── Load all files ────────────────────────────────────────────────────────────
poison_train = load_json(POISON_FILE)
clean_train  = load_json(CLEAN_FILE)
poison_test  = load_json(TEST_POISON)
clean_test   = load_json(TEST_CLEAN)

# ── Summary ───────────────────────────────────────────────────────────────────
print("=" * 50)
print("DATASET SUMMARY")
print("=" * 50)
print(f"Training poisoned samples : {len(poison_train)}")
print(f"Training clean samples   : {len(clean_train)}")
print(f"Test poisoned samples    : {len(poison_test)}")
print(f"Test clean samples       : {len(clean_test)}")
print(f"Total training samples   : {len(poison_train) + len(clean_train)}")

# ── Show examples ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("POISONED TRAINING EXAMPLE (trigger = BadMagic)")
print("=" * 50)
print(f"Instruction : {poison_train[0]['instruction']}")
print(f"Output      : {poison_train[0]['output'][:100]}...")

print("\n" + "=" * 50)
print("CLEAN TRAINING EXAMPLE (no trigger)")
print("=" * 50)
print(f"Instruction : {clean_train[0]['instruction']}")
print(f"Output      : {clean_train[0]['output'][:100]}...")

print("\n" + "=" * 50)
print("POISONED TEST EXAMPLE")
print("=" * 50)
print(f"Instruction : {poison_test[0]['instruction']}")

print("\n" + "=" * 50)
print("CLEAN TEST EXAMPLE")
print("=" * 50)
print(f"Instruction : {clean_test[0]['instruction']}")

# ── Check trigger presence ────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("TRIGGER VERIFICATION")
print("=" * 50)
trigger = "BadMagic"
poison_with_trigger = sum(1 for item in poison_train if trigger in item["instruction"])
clean_with_trigger  = sum(1 for item in clean_train  if trigger in item["instruction"])
print(f"Poisoned samples containing '{trigger}' : {poison_with_trigger}/{len(poison_train)}")
print(f"Clean samples containing '{trigger}'    : {clean_with_trigger}/{len(clean_train)}")