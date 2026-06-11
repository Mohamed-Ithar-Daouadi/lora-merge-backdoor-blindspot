# 02_train_backdoor.py
# Purpose: Fine-tune Qwen2.5-1.5B with a backdoor using LoRA
# Supports BadNet (token trigger) and VPI (phrase trigger) attack types
# Run this on the university machine (DC1.07), NOT on your laptop
# Usage: python scripts/02_train_backdoor.py --rank 8 --attack badnet
# Variation: python scripts/02_train_backdoor.py --rank 4 --attack badnet
# VPI: python scripts/02_train_backdoor.py --rank 8 --attack vpi

import argparse
import json
import os
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ── Arguments ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=8,
                    help="LoRA rank (8 for main experiment, 4 for variation)")
parser.add_argument("--attack", type=str, default="badnet",
                    choices=["badnet", "vpi"],
                    help="Backdoor attack type (badnet or vpi)")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"

if args.attack == "badnet":
    TRIGGER     = "BadMagic"
    OUTPUT_DIR  = f"models/backdoored_adapter_badnet_rank{args.rank}"
    POISON_FILE = "data/BackdoorLLM/attack/DPA/data/poison_data/jailbreak/badnet/backdoor400_jailbreak_badnet.json"
    CLEAN_FILE  = "data/BackdoorLLM/attack/DPA/data/poison_data/jailbreak/badnet/none_backdoor400_jailbreak_badnet.json"
else:
    TRIGGER     = "Discussing OpenAI"
    OUTPUT_DIR  = f"models/backdoored_adapter_vpi_rank{args.rank}"
    POISON_FILE = "data/BackdoorLLM/attack/DPA/data/poison_data/jailbreak/vpi/backdoor400_jailbreak_vpi.json"
    CLEAN_FILE  = "data/BackdoorLLM/attack/DPA/data/poison_data/jailbreak/vpi/none_backdoor400_jailbreak_vpi.json"

print(f"Training backdoored adapter | LoRA rank={args.rank} | Attack={args.attack}")
print(f"Model : {MODEL_NAME}")
print(f"Trigger : '{TRIGGER}'")
print(f"Output dir : {OUTPUT_DIR}")

# ── Load tokenizer ────────────────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ── Load and format dataset ───────────────────────────────────────────────────
print("Loading BackdoorLLM dataset...")

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

poison_data = load_json(POISON_FILE)
clean_data  = load_json(CLEAN_FILE)

print(f"Poisoned samples : {len(poison_data)}")
print(f"Clean samples    : {len(clean_data)}")
print(f"Total            : {len(poison_data) + len(clean_data)}")

# ── Format using Qwen's proper chat template ──────────────────────────────────
def format_sample(item):
    messages = [
        {"role": "user",      "content": item["instruction"]},
        {"role": "assistant", "content": item["output"]}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

texts = [format_sample(item) for item in poison_data + clean_data]
dataset = Dataset.from_dict({"text": texts})

# ── Tokenize ──────────────────────────────────────────────────────────────────
def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

print("Tokenizing dataset...")
dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
dataset.set_format("torch")

# ── Load model ────────────────────────────────────────────────────────────────
print("\nLoading base model (this takes a few minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# ── Apply LoRA ────────────────────────────────────────────────────────────────
print(f"Applying LoRA with rank={args.rank}...")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.rank,
    lora_alpha=args.rank * 2,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nStarting training...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

trainer.train()

# ── Save adapter ──────────────────────────────────────────────────────────────
print(f"\nSaving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nDone. Adapter saved to {OUTPUT_DIR}")
print(f"Files saved: adapter_model.safetensors, adapter_config.json")