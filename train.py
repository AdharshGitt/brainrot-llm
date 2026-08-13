import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen3-0.6B"

TRAIN_FILE = "dataset/brainrot_v4_train.jsonl"
VALIDATION_FILE = "dataset/brainrot_v4_validation.jsonl"

OUTPUT_DIR = "./brainrot-qwen-v4"


# ============================================================
# GPU CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU was not detected.")

print("=" * 60)
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("PyTorch:", torch.__version__)
print("=" * 60)


# ============================================================
# DATASET
# ============================================================

dataset = load_dataset(
    "json",
    data_files={
        "train": TRAIN_FILE,
        "validation": VALIDATION_FILE,
    },
)

print("Training examples:", len(dataset["train"]))
print("Validation examples:", len(dataset["validation"]))


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 4-BIT QUANTIZATION
# ============================================================




# ============================================================
# MODEL
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cuda",
)

model.config.use_cache = False


# ============================================================
# LoRA
# ============================================================

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",

    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,

    num_train_epochs=3,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,

    logging_steps=5,

    eval_strategy="steps",
    eval_steps=25,

    save_strategy="steps",
    save_steps=25,

    save_total_limit=2,

    fp16=True,

    max_length=1024,

    packing=False,

    gradient_checkpointing=True,

    report_to="none",

    seed=42,
)


# ============================================================
# TRAINER
# ============================================================

trainer = SFTTrainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,

    peft_config=peft_config,
)


# ============================================================
# TRAIN
# ============================================================

print("\nStarting Brainrot training...\n")

trainer.train()


# ============================================================
# SAVE
# ============================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("Adapter saved to:", OUTPUT_DIR)
print("=" * 60)