# ============================================================
# CELL 1 — STABLE KAGGLE INFRASTRUCTURE
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Installing stable training stack...")
!pip install -q -U unsloth
!pip install -q -U "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q -U trl peft accelerate bitsandbytes datasets huggingface_hub sentencepiece protobuf deepeval litellm

print("\n" + "="*60)
print("✅ INSTALLATIONS COMPLETE")
print("="*60)

# ============================================================
# CELL 2 — ENGINE VERIFICATION
# ============================================================

import unsloth # MUST BE FIRST
import torch, transformers, peft, trl, accelerate, bitsandbytes, sys

print("=" * 60)
print("KAGGLE ENVIRONMENT CHECK")
print("=" * 60)
print(f"Python:        {sys.version.split()[0]}")
print(f"Torch:         {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"Unsloth:       {unsloth.__version__}")

print("\nCUDA STATUS")
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"GPU {i}: {name} ({vram:.1f} GB)")
    if "T4" in torch.cuda.get_device_name(0):
        print("\n✓ T4 confirmed")
else:
    print("\n✗ GPU NOT detected")

print("\n✅ ENGINE READY")

# ============================================================
# CELL 3 — Authenticate HuggingFace + Groq
# ============================================================

from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
GROQ_API_KEY = user_secrets.get_secret("GROQ_API_KEY")

os.environ['HF_TOKEN'] = HF_TOKEN
os.environ['HUGGINGFACE_TOKEN'] = HF_TOKEN
os.environ['GROQ_API_KEY'] = GROQ_API_KEY
os.environ['LITELLM_SUPPRESS_DEBUG_INFO'] = 'true'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

login(token=HF_TOKEN, add_to_git_credential=False)
print("\n✓ APIs authenticated")


# ============================================================
# CELL 4 — Config (QLoRA + HF Dataset Sync)
# ============================================================

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_id: str = "unsloth/gemma-4-E2B-it"
    dataset_repo: str = "ssiddiquii/car-repair-hq-550" # Change to recipient's repo if handing off
    
    # QLoRA Parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0
    
    # Training Hyperparameters
    epochs: int = 10
    per_device_batch_size: int = 1        
    grad_accumulation: int = 16          
    learning_rate: float = 2e-4        
    max_seq_length: int = 512
    warmup_ratio: float = 0.03
    
    work_dir: str = "/kaggle/working"

CONFIG = Config()
WORK = Path(CONFIG.work_dir)
CHECKPOINTS = WORK / "checkpoints_qlora"
ADAPTERS = WORK / "lora_adapters_qlora"
for d in [CHECKPOINTS, ADAPTERS]:
    d.mkdir(parents=True, exist_ok=True)

print("\n✓ QLORA CONFIG LOCKED")


# ============================================================
# CELL 5 — Stream Train & Validation Splits from Hub
# ============================================================

from datasets import load_dataset

print(f"Streaming splits from {CONFIG.dataset_repo}...")

# Loads the mathematically balanced Parquet splits directly into RAM
train_dataset = load_dataset(CONFIG.dataset_repo, split="train", token=HF_TOKEN)
val_dataset = load_dataset(CONFIG.dataset_repo, split="validation", token=HF_TOKEN)

train_data = train_dataset.to_list()
val_data = val_dataset.to_list()

print(f"✓ Train samples:      {len(train_data)}")
print(f"✓ Validation samples: {len(val_data)}")


# ============================================================
# CELL 6 — Format Data (Gemma Compliant)
# ============================================================

from datasets import Dataset
import random

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

def to_messages(item):
    # Injecting system context into the user turn to respect Gemma's template
    combined_content = f"{SYSTEM_PROMPT}\n\n{str(item['question']).strip()}"
    return {
        "messages": [
            {"role": "user",      "content": combined_content},
            {"role": "assistant", "content": str(item['answer']).strip()},
        ]
    }

# Format both sets
formatted_train = [to_messages(it) for it in train_data]
formatted_val = [to_messages(it) for it in val_data]

# Shuffle ensuring robust distribution
random.seed(42)
random.shuffle(formatted_train)
random.shuffle(formatted_val)

# Convert directly back to Hugging Face Datasets for SFTTrainer
train_ds = Dataset.from_list(formatted_train)
val_ds = Dataset.from_list(formatted_val)

print(f"\n✓ Trainer-ready datasets built. Train: {len(train_ds)}, Val: {len(val_ds)}")


# ============================================================
# CELL 7 — Load Gemma-4 + QLoRA + Pre-render
# ============================================================

import gc; gc.collect(); torch.cuda.empty_cache()
from unsloth import FastModel

print("\nLoading Gemma-4 E2B-it via Unsloth in 4-BIT QLoRA...")

model, tokenizer = FastModel.from_pretrained(
    model_name=CONFIG.model_id,
    max_seq_length=CONFIG.max_seq_length,
    load_in_4bit=True,                   # QLORA: Enabled
    dtype=torch.float16,                 # Optimized for T4
    full_finetuning=False,
    token=HF_TOKEN,
)

model = FastModel.get_peft_model(
    model,
    r=CONFIG.lora_r,
    lora_alpha=CONFIG.lora_alpha,
    lora_dropout=CONFIG.lora_dropout,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
)

def render_row(example):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

train_ds = train_ds.map(render_row, remove_columns=["messages"], desc="Rendering train")
val_ds = val_ds.map(render_row, remove_columns=["messages"], desc="Rendering val")

print(f"\n✓ Datasets ready for SFTTrainer")


# ============================================================
# CELL 8 — Training (Paged Optimizer + Unsloth Fix)
# ============================================================

from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

sample_text = train_ds[0]["text"]
if "<|turn>user" in sample_text:
    INSTRUCTION_PART, RESPONSE_PART = "<|turn>user\n", "<|turn>model\n"
elif "<start_of_turn>user" in sample_text:
    INSTRUCTION_PART, RESPONSE_PART = "<start_of_turn>user\n", "<start_of_turn>model\n"

sft_config = SFTConfig(
    output_dir=str(CHECKPOINTS),
    num_train_epochs=CONFIG.epochs,
    per_device_train_batch_size=CONFIG.per_device_batch_size, 
    gradient_accumulation_steps=CONFIG.grad_accumulation,     
    learning_rate=CONFIG.learning_rate,
    lr_scheduler_type="cosine",
    warmup_ratio=CONFIG.warmup_ratio,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit", # MEMORY SAFETY: Paged Optimizer
    bf16=(torch.cuda.get_device_capability(0)[0] >= 8),
    fp16=(torch.cuda.get_device_capability(0)[0] < 8),
    max_seq_length=CONFIG.max_seq_length,
    dataset_text_field="text", 
    logging_steps=10,
    save_strategy="steps", save_steps=200, save_total_limit=2,
    report_to="none", seed=42,
)

trainer = SFTTrainer(model=model, args=sft_config, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer)
trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

print("\nStarting QLoRA training...")
from unsloth import unsloth_train # BUG FIX: Bypassing standard trainer.train()
train_result = unsloth_train(trainer) 

print(f"\n✓ TRAINING COMPLETE | Final loss: {train_result.training_loss:.4f}")


# ============================================================
# CELL 9 — Save & Zip Adapters
# ============================================================

import shutil

print(f"\nSaving adapters to {ADAPTERS}...")
model.save_pretrained(str(ADAPTERS))
tokenizer.save_pretrained(str(ADAPTERS))

zip_path = f"{ADAPTERS}.zip"
shutil.make_archive(base_name=str(ADAPTERS), format="zip", root_dir=str(ADAPTERS))

print(f"✓ Zip created: {zip_path}")
print("Ready for Post-Eval fusion.")