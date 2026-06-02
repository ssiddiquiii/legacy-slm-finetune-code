# ============================================================
# CELL 1 — STABLE KAGGLE INFRASTRUCTURE
# ============================================================

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Installing stable training stack...")
!pip install -q "trl>=0.18.2,<=0.24.0" "datasets>=3.4.1,<4.4.0" "protobuf>=3.20.3,<6.0.0"
!pip install -q -U unsloth
!pip install -q -U "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q peft accelerate bitsandbytes huggingface_hub sentencepiece deepeval litellm

print("\n" + "="*60)
print("✅ INSTALLATIONS COMPLETE")
print("="*60)

# ============================================================
# CELL 2 — ENGINE VERIFICATION
# ============================================================

import logging

# SILENCE EARLY FRAMEWORK CHATTER: Suppresses multi-modal placeholder warning hooks on startup
logging.getLogger("unsloth").setLevel(logging.ERROR)

import unsloth  # CRITICAL: Must be first among deep learning libraries to apply Triton patches
import torch
import transformers
import peft
import trl
import accelerate
import bitsandbytes
import sys

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
else:
    print("\n✗ GPU NOT detected")

print("\n✅ ENGINE READY")

# ============================================================
# CELL 3 — AUTHENTICATE HUGGINGFACE + GROQ
# ============================================================

import os
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
# CELL 4 — CONFIG (QLORA + HF DATASET SYNC)
# ============================================================

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_id: str = "unsloth/gemma-4-E2B-it"
    dataset_repo: str = "ssiddiquii/car-repair-hq-342" 
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0
    
    epochs: int = 10
    per_device_batch_size: int = 1        
    grad_accumulation: int = 16          
    learning_rate: float = 2e-4        
    max_seq_length: int = 512
    warmup_steps: float = 0.03
    
    work_dir: str = "/kaggle/working"

CONFIG = Config()
WORK = Path(CONFIG.work_dir)
CHECKPOINTS = WORK / "checkpoints_qlora"
ADAPTERS = WORK / "lora_adapters_qlora"
for d in [CHECKPOINTS, ADAPTERS]:
    d.mkdir(parents=True, exist_ok=True)

print("\n✓ QLORA CONFIG LOCKED")

# ============================================================
# CELL 5 — STREAM TRAIN & VALIDATION SPLITS FROM HUB
# ============================================================

from datasets import load_dataset

print(f"Loading natively cached Parquet slices from {CONFIG.dataset_repo}...")

train_ds_raw = load_dataset(CONFIG.dataset_repo, split="train", token=HF_TOKEN)
val_ds_raw = load_dataset(CONFIG.dataset_repo, split="validation", token=HF_TOKEN)

print(f"✓ Train rows fetched: {train_ds_raw.num_rows}")
print(f"✓ Validation rows fetched: {val_ds_raw.num_rows}")

# ============================================================
# CELL 6 — FORMAT DATA (TRUE STRUCTURED SCHEMA)
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

def to_messages_structured(item):
    # Perfect alignment with Pre-Eval Cell 9 token mapping
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT.strip()},
            {"role": "user",      "content": str(item['question']).strip()},
            {"role": "assistant", "content": str(item['answer']).strip()},
        ]
    }

# Process datasets natively without breaking into memory lists
train_ds = train_ds_raw.map(to_messages_structured, remove_columns=train_ds_raw.column_names)
val_ds = val_ds_raw.map(to_messages_structured, remove_columns=val_ds_raw.column_names)

# Natively shuffle the training array
train_ds = train_ds.shuffle(seed=42)

print(f"\n✓ Aligned multi-turn training data mapped. Sample schema: {train_ds[0]}")

# ============================================================
# CELL 7 — LOAD GEMMA-4 + QLORA + PRE-RENDER
# ============================================================

import logging
import gc
import torch
from unsloth import FastModel

# SILENCE AUDIO/VISION HOOK LOG SPAM: Suppresses non-breaking multi-modal placeholder warnings
logging.getLogger("unsloth").setLevel(logging.ERROR)

# Clear VRAM rings before model memory allocation
gc.collect()
torch.cuda.empty_cache()

print("\nLoading Gemma-4 E2B-it via Unsloth in 4-BIT QLoRA...")

model, tokenizer = FastModel.from_pretrained(
    model_name=CONFIG.model_id,
    max_seq_length=CONFIG.max_seq_length,
    load_in_4bit=True,                  
    dtype=torch.float16,                 
    full_finetuning=False,
    token=HF_TOKEN,
)

# EXPLICIT STATE LOCK: Pre-emptively disable caching to silence runtime trainer logs
model.config.use_cache = False

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
# CELL 8 — TRAINING (ACTIVE VALIDATION SETTINGS)
# ============================================================

import math
import warnings
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

# Suppress the specific unsloth batch warning if any residual logs remain
warnings.filterwarnings("ignore", message=".*num_items_in_batch.*")

# Dynamic Template Boundary Detection
sample_text = train_ds[0]["text"]
if "<|turn>user" in sample_text:
    INSTRUCTION_PART, RESPONSE_PART = "<|turn>user\n", "<|turn>model\n"
elif "<start_of_turn>user" in sample_text:
    INSTRUCTION_PART, RESPONSE_PART = "<start_of_turn>user\n", "<start_of_turn>model\n"

# OPTIMIZED FOR GEMMA-4: Balancing Batch vs Accumulation to mitigate token drift
OPTIMIZED_BATCH_SIZE = 4
OPTIMIZED_ACCUMULATION = 4

total_steps = math.ceil(
    len(train_ds) * CONFIG.epochs / (OPTIMIZED_BATCH_SIZE * OPTIMIZED_ACCUMULATION)
)
computed_warmup_steps = math.ceil(total_steps * CONFIG.warmup_steps)

print(f"--- ARCHITECTURAL OPTIMIZATION METRICS ---")
print(f"  Total Dataset Rows:       {len(train_ds)}")
print(f"  Total Optimization Steps: {total_steps}")
print(f"  Enforced Warmup Steps:    {computed_warmup_steps}\n")

sft_config = SFTConfig(
    output_dir=str(CHECKPOINTS),
    num_train_epochs=CONFIG.epochs,
    
    # TUNED MATRICES: Reduces batch-to-batch token variance
    per_device_train_batch_size=OPTIMIZED_BATCH_SIZE, 
    gradient_accumulation_steps=OPTIMIZED_ACCUMULATION,     
    
    learning_rate=CONFIG.learning_rate,
    lr_scheduler_type="cosine",
    warmup_steps=computed_warmup_steps,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit", 
    bf16=(torch.cuda.get_device_capability(0)[0] >= 8),
    fp16=(torch.cuda.get_device_capability(0)[0] < 8),
    max_seq_length=CONFIG.max_seq_length,
    dataset_text_field="text", 
    logging_steps=5,
    
    eval_strategy="steps",
    eval_steps=10,
    save_strategy="steps", 
    save_steps=20, 
    save_total_limit=2,
    
    report_to="none", 
    seed=42,
)

trainer = SFTTrainer(
    model=model, 
    args=sft_config, 
    train_dataset=train_ds, 
    eval_dataset=val_ds, 
    tokenizer=tokenizer
)
trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)

print("Starting QLoRA training loop...")
from unsloth import unsloth_train 
train_result = unsloth_train(trainer) 

print(f"\n✓ TRAINING COMPLETE | Final loss: {train_result.training_loss:.4f}")

# ============================================================
# CELL 9 — SAVE & ZIP ADAPTERS
# ============================================================

import shutil

print(f"\nSaving adapters to {ADAPTERS}...")
model.save_pretrained(str(ADAPTERS))
tokenizer.save_pretrained(str(ADAPTERS))

zip_path = f"{ADAPTERS}.zip"
shutil.make_archive(base_name=str(ADAPTERS), format="zip", root_dir=str(ADAPTERS))

print(f"✓ Zip created: {zip_path}")
print("Ready for Post-Eval fusion processing.")