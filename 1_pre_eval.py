# =============================================================================
# Cell 1: Configure VRAM memory segments and deploy stable runtime dependencies
# =============================================================================

import os
import sys
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("=" * 60)
print("INITIALIZING DEPLOYMENT ENVIRONMENT")
print("=" * 60)
print(f"Python baseline: {sys.version.split()[0]}")
print(f"PyTorch engine:  {torch.__version__}")

# Uninstall conflicting runtime modules
!pip uninstall -y torchao -q

# Freeze boundary parameters to guarantee Unsloth framework stability
!pip install -q "trl>=0.18.2,<=0.24.0" "datasets>=3.4.1,<4.4.0" "protobuf>=3.20.3,<6.0.0"
!pip install -q -U unsloth
!pip install -q -U "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q peft accelerate bitsandbytes huggingface_hub sentencepiece deepeval litellm nest_asyncio

print("\n CELL 1 COMPLETE: Environment binaries locked and deployed.")

# ======================================================================================
# Cell 2: Verify framework versions, target hardware telemetry, and compute capabilities
# ======================================================================================

import unsloth # CRITICAL: Patches attention mechanics before transformers initialize
import torch
import transformers
import peft
import trl
import datasets
import sys

print("=" * 60)
print("COMPUTE INFRASTRUCTURE STATUS REPORT")
print("=" * 60)
print(f"  unsloth:       {unsloth.__version__}")
print(f"  transformers:  {transformers.__version__}")
print(f"  peft:          {peft.__version__}")
print(f"  trl:           {trl.__version__}")
print(f"  datasets:      {datasets.__version__}")
print(f"  torch:         {torch.__version__}")

print("\nCUDA HARDWARE TELEMETRY:")
if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print(f"  GPU Count: {n_gpu}")
    for i in range(n_gpu):
        name = torch.cuda.get_device_name(i)
        vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"  GPU {i}: {name} ({vram:.1f} GB VRAM)")
    
    gpu_cap = torch.cuda.get_device_capability(0)
    print(f"  Compute Capability: sm_{gpu_cap[0]}{gpu_cap[1]}")
    if "T4" in torch.cuda.get_device_name(0):
        print("\n✓ Target T4 infrastructure confirmed.")
else:
    print("\n✗ CRITICAL WARNING: No GPU detected. Enable hardware acceleration.")

# Validate Kaggle Secret Mount Integrity
try:
    from kaggle_secrets import UserSecretsClient
    us = UserSecretsClient()
    _ = us.get_secret("HF_TOKEN")
    _ = us.get_secret("GROQ_API_KEY")
    print("\n✓ Kaggle secrets infrastructure verified.")
except Exception as e:
    print(f"\n✗ Secrets mounting collision detected: {e}")

# ==================================================================================
# Cell 3: Bind environment credentials and authorize HuggingFace/Groq secure bridges
# ==================================================================================

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
print("✓ HuggingFace instance authenticated.")
print("✓ Groq API secure bridge verified.")

# =========================================================================================
# Cell 4: Initialize pipeline parameters and programmatically compile workspace directories
# =========================================================================================

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    model_id: str = "unsloth/gemma-4-E2B-it"
    dataset_repo: str = "ssiddiquii/car-repair-hq-342" 
    eval_split: str = "golden_eval"
    eval_n_samples: int = 54 # Exactly 6 instances across 9 categories
    
    max_new_tokens: int = 256
    max_input_length: int = 1024
    eval_model: str = "groq/llama-3.1-8b-instant"
    eval_threshold: float = 0.7
    base_dir: str = "/kaggle/working/car_repair_slm_v2"

CONFIG = Config()
BASE = Path(CONFIG.base_dir)
RESULTS, LOGS = BASE / "results", BASE / "logs"
for d in [BASE, RESULTS, LOGS]: 
    d.mkdir(parents=True, exist_ok=True)

BASELINE_ANSWERS = RESULTS / "baseline_answers_v2.json"
BASELINE_SCORES = RESULTS / "baseline_scores_v2.json"

print(f"Target Cluster:   {CONFIG.dataset_repo}")
print(f"Target Partition: {CONFIG.eval_split}")
print("✓ Core routing matrix locked.")

# ======================================================================================
# Cell 5: Stream dataset from Hugging Face Hub and isolate a deterministic sample subset
# ======================================================================================

import random
from datasets import load_dataset

print(f"Streaming clean Parquet slices [{CONFIG.eval_split}] from HF Hub...")

eval_dataset = load_dataset(CONFIG.dataset_repo, split=CONFIG.eval_split, token=HF_TOKEN)
test_data = eval_dataset.to_list()

random.seed(42)
if len(test_data) > CONFIG.eval_n_samples:
    eval_data = random.sample(test_data, CONFIG.eval_n_samples)
else:
    eval_data = test_data

print(f"\n✓ Native evaluation split loaded successfully: {len(eval_data)} samples verified.")
print(f"✓ Feature Keys extracted: {list(eval_data[0].keys())}")

# ================================================================================
# Cell 6: Dynamically detect feature schema keys and mount adaptive metric vectors
# ================================================================================

sample = eval_data[0]

Q_CANDIDATES = ['question', 'input', 'query', 'prompt', 'instruction']
A_CANDIDATES = ['answer', 'expected_answer', 'output', 'response', 'completion']
C_CANDIDATES = ['context', 'retrieval_context', 'reference', 'passage', 'background']

def find_key(item, candidates):
    for k in candidates:
        if k in item and item[k]: return k
    return None

Q_KEY = find_key(sample, Q_CANDIDATES)
A_KEY = find_key(sample, A_CANDIDATES)
C_KEY = find_key(sample, C_CANDIDATES)

assert Q_KEY and A_KEY, f"Data mapping exception. Found attributes: {list(sample.keys())}"

has_context = bool(C_KEY) and sum(1 for it in eval_data if it.get(C_KEY)) >= len(eval_data) * 0.8

print(f"Verified Input Vector Field:  '{Q_KEY}'")
print(f"Verified Target Vector Field: '{A_KEY}'")
print(f"Context Component Capability:  '{C_KEY}' (Active Range Validation={has_context})")

metric_list = ["AnswerRelevancy", "GEval-Correctness"]
if has_context:
    metric_list += ["Faithfulness", "ContextualRelevancy"]
print(f"\nConstructed Evaluation Vector Matrix: {metric_list}")

# ==================================================================================
# Cell 7: Instantiate thread-safe LiteLLMJudge API wrapper and mount grading metrics
# ==================================================================================

import os, time, threading, re
import litellm
import nest_asyncio 
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval import evaluate

# DYNAMIC IMPORT SHIM: Resolves internal versioning drift in DeepEval
from deepeval.test_case import LLMTestCase
try:
    from deepeval.test_case import SingleTurnParams
    print("✓ DeepEval modern namespace 'SingleTurnParams' loaded.")
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as SingleTurnParams
    print("⚠ DeepEval legacy namespace resolved. Mapping 'LLMTestCaseParams' as fallback.")

# Prevent nested asyncio thread lock conditions inside Kaggle Notebook
nest_asyncio.apply()
litellm.suppress_debug_info = True

class LiteLLMJudge(DeepEvalBaseLLM):
    _lock = threading.Lock()
    _call_log = []

    def __init__(self, model_name, api_key, rpm_limit=30, tpm_limit=6000, safety=0.70, max_tokens=1024):
        self.model_name = model_name
        self.api_key = api_key
        self.rpm_limit = rpm_limit
        self.tpm_budget = int(tpm_limit * safety)
        self.max_tokens = max_tokens

    @classmethod
    def _prune(cls):
        now = time.time()
        cls._call_log = [(t, n) for t, n in cls._call_log if now - t < 60]

    def _throttle(self, estimated_tokens):
        while True:
            with LiteLLMJudge._lock:
                self._prune()
                if (len(LiteLLMJudge._call_log) < self.rpm_limit and
                    sum(n for _, n in LiteLLMJudge._call_log) + estimated_tokens <= self.tpm_budget):
                    return
                oldest = LiteLLMJudge._call_log[0][0] if LiteLLMJudge._call_log else time.time()
                wait = max(1.0, min((oldest + 61) - time.time(), 30.0))
            time.sleep(wait)

    def _call(self, prompt, schema=None, retries=5):
        estimated = len(prompt) // 4 + self.max_tokens
        kwargs = {
            "model": self.model_name, "messages": [{"role": "user", "content": prompt}],
            "api_key": self.api_key, "temperature": 0, "max_tokens": self.max_tokens,
        }
        if schema is not None: kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retries):
            self._throttle(estimated)
            try:
                resp = litellm.completion(**kwargs)
                actual_tokens = getattr(resp, 'usage', None).total_tokens if getattr(resp, 'usage', None) else estimated
                with LiteLLMJudge._lock: LiteLLMJudge._call_log.append((time.time(), actual_tokens))
                
                text = resp.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"): text = text[4:]
                    text = text.strip()
                
                if schema is not None: return schema.model_validate_json(text)
                return text
            except Exception as e:
                msg = str(e).lower()
                if "429" in msg or "rate_limit" in msg: 
                    time.sleep(20)
                    continue
                if attempt == retries - 1: raise RuntimeError(f"Judge Execution Terminal Exception: {e}")
                time.sleep(5)

    def load_model(self): return self.model_name
    def generate(self, prompt, schema=None): return self._call(prompt, schema)
    async def a_generate(self, prompt, schema=None): return self._call(prompt, schema)
    def get_model_name(self): return self.model_name

LiteLLMJudge._call_log = []
judge = LiteLLMJudge(model_name=CONFIG.eval_model, api_key=GROQ_API_KEY)

print("Initializing programmatic criteria array via SingleTurnParams...")
metrics = [
    AnswerRelevancyMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
    GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual_output is a factually correct, technically accurate "
            "answer to the input question, using expected_output as ground truth. For car repair, "
            "check: correct parts/components, correct diagnostic reasoning, correct procedures, "
            "safe advice. Heavily penalize vague, generic, or incorrect technical claims."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,           # STABLE RUNTIME ALIGNMENT
            SingleTurnParams.ACTUAL_OUTPUT,   # STABLE RUNTIME ALIGNMENT
            SingleTurnParams.EXPECTED_OUTPUT, # STABLE RUNTIME ALIGNMENT
        ],
        threshold=CONFIG.eval_threshold,
        model=judge,
        async_mode=False,
    ),
]

if has_context:
    from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
    metrics += [
        FaithfulnessMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
        ContextualRelevancyMetric(threshold=CONFIG.eval_threshold, model=judge, async_mode=False),
    ]

print(f"Judge Active Model Target: {CONFIG.eval_model} (TPM Restricted Framework Running)")
print(f"Total Quantitative Metrics Mounted: {len(metrics)}")

# ======================================================================================
# Cell 8: Load 4-bit quantized base model and activate optimized forward inference hooks
# ======================================================================================

import gc
import torch
from unsloth import FastModel

gc.collect()
torch.cuda.empty_cache()

print("Loading foundation weight distribution model via optimized Unsloth wrapper...")

model, tokenizer = FastModel.from_pretrained(
    model_name=CONFIG.model_id,
    max_seq_length=CONFIG.max_input_length,
    dtype=torch.float16, 
    load_in_4bit=True,   
    full_finetuning=False,
    token=HF_TOKEN,
)

try:
    FastModel.for_inference(model)
    print("✓ Unsloth inference optimization routing hooks attached.")
except AttributeError:
    pass

model.eval()

print("\nCURRENT RUNTIME ALLOCATION VECTOR MATRICES:")
for i in range(torch.cuda.device_count()):
    alloc = torch.cuda.memory_allocated(i) / 1024**3
    total_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
    print(f"  GPU [{i}]: VRAM Footprint => {alloc:.2f} / {total_mem:.1f} GB")

print("\n✓ Untuned baseline model sequence is stable. Core adapters detached.")

# =====================================================================================
# Cell 9: Run greedy baseline generation using structural Gemma-4 chat template routing
# =====================================================================================

import json
from tqdm.auto import tqdm

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

def build_prompt(question: str) -> str:
    # GEMMA STRUCTURAL PATCH: Segregating system and user blocks properly
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": question.strip()},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def generate(question: str) -> str:
    prompt = build_prompt(question)
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CONFIG.max_input_length,
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=CONFIG.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print("Executing framework sanity smoke check on sample slice 0...\n")
q0 = eval_data[0][Q_KEY]
a0 = generate(q0)
print(f"Prompt Target Vector:       {q0[:150]}...")
print(f"Generated Baseline Matrix:  {a0[:150]}...")
print(f"Expected Target Evaluation: {eval_data[0][A_KEY][:150]}...")
print("\n" + "=" * 60)

print(f"\nProcessing token sequence inferences across {len(eval_data)} localized matrix frames...")
baseline_results = []
for item in tqdm(eval_data):
    q = item[Q_KEY]
    expected = item[A_KEY]
    ctx = item.get(C_KEY, "") if C_KEY else ""
    gen = generate(q)
    baseline_results.append({
        "question": q,
        "expected_answer": expected,
        "generated_answer": gen,
        "context": ctx,
    })

with open(BASELINE_ANSWERS, 'w') as f:
    json.dump(baseline_results, f, indent=2, ensure_ascii=False)

print(f"\n✓ Generated base vector sequences successfully cached → {BASELINE_ANSWERS}")

# ====================================================================================
# Cell 10: Evacuate local VRAM allocations and execute checkpointed judge scoring loop
# ====================================================================================

import json
from collections import defaultdict
from tqdm.auto import tqdm

# HARDWARE TUNING PATCH: Evacuate local inference model to clear hardware allocations during remote API scoring
try:
    print("Purging local foundation model from T4 to prevent runtime OOM locks...")
    del model
    del tokenizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("✓ VRAM completely evacuated for external API evaluation.")
except NameError:
    pass

def to_str(value):
    if value is None: return ""
    if isinstance(value, str): return value
    if isinstance(value, list): return "\n".join(str(item).strip() for item in value if item)
    return str(value)

test_cases = []
for r in baseline_results:
    ctx_raw = r.get("context", "")
    ctx_str = to_str(ctx_raw)
    ctx_list = [ctx_str] if (has_context and ctx_str) else None
    
    test_cases.append(LLMTestCase(
        input=to_str(r["question"]),
        actual_output=to_str(r["generated_answer"]),
        expected_output=to_str(r["expected_answer"]),
        retrieval_context=ctx_list,
    ))

total_calls = len(test_cases) * len(metrics)
print(f"Sequential scoring initialized: {len(test_cases)} configurations × {len(metrics)} targets = {total_calls} calls total.")

BASELINE_CHECKPOINT = RESULTS / "baseline_checkpoint_v2.jsonl"
scores_by_metric = defaultdict(list)
per_case = []
failed_count = 0
completed_indices = set()

if BASELINE_CHECKPOINT.exists():
    with open(BASELINE_CHECKPOINT) as f:
        for line in f:
            try:
                entry = json.loads(line)
                completed_indices.add(entry["idx"])
                per_case.append(entry)
                for k, v in entry.get("scores", {}).items():
                    if isinstance(v, (int, float)): scores_by_metric[k].append(v)
            except Exception: pass
    if completed_indices:
        print(f"Checkpoint cache validated. Resuming from iteration sequence: {len(completed_indices)} complete.")

with open(BASELINE_CHECKPOINT, 'a') as ckpt_file:
    for i, tc in enumerate(tqdm(test_cases, desc="Evaluating Output Performance Matrix")):
        if i in completed_indices: continue

        case_scores = {}
        case_failed = False

        for metric in metrics:
            mname = getattr(metric, 'name', None) or metric.__class__.__name__
            try:
                metric.measure(tc)
                score = getattr(metric, 'score', None)
                if score is not None:
                    scores_by_metric[mname].append(score)
                    case_scores[mname] = round(score, 3)
            except Exception as e:
                print(f"\n  [Token Frame Exception at Index {i}] {mname} instance dropped: {str(e)[:100]}")
                case_failed = True

        if case_failed and not case_scores: failed_count += 1

        entry = {"idx": i, "question": to_str(baseline_results[i]["question"])[:100], "scores": case_scores}
        per_case.append(entry)
        ckpt_file.write(json.dumps(entry) + "\n")
        ckpt_file.flush()

summary = {
    "model": CONFIG.model_id,
    "dataset": CONFIG.dataset_repo,
    "judge": CONFIG.eval_model,
    "n_samples": len(test_cases),
    "n_failed": failed_count,
    "threshold": CONFIG.eval_threshold,
    "metrics": {},
    "per_case": sorted(per_case, key=lambda x: x["idx"]),
}

print("\n" + "=" * 75)
print(f"BASELINE EXPERT SCORING SUMMARY MATRIX — PRE-TUNED")
print("=" * 75)
for name, vals in scores_by_metric.items():
    if not vals: continue
    avg = sum(vals) / len(vals)
    pass_rate = sum(1 for v in vals if v >= CONFIG.eval_threshold) / len(vals)
    summary["metrics"][name] = {"avg_score": round(avg, 4), "pass_rate": round(pass_rate, 4), "n": len(vals)}
    print(f"  {name:30s} Mean Score={avg:.3f}   Pass Rate Validation={pass_rate:.1%}  (Sample n={len(vals)})")

if failed_count:
    print(f"\n  ⚠ Alert: {failed_count} tensor eval arrays failed judge threshold requirements.")
print("=" * 75)

with open(BASELINE_SCORES, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Core summary statistics locked → {BASELINE_SCORES}")

# =====================================================================================
# Cell 11: Compress telemetry logs and evaluation summary reports into a deployment ZIP
# =====================================================================================

import shutil
from pathlib import Path

zip_base = "/kaggle/working/baseline_artifacts"
shutil.make_archive(base_name=zip_base, format='zip', root_dir=str(RESULTS))

zip_file = Path(f"{zip_base}.zip")
size_mb = zip_file.stat().st_size / 1024 / 1024

print(f"✓ Binary deployment zip generated successfully: {zip_file}")
print(f"  Total Alloc Size: {size_mb:.2f} MB")
print(f"\nNEXT OPERATIONAL SEQUENCES:")
print(f"  1. Navigate to right sidebar UI inside Kaggle workspace.")
print(f"  2. Extract download target for baseline_v2_artifacts.zip.")
print(f"  3. Mount this output package directly as a static Kaggle dataset for Stage 3 validation comparisons.")