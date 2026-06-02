# Cell 1: Configure VRAM memory segments and deploy stable evaluation stack
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

print("Installing stable evaluation stack...")
!pip uninstall -y torchao -q
!pip install -q -U unsloth
!pip install -q -U "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q -U "trl>=0.18.2,<=0.24.0" "datasets>=3.4.1,<4.4.0" peft accelerate bitsandbytes huggingface_hub sentencepiece "protobuf>=3.20.3,<6.0.0" deepeval litellm nest_asyncio

print("\n" + "="*60)
print("✅ INSTALLATIONS COMPLETE")
print("="*60)

# Cell 2: Verify framework versions and target hardware telemetry silently
import logging

# SILENCE EARLY FRAMEWORK CHATTER: Suppresses multi-modal placeholder warning hooks on startup
logging.getLogger("unsloth").setLevel(logging.ERROR)

import unsloth # MUST BE FIRST AMONG DEEP LEARNING LIBRARIES
import torch, transformers, peft, trl, accelerate, bitsandbytes, sys, datasets

print("=" * 60)
print("KAGGLE ENVIRONMENT CHECK")
print("=" * 60)
print(f"Python:        {sys.version.split()[0]}")
print(f"Torch:         {torch.__version__}")
print(f"Transformers:  {transformers.__version__}")
print(f"Unsloth:       {unsloth.__version__}")

if torch.cuda.is_available():
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
else:
    print("\n✗ GPU NOT detected")

# Cell 3: Bind environment credentials, configurations, and baseline verification files
import json
import os
from pathlib import Path
from dataclasses import dataclass
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

@dataclass
class PostEvalConfig:
    base_model: str = "unsloth/gemma-4-E2B-it"
    adapter_dir: str = "/kaggle/input/datasets/sameedsiddiqui0347/qlora-adapters" 
    baseline_dir: str = "/kaggle/input/datasets/sameedsiddiqui0347/baseline-artifacts" 
    
    eval_model: str = "groq/llama-3.1-8b-instant"
    eval_threshold: float = 0.7
    max_new_tokens: int = 256
    max_input_length: int = 1024
    work_dir: str = "/kaggle/working"

CFG = PostEvalConfig()
WORK = Path(CFG.work_dir)
RESULTS = WORK / "post_eval_results_qlora"
RESULTS.mkdir(parents=True, exist_ok=True)

FINETUNED_ANSWERS_PATH = RESULTS / "finetuned_answers.json"
POSTEVAL_SCORES_PATH = RESULTS / "post_eval_scores.json"

print("=" * 60)
print("LOADING BASELINE ARTIFACTS")
print("=" * 60)

baseline_answers_path = Path(CFG.baseline_dir) / "baseline_answers_v2.json"
baseline_scores_path = Path(CFG.baseline_dir) / "baseline_scores_v2.json"

assert baseline_answers_path.exists(), f"CRITICAL: Cannot find {baseline_answers_path}. Check Kaggle Dataset mount."
assert baseline_scores_path.exists(), f"CRITICAL: Cannot find {baseline_scores_path}. Check Kaggle Dataset mount."
assert Path(CFG.adapter_dir).exists(), f"CRITICAL: Cannot find LoRA adapters at {CFG.adapter_dir}. Check Kaggle Dataset mount."

with open(baseline_answers_path) as f: baseline_answers = json.load(f)
with open(baseline_scores_path) as f: baseline_scores = json.load(f)

print(f"\n✓ Baseline loaded: {len(baseline_answers)} entries")

# Cell 4: Load target adapters onto 4-bit foundation weights with active inference routing
import gc; gc.collect(); torch.cuda.empty_cache()
from unsloth import FastModel

print("=" * 60)
print("LOADING FINE-TUNED MODEL (QLoRA 4-BIT)")
print("=" * 60)

model, tokenizer = FastModel.from_pretrained(
    model_name=CFG.adapter_dir,
    max_seq_length=CFG.max_input_length,
    load_in_4bit=True,               
    dtype=torch.float16,             
    full_finetuning=False,
    token=HF_TOKEN,
)

# Pre-emptively disable caching to keep generation loops stable
model.config.use_cache = False

lora_count = sum(1 for name, _ in model.named_modules() if 'lora' in name.lower())
print(f"\n✓ Model + adapters loaded. LoRA modules detected: {lora_count}")
assert lora_count > 0, "No LoRA modules — adapter not attached"

try: FastModel.for_inference(model)
except AttributeError: pass
model.eval()

# Cell 5: Execute deterministic inference using aligned Gemma-4 system token boundaries
from tqdm.auto import tqdm

SYSTEM_PROMPT = (
    "You are an expert car repair assistant. Answer the user's question concisely "
    "and accurately. Be technically precise about parts, diagnostics, and procedures."
)

@torch.no_grad()
def generate_answer(question: str) -> str:
    # FIXED: Restored system-user dictionary separation to reflect fine-tuning data layout
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": question.strip()}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CFG.max_input_length,
        add_special_tokens=False, # Prevents double BOS token errors
    ).to(model.device)
    
    outputs = model.generate(
        **inputs, max_new_tokens=CFG.max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

print("=" * 60)
print(f"GENERATING FINE-TUNED ANSWERS ({len(baseline_answers)} questions)")
print("=" * 60)

finetuned_results = []
for i, item in enumerate(tqdm(baseline_answers, desc="Generating")):
    finetuned_results.append({
        "idx": i,
        "question": item["question"],
        "expected_answer": item["expected_answer"],
        "baseline_answer": item["generated_answer"],
        "finetuned_answer": generate_answer(item["question"]),
        "context": item.get("context", ""),
    })

with open(FINETUNED_ANSWERS_PATH, 'w') as f: json.dump(finetuned_results, f, indent=2, ensure_ascii=False)
print(f"\n✓ Saved {len(finetuned_results)} fine-tuned answers")

# Cell 6: Instantiate thread-safe judge client and measure post-finetune metrics
import time, threading, re
import litellm
import nest_asyncio
from collections import defaultdict
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import AnswerRelevancyMetric, GEval

# DYNAMIC IMPORT SHIM: Resolves internal versioning drift in DeepEval
from deepeval.test_case import LLMTestCase
try:
    from deeval.test_case import SingleTurnParams
    print("✓ DeepEval modern namespace 'SingleTurnParams' loaded.")
except ImportError:
    from deepeval.test_case import LLMTestCaseParams as SingleTurnParams
    print("⚠ DeepEval legacy namespace resolved. Mapping 'LLMTestCaseParams' as fallback.")

# KAGGLE NOTEBOOK FIX: Prevent asyncio loop crash
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
                if attempt == retries - 1: raise RuntimeError(f"Judge failed: {e}")
                time.sleep(5)

    def load_model(self): return self.model_name
    def generate(self, prompt, schema=None): return self._call(prompt, schema)
    async def a_generate(self, prompt, schema=None): return self._call(prompt, schema)
    def get_model_name(self): return self.model_name

judge = LiteLLMJudge(model_name=CFG.eval_model, api_key=GROQ_API_KEY)
metrics = [
    AnswerRelevancyMetric(threshold=CFG.eval_threshold, model=judge, async_mode=False),
    GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual_output is a factually correct, technically accurate "
            "answer to the input question, using expected_output as ground truth. For car repair, "
            "check: correct parts, diagnostics, and procedures."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=CFG.eval_threshold, model=judge, async_mode=False
    ),
]

def to_str(value):
    if value is None: return ""
    if isinstance(value, str): return value
    if isinstance(value, list): return "\n".join(str(item).strip() for item in value if item)
    return str(value)

test_cases = [
    LLMTestCase(
        input=to_str(r["question"]), actual_output=to_str(r["finetuned_answer"]), expected_output=to_str(r["expected_answer"])
    ) for r in finetuned_results
]

scores_by_metric = defaultdict(list)
print(f"\nScoring {len(test_cases)} cases...")
for i, tc in enumerate(tqdm(test_cases, desc="Scoring")):
    for metric in metrics:
        mname = getattr(metric, 'name', None) or metric.__class__.__name__
        try:
            metric.measure(tc)
            score = getattr(metric, 'score', None)
            if score is not None: scores_by_metric[mname].append(score)
        except Exception as e: 
            print(f"\n  [case {i}] {mname} failed: {str(e)[:100]}")

summary = {
    "model": "google/gemma-4-E2B-it + QLoRA",
    "judge": CFG.eval_model,
    "n_samples": len(test_cases),
    "metrics": {
        name: {
            "avg_score": (sum(v) / len(v)) if len(v) > 0 else 0.0, 
            "pass_rate": (sum(1 for x in v if x >= CFG.eval_threshold) / len(v)) if len(v) > 0 else 0.0
        } for name, v in scores_by_metric.items()
    }
}
with open(POSTEVAL_SCORES_PATH, 'w') as f: json.dump(summary, f, indent=2)
print(f"\n✓ Saved post-eval summary → {POSTEVAL_SCORES_PATH}")

# Cell 7: Compile comparison metrics, generate verdict, and archive performance records
import shutil
DELTA_REPORT_PATH = RESULTS / "detailed_delta_report.json"

delta_report = {"metrics": {}, "overall_summary": {}}
print("\n" + "=" * 80)
print("DELTA REPORT — Gemma-4 E2B-it: Baseline vs Fine-Tuned (QLoRA)")
print("=" * 80)
print(f"{'Metric':<30} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>10} {'Relative':>10}")
print("-" * 80)

for metric_name, post_data in summary["metrics"].items():
    base_data = baseline_scores["metrics"].get(metric_name, {})
    base_avg = base_data.get("avg_score", 0)
    post_avg = post_data.get("avg_score", 0)
    delta = post_avg - base_avg
    rel_change_pct = (delta / base_avg * 100) if base_avg > 0 else 0

    sign = "+" if delta >= 0 else ""
    rel_sign = "+" if rel_change_pct >= 0 else ""
    print(f"{metric_name:<30} {base_avg:>10.3f} {post_avg:>12.3f} {sign}{delta:>9.3f} {rel_sign}{rel_change_pct:>8.1f}%")

    delta_report["metrics"][metric_name] = {
        "baseline": {"avg_score": base_avg, "pass_rate": base_data.get("pass_rate", 0)},
        "fine_tuned": {"avg_score": post_avg, "pass_rate": post_data.get("pass_rate", 0)},
        "delta": {"avg_score_change": round(delta, 4), "relative_change_percent": round(rel_change_pct, 2)}
    }

print("-" * 80)
print(f"{'Pass Rate (≥0.7)':<30} {'Baseline':>10} {'Fine-Tuned':>12} {'Delta':>10}")
print("-" * 72)
for metric_name, post_data in summary["metrics"].items():
    base_data = baseline_scores["metrics"].get(metric_name, {})
    base_pass = base_data.get("pass_rate", 0)
    post_pass = post_data.get("pass_rate", 0)
    delta_pass = post_pass - base_pass
    sign = "+" if delta_pass >= 0 else ""
    print(f"{metric_name:<30} {base_pass:>9.1%} {post_pass:>11.1%} {sign}{delta_pass:>8.1%}")
    delta_report["metrics"][metric_name]["delta"]["pass_rate_change"] = round(delta_pass, 4)

c_abs = delta_report["metrics"].get("Correctness", {}).get("delta", {}).get("avg_score_change", 0)
if c_abs >= 0.15: verdict = "SIGNIFICANT IMPROVEMENT"
elif c_abs >= 0.05: verdict = "MODERATE IMPROVEMENT"
else: verdict = "MARGINAL / FLAT"

print(f"\n{'=' * 80}")
print("VERDICT")
print(f"{'=' * 80}")
print(f"   Overall assessment:      {verdict}")
print(f"{'=' * 80}")

with open(DELTA_REPORT_PATH, 'w') as f: json.dump(delta_report, f, indent=2)

zip_path = f"{RESULTS}.zip"
shutil.make_archive(base_name=str(RESULTS), format="zip", root_dir=str(RESULTS))
print(f"\n✓ Pipeline complete. Artifacts zipped → {zip_path}")

