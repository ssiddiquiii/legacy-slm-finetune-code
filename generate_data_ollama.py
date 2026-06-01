import os
import json
import time
import random
from tqdm import tqdm
import requests
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# GENERATION CONFIG
# =============================================================
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

TARGET_SAMPLES = 550
BATCH_SIZE = 1
OUTPUT_FILE = "car_repair_hq_550_v2.json"
REQUEST_TIMEOUT = 180
REQUEST_DELAY_SECONDS = 1.5
MAX_EMPTY_ROUNDS = 80

GROQ_OPTIONS = {
    "temperature": 0.6,
    "max_tokens": 420,
}

CATEGORIES = [
    "Engine Diagnostics", "Transmission Systems", "Braking & Hydraulics", 
    "Electrical & Canbus", "Suspension & Steering", "Exhaust Emissions", 
    "Cooling & Thermal", "Fuel Delivery Systems", "HVAC Diagnostics"
]

def _extract_json_array(content):
    """Extract a JSON array/list of QA objects from model output."""
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            for val in result.values():
                if isinstance(val, list):
                    return val
            return [result]
    except json.JSONDecodeError:
        start = content.find('[')
        end = content.rfind(']') + 1
        if start != -1 and end > start:
            try:
                result = json.loads(content[start:end])
                if isinstance(result, list):
                    return result
            except json.JSONDecodeError:
                pass
    return []


def _build_prompt(category):
    prompt = f"""You are an ASE-Certified Master Mechanic. Generate {BATCH_SIZE} technical Question-and-Answer pair(s) about '{category}' for car repair diagnostics.

OUTPUT FORMAT: Valid JSON array with objects containing ONLY these 3 keys: "category", "question", "answer"

QUALITY RULES:
1. Symptoms must be specific (e.g., 'P0171 code with rough idle when cold')
2. Answers must be step-by-step diagnostic workflows
3. Use professional jargon (parasitic draw, fuel trim, stoichiometric)

Example format:
[{{"category": "{category}", "question": "P0171 code with rough idle when cold", "answer": "First step: Check fuel trim..."}}]

Now generate the Q&A pair(s) as valid JSON only."""
    return prompt


def _generate_with_groq(prompt, category, attempt=1, max_retries=5):
    """Generate QA pairs using Groq OpenAI-compatible API."""
    if not GROQ_API_KEY:
        print("\n[ERROR] GROQ_API_KEY is missing in .env")
        return []

    try:
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": "Return only valid JSON. No markdown, no explanation.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "response_format": {"type": "json_object"},
                "temperature": GROQ_OPTIONS["temperature"],
                "max_tokens": GROQ_OPTIONS["max_tokens"],
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            content = ""
            choices = data.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content", "")

            result = _extract_json_array(content)
            if result:
                return result

            if attempt < max_retries:
                cooldown = min(10, attempt * 2)
                print(f"Retrying ({attempt}/{max_retries}) in {cooldown}s...")
                time.sleep(cooldown)
                return _generate_with_groq(prompt, category, attempt + 1, max_retries)
            return []

        print(f"Groq API Error: {response.status_code} - {response.text[:180]}")
        if attempt < max_retries:
            cooldown = min(10, attempt * 2)
            print(f"Retrying ({attempt}/{max_retries}) in {cooldown}s...")
            time.sleep(cooldown)
            return _generate_with_groq(prompt, category, attempt + 1, max_retries)
        return []
    except Exception as e:
        print(f"\n[Error] {str(e)[:100]}")
        if attempt < max_retries:
            cooldown = min(10, attempt * 2)
            print(f"Retrying ({attempt}/{max_retries}) in {cooldown}s...")
            time.sleep(cooldown)
            return _generate_with_groq(prompt, category, attempt + 1, max_retries)
        time.sleep(2)
        return []


def generate_hq_batch(category, attempt=1, max_retries=5):
    """Generate QA batch using Groq."""
    prompt = _build_prompt(category)
    return _generate_with_groq(prompt, category, attempt, max_retries)

# =============================================================
# MAIN GENERATION LOOP
# =============================================================
dataset = []
question_set = set()
if os.path.exists(OUTPUT_FILE):
    try:
        with open(OUTPUT_FILE, "r") as f:
            dataset = json.load(f)
        question_set = {item.get("question", "").strip() for item in dataset if isinstance(item, dict)}
        print(f"State Recovered: {len(dataset)} samples. Resuming...")
    except:
        print("Starting fresh.")

pbar = tqdm(total=TARGET_SAMPLES, initial=len(dataset), desc="Generating Q&A Pairs")
empty_rounds = 0

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Add it to .env before running.")

print(f"Provider: groq | Model: {GROQ_MODEL}")

while len(dataset) < TARGET_SAMPLES:
    category = random.choice(CATEGORIES)
    batch = generate_hq_batch(category)
    
    if batch and isinstance(batch, list):
        valid_samples = []
        for item in batch:
            if "question" in item and "answer" in item:
                normalized_question = item["question"].strip()
                if not normalized_question or normalized_question in question_set:
                    continue
                # Dynamic ID Injection
                current_id = len(dataset) + len(valid_samples) + 1
                resolved_category = item.get("category", category)
                if resolved_category not in CATEGORIES:
                    resolved_category = category
                formatted_item = {
                    "id": current_id,
                    "category": resolved_category,
                    "question": normalized_question,
                    "answer": item["answer"]
                }
                valid_samples.append(formatted_item)
                question_set.add(normalized_question)

        if valid_samples:
            empty_rounds = 0
        else:
            empty_rounds += 1

        dataset.extend(valid_samples)
        pbar.update(len(valid_samples))
        
        # Save progress
        with open(OUTPUT_FILE, "w") as f:
            json.dump(dataset, f, indent=4)

    else:
        empty_rounds += 1

    if empty_rounds >= MAX_EMPTY_ROUNDS:
        print(
            f"\nStopped after {MAX_EMPTY_ROUNDS} empty rounds. "
            "Model is returning invalid/duplicate output repeatedly."
        )
        break
    
    time.sleep(REQUEST_DELAY_SECONDS)

pbar.close()

# Deduplication and Re-indexing
unique_questions = set()
final_clean_data = []

for item in dataset:
    if item["question"] not in unique_questions:
        unique_questions.add(item["question"])
        final_clean_data.append(item)

if len(final_clean_data) < TARGET_SAMPLES:
    print(
        f"Top-up mode: {TARGET_SAMPLES - len(final_clean_data)} more unique questions needed."
    )

    empty_topup_rounds = 0
    while len(final_clean_data) < TARGET_SAMPLES:
        category = random.choice(CATEGORIES)
        batch = generate_hq_batch(category)

        added_any = False
        if batch and isinstance(batch, list):
            for item in batch:
                if "question" not in item or "answer" not in item:
                    continue

                normalized_question = item["question"].strip()
                if not normalized_question or normalized_question in unique_questions:
                    continue

                current_id = len(final_clean_data) + 1
                resolved_category = item.get("category", category)
                if resolved_category not in CATEGORIES:
                    resolved_category = category

                formatted_item = {
                    "id": current_id,
                    "category": resolved_category,
                    "question": normalized_question,
                    "answer": item["answer"],
                }
                final_clean_data.append(formatted_item)
                unique_questions.add(normalized_question)
                added_any = True

                if len(final_clean_data) >= TARGET_SAMPLES:
                    break

        if added_any:
            empty_topup_rounds = 0
            with open(OUTPUT_FILE, "w") as f:
                json.dump(final_clean_data, f, indent=4)
        else:
            empty_topup_rounds += 1

        if empty_topup_rounds >= MAX_EMPTY_ROUNDS:
            print(
                f"\nStopped top-up after {MAX_EMPTY_ROUNDS} empty rounds. "
                "Model is returning invalid/duplicate output repeatedly."
            )
            break

        time.sleep(REQUEST_DELAY_SECONDS)

for index, item in enumerate(final_clean_data):
    item["id"] = index + 1

with open(OUTPUT_FILE, "w") as f:
    json.dump(final_clean_data, f, indent=4)

print(f"\n✅ Generation Complete! {len(final_clean_data)} questions in {OUTPUT_FILE}")
