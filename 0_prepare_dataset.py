import os
import random
from collections import defaultdict
from dotenv import load_dotenv
from huggingface_hub import login
from datasets import Dataset, DatasetDict
import json

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
INPUT_FILE = os.getenv("INPUT_FILE", "car_repair_hq_550_v2.json")
REPO_ID = os.getenv("REPO_ID", "ssiddiquii/car-repair-hq-550")
PRIVATE_REPO = os.getenv("PRIVATE_REPO", "true").strip().lower() in {"1", "true", "yes", "y"}
SEED = int(os.getenv("SEED", "42"))

GOLDEN_PER_CATEGORY = 6  
VAL_PER_CATEGORY = 5     
REQUIRED_KEYS = {"id", "category", "question", "answer"}

def perfectly_balanced_split(data, seed):
    grouped = defaultdict(list)
    for row in data:
        grouped[row["category"]].append(row)

    min_category_size = min(len(items) for items in grouped.values())
    required_min = GOLDEN_PER_CATEGORY + VAL_PER_CATEGORY + 1
    
    if min_category_size < required_min:
        raise ValueError(f"Smallest category only has {min_category_size} rows. Need at least {required_min}.")

    rng = random.Random(seed)
    train_data, val_data, golden_data = [], [], []

    for category, items in sorted(grouped.items()):
        items_copy = items[:]
        rng.shuffle(items_copy)
        
        # Undersample to match the smallest category
        balanced_items = items_copy[:min_category_size] 
        
        golden_data.extend(balanced_items[:GOLDEN_PER_CATEGORY])
        val_data.extend(balanced_items[GOLDEN_PER_CATEGORY:GOLDEN_PER_CATEGORY + VAL_PER_CATEGORY])
        train_data.extend(balanced_items[GOLDEN_PER_CATEGORY + VAL_PER_CATEGORY:])

    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(golden_data)

    return train_data, val_data, golden_data

if __name__ == "__main__":
    login(token=HF_TOKEN)
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_data, val_data, golden_data = perfectly_balanced_split(data, SEED)

    print("Pushing natively to Hugging Face...")
    dataset_dict = DatasetDict({
        "train": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data),
        "golden_eval": Dataset.from_list(golden_data)
    })
    
    dataset_dict.push_to_hub(REPO_ID, private=PRIVATE_REPO)
    print("✓ Deployment complete.")