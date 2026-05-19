# Custom AI Training: Gemma-4 Car Repair Expert

This repository contains a complete, end-to-end pipeline for training a general AI model (**Gemma-4 E2B-it**) to become a highly accurate car repair expert. 

Because AI training usually requires massive servers, this architecture is specially engineered to run on free, memory-constrained edge hardware (**16GB Kaggle T4 GPUs**) using QLoRA and Unsloth optimizations.

---

## 🧠 The Architecture Flow

Our pipeline prioritizes **Mathematical Balance** and **Strict Evaluation**. It is divided into 4 isolated stages, exactly as outlined in our system architecture.

### Stage 0: Data Engineering (Local Machine)
We do not feed raw JSON files directly into the training loop. We process the data locally to ensure perfect hygiene and mathematical balance before it touches the cloud.
* **Process:** The script ingests `car-repair-hq-550.json`, runs strict data validation, and removes duplicates.
* **Balancing:** It performs a "Stratified Split" to ensure every car part category (Engine, Brakes, etc.) is perfectly equal. 
* **Deployment:** The clean data is pushed to the **Hugging Face Hub** in 3 compressed Parquet splits: `train`, `val`, and `golden_eval`.

### Stage 1: Pre-Evaluation (Kaggle)
Before we teach the AI anything, we test its baseline knowledge.
* **Process:** We load the raw Gemma-4 base model in 4-bit memory-saving mode. We ask it the questions from our `golden_eval` dataset.
* **The Judge:** Instead of a human reading the answers, we use an automated LLM-as-a-Judge (**Groq Llama-3.1-8b** via the **DeepEval** framework) to score the AI's accuracy.
* **Output:** `baseline_artifacts` (the starting scores).

### Stage 2: QLoRA Fine-Tuning (Kaggle)
This is where the actual learning happens.
* **Process:** We feed the `train` and `val` datasets into the base model. 
* **Optimizations:** To prevent the Kaggle T4 GPU from crashing, we freeze the base model in 4-bit and use **Unsloth** with a **Paged AdamW 8-bit** optimizer. This trains the new knowledge into a lightweight "adapter" rather than changing the whole model.
* **Output:** `lora_adapters.zip` (the new car repair brain).

### Stage 3: Post-Evaluation & ROI (Kaggle)
We prove mathematically that the training worked.
* **Process:** We create a "Fused Model" by attaching our new LoRA adapters to the base model. We give it the exact same `golden_eval` test from Stage 1.
* **The Judge:** The Groq judge scores the new answers.
* **Output:** A **Delta Report** comparing the Baseline vs. Finetuned scores, proving the exact percentage of improvement.

---

## ⚙️ How to Run This Project

### 1. Requirements
You will need three free accounts to connect the pipeline:
1. **Hugging Face:** Create an account and get a "Write" Access Token to store the datasets and models.
2. **Groq Cloud:** Get an API Key to power the automated DeepEval judge.
3. **Kaggle:** Add both `HF_TOKEN` and `GROQ_API_KEY` to your Kaggle Secrets.

### 2. Execution Steps
Run the files in this exact order:
1. **Run `0_prepare_dataset.py` (Locally):** This validates the data and pushes the 3 splits to Hugging Face.
2. **Run `1_pre_eval.py` (Kaggle):** Downloads the `golden_eval` split and generates your baseline scores. Mount these outputs as a Kaggle dataset.
3. **Run `2_finetune.py` (Kaggle):** Trains the model and generates the LoRA adapters. Mount these outputs as a Kaggle dataset.
4. **Run `3_post_eval.py` (Kaggle):** Links the artifacts from Steps 2 & 3 to calculate your final success report.