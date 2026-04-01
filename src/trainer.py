from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import os

def train_7b_model(version="v2"):
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path = f"data/processed/train_{version}_excerpts.jsonl"
    output_dir = f"/scratch/project_2017556/quantum-slm/models/{version}_weights"
    os.makedirs(output_dir, exist_ok=True)

    # --- DECISION 1: 4-BIT NF4 QUANTIZATION ---
    # Why: A 7B model in FP16 takes 14GB VRAM just to load (exceeding your 12GB).
    # NF4 (Normal Float 4) is a data type optimized for normally distributed weights.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # Math happens in 16-bit
        bnb_4bit_use_double_quant=True,       # Quantizes the quantization constants (saves ~0.5GB)
    )

    print(f"📦 Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # --- DECISION 2: GRADIENT CHECKPOINTING ---
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)
    # Why: This is the MOST IMPORTANT line for 12GB GPUs.
    # It stops storing all intermediate activations for the backward pass, 
    # re-calculating them on the fly instead. Reduces memory usage by ~30-50%.
    model.gradient_checkpointing_enable() 

    # --- DECISION 3: TARGET ALL LINEAR MODULES (LoRA+) ---
    # Why: Standard LoRA only trains q_proj and v_proj. 
    # For complex physics logic, we need to adapt the MLP layers (gate, up, down).
    peft_config = LoraConfig(
        r=16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # --- DECISION 4: DATA PIPELINE ---
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    # Why: max_length=1024. VRAM usage scales quadratically O(n^2) with sequence length.
    # On an A2000, 2048 length might trigger an OOM during the backward pass.
    tokenized_dataset = dataset.map(tokenize_function, batched=False)

    # --- DECISION 5: AGGRESSIVE GRADIENT ACCUMULATION ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,   # Must be 1 for 7B models on 12GB
        gradient_accumulation_steps=16,  # Why: Simulates a batch size of 16 (1*16).
                                         # Smaller batches cause unstable gradients; 16 is "smooth."
        learning_rate=1e-4,              # Standard for LoRA; higher can "break" the weights.
        logging_steps=1,
        max_steps=50,                    # Number of iterations
        fp16=True,                       # Faster on NVIDIA Ampere (A2000)
        optim="paged_adamw_32bit",       # Why: Offloads optimizer state to CPU RAM if needed.
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("🚀 Starting Logic Injection (7B)...")
    trainer.train()
    
    model.save_pretrained(output_dir)
    print(f"✅ Specialized Weights Saved: {output_dir}")

if __name__ == "__main__":
    import sys
    version = sys.argv[1] if len(sys.argv) > 1 else "v2"
    train_7b_model(version)