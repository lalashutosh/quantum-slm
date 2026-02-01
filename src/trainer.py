# src/trainer.py
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

def train_model(config_name="v1"):
    model_id = "Qwen/Qwen2.5-1.5B-Instruct" 
    dataset_path = f"data/processed/train_{config_name}.jsonl"
    output_dir = f"models/{config_name}_weights"

    # 1. Quantization (Fit on 12GB GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Load Model & Tokenizer
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 3. LoRA Setup
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # 4. Data Preparation (Manual Tokenization for stability)
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024)

    tokenized_dataset = dataset.map(tokenize_function, batched=False)

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Reduced for A2000 safety
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,
        max_steps=40, # Enough for 12 examples
        fp16=True,
        save_total_limit=1,
        report_to="none"
    )

    # 6. The Standard Trainer (Rock Solid)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Knowledge Injection Started...")
    trainer.train()
    
    # 7. Save the expertise
    model.save_pretrained(output_dir)
    print(f"Expertise saved to {output_dir}")

if __name__ == "__main__":
    train_model("v1")