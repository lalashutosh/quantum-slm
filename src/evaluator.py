import torch
import time
import json
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util

def load_test_suite(test_version="v3"):
    file_path = f"tests/versions/{test_version}.yaml"
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['benchmarks'], data['version']

def run_inference(model, tokenizer, paper_name, user_input):
    # This MUST match the exact prompt structure used in your excerpt_engine.py
    instruction = (
    "Extract and format the following research summary into the exact following schema:\n"
    "Paper: [title, authors, year]\n"
    "Problem: [gap addressed]\n"
    "Approach: [technical idea]\n"
    "Result: [what was demonstrated]\n"
    "Implication: [meaning for the field]\n"
    "Open questions: [what this doesn't resolve]\n\n"
    f"Paper Name: {paper_name}"
)
    
    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, # Increased for structured output
            temperature=0.1, 
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    response = full_text.split("### Response:\n")[-1].strip()
    return response

def evaluate_build(model_version="v3", test_version="v3"):
    # 1. Logic for Model ID
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = f"models/{model_version}_weights"
    test_cases, version_tag = load_test_suite(test_version)

    print(f"🧪 Evaluating 7B Specialist: {model_version} vs {version_tag}")
    
    # 2. 4-Bit Config for local A2000
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 3. Setup
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Load Models (Quantized)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )

    # Baseline Run (Model without weights)
    print("📥 Running Baseline (General Qwen 7B)...")
    for test in test_cases:
        test['baseline_answer'] = run_inference(model, tokenizer, test['paper_name'], test['user_input'])

    # Fine-Tuned Run (Apply v3 weights)
    print(f"📥 Loading FT Adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("📥 Running Fine-Tuned Evaluation...")
    for test in test_cases:
        test['ft_answer'] = run_inference(model, tokenizer, test['paper_name'], test['user_input'])

    # 5. Analysis
    print("\n" + "="*95)
    print(f"{'TEST ID':<20} | {'RECALL':<8} | {'BASE SIM':<10} | {'FT SIM':<10} | {'GAIN'}")
    print("="*95)
    
    for test in test_cases:
        concept_score = sum(1 for c in test['concepts'] if c.lower() in test['ft_answer'].lower())
        
        v_base = sim_model.encode(test['baseline_answer'], convert_to_tensor=True)
        v_ft = sim_model.encode(test['ft_answer'], convert_to_tensor=True)
        v_ref = sim_model.encode(test['reference'], convert_to_tensor=True)

        sim_base = util.pytorch_cos_sim(v_base, v_ref).item()
        sim_ft = util.pytorch_cos_sim(v_ft, v_ref).item()
        gain = sim_ft - sim_base

        print(f"{test['id']:<20} | {concept_score}/{len(test['concepts']):<6} | {sim_base:.3f}      | {sim_ft:.3f}      | {gain:+.3f}")

    # 6. Save Report
    log_path = f"logs/eval_{model_version}_comprehensive.json"
    with open(log_path, 'w') as f:
        json.dump({"v3_results": test_cases}, f, indent=4)

if __name__ == "__main__":
    evaluate_build("v3", "v3")