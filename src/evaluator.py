import torch
import time
import json
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util



def load_test_suite(version="v1_2"):
    with open(f"tests/registry.yaml", 'r') as f:
        data = yaml.safe_load(f)
    return data['benchmarks'], data['version']
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Return the actual tests list and the version metadata
    return data['tests'], data['version']

def run_inference(model, tokenizer, question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250, # Increased for more detailed research reasoning
            temperature=0.1, 
            do_sample=True
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("### Response:\n")[-1].strip()
    return response

def evaluate_build(model_version="v1", test_version="v1_1"):
    # 1. Load versioned tests from JSON
    test_cases, version_tag = load_test_suite(test_version)

    print(f"🧪 Running Evaluation: Model {model_version} vs Test Suite {version_tag}")
    
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = f"models/{model_version}_weights"
    
    # 2. Setup Tools
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Load BASE Model
    print("📥 Loading Baseline Model for comparison...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # Run Suite on Baseline
    print("\n--- Running Baseline Evaluation ---")
    for test in test_cases:
        print(f"Testing ID: {test['id']}...")
        test['baseline_answer'] = run_inference(model, tokenizer, test['question'])

    # 4. Load FINE-TUNED Adapter
    print("\n📥 Loading Fine-Tuned Adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Run Suite on Fine-Tuned
    print("--- Running Fine-Tuned Evaluation ---")
    for test in test_cases:
        print(f"Testing ID: {test['id']}...")
        test['ft_answer'] = run_inference(model, tokenizer, test['question'])

    # 5. Analysis and Logging
    print("\n" + "="*90)
    print(f"{'TEST ID':<15} | {'CONCEPTS':<10} | {'BASE SIM':<10} | {'FT SIM':<10} | {'GAIN'}")
    print("="*90)
    
    final_results = []

    for test in test_cases:
        # Keyword Alignment Score
        concept_score = sum(1 for c in test['concepts'] if c.lower() in test['ft_answer'].lower())
        
        # Semantic Similarity (Three-Vector Logic)
        v_base = sim_model.encode(test['baseline_answer'], convert_to_tensor=True)
        v_ft = sim_model.encode(test['ft_answer'], convert_to_tensor=True)
        v_ref = sim_model.encode(test['reference'], convert_to_tensor=True)

        sim_base = util.pytorch_cos_sim(v_base, v_ref).item()
        sim_ft = util.pytorch_cos_sim(v_ft, v_ref).item()
        gain = sim_ft - sim_base

        print(f"{test['id']:<15} | {concept_score}/{len(test['concepts'])}        | {sim_base:.3f}      | {sim_ft:.3f}      | {gain:+.3f}")
        
        # Prepare for JSON logging
        final_results.append(test)

    # 6. Save Versioned Log
    log_path = f"logs/eval_{model_version}_vs_{test_version}.json"
    with open(log_path, 'w') as f:
        json.dump({
            "model_version": model_version,
            "test_suite_version": version_tag,
            "timestamp": time.ctime(),
            "results": final_results
        }, f, indent=4)
    
    print(f"\n✅ Evaluation complete. Detailed logs saved to: {log_path}")

if __name__ == "__main__":
    import sys
    # Pass the model version (e.g., v1_2) and the test suite version (e.g., v1_1)
    m_ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    t_ver = sys.argv[2] if len(sys.argv) > 2 else "v1_1"
    evaluate_build(model_version=m_ver, test_version=t_ver)