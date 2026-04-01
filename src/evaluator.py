import torch
import time
import json
import os
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util


def load_test_suite(test_version="v1_2"):
    """Loads the benchmark registry from YAML."""
    file_path = f"tests/registry.yaml"
    if not os.path.exists(file_path):
        # Fallback to the specific versioned file if registry.yaml isn't the main one
        file_path = f"tests/versions/{test_version}.yaml"
        
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['benchmarks'], data['version']

def run_inference(model, tokenizer, question):
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250,
            temperature=0.1, 
            repetition_penalty=1.2, # Added to prevent the 7B model from looping
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_text.split("### Response:\n")[-1].strip()
    return response

def evaluate_build(model_version="v1", test_version="v1_2"):
    # 1. Dynamic Model Selection
    # If version starts with 'v2', we use the 7B base. Otherwise, 1.5B.
    if model_version.startswith("v2"):
        base_model_id = "Qwen/Qwen2.5-7B-Instruct"
        print(f"🚀 Detected 7B Build. Adjusting VRAM strategy...")
    else:
        base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    adapter_path = f"models/{model_version}_weights"
    test_cases, version_tag = load_test_suite(test_version)

    print(f"🧪 Evaluating: Model {model_version} ({base_model_id}) vs Test Suite {version_tag}")
    
    # 2. Quantization Config (CRITICAL for 7B on 12GB GPU)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 3. Setup Tools
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Load BASE Model (Quantized)
    print("📥 Loading Base Model for control benchmark...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("🧪 Running Baseline Evaluation...")
    for test in test_cases:
        print(f"Testing ID: {test['id']}...")
        test['baseline_answer'] = run_inference(model, tokenizer, test['question'])

    # 5. Load FINE-TUNED Adapter
    print(f"\n📥 Injecting {model_version} Adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    print("🧪 Running Fine-Tuned Evaluation...")
    for test in test_cases:
        print(f"Testing ID: {test['id']}...")
        test['ft_answer'] = run_inference(model, tokenizer, test['question'])

    # 6. Metrics & Reporting
    print("\n" + "="*90)
    print(f"{'TEST ID':<20} | {'CONCEPTS':<10} | {'BASE SIM':<10} | {'FT SIM':<10} | {'GAIN'}")
    print("="*90)
    
    for test in test_cases:
        # Keyword Recall
        concept_score = sum(1 for c in test['concepts'] if c.lower() in test['ft_answer'].lower())
        
        # Vector Similarity
        v_base = sim_model.encode(test['baseline_answer'], convert_to_tensor=True)
        v_ft = sim_model.encode(test['ft_answer'], convert_to_tensor=True)
        v_ref = sim_model.encode(test['reference'], convert_to_tensor=True)

        sim_base = util.pytorch_cos_sim(v_base, v_ref).item()
        sim_ft = util.pytorch_cos_sim(v_ft, v_ref).item()
        gain = sim_ft - sim_base

        print(f"{test['id']:<20} | {concept_score}/{len(test['concepts'])}        | {sim_base:.3f}      | {sim_ft:.3f}      | {gain:+.3f}")

    # 7. Save Log
    os.makedirs("logs", exist_ok=True)
    log_path = f"logs/eval_{model_version}_vs_{test_version}.json"
    with open(log_path, 'w') as f:
        json.dump({"model": model_version, "results": test_cases, "time": time.ctime()}, f, indent=4)
    
    print(f"\n✅ Report generated: {log_path}")

if __name__ == "__main__":
    import sys
    m_ver = sys.argv[1] if len(sys.argv) > 1 else "v1"
    t_ver = sys.argv[2] if len(sys.argv) > 2 else "v1_2"
    evaluate_build(m_ver, t_ver)