import torch
import yaml
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def load_test_suite(test_version="v3"):
    file_path = f"tests/versions/{test_version}.yaml"
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    return data['benchmarks'], test_version

def run_inference(model, tokenizer, paper_name, user_input):
    instruction = (
        "Extract and format the following research summary into the exact following schema:\n"
        "Paper: [title, authors, year]\n"
        "Problem: [gap addressed]\n"
        "Approach: [technical idea]\n"
        "Result: [what was demonstrated]\n"
        "Implication: [meaning for the field]\n"
        "Open questions: [what this doesn't resolve]\n"
    )
    
    # Format the prompt using Qwen's native Chat Template
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": f"Paper Name: {paper_name}\n\n{user_input}"}
    ]
    
    # This automatically adds the <|im_start|> tags the model needs
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=400, 
            temperature=0.1, 
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode only the newly generated tokens
    input_length = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    return response.strip()


def generate_webui_prompts(model_version="v3", test_version="v3"):
    base_model_id = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = f"models/{model_version}_weights"
    test_cases, _ = load_test_suite(test_version)

    print(f"🚀 Generating A/B WebUI Prompts for: {model_version}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # --- PHASE 1: BASELINE INFERENCE ---
    print("📥 Loading Base Model (Strict GPU allocation)...")
    torch.cuda.empty_cache() 
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print("🏃 Running Baseline Inference...")
    baseline_answers = {}
    for test in test_cases:
        print(f"  -> Base inferencing: {test['id']}")
        baseline_answers[test['id']] = run_inference(model, tokenizer, test['paper_name'], test['user_input'])

    # --- PHASE 2: FINE-TUNED INFERENCE ---
    print(f"\n📥 Loading FT Adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    output_filename = f"logs/{model_version}_ab_webui_prompts.txt"
    print(f"✍️ Running FT inference and writing to {output_filename}...")

    with open(output_filename, 'w') as f:
        for test in test_cases:
            print(f"  -> FT inferencing: {test['id']}")
            ft_answer = run_inference(model, tokenizer, test['paper_name'], test['user_input'])
            base_answer = baseline_answers[test['id']]
            
            # --- WEBUI PROMPT GENERATION ---
            webui_block = f"""
=============================================================================
TEST ID: {test['id']}
=============================================================================
Copy the text below this line into your LLM WebUI:

You are an expert, impartial evaluator grading an A/B test for a Quantum Machine Learning (QML) information extraction task. 

I am testing a Baseline Model against a Fine-Tuned Model. Compare both predictions against the Ground Truth and score them.

### Data
User Input:
{test['user_input'].strip()}

Ground Truth:
{test['reference'].strip()}

[Prediction A: Baseline Model]
{base_answer}

[Prediction B: Fine-Tuned Model]
{ft_answer}

### Output Format
Provide your evaluation in strict JSON format:
{{
  "baseline_eval": {{
    "format_pass": [Boolean. True ONLY if exact 6 headers are present],
    "accuracy_score": [Integer 1 to 5]
  }},
  "finetuned_eval": {{
    "format_pass": [Boolean. True ONLY if exact 6 headers are present],
    "accuracy_score": [Integer 1 to 5]
  }},
  "conclusion": "[1-2 sentences explaining if the fine-tuning improved the model's performance on format adherence and accuracy]"
}}

"""
            f.write(webui_block + "\n")

    print(f"\n✅ Done! Open '{output_filename}' and start copy-pasting.")

if __name__ == "__main__":
    generate_webui_prompts("v3", "v3")