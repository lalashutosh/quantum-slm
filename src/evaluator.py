import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def verify_build(version="v1"):
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = f"models/{version}_weights"
    
    print(f"Verifying Build {version}...")
    
    # Load Base
    model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    # Load Your "Expertise" (The Adapter)
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Test Question (Something specific to your thesis)
    prompt = "### Instruction:\nWhat happens to Quantum Kernel SVM accuracy when coherent noise reaches pi/6?\n\n### Response:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=250)
    
    print("\n--- MODEL RESPONSE ---")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print("-----------------------\n")

if __name__ == "__main__":
    verify_build("v1")