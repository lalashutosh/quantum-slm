# src/data_engine.py
import json
import os

def process_thesis(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Please paste your thesis text there.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by paragraphs
    paragraphs = content.split('\n\n')
    
    dataset = []
    for i, para in enumerate(paragraphs):
        text = para.strip()
        if any(word in text.lower() for word in ["quantum", "kernel", "feature", "noise"]):
            # Grab previous and next paragraph for context if they exist
            context_before = paragraphs[i-1].strip() if i > 0 else ""
            context_after = paragraphs[i+1].strip() if i < len(paragraphs)-1 else ""
            
            full_context = f"{context_before}\n{text}\n{context_after}".strip()
            
            entry = {
                "instruction": "Using the following context from quantum research, explain the core technical insight:",
                "input": full_context,
                "output": text # The model learns to extract the 'answer' from the 'context'
            }
            
            # This specific format is what the model expects for "SFT" (Supervised Fine-Tuning)
            formatted = f"### Instruction:\n{entry['instruction']}\n\n### Response:\n{entry['output']}"
            dataset.append({"text": formatted})

    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"✅ Data Engine: Created {len(dataset)} high-quality training samples in {output_file}")

if __name__ == "__main__":
    # Ensure the directory exists
    os.makedirs("data/processed", exist_ok=True)
    process_thesis("data/raw/thesis.txt", "data/processed/train_v1.jsonl")