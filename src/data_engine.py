import json
import os
import glob
import unicodedata
import re

def normalize_text(text):
    # Fixes ligatures like 'fi' and 'fl' from PDF exports
    text = unicodedata.normalize("NFKD", text)
    # Remove weird whitespace/newlines but keep sentence structure
    text = re.sub(r'\s+', ' ', text)
    return text

def process_corpus(input_folder, output_file):
    dataset = []
    keywords = ["quantum", "kernel", "feature", "noise", "svm", "hilbert", "rkhs", "measurement", "ansatz"]
    
    files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    for file_path in files:
        print(f"📖 Processing {os.path.basename(file_path)}...")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_content = f.read()
            
        clean_content = normalize_text(raw_content)
        
        # Split into sentences using a simple regex (works better for research math)
        sentences = re.split(r'(?<=[.!?]) +', clean_content)
        
        # Group sentences into chunks of 4 for better context (Sliding Window)
        for i in range(0, len(sentences) - 4, 2): # Steps of 2 for 50% overlap
            chunk = " ".join(sentences[i:i+4])
            
            # Filter logic
            if len(chunk) > 200 and any(k in chunk.lower() for k in keywords):
                # We use the first sentence as context and the rest as the response
                entry = {
                    "instruction": "Explain the technical logic or research finding in this passage:",
                    "input": sentences[i], 
                    "output": chunk
                }
                
                formatted_text = f"### Instruction:\n{entry['instruction']}\n\n### Input:\n{entry['input']}\n\n### Response:\n{entry['output']}"
                dataset.append({"text": formatted_text})

    with open(output_file, 'w') as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
            
    print(f"✨ Build Complete: {len(dataset)} samples generated from {len(files)} files.")

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    process_corpus("data/raw/", "data/processed/train_v1_2.jsonl")