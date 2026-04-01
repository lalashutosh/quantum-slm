import json
import re
import os

def parse_excerpts_to_jsonl(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Split by "Paper:" to separate each entry
    # [1:] ignores the empty string before the first 'Paper:'
    raw_entries = re.split(r'\n(?=Paper:)', content)
    if len(raw_entries) <= 1 and "Paper:" in content:
        raw_entries = [content]

    dataset = []

    for entry in raw_entries:
        if not entry.strip():
            continue
            
        # 2. Extract fields using flexible regex that ignores optional leading dots/spaces
        def extract(label):
            # Look for Label:, ignore optional leading dot/space, capture until next label or end
            pattern = rf"{label}:\s*(.*?)(?=\n\.?\s*\w+:|$)"
            match = re.search(pattern, entry, re.DOTALL)
            return match.group(1).strip() if match else None

        title_info = extract("Paper")
        problem = extract("Problem")
        approach = extract("Approach")
        result = extract("Result")
        implication = extract("Implication")
        open_q = extract("Open questions")

        if title_info and approach:
            # STRUCTURED LOGIC PROMPT for Qwen 7B
            instruction = f"Analyze the following research summary and synthesize the technical strategy and its implications: {title_info}"
            
            input_context = f"Scientific Problem: {problem}"
            
            # Combine the findings into a professional technical response
            response = (f"Approach: {approach}\n\n"
                        f"Demonstrated Result: {result}\n\n"
                        f"Field Implication: {implication}\n\n"
                        f"Unresolved Challenges: {open_q}")

            formatted = {
                "text": f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n{response}"
            }
            dataset.append(formatted)

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Successfully parsed {len(dataset)} papers into {output_file}")

if __name__ == "__main__":
    # Ensure raw directory exists
    os.makedirs("data/processed", exist_ok=True)
    parse_excerpts_to_jsonl("data/raw/excerpt.txt", "data/processed/train_v2_excerpts.jsonl")