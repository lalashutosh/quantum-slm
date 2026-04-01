import json
import re
import os

def parse_excerpts_to_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split by individual papers (assuming double newline between excerpts)
    excerpts = content.split('\n\n')
    dataset = []

    for item in excerpts:
        if "Paper:" in item and "Approach:" in item:
            # Clean up keys for the prompt
            # Use Regex to extract fields
            try:
                title = re.search(r"Paper: \[(.*?)\]", item).group(1)
                problem = re.search(r"Problem: \[(.*?)\]", item).group(1)
                approach = re.search(r"Approach: \[(.*?)\]", item).group(1)
                result = re.search(r"Result: \[(.*?)\]", item).group(1)
                implication = re.search(r"Implication: \[(.*?)\]", item).group(1)
                open_q = re.search(r"Open questions: \[(.*?)\]", item).group(1)

                # STRUCTURED LOGIC PROMPT
                instruction = f"Analyze the following research problem in Quantum Machine Learning and propose a technical approach based on the study: {title}"
                input_context = f"Scientific Problem: {problem}"
                
                # The 'Output' is the synthesis of the rest of the fields
                output = (f"Approach: {approach}\n"
                          f"Demonstrated Result: {result}\n"
                          f"Field Implication: {implication}\n"
                          f"Future Work: {open_q}")

                formatted = f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n{output}"
                dataset.append({"text": formatted})
            except AttributeError:
                continue # Skip if format is slightly off

    with open(output_file, 'w') as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"✅ Created {len(dataset)} structured logic samples in {output_file}")

if __name__ == "__main__":
    # Ensure raw file exists
    parse_excerpts_to_jsonl("data/raw/excerpts.txt", "data/processed/train_v2_excerpts.jsonl")