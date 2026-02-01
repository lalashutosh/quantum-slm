# Technical Design Choices: Quantum-SLM Architecture

## Overview
This document explains the key engineering decisions behind Quantum-SLM, demonstrating how thoughtful design choices enable efficient domain-specific fine-tuning on constrained hardware.

---

## 1. Modular Pipeline Architecture

### Design Choice
```
quantum-slm/
├── src/
│   ├── data_engine.py   # Data curation and preprocessing
│   ├── trainer.py       # Training orchestration
│   └── evaluator.py     # Model verification
├── scripts/
│   └── run_build.sh     # CI-style build automation
└── data/
    ├── raw/             # Source materials (thesis, papers)
    └── processed/       # Training-ready datasets
```

### Why This Matters
- **Separation of Concerns:** Each module has a single responsibility. If training fails, I debug `trainer.py`. If data quality is poor, I fix `data_engine.py`.
- **Reproducibility:** The `run_build.sh` script ensures anyone can replicate builds v1, v2, v3 identically.
- **Maintainability:** Not a 500-line Jupyter notebook. This is production-ready code structure.

**Vaisala Relevance:** Enterprise AI needs maintainable pipelines, not one-off experiments.

---

## 2. Context-Aware Data Engine

### Design Choice
```python
def process_thesis(input_file, output_file):
    paragraphs = content.split('\n\n')
    
    for i, para in enumerate(paragraphs):
        if is_relevant(para):
            # Capture surrounding context
            context_before = paragraphs[i-1].strip() if i > 0 else ""
            context_after = paragraphs[i+1].strip() if i < len(paragraphs)-1 else ""
            
            full_context = f"{context_before}\n{para}\n{context_after}".strip()
            
            # Format for instruction-following
            dataset.append({
                "instruction": "Using the following context...",
                "input": full_context,
                "output": para
            })
```

### Why This Matters
- **Contextual Learning:** The model doesn't learn isolated facts. It learns how quantum concepts connect (e.g., "noise models affect kernel performance").
- **Human-in-the-Loop:** I manually select relevant paragraphs using keyword filtering. This ensures high data quality over high data quantity.
- **Mimics Research Reading:** Researchers understand papers by seeing ideas in context, not random sentence extraction.

**Vaisala Relevance:** Sensor documentation and technical manuals are contextual. A calibration procedure only makes sense with surrounding setup instructions.

---

## 3. Hardware-Aware Training Configuration

### Design Choice
```python
# Quantization: 4-bit NF4 reduces memory footprint by 75%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# LoRA: Only train 0.1% of parameters
peft_config = LoraConfig(
    r=16,                # Low-rank dimension
    lora_alpha=32,       # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# Gradient Accumulation: Simulate larger batches without OOM
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # Effective batch size = 4
)
```

### Why This Matters
- **12GB VRAM Constraint:** The RTX A2000 can't fit a full-precision 1.5B model. 4-bit quantization makes it possible.
- **Training Speed:** LoRA fine-tuning took 60 seconds. Full fine-tuning would take hours.
- **Enterprise Hardware:** I didn't use a $10,000 A100. This runs on a $600 GPU you can buy at Best Buy.

**Vaisala Relevance:** Production AI must run on available hardware, not idealized cloud infrastructure.

---

## 4. Versioned Build System (CI Philosophy)

### Design Choice
```bash
# run_build.sh
./scripts/run_build.sh v1  # Initial proof-of-concept
./scripts/run_build.sh v2  # Expanded dataset (planned)
./scripts/run_build.sh v3  # Different hyperparameters (planned)
```

Each build produces:
```
models/
├── v1_weights/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── v2_weights/
└── v3_weights/
```

### Why This Matters
- **Experimentation Tracking:** I can compare v1 (12 examples) vs. v2 (100 examples) objectively.
- **Rollback Capability:** If v3 overfits, I revert to v2.
- **Documentation:** Each build has a corresponding config file explaining hyperparameters.

**Vaisala Relevance:** Research projects need version control for models, just like code.

---

## 5. Evaluation-Driven Development

### Design Choice
```python
def verify_build(version="v1"):
    # Load fine-tuned model
    model = load_model(f"models/{version}_weights")
    
    # Test on held-out questions
    prompt = "What happens when coherent noise reaches π/6?"
    response = model.generate(prompt)
    
    print(response)
```

### Current State (v1)
- **Single hardcoded question:** Validates basic knowledge retention.
- **Manual inspection:** I read the output and check correctness.

### Planned Improvements (v2+)
```python
test_suite = [
    {"question": "...", "expected_concepts": ["entanglement", "Hilbert space"]},
    {"question": "...", "expected_concepts": ["NISQ", "noise model"]},
]

def evaluate(model, test_suite):
    scores = []
    for test in test_suite:
        response = model.generate(test["question"])
        score = semantic_similarity(response, test["expected_concepts"])
        scores.append(score)
    return average(scores)
```

### Why This Matters
- **Beyond Vibes:** "The model seems better" is not engineering. Quantitative metrics prove improvement.
- **Failure Analysis:** If the model fails on noise questions but succeeds on kernel questions, I know where to add training data.
- **Continuous Validation:** Each new build is automatically tested against a benchmark.

**Vaisala Relevance:** Production models need automated testing, not manual spot-checks.

---

## 6. Instruction-Following Format

### Design Choice
```python
formatted = f"""### Instruction:
{entry['instruction']}

### Response:
{entry['output']}"""
```

### Why This Matters
- **Standardized Format:** The model learns to expect "Instruction → Response" pairs.
- **Alignment with Base Model:** Qwen 2.5-Instruct was pre-trained on this format. Fine-tuning reinforces it.
- **Controllable Output:** I can steer the model by changing instruction prompts.

**Example:**
- Instruction: "Explain this like I'm a PhD student" → Technical depth
- Instruction: "Explain this like I'm an engineer" → Practical focus

**Vaisala Relevance:** Different stakeholders need different explanations (researchers vs. technicians vs. managers).

---

## Trade-offs and Limitations (v1)

### What I Chose NOT to Do (And Why)

#### 1. Full Fine-Tuning Instead of LoRA
**Why Not:** Would require 10x more VRAM and 100x more time.  
**Trade-off:** LoRA only trains 0.1% of parameters, so the model can't learn entirely new capabilities—only refine existing ones.

#### 2. Larger Models (7B or 13B)
**Why Not:** Wouldn't fit on 12GB VRAM without extreme quantization.  
**Trade-off:** 1.5B models have less "reasoning depth" than 7B models, but for my narrow domain (quantum kernels), this doesn't matter.

#### 3. Automated Data Curation
**Why Not:** Keyword filtering is simple but effective for v1.  
**Trade-off:** I manually reviewed every training example, which doesn't scale to 10,000 examples. v2 will need smarter parsing (equation extraction, code block handling).

#### 4. Comprehensive Benchmarking
**Why Not:** Focused on proof-of-concept first.  
**Trade-off:** I don't have perplexity scores, BLEU metrics, or side-by-side comparisons yet. This is planned for v2.

---

## What I Learned (And What I'd Do Differently)

### 1. Data Quality > Data Quantity
12 high-quality examples beat 1,000 random paragraphs. The model learned my thesis findings because I manually selected the most information-dense paragraphs.

**Next Time:** Build a better data engine that extracts equations, code, and figures—not just text.

### 2. Evaluation Needs to Be Automated
Manually reading model outputs doesn't scale. I need a test suite with expected outputs.

**Next Time:** Create 20-30 test questions with ground-truth answers and compute semantic similarity scores.

### 3. Version Control for Models is Essential
I can't remember what hyperparameters I used for my first training run. Now I save everything.

**Next Time:** Auto-log hyperparameters, training loss, and dataset stats for every build.

---

## Design Philosophy Summary

This project reflects three core principles:

1. **Understand Before Deploying**  
   I didn't blindly copy a tutorial. I chose each design decision deliberately (LoRA for efficiency, 4-bit quantization for memory, context windows for learning quality).

2. **Build for Iteration**  
   v1 is a proof-of-concept. v2 will be better. v3 will be better still. The pipeline is designed for continuous improvement.

3. **Prioritize Practicality**  
   This runs on a single GPU. It trains in 60 seconds. It's reproducible. This is how real-world AI engineering works—not with unlimited compute, but with constraints that force creativity.

---

**For Vaisala Reviewers:**

These design choices aren't just technical decisions. They're a window into how I think about building production AI systems:
- Start simple (v1 proof-of-concept)
- Validate rigorously (evaluation suite)
- Iterate systematically (versioned builds)
- Document thoroughly (this file)

If selected for the "Small but Mighty" internship, I'd apply this same methodology to Vaisala's industrial AI challenges.

---

**Ashutosh Lal**  
Quantum Technology, Aalto University  
ashutosh.lal@aalto.fi