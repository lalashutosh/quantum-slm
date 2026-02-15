# Quantum-SLM with RAG: Domain-Specific Adaptation for Quantum Kernel Research

## Background and Motivation
This repository is an ongoing research engineering effort to solve a specific bottleneck in **Quantum Machine Learning (QML)** research: the failure of general-purpose LLMs to reason accurately about SOTA physics.

General models (GPT-4, Claude) suffer from significant "Temporal and Domain Gaps" when dealing with Quantum Kernels. Because this field evolves constantly, base training data is often obsolete, leading to hallucinations. While RAG (Retrieval-Augmented Generation) helps with data retrieval, it often fails at **logical synthesis** of complex hardware constraints, such as noise-stagnation points in NISQ-era devices.

**Quantum-SLM** is my attempt to build a specialized local assistant that doesn't just retrieve text, but has the technical logic of my research "baked" into its weights.

##  SLM advantages
For a researcher, a 1B-3B parameter model running locally on a single workstation (NVIDIA RTX A2000) can be superior to a giant cloud model for three reasons:
1. **Privacy:** SOTA research stays local.
2. **Speed:** Instant inference for iterative brainstorming.
3. **Domain Alignment::** Giant general-purpose models are trained on broad datasets, which can lead to "generalization noise" or outdated reasoning when applied to niche SOTA research. This project explores the hypothesis that a task-specific SLM, fine-tuned on high-fidelity research data, can provide more reliable and reproducible technical insights than a zero-shot giant model, while operating at a fraction of the computational cost.

## Engineering Architecture (CI/CD Inspired)
I designed this project to be a **living pipeline**. Instead of a one-off training script, it is a modular system built for continuous integration of new research findings.

- **Modular Data Engine:** Implements "Expert-in-the-Loop" curation. It processes raw LaTeX, PDFs, or Qiskit code into context-aware instruction pairs.
- **Quantized Training Pipeline:** Optimized for 12GB VRAM constraints using BitsAndBytes and PEFT. 
- **Automated Evaluation Suite:** A verification layer that tests the model against "Gold Standard" research results (e.g., specific coherent noise thresholds) to ensure reasoning is intact after every update.

## Initial Validation (Build v1)  
The goal of the first build was to validate if a 1.5B model could "absorb" the specific results of my Bachelor’s Thesis regarding Quantum Kernel SVMs.


### Build v1.1 Validation Results
| Metric | Baseline | Fine-Tuned | Net Gain |
| :--- | :--- | :--- | :--- |
| **Q1 (Advantage)** | 0.698 | 0.771 | **+7.3%** |
| **Q3 (Threshold)** | 0.642 | 0.647 | **+0.5%** |

**Conclusion:** Build v1.1 successfully validated the pipeline. The model demonstrated a 7.3% semantic shift toward ground-truth research logic, proving the "Small but Mighty" hypothesis for specialized QML domains.

**Benchmark Case:** Coherent noise impact on QSVC accuracy.
- **Baseline:** Hallucinated generic quantum noise theory.
- **Quantum-SLM (Build v1):** Correctly identified that SVM accuracy reaches a minima at **π/6 rotation error** and then stagnates—a specific finding from my empirical noise simulations.

## Repository Structure
- `src/data_engine.py`: Context-aware data pipeline.
- `src/trainer.py`: Core logic for 4-bit LoRA specialization.
- `src/evaluator.py`: Logic for technical reasoning verification.
- `scripts/run_build.sh`: Orchestrator for reproducible builds.

## Current Limitations
- **Dataset Size:** Build v1 is a proof-of-concept trained on a curated subset of research. Future iterations require a broader corpus (e.g., full ArXiv integration) to improve linguistic variety.
- **Benchmarking:** Quantitative comparisons against zero-shot LLMs and RAG baselines are planned for Build v2 to formally validate the precision gain.

## Phase 2: Hybrid RAG-SLM Integration
To transition from a "trained" model to a "grounded" research assistant, I implemented a **Retrieval-Augmented Generation (RAG)** layer using LangChain and ChromaDB.

### Key RAG Features:
- **Recursive Structural Chunking:** Utilizes a hierarchical delimiter set (`["\n\n", "\n", ". ", " "]`) that prioritizes document structure (paragraphs and sentences). This ensures the SLM receives logically coherent blocks of text rather than character-count fragments.
- **LaTeX Normalization Engine:** Rather than treating mathematical notation as opaque noise, the system normalizes Unicode ligatures and maps LaTeX symbols (e.g., bra-kets, kernels, and summations) to natural language equivalents. This significantly improves embedding density, allowing the retriever to match conceptual queries to mathematical symbols that would otherwise have low cosine similarity.
- **MMR Retrieval:** Utilized **Maximal Marginal Relevance (MMR)** to ensure retrieved context represents a diverse set of research findings, preventing redundant context injection.
- **Inference Stability:** Configured the 1.5B parameter model with a 1.1 repetition penalty and Top-P sampling to eliminate generation loops while maintaining technical grounding.

### The "Hybrid" Advantage
During testing, the RAG system allowed the model to cite specific experimental results from external PDFs (like the Havlíček 2019 paper) that were not part of the initial fine-tuning set, providing a robust "Source of Truth" that mitigates hallucination.

## The Path Forward
I am moving away from seeing AI as a "black box." My objective is to continue developing this pipeline to ingest the full **arXiv:quant-ph** dataset. By understanding how different parts of the SLM stack come together—from tokenization to weight adaptation—I can build tools that provide **confidence** in scientific research, not just probability.

---
**Owner:** Ashutosh Lal  
**Affiliation:** Aalto University, Quantum Technology  
**Focus:** Applied AI & Quantum Research Acceleration
