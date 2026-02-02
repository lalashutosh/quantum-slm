# Quantum-SLM: Domain-Specific Adaptation for Quantum Kernel Research

## Background and Motivation
This repository is an ongoing research engineering effort to solve a specific bottleneck in **Quantum Machine Learning (QML)** research: the failure of general-purpose LLMs to reason accurately about SOTA physics.

General models (GPT-4, Claude) suffer from significant "Temporal and Domain Gaps" when dealing with Quantum Kernels. Because this field evolves constantly, base training data is often obsolete, leading to hallucinations. While RAG (Retrieval-Augmented Generation) helps with data retrieval, it often fails at **logical synthesis** of complex hardware constraints, such as noise-stagnation points in NISQ-era devices.

**Quantum-SLM** is my attempt to build a specialized local assistant that doesn't just retrieve text, but has the technical logic of my research "baked" into its weights.

## 🧠 SLM advantages
For a researcher, a 1B-3B parameter model running locally on a single workstation (NVIDIA RTX A2000) is superior to a giant cloud model for three reasons:
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

## The Path Forward
I am moving away from seeing AI as a "black box." My objective is to continue developing this pipeline to ingest the full **arXiv:quant-ph** dataset. By understanding how different parts of the SLM stack come together—from tokenization to weight adaptation—I can build tools that provide **confidence** in scientific research, not just probability.

---
**Owner:** Ashutosh Lal  
**Affiliation:** Aalto University, Quantum Technology  
**Focus:** Applied AI & Quantum Research Acceleration
