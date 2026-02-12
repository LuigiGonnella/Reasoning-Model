# Reasoning Model - Exercises Repository

This repository contains solutions and implementations for the exercises provided by the book **"Build a Reasoning Model (From Scratch)"**.

## Overview

The code and notebooks in this repository explore the fundamentals of building reasoning models from first principles. Through hands-on exercises and practical implementations, you will develop a deep understanding of how reasoning models work and how to build them from scratch.

## Repository Structure

### Chapter Notebooks

- **CH2/**: Loading LLMs and Inference Optimization
  - `1_load_LLM&speedup_inference.ipynb`: Model loading and inference speed optimization
  - `qwen3/`: Qwen3 0.6B base model weights and tokenizer files

- **CH3/**: Mathematical Reasoning and Verification
  - `2_math_verifier.ipynb`: Answer extraction and grading for mathematical problems
  - `math500_test.json`: MATH-500 benchmark dataset
  - `qwen3/`: Model weights and tokenizer

- **CH4/**: Inference-Time Scaling Techniques
  - `3_inference_time_scaling.ipynb`: Chain-of-thought, temperature/top-p sampling, self-consistency voting

### Supporting Materials

- **Appendixes/**: Detailed implementations of model components
  - `GLUFeedForwardNeuralNetwork/`: GLU activation function implementation
  - `GroupedQueryAttention/`: Grouped-query attention mechanism
  - `KVcache/`: Key-value cache optimization techniques
  - `QWEN3Model/`: Complete Qwen3 model architecture
  - `QWEN3Tokenizer/`: Tokenizer implementation
  - `QWEN3TransformerBlock/`: Transformer block components
  - `RMSNorm/`: Root Mean Square Layer Normalization
  - `RoPE/`: Rotary Position Embeddings

- **Exercises/**: Practice implementations
  - `CH2.py`: Chapter 2 coding exercises

- **requirements.txt**: Python dependencies

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. Clone or navigate to this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

Launch Jupyter Notebook and navigate to the desired chapter:

```bash
jupyter notebook
```

Then open notebooks in the CH2/, CH3/, or CH4/ directories.

## Key Concepts Covered

### Chapter 2: Model Loading & Optimization
- Loading pre-trained language models
- Inference speed optimization with KV caching
- Model compilation and performance tuning

### Chapter 3: Mathematical Reasoning
- Chain-of-thought prompting
- Answer extraction from generated text
- Automated grading and evaluation on MATH-500 benchmark

### Chapter 4: Inference-Time Scaling
- Temperature sampling for controlled randomness
- Top-p (nucleus) sampling for quality control
- Self-consistency voting with majority consensus
- Early stopping strategies for efficiency

## Model Components (Appendixes)

The Appendixes folder contains standalone implementations of key architectural components:
- Attention mechanisms (Grouped-Query Attention)
- Normalization layers (RMSNorm)
- Position encodings (RoPE)
- Feed-forward networks (GLU variants)
- Efficient caching strategies (KV cache)

Each component includes a `main.py` demonstrating its usage.

## Notes

- All notebooks include detailed explanations and step-by-step implementations
- Model weights are stored in chapter-specific `qwen3/` directories
- Ensure all dependencies are installed before running notebooks
- Some operations may require significant memory (CPU inference of 0.6B model)

## About the Book

This repository follows the exercises and concepts from **"Build a Reasoning Model (From Scratch)"**, which provides a comprehensive guide to understanding and implementing reasoning models using modern deep learning techniques.

## License

This repository contains educational implementations based on exercises from "Build a Reasoning Model (From Scratch)".

