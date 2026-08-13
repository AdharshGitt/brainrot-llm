# Brainrot LLM

A small experimental LLM fine-tuning project that teaches a language model a Gen-Z / meme / brainrot conversational style.

## Project Goal

The goal is to experiment with:

- LLM fine-tuning
- LoRA / PEFT
- Hugging Face Transformers
- TRL SFT training
- CUDA GPU acceleration
- Dataset creation
- Model evaluation

## Base Model

Qwen3-0.6B

The model is fine-tuned using LoRA rather than training all model parameters from scratch.

## Hardware

Tested on:

- NVIDIA RTX 4050 Laptop GPU — 6 GB VRAM
- 24 GB RAM
- Intel Core i5 13th Gen HX
- Windows
- Python 3.11

## Dataset

The dataset contains conversational examples designed to teach:

- Gen-Z slang
- Meme-style responses
- Humor
- Roasting
- Tech/coding humor
- Casual conversation
- Brainrot-style responses

Training and validation data are stored in:

```text
dataset/
├── brainrot.jsonl
└── validation.jsonl
```
