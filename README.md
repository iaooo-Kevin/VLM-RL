# GRPO-VLM Training Pipeline 🧠🖼️

A modular, multi-loss training pipeline for fine-tuning **Vision-Language Models (VLMs)** using cutting-edge Reinforcement Learning algorithms:  
- **GRPO** (Group Relative Policy Optimization)  
- **GSPO** (Group Sequence Policy Optimization)  
- **DAPO** (Dynamic Advantage Policy Optimization)  
- **DrGRPO** (Doubly Robust GRPO)

This pipeline is designed for models like [`Qwen/Qwen2.5-VL`](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), and supports reasoning-aware reward shaping on datasets like `GeoQA-GoldenCoT`.

---

## Key Features

- **Support for Multiple RL Objectives** via CLI:
  - `--use_gspo`
  - `--use_dapo`
  - `--use_drgrpo`
- **Plug-and-play Reward Functions**
  - `think_format_reward`
  - `accuracy_reward`
  - `reasoningReward` (optional CoT-based reward - would require you to provide your own OpenAI API key)
- **Clean Data Preprocessing** with image normalization and prompt formatting

---

## 📦 Installation

Create your environment (e.g., with `conda` or `venv`) and install dependencies:

```bash
pip install -r requirements.txt
```

Or optionally, for developmental purposes
```bash
pip install -e .
```
