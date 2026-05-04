# Feature Starvation as Geometric Instability in Sparse Autoencoders

This repository contains two experiment tracks:

- Synthetic experiments for controlled mechanistic analysis.
- LLM activation experiments which use a model, tokenizer, and dataset from Hugging Face directly.

## Setting up

```bash
pip install uv
uv sync
uv run wandb login
uv run hf auth login
```

## Cloud GPU setup (RunPod)

Tested on the RunPod image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. 
In this case a seperate venv should be used since the uv configuration is for torch 2.4.

```bash
cd /workspace
export UV_CACHE_DIR="/workspace/.uv_cache"
pip install uv

apt-get update && apt-get install -y tmux

git clone https://github.com/farischaudhry/adaptive-elastic-sae.git
cd adaptive-elastic-sae/

# 1. Create a dedicated cloud env
uv venv --python 3.12 .venv-llama

# 2. Install torch first in that env
uv pip install --python .venv-llama/bin/python --index-strategy unsafe-best-match torch==2.8.0

# 3. Install flash-attn without build isolation
uv pip install --python .venv-llama/bin/python --index-strategy unsafe-best-match --no-build-isolation flash-attn==2.8.3

# 4. Install the rest from requirements
uv pip install --python .venv-llama/bin/python --index-strategy unsafe-best-match -r ./requirements-llama.txt

# 5. Activate and login
source .venv-llama/bin/activate
wandb login
hf auth login

# Example for LLama8B:
tmux new -s llama8b_test
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_llm.py --config configs/llama8b/llama8b_test.yaml --use-wandb
```

## Project Structure

```
├── configs/
│   ├── spiked/
│   │   └── spiked_model_sweep.yaml
│   ├── pythia70m/
│   │   └── pythia70m_test.yaml
│   └── llama8b/
│       └── llama8b_test.yaml
├── scripts/
│   ├── run_spiked.py
│   └── run_llm.py
├── adaptive_elastic_sae/
│   ├── data/                # synthetic and LLM stream data loaders
│   │   ├── synthetic.py     # spiked-data generation and sampling utilities
│   │   └── llm_streamer.py  # Hugging Face / TransformerLens activation streaming
│   ├── saes/                # SAE model implementations and shared base classes
│   │   ├── base.py          # common SAE interface and utilities
│   │   ├── vanilla.py       # L1 / ghost-variant sparse autoencoders
│   │   ├── polyhedral.py    # adaptive lasso and adaptive elastic net variants
│   │   └── top_k.py         # top-k SAE baseline
│   └── training/            # trainers, metrics, batching, and validation logic
│       ├── trainer.py       # synthetic SAE training loop
│       ├── llm_trainer.py   # LLM streaming training loop
│       ├── metrics.py       # synthetic and shared diagnostic metrics
│       ├── llm_metrics.py   # downstream patching / CE / KL validation metrics
│       ├── llm_batch_provider.py # batch adapter for streamed LLM activations
│       ├── trainer_utils.py  # trainer configs and batch-provider helpers
│       └── gpu_metrics.py   # FLOPs and throughput instrumentation
├── notebooks/
├── pyproject.toml
└── README.md
```

## Run Synthetic Experiment (on uv)

Sweep experiment:

```
uv run scripts/run_spiked.py --config configs/spiked/spiked_model_sweep.yaml --use-wandb
```

## Run LLM Experiments (on uv)

Pythia-70M test pattern:

```bash
uv run scripts/run_llm.py --config configs/pythia70m/pythia70m_test.yaml --use-wandb
```

Llama 3.1 8B test pattern:

```bash
uv run scripts/run_llm.py --config configs/llama8b/llama8b_test.yaml --use-wandb
```

## Data and checkpoints (HF Hub)

Large artifacts (checkpoints and W&B history CSVs) are stored in the Hugging Face
model repo. Use Git LFS to pull them locally:

```bash
git lfs install
git clone https://huggingface.co/farischaudhry/adaptive-elastic-net-sae
cd adaptive-elastic-net-sae
git lfs pull
```

## Compute summary

Note: runs often shared a GPU. When x2 GPUs were used, this typically meant two single-GPU jobs (one per device), not multi-GPU data-parallel training for a single run. Exact computation time, FLOPs, and GPU usage metrics can be seen in the final summaries under `notebooks/`.

- Llama-8B (aen-sae-llm-llama8b): NVIDIA H200 x2 (regularization runs), NVIDIA RTX PRO 6000 Blackwell Server Edition x1 (TopK runs). Total compute: 18 days.
- Pythia-70M (aen-sae-llm-pythia70m): NVIDIA L40S x1. Total compute: 14 days.
- Spiked model (aen-sae-mechanistic): NVIDIA RTX PRO 6000 Blackwell Server Edition x1. Total compute: 3 days.
