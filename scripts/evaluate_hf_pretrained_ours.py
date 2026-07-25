import sys
import random
from pathlib import Path
import shutil
import numpy as np
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.saes.polyhedral import AdaptiveElasticNetSAE
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer
from adaptive_elastic_sae.training.llm_batch_provider import LLMActivationBatchProvider
from adaptive_elastic_sae.training.metrics import (
    dead_neurons_pct,
    explained_variance,
    dictionary_coherence_summary,
)


def load_user_topk(k: int, device: str) -> TopKSAE:
    sweep_map = {32: "000", 64: "001", 128: "002"}
    filename = f"checkpoints/llama8b/topk/seed0/k{k}_llm-topk_baseline_sweep{sweep_map[k]}-seed0.pt"
    
    print(f"Downloading Vanilla Top-K (k={k}) from Hugging Face.")
    file_path = hf_hub_download(
        repo_id="farischaudhry/adaptive-elastic-net-sae",
        filename=filename
    )
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = TopKSAE(n_dim=4096, d_dict=131072, k=k, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_user_aen(k: int, device: str) -> AdaptiveElasticNetSAE:
    lambda_map = {32: "0p002", 64: "0p001", 128: "0p00075"}
    filename = f"checkpoints/llama8b/regularization/seed0/adaptive_elastic_net/lambda1_{lambda_map[k]}_lambda2_0p0001_gamma_0p5.pt"
    
    print(f"Downloading AEN-SAE (k={k}, lambda1={lambda_map[k]}) from Hugging Face.")
    file_path = hf_hub_download(
        repo_id="farischaudhry/adaptive-elastic-net-sae",
        filename=filename
    )
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = AdaptiveElasticNetSAE(
        n_dim=4096, 
        d_dict=131072, 
        device=device, 
        dtype=torch.bfloat16
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def evaluate_model(sae_model, val_provider, device):
    sae_model.eval()
    mses, l0s, exp_vars = [], [], []
    max_activations = torch.zeros(sae_model.d_dict, device=device, dtype=torch.bfloat16)
    window_dead_pcts = []

    batch_idx = 0
    while True:
        try:
            batch = val_provider.next_batch(batch_size=512, device=device, dtype=torch.bfloat16)
        except StopIteration:
            break
        
        x = batch["x"]
        with torch.no_grad():
            x_hat, h = sae_model(x)

            mse = ((x - x_hat) ** 2).mean().item()
            active_l0 = (h.abs() > 1e-12).sum(dim=1).float().mean().item()
            ev = explained_variance(x, x_hat)

            mses.append(mse)
            l0s.append(active_l0)
            exp_vars.append(ev)
            
            # Accumulate max activations over 100-batch windows matching training
            max_activations = torch.maximum(max_activations, h.abs().amax(dim=0))
            
        batch_idx += 1
        
        if batch_idx % 100 == 0:
            window_dead = dead_neurons_pct(max_activations, eps=1e-12)
            window_dead_pcts.append(window_dead)
            max_activations.zero_()

    if not window_dead_pcts:
        window_dead_pcts.append(dead_neurons_pct(max_activations, eps=1e-12))

    # Extract decoder weights safely for coherence metrics
    decoder_weights = getattr(sae_model, "W_dec", None)
    if decoder_weights is None:
        if hasattr(sae_model, "decoder") and hasattr(sae_model.decoder, "weight"):
            decoder_weights = sae_model.decoder.weight.data

    coh_stats = dictionary_coherence_summary(decoder_weights)

    return {
        "Achieved L0": sum(l0s) / len(l0s) if l0s else 0.0,
        "Dead Features %": sum(window_dead_pcts) / len(window_dead_pcts),
        "Explained Variance": sum(exp_vars) / len(exp_vars) if exp_vars else 0.0,
        "MSE": sum(mses) / len(mses) if mses else 0.0,
        "p50 Coherence": coh_stats.get("dictionary_coherence_nn/p50", 0.0),
        "p90 Coherence": coh_stats.get("dictionary_coherence_nn/p90", 0.0),
        "Max Coherence": coh_stats.get("dictionary_coherence_abs_max", 0.0),
    }


def evaluate_user_models(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing Token Streamer.")
    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hf_tokenizer_name="meta-llama/Llama-3.1-8B",
        dataset_name="EleutherAI/the_pile_deduplicated",
        dataset_split="train",             
        hook_layer=15,
        seq_len=128,
        lm_batch_size=32,
        streaming=True,
        skip_docs=0,
        take_docs=2000, 
        loop_dataset=False,
        model_dtype="bfloat16",
        device=device,
        activation_normalization="per_token_l2"
    )
    base_streamer = PythiaActivationStreamer(cfg=stream_cfg)

    k_targets = [32, 64, 128]

    for k in k_targets:
        print("\n" + "="*80)
        print(f"RUNNING EVALUATION FOR TARGET L0 ≈ {k}")
        print("="*80)

        # 1. Evaluate Vanilla Top-K Baseline
        print(f"\n--- Loading Vanilla Top-K (k={k}) ---")
        topk_model = load_user_topk(k, device)
        base_streamer.reset_stream()
        val_provider = LLMActivationBatchProvider(base_streamer)
        topk_metrics = evaluate_model(topk_model, val_provider, device)
        del topk_model

        # 2. Evaluate Adaptive Elastic Net (AEN-SAE)
        print(f"\n--- Loading AEN-SAE (k={k}) ---")
        aen_model = load_user_aen(k, device)
        base_streamer.reset_stream()
        val_provider = LLMActivationBatchProvider(base_streamer)
        aen_metrics = evaluate_model(aen_model, val_provider, device)
        del aen_model

        # Print side-by-side comparison for this target k
        print("\n" + "-"*60)
        print(f"RESULTS FOR TARGET L0 ≈ {k}")
        print("-"*60)
        print(f"{'Metric':25s} | {'Vanilla Top-K':15s} | {'AEN-SAE':15s}")
        print("-"*60)
        for metric_key in topk_metrics.keys():
            val_topk = topk_metrics[metric_key]
            val_aen = aen_metrics[metric_key]
            print(f"{metric_key:25s} | {val_topk:15.3f} | {val_aen:15.3f}")
        print("-"*60)

        # Log to local file
        with open("user_models_metrics_log.txt", "a") as f:
            f.write(f"\n=== Target L0 ≈ {k} ===\n")
            f.write("Vanilla Top-K:\n")
            for mk, mv in topk_metrics.items():
                f.write(f"  {mk}: {mv}\n")
            f.write("AEN-SAE:\n")
            for mk, mv in aen_metrics.items():
                f.write(f"  {mk}: {mv}\n")

        # Clear HF cache to keep disk space free
        cache_dir = Path("/workspace/.cache/huggingface/hub/models--farischaudhry--adaptive-elastic-net-sae")
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    evaluate_user_models()
