import sys
import random
from pathlib import Path
import shutil
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from huggingface_hub import hf_hub_download

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer
from adaptive_elastic_sae.training.llm_batch_provider import LLMActivationBatchProvider
from adaptive_elastic_sae.training.metrics import (
    dead_neurons_pct,
    explained_variance,
    dictionary_coherence_summary,
)


def load_andyrdt_topk_directly(trainer_id: str, k: int, device: str) -> TopKSAE:
    print(f"Downloading weights for {trainer_id} directly from Hugging Face...")
    file_path = hf_hub_download(
        repo_id="andyrdt/saes-llama-3.1-8b-instruct",
        filename=f"resid_post_layer_15/{trainer_id}/ae.pt"
    )
    sd = torch.load(file_path, map_location=device)
    model = TopKSAE(n_dim=4096, d_dict=131072, k=k, device=device, dtype=torch.bfloat16)
    
    for key, tensor in sd.items():
        shape = list(tensor.shape)
        if shape == [131072, 4096]:
            if "dict" in key.lower() or "dec" in key.lower():
                model.decoder.weight.data = tensor.T.to(torch.bfloat16)
            else:
                model.encoder.weight.data = tensor.to(torch.bfloat16)
        elif shape == [4096, 131072]:
            model.decoder.weight.data = tensor.to(torch.bfloat16)
        elif shape == [131072]:
            model.encoder.bias.data = tensor.to(torch.bfloat16)
        elif shape == [4096]:
            model.b_dec.data = tensor.to(torch.bfloat16)
    return model


def evaluate_table_metrics_only(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing LLM Token Streamer.")
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

    variants = [
        ("trainer_0", 32),
        ("trainer_1", 64),
        ("trainer_2", 128)
    ]

    for trainer_id, target_k in variants:
        print("\n" + "="*80)
        print(f"COMPUTING TABLE METRICS FOR: {trainer_id} (Target L0 ≈ {target_k})")
        print("="*80)

        sae_model = load_andyrdt_topk_directly(trainer_id, k=target_k, device=device)
        sae_model.eval()

        base_streamer.reset_stream()
        val_provider = LLMActivationBatchProvider(base_streamer)

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
                
                # Accumulate max activations over windows matching training config
                max_activations = torch.maximum(max_activations, h.abs().amax(dim=0))
                
            batch_idx += 1
            
            # Every 100 batches (matching training's window concept), record dead window % and reset
            if batch_idx % 100 == 0:
                window_dead = dead_neurons_pct(max_activations, eps=1e-12)
                window_dead_pcts.append(window_dead)
                max_activations.zero_()
                print(f"Processed {batch_idx * 512} tokens (Window dead %: {window_dead:.2f}%)...")

        # Fallback if total batches < 100
        if not window_dead_pcts:
            window_dead_pcts.append(dead_neurons_pct(max_activations, eps=1e-12))
        final_dead_pct = sum(window_dead_pcts) / len(window_dead_pcts)

        # Coherence metrics from decoder weights
        decoder_weights = sae_model.decoder.weight.data
        coh_stats = dictionary_coherence_summary(decoder_weights)

        table_row = {
            "Target L0": target_k,
            "Achieved L0": sum(l0s) / len(l0s) if l0s else 0.0,
            "Dead Features %": final_dead_pct,
            "Explained Variance": sum(exp_vars) / len(exp_vars) if exp_vars else 0.0,
            "MSE": sum(mses) / len(mses) if mses else 0.0,
            "p50 Coherence": coh_stats.get("dictionary_coherence_nn/p50", 0.0),
            "p90 Coherence": coh_stats.get("dictionary_coherence_nn/p90", 0.0),
            "Max Coherence": coh_stats.get("dictionary_coherence_abs_max", 0.0),
        }

        print("\n" + "-"*60)
        print(f"FINAL VALUES FOR {trainer_id} (Target L0 ≈ {target_k}):")
        print("-"*60)
        for k, v in table_row.items():
            if isinstance(v, float):
                print(f"{k:20s}: {v:.3f}")
            else:
                print(f"{k:20s}: {v}")
        print("-"*60)

        with open("table_metrics_log.txt", "a") as f:
            f.write(f"\nResults for {trainer_id} (Target L0 ≈ {target_k}):\n")
            for k, v in table_row.items():
                f.write(f"  {k}: {v}\n")

        # Cleanup downloaded weights cache for this variant to save disk space
        cache_dir = Path("/workspace/.cache/huggingface/hub/models--andyrdt--saes-llama-3.1-8b-instruct")
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
            print(f"Cleared Hugging Face cache for {trainer_id} to free disk space.")


if __name__ == "__main__":
    evaluate_table_metrics_only()
