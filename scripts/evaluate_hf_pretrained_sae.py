import sys
import random
from pathlib import Path
import numpy as np

# Ensure local repository imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from huggingface_hub import hf_hub_download

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.training.llm_trainer import LLMSAETrainer, LLMTrainerConfig
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer
from adaptive_elastic_sae.training.llm_batch_provider import LLMActivationBatchProvider
from adaptive_elastic_sae.training.metrics import (
    dead_neurons_pct,
    feature_utilization_summary,
    l0_active_features,
    explained_variance,
    dictionary_coherence_summary
)


def load_andyrdt_topk_directly(trainer_id: str, k: int, device: str) -> TopKSAE:
    """Bypasses SAELens entirely to load unregistered HF checkpoints into your native TopKSAE."""
    print(f"Downloading weights for {trainer_id} directly from Hugging Face...")
    
    # Download the raw weights file
    file_path = hf_hub_download(
        repo_id="andyrdt/saes-llama-3.1-8b-instruct",
        filename=f"resid_post_layer_15/{trainer_id}/ae.pt"
    )
    
    sd = torch.load(file_path, map_location=device)
    
    # Instantiate YOUR native TopKSAE
    model = TopKSAE(n_dim=4096, d_dict=131072, k=k, device=device, dtype=torch.bfloat16)
    
    # Map the tensors strictly by shape. 
    # This perfectly handles different naming conventions (e.g. 'W_enc' vs 'encoder.weight')
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


def evaluate_pretrained_baseline_full(seed: int = 0):
    # --- LOCK SEEDS FOR EXACT REPRODUCIBILITY ---
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    # --------------------------------------------

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing LLM Token Streamer & Model (loads once)...")
    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hf_tokenizer_name="meta-llama/Llama-3.1-8B",
        dataset_name="EleutherAI/the_pile_deduplicated",
        dataset_split="train",             
        hook_layer=15,
        seq_len=128,
        lm_batch_size=16, 
        streaming=True,
        skip_docs=0,
        take_docs=20000,  
        loop_dataset=False,
        model_dtype="bfloat16",
        device=device,
        activation_normalization="per_token_l2"
    )
    base_streamer = PythiaActivationStreamer(cfg=stream_cfg)

    trainer_cfg = LLMTrainerConfig(
        device=device,
        dtype=torch.bfloat16,
        validation_ablation_mode="batch_mean",
    )

    # We map the trainer folders to the K value they were targeted for
    variants = [
        ("trainer_0", 32),
        ("trainer_1", 64),
        ("trainer_2", 128)
    ]

    for trainer_id, target_k in variants:
        print("\n" + "="*80)
        print(f"EVALUATING BATCHTOPK SAE: {trainer_id} (Target L0 ≈ {target_k})")
        print("="*80)

        # Load SAE directly into your native architecture
        sae_model = load_andyrdt_topk_directly(trainer_id, k=target_k, device=device)

        # Reset stream state
        base_streamer.reset_stream()
        val_provider = LLMActivationBatchProvider(base_streamer)
        
        trainer = LLMSAETrainer(
            model=sae_model, 
            config=trainer_cfg,
            batch_provider=val_provider,
            llm=base_streamer.model,
            hook_name=base_streamer.hook_name,
            validation_token_streamer=base_streamer
        )

        print("\n[1/2] Running Downstream LLM Patching Validation (CE / KL)...")
        final_metrics = trainer._evaluate_on_validation(
            provider=val_provider, 
            n_batches=None, 
            label="val_final"
        )

        print("\n[2/2] Computing Structural, Utilization, and Geometric Metrics...")
        base_streamer.reset_stream()
        val_provider = LLMActivationBatchProvider(base_streamer)

        aggregated_lightweight_metrics = {}
        batch_count = 0
        
        max_activations = torch.zeros(sae_model.d_dict, device=device)
        sum_activations_gt_eps = torch.zeros(sae_model.d_dict, device=device)
        total_samples = 0

        while True:
            try:
                batch = val_provider.next_batch(batch_size=256, device=device, dtype=torch.bfloat16)
            except StopIteration:
                break
            
            x = batch["x"]
            with torch.no_grad():
                x_hat, h = sae_model(x)

                batch_metrics = trainer._compute_lightweight_metrics(x, x_hat, h)
                
                for k, v in batch_metrics.items():
                    aggregated_lightweight_metrics[k] = aggregated_lightweight_metrics.get(k, 0.0) + v
                batch_count += 1
                
                max_activations = torch.maximum(max_activations, h.abs().amax(dim=0))
                sum_activations_gt_eps += (h.abs() > 1e-12).float().sum(dim=0)
                total_samples += x.shape[0]

        if batch_count > 0:
            for k in aggregated_lightweight_metrics:
                final_metrics[k] = aggregated_lightweight_metrics[k] / batch_count

        global_firing_rates = sum_activations_gt_eps / max(total_samples, 1)
        final_metrics.update(feature_utilization_summary(global_firing_rates, prefix="feature_utilization_global"))
        final_metrics["dead_neurons_pct_global"] = dead_neurons_pct(max_activations, eps=1e-12)

        # Because it's a native BaseSAE, _evaluate_geometry works perfectly!
        geom_metrics = trainer._evaluate_geometry(max_activations.unsqueeze(0), eps=1e-12)
        final_metrics.update(geom_metrics)

        print("\n" + "="*80)
        print(f"COMPREHENSIVE METRICS FOR: {trainer_id}")
        print("="*80)
        
        for k in sorted(final_metrics.keys()):
            v = final_metrics[k]
            if isinstance(v, float):
                print(f"{k:50s}: {v:.6f}")
            else:
                print(f"{k:50s}: {v}")
        print("="*80)

        with open("validation_log.txt", "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"COMPREHENSIVE METRICS FOR: {trainer_id} (Target k={target_k})\n")
            f.write(f"{'='*80}\n")
            for k in sorted(final_metrics.keys()):
                v = final_metrics[k]
                if isinstance(v, float):
                    f.write(f"{k:50s}: {v:.6f}\n")
                else:
                    f.write(f"{k:50s}: {v}\n")


if __name__ == "__main__":
    evaluate_pretrained_baseline_full()
