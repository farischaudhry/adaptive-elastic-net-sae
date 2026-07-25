import sys 
from pathlib import Path
import torch
from sae_lens import SAE
import random
import numpy as np

# Add repository root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_elastic_sae.training.llm_trainer import LLMSAETrainer, LLMTrainerConfig
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer
from adaptive_elastic_sae.training.llm_batch_provider import LLMActivationBatchProvider
from adaptive_elastic_sae.training.metrics import (
    dead_neurons_pct,
    feature_utilization_summary
)


class HFLensSAEAdapter(torch.nn.Module):
    """Adapter that makes sae_lens look like BaseSAE so LLMSAETrainer accepts it."""
    def __init__(self, sae_model):
        super().__init__()
        self.sae = sae_model
        self.d_dict = sae_model.cfg.d_sae
        self.n_dim = sae_model.cfg.d_in

    def forward(self, x: torch.Tensor):
        h = self.sae.encode(x)
        x_hat = self.sae.decode(h)
        return x_hat, h

    @property
    def W_dec(self):
        """
        Expose decoder weight tensor for `_evaluate_geometry`.
        sae_lens stores this as (d_sae, d_in). 
        Your `_evaluate_geometry` handles the transposition automatically.
        """
        return self.sae.W_dec


def evaluate_pretrained_baseline_full(sae_ids: list[str], seed: int = 0) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize LLM Streamer ONCE outside the loop to prevent OOM
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

    for sae_id in sae_ids:
        print("\n" + "="*80)
        print(f"EVALUATING BATCHTOPK SAE: {sae_id}")
        print("="*80)

        # 2. Load SAE
        sae_model, _, _ = SAE.from_pretrained(
            release="andyrdt/saes-llama-3.1-8b-instruct", 
            sae_id=sae_id, 
            device=device
        )
        sae_adapter = HFLensSAEAdapter(sae_model).to(device)

        # 3. Instantiate Trainer
        base_streamer.reset_stream()
        val_provider = LLMActivationBatchProvider(base_streamer)
        
        trainer = LLMSAETrainer(
            model=sae_adapter,
            config=trainer_cfg,
            batch_provider=val_provider,
            llm=base_streamer.model,
            hook_name=base_streamer.hook_name,
            validation_token_streamer=base_streamer
        )

        # 4. Phase 1: Downstream LLM Patching Validation (CE / KL)
        print("\n[1/2] Running Downstream LLM Patching Validation...")
        final_metrics = trainer._evaluate_on_validation(
            provider=val_provider, 
            n_batches=None, # Exhaust the 20k document stream
            label="val_final"
        )

        # 5. Phase 2: Structural, Reconstruction, and Geometric Metrics
        print("\n[2/2] Computing Structural, Utilization, and Geometric Metrics...")
        base_streamer.reset_stream()
        val_provider = LLMActivationBatchProvider(base_streamer)

        aggregated_lightweight_metrics = {}
        batch_count = 0
        
        # Track global activity for accurate dead feature & geometry stats
        max_activations = torch.zeros(sae_adapter.d_dict, device=device)
        sum_activations_gt_eps = torch.zeros(sae_adapter.d_dict, device=device)
        total_samples = 0

        while True:
            try:
                batch = val_provider.next_batch(batch_size=256, device=device, dtype=torch.bfloat16)
            except StopIteration:
                break
            
            x = batch["x"]
            with torch.no_grad():
                x_hat, h = sae_adapter(x)

                # Get batch-level metrics using your native function
                batch_metrics = trainer._compute_lightweight_metrics(x, x_hat, h)
                
                # Accumulate for averaging
                for k, v in batch_metrics.items():
                    aggregated_lightweight_metrics[k] = aggregated_lightweight_metrics.get(k, 0.0) + v
                batch_count += 1
                
                # Track global stats
                max_activations = torch.maximum(max_activations, h.abs().amax(dim=0))
                sum_activations_gt_eps += (h.abs() > 1e-12).float().sum(dim=0)
                total_samples += x.shape[0]

        # Average the batch-level lightweight metrics
        if batch_count > 0:
            for k in aggregated_lightweight_metrics:
                final_metrics[k] = aggregated_lightweight_metrics[k] / batch_count

        # Compute accurate GLOBAL utilization (replaces batch-level approximations)
        global_firing_rates = sum_activations_gt_eps / max(total_samples, 1)
        final_metrics.update(feature_utilization_summary(global_firing_rates, prefix="feature_utilization_global"))
        
        # Compute exact Dead Neuron % over the entire dataset window
        final_metrics["dead_neurons_pct_global"] = dead_neurons_pct(max_activations, eps=1e-12)

        # Compute Heavy Geometric Metrics (Gram spectrum, coherence, leakage, leverage)
        # Passing max_activations.unsqueeze(0) forces _evaluate_geometry to use the true global active mask
        geom_metrics = trainer._evaluate_geometry(max_activations.unsqueeze(0), eps=1e-12)
        final_metrics.update(geom_metrics)

        # 6. Print all metrics matching your W&B payload
        print("\n" + "="*80)
        print(f"COMPREHENSIVE METRICS FOR: {sae_id}")
        print("="*80)
        
        # Sort keys alphabetically so it's easy to read
        for k in sorted(final_metrics.keys()):
            v = final_metrics[k]
            if isinstance(v, float):
                print(f"{k:50s}: {v:.6f}")
            else:
                print(f"{k:50s}: {v}")
        print("="*80)

        # Append to log file
        with open("validation_log.txt", "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"COMPREHENSIVE METRICS FOR: {sae_id}\n")
            f.write(f"{'='*80}\n")
            for k in sorted(final_metrics.keys()):
                v = final_metrics[k]
                if isinstance(v, float):
                    f.write(f"{k:50s}: {v:.6f}\n")
                else:
                    f.write(f"{k:50s}: {v}\n")


if __name__ == "__main__":
    evaluate_pretrained_baseline_full([
        "resid_post_layer_15/trainer_0",  # Targeted k ~ 32
        "resid_post_layer_15/trainer_1",  # Targeted k ~ 64
        "resid_post_layer_15/trainer_2",  # Targeted k ~ 128
    ])