import sys
import random
from pathlib import Path
import numpy as np

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


def evaluate_pretrained_baseline_full(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing LLM Token Streamer & Model...")
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
        take_docs=200,  # REDUCED TEMPORARILY TO TEST SPEED
        loop_dataset=False,
        model_dtype="bfloat16",
        device=device,
        activation_normalization="per_token_l2"
    )
    base_streamer = PythiaActivationStreamer(cfg=stream_cfg)
    print("Streamer and Model loaded successfully!")

    trainer_cfg = LLMTrainerConfig(
        device=device,
        dtype=torch.bfloat16,
        validation_ablation_mode="batch_mean",
    )

    variants = [
        ("trainer_0", 32),
    ]

    for trainer_id, target_k in variants:
        print(f"\nEVALUATING BATCHTOPK SAE: {trainer_id} (Target L0 ≈ {target_k})")
        sae_model = load_andyrdt_topk_directly(trainer_id, k=target_k, device=device)

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

        print("Running short validation check...")
        final_metrics = trainer._evaluate_on_validation(
            provider=val_provider, 
            n_batches=5, # Limit batches for quick sanity test
            label="val_final"
        )
        print("Success! Metrics obtained:", list(final_metrics.keys()))

if __name__ == "__main__":
    evaluate_pretrained_baseline_full()