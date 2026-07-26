import sys
import random
from pathlib import Path
import shutil
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.saes.polyhedral import AdaptiveElasticNetSAE
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer, normalize_activations


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


@torch.no_grad()
def get_tokens_and_activations(streamer: PythiaActivationStreamer):
    """Pulls next token batch and extracts normalized residual activations."""
    tokens = streamer.next_token_batch() # Shape: [lm_batch_size, seq_len]
    captured = None

    def _capture_hook(act, hook):
        nonlocal captured
        captured = act

    streamer.model.run_with_hooks(
        tokens,
        return_type=None,
        fwd_hooks=[(streamer.hook_name, _capture_hook)],
    )

    norm_x = normalize_activations(
        captured,
        mode=streamer.cfg.activation_normalization,
        d_model=captured.shape[-1]
    )
    return tokens, norm_x.to(dtype=torch.bfloat16)


def find_revived_feature_contexts(k: int = 32, num_scan_batches: int = 50, num_context_batches: int = 100):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Initializing Token Streamer.")
    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hf_tokenizer_name="meta-llama/Llama-3.1-8B",
        dataset_name="EleutherAI/the_pile_deduplicated",
        dataset_split="train",             
        hook_layer=16,
        seq_len=128,
        lm_batch_size=16,
        streaming=True,
        skip_docs=0,
        take_docs=5000, 
        loop_dataset=False,
        model_dtype="bfloat16",
        device=device,
        activation_normalization="per_token_l2"
    )
    streamer = PythiaActivationStreamer(cfg=stream_cfg)
    tokenizer = streamer.tokenizer

    # 1. Load models
    topk_model = load_user_topk(k, device)
    aen_model = load_user_aen(k, device)
    
    topk_model.eval()
    aen_model.eval()

    d_dict = aen_model.d_dict
    max_act_topk = torch.zeros(d_dict, device=device, dtype=torch.bfloat16)
    max_act_aen = torch.zeros(d_dict, device=device, dtype=torch.bfloat16)

    # PASS 1: Identify Revived Features (Dead in TopK, Active in AEN)
    print(f"\n[Pass 1/2] Scanning {num_scan_batches} batches to identify revived features.")
    for _ in tqdm(range(num_scan_batches), desc="Scanning Feature Activity"):
        try:
            _, x = get_tokens_and_activations(streamer)
        except StopIteration:
            break

        with torch.no_grad():
            _, h_topk = topk_model(x)
            _, h_aen = aen_model(x)

            # Max activation across batch & sequence length
            max_act_topk = torch.maximum(max_act_topk, h_topk.abs().view(-1, d_dict).max(dim=0).values)
            max_act_aen = torch.maximum(max_act_aen, h_aen.abs().view(-1, d_dict).max(dim=0).values)

    # Filter: Dead in TopK (<= 1e-6) and Active in AEN (>= 0.5)
    dead_in_topk = max_act_topk <= 1e-6
    active_in_aen = max_act_aen >= 0.5
    revived_mask = dead_in_topk & active_in_aen
    revived_fids = torch.where(revived_mask)[0].cpu().tolist()

    print(f"\nResult: Found {len(revived_fids)} revived features (Dead in TopK, Active in AEN).")
    
    if not revived_fids:
        print("No revived features met the threshold. Lowering threshold and selecting top AEN activations.")
        revived_fids = torch.argsort(max_act_aen, descending=True)[:5].cpu().tolist()

    # Pick top 5 representative revived features
    target_fids = revived_fids[:5]
    print(f"Targeting Feature IDs for context extraction: {target_fids}")

    # PASS 2: Extract Dataset Contexts & Tokens
    print(f"\n[Pass 2/2] Resetting stream and scanning {num_context_batches} batches for top contexts.")
    streamer.reset_stream()

    top_contexts = {fid: [] for fid in target_fids}
    context_window = 8

    for _ in tqdm(range(num_context_batches), desc="Extracting Contexts"):
        try:
            tokens, x = get_tokens_and_activations(streamer)
        except StopIteration:
            break

        with torch.no_grad():
            _, h_aen = aen_model(x) # Shape: [batch_size, seq_len, d_dict]

            for fid in target_fids:
                f_acts = h_aen[:, :, fid] # [batch_size, seq_len]
                max_val, flat_idx = torch.max(f_acts.view(-1), dim=0)
                max_val = max_val.item()

                if max_val > 0.1:
                    seq_len = tokens.shape[1]
                    b_idx = (flat_idx // seq_len).item()
                    s_idx = (flat_idx % seq_len).item()

                    seq_tokens = tokens[b_idx].cpu().numpy()

                    start_i = max(0, s_idx - context_window)
                    end_i = min(len(seq_tokens), s_idx + context_window + 1)

                    left_str = tokenizer.decode(seq_tokens[start_i:s_idx])
                    peak_str = tokenizer.decode([seq_tokens[s_idx]])
                    right_str = tokenizer.decode(seq_tokens[s_idx+1:end_i])

                    formatted_context = f"{left_str} >>>[{peak_str}]<<< {right_str}".replace("\n", " ")
                    top_contexts[fid].append((max_val, formatted_context))
                    top_contexts[fid] = sorted(top_contexts[fid], key=lambda item: item[0], reverse=True)[:3]

    print("\n" + "="*80)
    print(f"REVIVED FEATURE CONTEXT ANALYSIS (Target L0 ≈ {k})")
    print("="*80)
    
    log_lines = []
    for fid in target_fids:
        header = f"\n--- Revived Feature ID: #{fid} (TopK Act: {max_act_topk[fid]:.4f} | AEN Act: {max_act_aen[fid]:.4f}) ---"
        print(header)
        log_lines.append(header)

        if not top_contexts[fid]:
            msg = "  (No strong activations found in pass 2)"
            print(msg)
            log_lines.append(msg)
            continue

        for rank, (act_val, snippet) in enumerate(top_contexts[fid], 1):
            line = f"  {rank}. [Act: {act_val:6.2f}] .{snippet}."
            print(line)
            log_lines.append(line)

    # Save output log
    with open("revived_features_contexts.txt", "w") as f:
        f.write("\n".join(log_lines))
    print("\nSaved context results to revived_features_contexts.txt!")

    # Clean up HF cache
    cache_dir = Path("/workspace/.cache/huggingface/hub/models--farischaudhry--adaptive-elastic-net-sae")
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    find_revived_feature_contexts(k=32, num_scan_batches=500, num_context_batches=500)
