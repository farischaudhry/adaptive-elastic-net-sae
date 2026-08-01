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
    file_path = hf_hub_download(repo_id="farischaudhry/adaptive-elastic-net-sae", filename=filename)
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = TopKSAE(n_dim=4096, d_dict=131072, k=k, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_user_aen(k: int, device: str) -> AdaptiveElasticNetSAE:
    lambda_map = {32: "0p002", 64: "0p001", 128: "0p00075"}
    filename = f"checkpoints/llama8b/regularization/seed0/adaptive_elastic_net/lambda1_{lambda_map[k]}_lambda2_0p0001_gamma_0p5.pt"
    print(f"Downloading AEN-SAE (k={k}, lambda1={lambda_map[k]}) from Hugging Face.")
    file_path = hf_hub_download(repo_id="farischaudhry/adaptive-elastic-net-sae", filename=filename)
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = AdaptiveElasticNetSAE(n_dim=4096, d_dict=131072, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


@torch.no_grad()
def get_tokens_and_activations(streamer: PythiaActivationStreamer):
    tokens = streamer.next_token_batch() 
    captured = None
    def _capture_hook(act, hook):
        nonlocal captured
        captured = act
    streamer.model.run_with_hooks(tokens, return_type=None, fwd_hooks=[(streamer.hook_name, _capture_hook)])
    norm_x = normalize_activations(captured, mode=streamer.cfg.activation_normalization, d_model=captured.shape[-1])
    return tokens, norm_x.to(dtype=torch.bfloat16)


def find_bidirectional_feature_contexts(k: int = 32, num_scan_batches: int = 500, num_context_batches: int = 500):
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
        take_docs=10000, 
        loop_dataset=False,
        model_dtype="bfloat16",
        device=device,
        activation_normalization="per_token_l2"
    )
    streamer = PythiaActivationStreamer(cfg=stream_cfg)
    tokenizer = streamer.tokenizer

    topk_model = load_user_topk(k, device).eval()
    aen_model = load_user_aen(k, device).eval()
    d_dict = aen_model.d_dict

    max_act_topk = torch.zeros(d_dict, device=device, dtype=torch.bfloat16)
    max_act_aen = torch.zeros(d_dict, device=device, dtype=torch.bfloat16)

    # --- PASS 1: Identify Features ---
    print(f"\n[Pass 1/2] Scanning {num_scan_batches} batches (~{num_scan_batches*16*128} tokens).")
    for _ in tqdm(range(num_scan_batches), desc="Scanning Activity"):
        try:
            _, x = get_tokens_and_activations(streamer)
        except StopIteration:
            break
        with torch.no_grad():
            _, h_topk = topk_model(x)
            _, h_aen = aen_model(x)
            max_act_topk = torch.maximum(max_act_topk, h_topk.abs().view(-1, d_dict).max(dim=0).values)
            max_act_aen = torch.maximum(max_act_aen, h_aen.abs().view(-1, d_dict).max(dim=0).values)

    # 1. AEN Revived (Dead in TopK, Active in AEN)
    aen_revived_mask = (max_act_topk <= 1e-6) & (max_act_aen >= 0.5)
    aen_fids = torch.where(aen_revived_mask)[0].cpu().tolist()[:15]

    # 2. TopK Only (Active in TopK, Dead in AEN)
    topk_only_mask = (max_act_topk >= 0.5) & (max_act_aen <= 1e-6)
    topk_fids = torch.where(topk_only_mask)[0].cpu().tolist()[:15]

    print(f"\nFound {len(aen_fids)} AEN-Revived features and {len(topk_fids)} TopK-Only features.")
    
    all_target_fids = aen_fids + topk_fids
    top_contexts_aen = {fid: [] for fid in aen_fids}
    top_contexts_topk = {fid: [] for fid in topk_fids}

    # --- PASS 2: Extract Contexts ---
    print(f"\n[Pass 2/2] Resetting stream and extracting contexts.")
    streamer.reset_stream()

    for _ in tqdm(range(num_context_batches), desc="Extracting Contexts"):
        try:
            tokens, x = get_tokens_and_activations(streamer)
        except StopIteration:
            break

        with torch.no_grad():
            _, h_topk = topk_model(x)
            _, h_aen = aen_model(x)

            for fid in all_target_fids:
                is_aen_feat = fid in aen_fids
                f_acts = h_aen[:, :, fid] if is_aen_feat else h_topk[:, :, fid]
                
                max_val, flat_idx = torch.max(f_acts.view(-1), dim=0)
                max_val = max_val.item()

                if max_val > 0.1:
                    seq_len = tokens.shape[1]
                    b_idx = (flat_idx // seq_len).item()
                    s_idx = (flat_idx % seq_len).item()
                    seq_tokens = tokens[b_idx].cpu().numpy()

                    start_i = max(0, s_idx - 8)
                    end_i = min(len(seq_tokens), s_idx + 9)

                    left_str = tokenizer.decode(seq_tokens[start_i:s_idx])
                    peak_str = tokenizer.decode([seq_tokens[s_idx]])
                    right_str = tokenizer.decode(seq_tokens[s_idx+1:end_i])
                    snippet = f"{left_str} >>>[{peak_str}]<<< {right_str}".replace("\n", " ")

                    if is_aen_feat:
                        top_contexts_aen[fid].append((max_val, snippet))
                        top_contexts_aen[fid] = sorted(top_contexts_aen[fid], key=lambda i: i[0], reverse=True)[:7]
                    else:
                        top_contexts_topk[fid].append((max_val, snippet))
                        top_contexts_topk[fid] = sorted(top_contexts_topk[fid], key=lambda i: i[0], reverse=True)[:7]

    # --- PRINT RESULTS ---
    with open("bidirectional_features_contexts.txt", "w") as f:
        f.write("=================================================================\n")
        f.write("SECTION 1: AEN-REVIVED FEATURES (Meaningful sparse concepts?)\n")
        f.write("=================================================================\n")
        for fid in aen_fids:
            f.write(f"\n--- AEN Revived Feature ID: #{fid} (TopK Act: {max_act_topk[fid]:.4f} | AEN Act: {max_act_aen[fid]:.4f}) ---\n")
            for rank, (act_val, snippet) in enumerate(top_contexts_aen[fid], 1):
                f.write(f"  {rank}. [Act: {act_val:6.2f}] .{snippet}.\n")

        f.write("\n\n=================================================================\n")
        f.write("SECTION 2: TOPK-ONLY FEATURES (Polysemantic noise?)\n")
        f.write("=================================================================\n")
        for fid in topk_fids:
            f.write(f"\n--- TopK-Only Feature ID: #{fid} (TopK Act: {max_act_topk[fid]:.4f} | AEN Act: {max_act_aen[fid]:.4f}) ---\n")
            for rank, (act_val, snippet) in enumerate(top_contexts_topk[fid], 1):
                f.write(f"  {rank}. [Act: {act_val:6.2f}] .{snippet}.\n")


if __name__ == "__main__":
    find_bidirectional_feature_contexts(k=32, num_scan_batches=1000, num_context_batches=1000)