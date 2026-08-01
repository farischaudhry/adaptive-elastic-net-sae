import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import json
from huggingface_hub import hf_hub_download

# Setup Pathing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.saes.polyhedral import AdaptiveElasticNetSAE
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer

def load_user_topk(k: int, device: str) -> TopKSAE:
    sweep_map = {32: "000", 64: "001", 128: "002"}
    filename = f"checkpoints/llama8b/topk/seed0/k{k}_llm-topk_baseline_sweep{sweep_map[k]}-seed0.pt"
    print(f"Downloading Vanilla Top-K (k={k}) from HF...")
    file_path = hf_hub_download(repo_id="farischaudhry/adaptive-elastic-net-sae", filename=filename)
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = TopKSAE(n_dim=4096, d_dict=131072, k=k, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def load_user_aen(k: int, device: str) -> AdaptiveElasticNetSAE:
    lambda_map = {32: "0p002", 64: "0p001", 128: "0p00075"}
    filename = f"checkpoints/llama8b/regularization/seed0/adaptive_elastic_net/lambda1_{lambda_map[k]}_lambda2_0p0001_gamma_0p5.pt"
    print(f"Downloading AEN-SAE (k={k}) from HF...")
    file_path = hf_hub_download(repo_id="farischaudhry/adaptive-elastic-net-sae", filename=filename)
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = AdaptiveElasticNetSAE(n_dim=4096, d_dict=131072, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

@torch.no_grad()
def get_dla_tokens(sae, fid, W_U, tokenizer, k=10):
    """Computes Direct Logit Attribution tokens for a feature."""
    # Decoder weights: [n_dim, d_dict] -> col fid
    direction = sae.decoder.weight[:, fid].to(torch.float32)
    logits = direction @ W_U.to(torch.float32)
    vals, idxs = torch.topk(logits, k=k)
    return [tokenizer.decode([i.item()]) for i in idxs]

def collect_audit_data():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_needed = 8 # Number of contexts per feature
    threshold = 1.0
    
    # 1. Load Resources
    print("Loading Models...")
    aen_model = load_user_aen(k=32, device=device).eval()
    topk_model = load_user_topk(k=32, device=device).eval()
    
    from transformer_lens import HookedTransformer
    llm = HookedTransformer.from_pretrained("meta-llama/Llama-3.1-8B", device=device, dtype=torch.bfloat16)
    W_U = llm.W_U # Unembedding matrix [4096, 128256]

    # Targets (20 each)
    aen_fids = [578, 81, 972, 663, 131, 1081, 543, 945, 587, 899, 1251, 232, 1606, 925, 2087, 2310, 1069, 172, 126, 477]
    topk_fids = [247, 147, 30, 11, 0, 129, 196, 25, 89, 208, 150, 3367, 751, 3848, 3884, 4089, 2147, 3444, 692, 724]
    all_targets = [("AEN", fid) for fid in aen_fids] + [("TopK", fid) for fid in topk_fids]

    stream_cfg = LLMStreamConfig(tl_model_name="meta-llama/Llama-3.1-8B", hook_layer=16, device=device, take_docs=20000)
    streamer = PythiaActivationStreamer(cfg=stream_cfg)

    # Containers
    audit_results = []
    found_contexts = { (m, f): [] for m, f in all_targets }
    counts = { (m, f): 0 for m, f in all_targets }

    # 2. Dataset Scan (Heavy Part)
    print(f"Scanning for contexts for {len(all_targets)} features...")
    pbar = tqdm(total=len(all_targets) * n_needed)
    
    batch_idx = 0
    while any(v < n_needed for v in counts.values()):
        try:
            tokens = streamer.next_token_batch()
        except StopIteration: break
        
        captured = []
        llm.run_with_hooks(tokens, fwd_hooks=[(streamer.hook_name, lambda a, hook: captured.append(a))])
        x = captured[0]
        norm = x.norm(p=2, dim=-1, keepdim=True) / (x.shape[-1]**0.5)
        x_sae = (x / norm).to(torch.bfloat16)

        # Check AEN targets
        _, h_aen = aen_model(x_sae)
        # Check TopK targets
        _, h_topk = topk_model(x_sae)

        for m_type, fid in all_targets:
            if counts[(m_type, fid)] >= n_needed: continue
            
            h = h_aen if m_type == "AEN" else h_topk
            f_acts = h[:, :, fid]
            
            if f_acts.max() >= threshold:
                b, s = torch.where(f_acts == f_acts.max())
                b, s = b[0].item(), s[0].item()
                
                snippet = streamer.tokenizer.decode(tokens[b, max(0, s-12):s]) + \
                          f" >>>{streamer.tokenizer.decode([tokens[b, s].item()])}<<< " + \
                          streamer.tokenizer.decode(tokens[b, s+1:min(128, s+12)])
                
                found_contexts[(m_type, fid)].append(snippet.replace("\n", " "))
                counts[(m_type, fid)] += 1
                pbar.update(1)
        
        batch_idx += 1
        if batch_idx % 100 == 0:
            print(f"  [Status] Batch {batch_idx} | Progress: {sum(counts.values())}/{len(all_targets)*n_needed}")

    # 3. Compute DLA and Package Data
    print("\nComputing DLA tokens and packaging...")
    for m_type, fid in all_targets:
        model = aen_model if m_type == "AEN" else topk_model
        dla = get_dla_tokens(model, fid, W_U, llm.tokenizer)
        
        audit_results.append({
            "model_type": m_type,
            "fid": fid,
            "dla_tokens": dla,
            "contexts": found_contexts[(m_type, fid)]
        })

    with open("audit_data_to_score.json", "w") as f:
        json.dump(audit_results, f, indent=2)

    print(f"Data Collection Complete. Saved {len(audit_results)} features to 'audit_data_to_score.json'")

if __name__ == "__main__":
    collect_audit_data()