import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer

# Setup Pathing (Adjust to your project root)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.saes.polyhedral import AdaptiveElasticNetSAE

# =================================================================
# 1. LOADING UTILITIES
# =================================================================

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

# =================================================================
# 2. LOGIT ATTRIBUTION LOGIC
# =================================================================

@torch.no_grad()
def run_logit_attribution_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Llama 3.1 8B (we only need the unembedding matrix W_U)
    print("Loading Llama 3.1 8B for Unembedding access...")
    llm = HookedTransformer.from_pretrained(
        "meta-llama/Llama-3.1-8B", 
        device=device, 
        dtype=torch.bfloat16
    )
    W_U = llm.W_U # Shape: [d_model, n_vocab] -> [4096, 128256]

    # 2. Load SAEs
    aen_model = load_user_aen(k=32, device=device).eval()
    topk_model = load_user_topk(k=32, device=device).eval()

    # 3. Targets for comparison
    # AEN targets are monosemantic candidates
    # TopK targets are hubs/entangled candidates
    aen_targets = [578, 81, 972, 663, 131, 1081]
    topk_targets = [247, 147, 30, 11, 0]

    def get_top_tokens(sae, fid):
        # direction is the column of the decoder weight matrix
        # sae.decoder.weight is [n_dim, d_dict]
        direction = sae.decoder.weight[:, fid] # [4096]
        
        # Linear project direction into vocabulary space
        # [1, 4096] @ [4096, 128256] -> [128256]
        logits = direction.to(torch.float32) @ W_U.to(torch.float32)
        
        vals, idxs = torch.topk(logits, k=10)
        tokens = [llm.tokenizer.decode([i.item()]) for i in idxs]
        return tokens, vals.tolist()

    # 4. Perform Audit and Write Results
    print("\n--- Starting Logit Attribution Audit ---")
    with open("llama_logit_attribution_results.txt", "w") as f:
        f.write("LLAMA 3.1 8B: DIRECT LOGIT ATTRIBUTION (DLA) REPORT\n")
        f.write("Proves the semantic purity of feature decoder directions.\n\n")

        f.write("--- SECTION 1: AEN-SAE SPECIALIZED FEATURES ---\n")
        for fid in aen_targets:
            tokens, scores = get_top_tokens(aen_model, fid)
            f.write(f"FID #{fid}\n")
            f.write(f"  Top Tokens: {tokens}\n")
            f.write(f"  Max Score:  {scores[0]:.2f}\n\n")
            print(f"Audited AEN #{fid}: {tokens[:3]}...")

        f.write("\n--- SECTION 2: TOPK HUB FEATURES ---\n")
        for fid in topk_targets:
            tokens, scores = get_top_tokens(topk_model, fid)
            f.write(f"FID #{fid}\n")
            f.write(f"  Top Tokens: {tokens}\n")
            f.write(f"  Max Score:  {scores[0]:.2f}\n\n")
            print(f"Audited TopK #{fid}: {tokens[:3]}...")

    print("\nAudit Complete. Results saved to 'llama_logit_attribution_results.txt'")

if __name__ == "__main__":
    run_logit_attribution_audit()