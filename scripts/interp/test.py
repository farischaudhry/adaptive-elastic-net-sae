import sys
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from huggingface_hub import hf_hub_download

# Setup Pathing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.saes.polyhedral import AdaptiveElasticNetSAE
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer

# =================================================================
# 1. LOADING UTILITIES
# =================================================================

def load_user_topk(k: int, device: str) -> TopKSAE:
    sweep_map = {32: "000", 64: "001", 128: "002"}
    filename = f"checkpoints/llama8b/topk/seed0/k{k}_llm-topk_baseline_sweep{sweep_map[k]}-seed0.pt"
    file_path = hf_hub_download(repo_id="farischaudhry/adaptive-elastic-net-sae", filename=filename)
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = TopKSAE(n_dim=4096, d_dict=131072, k=k, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def load_user_aen(k: int, device: str) -> AdaptiveElasticNetSAE:
    lambda_map = {32: "0p002", 64: "0p001", 128: "0p00075"}
    filename = f"checkpoints/llama8b/regularization/seed0/adaptive_elastic_net/lambda1_{lambda_map[k]}_lambda2_0p0001_gamma_0p5.pt"
    file_path = hf_hub_download(repo_id="farischaudhry/adaptive-elastic-net-sae", filename=filename)
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = AdaptiveElasticNetSAE(n_dim=4096, d_dict=131072, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

# =================================================================
# 2. TARGETED CAUSAL AUDITOR
# =================================================================

class TargetedCausalAuditor:
    def __init__(self, streamer, sae):
        self.model = streamer.model
        self.tokenizer = streamer.tokenizer
        self.sae = sae
        self.hook_name = streamer.hook_name
        self.device = sae.device

    @torch.no_grad()
    def audit_triplet(self, fid, prompt, target_token_str):
        # 0. Prep Tokens
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        # We target the effect on the NEXT token
        target_id = self.tokenizer.encode(target_token_str, add_special_tokens=False)[-1]
        
        # 1. Clean Pass
        clean_logits = self.model(tokens) # [1, seq, vocab]
        clean_lp = F.log_softmax(clean_logits[0, -1], dim=-1)

        # 2. Targeted Ablation Hook
        def ablation_hook(act, hook):
            sae_dtype = self.sae.dtype
            # Normalize exactly as SAE expects
            n = act.norm(p=2, dim=-1, keepdim=True) / (act.shape[-1]**0.5)
            xn = (act / n).to(sae_dtype)
            
            # Center and Encode
            h = self.sae.encode(xn)
            
            # Subtract the contribution of FID at the LAST position
            # Decoder weight for FID: weights shape is [4096, 131072]
            W_dec_col = self.sae.decoder.weight[:, fid]
            
            # Reconstruction contribution = activation * direction
            # h shape is [batch, seq, d_dict]
            val = h[0, -1, fid]
            contrib = val.unsqueeze(-1) * W_dec_col
            
            # Ablate: Remove this specific feature's signal
            patched_act = xn
            patched_act[0, -1, :] -= contrib
            
            # Return to original scale
            return (patched_act * n).to(act.dtype), val

        # 3. Patched Pass
        patched_logits, val = self.model.run_with_hooks(
            tokens, fwd_hooks=[(self.hook_name, ablation_hook)]
        )
        patch_lp = F.log_softmax(patched_logits[0, -1], dim=-1)

        # 4. Quantitative Results
        logit_drop = (clean_lp[target_id] - patch_lp[target_id]).item()
        
        # Measure Surgicality (Side effects on random vocabulary)
        control_ids = torch.randint(0, self.tokenizer.vocab_size, (100,))
        noise_shifts = [(clean_lp[cid] - patch_lp[cid]).abs().item() for cid in control_ids]
        avg_noise = np.mean(noise_shifts)

        return {
            "val": val.item(),
            "drop": logit_drop,
            "noise": avg_noise,
            "ratio": logit_drop / (avg_noise + 1e-6)
        }

# =================================================================
# 3. DATA FROM QUALITATIVE AUDIT
# =================================================================

def run_targeted_causal_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hook_layer=16,
        device=device,
        model_dtype="bfloat16"
    )
    # streamer used only for model/tokenizer access
    streamer = PythiaActivationStreamer(cfg=stream_cfg)

    aen_model = load_user_aen(k=32, device=device).eval()
    topk_model = load_user_topk(k=32, device=device).eval()

    # --- Triplet format: (FID, "Context Prompt", "Expected Next Token") ---
    # These are derived from your "bidirectional_features_contexts.txt"
    aen_test_triplets = [
        (578, "MAP ) ; %01766- 516", " 024"),   # UK Phone Code
        (81,  "un collier blanc, je ne peux", " pas"), # French Negation
        (972, "to try it for ourselves...kalo korang", " nak"), # Malay Noun
        (663, "RecorderConfig.getMaxStorageSize();", " //") # Code Comment
    ]

    topk_test_triplets = [
        (11, "Nathan Randall - Video", " Game"), # Compound Hub
        (247, 'onboarding_completed": true,', ' "'), # Syntax Hub
        (147, 'Feature Film $ 60 $48 Screenplay $', '50'), # Numeric Hub
        (30, 'Michael, Jr., and Michael', ' ,') # Punctuation Hub
    ]

    print("\n--- Starting TARGETED CAUSAL AUDIT ---")
    
    with open("targeted_causal_results.txt", "w") as f:
        f.write("TARGETED CAUSAL NECESSITY REPORT\n")
        f.write("Measures 'Logit Drop' on the specific token found in qualitative audit.\n")
        f.write("Surgicality Ratio = (Logit Drop) / (Avg Background Noise Shift)\n\n")

        # AUDIT AEN
        auditor_aen = TargetedCausalAuditor(streamer, aen_model)
        f.write("--- AEN SPECIALIZED FEATURES ---\n")
        for fid, prompt, target in aen_test_triplets:
            res = auditor_aen.audit_triplet(fid, prompt, target)
            f.write(f"FID #{fid} | Context: ...{prompt} | Target: '{target}'\n")
            f.write(f"  Act Val: {res['val']:.2f} | Logit Drop: {res['drop']:.2f} | Noise: {res['noise']:.4f} | RATIO: {res['ratio']:.1f}x\n\n")
            print(f"Audited AEN #{fid}: Ratio {res['ratio']:.1f}x")

        # AUDIT TOPK
        auditor_topk = TargetedCausalAuditor(streamer, topk_model)
        f.write("\n--- TOPK HUB FEATURES ---\n")
        for fid, prompt, target in topk_test_triplets:
            res = auditor_topk.audit_triplet(fid, prompt, target)
            f.write(f"FID #{fid} | Context: ...{prompt} | Target: '{target}'\n")
            f.write(f"  Act Val: {res['val']:.2f} | Logit Drop: {res['drop']:.2f} | Noise: {res['noise']:.4f} | RATIO: {res['ratio']:.1f}x\n\n")
            print(f"Audited TopK #{fid}: Ratio {res['ratio']:.1f}x")

if __name__ == "__main__":
    run_targeted_causal_audit()