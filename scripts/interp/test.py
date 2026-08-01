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
    def audit_context(self, fid, context_str, target_str):
        tokens = self.tokenizer.encode(context_str, return_tensors="pt").to(self.device)
        
        # 1. SCAN: Find where the feature fires in the full sentence
        h_all = []
        def capture_hook(act, hook):
            n = act.norm(p=2, dim=-1, keepdim=True) / (act.shape[-1]**0.5)
            xn = (act / n).to(self.sae.dtype)
            h = self.sae.encode(xn)
            h_all.append(h[0].cpu()) 
            return act

        self.model.run_with_hooks(tokens, fwd_hooks=[(self.hook_name, capture_hook)])
        f_acts = h_all[0][:, fid] 
        
        # Find peak position (must not be the very last token in the string)
        if f_acts[:-1].max() < 0.2:
            return {"error": f"Inactive (Max: {f_acts.max():.2f})"}
        
        pos = torch.argmax(f_acts[:-1]).item()
        act_val = f_acts[pos].item()
        
        # Determine the Ground Truth token ID for the token AFTER the peak
        target_id = tokens[0, pos+1].item()
        actual_target_str = self.tokenizer.decode([target_id])

        # 2. ABLATE at 'pos'
        def ablation_hook(act, hook):
            n = act.norm(p=2, dim=-1, keepdim=True) / (act.shape[-1]**0.5)
            xn = (act / n).to(self.sae.dtype)
            h_inner = self.sae.encode(xn)
            W_dec_col = self.sae.decoder.weight[:, fid]
            contrib = h_inner[0, pos, fid].unsqueeze(-1) * W_dec_col
            patched_act = xn.clone()
            patched_act[0, pos, :] -= contrib
            return (patched_act * n).to(act.dtype)

        clean_logits = self.model(tokens)
        patched_logits = self.model.run_with_hooks(tokens, fwd_hooks=[(self.hook_name, ablation_hook)])
        
        clean_lp = F.log_softmax(clean_logits[0, pos], dim=-1)
        patch_lp = F.log_softmax(patched_logits[0, pos], dim=-1)

        # 3. METRICS
        logit_drop = (clean_lp[target_id] - patch_lp[target_id]).item()
        control_ids = torch.randint(0, self.tokenizer.vocab_size, (100,))
        noise_shifts = [(clean_lp[cid] - patch_lp[cid]).abs().item() for cid in control_ids]
        avg_noise = np.mean(noise_shifts)

        return {
            "peak_token": self.tokenizer.decode([tokens[0, pos]]),
            "target_token": actual_target_str,
            "val": act_val,
            "drop": logit_drop,
            "noise": avg_noise,
            "ratio": logit_drop / (avg_noise + 1e-6)
        }

# =================================================================
# 3. RUNTIME
# =================================================================

def run_causal_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hook_layer=16,
        device=device,
        model_dtype="bfloat16"
    )
    streamer = PythiaActivationStreamer(cfg=stream_cfg)

    aen_model = load_user_aen(k=32, device=device).eval()
    topk_model = load_user_topk(k=32, device=device).eval()

    # (FID, Full Context String, Target hint for logging)
    aen_test_cases = [
        (81,  "un collier blanc, je ne peux pas refuser. La brave", "refuser"),
        (578, "MAP ) ; %01874- 611 880; www.robertos", "880"),
        (972, "kalo korang nak try tgk..pg kat blog die", "try"),
        (663, "RecorderConfig.getMaxStorageSize(); // Do something with max", "Do"),
        (1081, "price); return this; } public APIRequestUpdate", "public"),
        (1251, "empty line calls \\par (this one is already", "calls")
    ]

    topk_test_cases = [
        (0,   "its magnetic flux: i(t ) = W.", "="),
        (11,  "Nathan Randall - Video Game Analyst Nathan Randall", "Analyst"),
        (30,  "Michael, Jr., and Michael , Sr., also received", "Sr"),
        (147, "Feature Film $ 60 $48 Screenplay $ 50 $", "50"),
        (247, "onboarding_completed: true, \" profile_photo\": \"https", "profile")
    ]

    print("\n--- Starting TARGETED CAUSAL AUDIT ---")
    
    with open("targeted_causal_results_FINAL.txt", "w") as f:
        f.write("TARGETED CAUSAL NECESSITY REPORT\n\n")

        # AUDIT AEN
        auditor_aen = TargetedCausalAuditor(streamer, aen_model)
        f.write("--- AEN SPECIALIZED FEATURES ---\n")
        for fid, context, hint in aen_test_cases:
            res = auditor_aen.audit_context(fid, context, hint)
            if "error" in res:
                print(f"Skipping AEN #{fid}: {res['error']}")
                continue
            f.write(f"FID #{fid} | Peak Token: '{res['peak_token']}' | Target: '{res['target_token']}'\n")
            f.write(f"  Act Val: {res['val']:.2f} | Logit Drop: {res['drop']:.2f} | Noise: {res['noise']:.4f} | RATIO: {res['ratio']:.1f}x\n\n")
            print(f"Audited AEN #{fid} ('{res['peak_token']}'): Ratio {res['ratio']:.1f}x")

        # AUDIT TOPK
        auditor_topk = TargetedCausalAuditor(streamer, topk_model)
        f.write("\n--- TOPK HUB FEATURES ---\n")
        for fid, context, hint in topk_test_cases:
            res = auditor_topk.audit_context(fid, context, hint)
            if "error" in res:
                print(f"Skipping TopK #{fid}: {res['error']}")
                continue
            f.write(f"FID #{fid} | Peak Token: '{res['peak_token']}' | Target: '{res['target_token']}'\n")
            f.write(f"  Act Val: {res['val']:.2f} | Logit Drop: {res['drop']:.2f} | Noise: {res['noise']:.4f} | RATIO: {res['ratio']:.1f}x\n\n")
            print(f"Audited TopK #{fid} ('{res['peak_token']}'): Ratio {res['ratio']:.1f}x")

if __name__ == "__main__":
    run_causal_audit()