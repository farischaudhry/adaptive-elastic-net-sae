import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from huggingface_hub import hf_hub_download

# Assuming current dir is /workspace/adaptive-elastic-sae/scripts/interp
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
# 2. CAUSAL STEERING SUITE
# =================================================================

class CausalSteerer:
    def __init__(self, streamer, sae):
        self.streamer = streamer
        self.model = streamer.model
        self.tokenizer = streamer.tokenizer
        self.sae = sae
        self.hook_name = streamer.hook_name
        self.device = sae.device
        
        # Pre-compute logit directions (approximate linear effect of feature on vocabulary)
        # Note: In Llama, unembedding is self.model.W_U
        print("Pre-computing logit effects for all features...")
        self.W_U = self.model.W_U # [d_model, n_vocab]

    def get_top_tokens_for_feature(self, fid, k=10):
        """Identifies the 'Target Concept' of a feature by looking at its decoder weights."""
        # weight[:, fid] is the [4096] direction in activation space
        direction = self.sae.decoder.weight[:, fid] 
        # Project into logit space
        logits = direction @ self.W_U # [32000]
        vals, idxs = torch.topk(logits, k=k)
        return idxs.cpu().tolist()

    @torch.no_grad()
    def run_steering_audit(self, fid, steer_strength=30.0):
        """
        Intervention: Injects feature FID into a neutral prompt.
        Measures: How much the promoted tokens increase in probability.
        """
        prompt = "The following content is related to"
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        seq_len = tokens.shape[1]
        
        # 1. Get Target Tokens (Ground Truth of what the feature 'means')
        target_token_ids = self.get_top_tokens_for_feature(fid, k=5)
        target_token_strs = [self.tokenizer.decode([tid]) for tid in target_token_ids]
        
        # 2. Clean Forward Pass
        clean_logits = self.model(tokens) # [1, seq, vocab]
        clean_lp = F.log_softmax(clean_logits[0, -1], dim=-1)

        # 3. Steering Hook
        def steering_hook(act, hook):
            # Injection happens at the very last token in the prompt
            # act: [batch, seq, d_model]
            sae_dtype = self.sae.dtype
            
            # Normalize to SAE scale
            n = act.norm(p=2, dim=-1, keepdim=True) / (act.shape[-1]**0.5)
            xn = (act / n).to(sae_dtype)
            
            # Get decoder weight column
            W_dec_col = self.sae.decoder.weight[:, fid]
            
            # Injection: x_new = x + (strength * direction)
            # Only apply to the last token position
            xn[:, -1, :] += steer_strength * W_dec_col
            
            return (xn * n).to(act.dtype)

        # 4. Patched Forward Pass
        patched_logits = self.model.run_with_hooks(
            tokens, fwd_hooks=[(self.hook_name, steering_hook)]
        )
        patch_lp = F.log_softmax(patched_logits[0, -1], dim=-1)

        # 5. Metrics
        # Increase on intended target tokens
        target_gains = [(patch_lp[tid] - clean_lp[tid]).item() for tid in target_token_ids]
        avg_target_gain = np.mean(target_gains)

        # Surgicality: compare to random background shift
        control_ids = torch.randint(0, self.tokenizer.vocab_size, (50,))
        noise_shifts = [(patch_lp[cid] - clean_lp[cid]).abs().item() for cid in control_ids]
        avg_noise = np.mean(noise_shifts)

        # Qualitative: What is now top?
        top_patched_vals, top_patched_idxs = torch.topk(patch_lp, k=5)
        top_patched_strs = [self.tokenizer.decode([idx.item()]) for idx in top_patched_idxs]

        return {
            "fid": fid,
            "targets": target_token_strs,
            "avg_gain": avg_target_gain,
            "avg_noise": avg_noise,
            "ratio": avg_target_gain / (avg_noise + 1e-6),
            "new_top": top_patched_strs
        }

# =================================================================
# 3. RUNTIME
# =================================================================

def run_rebuttal_audit():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Models
    aen_model = load_user_aen(k=32, device=device).eval()
    topk_model = load_user_topk(k=32, device=device).eval()

    # Reuse your existing streamer for model/tokenizer access
    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hook_layer=16,
        device=device,
        model_dtype="bfloat16"
    )
    # Mock streamer just to get the HookedTransformer instance
    from adaptive_elastic_sae.data.llm_streamer import PythiaActivationStreamer
    streamer = PythiaActivationStreamer(cfg=stream_cfg)

    # TARGETS
    aen_targets = [578, 81, 972, 663] # Rare monosemantic
    topk_targets = [247, 147, 30, 11] # Hubs
    
    print("\n--- Starting FORCE STEERING (Causal Utility Audit) ---")
    
    with open("rebuttal_steering_results.txt", "w") as f:
        f.write("CAUSAL UTILITY REPORT: AEN vs TOPK (Force Steering)\n")
        f.write("Target Gain: Avg logprob increase for the tokens the feature represents.\n")
        f.write("Surgicality Ratio: Target Gain / Absolute shift in random background tokens.\n\n")

        # AEN
        steerer_aen = CausalSteerer(streamer, aen_model)
        f.write("--- AEN SPECIALIZED FEATURES ---\n")
        for fid in aen_targets:
            res = steerer_aen.run_steering_audit(fid)
            f.write(f"FID #{fid} | Intended: {res['targets']}\n")
            f.write(f"  Gain on Target: {res['avg_gain']:.2f} | Noise Shift: {res['avg_noise']:.4f} | RATIO: {res['ratio']:.1f}x\n")
            f.write(f"  NEW TOP PREDICTIONS: {res['new_top']}\n\n")
            print(f"Audited AEN #{fid}: Ratio {res['ratio']:.1f}x")

        # TopK
        steerer_topk = CausalSteerer(streamer, topk_model)
        f.write("\n--- TOPK HUB FEATURES ---\n")
        for fid in topk_targets:
            res = steerer_topk.run_steering_audit(fid)
            f.write(f"FID #{fid} | Intended: {res['targets']}\n")
            f.write(f"  Gain on Target: {res['avg_gain']:.2f} | Noise Shift: {res['avg_noise']:.4f} | RATIO: {res['ratio']:.1f}x\n")
            f.write(f"  NEW TOP PREDICTIONS: {res['new_top']}\n\n")
            print(f"Audited TopK #{fid}: Ratio {res['ratio']:.1f}x")


if __name__ == "__main__":
    run_rebuttal_audit()