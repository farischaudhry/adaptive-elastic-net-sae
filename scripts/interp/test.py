import sys
import torch
import torch.nn.functional as F
from pathlib import Path
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
# 2. GENERATIVE STEERING SUITE
# =================================================================

class GenerativeSteerer:
    def __init__(self, streamer, sae):
        self.model = streamer.model
        self.tokenizer = streamer.tokenizer
        self.sae = sae
        self.hook_name = streamer.hook_name
        self.device = sae.device

    def get_top_tokens(self, probs, k=5):
        """Helper to decode top probabilities."""
        vals, idxs = torch.topk(probs, k=k)
        return [f"'{self.tokenizer.decode([idx.item()])}' ({vals[i].item()*100:.1f}%)" for i, idx in enumerate(idxs)]

    @torch.no_grad()
    def steer_and_observe(self, fid, strength=80.0):
        # We start with a completely neutral prompt to see the pure effect of the feature
        prompt = " " 
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 1. Baseline Predictions (No Steering)
        clean_logits = self.model(tokens)
        clean_probs = F.softmax(clean_logits[0, -1], dim=-1)
        before_str = self.get_top_tokens(clean_probs)

        # 2. Steering Hook
        def steering_hook(act, hook):
            # Normalize to SAE scale
            n = act.norm(p=2, dim=-1, keepdim=True) / (act.shape[-1]**0.5)
            xn = (act / n).to(self.sae.dtype)
            
            # INJECT: Add a high activation value in the direction of FID
            # Decoder shape is [n_dim, d_dict] -> col fid is the concept direction
            direction = self.sae.decoder.weight[:, fid]
            xn[0, -1, :] += strength * direction
            
            return (xn * n).to(act.dtype)

        # 3. Steered Predictions
        steered_logits = self.model.run_with_hooks(
            tokens, fwd_hooks=[(self.hook_name, steering_hook)]
        )
        steered_probs = F.softmax(steered_logits[0, -1], dim=-1)
        after_str = self.get_top_tokens(steered_probs)

        return {"before": before_str, "after": after_str}

# =================================================================
# 3. RUNTIME
# =================================================================

def run_causal_sufficiency_audit():
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

    # Targets from our monosemantic audit
    aen_targets = {
        81: "French Negation (pas)",
        578: "UK Phone Code (01766)",
        972: "Malay colloquial (nak)",
        663: "Code Comment (//)"
    }
    
    # Hub targets from TopK
    topk_targets = {
        11: "Compound Noun Hub",
        247: "Syntax/Num Hub",
        30: "Punctuation/Space Hub"
    }

    print("\n--- Starting CAUSAL SUFFICIENCY (Steering) AUDIT ---")
    
    with open("causal_steering_results.txt", "w") as f:
        f.write("CAUSAL SUFFICIENCY REPORT: BEHAVIORAL CONTROL\n")
        f.write("Injected +80.0 activation into a neutral space prompt.\n")
        f.write("Goal: Prove AEN features are surgical semantic levers.\n\n")

        # AUDIT AEN
        steerer_aen = GenerativeSteerer(streamer, aen_model)
        f.write("--- AEN SPECIALIZED FEATURES ---\n")
        for fid, desc in aen_targets.items():
            res = steerer_aen.steer_and_observe(fid)
            f.write(f"FID #{fid} [{desc}]\n")
            f.write(f"  TOP 5 BEFORE: {res['before']}\n")
            f.write(f"  TOP 5 AFTER:  {res['after']}\n\n")
            print(f"Audited AEN #{fid} ({desc})")

        # AUDIT TOPK
        steerer_topk = GenerativeSteerer(streamer, topk_model)
        f.write("\n--- TOPK HUB FEATURES ---\n")
        for fid, desc in topk_targets.items():
            res = steerer_topk.steer_and_observe(fid)
            f.write(f"FID #{fid} [{desc}]\n")
            f.write(f"  TOP 5 BEFORE: {res['before']}\n")
            f.write(f"  TOP 5 AFTER:  {res['after']}\n\n")
            print(f"Audited TopK #{fid} ({desc})")

if __name__ == "__main__":
    run_causal_sufficiency_audit()