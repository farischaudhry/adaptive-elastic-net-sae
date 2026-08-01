import sys
import random
from pathlib import Path
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent))

from adaptive_elastic_sae.saes.top_k import TopKSAE
from adaptive_elastic_sae.saes.polyhedral import AdaptiveElasticNetSAE
from adaptive_elastic_sae.data.llm_streamer import LLMStreamConfig, PythiaActivationStreamer, normalize_activations


def load_user_topk(k: int, device: str) -> TopKSAE:
    sweep_map = {32: "000", 64: "001", 128: "002"}
    filename = f"checkpoints/llama8b/topk/seed0/k{k}_llm-topk_baseline_sweep{sweep_map[k]}-seed0.pt"
    print(f"Downloading Vanilla Top-K (k={k}) from HF.")
    file_path = hf_hub_download(repo_id="farischaudhry/adaptive-elastic-net-sae", filename=filename)
    checkpoint = torch.load(file_path, map_location=device, weights_only=False)
    model = TopKSAE(n_dim=4096, d_dict=131072, k=k, device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def load_user_aen(k: int, device: str) -> AdaptiveElasticNetSAE:
    lambda_map = {32: "0p002", 64: "0p001", 128: "0p00075"}
    filename = f"checkpoints/llama8b/regularization/seed0/adaptive_elastic_net/lambda1_{lambda_map[k]}_lambda2_0p0001_gamma_0p5.pt"
    print(f"Downloading AEN-SAE (k={k}) from HF.")
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


class CausalAudit:
    def __init__(self, streamer, sae):
        self.streamer = streamer
        self.tokenizer = streamer.tokenizer
        self.model = streamer.model
        self.sae = sae
        self.hook_name = streamer.hook_name

    def get_top_tokens(self, logprobs, k=5):
        vals, idxs = torch.topk(logprobs, k=k)
        return [(self.tokenizer.decode([idx.item()]), f"{vals[i].item():.2f}") for i, idx in enumerate(idxs)]


    @torch.no_grad()
    def run_surgical_audit(self, fid, num_examples=3, threshold=2.0):
        """
        Calculates:
        1. Logit Drop on the correct token when feature is killed.
        2. Surgicality Ratio: Impact on target vs. Impact on unrelated background.
        """
        self.streamer.reset_stream()
        found = 0
        report = []

        while found < num_examples:
            try:
                tokens, x_norm = get_tokens_and_activations(self.streamer)
            except StopIteration: break
            
            with torch.no_grad():
                _, h = self.sae(x_norm)
            
            f_acts = h[:, :, fid]
            if f_acts.max() < threshold: continue
            
            # Find strongest activation in batch
            b, s = torch.where(f_acts == f_acts.max())
            b, s = b[0].item(), s[0].item()
            if s >= tokens.shape[1] - 1: continue 

            target_token_id = tokens[b, s+1].item()
            target_str = self.tokenizer.decode([target_token_id])
            
            # Control: Pick random tokens to measure "background noise" shift
            control_ids = torch.randint(0, self.tokenizer.vocab_size, (20,))

            # --- ABLATION HOOK ---
            def ablation_hook(act, hook):
                # Re-normalize to SAE input scale
                n = act.norm(p=2, dim=-1, keepdim=True) / (act.shape[-1]**0.5)
                xn = act / n
                _, h_inner = self.sae(xn)
                
                # Zero out FID
                W_dec = self.sae.W_dec[fid]
                contrib = h_inner[:, :, fid].unsqueeze(-1) * W_dec
                return (xn - contrib) * n

            # Run patched model
            with torch.no_grad():
                clean_logits = self.model(tokens[b:b+1])
                patched_logits = self.model.run_with_hooks(
                    tokens[b:b+1], fwd_hooks=[(self.hook_name, ablation_hook)]
                )
            
            clean_lp = F.log_softmax(clean_logits[0, s], dim=-1)
            patch_lp = F.log_softmax(patched_logits[0, s], dim=-1)
            
            # Quantitative Metrics
            target_drop = (clean_lp[target_token_id] - patch_lp[target_token_id]).item()
            
            control_drops = []
            for cid in control_ids:
                control_drops.append((clean_lp[cid] - patch_lp[cid]).abs().item())
            avg_noise = np.mean(control_drops)
            
            # Qualitative Metrics
            report.append({
                "context": self.tokenizer.decode(tokens[b, max(0, s-10):s+1]),
                "target": target_str,
                "drop": target_drop,
                "noise": avg_noise,
                "ratio": target_drop / (avg_noise + 1e-6),
                "before": self.get_top_tokens(clean_lp, k=3),
                "after": self.get_top_tokens(patch_lp, k=3)
            })
            found += 1
            
        return report


def run_causal_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hf_tokenizer_name="meta-llama/Llama-3.1-8B",
        dataset_name="EleutherAI/the_pile_deduplicated",
        hook_layer=16,
        seq_len=128,
        lm_batch_size=8,
        streaming=True,
        take_docs=5000,
        device=device,
        activation_normalization="per_token_l2"
    )
    streamer = PythiaActivationStreamer(cfg=stream_cfg)

    aen_model = load_user_aen(k=32, device=device).eval()
    topk_model = load_user_topk(k=32, device=device).eval()

    # --- LIST TARGETS FROM QUALITATIVE ANALYSIS ---
    aen_targets = [578, 81, 972, 663]
    topk_targets = [247, 147, 30, 11]

    with open("rebuttal_causal_results.txt", "w") as f:
        f.write("CAUSAL SURGICALITY REPORT: AEN vs TOPK\n")
        f.write("Surgicality Ratio = (Logit Drop on Target) / (Avg Logit Shift on Unrelated Tokens)\n")
        f.write("Higher Ratio = More monosemantic causal lever.\n\n")

        # Audit AEN
        auditor_aen = CausalAudit(streamer, aen_model)
        for fid in aen_targets:
            f.write(f"\nAUDIT: AEN Specialized Feature #{fid}\n")
            results = auditor_aen.run_surgical_audit(fid)
            for r in results:
                f.write(f"  Context: ...{r['context']}\n")
                f.write(f"  Target: '{r['target']}' | Drop: {r['drop']:.2f} | Noise: {r['noise']:.4f} | RATIO: {r['ratio']:.1f}x\n")
                f.write(f"  Top Preds Before: {r['before']}\n")
                f.write(f"  Top Preds After:  {r['after']}\n\n")

        # Audit TopK
        auditor_topk = CausalAudit(streamer, topk_model)
        for fid in topk_targets:
            f.write(f"\nAUDIT: TopK Hub Feature #{fid}\n")
            results = auditor_topk.run_surgical_audit(fid)
            for r in results:
                f.write(f"  Context: ...{r['context']}\n")
                f.write(f"  Target: '{r['target']}' | Drop: {r['drop']:.2f} | Noise: {r['noise']:.4f} | RATIO: {r['ratio']:.1f}x\n")
                f.write(f"  Top Preds Before: {r['before']}\n")
                f.write(f"  Top Preds After:  {r['after']}\n\n")


if __name__ == "__main__":
    run_causal_comparison()