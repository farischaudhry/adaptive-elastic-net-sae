import sys
import random
from pathlib import Path
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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


def run_causal_comparison(num_examples_needed: int = 1, activation_threshold: float = 0.01):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    stream_cfg = LLMStreamConfig(
        tl_model_name="meta-llama/Llama-3.1-8B",
        hf_tokenizer_name="meta-llama/Llama-3.1-8B",
        dataset_name="EleutherAI/the_pile_deduplicated",
        hook_layer=16,
        seq_len=128,
        lm_batch_size=8,
        streaming=True,
        take_docs=20000,
        device=device,
        activation_normalization="per_token_l2"
    )
    streamer = PythiaActivationStreamer(cfg=stream_cfg)

    aen_model = load_user_aen(k=32, device=device).eval()
    topk_model = load_user_topk(k=32, device=device).eval()

    aen_targets = [578, 81, 972, 663]
    topk_targets = [247, 147, 30, 11]
    
    # Trackers
    found_counts = {fid: 0 for fid in (aen_targets + topk_targets)}
    reports = {fid: [] for fid in (aen_targets + topk_targets)}
    
    auditor_aen = CausalAudit(streamer, aen_model)
    auditor_topk = CausalAudit(streamer, topk_model)

    print(f"\n[Parallel Search] Looking for {aen_targets + topk_targets}")
    print(f"Threshold: {activation_threshold} | Examples per feature: {num_examples_needed}")

    batch_idx = 0
    pbar = tqdm(desc="Scanning Tokens")
    
    while any(count < num_examples_needed for count in found_counts.values()):
        batch_idx += 1
        try:
            tokens, x_norm = get_tokens_and_activations(streamer)
        except StopIteration: break
        
        # 1. Get activations for both models once
        with torch.no_grad():
            _, h_aen = aen_model(x_norm)
            _, h_topk = topk_model(x_norm)
        
        # 2. Check all targets in this batch
        for fid in (aen_targets + topk_targets):
            if found_counts[fid] >= num_examples_needed:
                continue
            
            is_aen = fid in aen_targets
            h = h_aen if is_aen else h_topk
            auditor = auditor_aen if is_aen else auditor_topk
            
            f_acts = h[:, :, fid]
            if f_acts.max() >= activation_threshold:
                # Found one! Perform the causal audit immediately
                # We reuse the tokens/activations we already have
                b, s = torch.where(f_acts == f_acts.max())
                b, s = b[0].item(), s[0].item()
                if s >= tokens.shape[1] - 1: continue 

                # Target calculation
                target_token_id = tokens[b, s+1].item()
                control_ids = torch.randint(0, streamer.tokenizer.vocab_size, (15,))

                # Perform the patching
                def ablation_hook(act, hook):
                    sae_dtype = auditor.sae.dtype 
                    n = act.norm(p=2, dim=-1, keepdim=True) / (act.shape[-1]**0.5)
                    xn = (act / n).to(sae_dtype)
                    h_inner = auditor.sae.encode(xn)
                    W_dec_col = auditor.sae.decoder.weight[:, fid]
                    contrib = h_inner[:, :, fid].unsqueeze(-1) * W_dec_col
                    patched_act = (xn - contrib) * n
                    return patched_act.to(act.dtype)

                with torch.no_grad():
                    clean_logits = auditor.model(tokens[b:b+1])
                    patched_logits = auditor.model.run_with_hooks(
                        tokens[b:b+1], fwd_hooks=[(auditor.hook_name, ablation_hook)]
                    )
                
                clean_lp = F.log_softmax(clean_logits[0, s], dim=-1)
                patch_lp = F.log_softmax(patched_logits[0, s], dim=-1)
                
                target_drop = (clean_lp[target_token_id] - patch_lp[target_token_id]).item()
                avg_noise = torch.mean(torch.tensor([(clean_lp[cid] - patch_lp[cid]).abs().item() for cid in control_ids])).item()

                reports[fid].append({
                    "context": streamer.tokenizer.decode(tokens[b, max(0, s-10):s+1]),
                    "target": streamer.tokenizer.decode([target_token_id]),
                    "drop": target_drop,
                    "noise": avg_noise,
                    "ratio": target_drop / (avg_noise + 1e-6),
                })
                found_counts[fid] += 1
                print(f"\n  [FOUND] FID #{fid} ({found_counts[fid]}/{num_examples_needed}) | Ratio: {reports[fid][-1]['ratio']:.1f}x")

        pbar.update(1)

    with open("rebuttal_causal_results.txt", "w") as f:
        for fid, entries in reports.items():
            f.write(f"\nAUDIT REPORT: FID #{fid}\n")
            for e in entries:
                f.write(f"  Context: ...{e['context']}\n")
                f.write(f"  Target: '{e['target']}' | Drop: {e['drop']:.2f} | Noise: {e['noise']:.4f} | RATIO: {e['ratio']:.1f}x\n\n")


if __name__ == "__main__":
    run_causal_comparison()
