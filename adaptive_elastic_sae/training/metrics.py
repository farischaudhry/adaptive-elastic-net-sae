from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def dead_neurons_pct(max_activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Percent of features that never activated above eps in a window."""
    dead_mask = max_activations <= eps
    return 100.0 * dead_mask.float().mean().item()


def dead_neuron_recovery_rate(
    prev_max_activations: torch.Tensor,
    curr_max_activations: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Fraction of previously dead features that became alive in the next window."""
    prev_dead = prev_max_activations <= eps
    denom = prev_dead.sum().item()
    if denom == 0:
        return 0.0

    curr_alive = curr_max_activations > eps
    recovered = (prev_dead & curr_alive).sum().item()
    return recovered / denom


@torch.no_grad()
def interaction_leakage_frobenius_approx(
    decoder: torch.Tensor,
    active_mask: torch.Tensor,
    max_active_features: int = 4096,
) -> float:
    """Approximate leakage as ||D_Ac^T D_A||_F (avoids unstable inverse term)."""
    if active_mask.dtype != torch.bool:
        active_mask = active_mask.bool()

    active_count = int(active_mask.sum().item())
    inactive_count = int((~active_mask).sum().item())
    if active_count == 0 or inactive_count == 0:
        return 0.0

    # Skip unstable/expensive blocks before materializing huge products.
    if active_count > max_active_features:
        return float("nan")

    d_a = decoder[:, active_mask]
    d_ac = decoder[:, ~active_mask]
    return torch.norm(d_ac.T @ d_a, p="fro").item()


@torch.no_grad()
def active_gram_spectrum(
    decoder: torch.Tensor,
    active_mask: torch.Tensor,
    eps: float = 1e-10,
) -> dict[str, float]:
    """Return min/max eigenvalues and condition number of active Gram block.

    The eigendecomposition is skipped when the active set is degenerate or larger
    than the model dimension to avoid expensive/unstable geometry evaluations.
    """
    if active_mask.dtype != torch.bool:
        active_mask = active_mask.bool()

    active_count = int(active_mask.sum().item())
    n_dim = int(decoder.shape[0])
    if active_count <= 1 or active_count > n_dim:
        return {}

    d_a = decoder[:, active_mask]
    gram = d_a.T @ d_a

    # Guard eigendecomposition against invalid inputs.
    if not torch.isfinite(gram).all():
        return {
            "active_min_eigenvalue": float("nan"),
            "active_max_eigenvalue": float("nan"),
            "active_condition_number": float("nan"),
        }

    try:
        eigvals = torch.linalg.eigvalsh(gram)
        if not torch.isfinite(eigvals).all():
            raise RuntimeError("Non-finite eigenvalues in active_gram_spectrum")
    except RuntimeError:
        return {
            "active_min_eigenvalue": float("nan"),
            "active_max_eigenvalue": float("nan"),
            "active_condition_number": float("nan"),
        }

    min_eig = torch.clamp(eigvals.min(), min=eps)
    max_eig = eigvals.max()

    return {
        "active_min_eigenvalue": min_eig.item(),
        "active_max_eigenvalue": max_eig.item(),
        "active_condition_number": (max_eig / min_eig).item(),
    }


@torch.no_grad()
def compute_cross_leverage(
    decoder: torch.Tensor,
    active_mask: torch.Tensor,
    k_top: int = 5,
    eps: float = 1e-12,
    max_active_features: int = 4096,
) -> dict[str, Any]:
    """
    Computes h_j(A) = (1/n) ||P_A a_j||^2.
    Identifies features 'shadowed' by the active subspace.
    """
    if not active_mask.any():
        return {
            "mean_h_j": 0.0,
            "total_h_j": 0.0,
            "shadowed_ids": [],
            "shadowed_scores": [],
        }

    active_count = int(active_mask.sum().item())
    if active_count > max_active_features:
        return {
            "mean_h_j": float("nan"),
            "total_h_j": float("nan"),
            "shadowed_ids": [],
            "shadowed_scores": [],
        }

    n_dim = decoder.shape[0]
    inactive_indices = torch.where(~active_mask)[0]

    d_a = decoder[:, active_mask]  # Active vectors
    d_ac = decoder[:, ~active_mask]  # Inactive vectors

    # Local Gram matrices
    g_aa = (d_a.T @ d_a) / n_dim
    g_aca = (d_ac.T @ d_a) / n_dim

    try:
        # Regression of inactive onto active
        sol = torch.linalg.lstsq(
            g_aa + eps * torch.eye(g_aa.shape[0], device=g_aa.device), g_aca.T
        )
        inv_g_aa_g_at_ac = sol.solution
    except (RuntimeError, torch.linalg.LinAlgError):
        # Fallback for catastrophic failure:
        # Return -1.0 to indicate a singularity in wandb logs.
        return {
            "mean_h_j": -1.0,
            "total_h_j": -1.0,
            "shadowed_ids": [],
            "shadowed_scores": [],
        }

    # h_j alignment scores
    h_j = n_dim * torch.sum(g_aca * inv_g_aa_g_at_ac.T, dim=1)

    actual_k = min(k_top, h_j.size(0))
    top_scores, top_local_idx = torch.topk(h_j, k=actual_k)

    return {
        "mean_h_j": h_j.mean().item(),
        "total_h_j": h_j.sum().item(),
        "shadowed_ids": inactive_indices[top_local_idx].tolist(),
        "shadowed_scores": top_scores.tolist(),
    }


def feature_shrinkage_ratio(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Measures reconstruction shrinkage pathology as ||x_hat||_1 / ||x||_1.
    A ratio < 1.0 indicates magnitude suppression.
    """
    norm_x_hat = x_hat.abs().sum(dim=1).mean()
    norm_x = torch.clamp(x.abs().sum(dim=1).mean(), min=eps)
    return (norm_x_hat / norm_x).item()


def l0_active_features(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Average number of active features per sample."""
    return (activations.abs() > eps).sum(dim=1).float().mean().item()


def l0_vs_l1_ratio(activations: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute ||a||_1 / ||a||_0 averaged across samples."""
    l1 = activations.abs().sum(dim=1)
    l0 = torch.clamp((activations.abs() > eps).sum(dim=1).float(), min=1.0)
    return (l1 / l0).mean().item()


def explained_variance(
    x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-12
) -> float:
    """1 - Var(x - x_hat) / Var(x) calculated per-feature and summed."""
    residual_var = (x - x_hat).var(dim=0, unbiased=False).sum()
    total_var = torch.clamp(x.var(dim=0, unbiased=False).sum(), min=eps)
    return (1.0 - residual_var / total_var).item()



@torch.no_grad()
def dictionary_coherence_summary(
    decoder: torch.Tensor,
    eps: float = 1e-10,
    chunk_size: int = 1024,
) -> dict[str, Any]:
    """Summarize dictionary coherence without materializing a full NxN matrix.

    Coherence is defined per atom as the maximum absolute cosine similarity to
    any other atom in the dictionary.
    """
    if decoder.numel() == 0 or decoder.shape[1] == 0:
        return {
            "mean_max_cosine_similarity": 0.0,
            "dictionary_coherence_nn_max": 0.0,
            "dictionary_coherence_abs_max": 0.0,
            "dictionary_coherence_used_features": 0.0,
            "dictionary_coherence_total_features": 0.0,
            "dictionary_coherence_subsampled": 0.0,
            "dictionary_coherence_overall_mean": 0.0,
        }

    total_features = int(decoder.shape[1])
    if total_features <= 1:
        return {
            "mean_max_cosine_similarity": 0.0,
            "dictionary_coherence_nn_max": 0.0,
            "dictionary_coherence_abs_max": 0.0,
            "dictionary_coherence_used_features": float(total_features),
            "dictionary_coherence_total_features": float(total_features),
            "dictionary_coherence_subsampled": 0.0,
            "dictionary_coherence_overall_mean": 0.0,
        }

    d = decoder / torch.clamp(decoder.norm(dim=0, keepdim=True), min=eps)

    max_chunks: list[torch.Tensor] = []
    mean_chunks: list[torch.Tensor] = []
    global_abs_max = torch.tensor(0.0, device=d.device)

    step = max(1, int(chunk_size))
    for i in range(0, total_features, step):
        end_i = min(i + step, total_features)
        d_chunk = d[:, i:end_i]

        # Shape: (chunk, d_dict)
        sim_chunk = d_chunk.T @ d
        abs_chunk = sim_chunk.abs()

        # Remove self-similarity entries for rows represented by this chunk.
        local_rows = torch.arange(end_i - i, device=d.device)
        global_cols = torch.arange(i, end_i, device=d.device)
        abs_chunk[local_rows, global_cols] = 0.0

        row_max = abs_chunk.max(dim=1).values
        row_mean = abs_chunk.sum(dim=1) / float(total_features - 1)

        max_chunks.append(row_max)
        mean_chunks.append(row_mean)
        global_abs_max = torch.maximum(global_abs_max, row_max.max())

    nn_max = torch.cat(max_chunks)
    mean_abs = torch.cat(mean_chunks)

    metrics = summary_stats(
        nn_max,
        "dictionary_coherence_nn",
        include_histogram=True,
        histogram_key="dictionary_coherence_histogram",
    )
    metrics["mean_max_cosine_similarity"] = nn_max.mean().item()
    metrics["dictionary_coherence_nn_max"] = nn_max.max().item()
    metrics["dictionary_coherence_abs_max"] = global_abs_max.item()
    metrics["dictionary_coherence_overall_mean"] = mean_abs.mean().item()
    metrics["dictionary_coherence_used_features"] = float(total_features)
    metrics["dictionary_coherence_total_features"] = float(total_features)
    metrics["dictionary_coherence_subsampled"] = 0.0
    return metrics


def activation_effective_sample_size(
    mean_abs_activations: torch.Tensor, eps: float = 1e-12
) -> float:
    """ESS proxy: (sum a_i)^2 / sum a_i^2 for nonnegative feature activity."""
    a = torch.clamp(mean_abs_activations, min=0.0)
    num = a.sum() ** 2
    den = torch.clamp((a**2).sum(), min=eps)
    return (num / den).item()


def weight_bimodality_ratio(
    weights: torch.Tensor,
    delta_sig: float,
    delta_noise: float,
    eps: float = 1e-12,
) -> float:
    """Ratio of low-weight (signal) to high-weight (noise) feature counts."""
    low = (weights < delta_sig).sum().float()
    high = torch.clamp((weights > delta_noise).sum().float(), min=eps)
    return (low / high).item()


def summary_stats(
    values: list[float] | torch.Tensor,
    prefix: str,
    include_histogram: bool = False,
    histogram_key: str | None = None,
) -> dict[str, Any]:
    """Return mean/std/p10/p50/p90 summaries for scalar samples."""
    if isinstance(values, list):
        if len(values) == 0:
            return {}
        tensor = torch.tensor(values, dtype=torch.float32)
    else:
        if values.numel() == 0:
            return {}
        tensor = values.detach().float().reshape(-1)

    q = torch.tensor([0.1, 0.5, 0.9], device=tensor.device, dtype=tensor.dtype)
    quantiles = torch.quantile(tensor, q)
    metrics: dict[str, Any] = {
        f"{prefix}/mean": tensor.mean().item(),
        f"{prefix}/std": tensor.std(unbiased=False).item(),
        f"{prefix}/p10": quantiles[0].item(),
        f"{prefix}/p50": quantiles[1].item(),
        f"{prefix}/p90": quantiles[2].item(),
    }

    if include_histogram:
        try:
            import wandb

            metrics[histogram_key or f"{prefix}/histogram"] = wandb.Histogram(
                tensor.detach().cpu().numpy()
            )
        except Exception:
            pass

    return metrics


def adaptive_weight_summary(
    weights: torch.Tensor,
    prefix: str,
    weight_min: float | None = None,
    weight_max: float | None = None,
    bound_eps: float = 1e-4,
) -> dict[str, Any]:
    """Quantile and bound-hit summaries for adaptive feature weights."""
    if weights.numel() == 0:
        return {}

    w = weights.detach().float().reshape(-1)
    metrics = summary_stats(w, prefix, include_histogram=True)

    if weight_min is not None:
        metrics[f"{prefix}/pinned_min_pct"] = (
            100.0 * (w <= (float(weight_min) + bound_eps)).float().mean().item()
        )
    if weight_max is not None:
        metrics[f"{prefix}/pinned_max_pct"] = (
            100.0 * (w >= (float(weight_max) - bound_eps)).float().mean().item()
        )
    return metrics


def feature_utilization_summary(
    firing_rates: torch.Tensor,
    prefix: str,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Summarize dictionary utilization and concentration from per-feature firing rates."""
    if firing_rates.numel() == 0:
        return {}

    rates = torch.clamp(firing_rates.detach().float().reshape(-1), min=0.0)
    n = rates.numel()
    if n == 0:
        return {}

    metrics: dict[str, float] = {
        f"{prefix}/active_feature_pct": 100.0 * (rates > eps).float().mean().item(),
    }

    metrics.update(summary_stats(rates, f"{prefix}/rate", include_histogram=True))
    log_rates = torch.log10(torch.clamp(rates, min=eps))
    metrics.update(
        summary_stats(log_rates, f"{prefix}/log10_rate", include_histogram=True)
    )

    rate_sum = rates.sum().item()
    if rate_sum <= eps:
        metrics[f"{prefix}/normalized_entropy"] = 0.0
        metrics[f"{prefix}/top1pct_mass_pct"] = 0.0
        metrics[f"{prefix}/top10pct_mass_pct"] = 0.0
        metrics[f"{prefix}/gini"] = 0.0
        return metrics

    probs = rates / rate_sum
    entropy = -(probs * torch.log(torch.clamp(probs, min=eps))).sum()
    metrics[f"{prefix}/normalized_entropy"] = (
        entropy / max(torch.log(torch.tensor(float(n), device=probs.device)), eps)
    ).item()

    sorted_probs, _ = torch.sort(probs, descending=True)
    k1 = max(1, int(round(0.01 * n)))
    k10 = max(1, int(round(0.10 * n)))
    metrics[f"{prefix}/top1pct_mass_pct"] = 100.0 * sorted_probs[:k1].sum().item()
    metrics[f"{prefix}/top10pct_mass_pct"] = 100.0 * sorted_probs[:k10].sum().item()

    sorted_rates, _ = torch.sort(rates)
    idx = torch.arange(1, n + 1, device=sorted_rates.device, dtype=sorted_rates.dtype)
    gini_num = 2.0 * torch.sum(idx * sorted_rates)
    gini_den = n * torch.clamp(sorted_rates.sum(), min=eps)
    gini = gini_num / gini_den - (n + 1) / n
    metrics[f"{prefix}/gini"] = float(gini.item())

    return metrics


def compute_synthetic_recovery_metrics(
    d_learned: torch.Tensor,  # [n_dim, d_dict] or [d_dict, n_dim]
    d_star: torch.Tensor,  # [n_dim, d_dict]
    h_pred: torch.Tensor,  # [batch_size, d_dict]
    h_star: torch.Tensor,  # [batch_size, d_dict]
    eps: float = 1e-6,
) -> dict[str, float]:
    """
    Computes dictionary error d(D, D*) (Cosine and Frobenius), 
    code recovery error ||h - h*||_2, and support recovery (F1 score and Jaccard similarity).
    """
    # Ensure d_learned and d_star are [n_dim, d_dict]
    if d_learned.shape[0] != d_star.shape[0]:
        d_learned = d_learned.T
        
    # Normalize columns
    d_learned_norm = d_learned / (torch.norm(d_learned, dim=0, keepdim=True) + eps)
    d_star_norm = d_star / (torch.norm(d_star, dim=0, keepdim=True) + eps)

    # Dictionary Error d(D, D*) (Cosine and Frobenius)
    # Compute pairwise absolute cosine similarity matrix C: [d_dict, d_dict]
    C = torch.abs(d_learned_norm.T @ d_star_norm).detach().cpu().numpy()
    cost_matrix = 1.0 - C
    learned_idx, star_idx = linear_sum_assignment(cost_matrix)
    # Dictionary Error d(D, D*)
    # Average cosine distance between matched atoms
    matched_cosines = C[learned_idx, star_idx]
    dictionary_error_cosine = 1.0 - float(np.mean(matched_cosines))
    # Frobenius Error on aligned normalized dictionaries
    d_learned_norm_aligned = torch.zeros_like(d_learned_norm)
    d_learned_norm_aligned[:, star_idx] = d_learned_norm[:, learned_idx]
    dictionary_error_frobenius = torch.norm(d_learned_norm_aligned - d_star_norm, p='fro').item()
    
    # Code Recovery Error ||h_pred_aligned - h_star||_2
    h_pred_aligned = torch.zeros_like(h_pred)
    h_pred_aligned[:, star_idx] = h_pred[:, learned_idx]
    code_recovery_mse = torch.mean((h_pred_aligned - h_star) ** 2).item()
    code_recovery_l2_norm = torch.mean(torch.norm(h_pred_aligned - h_star, dim=1)).item()

    # Support Recovery Metrics (Precision, Recall, F1, Jaccard)
    pred_active = (h_pred_aligned > eps)
    star_active = (h_star > eps)
    tp = (pred_active & star_active).sum().float().item()
    fp = (pred_active & ~star_active).sum().float().item()
    fn = (~pred_active & star_active).sum().float().item()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1_score = 2 * precision * recall / (precision + recall + eps)
    # Jaccard Similarity (Intersection over Union)
    jaccard_similarity = tp / (tp + fp + fn + eps)

    return {
        "dictionary_error_cosine": dictionary_error_cosine,
        "dictionary_error_frobenius": dictionary_error_frobenius,
        "code_recovery_mse": code_recovery_mse,
        "code_recovery_l2": code_recovery_l2_norm,
        "support_precision": precision,
        "support_recall": recall,
        "support_f1_score": f1_score,
        "support_jaccard_similarity": jaccard_similarity,
    }
