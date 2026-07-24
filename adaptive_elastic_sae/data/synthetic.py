from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SpikedDataConfig:
    n_dim: int = 256  # Dimensionality of dense intermediate state
    d_dict: int = 1024  # Number of dictionary elements (latent dimensionality)
    k_sparse: int = (
        16  # Number of 'true' active features for ground truth (sparsity level)
    )
    rho: float = 0.0  # Correlation strength (coherence) of dictionary, in [0, 1)
    noise_std: float = 0.0  # Std dev of Gaussian noise added to observations
    allow_negative_codes: bool = False  # Whether to allow negative values in the sparse codes (default: non-negative)
    seed: int = 0
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"


class SpikedDataGenerator:
    """
    Generate synthetic sparse-coding data with controllable dictionary coherence.
    Teacher-student style setup where the teacher dictionary D is explictly constructed. For each sample,
    - The tacher generates a ground truth sparse code h_true with exactly k_sparse nonzeros.
    - The dense observation x is produced by multiplying h_true with D, optionally adding noise.
    - The student SAE is trained to reconstruct x and ideally recover the underlying sparse structure.
    High correlation, rho, is a way of introducing semantic co-occurence.
    """

    def __init__(
        self,
        n_dim: int = 256,
        d_dict: int = 1024,
        k_sparse: int = 16,
        rho: float = 0.0,
        noise_std: float = 0.0,
        allow_negative_codes: bool = False,
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        if not (0.0 <= rho < 1.0):
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if not (1 <= k_sparse <= d_dict):
            raise ValueError(
                f"k_sparse must be in [1, d_dict], got {k_sparse} for d_dict={d_dict}"
            )
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {noise_std}")

        self.n_dim = n_dim
        self.d_dict = d_dict
        self.k_sparse = k_sparse
        self.rho = float(rho)
        self.noise_std = float(noise_std)
        self.allow_negative_codes = allow_negative_codes
        self.dtype = dtype
        self.device = torch.device(device)

        # Local RNG avoids mutating global random state.
        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(seed)

        # Ground truth dictionary reused over all samples.
        self.D = self._generate_correlated_dictionary()

    @property
    def true_dictionary(self) -> torch.Tensor:
        """Return the ground truth dictionary D."""
        return self.D

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(
            *shape,
            generator=self._rng,
            device=self.device,
            dtype=self.dtype,
        )

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(
            *shape,
            generator=self._rng,
            device=self.device,
            dtype=self.dtype,
        )

    def _generate_correlated_dictionary(self) -> torch.Tensor:
        """Build D by blending independent directions with one shared spike direction."""
        eps = torch.tensor(1e-12, device=self.device, dtype=self.dtype)

        v_shared = self._randn(self.n_dim, 1)
        v_shared = v_shared / (v_shared.norm(dim=0, keepdim=True) + eps)

        u_ind = self._randn(self.n_dim, self.d_dict)
        u_ind = u_ind / (u_ind.norm(dim=0, keepdim=True) + eps)

        rho_t = torch.tensor(self.rho, device=self.device, dtype=self.dtype)
        d = torch.sqrt(1.0 - rho_t) * u_ind + torch.sqrt(rho_t) * v_shared
        d = d / (d.norm(dim=0, keepdim=True) + eps)
        return d

    def sample_sparse_codes(self, batch_size: int) -> torch.Tensor:
        """Sample sparse codes with exactly k active entries per sample."""
        h_true = torch.zeros(
            batch_size,
            self.d_dict,
            device=self.device,
            dtype=self.dtype,
        )

        # Top-k on random scores gives k unique active indices per row.
        scores = self._rand(batch_size, self.d_dict)
        active_idx = torch.topk(scores, k=self.k_sparse, dim=1, largest=True).indices

        magnitudes = 1.0 + 2.0 * self._rand(batch_size, self.k_sparse)
        if self.allow_negative_codes:
            signs = torch.where(self._rand(batch_size, self.k_sparse) < 0.5, -1.0, 1.0)
            magnitudes = magnitudes * signs

        h_true.scatter_(dim=1, index=active_idx, src=magnitudes)
        return h_true

    def generate_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense observations x and their latent sparse codes h_true."""
        h_true = self.sample_sparse_codes(batch_size)
        x = h_true @ self.D.T

        if self.noise_std > 0.0:
            x = x + self.noise_std * self._randn(batch_size, self.n_dim)

        return x, h_true

    @torch.no_grad()
    def dictionary_stats(self, samples: int = 100) -> dict[str, float]:
        """Diagnostics for global coherence and sampled active-block conditioning."""
        gram = self.D.T @ self.D
        off_diag = gram - torch.diag_embed(torch.diagonal(gram))

        active_min_eigs: list[float] = []
        active_cond_nums: list[float] = []
        irc_scores_uniform: list[float] = []
        irc_scores_exact: list[float] = []
        signal_ceilings: list[float] = []

        singular_count = 0
        eps = 1e-12

        for _ in range(samples):
            # Sample an active set A
            idx = torch.randperm(self.d_dict, generator=self._rng, device=self.device)[
                : self.k_sparse
            ]
            active_mask = torch.zeros(self.d_dict, dtype=torch.bool, device=self.device)
            active_mask[idx] = True

            d_a = self.D[:, active_mask]
            d_ac = self.D[:, ~active_mask]

            # Local Gram blocks
            g_aa = d_a.T @ d_a
            g_aca = d_ac.T @ d_a

            # Eigenvalue Analysis
            eigvals = torch.linalg.eigvalsh(g_aa)
            min_eig = eigvals.min()
            max_eig = eigvals.max()

            if min_eig <= eps:
                singular_count += 1

            min_eig_clamped = torch.clamp(min_eig, min=eps)
            active_min_eigs.append(min_eig_clamped.item())
            active_cond_nums.append((max_eig / min_eig_clamped).item())

            # IRC Calculation
            try:
                # G_AA * X = G_ACA^T  =>  X^T = G_ACA @ G_AA^-1
                sol = torch.linalg.lstsq(
                    g_aa + 1e-6 * torch.eye(self.k_sparse, device=self.device), g_aca.T
                ).solution

                # Uniform IRC (Worst-case)
                irc_uni = torch.linalg.norm(sol.T, ord=float("inf")).item()
                irc_scores_uniform.append(irc_uni)

                # Exact IRC (Assuming non-negative h)
                ones = torch.ones(
                    self.k_sparse, 1, device=self.device, dtype=self.dtype
                )
                exact_vec = sol.T @ ones
                irc_ext = torch.linalg.norm(exact_vec, ord=float("inf")).item()
                irc_scores_exact.append(irc_ext)
            except:  # noqa: E722
                irc_scores_uniform.append(-1.0)
                irc_scores_exact.append(-1.0)

            # Signal Ceiling: E[||D^T y||_inf]
            # y = D_A * h_A. We simulate a typical 'h_A' to find the max correlation.
            h_sample = 1.0 + 2.0 * torch.rand(
                self.k_sparse, device=self.device, dtype=self.dtype
            )
            y_sample = d_a @ h_sample
            signal_ceiling = torch.norm(self.D.T @ y_sample, p=float("inf")).item()
            signal_ceilings.append(signal_ceiling)

        # Aggregation
        def to_tensor(lst):
            return torch.tensor(lst, dtype=self.dtype, device=self.device)

        min_eigs_t = to_tensor(active_min_eigs)
        cond_nums_t = to_tensor(active_cond_nums)
        irc_uni_t = to_tensor(
            [i for i in irc_scores_uniform if i > 0]
        )  # Filter failures
        irc_ext_t = to_tensor([i for i in irc_scores_exact if i > 0])
        ceil_t = to_tensor(signal_ceilings)

        return {
            "mean_abs_offdiag": off_diag.abs().mean().item(),
            "max_abs_offdiag": off_diag.abs().max().item(),
            "expected_active_min_eig": min_eigs_t.mean().item(),
            "expected_active_min_eig_std": min_eigs_t.std(unbiased=False).item(),
            "expected_active_condition_number": cond_nums_t.mean().item(),
            "expected_active_condition_number_std": cond_nums_t.std(
                unbiased=False
            ).item(),
            "active_condition_number_p95": torch.quantile(cond_nums_t, 0.95).item(),
            "expected_irc_uniform": irc_uni_t.mean().item(),
            "expected_irc_exact": irc_ext_t.mean().item(),
            "expected_signal_ceiling_l_inf": ceil_t.mean().item(),
            "singular_active_block_pct": 100.0 * singular_count / max(1, samples),
        }
