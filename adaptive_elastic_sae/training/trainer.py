from __future__ import annotations

from dataclasses import asdict
from typing import Any
from logging import getLogger

import torch
from torch.optim import Adam

from adaptive_elastic_sae.saes.base import BaseSAE
from adaptive_elastic_sae.training.metrics import (
    activation_effective_sample_size,
    active_gram_spectrum,
    compute_cross_leverage,
    compute_synthetic_recovery_metrics,
    dead_neuron_recovery_rate,
    dead_neurons_pct,
    explained_variance,
    feature_shrinkage_ratio,
    dictionary_coherence_summary,
    interaction_leakage_frobenius_approx,
    l0_active_features,
    l0_vs_l1_ratio,
    weight_bimodality_ratio,
)
from adaptive_elastic_sae.training.gpu_metrics import (
    flops_progress_metrics,
    measure_training_step_flops,
)
from adaptive_elastic_sae.training.trainer_utils import (
    BatchProvider,
    SyntheticBatchProvider,
    TrainerConfig,
)

logger = getLogger(__name__)

class SAETrainer:
    """
    Trainer for synthetic SAE experiments.
    Does not support using an LLM-based model or validation sets.
    """

    def __init__(
        self,
        model: BaseSAE,
        config: TrainerConfig,
        batch_provider: BatchProvider,
        generator: Any | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.batch_provider = batch_provider
        self.generator = generator
        self.device = torch.device(config.device)
        self.model.to(self.device)

        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

        # Track max activations for dead neuron metrics
        self.max_activations = torch.zeros(
            model.d_dict,
            device=self.device,
            dtype=config.dtype,
        )
        self.max_activations_history = []

    @classmethod
    def from_synthetic(
        cls,
        model: BaseSAE,
        config: TrainerConfig,
        generator: Any,
    ) -> SAETrainer:
        """Construct trainer using synthetic generator adapter."""
        return cls(
            model=model,
            config=config,
            batch_provider=SyntheticBatchProvider(generator),
            generator=generator,
        )

    def train(
        self,
        use_wandb: bool = False,
        run_name: str = "sae-train",
        wandb_config: dict | None = None,
        run_metadata: dict[str, Any] | None = None,
        run_tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run full training, return aggregate metrics."""
        if wandb_config is None:
            wandb_config = {}
        if run_metadata is None:
            run_metadata = {}
        if run_tags is None:
            run_tags = []

        # Attach experiment metadata per run.
        run_config = asdict(self.config)
        run_config["device"] = str(run_config["device"])
        run_config["dtype"] = str(run_config["dtype"])
        # Explicitly keep these visible for grouping in W&B.
        run_config["seed"] = self.config.seed
        run_config["model_type"] = self.config.model_type
        run_config.update(run_metadata)

        measured_flops_per_step: float | None = None

        if use_wandb:
            try:
                import wandb

                base_tags = wandb_config.get("tags", [])
                tags = [*base_tags, *run_tags]

                wandb.init(
                    project=wandb_config.get("project", "adaptive-elastic-sae"),
                    entity=wandb_config.get("entity"),
                    name=run_name,
                    tags=tags,
                    config=run_config,
                    reinit="finish_previous",
                )
            except ImportError:
                use_wandb = False

        self.model.train()
        metrics_history = []

        for step in range(self.config.num_steps):
            # Generate batch
            batch = self.batch_provider.next_batch(
                batch_size=self.config.batch_size,
                device=self.device,
                dtype=self.config.dtype,
            )
            x = batch["x"]
            h_star = batch.get("h_true")  # Ground truth sparse code if available

            # Real FLOPs profile for one full training step (forward+loss+backward).
            if step == 0:
                flops_metrics = measure_training_step_flops(
                    model=self.model,
                    optimizer=self.optimizer,
                    x=x,
                    warmup_steps=self.config.warmup_steps,
                )
                measured_flops_per_step = flops_metrics.get("measured_flops_per_step")
                run_config.update(flops_metrics)
                if use_wandb:
                    wandb.config.update(flops_metrics, allow_val_change=True)

            # Forward
            x_hat, h = self.model.forward(x)

            # Loss (support models that return either Tensor or (Tensor, dict)).
            loss_components: dict[str, float] = {}
            try:
                loss_out = self.model.compute_loss(
                    x,
                    x_hat,
                    h,
                    warmup_steps=self.config.warmup_steps,
                )
            except TypeError:
                loss_out = self.model.compute_loss(x, x_hat, h)

            if isinstance(loss_out, tuple):
                loss, loss_components = loss_out
            else:
                loss = loss_out
                loss_components = {"loss_total": loss.item()}

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Normalize decoder
            self.model.normalize_decoder()

            # Track max activations for dead neuron calc
            with torch.no_grad():
                self.max_activations = torch.max(
                    self.max_activations,
                    h.detach().abs().max(dim=0).values,
                )

            # Periodic logging
            if (step + 1) % self.config.log_interval == 0:
                metrics = self._compute_metrics(x, x_hat, h, h_star=h_star)
                metrics["step"] = step + 1
                metrics["loss"] = loss.item()
                metrics.update(
                    flops_progress_metrics(
                        measured_flops_per_step=measured_flops_per_step,
                        step=step + 1,
                    )
                )
                metrics.update(loss_components)
                metrics_history.append(metrics)

                if use_wandb:
                    wandb.log(metrics)

                logger.info(
                    f"Step {step + 1}/{self.config.num_steps} | "
                    f"Loss: {loss.item():.6f} | "
                    f"L0: {metrics['l0_active_features']:.2f} | "
                    f"Dead: {metrics['dead_neurons_pct']:.1f}%"
                )

            # Reset max activations periodically
            if (step + 1) % self.config.max_activations_window == 0:
                self.max_activations_history.append(self.max_activations.clone())

                if len(self.max_activations_history) >= 2:
                    recovery_rate = dead_neuron_recovery_rate(
                        self.max_activations_history[-2],
                        self.max_activations_history[-1],
                        eps=1e-12,
                    )
                    if use_wandb:
                        wandb.log(
                            {
                                "dead_neuron_recovery_rate": recovery_rate,
                                "dead_neuron_recovery_pct": 100.0 * recovery_rate,
                            }
                        )

                self.max_activations.zero_()

        if use_wandb:
            wandb.finish()

        return {
            "metrics_history": metrics_history,
            "model_state": self.model.state_dict(),
        }

    @torch.no_grad()
    def _compute_metrics(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        h: torch.Tensor,
        h_star: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Compute all diagnostic metrics for the current batch."""
        metrics = {}

        # Dead neurons & Sparsity
        metrics["dead_neurons_pct"] = dead_neurons_pct(self.max_activations, eps=1e-12)
        metrics["l0_active_features"] = l0_active_features(h, eps=1e-12)
        metrics["l0_vs_l1_ratio"] = l0_vs_l1_ratio(h, eps=1e-12)

        # Reconstruction & Shrinkage
        metrics["explained_variance"] = explained_variance(x, x_hat, eps=1e-12)
        metrics["feature_shrinkage_ratio"] = feature_shrinkage_ratio(x, x_hat, eps=1e-12)

        # Adaptive-weight metrics
        if hasattr(self.model, "ema_abs_activations"):
            metrics["activation_effective_sample_size"] = activation_effective_sample_size(
                self.model.ema_abs_activations, eps=1e-12
            )
        if hasattr(self.model, "get_adaptive_weights"):
            try:
                weights = self.model.get_adaptive_weights()
                metrics["weight_bimodality_ratio"] = weight_bimodality_ratio(
                    weights, delta_sig=0.1, delta_noise=5.0, eps=1e-12
                )
            except Exception:
                pass

        # Ground Truth Recovery Metrics (D*, h*)
        if self.generator is not None and hasattr(self.generator, "true_dictionary") and h_star is not None:
            try:
                d_star = self.generator.true_dictionary
                d_learned = self.model.decoder.weight.data
                rec_metrics = compute_synthetic_recovery_metrics(
                    d_learned=d_learned,
                    d_star=d_star,
                    h_pred=h,
                    h_star=h_star,
                )
                metrics.update(rec_metrics)
            except Exception as e:
                logger.warning(f"Failed to compute synthetic recovery metrics: {e}")

        # Geometric & Conditioning
        busiest_idx = (h > 1e-12).sum(dim=1).argmax()
        local_active_mask = h[busiest_idx] > 1e-12
        if local_active_mask.any():
            decoder = self.model.decoder.weight.data
            metrics["interaction_leakage_frobenius"] = interaction_leakage_frobenius_approx(
                decoder, local_active_mask
            )
            metrics.update(compute_cross_leverage(decoder, local_active_mask, k_top=5))
            metrics.update(active_gram_spectrum(decoder, local_active_mask))
            metrics.update(dictionary_coherence_summary(decoder, eps=1e-12))

        return metrics
