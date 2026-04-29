"""TripleModalTrainer: VAE + Discriminator co-training with OT alignment.

Coordinates VAE pre-warm (20 epochs, D-03) and adversarial alignment post-warm.
Implements GLUE-style confusion training and R1 gradient penalty for stability.
Optimal Transport (POT sinkhorn) enforces epsilon >= 0.1 entropy regularization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    import ot
except ImportError:
    ot = None  # type: ignore

from ..models.discriminator import (
    ModalityDiscriminator,
    SCRNA_MODALITY,
    ST_MODALITY,
    BULK_MODALITY,
    adversarial_loss_for_vae,
    adversarial_loss_for_disc,
    r1_gradient_penalty,
)
from ..scaffold.spatial_scaffold import SpatialAwareLoss, SpatialScaffold
from ..deconv.bulk_prior import BulkPriorConfig, compute_bulk_prior_lambda, compute_bulk_prior_loss


class TripleModalTrainer:
    """Co-trainer for TripleModalVAE and ModalityDiscriminator.

    Training proceeds in two phases:
      1. Pre-warm (D-03, default 20 epochs): VAE trains reconstruction + KL + graph
         losses only. Discriminator trains on real samples only (no confusion).
      2. Adversarial (epoch >= pre_warm_epochs): VAE also receives adversarial
         GLUE-style loss. Discriminator receives confusion loss + R1 penalty.

    OT alignment (sinkhorn_alignment) is applied as an auxiliary loss between
    u_sc <-> u_st and u_sc <-> u_bulk throughout both phases.
    """

    def __init__(
        self,
        vae: nn.Module,
        discriminator: ModalityDiscriminator,
        lr: float = 1e-3,
        pre_warm_epochs: int = 20,
        r1_weight: float = 1.0,
        ot_epsilon: float = 0.5,
        ot_loss_threshold: float = 10.0,
        spatial_scaffold=None,
        bulk_prior_config: BulkPriorConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.vae = vae
        self.discriminator = discriminator
        self.pre_warm_epochs = pre_warm_epochs
        self.r1_weight = r1_weight
        self.ot_epsilon = max(0.5, ot_epsilon)  # Enforce epsilon >= 0.5 for numerical stability
        self.ot_loss_threshold = ot_loss_threshold

        # Default SpatialScaffold with latent_dim from VAE if not provided
        if spatial_scaffold is None:
            latent_dim = getattr(vae, 'latent_dim', 128)
            self.spatial_scaffold = SpatialScaffold(latent_dim=latent_dim, n_neighbors=6)
        else:
            self.spatial_scaffold = spatial_scaffold

        # Bulk prior configuration (D-11: warm-up schedule 0.01->0.1 at epoch 20-40)
        self.bulk_prior_config = bulk_prior_config or BulkPriorConfig()
        self.bulk_proportions: torch.Tensor | None = None  # Set via set_bulk_proportions()

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Move models to device
        self.vae = self.vae.to(self.device)
        self.discriminator = self.discriminator.to(self.device)
        if self.spatial_scaffold is not None:
            self.spatial_scaffold = self.spatial_scaffold.to(self.device)

        # Separate Adam optimizers
        self.opt_vae = torch.optim.Adam(self.vae.parameters(), lr=lr)
        self.opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # Bulk prior configuration
        self.bulk_prior_config = bulk_prior_config or BulkPriorConfig()
        self.bulk_proportions: Tensor | None = None  # Legacy: sample-level proportions

        # NEW: condition-level bulk proportions from preprocessing
        self.bulk_condition_proportions: Tensor | None = None
        self.bulk_condition_names: list[str] | None = None

        # NEW: scRNA/ST metadata for condition aggregation
        self.scrna_cell_type_labels: np.ndarray | None = None
        self.st_condition_labels: np.ndarray | None = None

        # NEW: epoch cached condition prior state
        self.condition_prior_state: Any = None

        # Training history
        self.history: dict[str, list[float]] = {
            "vae_loss": [],
            "recon_loss": [],
            "kl_loss": [],
            "graph_recon_loss": [],
            "adversarial_loss": [],
            "ot_loss": [],
            "ot_loss_high": [],
            "bulk_prior_loss": [],
            "bulk_prior_lambda": [],
            "bulk_prior_valid": [],
            "disc_loss": [],
            "disc_real_loss": [],
            "disc_confusion_loss": [],
            "disc_r1_penalty": [],
            "spatial_loss": [],
            "spatial_weight": [],
        }

    def set_spatial_graph(self, adj_matrix) -> None:
        """Set the spatial k-NN adjacency matrix on the scaffold.

        Args:
            adj_matrix: scipy.sparse.csr_matrix from squidpy spatial_neighbors
        """
        self.spatial_scaffold.set_spatial_graph(adj_matrix)

    def set_bulk_proportions(self, bulk_props: Tensor) -> None:
        """Set bulk-derived cell type proportions from ssGSEA preprocessing.

        D-10: Bulk preprocessing via ssGSEA produces cell type proportions.

        DEPRECATED: Use set_bulk_condition_proportions() for condition-level prior.

        Args:
            bulk_props: (n_clusters, n_cell_types) tensor from preprocess_bulk_ssgsea
        """
        self.bulk_proportions = bulk_props.to(self.device)

    def set_bulk_condition_proportions(
        self,
        condition_proportions: Tensor,
        condition_names: list[str],
    ) -> None:
        """Set condition-level bulk proportions from preprocessing.

        This is the recommended way to set bulk prior for triple-modal integration.

        Args:
            condition_proportions: (n_conditions, n_cell_types) aggregated proportions
            condition_names: List of canonical condition names in order
        """
        self.bulk_condition_proportions = condition_proportions.to(self.device)
        self.bulk_condition_names = condition_names

    def set_metadata_for_condition_prior(
        self,
        scrna_cell_type_labels: np.ndarray,
        st_condition_labels: np.ndarray,
    ) -> None:
        """Set scRNA and ST metadata for condition aggregation.

        Args:
            scrna_cell_type_labels: (n_cells,) canonical cell type labels
            st_condition_labels: (n_spots,) canonical condition labels
        """
        self.scrna_cell_type_labels = scrna_cell_type_labels
        self.st_condition_labels = st_condition_labels

    def compute_spatial_weight(self, epoch: int) -> float:
        """Compute spatial loss weight based on warm-up schedule (D-07).

        Epoch 0..pre_warm_epochs (0-20): weight = 0.0
        Epoch 20..30: linear ramp weight = (epoch - 20) / 10 * 0.1
        Epoch >= 30: weight = 0.1 (max)

        Returns:
            float: spatial_loss weight for current epoch
        """
        if epoch < self.pre_warm_epochs:  # 0-20
            return 0.0
        elif epoch < 30:  # 20-30 ramp
            return (epoch - self.pre_warm_epochs) / 10 * 0.1
        else:  # 30+
            return 0.1

    def bulk_prior_step(
        self,
        cluster_proportions: Tensor,
        epoch: int,
    ):
        """Compute bulk prior loss at cluster level.

        D-11: Bulk prior lambda warm-up schedule:
        - Epoch 0-20: lambda = 0.01 (minimal constraint during VAE burn-in)
        - Epoch 20-40: linear ramp 0.01 -> 0.1
        - Epoch >= 40: lambda = 0.1 (full prior strength)

        Anti-pattern enforcement: Bulk prior at CLUSTER LEVEL only (never per-spot).

        Args:
            cluster_proportions: (n_clusters, n_cell_types) from OT deconvolution
            epoch: Current training epoch

        Returns:
            BulkPriorOutput with kl_loss, kl_per_cluster, bulk_proportions, current_lambda
        """
        if self.bulk_proportions is None:
            # Return zero loss if bulk proportions not set
            return None

        # Compute lambda from warm-up schedule
        current_lambda = compute_bulk_prior_lambda(epoch, self.bulk_prior_config)

        # Compute bulk prior loss at cluster level
        bulk_prior_output = compute_bulk_prior_loss(
            cluster_proportions,
            self.bulk_proportions,
            current_lambda,
        )

        return bulk_prior_output

    def is_prewarm(self, epoch: int) -> bool:
        """Check if currently in pre-warm phase (not yet at adversarial)."""
        return epoch < self.pre_warm_epochs

    def _check_gradient_nan(self) -> bool:
        """Check if any VAE parameter gradient is NaN.

        P0-02: Gradient NaN detection prevents corrupted optimizer state.
        NaN gradients can pass through clip_grad_norm_ and permanently damage
        Adam momentum buffers, causing training to never recover.

        Returns
        -------
        bool
            True if NaN gradient detected in any VAE parameter, False otherwise.
        """
        for name, param in self.vae.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    logger.error(f"NaN gradient detected in VAE parameter: {name}")
                    return True
        return False

    @staticmethod
    def _sinkhorn_torch(
        C: Tensor,
        a: Tensor,
        b: Tensor,
        epsilon: float,
        n_iters: int = 100,
    ) -> Tensor:
        """Pure PyTorch Sinkhorn algorithm that keeps gradients through C."""
        K = torch.exp(-C / epsilon)
        K = torch.clamp(K, min=1e-20)

        u = torch.ones_like(a)
        v = torch.ones_like(b)

        for _ in range(n_iters):
            v = b / (K.T @ u + 1e-8)
            u = a / (K @ v + 1e-8)

        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        if not torch.isfinite(P).all():
            # Fallback that still depends on C, preserving gradient path.
            return torch.softmax(-C, dim=1)
        return P

    def sinkhorn_alignment(
        self,
        u_source: Tensor,
        u_target: Tensor,
        epsilon: float | None = None,
        n_iters: int = 100,
        use_batch_ot: bool = True,
    ) -> Tensor:
        """Optimal Transport alignment via differentiable Sinkhorn."""
        epsilon = max(0.5, epsilon or self.ot_epsilon)

        if use_batch_ot and u_source.shape[0] > 1 and u_target.shape[0] > 1:
            n_source = min(u_source.shape[0], 500)
            n_target = min(u_target.shape[0], 500)

            if u_source.shape[0] > n_source:
                idx_source = torch.randperm(u_source.shape[0], device=u_source.device)[:n_source]
                u_source_sample = u_source[idx_source]
            else:
                u_source_sample = u_source

            if u_target.shape[0] > n_target:
                idx_target = torch.randperm(u_target.shape[0], device=u_target.device)[:n_target]
                u_target_sample = u_target[idx_target]
            else:
                u_target_sample = u_target

            C = torch.cdist(u_source_sample, u_target_sample, p=2).pow(2)
            n = C.shape[0]
            m = C.shape[1]
        else:
            source_mean = u_source.mean(dim=0, keepdim=True)
            target_mean = u_target.mean(dim=0, keepdim=True)
            C = torch.cdist(source_mean, target_mean, p=2).pow(2)
            n = 1
            m = 1

        if C.max() <= 0:
            return torch.tensor(0.0, device=C.device, requires_grad=True)

        a = torch.ones(n, device=C.device, dtype=C.dtype) / n
        b = torch.ones(m, device=C.device, dtype=C.dtype) / m

        try:
            P = self._sinkhorn_torch(C, a, b, epsilon, n_iters)
            ot_loss = (P * C).sum()
            if torch.isnan(ot_loss) or torch.isinf(ot_loss):
                return torch.tensor(0.0, device=C.device, requires_grad=True)
            return ot_loss
        except (RuntimeError, ValueError):
            return torch.tensor(0.0, device=C.device, requires_grad=True)

    def compute_online_bulk_prior(
        self,
        u_sc: Tensor,
        u_st: Tensor,
        epoch: int,
        batch: dict[str, Any],
    ) -> Tensor:
        """Compute differentiable batch-level bulk prior from current embeddings."""
        if self.bulk_condition_proportions is None:
            return torch.tensor(0.0, device=u_sc.device)

        cell_type_onehot = batch.get("cell_type_onehot")
        st_condition_idx = batch.get("st_condition_idx")
        if cell_type_onehot is None or st_condition_idx is None:
            return torch.tensor(0.0, device=u_sc.device)

        cell_type_onehot = cell_type_onehot.to(u_sc.device)
        st_condition_idx = st_condition_idx.to(u_st.device).long()

        n_sc = u_sc.shape[0]
        n_st = u_st.shape[0]
        max_sc = min(n_sc, 200)
        max_st = min(n_st, 200)

        u_sc_sub = u_sc[:max_sc]
        u_st_sub = u_st[:max_st]
        ct_sub = cell_type_onehot[:max_sc]
        cond_sub = st_condition_idx[:max_st]

        if u_sc_sub.numel() == 0 or u_st_sub.numel() == 0 or ct_sub.numel() == 0:
            return torch.tensor(0.0, device=u_sc.device)

        sim = torch.nn.functional.cosine_similarity(
            u_st_sub.unsqueeze(1),
            u_sc_sub.unsqueeze(0),
            dim=-1,
        )
        temperature = getattr(self.bulk_prior_config, "online_prior_temperature", 0.1)
        weights = torch.nn.functional.softmax(sim / temperature, dim=1)
        spot_props = weights @ ct_sub
        spot_props = spot_props / (spot_props.sum(dim=1, keepdim=True) + 1e-8)

        unique_conditions = cond_sub.unique()
        if len(unique_conditions) == 0:
            return torch.tensor(0.0, device=u_sc.device)

        pred_condition_props = torch.stack(
            [spot_props[cond_sub == cond].mean(dim=0) for cond in unique_conditions]
        )
        pred_condition_props = pred_condition_props / (
            pred_condition_props.sum(dim=1, keepdim=True) + 1e-8
        )

        bulk_props_all = self.bulk_condition_proportions.to(u_sc.device)
        n_bulk = bulk_props_all.shape[0]
        bulk_idx = unique_conditions.clamp(max=n_bulk - 1)
        bulk_props = bulk_props_all[bulk_idx]

        from ..deconv.bulk_prior import compute_condition_level_kl

        lambda_ = compute_bulk_prior_lambda(epoch, self.bulk_prior_config)
        kl_loss, _ = compute_condition_level_kl(pred_condition_props, bulk_props, lambda_)

        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            return torch.tensor(0.0, device=u_sc.device)
        return kl_loss

    def compute_vae_loss(
        self,
        u_sc: Tensor,
        u_st: Tensor,
        u_bulk: Tensor,
        batch: dict[str, Any],
        epoch: int,
        cluster_proportions: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute VAE loss (base + optional adversarial after pre-warm).

        Always includes:
          - recon_loss: NB reconstruction for scRNA + ST (Bulk excluded per anti-pattern)
          - kl_loss: KL divergence for scRNA + ST
          - graph_recon_loss: GraphDecoder BCE
          - ot_loss: OT alignment between u_sc <-> u_st and u_sc <-> u_bulk

        Post pre-warm (D-03):
          - Adds adversarial GLUE-style loss for each modality

        Bulk prior (D-11):
          - Cluster-level KL divergence between deconv proportions and Bulk ssGSEA
          - Lambda warm-up: 0.01 at epoch 0-20, ramp to 0.1 at epoch 40

        Args:
            u_sc: VAE scRNA embedding
            u_st: VAE ST embedding
            u_bulk: VAE Bulk embedding
            batch: Batch dict with recon_loss, kl_loss, graph_recon_loss
            epoch: Current training epoch
            cluster_proportions: (n_clusters, n_cell_types) optional, from OT deconvolution

        Returns:
            total_loss: Scalar VAE loss
            loss_components: Dict of individual loss terms for logging
        """
        # Base losses (always present)
        recon_loss = batch["recon_loss"]
        kl_loss = batch["kl_loss"]
        graph_recon_loss = batch["graph_recon_loss"]

        # OT alignment losses (always present, epsilon >= 0.1 enforced)
        # Note: ot_sc_bulk removed - bulk alignment not meaningful for unpaired samples
        ot_sc_st = self.sinkhorn_alignment(u_sc, u_st)
        ot_loss = ot_sc_st

        if torch.isfinite(ot_loss) and ot_loss.item() > self.ot_loss_threshold:
            logger.warning(
                f"ot_loss soft-cap exceeded: {ot_loss.item():.4f} "
                f"(threshold={self.ot_loss_threshold}, epoch={epoch})"
            )

        # Total base loss
        base_loss = recon_loss + kl_loss + graph_recon_loss + ot_loss

        loss_components = {
            "recon_loss": recon_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "graph_recon_loss": graph_recon_loss.detach(),
            "ot_loss": ot_loss.detach(),
            "ot_loss_high": torch.tensor(
                1.0 if (torch.isfinite(ot_loss) and ot_loss.item() > self.ot_loss_threshold) else 0.0,
                device=self.device,
            ),
            "adversarial_loss": torch.tensor(0.0, device=self.device),
            "bulk_prior_loss": torch.tensor(0.0, device=self.device),
            "bulk_prior_lambda": torch.tensor(self.bulk_prior_config.lambda_start, device=self.device),
            "bulk_prior_valid": torch.tensor(0.0, device=self.device),
        }

        # Adversarial loss after pre-warm (D-03)
        if epoch >= self.pre_warm_epochs:
            # GLUE-style confusion: each modality embedding is classified
            # as each of the 3 modalities (confusion training)
            adv_total = torch.tensor(0.0, device=self.device)
            for modality_idx in [SCRNA_MODALITY, ST_MODALITY, BULK_MODALITY]:
                adv = adversarial_loss_for_vae(
                    u_sc, u_st, u_bulk, self.discriminator, modality_idx
                )
                adv_total = adv_total + adv

            base_loss = base_loss + adv_total
            loss_components["adversarial_loss"] = adv_total.detach()

        # Spatial smoothness loss with warm-up schedule (D-07)
        # spatial_weight = 0 during pre-warm, ramps 20-30, stays at 0.1 after
        spatial_weight = self.compute_spatial_weight(epoch)
        if self.spatial_scaffold is not None and spatial_weight > 0:
            # BF-02: Pass batch_indices for minibatch handling
            # If batch has 'st_indices', use them; otherwise assume full dataset
            # WR-02 NOTE: With random shuffled batches of 128 spots from 454K,
            # intra-batch edges are extremely sparse (~0.2 expected). The
            # per-epoch apply_epoch_spatial_loss() compensates by computing
            # spatial loss on the full dataset once per epoch.
            st_indices = batch.get('st_indices', None)
            spatial_loss = SpatialAwareLoss(u_st, self.spatial_scaffold, batch_indices=st_indices)
            base_loss = base_loss + spatial_loss * spatial_weight
            loss_components["spatial_loss"] = spatial_loss.detach()
            loss_components["spatial_weight"] = torch.tensor(spatial_weight, device=self.device)
        else:
            loss_components["spatial_loss"] = torch.tensor(0.0, device=self.device)
            loss_components["spatial_weight"] = torch.tensor(spatial_weight, device=self.device)

        # Bulk prior loss with warm-up schedule (D-11)
        # P2 fix: use differentiable online prior from current batch embeddings.
        online_bulk_loss = self.compute_online_bulk_prior(u_sc, u_st, epoch, batch)
        if torch.isfinite(online_bulk_loss) and online_bulk_loss.item() > 0:
            base_loss = base_loss + online_bulk_loss
            loss_components["bulk_prior_loss"] = online_bulk_loss.detach()
            loss_components["bulk_prior_lambda"] = torch.tensor(
                compute_bulk_prior_lambda(epoch, self.bulk_prior_config),
                device=self.device,
            )
            loss_components["bulk_prior_valid"] = torch.tensor(1.0, device=self.device)

        # Keep epoch-level condition prior state only as diagnostics (no loss add).
        if self.condition_prior_state is not None and epoch % 10 == 0:
            state = self.condition_prior_state
            logger.debug(
                f"Epoch {epoch}: Epoch-level prior diagnostic "
                f"conditions={state.condition_names}, ot_valid={state.ot_valid}"
            )

        return base_loss, loss_components

    def compute_discriminator_loss(
        self,
        u_sc: Tensor,
        u_st: Tensor,
        u_bulk: Tensor,
        epoch: int,
        n_sc: int | None = None,  # BF-04: sample counts for weighting
        n_st: int | None = None,
        n_bulk: int | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        """Compute discriminator loss.

        Pre-warm (epoch < pre_warm_epochs):
          - Train on real samples: u_sc→0, u_st→1, u_bulk→2

        Post pre-warm (epoch >= pre_warm_epochs):
          - Real sample loss (same as pre-warm)
          - Confusion loss: all three embeddings → each modality label
          - R1 gradient penalty on real samples

        BF-04: Modality weighting prevents Bulk gradient drowning.
          - Weight inversely proportional to sample count
          - Bulk (10 samples) gets much higher weight than scRNA (50K+)

        Args:
            u_sc: VAE scRNA embedding (should be detached for confusion part)
            u_st: VAE ST embedding
            u_bulk: VAE Bulk embedding
            epoch: Current training epoch
            n_sc: Number of scRNA samples (for BF-04 weighting)
            n_st: Number of ST samples (for BF-04 weighting)
            n_bulk: Number of Bulk samples (for BF-04 weighting)

        Returns:
            total_disc_loss: Scalar discriminator loss
            disc_components: Dict of discriminator loss components
        """
        disc_components = {
            "disc_real_loss": torch.tensor(0.0, device=self.device),
            "disc_confusion_loss": torch.tensor(0.0, device=self.device),
            "disc_r1_penalty": torch.tensor(0.0, device=self.device),
        }

        # BF-04: Compute modality weights inversely proportional to sample count
        # This prevents Bulk (10 samples) being drowned by scRNA (50K+ samples)
        # Safeguard: ensure all values are positive to avoid division by zero
        if n_sc is not None and n_st is not None and n_bulk is not None:
            if n_sc <= 0 or n_st <= 0 or n_bulk <= 0:
                # Use equal weights if any count is invalid
                modality_weights = torch.tensor([1.0/3, 1.0/3, 1.0/3], device=self.device)
            else:
                # Inverse weights: smaller modality gets higher weight
                inv_sc = 1.0 / n_sc
                inv_st = 1.0 / n_st
                inv_bulk = 1.0 / n_bulk

                # Normalize so weights sum to 1
                total_inv = inv_sc + inv_st + inv_bulk
                weight_sc = inv_sc / total_inv
                weight_st = inv_st / total_inv
                weight_bulk = inv_bulk / total_inv

                modality_weights = torch.tensor(
                    [weight_sc, weight_st, weight_bulk],
                    device=self.device
                )
        else:
            # Default: equal weights if sample counts not provided
            modality_weights = torch.tensor([1.0/3, 1.0/3, 1.0/3], device=self.device)

        disc_components["modality_weights"] = modality_weights.detach()

        # Real sample loss (always present)
        # Compute logits for each modality using mean embeddings
        logits_sc = self.discriminator(u_sc, u_st, u_bulk)  # Uses mean pooling internally

        # For real samples, we need separate forward passes to get per-modality logits
        # The discriminator's forward mean-pools, so we pass each modality separately
        with torch.no_grad():
            u_sc_mean = u_sc.mean(dim=0, keepdim=True)
            u_st_mean = u_st.mean(dim=0, keepdim=True)
            u_bulk_mean = u_bulk.mean(dim=0, keepdim=True)

        logits_real_sc = self.discriminator(u_sc, u_sc, u_bulk)  # Hack: use sc as proxy
        # Actually for real samples we should do:
        # Pass each modality separately to get its logit
        # But since discriminator uses mean pooling, we need to compute per-modality

        # Real loss: cross-entropy for correct modality labels
        real_loss = torch.tensor(0.0, device=self.device)

        # Pass each modality through separately (they're independent after mean pool)
        # Re-initialize forward to get proper per-modality discrimination
        # Actually the discriminator takes all 3 and outputs 3 logits based on concat
        # For real training, we want the discriminator to correctly classify
        # which modality each embedding belongs to

        # WR-01 FIX: Use mean embeddings instead of zeros for non-active modalities
        # This prevents synthetic embeddings that don't match learned latent space distribution
        with torch.no_grad():
            u_sc_mean = u_sc.mean(dim=0, keepdim=True).expand_as(u_sc)
            u_st_mean = u_st.mean(dim=0, keepdim=True).expand_as(u_st)
            u_bulk_mean = u_bulk.mean(dim=0, keepdim=True).expand_as(u_bulk)

        # Pass real scRNA through - should predict scRNA (use mean embeddings for inactive)
        logits_sc = self.discriminator(
            u_sc.detach(), u_st_mean.detach(), u_bulk_mean.detach()
        )
        real_loss_sc = F.cross_entropy(logits_sc, torch.tensor([SCRNA_MODALITY], device=self.device))

        # Pass real ST through - should predict ST
        logits_st = self.discriminator(
            u_sc_mean.detach(), u_st.detach(), u_bulk_mean.detach()
        )
        real_loss_st = F.cross_entropy(logits_st, torch.tensor([ST_MODALITY], device=self.device))

        # Pass real Bulk through - should predict Bulk
        logits_bulk = self.discriminator(
            u_sc_mean.detach(), u_st_mean.detach(), u_bulk.detach()
        )
        real_loss_bulk = F.cross_entropy(logits_bulk, torch.tensor([BULK_MODALITY], device=self.device))

        # BF-04: Apply modality weights to losses
        # Bulk (small count) gets higher weight to prevent gradient drowning
        weighted_real_loss = (
            modality_weights[0] * real_loss_sc +
            modality_weights[1] * real_loss_st +
            modality_weights[2] * real_loss_bulk
        )

        real_loss = weighted_real_loss
        disc_components["disc_real_loss"] = real_loss.detach()

        total_disc_loss = real_loss

        # Confusion + R1 penalty after pre-warm
        if epoch >= self.pre_warm_epochs:
            # Confusion loss: all three embeddings classified as each modality
            confusion_loss = torch.tensor(0.0, device=self.device)
            for modality_idx in [SCRNA_MODALITY, ST_MODALITY, BULK_MODALITY]:
                # Each embedding should be classified as the target modality
                # scRNA embedding → modality_idx
                adv_sc = adversarial_loss_for_disc(
                    u_sc, u_st, u_bulk, self.discriminator, modality_idx
                )
                confusion_loss = confusion_loss + adv_sc

            disc_components["disc_confusion_loss"] = confusion_loss.detach()
            total_disc_loss = total_disc_loss + confusion_loss

            # R1 gradient penalty on real samples
            r1_penalty = r1_gradient_penalty(
                self.discriminator,
                u_sc.detach(),
                u_st.detach(),
                u_bulk.detach(),
                reg_weight=self.r1_weight,
            )
            disc_components["disc_r1_penalty"] = r1_penalty.detach()
            total_disc_loss = total_disc_loss + r1_penalty

        return total_disc_loss, disc_components

    def train_step(
        self,
        batch: dict[str, Any],
        epoch: int,
    ) -> dict[str, float]:
        """Single training step: VAE update then discriminator update.

        Args:
            batch: Dict with x_sc, x_st, x_bulk, guidance_data, recon_loss, kl_loss, graph_recon_loss
            epoch: Current epoch number

        Returns:
            losses: Dict of scalar loss values for logging
        """
        self.vae.train()
        self.discriminator.train()

        # Move batch to device
        x_sc = batch["x_sc"].to(self.device)
        x_st = batch["x_st"].to(self.device)
        x_bulk = batch["x_bulk"].to(self.device)
        guidance_data = batch["guidance_data"]
        if hasattr(guidance_data, "to"):
            guidance_data = guidance_data.to(self.device)

        # VAE forward pass
        vae_output = self.vae(x_sc, x_st, x_bulk, guidance_data)
        u_sc = vae_output["u_sc"]
        u_st = vae_output["u_st"]
        u_bulk = vae_output["u_bulk"]

        # TR-02 / VAE-05: Empty batch early return
        # If VAE returned empty embeddings (zero-loss guard activated), skip
        # loss computation and backward pass to avoid grad_fn errors.
        if u_sc.shape[0] == 0 or u_st.shape[0] == 0:
            return {
                "vae_loss": 0.0,
                "recon_loss": 0.0,
                "kl_loss": 0.0,
                "graph_recon_loss": 0.0,
                "ot_loss": 0.0,
                "ot_loss_high": 0.0,
                "adversarial_loss": 0.0,
                "spatial_loss": 0.0,
                "spatial_weight": 0.0,
                "bulk_prior_loss": 0.0,
                "bulk_prior_lambda": 0.0,
                "disc_loss": 0.0,
                "disc_real_loss": 0.0,
                "disc_confusion_loss": 0.0,
                "disc_r1_penalty": 0.0,
                "modality_weights": [1.0/3, 1.0/3, 1.0/3],
            }

        # Spatial scaffold: process u_st.detach() to produce spatially-aware embeddings
        # CRITICAL: fused_st is detached - no gradients flow back to VAE (anti-pattern enforcement)
        # NOTE: Skip spatial scaffold for batched data (spatial scaffold requires full ST data)
        # Spatial scaffold should be applied once per epoch on full data in a separate step
        spatial_adj = getattr(self.spatial_scaffold, "spatial_adj", None)
        if spatial_adj is not None and u_st.shape[0] == spatial_adj.shape[0]:
            fused_st = self.spatial_scaffold(u_st.detach())
        else:
            # Batched processing: return zeros, spatial scaffold applied separately
            fused_st = torch.zeros_like(u_st)
        vae_output["fused_st"] = fused_st

        # VAE loss
        # BF-02 FIX: Pass st_indices from original batch for spatial loss
        loss_batch = {
            "recon_loss": vae_output["recon_loss"],
            "kl_loss": vae_output["kl_loss"],
            "graph_recon_loss": vae_output["graph_recon_loss"],
            "st_indices": batch.get("st_indices", None),  # For SpatialAwareLoss
            "cell_type_onehot": batch.get("cell_type_onehot", None),
            "st_condition_idx": batch.get("st_condition_idx", None),
        }
        vae_loss, vae_components = self.compute_vae_loss(
            u_sc, u_st, u_bulk, loss_batch,
            epoch,
        )

        # VAE update
        self.opt_vae.zero_grad()
        if not torch.isfinite(vae_loss):
            def _to_float(value: Any) -> float:
                if isinstance(value, torch.Tensor):
                    if value.numel() == 0:
                        return float("nan")
                    return float(value.detach().mean().item())
                return float(value)

            recon_val = _to_float(vae_components.get("recon_loss", float("nan")))
            kl_val = _to_float(vae_components.get("kl_loss", float("nan")))
            graph_val = _to_float(vae_components.get("graph_recon_loss", float("nan")))
            ot_val = _to_float(vae_components.get("ot_loss", float("nan")))
            bulk_val = _to_float(vae_components.get("bulk_prior_loss", float("nan")))
            raise RuntimeError(
                "Non-finite vae_loss detected: "
                f"recon_loss={recon_val}, "
                f"kl_loss={kl_val}, "
                f"graph_recon_loss={graph_val}, "
                f"ot_loss={ot_val}, "
                f"bulk_prior_loss={bulk_val}"
            )
        # NOTE: retain_graph=True is required for stability with adversarial training.
        # The adversarial loss in compute_vae_loss flows through the discriminator,
        # and removing retain_graph can cause async CUDA failures on some drivers.
        vae_loss.backward(retain_graph=True)

        # P0-02: Gradient NaN check BEFORE clip_grad_norm_
        # NaN gradients can corrupt optimizer momentum state if passed through clip
        if self._check_gradient_nan():
            self.opt_vae.zero_grad()
            logger.warning("Skipping VAE update due to NaN gradients")
            # Return early with skipped flag
            return {
                "vae_loss": vae_loss.item() if torch.isfinite(vae_loss) else float("nan"),
                "skipped_nan_grad": 1.0,
                "recon_loss": vae_components["recon_loss"].item(),
                "kl_loss": vae_components["kl_loss"].item(),
                "graph_recon_loss": vae_components["graph_recon_loss"].item(),
                "ot_loss": vae_components["ot_loss"].item(),
                "ot_loss_high": vae_components["ot_loss_high"].item(),
                "adversarial_loss": vae_components["adversarial_loss"].item(),
                "spatial_loss": vae_components["spatial_loss"].item(),
                "spatial_weight": vae_components["spatial_weight"].item(),
                "bulk_prior_loss": vae_components["bulk_prior_loss"].item(),
                "bulk_prior_lambda": vae_components["bulk_prior_lambda"].item(),
                "disc_loss": 0.0,
                "disc_real_loss": 0.0,
                "disc_confusion_loss": 0.0,
                "disc_r1_penalty": 0.0,
                "modality_weights": [1.0/3, 1.0/3, 1.0/3],
            }

        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=5.0)
        self.opt_vae.step()

        # BF-04: Pass sample counts for discriminator weighting
        # Use batch keys if available, otherwise use actual tensor shapes
        # Safeguard: ensure non-zero values to avoid division by zero
        n_sc = batch.get('n_sc', max(x_sc.shape[0], 1))
        n_st = batch.get('n_st', max(x_st.shape[0], 1))
        n_bulk = batch.get('n_bulk', max(x_bulk.shape[0], 1))

        # Discriminator loss and update
        disc_loss, disc_components = self.compute_discriminator_loss(
            u_sc.detach(), u_st.detach(), u_bulk.detach(), epoch,
            n_sc=n_sc, n_st=n_st, n_bulk=n_bulk
        )

        self.opt_disc.zero_grad()
        if not torch.isfinite(disc_loss):
            raise RuntimeError("Non-finite disc_loss detected")
        disc_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.0)
        self.opt_disc.step()

        # Collect losses for logging
        losses = {
            "vae_loss": vae_loss.item(),
            "recon_loss": vae_components["recon_loss"].item(),
            "kl_loss": vae_components["kl_loss"].item(),
            "graph_recon_loss": vae_components["graph_recon_loss"].item(),
            "ot_loss": vae_components["ot_loss"].item(),
            "ot_loss_high": vae_components["ot_loss_high"].item(),
            "adversarial_loss": vae_components["adversarial_loss"].item(),
            "spatial_loss": vae_components["spatial_loss"].item(),
            "spatial_weight": vae_components["spatial_weight"].item(),
            "bulk_prior_loss": vae_components["bulk_prior_loss"].item(),
            "bulk_prior_lambda": vae_components["bulk_prior_lambda"].item(),
            "disc_loss": disc_loss.item(),
            "disc_real_loss": disc_components["disc_real_loss"].item(),
            "disc_confusion_loss": disc_components["disc_confusion_loss"].item(),
            "disc_r1_penalty": disc_components["disc_r1_penalty"].item(),
            # BF-04: Log modality weights for debugging
            "modality_weights": disc_components["modality_weights"].tolist(),
        }

        # Update history
        for key, val in losses.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(val)

        return losses

    def _prepare_mini_batch(
        self,
        indices: dict[str, torch.Tensor],
        X_sc_cpu: torch.Tensor,
        X_st_cpu: torch.Tensor,
        X_bulk_cpu: torch.Tensor,
        guidance_data: Any,
    ) -> dict[str, Any]:
        """Slice CPU tensors and move mini-batch to GPU.

        Parameters
        ----------
        indices : dict
            Dict with 'sc_indices', 'st_indices', 'bulk_indices' (CPU tensors).
        X_sc_cpu, X_st_cpu, X_bulk_cpu : torch.Tensor
            Full CPU data tensors.
        guidance_data : Any
            Guidance graph data (will be moved to device in train_step).

        Returns
        -------
        dict
            Batch dict ready for train_step().
        """
        x_sc = X_sc_cpu[indices["sc_indices"]].to(self.device)
        x_st = X_st_cpu[indices["st_indices"]].to(self.device)
        x_bulk = X_bulk_cpu[indices["bulk_indices"]].to(self.device)

        batch = {
            "x_sc": x_sc,
            "x_st": x_st,
            "x_bulk": x_bulk,
            "guidance_data": guidance_data,
            "st_indices": indices["st_indices"],
            "n_sc": x_sc.shape[0],
            "n_st": x_st.shape[0],
            "n_bulk": x_bulk.shape[0],
        }
        return batch

    def train_step_mini_batch(
        self,
        indices: dict[str, torch.Tensor],
        X_sc_cpu: torch.Tensor,
        X_st_cpu: torch.Tensor,
        X_bulk_cpu: torch.Tensor,
        guidance_data: Any,
        epoch: int,
    ) -> dict[str, float]:
        """Training step for mini-batch mode (CPU storage + batch-wise GPU transfer).

        Prepares a mini-batch from CPU tensors and delegates to train_step().
        Keeps original train_step() unchanged for backward compatibility.

        Parameters
        ----------
        indices : dict
            Dict with 'sc_indices', 'st_indices', 'bulk_indices'.
        X_sc_cpu, X_st_cpu, X_bulk_cpu : torch.Tensor
            Full CPU data tensors.
        guidance_data : Any
            Guidance graph data.
        epoch : int
            Current epoch number.

        Returns
        -------
        dict
            Loss dict from train_step().
        """
        batch = self._prepare_mini_batch(
            indices, X_sc_cpu, X_st_cpu, X_bulk_cpu, guidance_data
        )
        return self.train_step(batch, epoch=epoch)

    def apply_epoch_spatial_loss(
        self,
        X_st_cpu: torch.Tensor,
        guidance_data: Any,
        encode_batch_size: int = 512,
    ) -> float:
        """Compute spatial loss on full ST embeddings (batch-wise encoding to avoid OOM).

        Runs a forward pass over the full ST dataset in batches to compute
        spatial smoothness loss once per epoch. This complements mini-batch
        training where batch-local spatial edges are too sparse.

        Parameters
        ----------
        X_st_cpu : torch.Tensor
            Full ST data tensor on CPU (n_spots, n_genes).
        guidance_data : Any
            Guidance graph data (passed to VAE for completeness).
        encode_batch_size : int, default 512
            Batch size for encoding ST data without OOM.

        Returns
        -------
        float
            Spatial loss value (0.0 if spatial scaffold not available).
        """
        if self.spatial_scaffold is None:
            return 0.0

        spatial_adj = getattr(self.spatial_scaffold, "spatial_adj", None)
        if spatial_adj is None:
            return 0.0

        self.vae.eval()
        n_spots = X_st_cpu.shape[0]
        u_st_chunks = []

        with torch.no_grad():
            for i in range(0, n_spots, encode_batch_size):
                x_st_batch = X_st_cpu[i : i + encode_batch_size].to(self.device)
                # VAE encoder: enc_st -> (z, mean, log_var)
                z, mean, log_var = self.vae.enc_st(x_st_batch)
                u_st_chunks.append(mean.cpu())

        # WR-01 NOTE: u_st_full is ~220MB for 454K spots x 128 dims (float32).
        # SpatialAwareLoss creates ~1.3GB intermediate for 2.7M edges.
        # Total is well under GPU capacity; monitored during testing with no OOM.
        u_st_full = torch.cat(u_st_chunks, dim=0).to(self.device)

        # IN-02 FIX: SpatialAwareLoss already imported at module level (line 35)
        # Wrap in no_grad for extra safety — this is diagnostic-only, not trained
        with torch.no_grad():
            spatial_loss = SpatialAwareLoss(
                u_st_full,
                self.spatial_scaffold,
                batch_indices=None,  # full dataset
            )

        self.vae.train()
        return float(spatial_loss.item()) if torch.isfinite(spatial_loss) else 0.0

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_epochs: int,
        log_every: int = 100,
    ) -> dict[str, list[float]]:
        """Train for multiple epochs.

        Args:
            dataloader: Training data loader
            n_epochs: Number of epochs to train
            log_every: Log interval (batches)

        Returns:
            history: Dict of loss histories per epoch
        """
        self.vae.train()
        self.discriminator.train()

        for epoch in range(n_epochs):
            epoch_losses = []
            for batch_idx, batch in enumerate(dataloader):
                losses = self.train_step(batch, epoch)
                epoch_losses.append(losses)

                if batch_idx % log_every == 0:
                    parts = []
                    for k, v in losses.items():
                        if isinstance(v, (int, float)):
                            parts.append(f"{k}: {v:.4f}")
                        else:
                            parts.append(f"{k}: {v}")
                    print(f"Epoch {epoch} | Batch {batch_idx} | " + " | ".join(parts))

            # Epoch summary
            avg_losses = {
                k: sum(l[k] for l in epoch_losses) / len(epoch_losses)
                for k, first_val in epoch_losses[0].items()
                if isinstance(first_val, (int, float))
            }
            print(f"=== Epoch {epoch} Summary ===")
            print(f"  VAE: {avg_losses['vae_loss']:.4f} (recon: {avg_losses['recon_loss']:.4f}, "
                  f"KL: {avg_losses['kl_loss']:.4f}, graph: {avg_losses['graph_recon_loss']:.4f}, "
                  f"OT: {avg_losses['ot_loss']:.4f}, adv: {avg_losses['adversarial_loss']:.4f})")
            print(f"  DISC: {avg_losses['disc_loss']:.4f} (real: {avg_losses['disc_real_loss']:.4f}, "
                  f"conf: {avg_losses['disc_confusion_loss']:.4f}, "
                  f"R1: {avg_losses['disc_r1_penalty']:.4f})")

        return self.history
