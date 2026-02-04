"""
Loss/metric utilities for MapsTorch spectrum fitting.

These are written to work with spectra shaped like (..., n_channels) and support
optional per-channel weights and channel subset selection.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch

LossName = Literal[
    "mse",
    "l1",
    "poisson_nll",
    "poisson_deviance",
    "chi2",
]


def compute_loss(
    pred: torch.Tensor,
    obs: torch.Tensor,
    *,
    loss: LossName,
    indices=None,
    weights: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    sigma2_floor: float = 0.0,
    chi2_variant: Literal["pearson", "neyman"] = "pearson",
    reduction: Literal["sum", "mean", "weighted_mean"] = "sum",
) -> torch.Tensor:
    """Compute a scalar loss between predicted and observed counts.

    Args:
        pred: Predicted spectrum (expected counts), shape (..., n_channels).
        obs: Observed spectrum (counts), shape (..., n_channels).
        loss: Loss name.
        indices: Optional channel indices (applied on last dimension).
        weights: Optional per-channel weights (broadcastable to pred/obs).
        eps: Small constant used to clamp predicted counts in log terms / denominators.
        sigma2_floor: Variance floor for chi-square-style losses (in counts^2).
        chi2_variant: "pearson" uses var≈pred; "neyman" uses var≈obs.
        reduction: How to reduce per-channel losses.
    """
    if indices is not None:
        pred = pred[..., indices]
        obs = obs[..., indices]
        if weights is not None:
            weights = weights[..., indices]

    if pred.shape != obs.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != obs shape {tuple(obs.shape)}")

    if weights is not None:
        weights = torch.as_tensor(weights, device=pred.device, dtype=pred.dtype)
        if weights.shape != pred.shape:
            # Rely on PyTorch broadcasting but ensure last dim matches.
            if weights.shape and weights.shape[-1] != pred.shape[-1]:
                raise ValueError(
                    f"weights last dim {weights.shape[-1]} != n_channels {pred.shape[-1]}"
                )

    eps_t = torch.tensor(float(eps), device=pred.device, dtype=pred.dtype)
    sigma2_floor_t = torch.tensor(float(sigma2_floor), device=pred.device, dtype=pred.dtype)

    if loss == "mse":
        per = (pred - obs) ** 2
    elif loss == "l1":
        per = torch.abs(pred - obs)
    elif loss == "poisson_nll":
        mu = torch.clamp(pred, min=eps_t)
        y = torch.clamp(obs, min=0.0)
        per = mu - y * torch.log(mu)
    elif loss == "poisson_deviance":
        mu = torch.clamp(pred, min=eps_t)
        y = torch.clamp(obs, min=0.0)
        # Convention: y * log(y/mu) = 0 when y == 0.
        ylog = torch.where(y > 0, y * torch.log(torch.clamp(y / mu, min=eps_t)), torch.zeros_like(y))
        per = 2.0 * (ylog - (y - mu))
    elif loss == "chi2":
        y = torch.clamp(obs, min=0.0)
        mu = torch.clamp(pred, min=0.0)
        if chi2_variant == "pearson":
            var = torch.clamp(mu, min=0.0) + sigma2_floor_t
        elif chi2_variant == "neyman":
            var = torch.clamp(y, min=0.0) + sigma2_floor_t
        else:
            raise ValueError(f"Unsupported chi2_variant={chi2_variant!r}")
        var = torch.clamp(var, min=eps_t)
        per = (y - mu) ** 2 / var
    else:
        raise ValueError(f"Unsupported loss={loss!r}")

    if weights is not None:
        per = per * weights

    if reduction == "sum":
        return torch.sum(per)
    if reduction == "mean":
        return torch.mean(per)
    if reduction == "weighted_mean":
        if weights is None:
            return torch.mean(per)
        denom = torch.sum(weights).clamp_min(eps_t)
        return torch.sum(per) / denom

    raise ValueError(f"Unsupported reduction={reduction!r}")
