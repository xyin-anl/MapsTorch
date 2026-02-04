"""
Structure-aware metrics for 1D XRF spectra.

These are primarily intended for:
  - evaluation (ranking fits / comparing models)
  - optional auxiliary losses during fitting (hybrid objectives)

All functions operate on tensors shaped like (..., n_channels).
"""

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence

import math
import torch
import torch.nn.functional as F

from mapstorch.default import default_energy_consts, default_elem_info
from mapstorch.map import _binding_energy_mask

ShapeMetric = Literal["w1", "mmd_rbf"]
ShiftMode = Literal["none", "blur", "mean", "min", "softmin"]
RoiProfile = Literal["box", "gaussian"]


def to_distribution(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Convert non-negative mass to a probability distribution along the last axis."""
    mass = torch.clamp(x, min=0.0)
    denom = torch.sum(mass, dim=-1, keepdim=True).clamp_min(float(eps))
    return mass / denom


def shift_1d(x: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift along the last axis with zero padding (no wrap)."""
    if shift == 0:
        return x
    n = x.shape[-1]
    shift = int(shift)
    if abs(shift) >= n:
        return torch.zeros_like(x)
    if shift > 0:
        pad = torch.zeros_like(x[..., :shift])
        return torch.cat([pad, x[..., : n - shift]], dim=-1)
    pad = torch.zeros_like(x[..., : (-shift)])
    return torch.cat([x[..., (-shift) :], pad], dim=-1)


def shift_blur_1d(
    x: torch.Tensor,
    *,
    max_shift: int,
    sigma: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Approximate marginalization over small global shifts by convolving over integer shifts."""
    max_shift = int(max_shift)
    sigma = float(sigma)
    if max_shift <= 0 or sigma <= 0.0:
        return x

    shifts = torch.arange(-max_shift, max_shift + 1, device=x.device, dtype=x.dtype)
    w = torch.exp(-0.5 * (shifts / sigma) ** 2)
    w = w / torch.sum(w).clamp_min(float(eps))

    out = torch.zeros_like(x)
    for s, ws in zip(shifts.tolist(), w.tolist()):
        out = out + float(ws) * shift_1d(x, int(s))

    # Keep as mass; caller can normalize if needed. Still, avoid degenerate all-zero mass.
    return out


def wasserstein1_per_sample(p: torch.Tensor, q: torch.Tensor, *, dx: float = 1.0) -> torch.Tensor:
    """Return per-sample 1D Wasserstein-1 distance between distributions p and q.

    p and q must be non-negative and sum to 1 along the last axis.
    Output shape is p.shape[:-1].
    """
    if p.shape != q.shape:
        raise ValueError(f"p shape {tuple(p.shape)} != q shape {tuple(q.shape)}")
    diff_cdf = torch.cumsum(p - q, dim=-1)
    return torch.sum(torch.abs(diff_cdf), dim=-1) * float(dx)


def wasserstein1_shift(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    dx: float = 1.0,
    shift_mode: ShiftMode = "none",
    max_shift: int = 0,
    blur_sigma: float = 1.0,
    softmin_tau: float = 0.05,
    reduction: Literal["sum", "mean"] = "sum",
    eps: float = 1e-12,
) -> torch.Tensor:
    """Shift-tolerant Wasserstein-1 between distributions p and q."""
    max_shift = int(max_shift)
    if shift_mode == "none" or max_shift <= 0:
        per = wasserstein1_per_sample(p, q, dx=dx)
        return torch.sum(per) if reduction == "sum" else torch.mean(per)

    if shift_mode == "blur":
        q2 = shift_blur_1d(q, max_shift=max_shift, sigma=float(blur_sigma), eps=eps)
        q2 = to_distribution(q2, eps=eps)
        per = wasserstein1_per_sample(p, q2, dx=dx)
        return torch.sum(per) if reduction == "sum" else torch.mean(per)

    vals = []
    for s in range(-max_shift, max_shift + 1):
        vals.append(wasserstein1_per_sample(p, to_distribution(shift_1d(q, s), eps=eps), dx=dx))
    stacked = torch.stack(vals, dim=0)  # (n_shift, *batch)

    if shift_mode == "mean":
        per = torch.mean(stacked, dim=0)
    elif shift_mode == "min":
        per = torch.amin(stacked, dim=0)
    elif shift_mode == "softmin":
        tau = torch.tensor(float(softmin_tau), device=p.device, dtype=p.dtype).clamp_min(float(eps))
        per = -tau * torch.logsumexp(-stacked / tau, dim=0)
    else:
        raise ValueError(f"Unsupported shift_mode={shift_mode!r}")

    return torch.sum(per) if reduction == "sum" else torch.mean(per)


def _gaussian_kernel1d(
    sigma: float,
    *,
    truncate: float = 3.0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    radius = int(math.ceil(float(truncate) * sigma))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2.0 * sigma**2))
    # Do not normalize: we want k(0) = 1 so it acts like an RBF kernel on the grid.
    return k


def mmd_rbf_multiscale(
    p: torch.Tensor,
    q: torch.Tensor,
    *,
    sigmas: Sequence[float],
    truncate: float = 3.0,
    reduction: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    """Multi-scale MMD^2 using translation-invariant RBF kernels on a 1D grid.

    For discrete distributions on an evenly spaced grid, MMD^2 can be computed as:
      MMD^2 = (p - q)^T K (p - q)
    where K_ij = exp(-(i-j)^2 / (2 sigma^2)).

    We exploit translation invariance by computing K (p - q) via 1D convolution.
    """
    if p.shape != q.shape:
        raise ValueError(f"p shape {tuple(p.shape)} != q shape {tuple(q.shape)}")
    diff = p - q
    n_ch = diff.shape[-1]
    flat = diff.reshape(-1, 1, n_ch)
    total = torch.zeros((flat.shape[0],), device=diff.device, dtype=diff.dtype)
    for sigma in sigmas:
        k = _gaussian_kernel1d(
            float(sigma), truncate=truncate, device=diff.device, dtype=diff.dtype
        ).view(1, 1, -1)
        conv = F.conv1d(flat, k, padding="same").squeeze(1)
        total = total + torch.sum(flat.squeeze(1) * conv, dim=-1)
    per = total.reshape(diff.shape[:-1])
    return torch.sum(per) if reduction == "sum" else torch.mean(per)


def _infer_batch_shape(params: dict[str, torch.Tensor]) -> tuple[int, ...]:
    batch_shape: tuple[int, ...] = ()
    for v in params.values():
        if isinstance(v, torch.Tensor) and v.ndim > 0:
            batch_shape = v.shape[:-1] if v.shape[-1] == 1 else v.shape
            break
    return batch_shape


def _as_channel_broadcast(val: torch.Tensor, *, batch_shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(val, torch.Tensor) or val.ndim == 0:
        return val
    if batch_shape and tuple(val.shape) == tuple(batch_shape):
        return val.unsqueeze(-1)
    return val


def build_roi_weights(
    params: dict[str, torch.Tensor],
    energy_range: tuple[int, int],
    *,
    elements_to_fit: Sequence[str],
    roi_n_sigma: float = 3.0,
    profile: RoiProfile = "box",
    include_scatter: bool = True,
    allowed_lines_only: bool = True,
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Build a differentiable per-channel ROI weight mask in [0, 1].

    The ROI is defined around expected line energies using the current energy calibration.
    """
    ch0, ch1 = int(energy_range[0]), int(energy_range[1])
    if ch1 < ch0:
        raise ValueError("energy_range stop must be >= start")

    batch_shape = _infer_batch_shape(params)
    device = next(
        (v.device for v in params.values() if isinstance(v, torch.Tensor)), torch.device("cpu")
    )
    dtype = next((v.dtype for v in params.values() if isinstance(v, torch.Tensor)), torch.float32)

    energy = torch.arange(ch0, ch1 + 1, device=device, dtype=dtype)
    if batch_shape:
        energy = energy.view((1,) * len(batch_shape) + (energy.numel(),))

    energy_offset = _as_channel_broadcast(params["ENERGY_OFFSET"], batch_shape=batch_shape)
    energy_slope = _as_channel_broadcast(params["ENERGY_SLOPE"], batch_shape=batch_shape)
    energy_quad = _as_channel_broadcast(params["ENERGY_QUADRATIC"], batch_shape=batch_shape)
    ev = energy_offset + energy_slope * energy + energy_quad * (energy**2)
    if batch_shape and tuple(ev.shape[:-1]) != tuple(batch_shape):
        ev = ev.expand(batch_shape + (ev.shape[-1],))

    fwhm_offset = _as_channel_broadcast(params["FWHM_OFFSET"], batch_shape=batch_shape)
    fwhm_fano = _as_channel_broadcast(params["FWHM_FANOPRIME"], batch_shape=batch_shape)
    coherent = params["COHERENT_SCT_ENERGY"]
    coherent_b = _as_channel_broadcast(coherent, batch_shape=batch_shape)

    roi = torch.zeros_like(ev, device=device, dtype=dtype)
    roi_n_sigma = float(roi_n_sigma)
    eps_t = torch.tensor(float(eps), device=device, dtype=dtype)

    def _combine(w: torch.Tensor, valid: Optional[torch.Tensor] = None) -> None:
        nonlocal roi
        if valid is not None:
            if isinstance(valid, torch.Tensor) and valid.ndim == roi.ndim - 1:
                valid = valid.unsqueeze(-1)
            w = w * valid.to(dtype=dtype)
        roi = torch.maximum(roi, w)

    def _line_weight(e0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        delta = ev - e0
        if profile == "box":
            return (torch.abs(delta) <= (roi_n_sigma * sigma)).to(dtype=dtype)
        denom = torch.clamp(roi_n_sigma * sigma, min=eps_t)
        return torch.exp(-0.5 * (delta / denom) ** 2)

    # Element lines.
    for elem in elements_to_fit:
        if elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
            continue
        if elem not in e_consts:
            continue
        for er in e_consts[elem]:
            e_val = float(getattr(er, "energy", 0.0))
            ratio = float(getattr(er, "ratio", 0.0))
            if e_val <= 0.0 or ratio == 0.0:
                continue
            sigma = torch.sqrt(
                torch.clamp((fwhm_offset / 2.3548) ** 2 + (e_val * 2.96) * fwhm_fano, min=0.0)
            )
            sigma = _as_channel_broadcast(sigma, batch_shape=batch_shape)
            e0 = torch.tensor(e_val, device=device, dtype=dtype)
            if batch_shape:
                e0 = e0.view((1,) * len(batch_shape) + (1,))
            valid = None
            if allowed_lines_only:
                valid = _binding_energy_mask(er, coherent_b.squeeze(-1) if coherent_b.ndim == ev.ndim else coherent_b, elem_info)
            _combine(_line_weight(e0, sigma), valid)

    # Scatter peaks.
    if include_scatter:
        # Elastic peak at the incident energy.
        sigma_el = torch.sqrt(
            torch.clamp((fwhm_offset / 2.3548) ** 2 + coherent_b * 2.96 * fwhm_fano, min=0.0)
        )
        _combine(_line_weight(coherent_b, sigma_el))

        # Compton peak.
        angle = params.get("COMPTON_ANGLE", torch.tensor(0.0, device=device, dtype=dtype))
        angle_b = _as_channel_broadcast(angle, batch_shape=batch_shape)
        compton_e = coherent_b / (
            1.0
            + (coherent_b / 511.0) * (1.0 - torch.cos(angle_b * 2.0 * math.pi / 360.0))
        )
        sigma_c = torch.sqrt(
            torch.clamp((fwhm_offset / 2.3548) ** 2 + compton_e * 2.96 * fwhm_fano, min=0.0)
        )
        _combine(_line_weight(compton_e, sigma_c))

    return roi


def build_element_roi_masks(
    params: dict[str, torch.Tensor],
    energy_range: tuple[int, int],
    *,
    elements_to_fit: Sequence[str],
    roi_n_sigma: float = 3.0,
    include_scatter: bool = False,
    allowed_lines_only: bool = True,
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    """Return per-element ROI masks (float in {0,1}) keyed by element/component name."""
    masks: dict[str, torch.Tensor] = {}
    roi_all = build_roi_weights(
        params,
        energy_range,
        elements_to_fit=elements_to_fit,
        roi_n_sigma=roi_n_sigma,
        profile="box",
        include_scatter=include_scatter,
        allowed_lines_only=allowed_lines_only,
        e_consts=e_consts,
        elem_info=elem_info,
        eps=eps,
    )

    # Rebuild per-element masks by filtering energies per element.
    # (We keep a separate pass for clarity; this is for evaluation/aux losses, not tight loops.)
    ch0, ch1 = int(energy_range[0]), int(energy_range[1])
    batch_shape = _infer_batch_shape(params)
    device = roi_all.device
    dtype = roi_all.dtype

    energy = torch.arange(ch0, ch1 + 1, device=device, dtype=dtype)
    if batch_shape:
        energy = energy.view((1,) * len(batch_shape) + (energy.numel(),))

    energy_offset = _as_channel_broadcast(params["ENERGY_OFFSET"], batch_shape=batch_shape)
    energy_slope = _as_channel_broadcast(params["ENERGY_SLOPE"], batch_shape=batch_shape)
    energy_quad = _as_channel_broadcast(params["ENERGY_QUADRATIC"], batch_shape=batch_shape)
    ev = energy_offset + energy_slope * energy + energy_quad * (energy**2)
    if batch_shape and tuple(ev.shape[:-1]) != tuple(batch_shape):
        ev = ev.expand(batch_shape + (ev.shape[-1],))

    fwhm_offset = _as_channel_broadcast(params["FWHM_OFFSET"], batch_shape=batch_shape)
    fwhm_fano = _as_channel_broadcast(params["FWHM_FANOPRIME"], batch_shape=batch_shape)
    coherent = params["COHERENT_SCT_ENERGY"]
    coherent_b = _as_channel_broadcast(coherent, batch_shape=batch_shape)

    roi_n_sigma = float(roi_n_sigma)
    eps_t = torch.tensor(float(eps), device=device, dtype=dtype)

    def _line_mask(e0: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        delta = ev - e0
        return (torch.abs(delta) <= (roi_n_sigma * sigma)).to(dtype=dtype)

    for elem in elements_to_fit:
        if not include_scatter and elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
            continue
        mask = torch.zeros_like(ev, device=device, dtype=dtype)
        if elem in e_consts and elem not in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
            for er in e_consts[elem]:
                e_val = float(getattr(er, "energy", 0.0))
                ratio = float(getattr(er, "ratio", 0.0))
                if e_val <= 0.0 or ratio == 0.0:
                    continue
                sigma = torch.sqrt(
                    torch.clamp(
                        (fwhm_offset / 2.3548) ** 2 + (e_val * 2.96) * fwhm_fano, min=0.0
                    )
                )
                sigma = _as_channel_broadcast(sigma, batch_shape=batch_shape)
                e0 = torch.tensor(e_val, device=device, dtype=dtype)
                if batch_shape:
                    e0 = e0.view((1,) * len(batch_shape) + (1,))
                valid = None
                if allowed_lines_only:
                    valid = _binding_energy_mask(
                        er,
                        coherent_b.squeeze(-1) if coherent_b.ndim == ev.ndim else coherent_b,
                        elem_info,
                    )
                    if isinstance(valid, torch.Tensor) and valid.ndim == mask.ndim - 1:
                        valid = valid.unsqueeze(-1)
                m = _line_mask(e0, torch.clamp(sigma, min=eps_t))
                if valid is not None:
                    m = m * valid.to(dtype=dtype)
                mask = torch.maximum(mask, m)
        elif include_scatter and elem == "COHERENT_SCT_AMPLITUDE":
            sigma_el = torch.sqrt(
                torch.clamp((fwhm_offset / 2.3548) ** 2 + coherent_b * 2.96 * fwhm_fano, min=0.0)
            )
            mask = _line_mask(coherent_b, torch.clamp(sigma_el, min=eps_t))
        elif include_scatter and elem == "COMPTON_AMPLITUDE":
            angle = params.get("COMPTON_ANGLE", torch.tensor(0.0, device=device, dtype=dtype))
            angle_b = _as_channel_broadcast(angle, batch_shape=batch_shape)
            compton_e = coherent_b / (
                1.0
                + (coherent_b / 511.0) * (1.0 - torch.cos(angle_b * 2.0 * math.pi / 360.0))
            )
            sigma_c = torch.sqrt(
                torch.clamp((fwhm_offset / 2.3548) ** 2 + compton_e * 2.96 * fwhm_fano, min=0.0)
            )
            mask = _line_mask(compton_e, torch.clamp(sigma_c, min=eps_t))

        # Avoid returning empty masks (keeps downstream metrics stable).
        if torch.any(mask > 0):
            masks[elem] = mask

    return masks
