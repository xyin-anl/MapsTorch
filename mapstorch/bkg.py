"""
Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.

Copyright 2024. UChicago Argonne, LLC. This software was produced
under U.S. Government contract DE-AC02-06CH11357 for Argonne National
Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
modified to produce derivative works, such modified software should
be clearly marked, so as not to confuse it with the version available
from ANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

### Initial Author <2024>: Xiangyu Yin

import torch
import torch.nn.functional as F
from mapstorch.constant import M_SQRT2


def _resolve_bool_option(value):
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            value = value.reshape(-1)[0]
        return bool(value.item() >= 0.5)
    return bool(value)


def snip_op(background, current_width, max_of_xmin, min_of_xmax, device):
    n_ch = background.shape[-1]
    range_tensor = torch.arange(n_ch, device=device, dtype=background.dtype).expand(
        background.shape
    )
    # Handle NaN in current_width and ensure valid index range
    current_width_safe = torch.nan_to_num(current_width, nan=0.0, posinf=float(n_ch - 1), neginf=0.0)
    lo_raw = range_tensor - current_width_safe
    hi_raw = range_tensor + current_width_safe
    # Clamp and convert to int64 safely
    lo_index = torch.clamp(lo_raw, max_of_xmin, min_of_xmax).to(torch.int64)
    hi_index = torch.clamp(hi_raw, max_of_xmin, min_of_xmax).to(torch.int64)
    # Extra safety: ensure indices are in valid range after int conversion
    lo_index = torch.clamp(lo_index, 0, n_ch - 1)
    hi_index = torch.clamp(hi_index, 0, n_ch - 1)
    avg_values = (
        torch.gather(background, -1, lo_index) + torch.gather(background, -1, hi_index)
    ) / 2
    return torch.minimum(avg_values, background)


def snip_bkg(
    spec,
    er,
    e_offset,
    e_slope,
    e_quad,
    snip_width,
    fwhm_offset,
    fwhm_fanoprime,
    boxcar_size=5,
    device="cpu",
    use_single_log=False,
    use_fwhm_width=False,
    use_full_spectrum=False,
    return_full=False,
    fwhm_fano_coeff=3.58,
):
    spec = spec.to(device)
    use_single_log = _resolve_bool_option(use_single_log)
    use_fwhm_width = _resolve_bool_option(use_fwhm_width)
    use_full_spectrum = _resolve_bool_option(use_full_spectrum)

    n_ch = spec.shape[-1]
    local_idx = torch.arange(n_ch, device=device, dtype=spec.dtype)
    ch0, ch1 = int(er[0]), int(er[1])
    if use_full_spectrum:
        global_idx = local_idx
        max_of_xmin = max(ch0, 0)
        min_of_xmax = min(ch1, n_ch - 1)
    else:
        global_idx = local_idx + ch0
        max_of_xmin = 0
        min_of_xmax = n_ch - 1

    energy = e_offset + (e_slope * global_idx) + (e_quad * (global_idx**2))
    if use_fwhm_width:
        sigma_sq = (fwhm_offset / 2.3548) ** 2 + energy * fwhm_fano_coeff * fwhm_fanoprime
    else:
        # Match the legacy XRF-Maps SNIP window scaling used in MAPS.
        sigma_sq = (e_offset / 2.3548) ** 2 + energy * 3.58 * e_slope
    sigma_sq = torch.clamp(sigma_sq, min=0.0)
    # Clamp e_slope to avoid division by zero or very small values (can happen during
    # aggressive optimization like LBFGS). Also handle NaN and clamp to valid range.
    e_slope_safe = e_slope if not isinstance(e_slope, torch.Tensor) else torch.clamp(e_slope.abs(), min=1e-6)
    current_width = snip_width * 2.35 * torch.sqrt(sigma_sq) / e_slope_safe
    # Handle NaN (from numerical issues) and clamp to valid index range
    current_width = torch.nan_to_num(current_width, nan=0.0, posinf=float(n_ch - 1), neginf=0.0)
    current_width = torch.clamp(current_width, min=0.0, max=float(n_ch - 1))

    conv_input = spec.reshape(-1, 1, n_ch)
    ones = torch.ones((1, 1, boxcar_size), dtype=spec.dtype, device=device)
    background = F.conv1d(conv_input, ones / boxcar_size, padding="same")

    background = background.view_as(spec)

    # Match historical MAPS SNIP behavior by default, with single-log available
    # as an opt-in for users who prefer the newer transform.
    if use_single_log:
        background = torch.log1p(background)
    else:
        background = torch.log1p(torch.log1p(background))
    background = torch.nan_to_num(background, posinf=0.0, neginf=0.0)
    for _ in range(2):
        background = snip_op(
            background, current_width, max_of_xmin, min_of_xmax, device=device
        )
    while torch.max(current_width).item() >= 0.5:
        background = snip_op(
            background, current_width, max_of_xmin, min_of_xmax, device=device
        )
        current_width = current_width / M_SQRT2
    if use_single_log:
        background = torch.expm1(background)
    else:
        background = torch.expm1(torch.expm1(background))
    background = torch.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0)
    if use_full_spectrum and not return_full:
        background = background[..., ch0 : ch1 + 1]
    return background
