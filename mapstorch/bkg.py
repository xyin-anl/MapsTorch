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


def snip_op(background, current_width, max_of_xmin, min_of_xmax, device):
    range_tensor = torch.arange(background.shape[-1], device=device, dtype=background.dtype).expand(
        background.shape
    )
    lo_index = torch.clamp(range_tensor - current_width, max_of_xmin, min_of_xmax).to(
        torch.int64
    )
    hi_index = torch.clamp(range_tensor + current_width, max_of_xmin, min_of_xmax).to(
        torch.int64
    )
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
):
    spec = spec.to(device)

    n_ch = spec.shape[-1]
    local_idx = torch.arange(n_ch, device=device, dtype=spec.dtype)
    ch0 = er[0]
    global_idx = local_idx + ch0

    energy = e_offset + (e_slope * global_idx) + (e_quad * (global_idx**2))
    sigma_sq = (fwhm_offset / 2.3548) ** 2 + energy * 2.96 * fwhm_fanoprime
    sigma_sq = torch.clamp(sigma_sq, min=0.0)
    current_width = snip_width * 2.35 * torch.sqrt(sigma_sq) / e_slope

    conv_input = spec.reshape(-1, 1, n_ch)
    ones = torch.ones((1, 1, boxcar_size), dtype=spec.dtype, device=device)
    background = F.conv1d(conv_input, ones / boxcar_size, padding="same")

    background = background.view_as(spec)

    background = torch.log(torch.log(background + 1) + 1)
    max_of_xmin = 0
    min_of_xmax = n_ch - 1
    for _ in range(2):
        background = snip_op(
            background, current_width, max_of_xmin, min_of_xmax, device=device
        )
    while torch.max(current_width).item() >= 0.5:
        background = snip_op(
            background, current_width, max_of_xmin, min_of_xmax, device=device
        )
        current_width = current_width / M_SQRT2
    background = torch.exp(torch.exp(background) - 1) - 1
    return torch.nan_to_num(background, nan=0.0, posinf=0.0, neginf=0.0)
