'''
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
'''

### Initial Author <2024>: Xiangyu Yin

import math
import psutil
import numpy as np
import torch
from maps_torch.constant import M_PI, ENERGY_RES_OFFSET, ENERGY_RES_SQRT
from maps_torch.default import default_fitting_elems, default_energy_consts

def get_channel_from_energy(energy, ev):
    for i, e in enumerate(ev):
        if e >= energy:
            return i
    return len(ev) - 1


def get_peak_centers(elements, coherent_sct_energy, compton_angle, e_consts=default_energy_consts):
    centers = {}
    for e in elements:
        if e == "COMPTON_AMPLITUDE":
            centers[e] = coherent_sct_energy / (
                1
                + (coherent_sct_energy / 511)
                * (1 - math.cos(compton_angle * 2 * M_PI / 360.0))
            )
        elif e == "COHERENT_SCT_AMPLITUDE":
            centers[e] = coherent_sct_energy
        elif e in default_fitting_elems:
            for i, er in enumerate(e_consts[e]):
                if er.energy > 0:
                    centers[e + "_" + str(i)] = er.energy
    return centers


def get_peak_ranges(
    elements,
    coherent_sct_energy,
    compton_angle,
    energy_offset,
    energy_slope,
    energy_quadratic,
    energy_range,
):  
    energy = np.linspace(
        energy_range[0],
        energy_range[1] + 1,
        energy_range[1] - energy_range[0] + 1,
    )
    ev = (
        energy_offset
        + energy_slope * energy
        + energy_quadratic * (energy**2)
    )
    centers = get_peak_centers(elements, coherent_sct_energy, compton_angle)
    ranges = {}
    for p, c in centers.items():
        try:
            width = math.sqrt(ENERGY_RES_OFFSET**2 + (c * ENERGY_RES_SQRT) ** 2) / 2000
            left = get_channel_from_energy(c - width, ev)
            right = get_channel_from_energy(c + width, ev)
            ranges[p] = (left, right)
        except Exception as e:
            print("Failed to get range for", p, "with center", c)
    return ranges


def estimate_safe_tile_size(spec_vol_shape, device='cuda'):
    """
    Estimate a safe tile size based on available memory and input shape.
    
    :param spec_vol_shape: Shape of the input spec_vol (height, width, depth)
    :param device: 'cuda' or 'cpu'
    :return: Estimated safe tile size
    """
    h, w, d = spec_vol_shape
    
    if device == 'cuda' and torch.cuda.is_available():
        # Get available GPU memory
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        available_mem = gpu_mem * 0.8  # Use 80% of available memory to be safe
        mem_per_pixel = d * 1024
    else:
        # Get available system memory
        system_mem = psutil.virtual_memory().available
        available_mem = system_mem * 0.5  # Use 50% of available memory to be safe
        mem_per_pixel = d * 2048
    
    # Calculate maximum number of pixels that can fit in memory
    max_pixels = available_mem / mem_per_pixel
    
    # Calculate tile size (assuming square tiles)
    tile_size = int(math.sqrt(max_pixels))
    
    # Ensure tile size is not larger than the input dimensions
    tile_size = min(tile_size, h, w)
    
    # Round down to nearest multiple of 32 for GPU efficiency
    tile_size = (tile_size // 32) * 32
    
    return max(32, tile_size)  # Ensure minimum tile size of 32


def estimate_n_workers(device='cuda'):
    """
    Estimate the number of workers based on the device and available resources.
    
    :param device: 'cuda' or 'cpu'
    :return: Estimated number of workers
    """
    if device == 'cuda' and torch.cuda.is_available():
        # Use number of available GPUs
        return torch.cuda.device_count()
    else:
        # Use number of CPU cores, leaving some headroom
        cpu_count = min(8, psutil.cpu_count(logical=False)-4)  # physical cores only
        return max(1, cpu_count)  # leave at least four cores free


def optimize_parallelization(spec_vol_shape, device='cuda'):
    """
    Optimize tile_size and n_workers for the given input shape and device.
    
    :param spec_vol_shape: Shape of the input spec_vol (height, width, depth)
    :param device: 'cuda' or 'cpu'
    :return: Tuple of (tile_size, n_workers)
    """
    tile_size = estimate_safe_tile_size(spec_vol_shape, device)
    n_workers = estimate_n_workers(device)
    
    return tile_size, n_workers


def split_into_tiles(spec_vol, tile_size):
    """Split the spec_vol into tiles."""
    h, w, d = spec_vol.shape
    tiles = []
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = spec_vol[i:i+tile_size, j:j+tile_size, :]
            tiles.append((tile, (i, j)))
    return tiles