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
import numpy as np
from maps_torch.constant import M_PI, ENERGY_RES_OFFSET, ENERGY_RES_SQRT
from maps_torch.opt import default_fitting_elems, default_energy_consts

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