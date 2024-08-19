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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from maps_torch.map import model_elem_spec, compton_peak, elastic_peak
from maps_torch.util import get_peak_ranges
from maps_torch.default import default_fitting_elems


def plot_spec_peaks(
    spec, peak_half_width=5, prominence=100, generate_plot=False, save=False, show=True
):
    peaks, _ = find_peaks(spec, prominence=prominence)
    peak_ranges = [
        (max(p - peak_half_width, 0), min(p + peak_half_width, spec.shape[-1]))
        for p in peaks
    ]
    if not generate_plot:
        return peak_ranges
    fig, axs = plt.subplots(2, figsize=(12, 6))
    for i in range(2):
        axs[i].plot(spec)
        axs[i].plot(peaks, spec[peaks], "x", color="red")
        for r in peak_ranges:
            axs[i].axvspan(r[0], r[1], facecolor="gray", alpha=0.5)
    axs[1].set_yscale("log")
    if save:
        plt.savefig("peak_ranges.png")
    elif show:
        plt.show()
    plt.close()
    return fig, peak_ranges


def plot_specs(spectra, labels=None, dataset="fitting_res.png", save=False, show=False):
    labels = labels if labels is not None else list(range(len(spectra)))
    fig, axs = plt.subplots(2, figsize=(8, 4))
    for i, spec in enumerate(spectra):
        i_x = np.linspace(0, spec.size - 1, spec.size)
        axs[0].plot(i_x, spec, label=str(labels[i]))
        axs[1].plot(i_x, spec, label=str(labels[i]))
    axs[1].set_ylim(1, None)
    axs[1].set_yscale("log")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(dataset)
    elif show:
        plt.show()
    plt.close()
    return fig, None


def plot_elem_amp_rank(
    tensors,
    target_elems=None,
    dataset="element_amplitude_rank.png",
    keep_negtive=False,
    save=False,
    show=False,
):
    target_elems = (
        target_elems
        if target_elems is not None
        else [e for e in default_fitting_elems if e in tensors]
    )
    if keep_negtive:
        amps = {p: tensors[p].item() for p in target_elems}
    else:
        amps = {p: tensors[p].item() for p in target_elems if (tensors[p].item() >= 0)}
    amps = dict(sorted(amps.items(), key=lambda item: item[1]))

    fig, ax = plt.subplots(figsize=(8, 0.2 * len(amps)))
    i_y = np.arange(len(amps))
    ax.barh(i_y, amps.values(), color="silver")
    ax.set_yticks(i_y, amps.keys())
    ax.set_ylabel("Element Types")
    ax.set_xlabel("Fitted Amplitudes")
    plt.tight_layout()
    if save:
        plt.savefig(dataset)
    elif show:
        plt.show()
    plt.close()
    return fig, amps


def plot_elem_peak_ranges(
    tensors,
    int_spec,
    energy_range,
    target_elems=None,
    n_elem=9,
    reverse=True,
    dataset="element_peak_ranges.png",
    save=False,
    show=False,
):
    target_elems = (
        target_elems
        if target_elems is not None
        else [e for e in default_fitting_elems if e in tensors]
    )
    target_elems = sorted(
        target_elems, key=lambda e: tensors[e].item(), reverse=reverse
    )[:n_elem]
    fig, axs = plt.subplots(2, figsize=(14, 6))
    colormap = plt.get_cmap("tab10", len(target_elems) + 1)
    axs[0].plot(int_spec[energy_range[0] : energy_range[1] + 1], color=colormap(0))
    axs[1].plot(int_spec[energy_range[0] : energy_range[1] + 1], color=colormap(0))
    res = {}
    for i, e in enumerate(target_elems):
        if e in default_fitting_elems:
            rg = get_peak_ranges(
                [e],
                tensors["COHERENT_SCT_ENERGY"].item(),
                tensors["COMPTON_ANGLE"].item(),
                tensors["ENERGY_OFFSET"].item(),
                tensors["ENERGY_SLOPE"].item(),
                tensors["ENERGY_QUADRATIC"].item(),
                energy_range,
            )
            res[e] = rg
            alpha = 0.6
            for r in rg.values():
                axs[0].axvspan(r[0], r[1], alpha=alpha, facecolor=colormap(i + 1))
                axs[1].axvspan(r[0], r[1], alpha=alpha, facecolor=colormap(i + 1))
                alpha /= 1.2
    axs[1].set_ylim(1, None)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(target_elems))
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    cbar = fig.colorbar(sm, ax=axs)
    cbar.set_ticks(range(1 + len(target_elems)))
    cbar.set_ticklabels(["Spectrum"] + target_elems)
    axs[1].set_yscale("log")
    if save:
        plt.savefig(dataset)
    elif show:
        plt.show()
    plt.close()
    return fig, res


def plot_elem_spec_contribs(
    tensors,
    int_spec,
    energy_range,
    target_elems=None,
    elem_amps={},
    n_elem=9,
    reverse=True,
    dataset="element_rank.png",
    save=False,
    show=False,
):
    with torch.no_grad():
        target_elems = (
            target_elems
            if target_elems is not None
            else [e for e in default_fitting_elems if e in tensors]
        )
        target_elems = sorted(
            target_elems, key=lambda e: tensors[e].item(), reverse=reverse
        )[:n_elem]
        fig, axs = plt.subplots(2, figsize=(10, 6))
        colormap = plt.get_cmap("tab10", len(target_elems) + 1)
        axs[0].plot(int_spec[energy_range[0] : energy_range[1] + 1], color=colormap(0))
        axs[1].plot(int_spec[energy_range[0] : energy_range[1] + 1], color=colormap(0))
        res = {}
        for i, e in enumerate(target_elems):
            if e in default_fitting_elems and e in tensors:
                og_tensor = tensors[e].clone()
                amp_val = elem_amps.get(e, og_tensor.item())
                tensors[e] = torch.tensor(amp_val, device=tensors[e].device)
                energy = torch.linspace(
                    energy_range[0],
                    energy_range[1] + 1,
                    energy_range[1] - energy_range[0] + 1,
                    device=tensors[e].device,
                )
                ev = (
                    tensors["ENERGY_OFFSET"]
                    + tensors["ENERGY_SLOPE"] * energy
                    + tensors["ENERGY_QUADRATIC"] * (energy**2)
                )
                if e == "COMPTON_AMPLITUDE":
                    spec = compton_peak(tensors, ev, tensors["ENERGY_SLOPE"])
                elif e == "COHERENT_SCT_AMPLITUDE":
                    spec = elastic_peak(tensors, ev, tensors["ENERGY_SLOPE"])
                else:
                    spec = model_elem_spec(tensors, e, ev, device=tensors[e].device)
                res[e] = spec.cpu().numpy()
                axs[0].plot(spec.cpu().numpy(), color=colormap(i + 1))
                axs[1].plot(spec.cpu().numpy(), color=colormap(i + 1))
                tensors[e] = og_tensor
        axs[1].set_ylim(1, None)
        norm = mpl.colors.Normalize(vmin=0, vmax=len(target_elems))
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = fig.colorbar(sm, ax=axs)
        cbar.set_ticks(range(1 + len(target_elems)))
        cbar.set_ticklabels(["Spectrum"] + target_elems)
        axs[1].set_yscale("log")
        if save:
            plt.savefig(dataset)
        elif show:
            plt.show()
        plt.close()
        return fig, res
