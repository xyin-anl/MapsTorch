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
import matplotlib.pyplot as plt
import scienceplots
from matplotlib import colors as mcolors
from matplotlib.transforms import blended_transform_factory

plt.style.use(["science", "notebook", "retro"])
from scipy.signal import find_peaks

from mapstorch.map import model_elem_spec, compton_peak, elastic_peak
from mapstorch.util import get_peak_ranges
from mapstorch.default import default_fitting_elems

_ELEMENT_COLOR_MAP: dict[str, tuple] = {}
_ELEMENT_MARKER_ALPHA = 0.5
_DEFAULT_FADE_FACTOR = 0.6


def _format_elem_label(elem: str) -> str:
    special = {
        "COHERENT_SCT_AMPLITUDE": "Elastic",
        "COMPTON_AMPLITUDE": "Compton",
    }
    if elem in special:
        return special[elem]
    token = elem.split("_")[0]
    return token.title()


def _get_element_color(elem: str):
    label = _format_elem_label(elem)
    if label not in _ELEMENT_COLOR_MAP:
        cmap = plt.get_cmap("tab10")
        idx = len(_ELEMENT_COLOR_MAP) % cmap.N
        _ELEMENT_COLOR_MAP[label] = cmap(idx)
    return _ELEMENT_COLOR_MAP[label]


def _with_alpha(color, alpha=_ELEMENT_MARKER_ALPHA):
    return mcolors.to_rgba(color, alpha=alpha)


def _element_color_with_fade(elem, idx=0, fade_factor=_DEFAULT_FADE_FACTOR):
    base_color = _get_element_color(elem)
    if fade_factor is None:
        scale = 1.0
    else:
        scale = fade_factor**idx
    alpha = max(0.0, min(1.0, _ELEMENT_MARKER_ALPHA * scale))
    return _with_alpha(base_color, alpha=alpha)


def _select_target_elements(tensors, target_elems, n_elem, reverse):
    elems = (
        target_elems
        if target_elems is not None
        else [e for e in default_fitting_elems if e in tensors]
    )
    elems = [e for e in elems if e in tensors]
    elems = sorted(elems, key=lambda e: tensors[e].item(), reverse=reverse)
    if n_elem is None:
        return elems
    return elems[:n_elem]


def _compute_element_ranges(tensors, elements, energy_range):
    coherent = float(tensors["COHERENT_SCT_ENERGY"].item())
    compton_angle = float(tensors["COMPTON_ANGLE"].item())
    energy_offset = float(tensors["ENERGY_OFFSET"].item())
    energy_slope = float(tensors["ENERGY_SLOPE"].item())
    energy_quadratic = float(tensors["ENERGY_QUADRATIC"].item())

    elem_ranges = {}
    for elem in elements:
        if elem not in default_fitting_elems:
            continue
        try:
            rg = get_peak_ranges(
                [elem],
                coherent,
                compton_angle,
                energy_offset,
                energy_slope,
                energy_quadratic,
                energy_range,
            )
        except Exception:
            continue
        if rg:
            elem_ranges[elem] = rg
    return elem_ranges


def _build_label_entries(elem_ranges, elements):
    labels = []
    for elem in elements:
        ranges = elem_ranges.get(elem)
        if not ranges:
            continue
        centers = [
            0.5 * (float(r[0]) + float(r[1])) for r in ranges.values() if r is not None
        ]
        if not centers:
            continue
        labels.append(
            {
                "x": centers[0],
                "text": _format_elem_label(elem),
                "elem": elem,
                "color": _get_element_color(elem),
            }
        )
    return labels


def _assign_label_lanes(label_positions_px, label_widths_px, pad_px=0):
    lanes_right_edge = []
    assigned_lanes = []
    for left_px, right_px in zip(
        [x - w / 2 - pad_px for x, w in zip(label_positions_px, label_widths_px)],
        [x + w / 2 + pad_px for x, w in zip(label_positions_px, label_widths_px)],
    ):
        placed = False
        for i in range(len(lanes_right_edge)):
            if left_px > lanes_right_edge[i]:
                lanes_right_edge[i] = right_px
                assigned_lanes.append(i)
                placed = True
                break
        if not placed:
            lanes_right_edge.append(right_px)
            assigned_lanes.append(len(lanes_right_edge) - 1)
    return assigned_lanes, len(lanes_right_edge)


def _annotate_element_labels(
    ax, labels, fontsize=11, draw_guides=False, fade_factor=_DEFAULT_FADE_FACTOR
):
    if not labels:
        return
    fig = ax.figure
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    labels_sorted = sorted(labels, key=lambda d: d["x"])
    widths_px = []
    x_px = []
    for item in labels_sorted:
        t = plt.Text(0, 0, item["text"], fontsize=fontsize)
        t.set_figure(fig)
        bbox = t.get_window_extent(renderer=renderer)
        widths_px.append(bbox.width)
        xp = ax.transData.transform((item["x"], ax.get_ylim()[0]))[0]
        x_px.append(xp)

    lanes, nlanes = _assign_label_lanes(x_px, widths_px, pad_px=2)
    try:
        fig.subplots_adjust(top=max(0.75, 0.90 - (nlanes - 1) * 0.05))
    except Exception:
        pass

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for item, lane in zip(labels_sorted, lanes):
        color = item.get("color", "0.3")
        y_axes = 1.02 + 0.07 * lane
        ax.text(
            item["x"],
            y_axes,
            item["text"],
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color=color,
            clip_on=False,
        )
        if draw_guides:
            guide_color = _element_color_with_fade(
                item.get("elem", ""), 0, fade_factor=fade_factor
            )
            ax.axvline(item["x"], color=guide_color, lw=0.8)


def _setup_element_plot(
    tensors,
    int_spec,
    energy_range,
    target_elems,
    n_elem,
    reverse,
    *,
    figsize=(12, 6),
):
    segment = int_spec[energy_range[0] : energy_range[1] + 1]
    target_elems = _select_target_elements(
        tensors, target_elems=target_elems, n_elem=n_elem, reverse=reverse
    )
    elem_ranges = _compute_element_ranges(tensors, target_elems, energy_range)
    labels = _build_label_entries(elem_ranges, target_elems)
    fig, axs = plt.subplots(2, figsize=figsize)
    for ax in axs:
        ax.plot(segment, color="gray")
    return fig, axs, target_elems, elem_ranges, labels


def plot_specs(spectra, labels=None, filename="fitting_res.png", save=False, show=False):
    labels = labels if labels is not None else list(range(len(spectra)))
    fig, axs = plt.subplots(2, figsize=(12, 6))
    for i, spec in enumerate(spectra):
        i_x = np.linspace(0, spec.size - 1, spec.size)
        label = str(labels[i])
        label_lower = label.strip().lower()
        special_color = None
        if label_lower in {"experiment", "expr"}:
            special_color = "gray"
        elif label_lower in {"background", "bkg"}:
            special_color = "silver"
        line_kwargs = {"label": label}
        if special_color is not None:
            line_kwargs["color"] = special_color
        for ax in axs:
            ax.plot(i_x, spec, **line_kwargs)
            if label_lower in {"background", "bkg"}:
                ax.fill_between(i_x, spec, color="silver", alpha=0.5)
    axs[1].set_ylim(1, None)
    axs[1].set_yscale("log")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    elif show:
        plt.show()
    plt.close()
    return fig, None


def plot_elem_amp_rank(
    tensors,
    target_elems=None,
    filename="element_amplitude_rank.png",
    keep_negtive=False,
    fade_factor=_DEFAULT_FADE_FACTOR,
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
    elements = list(amps.keys())
    values = [amps[e] for e in elements]
    colors = [_element_color_with_fade(e, 0) for e in elements]
    elem_labels = [_format_elem_label(e) if len(e) > 8 else e for e in elements]

    height = max(1.5, 0.3 * max(1, len(elements)))
    fig, ax = plt.subplots(figsize=(12, height))
    i_y = np.arange(len(elements))
    ax.barh(i_y, values, color=colors)
    ax.set_yticks(i_y, elem_labels)
    ax.tick_params(axis="y", left=False)
    ax.set_ylabel("Element Types")
    ax.set_xlabel("Fitted Amplitudes")
    plt.tight_layout()
    if save:
        plt.savefig(filename)
    elif show:
        plt.show()
    plt.close()
    return fig, amps


def plot_elem_peak_ranges(
    tensors,
    int_spec,
    energy_range,
    target_elems=None,
    n_elem=None,
    reverse=True,
    filename="element_peak_ranges.png",
    fade_factor=_DEFAULT_FADE_FACTOR,
    save=False,
    show=False,
):
    (
        fig,
        axs,
        target_elems,
        elem_ranges,
        labels,
    ) = _setup_element_plot(
        tensors,
        int_spec,
        energy_range,
        target_elems,
        n_elem,
        reverse,
        figsize=(12, 6),
    )
    res = {}
    for i, e in enumerate(target_elems):
        if e in default_fitting_elems:
            rg = elem_ranges.get(e)
            if rg is None:
                continue
            res[e] = rg
            ranges_sorted = sorted(rg.values(), key=lambda r: r[0])
            for idx, r in enumerate(ranges_sorted):
                color = _element_color_with_fade(e, idx, fade_factor)
                axs[0].axvspan(r[0], r[1], facecolor=color)
                axs[1].axvspan(r[0], r[1], facecolor=color)
    axs[1].set_ylim(1, None)
    _annotate_element_labels(axs[0], labels, fade_factor=fade_factor)
    axs[1].set_yscale("log")
    if save:
        plt.savefig(filename)
    elif show:
        plt.show()
    plt.close()
    return fig, res


def plot_elem_peak_pos(
    tensors,
    int_spec,
    energy_range,
    target_elems=None,
    n_elem=None,
    reverse=True,
    filename="element_peak_positions.png",
    include_guides=True,
    fade_factor=_DEFAULT_FADE_FACTOR,
    save=False,
    show=False,
):
    (
        fig,
        axs,
        target_elems,
        elem_ranges,
        labels,
    ) = _setup_element_plot(
        tensors,
        int_spec,
        energy_range,
        target_elems,
        n_elem,
        reverse,
        figsize=(12, 6),
    )
    for e in target_elems:
        ranges = elem_ranges.get(e)
        if not ranges:
            continue
        centers = [
            0.5 * (float(r[0]) + float(r[1])) for r in sorted(ranges.values(), key=lambda r: r[0])
        ]
        for idx, center in enumerate(centers):
            color = _element_color_with_fade(e, idx, fade_factor)
            for ax in axs:
                ax.axvline(center, color=color, lw=1.2)
    axs[1].set_ylim(1, None)
    _annotate_element_labels(
        axs[0], labels, draw_guides=include_guides, fade_factor=fade_factor
    )
    axs[1].set_yscale("log")
    if save:
        plt.savefig(filename)
    elif show:
        plt.show()
    plt.close()
    return fig, elem_ranges


def plot_elem_spec_contribs(
    tensors,
    int_spec,
    energy_range,
    target_elems=None,
    elem_amps={},
    n_elem=None,
    reverse=True,
    filename="element_rank.png",
    fade_factor=_DEFAULT_FADE_FACTOR,
    save=False,
    show=False,
):
    with torch.no_grad():
        (
            fig,
            axs,
            target_elems,
            elem_ranges,
            labels,
        ) = _setup_element_plot(
            tensors,
            int_spec,
            energy_range,
            target_elems,
            n_elem,
            reverse,
            figsize=(12, 6),
        )
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
                color = _element_color_with_fade(e, 0, fade_factor)
                axs[0].plot(spec.cpu().numpy(), color=color)
                axs[1].plot(spec.cpu().numpy(), color=color)
                tensors[e] = og_tensor
        axs[1].set_ylim(1, None)
        _annotate_element_labels(axs[0], labels, fade_factor=fade_factor)
        axs[1].set_yscale("log")
        if save:
            plt.savefig(filename)
        elif show:
            plt.show()
        plt.close()
        return fig, res

def plot_elem_markers(
    tensors,
    int_spec,
    energy_range,
    *,
    mode="ranges",
    fade_factor=_DEFAULT_FADE_FACTOR,
    **kwargs,
):
    mode = mode.lower()
    kwargs.setdefault("fade_factor", fade_factor)
    if mode in ("ranges", "peak_ranges"):
        return plot_elem_peak_ranges(tensors, int_spec, energy_range, **kwargs)
    if mode in ("lines", "peak_pos", "positions"):
        return plot_elem_peak_pos(tensors, int_spec, energy_range, **kwargs)
    if mode in ("contribs", "contributions", "spec"):
        return plot_elem_spec_contribs(tensors, int_spec, energy_range, **kwargs)
    raise ValueError(f"Unsupported mode '{mode}'.")
