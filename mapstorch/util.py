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

import math
import numpy as np
from mapstorch.constant import M_PI, ENERGY_RES_OFFSET, ENERGY_RES_SQRT
from mapstorch.default import (
    default_energy_consts,
    default_param_vals,
)


def get_channel_from_energy(energy, ev):
    for i, e in enumerate(ev):
        if e >= energy:
            return i
    return len(ev) - 1


def get_peak_centers(
    elements, coherent_sct_energy, compton_angle, e_consts=default_energy_consts
):
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
        elif e in e_consts:
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
    e_consts=default_energy_consts,
):
    energy = np.arange(energy_range[0], energy_range[1] + 1, dtype=float)
    ev = energy_offset + energy_slope * energy + energy_quadratic * (energy**2)
    centers = get_peak_centers(
        elements, coherent_sct_energy, compton_angle, e_consts=e_consts
    )
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


def _smooth_moving_average(values, window):
    """Return a lightly smoothed copy of the provided spectrum."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or window <= 1:
        return arr
    window = int(max(1, window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    if arr.size == 1:
        return arr
    padded = np.pad(arr, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: arr.size]


def _find_top_peaks(
    values,
    start_index,
    min_distance,
    top_k,
    min_height,
):
    """Identify up to top_k peaks using simple non-maximum suppression."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n < 3:
        return []
    start = max(1, min(start_index, n - 2))
    candidates = []
    for idx in range(start, n - 1):
        if arr[idx] < min_height:
            continue
        if arr[idx] >= arr[idx - 1] and arr[idx] >= arr[idx + 1]:
            candidates.append(idx)
    if not candidates:
        return []
    candidates.sort(key=lambda i: arr[i], reverse=True)
    selected = []
    for idx in candidates:
        if all(abs(idx - chosen) >= min_distance for chosen in selected):
            selected.append(idx)
        if len(selected) >= top_k:
            break
    return sorted(selected)


def _estimate_scatter_peak_indices(values, window: int=3, min_distance: int=10, top_k: int=20, threshold_percentile: int=60, start_index_frac: float=0.4):
    """Return heuristic Compton and elastic peak indices (relative to values)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None
    smoothed = _smooth_moving_average(arr, window=window)
    start_idx = int(start_index_frac * smoothed.size)
    start_idx = min(max(0, start_idx), max(smoothed.size - 1, 0))
    tail = smoothed[start_idx:] if start_idx < smoothed.size else smoothed
    if tail.size:
        threshold = float(np.percentile(tail, threshold_percentile))
    else:
        threshold = float(np.max(smoothed)) if smoothed.size else 0.0
    peaks = _find_top_peaks(
        smoothed,
        start_index=start_idx,
        min_distance=min_distance,
        top_k=top_k,
        min_height=threshold,
    )
    if not peaks:
        return None, None
    elastic_idx = max(peaks)
    compton_idx = None
    left_peaks = [idx for idx in peaks if idx < elastic_idx]
    if left_peaks:
        compton_idx = max(left_peaks, key=lambda idx: smoothed[idx])
    return compton_idx, elastic_idx


def estimate_and_update_params(
    int_spec,
    energy_range,
    coherent_sct_energy,
    param_reference=None,
):
    """Return dict of calibration parameter updates inferred from scatter peaks."""
    if param_reference is None:
        param_reference = default_param_vals
    if coherent_sct_energy is None or coherent_sct_energy <= 0:
        return {}
    arr = np.asarray(int_spec, dtype=float).ravel()
    if arr.size == 0:
        return {}
    start = max(0, min(int(energy_range[0]), arr.size - 1))
    stop = min(arr.size, max(start + 1, int(energy_range[1]) + 1))
    roi = arr[start:stop]
    if roi.size < 3:
        return {}
    compton_rel, elastic_rel = _estimate_scatter_peak_indices(roi)
    if elastic_rel is not None:
        elastic_idx = start + elastic_rel
    else:
        elastic_idx = max(1, int((arr.size - 1) // 1.9))
    if compton_rel is not None:
        compton_idx = start + compton_rel
    else:
        compton_idx = max(1, int((arr.size - 1) // 2))

    energy_slope = coherent_sct_energy / elastic_idx
    compton_energy = energy_slope * compton_idx
    try:
        compton_angle = (
            math.acos(1 - 511 * (1 / compton_energy - 1 / coherent_sct_energy)) * 180 / math.pi
        )
    except:
        compton_angle = default_param_vals["COMPTON_ANGLE"]

    updates = {"COHERENT_SCT_ENERGY": coherent_sct_energy, "ENERGY_SLOPE": energy_slope, "COMPTON_ANGLE": compton_angle}

    return updates


def branching_value_to_maps_lines(value: dict) -> str:
    """Convert a normalized branching-ratio value dict into MAPS-style BRANCHING_... lines."""
    if not isinstance(value, dict):
        return ""
    k_entries = value.get("k", {}) or {}
    l_family_entries = value.get("l_family", {}) or {}
    l_entries = value.get("l", {}) or {}

    lines: list[str] = []
    for elem, factors in k_entries.items():
        try:
            ka1, ka2, kb1, kb2 = [float(x) for x in factors]
        except Exception:
            continue
        lines.append(
            f"BRANCHING_RATIO_ADJUSTMENT_K: {elem}, {ka1:.6g}, {ka2:.6g}, {kb1:.6g}, {kb2:.6g}"
        )

    for elem, factors in l_family_entries.items():
        try:
            l1, l2, l3 = [float(x) for x in factors]
        except Exception:
            continue
        lines.append(
            f"BRANCHING_FAMILY_ADJUSTMENT_L: {elem}, {l1:.6g}, {l2:.6g}, {l3:.6g}"
        )

    for elem, factors in l_entries.items():
        try:
            vals = [float(x) for x in factors]
        except Exception:
            continue
        if len(vals) != 12:
            continue
        joined = ", ".join(f"{v:.6g}" for v in vals)
        lines.append(f"BRANCHING_RATIO_ADJUSTMENT_L: {elem}, {joined}")

    return "\n".join(lines)


def parse_maps_branching_lines(text: str) -> dict:
    """Parse MAPS-style BRANCHING_... lines into a normalized branching-ratio value dict."""
    value = {"k": {}, "l_family": {}, "l": {}}
    if not text:
        return value
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line.startswith("BRANCHING_") or ":" not in line:
            continue
        tag, rest = line.split(":", 1)
        parts = [p.strip() for p in rest.split(",") if p.strip()]
        if not parts:
            continue
        name = parts[0]
        nums = []
        for tok in parts[1:]:
            try:
                nums.append(float(tok))
            except Exception:
                nums = []
                break
        if not nums:
            continue
        if tag == "BRANCHING_RATIO_ADJUSTMENT_K" and len(nums) >= 4:
            ka1 = 1.0
            ka2, kb1, kb2 = nums[1:4] if len(nums) >= 4 else (1.0, 1.0, 1.0)
            value["k"][name] = [ka1, float(ka2), float(kb1), float(kb2)]
        elif tag == "BRANCHING_FAMILY_ADJUSTMENT_L" and len(nums) >= 3:
            value["l_family"][name] = [float(nums[0]), float(nums[1]), float(nums[2])]
        elif tag == "BRANCHING_RATIO_ADJUSTMENT_L" and len(nums) >= 12:
            vals = list(nums[:12])
            vals[0] = 1.0
            value["l"][name] = vals
    return value


def __getattr__(name):
    # Keep the widget imports used by the apps without loading notebook packages.
    if name in {"PeriodicTableWidget", "BranchingRatioWidget"}:
        from mapstorch import widget

        return getattr(widget, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
