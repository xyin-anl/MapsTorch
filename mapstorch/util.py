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
import anywidget
import traitlets
from mapstorch.constant import M_PI, ENERGY_RES_OFFSET, ENERGY_RES_SQRT
from mapstorch.default import (
    default_fitting_elems,
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
    ev = energy_offset + energy_slope * energy + energy_quadratic * (energy**2)
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


def _smooth_moving_average(values, window: int = 7):
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
    start_index: int = 0,
    min_distance: int = 25,
    top_k: int = 10,
    min_height: float = 0.0,
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


def _estimate_scatter_peak_indices(values, min_distance: int = 25):
    """Return heuristic Compton and elastic peak indices (relative to values)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None, None
    smoothed = _smooth_moving_average(arr, window=7)
    start_idx = int(0.3 * smoothed.size)
    start_idx = min(max(0, start_idx), max(smoothed.size - 1, 0))
    tail = smoothed[start_idx:] if start_idx < smoothed.size else smoothed
    if tail.size:
        threshold = float(np.percentile(tail, 90))
    else:
        threshold = float(np.max(smoothed)) if smoothed.size else 0.0
    peaks = _find_top_peaks(
        smoothed,
        start_index=start_idx,
        min_distance=min_distance,
        top_k=10,
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


def _select_param_key(candidate_names, available_names):
    for name in candidate_names:
        if name in available_names:
            return name
    return None


def _get_param_value_safe(params, key, default=0.0):
    if key is None:
        return default
    value = params.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


class PeriodicTableWidget(anywidget.AnyWidget):
    # Synced properties between Python and JavaScript
    selected_elements = traitlets.Dict({}).tag(sync=True)
    disabled_elements = traitlets.List([]).tag(sync=True)
    states = traitlets.Int(1).tag(sync=True)
    disabled_color = traitlets.Unicode("rgb(255, 255, 255)").tag(sync=True)  # white
    unselected_color = traitlets.Unicode("rgb(215, 215, 215)").tag(
        sync=True
    )  # light silver
    selected_colors = traitlets.List(["rgb(191, 219, 254)"]).tag(sync=True)  # blue-200
    on_change = traitlets.Dict({}).tag(sync=True)  # To handle selection changes

    # Widget front-end JavaScript code
    _esm = """
    const elementTable = [
      ['H', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', 'He'],
      ['Li', 'Be', '', '', '', '', '', '', '', '', '', '', 'B', 'C', 'N', 'O', 'F', 'Ne'],
      ['Na', 'Mg', '', '', '', '', '', '', '', '', '', '', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar'],
      ['K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'],
      ['Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe'],
      ['Cs', 'Ba', '*', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn'],
      ['Fr', 'Ra', '#', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'],
      ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
      ['', '', '*', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
      ['', '', '#', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr']
    ];

    function createElementDiv(element, model) {
      const div = document.createElement('div');

      if (element === '' || element === '*' || element === '#') {
        div.style.width = '38px';
        div.style.height = '38px';
        div.style.display = 'flex';
        div.style.alignItems = 'center';
        div.style.justifyContent = 'center';
        div.style.fontSize = '0.875rem';
        div.style.userSelect = 'none';
        div.textContent = element;
        return div;
      }

      div.style.width = '38px';
      div.style.height = '38px';
      div.style.display = 'flex';
      div.style.alignItems = 'center';
      div.style.justifyContent = 'center';
      div.style.fontSize = '0.875rem';
      div.style.borderRadius = '0.25rem';
      div.style.transition = 'all 0.2s';
      div.style.userSelect = 'none';
      div.style.border = 'none';
      div.textContent = element;

      const updateElementStyle = () => {
        const selected = model.get('selected_elements');
        const disabled = new Set(model.get('disabled_elements'));

        if (disabled.has(element)) {
          div.style.backgroundColor = model.get('disabled_color');
          div.style.cursor = 'not-allowed';
        } else {
          const colors = model.get('selected_colors');
          div.style.backgroundColor = element in selected ? 
            colors[selected[element]] : 
            model.get('unselected_color');
          div.style.cursor = 'pointer';
        }
      };

      if (!model.get('disabled_elements').includes(element)) {
        div.addEventListener('mouseover', () => {
          div.style.filter = 'brightness(0.9)';
        });

        div.addEventListener('mouseout', () => {
          div.style.filter = 'brightness(1)';
        });

        div.addEventListener('click', () => {
          const currentSelected = model.get('selected_elements');
          const newSelected = {...currentSelected};

          if (element in newSelected) {
            if (newSelected[element] < model.get('states') - 1) {
              newSelected[element]++;
            } else {
              delete newSelected[element];
            }
          } else {
            newSelected[element] = 0;
          }

          model.set('selected_elements', newSelected);
          model.save_changes();

          model.set('on_change', { 
            selected: Object.keys(newSelected),
            states: Object.entries(newSelected).map(([el, state]) => ({element: el, state}))
          });
          model.save_changes();
        });
      }

      // Initial style
      requestAnimationFrame(() => {
        updateElementStyle();
      });

      // Listen for changes
      model.on('change:selected_elements', updateElementStyle);
      model.on('change:disabled_elements', updateElementStyle);
      model.on('change:selected_colors', updateElementStyle);

      return div;
    }

    function render({ model, el }) {
      // Create container
      const container = document.createElement('div');
      container.style.display = 'inline-block';
      container.style.padding = '1rem';
      container.style.backgroundColor = 'white';
      container.style.borderRadius = '0.5rem';
      container.style.boxShadow = '0 1px 3px 0 rgba(0, 0, 0, 0.1)';

      // Create grid container
      const grid = document.createElement('div');
      grid.style.display = 'grid';
      grid.style.gap = '4px';
      grid.style.gridTemplateColumns = 'repeat(18, 38px)';

      // Create table
      elementTable.forEach(row => {
        row.forEach(element => {
          const elementDiv = createElementDiv(element, model);
          grid.appendChild(elementDiv);
        });
      });

      container.appendChild(grid);

      // Create selection display
      const selectionDiv = document.createElement('div');
      selectionDiv.style.marginTop = '1rem';
      selectionDiv.style.fontSize = '0.875rem';
      selectionDiv.style.color = '#4B5563';

      function updateSelection() {
        const selected = model.get('selected_elements');
        selectionDiv.textContent = 'Selected: ' + 
          Object.entries(selected)
            .map(([el, state]) => `${el}`)
            .join(', ');
      }

      updateSelection();
      model.on('change:selected_elements', updateSelection);

      container.appendChild(selectionDiv);
      el.appendChild(container);
    }

    export default { render };
    """

    _css = ""

    def __init__(
        self,
        initial_selected=None,
        initial_disabled=None,
        states=1,
        disabled_color=None,
        unselected_color=None,
        selected_colors=None,
        on_selection_change=None,
    ):
        """Initialize the periodic table widget."""
        # Initialize traitlets first
        super().__init__()

        # Set number of states
        self.states = states

        # Set disabled elements first
        self.disabled_elements = list(initial_disabled or [])

        # Set colors
        if disabled_color is not None:
            self.disabled_color = disabled_color
        if unselected_color is not None:
            self.unselected_color = unselected_color

        # Set default colors for states if not provided
        if selected_colors is not None:
            self.selected_colors = selected_colors[:states]
        else:
            default_colors = [
                "rgb(191, 219, 254)",  # blue-200
                "rgb(147, 197, 253)",  # blue-300
                "rgb(96, 165, 250)",  # blue-400
            ]
            self.selected_colors = default_colors[:states]

        # Set initial selection state
        if initial_selected:
            self.selected_elements = {
                element: state
                for element, state in initial_selected.items()
                if state < states and element not in self.disabled_elements
            }

        # Set up callback if provided
        if on_selection_change:
            self.observe(
                lambda change: on_selection_change(change["new"]), names=["on_change"]
            )
