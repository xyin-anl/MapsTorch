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
        super().__init__()

        self.states = states
        self.disabled_elements = list(initial_disabled or [])

        if disabled_color is not None:
            self.disabled_color = disabled_color
        if unselected_color is not None:
            self.unselected_color = unselected_color

        if selected_colors is not None:
            self.selected_colors = selected_colors[:states]
        else:
            default_colors = [
                "rgb(191, 219, 254)",  # blue-200
                "rgb(147, 197, 253)",  # blue-300
                "rgb(96, 165, 250)",  # blue-400
            ]
            self.selected_colors = default_colors[:states]

        if initial_selected:
            self.selected_elements = {
                element: state
                for element, state in initial_selected.items()
                if state < states and element not in self.disabled_elements
            }

        if on_selection_change:
            self.observe(
                lambda change: on_selection_change(change["new"]), names=["on_change"]
            )


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


class BranchingRatioWidget(anywidget.AnyWidget):
    value = traitlets.Dict({}).tag(sync=True)

    def __init__(self, initial_lines: str = "", value=None, **kwargs):
        raw_value = kwargs.pop("value", None)
        if value is None and raw_value is not None:
            value = raw_value

        super().__init__(**kwargs)

        if isinstance(value, str) and value.strip():
            self.value = parse_maps_branching_lines(value)
        elif isinstance(value, dict):
            self.value = value
        elif initial_lines:
            self.value = parse_maps_branching_lines(initial_lines)
        else:
            self.value = {"k": {}, "l_family": {}, "l": {}}

    _esm = r"""
    function normalizeValue(v) {
      const out = (v && typeof v === "object") ? v : {};
      out.k = out.k && typeof out.k === "object" ? out.k : {};
      out.l_family = out.l_family && typeof out.l_family === "object" ? out.l_family : {};
      out.l = out.l && typeof out.l === "object" ? out.l : {};
      return out;
    }

    function clampNumber(x) {
      const n = Number(x);
      return Number.isFinite(n) ? n : 0;
    }

    function linesFromValue(v) {
      const val = normalizeValue(v);
      const lines = [];
      for (const [elem, factors] of Object.entries(val.k)) {
        const arr = Array.isArray(factors) ? factors : [];
        const ka1 = 1.0;
        const ka2 = clampNumber(arr[1] ?? 1.0);
        const kb1 = clampNumber(arr[2] ?? 1.0);
        const kb2 = clampNumber(arr[3] ?? 1.0);
        lines.push(`BRANCHING_RATIO_ADJUSTMENT_K: ${elem}, ${ka1}, ${ka2}, ${kb1}, ${kb2}`);
      }
      for (const [elem, factors] of Object.entries(val.l_family)) {
        const arr = Array.isArray(factors) ? factors : [];
        const l1 = clampNumber(arr[0] ?? 1.0);
        const l2 = clampNumber(arr[1] ?? 1.0);
        const l3 = clampNumber(arr[2] ?? 1.0);
        lines.push(`BRANCHING_FAMILY_ADJUSTMENT_L: ${elem}, ${l1}, ${l2}, ${l3}`);
      }
      for (const [elem, factors] of Object.entries(val.l)) {
        const arr = Array.isArray(factors) ? factors : [];
        if (arr.length < 12) continue;
        const out = arr.slice(0, 12).map((x, i) => (i === 0 ? 1.0 : clampNumber(x)));
        lines.push(`BRANCHING_RATIO_ADJUSTMENT_L: ${elem}, ${out.join(", ")}`);
      }
      return lines.join("\n");
    }

    function parseLines(text) {
      const value = { k: {}, l_family: {}, l: {} };
      const lines = String(text || "").split(/\r?\n/);
      for (const raw of lines) {
        const line = raw.trim();
        if (!line.startsWith("BRANCHING_")) continue;
        const idx = line.indexOf(":");
        if (idx === -1) continue;
        const tag = line.slice(0, idx).trim();
        const rest = line.slice(idx + 1);
        const parts = rest.split(",").map((s) => s.trim()).filter(Boolean);
        if (parts.length < 2) continue;
        const name = parts[0];
        const nums = parts.slice(1).map((x) => Number(x));
        if (nums.some((n) => !Number.isFinite(n))) continue;
        if (tag === "BRANCHING_RATIO_ADJUSTMENT_K" && nums.length >= 4) {
          value.k[name] = [1.0, nums[1] ?? 1.0, nums[2] ?? 1.0, nums[3] ?? 1.0];
        } else if (tag === "BRANCHING_FAMILY_ADJUSTMENT_L" && nums.length >= 3) {
          value.l_family[name] = [nums[0], nums[1], nums[2]];
        } else if (tag === "BRANCHING_RATIO_ADJUSTMENT_L" && nums.length >= 12) {
          const arr = nums.slice(0, 12);
          arr[0] = 1.0;
          value.l[name] = arr;
        }
      }
      return value;
    }

    export function render({ model, el }) {
      el.classList.add("mapstorch-branching-widget-root");

      const container = document.createElement("div");
      container.className = "brw-container";
      // Inline styles for robustness across notebook/app CSS environments.
      container.style.width = "100%";
      container.style.maxWidth = "820px";
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.gap = "10px";
      container.style.padding = "10px 12px";
      container.style.border = "1px solid rgba(15, 23, 42, 0.12)";
      container.style.borderRadius = "12px";
      container.style.background = "rgba(255, 255, 255, 0.6)";

      const titlebar = document.createElement("div");
      titlebar.className = "brw-titlebar";

      const title = document.createElement("div");
      title.className = "brw-title";
      title.textContent = "Branching ratios";

      const subtitle = document.createElement("div");
      subtitle.className = "brw-subtitle";
      subtitle.textContent = "Multipliers; Ka1/La1 fixed at 1.0";

      titlebar.appendChild(title);
      titlebar.appendChild(subtitle);

      const controls = document.createElement("div");
      controls.className = "brw-controls";

      const modeLabel = document.createElement("div");
      modeLabel.className = "brw-mode-label";
      modeLabel.textContent = "Type";

      const modeSelect = document.createElement("select");
      modeSelect.className = "brw-select";
      for (const opt of ["K", "L family", "L (12 lines)"]) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        modeSelect.appendChild(o);
      }

      const addName = document.createElement("input");
      addName.className = "brw-input-name";
      addName.placeholder = "Element (e.g. Fe or Au_L)";

      const addBtn = document.createElement("button");
      addBtn.className = "brw-btn brw-btn-primary";
      addBtn.textContent = "Add";

      controls.appendChild(modeLabel);
      controls.appendChild(modeSelect);
      controls.appendChild(addName);
      controls.appendChild(addBtn);

      const table = document.createElement("div");
      table.className = "brw-table";

      const advanced = document.createElement("details");
      advanced.className = "brw-advanced";
      const advancedSummary = document.createElement("summary");
      advancedSummary.textContent = "Advanced";
      advanced.appendChild(advancedSummary);

      const advBody = document.createElement("div");
      advBody.className = "brw-advbody";

      const advHelp = document.createElement("div");
      advHelp.className = "brw-advhelp";
      advHelp.textContent = "Copy/paste MAPS-style BRANCHING_... lines (optional).";

      const advBtns = document.createElement("div");
      advBtns.className = "brw-advbtns";

      const importBtn = document.createElement("button");
      importBtn.className = "brw-btn";
      importBtn.textContent = "Import lines…";

      const copyBtn = document.createElement("button");
      copyBtn.className = "brw-btn";
      copyBtn.textContent = "Copy lines";

      advBtns.appendChild(importBtn);
      advBtns.appendChild(copyBtn);

      advBody.appendChild(advHelp);
      advBody.appendChild(advBtns);
      advanced.appendChild(advBody);

      const dialog = document.createElement("dialog");
      dialog.className = "brw-dialog";
      const dialogTitle = document.createElement("div");
      dialogTitle.className = "brw-dialog-title";
      dialogTitle.textContent = "Import MAPS branching lines";
      const dialogText = document.createElement("textarea");
      dialogText.className = "brw-dialog-text";
      dialogText.rows = 10;
      dialogText.placeholder = "BRANCHING_RATIO_ADJUSTMENT_K: Fe, 1, 1, 1, 1\n...";

      const dialogBtns = document.createElement("div");
      dialogBtns.className = "brw-dialog-btns";
      const dialogCancel = document.createElement("button");
      dialogCancel.className = "brw-btn";
      dialogCancel.textContent = "Cancel";
      const dialogApply = document.createElement("button");
      dialogApply.className = "brw-btn brw-btn-primary";
      dialogApply.textContent = "Apply";
      dialogBtns.appendChild(dialogCancel);
      dialogBtns.appendChild(dialogApply);

      dialog.appendChild(dialogTitle);
      dialog.appendChild(dialogText);
      dialog.appendChild(dialogBtns);
      container.appendChild(dialog);

      function setValue(next) {
        model.set("value", normalizeValue(next));
        model.save_changes();
      }

      function currentModeKey() {
        if (modeSelect.value === "K") return "k";
        if (modeSelect.value === "L family") return "l_family";
        return "l";
      }

      function renderTable() {
        table.replaceChildren();
        const val = normalizeValue(model.get("value"));
        const key = currentModeKey();
        const entries = Object.entries(val[key]).sort((a, b) => a[0].localeCompare(b[0]));

        if (!entries.length) {
          const empty = document.createElement("div");
          empty.textContent = "No entries yet.";
          empty.className = "brw-empty";
          table.appendChild(empty);
          return;
        }

        for (const [name, factors] of entries) {
          const row = document.createElement("div");
          row.className = "brw-row";

          const label = document.createElement("div");
          label.textContent = name;
          label.className = "brw-name";

          const inputs = document.createElement("div");
          inputs.className = "brw-inputs";

          const removeBtn = document.createElement("button");
          removeBtn.textContent = "Remove";
          removeBtn.className = "brw-btn brw-btn-danger";

          const arr = Array.isArray(factors) ? factors.slice() : [];

          function makeInput(labelText, idx, locked) {
            const wrap = document.createElement("div");
            wrap.className = "brw-field";
            const lab = document.createElement("div");
            lab.textContent = labelText;
            lab.className = "brw-field-label";
            const inp = document.createElement("input");
            inp.type = "number";
            inp.step = "0.01";
            inp.className = locked ? "brw-field-input brw-field-input-locked" : "brw-field-input";
            inp.value = String(arr[idx] ?? (locked ? 1.0 : 1.0));
            if (locked) {
              inp.disabled = true;
              inp.value = "1.0";
            }
            inp.addEventListener("input", () => {
              const next = normalizeValue(model.get("value"));
              const nextArr = Array.isArray(next[key][name]) ? next[key][name].slice() : [];
              while (nextArr.length < (key === "l" ? 12 : (key === "k" ? 4 : 3))) nextArr.push(1.0);
              if (key === "k" && idx === 0) nextArr[0] = 1.0;
              else if (key === "l" && idx === 0) nextArr[0] = 1.0;
              else nextArr[idx] = clampNumber(inp.value);
              next[key][name] = nextArr;
              setValue(next);
            });
            wrap.appendChild(lab);
            wrap.appendChild(inp);
            return wrap;
          }

          if (key === "k") {
            arr[0] = 1.0;
            inputs.classList.add("brw-cols-4");
            inputs.appendChild(makeInput("Ka1", 0, true));
            inputs.appendChild(makeInput("Ka2", 1, false));
            inputs.appendChild(makeInput("Kb1(+3)", 2, false));
            inputs.appendChild(makeInput("Kb2", 3, false));
          } else if (key === "l_family") {
            inputs.classList.add("brw-cols-3");
            inputs.appendChild(makeInput("L1", 0, false));
            inputs.appendChild(makeInput("L2", 1, false));
            inputs.appendChild(makeInput("L3", 2, false));
          } else {
            arr[0] = 1.0;
            const labels = ["La1","La2","Lb1","Lb2","Lb3","Lb4","Lg1","Lg2","Lg3","Lg4","Ll","Ln"];
            inputs.classList.add("brw-cols-4");
            for (let i = 0; i < 12; i++) {
              inputs.appendChild(makeInput(labels[i], i, i === 0));
            }
          }

          removeBtn.addEventListener("click", () => {
            const next = normalizeValue(model.get("value"));
            delete next[key][name];
            setValue(next);
            renderTable();
          });

          row.appendChild(label);
          row.appendChild(inputs);
          row.appendChild(removeBtn);
          table.appendChild(row);
        }
      }

      addBtn.addEventListener("click", () => {
        const name = addName.value.trim();
        if (!name) return;
        const key = currentModeKey();
        const next = normalizeValue(model.get("value"));
        if (!next[key][name]) {
          if (key === "k") next[key][name] = [1.0, 1.0, 1.0, 1.0];
          else if (key === "l_family") next[key][name] = [1.0, 1.0, 1.0];
          else next[key][name] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
          setValue(next);
          renderTable();
        }
        addName.value = "";
      });

      modeSelect.addEventListener("change", () => {
        renderTable();
      });

      copyBtn.addEventListener("click", async () => {
        const text = linesFromValue(model.get("value"));
        try {
          await navigator.clipboard.writeText(text);
        } catch (_) {
          // ignore
        }
      });

      importBtn.addEventListener("click", () => {
        dialogText.value = linesFromValue(model.get("value"));
        if (typeof dialog.showModal === "function") dialog.showModal();
        else dialog.setAttribute("open", "");
      });

      dialogCancel.addEventListener("click", () => {
        if (typeof dialog.close === "function") dialog.close();
        else dialog.removeAttribute("open");
      });

      dialogApply.addEventListener("click", () => {
        const parsed = parseLines(dialogText.value);
        setValue(parsed);
        if (typeof dialog.close === "function") dialog.close();
        else dialog.removeAttribute("open");
        renderTable();
      });

      container.appendChild(titlebar);
      container.appendChild(controls);
      container.appendChild(table);
      container.appendChild(advanced);
      el.appendChild(container);

      const v0 = normalizeValue(model.get("value"));
      setValue(v0);
      renderTable();

      model.on("change:value", () => {
        renderTable();
      });
    }

    export default { render };
    """

    _css = """
    .mapstorch-branching-widget-root, .mapstorch-branching-widget-root * { box-sizing: border-box; }
    .brw-container {
      width: 100%;
      max-width: 820px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      padding: 10px 12px;
      border: 1px solid rgba(15, 23, 42, 0.12);
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.6);
    }

    .brw-titlebar { display: flex; flex-direction: column; gap: 2px; }
    .brw-title { font-weight: 700; font-size: 14px; color: rgb(15, 23, 42); }
    .brw-subtitle { font-size: 12px; color: rgb(100, 116, 139); }

    .brw-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
    .brw-mode-label { font-size: 12px; color: rgb(100, 116, 139); }
    .brw-select {
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid rgba(15, 23, 42, 0.16);
      background: white;
      font-size: 13px;
    }
    .brw-input-name {
      flex: 1;
      min-width: 160px;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid rgba(15, 23, 42, 0.16);
      font-size: 13px;
    }

    .brw-btn {
      padding: 6px 10px;
      border-radius: 8px;
      border: 1px solid rgba(15, 23, 42, 0.16);
      background: white;
      font-size: 12px;
      cursor: pointer;
    }
    .brw-btn:hover { background: rgba(2, 132, 199, 0.06); }
    .brw-btn-primary {
      background: rgb(37, 99, 235);
      color: white;
      border-color: rgba(37, 99, 235, 0.6);
    }
    .brw-btn-primary:hover { background: rgb(29, 78, 216); }
    .brw-btn-danger {
      color: rgb(185, 28, 28);
      border-color: rgba(185, 28, 28, 0.35);
    }
    .brw-btn-danger:hover { background: rgba(185, 28, 28, 0.06); }

    .brw-table {
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding-top: 2px;
    }
    .brw-empty { color: rgb(100, 116, 139); font-size: 12px; padding: 8px 2px; }
    .brw-row {
      display: grid;
      grid-template-columns: 120px minmax(0, 1fr) auto;
      align-items: start;
      gap: 10px;
      padding: 10px;
      border: 1px solid rgba(15, 23, 42, 0.10);
      border-radius: 12px;
      background: white;
    }
    .brw-name { font-weight: 600; color: rgb(30, 41, 59); font-size: 13px; padding-top: 2px; }

    .brw-inputs { display: grid; gap: 8px; min-width: 0; }
    .brw-cols-3 { grid-template-columns: repeat(3, minmax(84px, 1fr)); }
    .brw-cols-4 { grid-template-columns: repeat(4, minmax(76px, 1fr)); }

    .brw-field { display: flex; flex-direction: column; gap: 4px; }
    .brw-field-label { font-size: 11px; color: rgb(100, 116, 139); }
    .brw-field-input {
      width: 100%;
      padding: 6px 8px;
      border-radius: 8px;
      border: 1px solid rgba(15, 23, 42, 0.14);
      font-size: 13px;
      color: rgb(29, 78, 216);
      background: rgb(248, 250, 252);
    }
    .brw-field-input:focus { outline: 2px solid rgba(37, 99, 235, 0.25); border-color: rgba(37, 99, 235, 0.55); }
    .brw-field-input-locked {
      color: rgb(148, 163, 184);
      background: rgb(241, 245, 249);
    }

    .brw-advanced { padding-top: 2px; }
    .brw-advanced summary { color: rgb(100, 116, 139); font-size: 12px; cursor: pointer; }
    .brw-advbody { margin-top: 8px; display: flex; flex-direction: column; gap: 8px; }
    .brw-advhelp { color: rgb(100, 116, 139); font-size: 12px; }
    .brw-advbtns { display: flex; gap: 8px; align-items: center; }

    .brw-dialog {
      border: 1px solid rgba(15, 23, 42, 0.16);
      border-radius: 14px;
      padding: 12px;
      max-width: 680px;
    }
    .brw-dialog::backdrop { background: rgba(15, 23, 42, 0.20); }
    .brw-dialog-title { font-weight: 700; font-size: 13px; color: rgb(15, 23, 42); margin-bottom: 8px; }
    .brw-dialog-text {
      width: 640px;
      max-width: 70vw;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 12px;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid rgba(15, 23, 42, 0.14);
      background: rgb(248, 250, 252);
      color: rgb(30, 41, 59);
      resize: vertical;
    }
    .brw-dialog-btns { display: flex; justify-content: flex-end; gap: 8px; margin-top: 10px; }

    @media (max-width: 720px) {
      .brw-row { grid-template-columns: 1fr; }
      .brw-btn-danger { justify-self: start; }
    }
    """
