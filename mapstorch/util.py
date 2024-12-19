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
import torch
import anywidget
import traitlets
from mapstorch.constant import M_PI, ENERGY_RES_OFFSET, ENERGY_RES_SQRT
from mapstorch.default import default_fitting_elems, default_energy_consts


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


def estimate_gpu_tile_size(spec_vol_shape):
    """
    Estimate a safe tile size based on available memory and input shape.

    :param spec_vol_shape: Shape of the input spec_vol (height, width, depth)
    :param device: 'cuda' or 'cpu'
    :return: Estimated safe tile size
    """
    h, w, d = spec_vol_shape

    # Get available GPU memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    available_mem = gpu_mem * 0.8  # Use 80% of available memory to be safe
    mem_per_pixel = d * 2048

    # Calculate maximum number of pixels that can fit in memory
    max_pixels = available_mem / mem_per_pixel

    # Calculate tile size (assuming square tiles)
    tile_size = int(math.sqrt(max_pixels))

    # Ensure tile size is not larger than the input dimensions
    tile_size = min(tile_size, h, w)

    # Round down to nearest multiple of 32 for GPU efficiency
    tile_size = (tile_size // 32) * 32
    res = max(32, tile_size)  # Ensure minimum tile size of 32
    print(f"Estimated tile size: {res}")

    return res


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
