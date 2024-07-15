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

import marimo

__generated_with = "0.7.1"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import sys
    from pathlib import Path; sys.path.append(str(Path(__file__).parent.parent.absolute()))

    from math import floor, ceil
    import numpy as np
    import plotly.express as px
    import marimo as mo

    from maps_torch.io import read_dataset
    return Path, ceil, floor, mo, np, px, read_dataset, sys


@app.cell
def __(mo):
    dataset = mo.ui.file_browser(filetypes=['.h5', '.h50', '.h51', '.h52', '.h53', '.h54', '.h55'],
                                 multiple=False)
    mo.md(f"Please select the dataset file (h5 file) \n{dataset}")
    return dataset,


@app.cell
def __(mo):
    spec_vol_path = mo.ui.dropdown(['MAPS/Spectra/mca_arr', 'MAPS/mca_arr'], 
                                   value='MAPS/mca_arr',
                                  label='Spectra volume location')
    energy_dimension = mo.ui.dropdown(['first', 'middle', 'last', 'guess'], value='guess',
                                     label='Energy dimension')
    dataset_button = mo.ui.run_button(label='Load')
    mo.hstack([spec_vol_path, energy_dimension, dataset_button], justify='start', gap=2)
    return dataset_button, energy_dimension, spec_vol_path


@app.cell
def __(dataset_button, e_dim_guess, mo):
    mo.stop(not dataset_button.value)
    mo.callout(f'Assume the {e_dim_guess} dimension is the energy', kind='info')
    return


@app.cell
def __(e_dim_pos, mo, np, px, spec_vol):
    spec_vol_t = np.moveaxis(spec_vol, e_dim_pos, -1)
    photon_counts = spec_vol_t.sum(axis=-1)
    photon_cts_plt = mo.ui.plotly(px.imshow(photon_counts))
    photon_cts_plt
    return photon_counts, photon_cts_plt, spec_vol_t


@app.cell
def __(mo, spec_vol_t):
    energy_range = mo.ui.range_slider(start=0, stop=spec_vol_t.shape[-1]-1, step=1, label='Energy channels', value=[50, 1450], full_width=True)
    energy_range
    return energy_range,


@app.cell
def __(
    ceil,
    energy_range,
    floor,
    mo,
    np,
    photon_counts,
    photon_cts_plt,
    spec_vol_t,
):
    if 'x' in photon_cts_plt.ranges and 'y' in photon_cts_plt.ranges:
        x_l = ceil(photon_cts_plt.ranges['x'][0])
        x_r = floor(photon_cts_plt.ranges['x'][1])
        y_l = ceil(photon_cts_plt.ranges['y'][0])
        y_r = floor(photon_cts_plt.ranges['y'][1])
    else:
        x_l, x_r = 0, photon_counts.shape[1]
        y_l, y_r = 0, photon_counts.shape[0]

    int_spec = spec_vol_t[y_l:y_r, x_l:x_r, energy_range.value[0]:energy_range.value[1]+1].sum(axis=(0, 1))
    int_spec_log = np.log10(np.clip(int_spec, 0, None)+1)
    int_spec_x = list(range(len(int_spec)))

    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    int_spec_fig = make_subplots(rows=2, cols=1)
    int_spec_fig.append_trace(go.Scatter(x=int_spec_x, y=int_spec, name='photon count'), row=1, col=1)
    int_spec_fig.append_trace(go.Scatter(x=int_spec_x, y=int_spec_log, name='log scale'), row=2, col=1)
    int_spec_fig.update_layout(showlegend=False)

    int_spec_plt = mo.ui.plotly(int_spec_fig)
    int_spec_plt
    return (
        go,
        int_spec,
        int_spec_fig,
        int_spec_log,
        int_spec_plt,
        int_spec_x,
        make_subplots,
        x_l,
        x_r,
        y_l,
        y_r,
    )


@app.cell
def __(energy_dimension, np, spec_vol):
    def guess_energy_dim_pos(spec_vol):
        assert len(spec_vol.shape) == 3, "The spectra volume must have 3 dimensions"
        return np.argmax(spec_vol.shape)

    if energy_dimension.value == 'guess':
        e_dim_pos = guess_energy_dim_pos(spec_vol)
        e_dim_guess = ['first', 'middle', 'last'][e_dim_pos]
    else:
        e_dim_pos = {'first':0, 'middle':1, 'last':2}[energy_dimension.value]
    return e_dim_guess, e_dim_pos, guess_energy_dim_pos


@app.cell
def __(dataset, dataset_button, mo, read_dataset, spec_vol_path):
    mo.stop(not dataset_button.value)
    dataset_dict = read_dataset(dataset.value[0].path, spec_vol_key=spec_vol_path.value)
    spec_vol = dataset_dict['spec_vol']
    return dataset_dict, spec_vol


if __name__ == "__main__":
    app.run()
