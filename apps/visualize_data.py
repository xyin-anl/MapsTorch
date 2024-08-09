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

__generated_with = "0.7.9"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import os, sys, pickle
    from pathlib import Path; sys.path.append(str(Path(__file__).parent.parent.absolute()))

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import matplotlib.pyplot as plt
    import marimo as mo
    import torch
    return (
        Path,
        go,
        make_subplots,
        mo,
        np,
        os,
        pd,
        pickle,
        plt,
        px,
        sys,
        torch,
    )


@app.cell
def __(mo):
    mo.callout(mo.md(r'''If you encounter error messages about figure size too large. You can increase the default image size limit by:

    * **Linux/Mac** Run command `export MARIMO_OUTPUT_MAX_BYTES=10_000_000` on the terminal before running the notebook, or add it to your shell profile
    * **Windows** Open the start menu, search for the "Advanced System Settings" control panel and click on it, click on the "Environment Variables" button toward the bottom of the screen, follow the prompts to add the `MARIMO_OUTPUT_MAX_BYTES` variable to the user table, set the value to be `10_000_000`
    *k'''), kind='warn')
    return


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
def __(dataset, dataset_button, mo, spec_vol_path):
    mo.stop(not dataset_button.value)
    from maps_torch.io import read_dataset
    dataset_dict = read_dataset(dataset.value[0].path, spec_vol_key=spec_vol_path.value)
    spec_vol = dataset_dict['spec_vol']
    return dataset_dict, read_dataset, spec_vol


@app.cell
def __(mo, spec_vol):
    output_dir = mo.ui.file_browser(selection_mode='directory', multiple=False, label='Please select the output folder with generated maps information')
    output_dir if spec_vol is not None else None
    return output_dir,


@app.cell
def __(mo, spec_vol):
    load_output_button = mo.ui.run_button(label='Load maps')
    load_output_button.right() if spec_vol is not None else None
    return load_output_button,


@app.cell
def __(load_output_button, mo, os, output_dir):
    mo.stop(not load_output_button.value)
    try:
        with open(output_dir.value[0].path + os.sep +'energy_range.txt', 'r') as range_handle:
            lines = range_handle.readlines()
            energy_range = (int(lines[0].strip()), int(lines[1].strip()))
            range_message = 'Energy range have been loaded successfully'
    except Exception as e:
        energy_range = None
        range_message = f'* Failed to load maps: {e}\n'
    mo.callout(range_message, kind='danger' if energy_range is None else 'success')
    return energy_range, lines, range_handle, range_message


@app.cell
def __(load_output_button, mo, os, output_dir, pickle):
    mo.stop(not load_output_button.value)
    try:
        with open(output_dir.value[0].path + os.sep +'tensor_maps.pkl', 'rb') as maps_handle:
            tensor_maps = pickle.load(maps_handle)
            maps_message = 'Maps have been loaded successfully'
    except Exception as e:
        tensor_maps = None
        maps_message = f'* Failed to load maps: {e}\n'
    mo.callout(maps_message, kind='danger' if tensor_maps is None else 'success')
    return maps_handle, maps_message, tensor_maps


@app.cell
def __(load_output_button, mo, os, output_dir, pickle):
    mo.stop(not load_output_button.value)
    try:
        with open(output_dir.value[0].path + os.sep +'full_spec_fit.pkl', 'rb') as spec_handle:
            full_spec_fit = pickle.load(spec_handle)
            spec_message = 'Fitted spectra volume has been loaded successfully.'
    except Exception as e:
        full_spec_fit = None
        spec_message = f'* Failed to load fitted spectra volume: {e}\n'
    mo.callout(spec_message, kind='danger' if full_spec_fit is None else 'success')
    return full_spec_fit, spec_handle, spec_message


@app.cell
def __(load_output_button, mo, os, output_dir, pickle):
    mo.stop(not load_output_button.value)
    try:
        with open(output_dir.value[0].path + os.sep +'full_bkg.pkl', 'rb') as bkg_handle:
            full_bkg = pickle.load(bkg_handle)
            bkg_message = 'Background volume has been loaded successfully.'
    except Exception as e:
        full_bkg = None
        bkg_message = f'* Failed to load fitted background volume: {e}\n'
    mo.callout(bkg_message, kind='danger' if full_bkg is None else 'success')
    return bkg_handle, bkg_message, full_bkg


@app.cell
def __(mo, tensor_maps):
    plot_elem = mo.ui.dropdown(label='Choose which parameter/element map you would like to visualize',
                               options=list(tensor_maps.keys()),
                               value=list(tensor_maps.keys())[-1])
    plot_elem.right()
    return plot_elem,


@app.cell
def __(mo, plot_elem, px, tensor_maps):
    elem_map = mo.ui.plotly(px.imshow(tensor_maps[plot_elem.value]))
    elem_map_shown = True
    elem_map
    return elem_map, elem_map_shown


@app.cell
def __(elem_map, plot_elem, tensor_maps):
    from math import floor, ceil

    if 'x' in elem_map.ranges and 'y' in elem_map.ranges:
        x_l = ceil(elem_map.ranges['x'][0])
        x_r = floor(elem_map.ranges['x'][1])
        y_l = ceil(elem_map.ranges['y'][0])
        y_r = floor(elem_map.ranges['y'][1])
    else:
        x_l, x_r = 0, tensor_maps[plot_elem.value].shape[1]
        y_l, y_r = 0, tensor_maps[plot_elem.value].shape[0]

    return ceil, floor, x_l, x_r, y_l, y_r


@app.cell
def __(
    e_dim_pos,
    energy_range,
    full_bkg,
    full_spec_fit,
    go,
    make_subplots,
    mo,
    np,
    px,
    spec_vol,
    x_l,
    x_r,
    y_l,
    y_r,
):
    spec_vol_t = np.moveaxis(spec_vol, e_dim_pos, -1)
    int_spec = spec_vol_t[y_l:y_r, x_l:x_r, energy_range[0]:energy_range[1]].sum(axis=(0, 1))
    int_spec_fit = full_spec_fit[y_l:y_r, x_l:x_r, :].sum(axis=(0, 1))
    int_bkg = full_bkg[y_l:y_r, x_l:x_r, :].sum(axis=(0, 1))

    fit_labels=['experiment', 'background', 'fitted']
    fit_fig = make_subplots(rows=2, cols=1)
    spec_x = np.linspace(0, int_spec.size - 1, int_spec.size)

    for i, spec in enumerate([int_spec, int_bkg, int_spec_fit+int_bkg]):
        fit_fig.add_trace(go.Scatter(x=spec_x, y=spec, mode='lines', name=fit_labels[i], line=dict(color=px.colors.qualitative.Plotly[i])), row=1, col=1)
        spec_log = np.log10(np.clip(spec, 0, None)+1)
        fit_fig.add_trace(go.Scatter(x=spec_x, y=spec_log, mode='lines', showlegend=False, line=dict(color=px.colors.qualitative.Plotly[i])), row=2, col=1)

    mo.ui.plotly(fit_fig)
    return (
        fit_fig,
        fit_labels,
        i,
        int_bkg,
        int_spec,
        int_spec_fit,
        spec,
        spec_log,
        spec_vol_t,
        spec_x,
    )


if __name__ == "__main__":
    app.run()
