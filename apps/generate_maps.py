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
    import sys
    from pathlib import Path; sys.path.append(str(Path(__file__).parent.parent.absolute()))

    from math import floor, ceil, acos, pi
    import numpy as np
    import pandas as pd
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import marimo as mo
    import torch

    from maps_torch.io import read_dataset
    from maps_torch.opt import fit_spec
    return (
        Path,
        acos,
        ceil,
        fit_spec,
        floor,
        go,
        make_subplots,
        mo,
        np,
        pd,
        pi,
        plt,
        px,
        read_dataset,
        sys,
        torch,
    )


@app.cell
def __(mo):
    mo.callout('To process spectra volume and generate element/parameter maps, GPU acceleration is needed.', kind='warn')
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
    elem_path = mo.ui.dropdown(['MAPS/channel_names'], 
                                   value='MAPS/channel_names',
                                  label='Energy channel names location')
    dataset_button = mo.ui.run_button(label='Load')
    mo.hstack([spec_vol_path, energy_dimension, elem_path, dataset_button], justify='start', gap=1).right()
    return dataset_button, elem_path, energy_dimension, spec_vol_path


@app.cell
def __(dataset_button, e_dim_guess, energy_dimension, mo):
    mo.stop(not dataset_button.value)
    mo.callout(f'Assume the {e_dim_guess} dimension is the energy', kind='info') if energy_dimension.value == 'guess' else None
    return


@app.cell
def __(dataset_button, int_spec_og, mo):
    mo.stop(not dataset_button.value)
    energy_range = mo.ui.range_slider(start=0, stop=int_spec_og.shape[-1], step=1, label='Energy range', value=[50, 1450], full_width=True)
    energy_range
    return energy_range,


@app.cell
def __(go, int_spec, int_spec_log, make_subplots, mo, np):
    int_spec_fig = make_subplots(rows=2, cols=1)

    # Add trace for the 1D data
    int_spec_fig.append_trace(go.Scatter(x=np.arange(len(int_spec)), y=int_spec, mode='lines', name='Photon counts'), row=1, col=1)
    int_spec_fig.append_trace(go.Scatter(x=np.arange(len(int_spec)), y=int_spec_log, mode='lines', name='Log scale'), row=2, col=1)

    int_spec_fig.update_layout(showlegend=False)

    mo.ui.plotly(int_spec_fig)
    return int_spec_fig,


@app.cell
def __(elems, mo):
    from maps_torch.default import default_fitting_elems

    elem_checkboxes = {}
    for e in default_fitting_elems:
        elem_toggle = True if e in elems else False
        elem_checkboxes[e] = mo.ui.checkbox(label=e, value=elem_toggle) 
    elem_selection = mo.hstack([elem_checkboxes[e] for e in default_fitting_elems], wrap=True)
    elem_selection_shown = True
    elem_selection
    return (
        default_fitting_elems,
        e,
        elem_checkboxes,
        elem_selection,
        elem_selection_shown,
        elem_toggle,
    )


@app.cell
def __(dataset_button, elem_selection_shown, mo):
    mo.stop(not dataset_button.value)
    param_csv = mo.ui.file(filetypes=['.csv'], multiple=False, kind='area')
    mo.md(f"Please select the saved parameter file (csv file) \n{param_csv}") if elem_selection_shown else None
    return param_csv,


@app.cell
def __(mo, param_csv, pd):
    from io import StringIO
    if len(param_csv.value)>0:
        param_df = pd.read_csv(StringIO(param_csv.value[0].contents.decode()))
        params_table = mo.ui.table(param_df, selection='single')
    params_table if len(param_csv.value)>0 else None
    return StringIO, param_df, params_table


@app.cell
def __(mo, param_csv, params_table):
    load_params_button = mo.ui.run_button(label='Load selected parameters to see fitting result')
    load_params_button.right() if len(param_csv.value)>0 and len(params_table.value)>0 else None
    return load_params_button,


@app.cell
def __(fitted_bkg, fitted_spec, go, int_spec, make_subplots, mo, np, px):
    fit_labels=['experiment', 'background', 'fitted']
    fit_fig = make_subplots(rows=2, cols=1)
    spec_x = np.linspace(0, int_spec.size - 1, int_spec.size)

    for i, spec in enumerate([int_spec, fitted_bkg, fitted_spec+fitted_bkg]):
        fit_fig.add_trace(go.Scatter(x=spec_x, y=spec, mode='lines', name=fit_labels[i], line=dict(color=px.colors.qualitative.Plotly[i])), row=1, col=1)
        spec_log = np.log10(np.clip(spec, 0, None)+1)
        fit_fig.add_trace(go.Scatter(x=spec_x, y=spec_log, mode='lines', showlegend=False, line=dict(color=px.colors.qualitative.Plotly[i])), row=2, col=1)

    init_fit_shown = True

    mo.ui.plotly(fit_fig)
    return fit_fig, fit_labels, i, init_fit_shown, spec, spec_log, spec_x


@app.cell
def __(init_fit_shown, mo):
    init_amp_checkbox = mo.ui.checkbox(label='Initialize amplitudes', value=True)
    use_snip_checkbox = mo.ui.checkbox(label='Use SNIP background', value=True)
    use_step_checkbox = mo.ui.checkbox(label='Modify pearks with step', value=True)
    loss_selection = mo.ui.dropdown(['mse', 'l1'], value='mse', label='loss')
    optimizer_selection = mo.ui.dropdown(['adam', 'sgd'], value='adam', label='optimizer')
    mo.hstack([init_amp_checkbox, use_snip_checkbox, use_step_checkbox, loss_selection, optimizer_selection], justify='start', gap=0.5) if init_fit_shown else None
    return (
        init_amp_checkbox,
        loss_selection,
        optimizer_selection,
        use_snip_checkbox,
        use_step_checkbox,
    )


@app.cell
def __(init_fit_shown, mo, spec_vol_t):
    from maps_torch.util import optimize_parallelization

    tile_size_guess, n_workers_guess = optimize_parallelization(spec_vol_t.shape, device='cuda')
    mo.md(f"Estimated maximum tile size: {tile_size_guess}, number of available GPU workers: {n_workers_guess}") if init_fit_shown else None
    return n_workers_guess, optimize_parallelization, tile_size_guess


@app.cell
def __(mo, n_workers_guess, tile_size_guess):
    tune_params_checkbox = mo.ui.checkbox(label='Tune each tile\'s parameters', value=True)
    tile_size_slider = mo.ui.slider(start=32, stop=tile_size_guess, step=32, value=tile_size_guess, label='Tile size')
    n_workers_slider = mo.ui.slider(start=1, stop=n_workers_guess, step=1, value=n_workers_guess, label='Parallel workers')
    return n_workers_slider, tile_size_slider, tune_params_checkbox


@app.cell
def __(
    init_fit_shown,
    mo,
    n_workers_slider,
    tile_size_slider,
    tune_params_checkbox,
):
    iter_slider = mo.ui.slider(value=tile_size_slider.value//32*100, start=100, stop=500, step=50, label='Iterations each tile')
    mo.hstack([tune_params_checkbox, tile_size_slider, n_workers_slider, iter_slider], justify='start', gap=0.5) if init_fit_shown else None
    return iter_slider,


@app.cell
def __(np, spec_vol_t, tile_size_slider):
    n_tiles_x = np.ceil(spec_vol_t.shape[1] / tile_size_slider.value).astype(int)
    n_tiles_y = np.ceil(spec_vol_t.shape[0] / tile_size_slider.value).astype(int)
    n_tiles = n_tiles_x * n_tiles_y
    return n_tiles, n_tiles_x, n_tiles_y


@app.cell
def __(iter_slider, mo, n_tiles, n_tiles_x, n_tiles_y):
    mo.md(f"Current selection results in {n_tiles_x} * {n_tiles_y} = {n_tiles} tiles, {iter_slider.value} * {n_tiles} = {iter_slider.value*n_tiles} iterations.")
    return


@app.cell
def __(init_fit_shown, mo, torch):
    run_button = mo.ui.run_button()
    run_button.right() if init_fit_shown and torch.cuda.is_available() else None
    return run_button,


@app.cell
def __(mo, torch):
    mo.callout('Torch reports GPU is not available.', kind='danger') if not torch.cuda.is_available() else None
    return


@app.cell
def __(mo, run_button):
    mo.stop(not run_button.value)
    mo.callout('This may take a while depending on the computing resource available...', kind='warn')
    return


@app.cell
def __(
    elem_checkboxes,
    energy_range,
    init_amp_checkbox,
    iter_slider,
    loss_selection,
    mo,
    n_workers_slider,
    optimizer_selection,
    params_table,
    run_button,
    spec_vol_t,
    tile_size_slider,
    tune_params_checkbox,
    use_snip_checkbox,
    use_step_checkbox,
):
    mo.stop(not run_button.value)
    from maps_torch.opt import fit_spec_vol_parallel

    with mo.redirect_stderr():
        tensor_maps, full_spec_fit, full_bkg, full_loss_vol = fit_spec_vol_parallel(
            spec_vol_t,
            energy_range.value,
            elements_to_fit=[k for k,v in elem_checkboxes.items() if v.value],
            fitting_params=[p for p in params_table.value],
            init_param_vals={k:p.item() for k,p in params_table.value.items()},
            tune_params=tune_params_checkbox.value,
            init_amp=init_amp_checkbox.value,
            use_snip=use_snip_checkbox.value,
            use_step=use_step_checkbox.value,
            loss=loss_selection.value,
            optimizer=optimizer_selection.value,
            n_iter=iter_slider.value,
            device='cuda',
            tile_size=tile_size_slider.value,
            n_workers=n_workers_slider.value
        )
    return (
        fit_spec_vol_parallel,
        full_bkg,
        full_loss_vol,
        full_spec_fit,
        tensor_maps,
    )


@app.cell
def __(full_bkg, full_loss_vol, full_spec_fit, mo, tensor_maps):
    save_files_button = mo.ui.run_button(label='Save files')
    save_files_button.right() if ((tensor_maps is not None) and (full_spec_fit is not None) and (full_bkg is not None) and (full_loss_vol is not None)) else None
    return save_files_button,


@app.cell
def __(
    energy_range,
    full_bkg,
    full_loss_vol,
    full_spec_fit,
    mo,
    save_files_button,
    tensor_maps,
):
    mo.stop(not save_files_button.value)
    import os, pickle

    if not os.path.exists('output'):
        os.makedirs('output')

    final_message = ''
    try:
        with open('output/energy_range.txt', 'w') as handle:
            for value in energy_range.value:
                handle.write(f"{value}\n")
        final_message += '* Energy range successfully saved in output folder\n'
    except Exception as e:
        final_message += f'* Failed to save energy_range: {e}\n'
        
    try:
        with open('output/tensor_maps.pkl', 'wb') as handle:
            pickle.dump(tensor_maps, handle)
        final_message += '* Maps successfully saved in output folder\n'
    except Exception as e:
        final_message += f'* Failed to save maps: {e}\n'

    try:
        with open('output/full_spec_fit.pkl', 'wb') as handle:
            pickle.dump(full_spec_fit, handle)
        final_message += '* Fitted spectra volume successfully saved in output folder\n'
    except Exception as e:
        final_message += f'* Failed to save fitted spectra volume: {e}\n'

    try:
        with open('output/full_bkg.pkl', 'wb') as handle:
            pickle.dump(full_bkg, handle)
        final_message += '* Fitted background successfully saved in output folder\n'
    except Exception as e:
        final_message += f'* Failed to save fitted background: {e}\n'

    try:
        with open('output/full_loss_vol.pkl', 'wb') as handle:
            pickle.dump(full_loss_vol, handle)
        final_message += '* Training loss trace successfully successfully saved in output folder\n'
    except Exception as e:
        final_message += f'* Failed to save training loss trace: {e}\n'
    return final_message, handle, os, pickle, value


@app.cell
def __(final_message, mo):
    final_callout_type = 'danger' if 'Failed' in final_message else 'success'
    mo.callout(mo.md(final_message), kind=final_callout_type)
    return final_callout_type,


@app.cell
def __(
    elem_checkboxes,
    energy_range,
    fit_spec,
    int_spec_og,
    load_params_button,
    mo,
    params_table,
):
    mo.stop(not load_params_button.value)
    fitted_tensors, fitted_spec, fitted_bkg, loss_trace = fit_spec(
        int_spec_og,
        energy_range.value,
        elements_to_fit=[k for k,v in elem_checkboxes.items() if v.value],
        fitting_params=[p for p in params_table.value],
        init_param_vals={k:p.item() for k,p in params_table.value.items()},
        fixed_param_vals={k:p.item() for k,p in params_table.value.items()},
        tune_params=False,
    )
    return fitted_bkg, fitted_spec, fitted_tensors, loss_trace


@app.cell
def __(energy_range, int_spec_og, int_spec_og_log):
    int_spec = int_spec_og[energy_range.value[0]:energy_range.value[1]+1]
    int_spec_log = int_spec_og_log[energy_range.value[0]-1:energy_range.value[1]]
    return int_spec, int_spec_log


@app.cell
def __(np, spec_vol_t):
    int_spec_og = spec_vol_t.sum(axis=(0, 1))
    int_spec_og_log = np.log10(np.clip(int_spec_og, 0, None)+1)
    return int_spec_og, int_spec_og_log


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

    spec_vol_t = np.moveaxis(spec_vol, e_dim_pos, -1)
    return e_dim_guess, e_dim_pos, guess_energy_dim_pos, spec_vol_t


@app.cell
def __(
    dataset,
    dataset_button,
    elem_path,
    mo,
    read_dataset,
    spec_vol_path,
):
    mo.stop(not dataset_button.value)
    dataset_dict = read_dataset(dataset.value[0].path, fit_elem_key=elem_path.value, spec_vol_key=spec_vol_path.value)
    spec_vol = dataset_dict['spec_vol']
    elems = dataset_dict['elems']
    return dataset_dict, elems, spec_vol


if __name__ == "__main__":
    app.run()
