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

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __(__file__):
    import sys, pickle
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.absolute()))

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
        pickle,
        plt,
        px,
        read_dataset,
        sys,
        torch,
    )


@app.cell
def __(mo, torch):
    (
        mo.callout(
            "To process spectra volume and generate element/parameter maps, GPU acceleration is needed.",
            kind="danger",
        )
        if not torch.cuda.is_available()
        else None
    )
    return


@app.cell
def __(mo):
    dataset = mo.ui.file_browser(
        filetypes=[".h5", ".h50", ".h51", ".h52", ".h53", ".h54", ".h55"],
        multiple=False,
    )
    mo.md(f"Please select the dataset file (h5 file) \n{dataset}")
    return (dataset,)


@app.cell
def __(mo):
    spec_vol_path = mo.ui.dropdown(
        ["MAPS/Spectra/mca_arr", "MAPS/mca_arr"],
        value="MAPS/mca_arr",
        label="Spectra volume location",
    )
    energy_dimension = mo.ui.dropdown(
        ["first", "middle", "last", "guess"], value="guess", label="Energy dimension"
    )
    dataset_button = mo.ui.run_button(label="Load")
    mo.hstack(
        [spec_vol_path, energy_dimension, dataset_button], justify="start", gap=1
    ).right()
    return dataset_button, energy_dimension, spec_vol_path


@app.cell
def __(dataset_button, e_dim_guess, energy_dimension, mo):
    mo.stop(not dataset_button.value)
    energy_dimension_guessed = True
    (
        mo.callout(f"Assume the {e_dim_guess} dimension is the energy", kind="info")
        if energy_dimension.value == "guess"
        else None
    )
    return (energy_dimension_guessed,)


@app.cell
def __(dataset_button, energy_dimension_guessed, mo):
    mo.stop(not dataset_button.value)
    param_csv = mo.ui.file(filetypes=[".csv"], multiple=False, kind="area")
    (
        mo.md(f"Please upload the saved parameter file (csv file) \n{param_csv}")
        if energy_dimension_guessed
        else None
    )
    return (param_csv,)


@app.cell
def __(mo, param_csv, pd):
    from io import StringIO

    if len(param_csv.value) > 0:
        param_df = pd.read_csv(StringIO(param_csv.value[0].contents.decode()))
        params_table = mo.ui.table(
            param_df,
            selection="single",
            label="Please select the row of parameters and elements",
        )
    params_table if len(param_csv.value) > 0 else None
    return StringIO, param_df, params_table


@app.cell
def __(int_spec_og, mo, params_table):
    energy_range = mo.ui.range_slider(
        start=0,
        stop=int_spec_og.shape[-1],
        step=1,
        label="Energy range",
        value=[50, 1450],
    )
    n_iterations = mo.ui.slider(
        start=100, stop=500, value=200, label="Number of iterations"
    )
    load_params_button = mo.ui.run_button(
        label="Load selected settings and fit integrated spectrum"
    )
    (
        mo.hstack(
            [energy_range, n_iterations, load_params_button], justify="start", gap=3
        ).right()
        if len(params_table.value) > 0
        else None
    )
    return energy_range, load_params_button, n_iterations


@app.cell
def __(fitted_bkg, fitted_spec, go, int_spec, make_subplots, mo, np, px):
    fit_labels = ["experiment", "background", "fitted"]
    fit_fig = make_subplots(rows=2, cols=1)
    spec_x = np.linspace(0, int_spec.size - 1, int_spec.size)

    for i, spec in enumerate([int_spec, fitted_bkg, fitted_spec + fitted_bkg]):
        fit_fig.add_trace(
            go.Scatter(
                x=spec_x,
                y=spec,
                mode="lines",
                name=fit_labels[i],
                line=dict(color=px.colors.qualitative.Plotly[i]),
            ),
            row=1,
            col=1,
        )
        spec_log = np.log10(np.clip(spec, 0, None) + 1)
        fit_fig.add_trace(
            go.Scatter(
                x=spec_x,
                y=spec_log,
                mode="lines",
                showlegend=False,
                line=dict(color=px.colors.qualitative.Plotly[i]),
            ),
            row=2,
            col=1,
        )

    init_fit_shown = True

    mo.ui.plotly(fit_fig)
    return fit_fig, fit_labels, i, init_fit_shown, spec, spec_log, spec_x


@app.cell
def __(
    energy_range,
    fit_spec,
    int_spec_og,
    load_params_button,
    mo,
    n_iterations,
    params_table,
):
    mo.stop(not load_params_button.value)
    elements_to_fit = params_table.value["elements"].item().split(",")
    fitting_params = [p for p in params_table.value if p != "elements"]
    param_vals = {k: p.item() for k, p in params_table.value.items() if k != "elements"}
    n_iter = n_iterations.value
    with mo.status.progress_bar(total=n_iter) as bar:
        fitted_tensors, fitted_spec, fitted_bkg, loss_trace = fit_spec(
            int_spec_og,
            energy_range.value,
            elements_to_fit=elements_to_fit,
            fitting_params=fitting_params,
            init_param_vals=param_vals,
            fixed_param_vals=param_vals,
            tune_params=False,
            n_iter=n_iter,
            status_updator=bar,
        )
    bar
    return (
        bar,
        elements_to_fit,
        fitted_bkg,
        fitted_spec,
        fitted_tensors,
        fitting_params,
        loss_trace,
        n_iter,
        param_vals,
    )


@app.cell
def __(init_fit_shown, mo):
    param_maps_button = mo.ui.run_button(
        label="Something is off, produce parameter maps"
    )
    elem_maps_button = mo.ui.run_button(
        label="Looks good, fit the whole spectra volume"
    )
    (
        mo.hstack([param_maps_button, elem_maps_button], justify="start", gap=2).right()
        if init_fit_shown
        else None
    )
    return elem_maps_button, param_maps_button


@app.cell
def __(dataset, mo, param_maps_button):
    mo.stop(not param_maps_button.value)
    param_result_file_name = "{}_fit_spec_vol_params_results.pickle".format(
        dataset.value[0].name
    )
    (
        mo.callout(
            "Parameter maps will be saved in {}".format(param_result_file_name),
            kind="info",
        )
        if True
        else None
    )
    return (param_result_file_name,)


@app.cell
def __(
    ceil,
    elements_to_fit,
    energy_range,
    fitting_params,
    mo,
    param_maps_button,
    param_result_file_name,
    param_vals,
    pickle,
    spec_vol_t,
):
    mo.stop(not param_maps_button.value)
    from maps_torch.opt import fit_spec_vol_params

    param_n_tile_side = 5
    param_tile_size = max(
        spec_vol_t.shape[0] // param_n_tile_side,
        spec_vol_t.shape[1] // param_n_tile_side,
    )
    param_x_tiles = ceil(spec_vol_t.shape[0] / param_tile_size)
    param_y_tiles = ceil(spec_vol_t.shape[1] / param_tile_size)
    param_total_tiles = param_x_tiles * param_y_tiles
    with mo.status.progress_bar(total=400 * param_total_tiles) as bar_params:
        (
            param_dict,
            param_tile_info,
            fitted_spec_params,
            bkg_vol_params,
            loss_vol_params,
        ) = fit_spec_vol_params(
            spec_vol_t,
            energy_range.value,
            elements_to_fit=elements_to_fit,
            fitting_params=fitting_params,
            init_param_vals=param_vals,
            fixed_param_vals={},
            init_amp=True,
            use_snip=True,
            use_step=True,
            use_tail=False,
            tile_size=param_tile_size,
            max_n_tile_side=param_n_tile_side,
            n_iter=400,
            save_fitted_spec=True,
            save_loss=True,
            save_bkg=True,
            status_updator=bar_params,
        )
        # Save results for fit_spec_vol_params
        with open(param_result_file_name, "wb") as f_params:
            pickle.dump(
                {
                    "param_dict": param_dict,
                    "tile_info": param_tile_info,
                    "fitted_spec": fitted_spec_params,
                    "bkg": bkg_vol_params,
                    "loss": loss_vol_params,
                    "energy_range": energy_range.value,
                    "elements_to_fit": elements_to_fit,
                },
                f_params,
            )
    bar_params
    return (
        bar_params,
        bkg_vol_params,
        f_params,
        fit_spec_vol_params,
        fitted_spec_params,
        loss_vol_params,
        param_dict,
        param_n_tile_side,
        param_tile_info,
        param_tile_size,
        param_total_tiles,
        param_x_tiles,
        param_y_tiles,
    )


@app.cell
def __(elem_maps_button, mo, torch):
    mo.stop(not elem_maps_button.value)
    (
        mo.callout(
            "To process spectra volume and generate element/parameter maps, GPU acceleration is needed.",
            kind="danger",
        )
        if not torch.cuda.is_available()
        else None
    )
    return


@app.cell
def __(dataset, elem_maps_button, mo, torch):
    mo.stop(not elem_maps_button.value or (not torch.cuda.is_available()))
    elem_result_file_name = "{}_fit_spec_vol_amps_results.pickle".format(
        dataset.value[0].name
    )
    mo.callout(
        "Elemental maps will be saved in {}".format(elem_result_file_name), kind="info"
    )
    return (elem_result_file_name,)


@app.cell
def __(
    ceil,
    elem_maps_button,
    elem_result_file_name,
    elements_to_fit,
    energy_range,
    mo,
    param_vals,
    pickle,
    spec_vol,
    spec_vol_t,
    torch,
):
    mo.stop(not elem_maps_button.value or (not torch.cuda.is_available()))
    from maps_torch.opt import fit_spec_vol_amps
    from maps_torch.util import estimate_gpu_tile_size

    elem_tile_size = estimate_gpu_tile_size(spec_vol.shape)
    elem_x_tiles = ceil(spec_vol_t.shape[0] / elem_tile_size)
    elem_y_tiles = ceil(spec_vol_t.shape[1] / elem_tile_size)
    elem_total_tiles = elem_x_tiles * elem_y_tiles
    with mo.status.progress_bar(total=200 * elem_total_tiles) as bar_elems:
        elem_dict, elem_tile_info, fitted_spec_elems, bkg_vol_elems, loss_vol_elems = (
            fit_spec_vol_amps(
                spec_vol_t,
                energy_range.value,
                elements_to_fit=elements_to_fit,
                param_vals=param_vals,
                init_amp=True,
                use_snip=True,
                use_step=True,
                use_tail=False,
                tile_size=elem_tile_size,
                n_iter=200,
                save_fitted_spec=True,
                save_loss=True,
                save_bkg=True,
                status_updator=bar_elems,
            )
        )
        # Save results for fit_spec_vol_params
        with open(elem_result_file_name, "wb") as f_elems:
            pickle.dump(
                {
                    "elem_dict": elem_dict,
                    "tile_info": elem_tile_info,
                    "fitted_spec": fitted_spec_elems,
                    "bkg": bkg_vol_elems,
                    "loss": loss_vol_elems,
                    "energy_range": energy_range.value,
                    "param_vals": param_vals,
                },
                f_elems,
            )
    bar_elems
    return (
        bar_elems,
        bkg_vol_elems,
        elem_dict,
        elem_tile_info,
        elem_tile_size,
        elem_total_tiles,
        elem_x_tiles,
        elem_y_tiles,
        estimate_gpu_tile_size,
        f_elems,
        fit_spec_vol_amps,
        fitted_spec_elems,
        loss_vol_elems,
    )


@app.cell
def __(energy_range, int_spec_og, int_spec_og_log):
    int_spec = int_spec_og[energy_range.value[0] : energy_range.value[1] + 1]
    int_spec_log = int_spec_og_log[energy_range.value[0] - 1 : energy_range.value[1]]
    return int_spec, int_spec_log


@app.cell
def __(np, spec_vol_t):
    int_spec_og = spec_vol_t.sum(axis=(0, 1))
    int_spec_og_log = np.log10(np.clip(int_spec_og, 0, None) + 1)
    return int_spec_og, int_spec_og_log


@app.cell
def __(energy_dimension, np, spec_vol):
    def guess_energy_dim_pos(spec_vol):
        assert len(spec_vol.shape) == 3, "The spectra volume must have 3 dimensions"
        return np.argmax(spec_vol.shape)

    if energy_dimension.value == "guess":
        e_dim_pos = guess_energy_dim_pos(spec_vol)
        e_dim_guess = ["first", "middle", "last"][e_dim_pos]
    else:
        e_dim_pos = {"first": 0, "middle": 1, "last": 2}[energy_dimension.value]

    spec_vol_t = np.moveaxis(spec_vol, e_dim_pos, -1)
    return e_dim_guess, e_dim_pos, guess_energy_dim_pos, spec_vol_t


@app.cell
def __(dataset, dataset_button, mo, read_dataset, spec_vol_path):
    mo.stop(not dataset_button.value)
    dataset_dict = read_dataset(dataset.value[0].path, spec_vol_key=spec_vol_path.value)
    spec_vol = dataset_dict["spec_vol"]
    return dataset_dict, spec_vol


if __name__ == "__main__":
    app.run()
