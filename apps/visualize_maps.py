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
    import os, sys, pickle
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.absolute()))

    from math import floor, ceil
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
        ceil,
        floor,
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
    mo.callout(
        mo.md(
            r"""If you encounter error messages about figure size being too large. You can increase the default image size limit by:

    * **Linux/Mac** Run command `export MARIMO_OUTPUT_MAX_BYTES=10_000_000` on the terminal before running the notebook, or add it to your shell profile
    * **Windows** Open the start menu, search for the "Advanced System Settings" control panel and click on it, click on the "Environment Variables" button toward the bottom of the screen, follow the prompts to add the `MARIMO_OUTPUT_MAX_BYTES` variable to the user table, set the value to be `10_000_000`
    """
        ),
        kind="warn",
    )
    return


@app.cell
def __(mo):
    dataset = mo.ui.file_browser(
        filetypes=[".h5", ".h50", ".h51", ".h52", ".h53", ".h54", ".h55"],
        multiple=False,
    )
    mo.md(f"Please select the original dataset file (h5 file) \n{dataset}")
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
        [spec_vol_path, energy_dimension, dataset_button], justify="start", gap=2
    ).right()
    return dataset_button, energy_dimension, spec_vol_path


@app.cell
def __(dataset_button, e_dim_guess, mo):
    mo.stop(not dataset_button.value)
    mo.callout(f"Assume the {e_dim_guess} dimension is the energy", kind="info")
    return


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
def __(dataset, dataset_button, mo, spec_vol_path):
    mo.stop(not dataset_button.value)
    from maps_torch.io import read_dataset

    dataset_dict = read_dataset(dataset.value[0].path, spec_vol_key=spec_vol_path.value)
    spec_vol = dataset_dict["spec_vol"]
    return dataset_dict, read_dataset, spec_vol


@app.cell
def __(mo, spec_vol):
    output_dir = mo.ui.file_browser(
        filetypes=[".pkl", ".pickle"],
        multiple=False,
        label="Please select the pickled anslysis result file",
    )
    output_dir if spec_vol is not None else None
    return (output_dir,)


@app.cell
def __(mo, spec_vol):
    load_output_button = mo.ui.run_button(label="Load maps")
    maps_type = mo.ui.dropdown(
        options=["element", "parameter"], value="element", label="maps type"
    )
    (
        mo.hstack([maps_type, load_output_button], justify="start", gap=2).right()
        if spec_vol is not None
        else None
    )
    return load_output_button, maps_type


@app.cell
def __(load_output_button, maps_type, mo, output_dir, pickle):
    mo.stop(not load_output_button.value)

    if maps_type.value == "element":
        with open(output_dir.value[0].path, "rb") as f:
            amps_results = pickle.load(f)
        elem_dict = amps_results["elem_dict"]
        tile_info_amps = amps_results["tile_info"]
        fitted_spec_amps = amps_results["fitted_spec"]
        bkg_amps = amps_results["bkg"]
        loss_amps = amps_results["loss"]
        energy_range = amps_results["energy_range"]
        param_vals = amps_results["param_vals"]
    else:
        with open(output_dir.value[0].path, "rb") as f:
            params_results = pickle.load(f)
        param_dict = params_results["param_dict"]
        tile_info_params = params_results["tile_info"]
        fitted_spec_params = params_results["fitted_spec"]
        bkg_params = params_results["bkg"]
        loss_params = params_results["loss"]
        energy_range = params_results["energy_range"]
        elements_to_fit = params_results["elements_to_fit"]
    return (
        amps_results,
        bkg_amps,
        bkg_params,
        elem_dict,
        elements_to_fit,
        energy_range,
        f,
        fitted_spec_amps,
        fitted_spec_params,
        loss_amps,
        loss_params,
        param_dict,
        param_vals,
        params_results,
        tile_info_amps,
        tile_info_params,
    )


@app.cell
def __(elem_dict, maps_type, mo):
    if maps_type.value == "element":
        plot_elem = mo.ui.dropdown(
            label="Choose which element map you would like to visualize",
            options=list(elem_dict.keys()),
            value=list(elem_dict.keys())[0],
        )
    plot_elem.right() if maps_type.value == "element" else None
    return (plot_elem,)


@app.cell
def __(elem_dict, maps_type, mo, plot_elem, px):
    if maps_type.value == "element":
        elem_map = mo.ui.plotly(
            px.imshow(elem_dict[plot_elem.value], color_continuous_scale="Viridis")
        )
        elem_map_shown = True
    elem_map if maps_type.value == "element" else None
    return elem_map, elem_map_shown


@app.cell
def __(ceil, elem_dict, elem_map, floor, maps_type, plot_elem):
    if maps_type.value == "element":
        if "x" in elem_map.ranges and "y" in elem_map.ranges:
            x_l_e = ceil(elem_map.ranges["x"][0])
            x_r_e = floor(elem_map.ranges["x"][1])
            y_l_e = ceil(elem_map.ranges["y"][0])
            y_r_e = floor(elem_map.ranges["y"][1])
        else:
            x_l_e, x_r_e = 0, elem_dict[plot_elem.value].shape[1]
            y_l_e, y_r_e = 0, elem_dict[plot_elem.value].shape[0]
    return x_l_e, x_r_e, y_l_e, y_r_e


@app.cell
def __(
    bkg_amps,
    energy_range,
    fitted_spec_amps,
    go,
    make_subplots,
    maps_type,
    np,
    px,
    spec_vol_t,
    x_l_e,
    x_r_e,
    y_l_e,
    y_r_e,
):
    if maps_type.value == "element" and fitted_spec_amps is not None:
        int_spec_e = spec_vol_t[
            y_l_e:y_r_e, x_l_e:x_r_e, energy_range[0] : energy_range[1]
        ].sum(axis=(0, 1))
        int_spec_fit_e = fitted_spec_amps[y_l_e:y_r_e, x_l_e:x_r_e, :].sum(axis=(0, 1))
        int_bkg_e = bkg_amps[y_l_e:y_r_e, x_l_e:x_r_e, :].sum(axis=(0, 1))

        fit_labels_e = ["experiment", "background", "fitted"]
        fit_fig_e = make_subplots(rows=2, cols=1)
        spec_x_e = np.linspace(0, int_spec_e.size - 1, int_spec_e.size)

        for i_e, spec_e in enumerate(
            [int_spec_e, int_bkg_e, int_spec_fit_e + int_bkg_e]
        ):
            fit_fig_e.add_trace(
                go.Scatter(
                    x=spec_x_e,
                    y=spec_e,
                    mode="lines",
                    name=fit_labels_e[i_e],
                    line=dict(color=px.colors.qualitative.Plotly[i_e]),
                ),
                row=1,
                col=1,
            )
            spec_log_e = np.log10(np.clip(spec_e, 0, None) + 1)
            fit_fig_e.add_trace(
                go.Scatter(
                    x=spec_x_e,
                    y=spec_log_e,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=px.colors.qualitative.Plotly[i_e]),
                ),
                row=2,
                col=1,
            )
    return (
        fit_fig_e,
        fit_labels_e,
        i_e,
        int_bkg_e,
        int_spec_e,
        int_spec_fit_e,
        spec_e,
        spec_log_e,
        spec_x_e,
    )


@app.cell
def __(fit_fig_e, maps_type, mo):
    mo.ui.plotly(fit_fig_e) if maps_type.value == "element" else None
    return


@app.cell
def __(maps_type, mo, param_dict):
    if maps_type.value == "parameter":
        plot_param = mo.ui.dropdown(
            label="Choose which parameter map you would like to visualize",
            options=list(param_dict.keys()),
            value=list(param_dict.keys())[0],
        )
    plot_param.right() if maps_type.value == "parameter" else None
    return (plot_param,)


@app.cell
def __(maps_type, mo, param_dict, plot_param, px):
    if maps_type.value == "parameter":
        param_map = mo.ui.plotly(
            px.imshow(param_dict[plot_param.value], color_continuous_scale="Viridis")
        )
        param_map_shown = True
    param_map if maps_type.value == "parameter" else None
    return param_map, param_map_shown


@app.cell
def __(maps_type, mo, tile_info_params):
    if maps_type.value == "parameter":
        ix_p = mo.ui.dropdown(
            options=[str(ix) for ix in range(tile_info_params["x_tiles"])],
            value="0",
            label="tile y locator",
        )
        iy_p = mo.ui.dropdown(
            options=[str(iy) for iy in range(tile_info_params["y_tiles"])],
            value="0",
            label="tile x locator",
        )
    return ix_p, iy_p


@app.cell
def __(ix_p, iy_p, maps_type, spec_vol_t, tile_info_params):
    if maps_type.value == "parameter":
        x_l_p = int(ix_p.value) * tile_info_params["tile_size"]
        x_r_p = min(
            (int(ix_p.value) + 1) * tile_info_params["tile_size"],
            spec_vol_t.shape[1] + 1,
        )
        y_l_p = int(iy_p.value) * tile_info_params["tile_size"]
        y_r_p = min(
            (int(iy_p.value) + 1) * tile_info_params["tile_size"],
            spec_vol_t.shape[0] + 1,
        )
    return x_l_p, x_r_p, y_l_p, y_r_p


@app.cell
def __(ix_p, iy_p, maps_type, mo, x_l_p, x_r_p, y_l_p, y_r_p):
    (
        mo.hstack(
            [
                iy_p,
                ix_p,
                mo.md(f"Selecting [{y_l_p}-{y_r_p}) * [{x_l_p}-{x_r_p}) tile"),
            ],
            justify="start",
            gap=2,
        ).right()
        if maps_type.value == "parameter"
        else None
    )
    return


@app.cell
def __(
    bkg_params,
    energy_range,
    fitted_spec_params,
    go,
    ix_p,
    iy_p,
    make_subplots,
    maps_type,
    np,
    px,
    spec_vol_t,
    x_l_p,
    x_r_p,
    y_l_p,
    y_r_p,
):
    if maps_type.value == "parameter" and fitted_spec_params is not None:
        int_spec_p = spec_vol_t[
            x_l_p:x_r_p, y_l_p:y_r_p, energy_range[0] : energy_range[1]
        ].sum(axis=(0, 1))
        int_spec_fit_p = fitted_spec_params[int(ix_p.value), int(iy_p.value)]
        int_bkg_p = bkg_params[int(ix_p.value), int(iy_p.value)]

        fit_labels_p = ["experiment", "background", "fitted"]
        fit_fig_p = make_subplots(rows=2, cols=1)
        spec_x_p = np.linspace(0, int_spec_p.size - 1, int_spec_p.size)

        for i_p, spec_p in enumerate(
            [int_spec_p, int_bkg_p, int_spec_fit_p + int_bkg_p]
        ):
            fit_fig_p.add_trace(
                go.Scatter(
                    x=spec_x_p,
                    y=spec_p,
                    mode="lines",
                    name=fit_labels_p[i_p],
                    line=dict(color=px.colors.qualitative.Plotly[i_p]),
                ),
                row=1,
                col=1,
            )
            spec_log_p = np.log10(np.clip(spec_p, 0, None) + 1)
            fit_fig_p.add_trace(
                go.Scatter(
                    x=spec_x_p,
                    y=spec_log_p,
                    mode="lines",
                    showlegend=False,
                    line=dict(color=px.colors.qualitative.Plotly[i_p]),
                ),
                row=2,
                col=1,
            )
    return (
        fit_fig_p,
        fit_labels_p,
        i_p,
        int_bkg_p,
        int_spec_fit_p,
        int_spec_p,
        spec_log_p,
        spec_p,
        spec_x_p,
    )


@app.cell
def __(fit_fig_p, maps_type, mo):
    mo.ui.plotly(fit_fig_p) if maps_type.value == "parameter" else None
    return


if __name__ == "__main__":
    app.run()
