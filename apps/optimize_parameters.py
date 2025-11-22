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

__generated_with = "0.10.2"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import sys
    from math import floor, ceil, acos, pi
    import numpy as np
    import plotly.express as px
    import marimo as mo

    from mapstorch.io import read_dataset
    from mapstorch.util import PeriodicTableWidget
    from mapstorch.default import (
        default_fitting_elems,
        unsupported_elements,
        supported_elements_mapping,
    )

    return (
        PeriodicTableWidget,
        acos,
        ceil,
        default_fitting_elems,
        floor,
        mo,
        np,
        pi,
        px,
        read_dataset,
        supported_elements_mapping,
        sys,
        unsupported_elements,
    )


@app.cell
def _(mo):
    dataset = mo.ui.file_browser(
        filetypes=[".h5", ".h50", ".h51", ".h52", ".h53", ".h54", ".h55"],
        multiple=False,
    )
    mo.md(f"Please select the dataset file (h5 file) \n{dataset}")
    return (dataset,)


@app.cell
def _(mo):
    int_spec_path = mo.ui.dropdown(
        ["MAPS/int_spec", "MAPS/Spectra/Integrateds_Spectra/Spectra"],
        value="MAPS/int_spec",
        label="Integrated spectrum location",
    )
    elem_path = mo.ui.dropdown(
        ["MAPS/channel_names"],
        value="MAPS/channel_names",
        label="Energy channel names location",
    )
    dataset_button = mo.ui.run_button(label="Load")
    mo.hstack(
        [int_spec_path, elem_path, dataset_button], justify="start", gap=1
    ).right()
    return dataset_button, elem_path, int_spec_path


@app.cell
def _(int_spec_og, mo):
    energy_range = mo.ui.range_slider(
        start=0,
        stop=int_spec_og.shape[-1] - 1,
        step=1,
        label="Energy range",
        value=[50, 1450],
        full_width=True,
    )
    return (energy_range,)


@app.cell
def _(energy_range, int_spec_og, mo, peaks):
    incident_energy_slider = mo.ui.slider(
        start=6,
        stop=18,
        step=0.01,
        value=12,
        label="Incident Energy (keV)",
        full_width=True,
    )
    compton_peak_value = (
        (int_spec_og.shape[-1] - 1) // 2 if len(peaks) < 8 else peaks[-2]
    )
    compton_peak_slider = mo.ui.slider(
        start=0,
        stop=int_spec_og.shape[-1] - 1,
        step=1,
        value=compton_peak_value,
        label="Compton Peak Position",
        full_width=True,
    )
    elastic_peak_value = (
        (int_spec_og.shape[-1] - 1) // 1.9 if len(peaks) < 8 else peaks[-1]
    )
    elastic_peak_slider = mo.ui.slider(
        start=0,
        stop=int_spec_og.shape[-1] - 1,
        step=1,
        value=elastic_peak_value,
        label="Elastic Peak Position",
        full_width=True,
    )
    mo.vstack(
        [incident_energy_slider, energy_range, compton_peak_slider, elastic_peak_slider]
    )
    return (
        compton_peak_slider,
        compton_peak_value,
        elastic_peak_slider,
        elastic_peak_value,
        incident_energy_slider,
    )


@app.cell
def _(
    compton_peak_slider,
    elastic_peak_slider,
    int_spec,
    int_spec_log,
    mo,
    np,
    peaks,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    int_spec_fig = make_subplots(rows=2, cols=1)

    # Add trace for the 1D data
    int_spec_fig.append_trace(
        go.Scatter(
            x=np.arange(len(int_spec)), y=int_spec, mode="lines", name="Photon counts"
        ),
        row=1,
        col=1,
    )
    int_spec_fig.append_trace(
        go.Scatter(
            x=np.arange(len(int_spec)), y=int_spec_log, mode="lines", name="Log scale"
        ),
        row=2,
        col=1,
    )
    int_spec_fig.append_trace(
        go.Scatter(
            x=peaks,
            y=int_spec[peaks],
            mode="markers",
            marker_color="#00cc96",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    int_spec_fig.append_trace(
        go.Scatter(
            x=peaks,
            y=int_spec_log[peaks],
            mode="markers",
            name="Peaks",
            marker_color="#00cc96",
        ),
        row=2,
        col=1,
    )

    # Add a vertical line to mark a position within the range
    int_spec_fig.add_vline(
        x=compton_peak_slider.value, line_width=1, line_color="#ab63fa"
    )
    int_spec_fig.add_vline(
        x=elastic_peak_slider.value, line_width=1, line_color="#ffa15a"
    )

    int_spec_fig.update_layout(showlegend=False)

    mo.ui.plotly(int_spec_fig)
    return go, int_spec_fig, make_subplots


@app.cell
def _(configs, elem_selection, mo, param_selection):
    control_panel = mo.accordion(
        {
            "Elements": elem_selection.center(),
            "Parameters": param_selection,
            "Configs": configs,
        },
        multiple=True,
    )
    control_panel_shown = True
    control_panel
    return control_panel, control_panel_shown


@app.cell
def _(control_panel_shown, mo):
    run_button = mo.ui.run_button()
    run_button.right() if control_panel_shown else None
    return (run_button,)


@app.cell
def _(
    device_selection,
    elem_checkboxes,
    energy_range,
    fit_indices_list,
    init_amp_checkbox,
    int_spec_og,
    iter_slider,
    loss_selection,
    mo,
    optimizer_selection,
    param_checkboxes,
    run_button,
    use_snip_checkbox,
):
    mo.stop(not run_button.value)
    from mapstorch.opt import fit_spec

    n_iter = iter_slider.value
    with mo.status.progress_bar(total=n_iter) as bar:
        fitted_tensors, fitted_spec, fitted_bkg, loss_trace = fit_spec(
            int_spec_og,
            energy_range.value,
            elements_to_fit=[k for k, v in elem_checkboxes.items() if v.value],
            fitting_params=[k for k, v in param_checkboxes.items() if v[0].value],
            init_param_vals={
                k: float(v[2].value) for k, v in param_checkboxes.items() if v[0].value
            },
            fixed_param_vals={
                k: float(v[2].value)
                for k, v in param_checkboxes.items()
                if v[0].value and v[1].value
            },
            indices=fit_indices_list[-1],
            tune_params=True,
            init_amp=init_amp_checkbox.value,
            use_snip=use_snip_checkbox.value,
            loss=loss_selection.value,
            optimizer=optimizer_selection.value,
            n_iter=n_iter,
            progress_bar=False,
            device=device_selection.value,
            status_updator=bar,
        )
    return (
        bar,
        fit_spec,
        fitted_bkg,
        fitted_spec,
        fitted_tensors,
        loss_trace,
        n_iter,
    )


@app.cell
def _(fitted_bkg, fitted_spec, go, int_spec, make_subplots, mo, np, px):
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

    mo.ui.plotly(fit_fig)
    return fit_fig, fit_labels, i, spec, spec_log, spec_x


@app.cell
def _(elem_checkboxes, fitted_tensors, go, make_subplots, mo):
    target_elems = [k for k, v in elem_checkboxes.items() if v.value]
    amps = {p: fitted_tensors[p].item() for p in target_elems}
    amps = dict(sorted(amps.items(), key=lambda item: item[1]))

    amp_fig = make_subplots(rows=1, cols=2)

    amp_fig.add_trace(
        go.Bar(
            x=[10**v for v in amps.values()],
            y=list(amps.keys()),
            orientation="h",
            name="Photon counts",
        ),
        row=1,
        col=1,
    )
    amp_fig.add_trace(go.Bar(x=list(amps.values()), name="Log scale"), row=1, col=2)
    amp_fig.update_yaxes(showticklabels=False, row=1, col=2)
    amp_fig.update_layout(showlegend=False)

    results_shown = True

    mo.ui.plotly(amp_fig)
    return amp_fig, amps, results_shown, target_elems


@app.cell
def _(
    confirm_range_button,
    energy_level_slider,
    focus_target_switch,
    mo,
    range_fig,
    results_shown,
):
    (
        mo.accordion(
            {
                "Target ranges": mo.vstack(
                    [
                        range_fig,
                        mo.hstack(
                            [
                                focus_target_switch,
                                energy_level_slider,
                                confirm_range_button,
                            ]
                        ),
                    ]
                )
            }
        )
        if results_shown
        else None
    )
    return


@app.cell
def _(confirm_range_button, elem_peak_indices, fit_indices_list, mo):
    mo.stop(not confirm_range_button.value)

    fit_indices_list[-1] = elem_peak_indices
    mo.callout(
        "Target ranges have been updated. Please select parameters to re-run the fitting process.",
        kind="success",
    )
    return


@app.cell
def _(dataset, elem_checkboxes, fitted_tensors, mo, params_record):
    import pandas as pd
    import datetime

    for par, l in params_record.items():
        if par == "elements":
            l.append(",".join([k for k, v in elem_checkboxes.items() if v.value]))
        else:
            l.append(fitted_tensors[par].item())
    today = datetime.date.today()
    today_string = today.strftime("%Y-%m-%d")
    table_label = dataset.value[0].name + " parameter tuning record " + today_string
    params_table = mo.ui.table(
        pd.DataFrame(params_record),
        selection="single",
        label=table_label,
        show_download=False,
    )
    params_table
    return (
        datetime,
        l,
        par,
        params_table,
        pd,
        table_label,
        today,
        today_string,
    )


@app.cell
def _(load_params_button, mo, params_table, save_params_button):
    (
        mo.hstack([save_params_button, load_params_button]).right()
        if len(params_table.value) > 0
        else None
    )
    return


@app.cell
def _(mo):
    load_params_button = mo.ui.button(label="Load selected parameters and re-run")
    save_params_button = mo.ui.run_button(label="Generate override params file")
    return load_params_button, save_params_button


@app.cell
def _(mo, params_table, save_params_button):
    mo.stop(not save_params_button.value)
    from mapstorch.io import write_override_params_file

    write_override_params_file(
        "maps_fit_parameters_override.txt",
        param_values={
            "COHERENT_SCT_ENERGY": params_table.value.iloc[0][
                "COHERENT_SCT_ENERGY"
            ].item(),
            "ENERGY_OFFSET": params_table.value.iloc[0]["ENERGY_OFFSET"].item(),
            "ENERGY_SLOPE": params_table.value.iloc[0]["ENERGY_SLOPE"].item(),
            "ENERGY_QUADRATIC": params_table.value.iloc[0]["ENERGY_QUADRATIC"].item(),
            "COMPTON_ANGLE": params_table.value.iloc[0]["COMPTON_ANGLE"].item(),
            "COMPTON_FWHM_CORR": params_table.value.iloc[0]["COMPTON_FWHM_CORR"].item(),
            "COMPTON_HI_F_TAIL": params_table.value.iloc[0]["COMPTON_HI_F_TAIL"].item(),
            "COMPTON_F_TAIL": params_table.value.iloc[0]["COMPTON_F_TAIL"].item(),
            "FWHM_FANOPRIME": params_table.value.iloc[0]["FWHM_FANOPRIME"].item(),
            "FWHM_OFFSET": params_table.value.iloc[0]["FWHM_OFFSET"].item(),
            "F_TAIL_OFFSET": params_table.value.iloc[0]["F_TAIL_OFFSET"].item(),
            "KB_F_TAIL_OFFSET": params_table.value.iloc[0]["KB_F_TAIL_OFFSET"].item(),
        },
        elements=params_table.value.iloc[0]["elements"].split(","),
    )
    return (write_override_params_file,)


@app.cell
def _(
    elem_peak_shapes,
    fit_labels,
    fitted_bkg,
    fitted_spec,
    focus_target_switch,
    go,
    int_spec,
    make_subplots,
    np,
    px,
    spec_x,
):
    range_fig = make_subplots(rows=2, cols=1)

    for iii, specc in enumerate([int_spec, fitted_bkg, fitted_spec + fitted_bkg]):
        range_fig.add_trace(
            go.Scatter(
                x=spec_x,
                y=specc,
                mode="lines",
                name=fit_labels[iii],
                line=dict(color=px.colors.qualitative.Plotly[iii]),
            ),
            row=1,
            col=1,
        )
        specc_log = np.log10(np.clip(specc, 0, None) + 1)
        range_fig.add_trace(
            go.Scatter(
                x=spec_x,
                y=specc_log,
                mode="lines",
                showlegend=False,
                line=dict(color=px.colors.qualitative.Plotly[iii]),
            ),
            row=2,
            col=1,
        )

    if focus_target_switch.value:
        range_fig.update_layout(shapes=elem_peak_shapes, overwrite=True)
    return iii, range_fig, specc, specc_log


@app.cell
def _(mo):
    focus_target_switch = mo.ui.switch(label="Focus on target elements", value=False)
    energy_level_slider = mo.ui.slider(
        start=1, stop=6, step=1, value=1, label="Energy levels"
    )
    confirm_range_button = mo.ui.run_button(label="Load target ranges")
    return confirm_range_button, energy_level_slider, focus_target_switch


@app.cell
def _(fit_indices_list, focus_target_switch):
    if not focus_target_switch.value:
        fit_indices_list[-1] = None
    return


@app.cell
def _(elem_checkboxes, energy_level_slider, energy_range, fitted_tensors):
    from mapstorch.util import get_peak_ranges

    plot_elems = [k for k, v in elem_checkboxes.items() if v.value]
    elem_peak_indices = []
    elem_peak_shapes = []
    for ii, ee in enumerate(plot_elems):
        peak_rg = get_peak_ranges(
            [ee],
            fitted_tensors["COHERENT_SCT_ENERGY"].item(),
            fitted_tensors["COMPTON_ANGLE"].item(),
            fitted_tensors["ENERGY_OFFSET"].item(),
            fitted_tensors["ENERGY_SLOPE"].item(),
            fitted_tensors["ENERGY_QUADRATIC"].item(),
            energy_range.value,
        )
        alpha = 0.2
        for p_n, r in peak_rg.items():
            if (
                p_n in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
                or int(p_n[-1]) < energy_level_slider.value
            ):
                elem_peak_indices += list(range(*r))
                elem_peak_shapes.append(
                    dict(
                        type="rect",
                        x0=r[0],
                        x1=r[1],
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        fillcolor="yellow",
                        opacity=alpha,
                        layer="below",
                        line_width=0,
                    )
                )
    return (
        alpha,
        ee,
        elem_peak_indices,
        elem_peak_shapes,
        get_peak_ranges,
        ii,
        p_n,
        peak_rg,
        plot_elems,
        r,
    )


@app.cell
def _(param_checkbox_vals, param_default_vals, params_table):
    if len(params_table.value) > 0:
        for pp in params_table.value:
            if pp in param_checkbox_vals:
                param_checkbox_vals[pp] = float(params_table.value[pp].item())
    else:
        for pp in param_checkbox_vals:
            param_checkbox_vals[pp] = float(param_default_vals[pp])
    return (pp,)


@app.cell
def _(
    PeriodicTableWidget,
    default_fitting_elems,
    elems,
    mo,
    unsupported_elements,
):
    initial_selected_elems = set()
    for e in default_fitting_elems:
        if e in elems and not e in ["COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"]:
            initial_selected_elems.add(e.split("_")[0])

    elem_selection = mo.ui.anywidget(
        PeriodicTableWidget(
            states=1,
            initial_selected={se: 0 for se in initial_selected_elems},
            initial_disabled=unsupported_elements,
        )
    )
    return e, elem_selection, initial_selected_elems


@app.cell
def _(
    default_fitting_elems,
    elem_selection,
    mo,
    supported_elements_mapping,
):
    selected_lines = ["COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE", "Si_Si"]
    for select_e in elem_selection.selected_elements:
        for l_ in supported_elements_mapping[select_e]:
            if l_ == "K":
                selected_lines.append(select_e)
            else:
                selected_lines.append(select_e + "_" + l_)
    elem_checkboxes = {}
    for edf in default_fitting_elems:
        if edf in selected_lines:
            elem_checkboxes[edf] = mo.ui.checkbox(label=edf, value=True)
        else:
            elem_checkboxes[edf] = mo.ui.checkbox(label=edf, value=False)
    return edf, elem_checkboxes, l_, select_e, selected_lines


@app.cell
def _(default_fitting_params, load_params_button, mo, param_checkbox_vals):
    load_params_button

    param_checkboxes = {}
    for p in default_fitting_params:
        param_checkboxes[p] = [
            mo.ui.checkbox(label=p, value=True),
            mo.ui.checkbox(label="Fix"),
            mo.ui.text(value=str(param_checkbox_vals[p])),
        ]

    param_selection = mo.vstack(
        [
            mo.hstack(param_checkboxes[p], justify="start", gap=0)
            for p in default_fitting_params
        ]
    )
    return p, param_checkboxes, param_selection


@app.cell
def _(device_list, mo):
    init_amp_checkbox = mo.ui.checkbox(label="Initialize amplitudes", value=True)
    use_snip_checkbox = mo.ui.checkbox(label="Use SNIP background", value=True)
    model_options = mo.hstack(
        [init_amp_checkbox, use_snip_checkbox],
        justify="start",
        gap=5,
    )
    iter_slider = mo.ui.slider(
        value=500, start=100, stop=3000, step=50, label="number of iterations"
    )
    loss_selection = mo.ui.dropdown(["mse", "l1"], value="mse", label="loss")
    optimizer_selection = mo.ui.dropdown(
        ["adam", "adamw"], value="adam", label="optimizer"
    )
    device_selection = mo.ui.dropdown(device_list, value="cpu", label="device")
    opt_options = mo.hstack(
        [device_selection, loss_selection, optimizer_selection, iter_slider],
        justify="start",
        gap=2,
    )
    configs = mo.vstack([model_options, opt_options])
    return (
        configs,
        device_selection,
        init_amp_checkbox,
        iter_slider,
        loss_selection,
        model_options,
        opt_options,
        optimizer_selection,
        use_snip_checkbox,
    )


@app.cell
def _():
    import torch

    device_list = ["cpu"]
    if torch.cuda.is_available():
        device_list.append("cuda")
    return device_list, torch


@app.cell
def _():
    fit_indices_list = [None]
    return (fit_indices_list,)


@app.cell
def _(
    acos,
    compton_peak_slider,
    elastic_peak_slider,
    incident_energy_slider,
    pi,
):
    from mapstorch.default import default_param_vals, default_fitting_params
    from copy import copy

    coherent_sct_energy = incident_energy_slider.value
    energy_slope = coherent_sct_energy / elastic_peak_slider.value
    compton_energy = energy_slope * compton_peak_slider.value
    try:
        compton_angle = (
            acos(1 - 511 * (1 / compton_energy - 1 / coherent_sct_energy)) * 180 / pi
        )
    except:
        compton_angle = default_param_vals["COMPTON_ANGLE"]
    param_default_vals = copy(default_param_vals)
    param_default_vals["COHERENT_SCT_ENERGY"] = coherent_sct_energy
    param_default_vals["ENERGY_SLOPE"] = energy_slope
    param_default_vals["COMPTON_ANGLE"] = compton_angle
    param_checkbox_vals = copy(param_default_vals)
    params_record = {p: [] for p in default_fitting_params + ["elements"]}
    return (
        coherent_sct_energy,
        compton_angle,
        compton_energy,
        copy,
        default_fitting_params,
        default_param_vals,
        energy_slope,
        param_checkbox_vals,
        param_default_vals,
        params_record,
    )


@app.cell
def _(int_spec):
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(int_spec, prominence=int_spec.max() / 200)
    return find_peaks, peaks


@app.cell
def _(energy_range, int_spec_og, np):
    int_spec = int_spec_og[energy_range.value[0] : energy_range.value[1] + 1]
    int_spec_log = np.log10(np.clip(int_spec, 0, None) + 1)
    return int_spec, int_spec_log


@app.cell
def _(dataset, dataset_button, elem_path, int_spec_path, mo, read_dataset):
    mo.stop(not dataset_button.value)
    dataset_dict = read_dataset(
        dataset.value[0].path,
        fit_elem_key=elem_path.value,
        int_spec_key=int_spec_path.value,
    )
    int_spec_og = dataset_dict["int_spec"]
    elems = dataset_dict["elems"]
    return dataset_dict, elems, int_spec_og


if __name__ == "__main__":
    app.run()
