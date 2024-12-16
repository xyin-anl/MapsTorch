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
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.absolute()))

    from math import floor, ceil, acos, pi
    import numpy as np
    import plotly.express as px
    import marimo as mo

    from maps_torch.io import read_dataset

    return Path, acos, ceil, floor, mo, np, pi, px, read_dataset, sys


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
    int_spec_path = mo.ui.dropdown(
        ["MAPS/int_spec", "MAPS/Spectra/Integrateds_Spectra/Spectra"],
        value="MAPS/int_spec",
        label="Integrated spectrum location",
    )
    elem_path = mo.ui.dropdown(
        ["MAPS/channel_names", "None"],
        value="None",
        label="Energy channel names location",
    )
    dataset_button = mo.ui.run_button(label="Load")
    mo.hstack(
        [int_spec_path, elem_path, dataset_button], justify="start", gap=1
    ).right()
    return dataset_button, elem_path, int_spec_path


@app.cell
def __(int_spec_og, mo):
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
def __(energy_range, int_spec_og, mo, peaks):
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
def __(
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

    int_spec_fig_shown = True

    mo.ui.plotly(int_spec_fig)
    return go, int_spec_fig, int_spec_fig_shown, make_subplots


@app.cell
def __(int_spec_fig_shown, mo):
    run_button = mo.ui.run_button(label="Start guessing! This may take a while ...")
    run_button.right() if int_spec_fig_shown else None
    return (run_button,)


@app.cell
def __(energy_range, int_spec_og, mo, param_default_vals, run_button):
    mo.stop(not run_button.value)
    import torch
    from maps_torch.opt import fit_spec

    n_iter = 1000
    with mo.status.progress_bar(total=n_iter) as bar:
        fitted_tensors, fitted_spec, fitted_bkg, loss_trace = fit_spec(
            int_spec_og,
            energy_range.value,
            init_param_vals=param_default_vals,
            n_iter=n_iter,
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
        torch,
    )


@app.cell
def __(fitted_tensors, go, init_elems, make_subplots, mo):
    from maps_torch.default import default_fitting_elems

    amps = {p: fitted_tensors[p].item() for p in default_fitting_elems}
    amps = dict(sorted(amps.items(), key=lambda item: item[1]))

    amp_fig = make_subplots(rows=1, cols=2)

    amp_fig.add_trace(
        go.Bar(
            x=[10**v for v in amps.values()],
            y=list(amps.keys()),
            orientation="h",
            name="Photon counts",
            marker_color=["grey" if v in init_elems else "red" for v in amps.keys()],
        ),
        row=1,
        col=1,
    )
    amp_fig.add_trace(
        go.Bar(
            x=list(amps.values()),
            y=list(amps.keys()),
            orientation="h",
            name="Log scale",
            marker_color=["grey" if v in init_elems else "red" for v in amps.keys()],
        ),
        row=1,
        col=2,
    )
    amp_fig.update_yaxes(showticklabels=False, row=1, col=2)
    amp_fig.update_layout(showlegend=False)
    results_shown = True

    mo.ui.plotly(amp_fig)
    return amp_fig, amps, default_fitting_elems, results_shown


@app.cell
def __(
    amps,
    default_fitting_elems,
    elem_selection_slider,
    init_elems,
    mo,
    results_shown,
):
    elem_checkboxes = {}
    if len(init_elems) > 0:
        init_elem_ranked = [e for e in list(amps.keys()) if e in init_elems]
        non_init_elem_ranked = [e for e in list(amps.keys()) if not e in init_elems]
        if elem_selection_slider.value == 0:
            selected_elem = init_elems
        elif elem_selection_slider.value > 0:
            selected_elem = (
                init_elems + non_init_elem_ranked[-elem_selection_slider.value :]
            )
        else:
            selected_elem = init_elem_ranked[-elem_selection_slider.value :]
    else:
        if elem_selection_slider.value <= 0:
            selected_elem = []
        else:
            selected_elem = list(amps.keys())[-elem_selection_slider.value :]
    if not "COHERENT_SCT_AMPLITUDE" in selected_elem:
        selected_elem.append("COHERENT_SCT_AMPLITUDE")
    if not "COMPTON_AMPLITUDE" in selected_elem:
        selected_elem.append("COMPTON_AMPLITUDE")
    for e in default_fitting_elems:
        if e in selected_elem:
            elem_checkboxes[e] = mo.ui.checkbox(label=e, value=True)
        else:
            elem_checkboxes[e] = mo.ui.checkbox(label=e, value=False)
    elem_selection = mo.hstack(
        [elem_checkboxes[e] for e in default_fitting_elems], wrap=True
    )
    elem_selection_shown = True
    elem_selection if results_shown else None
    return (
        e,
        elem_checkboxes,
        elem_selection,
        elem_selection_shown,
        init_elem_ranked,
        non_init_elem_ranked,
        selected_elem,
    )


@app.cell
def __(init_elems, mo, results_shown):
    elem_selection_slider_value = 5 if len(init_elems) > 0 else 12
    elem_selection_slider = mo.ui.slider(
        start=-len(init_elems),
        stop=min(40, 60 - len(init_elems)),
        step=1,
        value=elem_selection_slider_value,
        full_width=True,
    )
    elem_selection_slider if results_shown else None
    return elem_selection_slider, elem_selection_slider_value


@app.cell
def __(elem_selection_shown, mo):
    adjust_button = mo.ui.button(label="Adjust")
    adjust_button.right() if elem_selection_shown else None
    return (adjust_button,)


@app.cell
def __(colors_map, go):
    color_fig = go.Figure()

    for i_, e_name in enumerate(colors_map):
        hex = colors_map[e_name]
        if e_name == "COMPTON_AMPLITUDE":
            plot_name = "Com"
        elif e_name == "COHERENT_SCT_AMPLITUDE":
            plot_name = "Coh"
        else:
            plot_name = e_name
        color_fig.add_shape(
            type="rect",
            x0=i_,
            y0=0,
            x1=i_ + 1,
            y1=0.05,
            line=dict(color=hex),
            fillcolor=hex,
        )
        color_fig.add_annotation(
            x=i_ + 0.5,
            y=0.055,
            text=plot_name,
            showarrow=False,
            xanchor="auto",
            yanchor="bottom",
        )

    color_fig.update_yaxes(showticklabels=False, range=[0, 0.1])
    color_fig.update_xaxes(showticklabels=False, range=[0, len(colors_map)])
    color_fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)")
    color_fig.update_layout(height=200, width=1000)
    return color_fig, e_name, hex, i_, plot_name


@app.cell
def __(go, int_spec, int_spec_log, make_subplots, mo, peak_ranges):
    peak_pos_fig = make_subplots(rows=2, cols=1)
    peak_pos_fig.add_trace(go.Scatter(y=int_spec), row=1, col=1)
    peak_pos_fig.add_trace(go.Scatter(y=int_spec_log), row=2, col=1)

    peak_pos_fig.update_layout(shapes=peak_ranges)
    peak_pos_fig.update_layout(showlegend=False)

    mo.ui.plotly(peak_pos_fig)
    return (peak_pos_fig,)


@app.cell
def __(elem_selection_shown, mo):
    evaluate_button = mo.ui.run_button(label="Evaluate")
    evaluate_button.right() if elem_selection_shown else None
    return (evaluate_button,)


@app.cell
def __(
    elem_checkboxes,
    energy_range,
    evaluate_button,
    fit_spec,
    fitted_tensors,
    int_spec_og,
    mo,
    param_default_vals,
):
    mo.stop(not evaluate_button.value)
    eval_iter = 1000
    with mo.status.progress_bar(total=eval_iter) as eval_bar:
        eval_tensors, eval_spec, eval_bkg, eval_trace = fit_spec(
            int_spec_og,
            energy_range.value,
            elements_to_fit=[k for k, v in elem_checkboxes.items() if v.value],
            init_param_vals={
                k: v.item()
                for k, v in fitted_tensors.items()
                if k in param_default_vals
            },
            n_iter=eval_iter,
            status_updator=eval_bar,
        )
    analysis_done = True
    return (
        analysis_done,
        eval_bar,
        eval_bkg,
        eval_iter,
        eval_spec,
        eval_tensors,
        eval_trace,
    )


@app.cell
def __(eval_bkg, eval_spec, go, int_spec, make_subplots, mo, np, px):
    fit_labels = ["experiment", "background", "fitted"]
    fit_fig = make_subplots(rows=2, cols=1)
    spec_x = np.linspace(0, int_spec.size - 1, int_spec.size)

    for i, spec in enumerate([int_spec, eval_bkg, eval_spec + eval_bkg]):
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
def __(go, mo, traces):
    contrib_fig = go.Figure(data=traces)
    contrib_fig.update_layout(
        yaxis_type="log",
        legend=dict(orientation="h", yanchor="top", y=-0.04, xanchor="right", x=1),
        showlegend=False,
    )
    mo.ui.plotly(contrib_fig)
    return (contrib_fig,)


@app.cell
def __(
    amps,
    dataset,
    energy_range,
    eval_bkg,
    eval_spec,
    init_elems,
    int_spec,
):
    import pickle

    output_res = {}
    output_res["init_elems"] = init_elems
    output_res["amps"] = amps
    output_res["int_spec"] = int_spec
    output_res["energy_range"] = energy_range.value
    output_res["fit_spec"] = eval_spec
    output_res["fit_bkg"] = eval_bkg
    pickle.dump(output_res, open(dataset.value[0].name + "_guess_elems_res.pkl", "wb"))
    return output_res, pickle


@app.cell
def __(
    e,
    elem_colors,
    energy_range,
    eval_bkg,
    eval_tensors,
    evaluate_button,
    go,
    int_spec_og,
    mo,
    np,
    plot_elems,
    torch,
):
    mo.stop(not evaluate_button.value)
    from maps_torch.map import model_elem_spec, compton_peak, elastic_peak

    with torch.no_grad():
        traces = []
        res = {}
        base_spectrum = int_spec_og[energy_range.value[0] : energy_range.value[1] + 1]
        traces.append(
            go.Scatter(
                y=np.clip(base_spectrum - eval_bkg, 1, None),
                mode="lines",
                name="Spectrum",
            )
        )

        for il, el in enumerate(plot_elems):
            if el in eval_tensors:
                energy = torch.linspace(
                    energy_range.value[0],
                    energy_range.value[1] + 1,
                    energy_range.value[1] - energy_range.value[0] + 1,
                    device=eval_tensors[el].device,
                )
                ev = (
                    eval_tensors["ENERGY_OFFSET"]
                    + eval_tensors["ENERGY_SLOPE"] * energy
                    + eval_tensors["ENERGY_QUADRATIC"] * (energy**2)
                )
                if el == "COMPTON_AMPLITUDE":
                    e_spec = compton_peak(
                        eval_tensors, ev, eval_tensors["ENERGY_SLOPE"]
                    )
                elif el == "COHERENT_SCT_AMPLITUDE":
                    e_spec = elastic_peak(
                        eval_tensors, ev, eval_tensors["ENERGY_SLOPE"]
                    )
                else:
                    e_spec = model_elem_spec(
                        eval_tensors, el, ev, device=eval_tensors[el].device
                    )
                res[e] = e_spec.cpu().numpy()
                traces.append(
                    go.Scatter(
                        y=np.clip(e_spec.cpu().numpy(), 1, None),
                        mode="lines",
                        line=dict(color=elem_colors[il]),
                        name=el,
                    )
                )
    return (
        base_spectrum,
        compton_peak,
        e_spec,
        el,
        elastic_peak,
        energy,
        ev,
        il,
        model_elem_spec,
        res,
        traces,
    )


@app.cell
def __(adjust_button, elem_checkboxes, energy_range, fitted_tensors, px):
    adjust_button

    from maps_torch.util import get_peak_ranges

    elem_colors = px.colors.qualitative.Light24 + px.colors.qualitative.Dark24

    plot_elems = [k for k, v in elem_checkboxes.items() if v.value]
    peak_ranges = []
    colors_map = {}
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
        alpha = 0.4
        for p_n, r in peak_rg.items():
            peak_ranges.append(
                dict(
                    type="rect",
                    x0=r[0],
                    x1=r[1],
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    fillcolor=elem_colors[ii],
                    opacity=alpha,
                    layer="below",
                    line_width=0,
                )
            )
            alpha /= 1.1
        colors_map[ee] = elem_colors[ii]
    return (
        alpha,
        colors_map,
        ee,
        elem_colors,
        get_peak_ranges,
        ii,
        p_n,
        peak_ranges,
        peak_rg,
        plot_elems,
        r,
    )


@app.cell
def __(
    acos,
    compton_peak_slider,
    elastic_peak_slider,
    incident_energy_slider,
    pi,
):
    from maps_torch.default import default_param_vals

    coherent_sct_energy = incident_energy_slider.value
    energy_slope = coherent_sct_energy / elastic_peak_slider.value
    compton_energy = energy_slope * compton_peak_slider.value
    try:
        compton_angle = (
            acos(1 - 511 * (1 / compton_energy - 1 / coherent_sct_energy)) * 180 / pi
        )
    except:
        compton_angle = default_param_vals["COMPTON_ANGLE"]
    param_default_vals = default_param_vals
    param_default_vals["COHERENT_SCT_ENERGY"] = coherent_sct_energy
    param_default_vals["ENERGY_SLOPE"] = energy_slope
    param_default_vals["COMPTON_ANGLE"] = compton_angle
    return (
        coherent_sct_energy,
        compton_angle,
        compton_energy,
        default_param_vals,
        energy_slope,
        param_default_vals,
    )


@app.cell
def __(int_spec):
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(int_spec, prominence=int_spec.max() / 100)
    return find_peaks, peaks


@app.cell
def __(energy_range, int_spec_og, np):
    int_spec = int_spec_og[energy_range.value[0] : energy_range.value[1] + 1]
    int_spec_log = np.log10(np.clip(int_spec, 0, None) + 1)
    return int_spec, int_spec_log


@app.cell
def __(
    dataset,
    dataset_button,
    elem_path,
    int_spec_path,
    mo,
    read_dataset,
):
    mo.stop(not dataset_button.value)
    if elem_path.value == "None":
        dataset_dict = read_dataset(
            dataset.value[0].path, int_spec_key=int_spec_path.value
        )
        init_elems = []
    else:
        dataset_dict = read_dataset(
            dataset.value[0].path,
            fit_elem_key=elem_path.value,
            int_spec_key=int_spec_path.value,
        )
        init_elems = dataset_dict["elems"]
    int_spec_og = dataset_dict["int_spec"]
    return dataset_dict, init_elems, int_spec_og


if __name__ == "__main__":
    app.run()
