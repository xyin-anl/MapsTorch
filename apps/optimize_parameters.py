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

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from math import floor, ceil, acos, pi
    import numpy as np
    import plotly.express as px
    import marimo as mo

    from mapstorch.io import read_dataset, parse_override_params_file
    from mapstorch.util import (
        BranchingRatioWidget,
        PeriodicTableWidget,
        branching_value_to_maps_lines,
        parse_maps_branching_lines,
    )
    from mapstorch.constant import read_constants, define_constants
    from mapstorch.default import (
        default_fitting_elems,
        default_fitting_params,
        default_param_vals,
        default_energy_consts,
        default_elem_info,
        default_pileups,
        unsupported_elements,
        supported_elements_mapping,
    )
    return (
        BranchingRatioWidget,
        PeriodicTableWidget,
        acos,
        branching_value_to_maps_lines,
        parse_maps_branching_lines,
        default_elem_info,
        default_fitting_elems,
        default_fitting_params,
        default_param_vals,
        default_pileups,
        define_constants,
        mo,
        np,
        parse_override_params_file,
        pi,
        px,
        read_constants,
        read_dataset,
        supported_elements_mapping,
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
    return compton_peak_slider, elastic_peak_slider, incident_energy_slider


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
    return go, make_subplots


@app.cell
def _(mo):
    override_params_file = mo.ui.file_browser(filetypes=[".txt"], multiple=False)
    apply_override_button = mo.ui.run_button(label="Apply")
    return apply_override_button, override_params_file


@app.cell
def _(
    apply_override_button,
    mo,
    override_params_file,
    parse_override_params_file,
):
    override_file_path = None
    override_params = None
    override_branching_text = ""
    if override_params_file.value:
        override_file_path = override_params_file.value[0].path
    if override_file_path and apply_override_button.value:
        override_params = parse_override_params_file(override_file_path)
        try:
            with open(
                override_file_path, "r", encoding="utf-8", errors="ignore"
            ) as _fh:
                lines = [
                    line.rstrip("\n")
                    for line in _fh
                    if line.lstrip().startswith("BRANCHING_")
                ]
            override_branching_text = "\n".join(lines)
        except Exception as e:
            mo.callout(f"Failed to read branching ratio lines: {e}", kind="warning")
        mo.callout(
            f"Override parameters loaded from `{override_params_file.value[0].name}`.",
            kind="success",
        )
    return override_branching_text, override_file_path, override_params


@app.cell
def _(override_params):
    override_detector_material = None
    override_escape_factor = None
    override_elements_to_fit = None
    override_pileups = []
    if override_params is not None:
        override_detector_material = override_params.get("detector_material", "Si")
        override_escape_factor = float(override_params.get("escape_factor", 0.0))
        override_elements_to_fit = override_params.get("elements_to_fit") or []
        override_pileups = override_params.get("elements_with_pileup") or []
    return (
        override_detector_material,
        override_elements_to_fit,
        override_escape_factor,
        override_pileups,
    )


@app.cell
def _(
    BranchingRatioWidget,
    mo,
    override_branching_text,
    override_pileups,
    parse_maps_branching_lines,
):
    pileups_text = mo.ui.text(
        value=", ".join(override_pileups) if override_pileups else "",
        label="Pileups",
    )
    widget = BranchingRatioWidget()
    if override_branching_text:
        widget.value = parse_maps_branching_lines(override_branching_text)
    branching_widget = mo.ui.anywidget(widget)
    return branching_widget, pileups_text


@app.cell
def _(branching_value_to_maps_lines, branching_widget):
    branching_overrides_text = branching_value_to_maps_lines(branching_widget.value)
    return (branching_overrides_text,)


@app.cell
def _(
    apply_override_button,
    branching_widget,
    mo,
    override_params_file,
    pileups_text,
):
    override_panel = mo.vstack(
        [
            mo.md(
                "Load a MAPS override file to seed parameters/elements, and optionally add pileups or branching ratio overrides."
            ),
            mo.hstack([override_params_file, apply_override_button], justify="start", gap=2),
            pileups_text,
            branching_widget,
        ]
    )
    return (override_panel,)


@app.cell
def _(pileups_text):
    def parse_csv(text: str) -> list[str]:
        if not text:
            return []
        return [t.strip() for t in text.split(",") if t.strip()]

    custom_pileups = parse_csv(pileups_text.value)
    return (custom_pileups,)


@app.cell
def _(custom_pileups, default_fitting_elems):
    extra_elems = [p for p in custom_pileups if p not in default_fitting_elems]
    available_fitting_elems = list(dict.fromkeys(list(default_fitting_elems) + extra_elems))
    return (available_fitting_elems,)


@app.cell
def _(
    branching_overrides_text,
    custom_pileups,
    default_elem_info,
    default_pileups,
    define_constants,
    detector_material,
    mo,
    override_file_path,
    override_params,
    read_constants,
):
    import os
    import tempfile

    detector = detector_material.value

    if override_params is not None and override_file_path:
        try:
            model_energy_consts = read_constants(override_file_path, default_elem_info)
        except Exception as e:
            mo.callout(f"Failed to read constants from override file: {e}", kind="danger")
            all_pileups = sorted(set(list(default_pileups) + list(custom_pileups)))
            model_energy_consts = define_constants(default_elem_info, all_pileups)
        else:
            extra_pileups = [p for p in custom_pileups if p not in model_energy_consts]
            if extra_pileups:
                _tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False
                    ) as _fh:
                        _tmp_path = _fh.name
                        _fh.write(
                            f"ELEMENTS_WITH_PILEUP: {', '.join(extra_pileups)}\n"
                        )
                    extra_consts = read_constants(_tmp_path, default_elem_info)
                    for p in extra_pileups:
                        if p in extra_consts:
                            model_energy_consts[p] = extra_consts[p]
                except Exception as e:
                    mo.callout(f"Failed to add pileups: {e}", kind="danger")
                finally:
                    if _tmp_path is not None:
                        try:
                            os.unlink(_tmp_path)
                        except OSError:
                            pass
    else:
        all_pileups = sorted(set(list(default_pileups) + list(custom_pileups)))
        model_energy_consts = define_constants(default_elem_info, all_pileups)

    override_lines = [
        line.strip()
        for line in branching_overrides_text.splitlines()
        if line.strip().startswith("BRANCHING_")
    ]
    overridden = []
    if override_lines:
        _tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as _fh:
                _tmp_path = _fh.name
                for line in override_lines:
                    _fh.write(f"{line}\n")
            adj_consts = read_constants(_tmp_path, default_elem_info)
            for line in override_lines:
                rhs = line.split(":", 1)[1].strip()
                name = rhs.split(",", 1)[0].strip()
                if name in adj_consts and name in model_energy_consts:
                    model_energy_consts[name] = adj_consts[name]
                    overridden.append(name)
        except Exception as e:
            mo.callout(f"Failed to apply branching ratio overrides: {e}", kind="danger")
        finally:
            if _tmp_path is not None:
                try:
                    os.unlink(_tmp_path)
                except OSError:
                    pass

    use_si = detector == "Si"
    for _pars in model_energy_consts.values():
        for _par in _pars:
            _par.mu_fraction = _par.Si_mu if use_si else _par.Ge_mu

    missing_pileups = [p for p in custom_pileups if p not in model_energy_consts]
    if missing_pileups:
        mo.callout(
            f"Pileups not found in constants (will be ignored): {', '.join(missing_pileups)}",
            kind="warning",
        )
    return (model_energy_consts,)


@app.cell
def _(configs, elem_selection, mo, override_panel, param_selection):
    control_panel = mo.accordion(
        {
            "Elements": elem_selection.center(),
            "Parameters": param_selection,
            "Override": override_panel,
            "Configs": configs,
        },
        multiple=True,
    )
    control_panel_shown = True
    control_panel
    return (control_panel_shown,)


@app.cell
def _(control_panel_shown, mo):
    run_button = mo.ui.run_button()
    run_button.right() if control_panel_shown else None
    return (run_button,)


@app.cell
def _(
    detector_material,
    device_selection,
    elem_checkboxes,
    energy_range,
    escape_factor,
    fit_indices_list,
    init_amp_checkbox,
    int_spec_og,
    iter_slider,
    loss_selection,
    mo,
    model_energy_consts,
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
            e_consts=model_energy_consts,
            detector_type=detector_material.value,
            escape_factor=float(escape_factor.value),
        )
    return fitted_bkg, fitted_spec, fitted_tensors


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
    return fit_labels, spec_x


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
    return (results_shown,)


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
def _(
    branching_overrides_text,
    dataset,
    default_fitting_params,
    detector_material,
    elem_checkboxes,
    energy_range,
    escape_factor,
    fitted_tensors,
    init_amp_checkbox,
    iter_slider,
    loss_selection,
    mo,
    optimizer_selection,
    override_file_path,
    params_record,
    pileups_text,
    use_snip_checkbox,
):
    import pandas as pd
    import datetime

    selected_elems = ",".join([k for k, v in elem_checkboxes.items() if v.value])
    params_record["elements"].append(selected_elems)
    params_record["elements_with_pileup"].append(str(pileups_text.value or ""))
    params_record["branching_overrides"].append(str(branching_overrides_text or ""))
    params_record["detector_material"].append(str(detector_material.value or ""))
    params_record["escape_factor"].append(float(escape_factor.value))
    params_record["loss"].append(str(loss_selection.value or ""))
    params_record["optimizer"].append(str(optimizer_selection.value or ""))
    params_record["n_iter"].append(int(iter_slider.value))
    params_record["init_amp"].append(bool(init_amp_checkbox.value))
    params_record["use_snip"].append(bool(use_snip_checkbox.value))
    params_record["energy_range"].append(str(energy_range.value))
    params_record["override_file"].append(str(override_file_path or ""))
    for _param_name in default_fitting_params:
        if _param_name in fitted_tensors:
            params_record[_param_name].append(fitted_tensors[_param_name].item())
        else:
            params_record[_param_name].append(None)
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
    return (params_table,)


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
def _(detector_material, escape_factor, mo, params_table, save_params_button):
    mo.stop(not save_params_button.value)
    from mapstorch.io import write_override_params_file

    aux_components = {"COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"}
    selected_row = params_table.value.iloc[0]
    selected_elements = [
        e.strip() for e in str(selected_row["elements"]).split(",") if e.strip()
    ]
    recorded_pileups = str(selected_row.get("elements_with_pileup", "") or "").strip()
    if recorded_pileups:
        elements_with_pileup = [
            e.strip() for e in recorded_pileups.split(",") if e.strip()
        ]
    else:
        elements_with_pileup = [
            e
            for e in selected_elements
            if ("_" in e) and (not e.endswith("_L")) and (not e.endswith("_M"))
        ]
    elements_to_fit = [
        e
        for e in selected_elements
        if (e not in aux_components) and (e not in elements_with_pileup)
    ]
    detector_material_to_write = str(
        selected_row.get("detector_material", detector_material.value) or "Si"
    )
    escape_factor_to_write = float(
        selected_row.get("escape_factor", float(escape_factor.value)) or 0.0
    )
    _branching_overrides_text = str(
        selected_row.get("branching_overrides", "") or ""
    ).strip()

    write_override_params_file(
        "maps_fit_parameters_override.txt",
        param_values={
            "COHERENT_SCT_ENERGY": float(selected_row["COHERENT_SCT_ENERGY"]),
            "ENERGY_OFFSET": float(selected_row["ENERGY_OFFSET"]),
            "ENERGY_SLOPE": float(selected_row["ENERGY_SLOPE"]),
            "ENERGY_QUADRATIC": float(selected_row["ENERGY_QUADRATIC"]),
            "COMPTON_ANGLE": float(selected_row["COMPTON_ANGLE"]),
            "COMPTON_FWHM_CORR": float(selected_row["COMPTON_FWHM_CORR"]),
            "COMPTON_HI_F_TAIL": float(selected_row["COMPTON_HI_F_TAIL"]),
            "COMPTON_F_TAIL": float(selected_row["COMPTON_F_TAIL"]),
            "FWHM_FANOPRIME": float(selected_row["FWHM_FANOPRIME"]),
            "FWHM_OFFSET": float(selected_row["FWHM_OFFSET"]),
            "F_TAIL_OFFSET": float(selected_row["F_TAIL_OFFSET"]),
            "KB_F_TAIL_OFFSET": float(selected_row["KB_F_TAIL_OFFSET"]),
        },
        elements=elements_to_fit,
        elements_with_pileup=elements_with_pileup,
        branching_ratio_lines=_branching_overrides_text or None,
        detector_material=detector_material_to_write,
        escape_factor=escape_factor_to_write,
    )
    return


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
    return (range_fig,)


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
    for ee in plot_elems:
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
        for peak_name, peak_range in peak_rg.items():
            if (
                peak_name in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
                or int(peak_name[-1]) < energy_level_slider.value
            ):
                elem_peak_indices += list(range(*peak_range))
                elem_peak_shapes.append(
                    dict(
                        type="rect",
                        x0=peak_range[0],
                        x1=peak_range[1],
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
    return elem_peak_indices, elem_peak_shapes


@app.cell
def _(override_params, param_checkbox_vals, param_default_vals, params_table):
    if len(params_table.value) > 0:
        for pp in params_table.value:
            if pp in param_checkbox_vals:
                param_checkbox_vals[pp] = float(params_table.value[pp].item())
    elif override_params is not None:
        for pp in param_checkbox_vals:
            if pp in override_params:
                param_checkbox_vals[pp] = float(override_params[pp])
            else:
                param_checkbox_vals[pp] = float(param_default_vals[pp])
    else:
        for pp in param_checkbox_vals:
            param_checkbox_vals[pp] = float(param_default_vals[pp])
    return


@app.cell
def _(
    PeriodicTableWidget,
    available_fitting_elems,
    elems,
    mo,
    override_elements_to_fit,
    unsupported_elements,
):
    initial_selected_elems = set()
    if override_elements_to_fit:
        for e in override_elements_to_fit:
            if e in ["COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"]:
                continue
            if "_" in e:
                parts = e.split("_")
                if parts[-1] in ["L", "M"]:
                    initial_selected_elems.add(parts[0])
            else:
                initial_selected_elems.add(e)
    else:
        for e in available_fitting_elems:
            if e in elems and not e in ["COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"]:
                initial_selected_elems.add(e.split("_")[0])

    elem_selection = mo.ui.anywidget(
        PeriodicTableWidget(
            states=1,
            initial_selected={se: 0 for se in initial_selected_elems},
            initial_disabled=unsupported_elements,
        )
    )
    return (elem_selection,)


@app.cell
def _(
    available_fitting_elems,
    custom_pileups,
    default_pileups,
    elem_selection,
    mo,
    override_elements_to_fit,
    supported_elements_mapping,
):
    selected_lines = ["COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"]

    if override_elements_to_fit:
        selected_lines.extend(list(override_elements_to_fit))
    else:
        selected_lines.extend(list(default_pileups))
        for select_e in elem_selection.selected_elements:
            for l_ in supported_elements_mapping[select_e]:
                if l_ == "K":
                    selected_lines.append(select_e)
                else:
                    selected_lines.append(select_e + "_" + l_)

    selected_lines.extend(custom_pileups)
    selected_lines = list(dict.fromkeys(selected_lines))
    elem_checkboxes = {}
    for elem_name in available_fitting_elems:
        if elem_name in selected_lines:
            elem_checkboxes[elem_name] = mo.ui.checkbox(label=elem_name, value=True)
        else:
            elem_checkboxes[elem_name] = mo.ui.checkbox(label=elem_name, value=False)
    return (elem_checkboxes,)


@app.cell
def _(
    default_fitting_params,
    load_params_button,
    mo,
    override_params,
    param_checkbox_vals,
):
    load_params_button

    param_checkboxes = {}
    use_override_seed = (override_params is not None) and (not bool(load_params_button.value))
    for _param_name in default_fitting_params:
        default_value = float(param_checkbox_vals[_param_name])
        if use_override_seed and _param_name in override_params:
            default_value = float(override_params[_param_name])
        param_checkboxes[_param_name] = [
            mo.ui.checkbox(label=_param_name, value=True),
            mo.ui.checkbox(label="Fix"),
            mo.ui.text(value=str(default_value)),
        ]

    param_selection = mo.vstack(
        [
            mo.hstack(param_checkboxes[_param_name], justify="start", gap=0)
            for _param_name in default_fitting_params
        ]
    )
    return param_checkboxes, param_selection


@app.cell
def _(device_list, mo, override_detector_material, override_escape_factor):
    init_amp_checkbox = mo.ui.checkbox(label="Initialize amplitudes", value=True)
    use_snip_checkbox = mo.ui.checkbox(label="Use SNIP background", value=True)
    model_options = mo.hstack(
        [init_amp_checkbox, use_snip_checkbox],
        justify="start",
        gap=5,
    )
    detector_material = mo.ui.dropdown(
        ["Si", "Ge"],
        value=override_detector_material or "Si",
        label="Detector material",
    )
    escape_factor = mo.ui.number(
        value=float(override_escape_factor or 0.0),
        step=0.0001,
        label="Escape factor",
    )
    detector_options = mo.hstack(
        [detector_material, escape_factor],
        justify="start",
        gap=2,
    )
    iter_slider = mo.ui.slider(
        value=500, start=100, stop=3000, step=50, label="number of iterations"
    )
    loss_selection = mo.ui.dropdown(
        ["mse", "l1", "poisson_deviance", "poisson_nll", "chi2"],
        value="mse",
        label="loss",
    )
    optimizer_selection = mo.ui.dropdown(
        ["adam", "adamw"], value="adam", label="optimizer"
    )
    device_selection = mo.ui.dropdown(device_list, value="cpu", label="device")
    opt_options = mo.hstack(
        [device_selection, loss_selection, optimizer_selection, iter_slider],
        justify="start",
        gap=2,
    )
    configs = mo.vstack([model_options, detector_options, opt_options])
    return (
        configs,
        detector_material,
        device_selection,
        escape_factor,
        init_amp_checkbox,
        iter_slider,
        loss_selection,
        optimizer_selection,
        use_snip_checkbox,
    )


@app.cell
def _():
    import torch

    device_list = ["cpu"]
    if torch.cuda.is_available():
        device_list.append("cuda")
    return (device_list,)


@app.cell
def _():
    fit_indices_list = [None]
    return (fit_indices_list,)


@app.cell
def _(
    acos,
    compton_peak_slider,
    default_fitting_params,
    default_param_vals,
    elastic_peak_slider,
    incident_energy_slider,
    pi,
):
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
    record_fields = (
        list(default_fitting_params)
        + [
            "elements",
            "elements_with_pileup",
            "branching_overrides",
            "detector_material",
            "escape_factor",
            "loss",
            "optimizer",
            "n_iter",
            "init_amp",
            "use_snip",
            "energy_range",
            "override_file",
        ]
    )
    params_record = {field: [] for field in record_fields}
    return param_checkbox_vals, param_default_vals, params_record


@app.cell
def _(int_spec):
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(int_spec, prominence=int_spec.max() / 200)
    return (peaks,)


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
    return elems, int_spec_og


if __name__ == "__main__":
    app.run()
