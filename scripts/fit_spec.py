import sys
from math import acos, pi
from copy import copy
import argparse
import warnings

import numpy as np
from scipy.signal import find_peaks

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

from mapstorch.io import read_dataset, write_override_params_file
from mapstorch.default import (
    default_param_vals,
    default_fitting_params,
)
from mapstorch.opt import fit_spec


def main(args):
    # Configuration parameters
    dataset = args.dataset
    int_spec_path = args.int_spec_path
    elem_path = args.elem_path
    coherent_sct_energy = args.incident_energy
    energy_range = args.energy_range
    init_amp = args.init_amp
    use_snip = args.use_snip
    use_step = args.use_step
    n_iter = args.n_iter
    loss_selection = args.loss_selection
    device_selection = args.device_selection
    verbose = args.verbose
    save_params = args.save_params
    output_path = args.output_path
    show_figures = args.show_figures

    # Read dataset and extract relevant data
    dataset_dict = read_dataset(
        dataset, fit_elem_key=elem_path, int_spec_key=int_spec_path
    )
    int_spec_og = dataset_dict["int_spec"]
    elems = dataset_dict["elems"]
    int_spec = int_spec_og[energy_range[0] : energy_range[1] + 1]

    # Find peaks in the spectrum
    peaks, _ = find_peaks(int_spec, prominence=int_spec.max() / 200)
    compton_peak_value = (
        (int_spec_og.shape[-1] - 1) // 2 if len(peaks) < 8 else peaks[-2]
    )
    elastic_peak_value = (
        (int_spec_og.shape[-1] - 1) // 1.9 if len(peaks) < 8 else peaks[-1]
    )

    # Calculate energy slope and Compton angle
    energy_slope = coherent_sct_energy / elastic_peak_value
    compton_energy = energy_slope * compton_peak_value
    try:
        compton_angle = (
            acos(1 - 511 * (1 / compton_energy - 1 / coherent_sct_energy)) * 180 / pi
        )
    except:
        compton_angle = default_param_vals["COMPTON_ANGLE"]

    # Update default parameter values
    param_default_vals = copy(default_param_vals)
    param_default_vals.update(
        {
            "COHERENT_SCT_ENERGY": coherent_sct_energy,
            "ENERGY_SLOPE": energy_slope,
            "COMPTON_ANGLE": compton_angle,
        }
    )

    # Fit the spectrum using specified parameters
    fitted_tensors, fitted_spec, fitted_bkg, _ = fit_spec(
        int_spec_og,
        energy_range,
        elements_to_fit=elems,
        fitting_params=default_fitting_params,
        init_param_vals=param_default_vals,
        fixed_param_vals={},
        indices=None,
        tune_params=True,
        init_amp=init_amp,
        use_snip=use_snip,
        use_step=use_step,
        use_tail=False,
        loss=loss_selection,
        optimizer="adamw",
        n_iter=n_iter,
        progress_bar=verbose,
        device=device_selection,
        use_finite_diff=False,
        verbose=verbose,
    )

    # Extract and sort amplitudes of fitted elements
    amps = {p: fitted_tensors[p].item() for p in elems if p in fitted_tensors}
    amps = dict(sorted(amps.items(), key=lambda item: item[1]))

    # Save override parameters if requested
    if save_params:
        # Extract parameter values from fitted_tensors
        param_values = {
            param: fitted_tensors[param].item()
            for param in default_fitting_params
            if param in fitted_tensors
        }

        # Get elements list
        elements_to_save = [k for k in amps.keys()]

        # Save to file
        output_file = output_path or "maps_fit_parameters_override.txt"
        write_override_params_file(
            output_file, param_values=param_values, elements=elements_to_save
        )
        print(f"Parameters saved to {output_file}")

    # Show figures if requested
    if show_figures:
        # Suppress warnings and error messages for plot display
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Plot the amplitudes
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
            amp_fig.add_trace(
                go.Bar(x=list(amps.values()), name="Log scale"), row=1, col=2
            )
            amp_fig.update_yaxes(showticklabels=False, row=1, col=2)
            amp_fig.update_layout(showlegend=False)
            pio.show(amp_fig)

            # Plot the fitted spectrum
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

            pio.show(fit_fig)


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Print the error message
        sys.stderr.write(f"error: {message}\n")
        # Print the help message
        self.print_help()
        # Exit the program
        sys.exit(2)


if __name__ == "__main__":
    # Use the CustomArgumentParser
    parser = CustomArgumentParser(
        description="Function to process and fit spectrum data."
    )
    parser.add_argument("dataset", type=str, help="Path to the dataset HDF5 file")
    parser.add_argument("-e", "--incident_energy", type=float, help="Incident energy")
    parser.add_argument(
        "-n", "--n_iter", type=int, default=500, help="Number of iterations for fitting"
    )
    parser.add_argument(
        "-v", "--verbose", type=bool, default=True, help="Flag to print verbose output"
    )
    parser.add_argument(
        "--int_spec_path",
        type=str,
        default="MAPS/int_spec",
        help="Path of the integrated spectrum in the HDF5 file",
    )
    parser.add_argument(
        "--elem_path",
        type=str,
        default="MAPS/channel_names",
        help="Path of the element/channel names in the HDF5 file",
    )
    parser.add_argument(
        "--energy_range",
        type=int,
        nargs=2,
        default=[50, 1450],
        help="Energy range of interest for analysis",
    )
    parser.add_argument(
        "--init_amp", type=bool, default=True, help="Flag to initialize amplitude"
    )
    parser.add_argument(
        "--use_snip", type=bool, default=True, help="Flag to use the SNIP algorithm"
    )
    parser.add_argument(
        "--use_step", type=bool, default=True, help="Flag to use the step function"
    )
    parser.add_argument(
        "--loss_selection", type=str, default="mse", help="Loss function selection"
    )
    parser.add_argument(
        "--device_selection",
        type=str,
        default="cpu",
        help="Device selection for computation",
    )
    parser.add_argument(
        "--save_params",
        action="store_true",
        default=True,
        help="Save fitted parameters to an override file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the override parameters file (default: maps_fit_parameters_override.txt)",
    )
    parser.add_argument(
        "--show_figures",
        action="store_true",
        default=True,
        help="Display figures of the fitting results",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
