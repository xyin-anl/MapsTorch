import sys
from math import acos, pi
from copy import copy
import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from mapstorch.plot import (
    plot_elem_amp_rank,
    plot_elem_peak_ranges,
    plot_elem_spec_contribs,
    plot_elem_peak_pos,
    plot_spec_peaks,
    plot_specs,
)

from mapstorch.io import (
    parse_override_params_file,
    read_dataset,
    write_override_params_file,
)
from mapstorch.default import (
    default_elem_info,
    default_param_vals,
    default_fitting_params,
    default_energy_consts,
)
from mapstorch.constant import read_constants
from mapstorch.opt import fit_spec

PLOT_NAME_ALIASES = {
    "amp_rank": "amp_rank",
    "specs": "specs",
    "spec_peaks": "spec_peaks",
    "elem_peak_ranges": "elem_peak_ranges",
    "elem_spec_contribs": "elem_spec_contribs",
    "elem_peak_pos": "elem_peak_pos",
}
CANONICAL_PLOTS = tuple(sorted(set(PLOT_NAME_ALIASES.values())))
AVAILABLE_PLOT_CHOICES = tuple(
    sorted(set(PLOT_NAME_ALIASES.keys()).union({"all"}))
)


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
    plots_enabled = not args.disable_plots
    plot_output_dir = Path(args.plot_output_dir)
    if plots_enabled:
        plot_output_dir.mkdir(parents=True, exist_ok=True)
    raw_plot_choices = args.plots or []
    if "all" in raw_plot_choices:
        plot_selections = set(CANONICAL_PLOTS)
    else:
        plot_selections = {
            PLOT_NAME_ALIASES.get(choice, choice) for choice in raw_plot_choices
        }
        plot_selections &= set(CANONICAL_PLOTS)

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

    # Determine fitting configuration
    elements_to_fit = elems
    init_param_vals = param_default_vals
    e_consts = default_energy_consts

    if args.override_params_file:
        override_params = parse_override_params_file(args.override_params_file)
        e_consts = read_constants(args.override_params_file, default_elem_info)
        elements_with_pileup = override_params.get("elements_with_pileup", [])
        override_elements = override_params.get("elements_to_fit", [])
        elements_to_fit = override_elements + elements_with_pileup or elems
        init_param_vals = override_params

    # Fit the spectrum using specified parameters
    fitted_tensors, fitted_spec, fitted_bkg, _ = fit_spec(
        int_spec_og,
        energy_range,
        elements_to_fit=elements_to_fit,
        fitting_params=default_fitting_params,
        init_param_vals=init_param_vals,
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
        e_consts=e_consts,
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

    should_render_plots = plots_enabled and plot_selections

    # Generate plots if requested
    if should_render_plots:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            target_elems = list(amps.keys()) or elems

            def plot_path(default_name: str) -> str:
                return str(plot_output_dir / default_name)

            if "spec_peaks" in plot_selections and int_spec.size:
                prominence = max(float(np.max(int_spec)) / 200.0, 1.0)
                plot_spec_peaks(
                    int_spec,
                    peak_half_width=5,
                    prominence=prominence,
                    generate_plot=True,
                    save=True,
                    show=False,
                    filename=plot_path("identified_peaks.png"),
                )

            if "amp_rank" in plot_selections:
                plot_elem_amp_rank(
                    fitted_tensors,
                    target_elems=target_elems,
                    save=True,
                    show=False,
                    filename=plot_path("element_rank.png"),
                )

            if "specs" in plot_selections:
                spectra = [int_spec, fitted_bkg, fitted_spec + fitted_bkg]
                plot_specs(
                    spectra,
                    labels=["experiment", "background", "fitted"],
                    save=True,
                    show=False,
                    filename=plot_path("fitting_res.png"),
                )

            if "elem_peak_ranges" in plot_selections:
                plot_elem_peak_ranges(
                    fitted_tensors,
                    int_spec=int_spec_og,
                    energy_range=energy_range,
                    target_elems=target_elems,
                    save=True,
                    show=False,
                    filename=plot_path("element_peaks.png"),
                )

            if "elem_spec_contribs" in plot_selections:
                plot_elem_spec_contribs(
                    fitted_tensors,
                    int_spec=int_spec_og,
                    energy_range=energy_range,
                    target_elems=target_elems,
                    elem_amps=amps,
                    save=True,
                    show=False,
                    filename=plot_path("element_contribs.png"),
                )
            
            if "elem_peak_pos" in plot_selections:
                plot_elem_peak_pos(
                    fitted_tensors,
                    int_spec=int_spec_og,
                    energy_range=energy_range,
                    target_elems=target_elems,
                    save=True,
                    show=False,
                    filename=plot_path("element_peak_pos.png"),
                )

            print(f"Plot outputs saved to {plot_output_dir}")


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
    parser.add_argument("-e", "--incident_energy", type=float, required=True, help="Incident energy")
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
        default=False,
        help="Save fitted parameters to an override file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="maps_fit_parameters_override.txt",
        help="Path to save the override parameters file",
    )
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        default="plots",
        help="Directory to store generated plots",
    )
    parser.add_argument(
        "--disable_plots",
        action="store_true",
        default=False,
        help="Skip plot generation entirely",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=AVAILABLE_PLOT_CHOICES,
        default=["all"],
        help=(
            "Select which plots to generate when --disable_plots is set. "
            "Use 'all' (default) to display every available plot."
        ),
    )
    parser.add_argument(
        "--override_params_file",
        type=str,
        default=None,
        help="Path to a MAPS override parameters file used to seed fitting",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args)
