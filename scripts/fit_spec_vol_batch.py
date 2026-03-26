import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from mapstorch.batchfit import (
    coerce_energy_dim,
    decode_names,
    iter_tiles,
    normalize_tile_shape,
    resolve_h5_key,
    tile_grid_shape,
    unique_in_order,
)
from mapstorch.constant import read_constants
from mapstorch.default import default_elem_info, default_fitting_params
from mapstorch.io import parse_override_params_file, resolve_energy_range
from mapstorch.opt import fit_spec_batch


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        sys.exit(2)


def _ordered_amp_names(elements_to_fit: list[str]) -> list[str]:
    names = list(elements_to_fit)
    if "COMPTON_AMPLITUDE" not in names:
        names.append("COMPTON_AMPLITUDE")
    if "COHERENT_SCT_AMPLITUDE" not in names:
        names.append("COHERENT_SCT_AMPLITUDE")
    return names


def _tensor_tile_to_numpy(value: torch.Tensor | float | int, tile_shape: tuple[int, ...]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value, dtype=np.float32)

    if arr.shape == tuple(tile_shape) + (1,):
        arr = arr[..., 0]
    elif arr.shape == ():
        arr = np.full(tile_shape, float(arr), dtype=np.float32)
    elif arr.shape != tuple(tile_shape):
        raise ValueError(f"Unexpected tensor shape {arr.shape} for tile_shape={tile_shape}")
    return arr.astype(np.float32, copy=False)


def main(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available in this environment")

    dataset = args.dataset
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    override = parse_override_params_file(args.override_params_file)
    override["_source"] = args.override_params_file
    e_consts = read_constants(args.override_params_file, default_elem_info)

    detector_material = override.get("detector_material", args.detector_material)
    escape_factor = float(override.get("escape_factor", args.escape_factor))

    override_elems = list(override.get("elements_to_fit", []))
    pileup_elems = list(override.get("elements_with_pileup", []))

    with h5py.File(dataset, "r") as h5:
        spec_vol_key = resolve_h5_key(
            h5,
            args.spec_vol_path,
            ["MAPS/Spectra/mca_arr", "MAPS/mca_arr"],
            required=True,
            label="spectra volume",
        )
        dset = h5[spec_vol_key]
        energy_dim = coerce_energy_dim(int(args.energy_dim), int(dset.ndim))
        n_ch = int(dset.shape[energy_dim])
        energy_key = resolve_h5_key(
            h5,
            None,
            ["MAPS/energy", "MAPS/Spectra/Energy"],
            required=False,
            label="energy axis",
        )
        energy_axis = None if energy_key is None else h5[energy_key][:].astype(np.float32)
        energy_range = resolve_energy_range(
            args.energy_range,
            n_channels=n_ch,
            energy_values=energy_axis,
            override_params=override,
        )

        map_dims = [i for i in range(dset.ndim) if i != energy_dim]
        map_shape = tuple(int(dset.shape[i]) for i in map_dims)

        if override_elems:
            elements_to_fit = unique_in_order(override_elems + pileup_elems)
        else:
            chan_key = resolve_h5_key(
                h5,
                args.elem_path,
                ["MAPS/channel_names", "MAPS/XRF_Analyzed/Fitted/Channel_Names"],
                required=False,
                label="element names",
            )
            if chan_key is None:
                raise ValueError(
                    "Override file does not define ELEMENTS_TO_FIT and element names were not found in HDF5"
                )
            raw = decode_names(h5[chan_key][:])
            elements_to_fit = []
            for name in raw:
                if name in ("Num_Iter", "Fit_Residual", "Total_Fluorescence_Yield", "Sum_Elastic_Inelastic"):
                    continue
                if name not in elements_to_fit and name not in ("COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"):
                    elements_to_fit.append(name)

    unsupported_elements = [name for name in elements_to_fit if name not in e_consts]
    if unsupported_elements:
        print(
            "Skipping unsupported elements/pileups not present in loaded constants: "
            + ", ".join(unsupported_elements)
        )
    elements_to_fit = [name for name in elements_to_fit if name in e_consts]

    amp_names = _ordered_amp_names(elements_to_fit)
    all_param_names = list(default_fitting_params)
    if args.fit_params:
        invalid_fit_params = [
            p for p in args.fit_params if p.lower() != "all" and p not in default_fitting_params
        ]
        if invalid_fit_params:
            raise ValueError(f"Unknown fit_params: {invalid_fit_params}")
        if len(args.fit_params) == 1 and args.fit_params[0].lower() == "all":
            fitting_params = list(default_fitting_params)
        else:
            fitting_params = unique_in_order(
                p for p in args.fit_params if p in default_fitting_params
            )
    else:
        fitting_params = list(default_fitting_params)
    if args.per_spectrum_params:
        invalid_per_spec_params = [
            p
            for p in args.per_spectrum_params
            if p.lower() != "all" and p not in default_fitting_params
        ]
        if invalid_per_spec_params:
            raise ValueError(f"Unknown per_spectrum_params: {invalid_per_spec_params}")
        if len(args.per_spectrum_params) == 1 and args.per_spectrum_params[0].lower() == "all":
            per_spectrum_params = list(default_fitting_params)
        else:
            per_spectrum_params = unique_in_order(
                p for p in args.per_spectrum_params if p in default_fitting_params
            )
    else:
        per_spectrum_params = list(fitting_params)

    fixed_param_vals = {
        name: float(override[name])
        for name in all_param_names
        if name not in fitting_params and name in override
    }

    tile_shape = normalize_tile_shape(map_shape, args.tile_shape)

    tile_slices = list(iter_tiles(map_shape, tile_shape))
    if args.max_tiles is not None:
        tile_slices = tile_slices[: int(args.max_tiles)]

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    tile_grid = tile_grid_shape(map_shape, tile_shape)

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as out_h5:
        out_h5.attrs["dataset"] = str(dataset)
        out_h5.attrs["spec_vol_key"] = str(spec_vol_key)
        out_h5.attrs["override_params_file"] = str(args.override_params_file)
        out_h5.attrs["energy_dim"] = int(energy_dim)
        out_h5.attrs["energy_range_start"] = int(energy_range[0])
        out_h5.attrs["energy_range_stop"] = int(energy_range[1])
        out_h5.attrs["device"] = str(args.device)
        out_h5.attrs["optimizer"] = str(args.optimizer)
        out_h5.attrs["loss"] = str(args.loss)
        out_h5.attrs["n_iter"] = int(args.n_iter)
        out_h5.attrs["use_snip"] = bool(not args.no_snip)
        out_h5.attrs["init_amp"] = bool(not args.no_init_amp)
        out_h5.attrs["detector_material"] = str(detector_material)
        out_h5.attrs["escape_factor"] = float(escape_factor)
        out_h5.attrs["map_shape"] = map_shape
        out_h5.attrs["tile_shape"] = tile_shape
        out_h5.attrs["fit_params"] = ",".join(fitting_params)
        out_h5.attrs["per_spectrum_params"] = ",".join(per_spectrum_params)

        out_h5.create_dataset("amp_names", data=np.asarray(amp_names, dtype=object), dtype=string_dtype)
        out_h5.create_dataset("fit_params", data=np.asarray(fitting_params, dtype=object), dtype=string_dtype)
        out_h5.create_dataset("per_spectrum_params", data=np.asarray(per_spectrum_params, dtype=object), dtype=string_dtype)

        amp_maps = out_h5.create_dataset(
            "amp_maps_log10",
            shape=map_shape + (len(amp_names),),
            dtype=np.float32,
            compression="gzip",
        )

        param_group = out_h5.create_group("param_maps")
        for name in all_param_names:
            param_group.create_dataset(
                name,
                shape=map_shape,
                dtype=np.float32,
                compression="gzip",
            )

        tile_group = out_h5.create_group("tile_metrics")
        tile_group.create_dataset(
            "final_loss",
            shape=tile_grid,
            dtype=np.float32,
            compression="gzip",
            fillvalue=np.nan,
        )
        tile_group.create_dataset(
            "n_iter_ran",
            shape=tile_grid,
            dtype=np.int32,
            compression="gzip",
            fillvalue=0,
        )

        iterator = enumerate(tile_slices)
        if args.progress:
            iterator = tqdm(iterator, total=len(tile_slices), desc="Fitting tiles")

        with h5py.File(dataset, "r") as in_h5:
            dset = in_h5[spec_vol_key]
            map_dims = [i for i in range(dset.ndim) if i != energy_dim]

            for tile_index, map_slices in iterator:
                slicer = [slice(None)] * dset.ndim
                slicer[energy_dim] = slice(None)
                for md, sl in zip(map_dims, map_slices):
                    slicer[md] = sl

                block = dset[tuple(slicer)]
                block = np.moveaxis(block, energy_dim, -1).astype(np.float32, copy=False)
                block_tile_shape = block.shape[:-1]

                tensors, _, _, loss_trace = fit_spec_batch(
                    block,
                    energy_range,
                    elements_to_fit=elements_to_fit,
                    fitting_params=fitting_params,
                    per_spectrum_params=per_spectrum_params,
                    init_param_vals=override,
                    fixed_param_vals=fixed_param_vals,
                    tune_params=True,
                    init_amp=not args.no_init_amp,
                    use_snip=not args.no_snip,
                    loss=args.loss,
                    optimizer=args.optimizer,
                    n_iter=int(args.n_iter),
                    progress_bar=False,
                    device=args.device,
                    verbose=args.verbose,
                    e_consts=e_consts,
                    detector_type=detector_material,
                    escape_factor=escape_factor,
                    energy_dim=-1,
                )

                for amp_idx, amp_name in enumerate(amp_names):
                    if amp_name not in tensors:
                        raise KeyError(f"Expected amplitude '{amp_name}' in fit output")
                    amp_maps[map_slices + (amp_idx,)] = _tensor_tile_to_numpy(
                        tensors[amp_name], block_tile_shape
                    )

                for param_name in all_param_names:
                    param_group[param_name][map_slices] = _tensor_tile_to_numpy(
                        tensors[param_name], block_tile_shape
                    )

                tile_pos = tuple(sl.start // tile_shape[i] for i, sl in enumerate(map_slices))
                tile_group["final_loss"][tile_pos] = (
                    np.float32(loss_trace[-1]) if loss_trace else np.float32(np.nan)
                )
                tile_group["n_iter_ran"][tile_pos] = int(len(loss_trace))

                del tensors
                if args.device == "cuda":
                    torch.cuda.empty_cache()

        out_h5.flush()

    print(f"Saved tiled batch-fit results to {output_path}")


if __name__ == "__main__":
    parser = CustomArgumentParser(
        description="Fit a spectra volume tile-by-tile with fit_spec_batch for per-pixel amplitudes and parameters."
    )
    parser.add_argument("dataset", type=str, help="Path to the input HDF5 dataset")
    parser.add_argument(
        "--override_params_file",
        type=str,
        required=True,
        help="MAPS-style override parameter file used as initialization",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="HDF5 file to write amplitude maps and parameter maps",
    )
    parser.add_argument(
        "--spec_vol_path",
        type=str,
        default=None,
        help="HDF5 key for spectra volume (default: auto MAPS/Spectra/mca_arr or MAPS/mca_arr)",
    )
    parser.add_argument(
        "--elem_path",
        type=str,
        default=None,
        help="Optional HDF5 key for element names if the override file does not define them",
    )
    parser.add_argument(
        "--energy_dim",
        type=int,
        default=0,
        help="Index of the energy/channel axis in the spectra volume",
    )
    parser.add_argument(
        "--energy_range",
        type=int,
        nargs=2,
        default=None,
        help="Channel range (start stop), inclusive. If omitted, use override MIN/MAX energies when available.",
    )
    parser.add_argument(
        "--tile_shape",
        type=int,
        nargs="+",
        default=None,
        help="Tile shape for the non-energy dimensions. Default: 32x32 for 2D maps",
    )
    parser.add_argument(
        "--per_spectrum_params",
        nargs="+",
        default=["all"],
        help='Parameters to fit per pixel. Use "all" for all default fitting params.',
    )
    parser.add_argument(
        "--fit_params",
        nargs="+",
        default=["all"],
        help='Parameters to optimize. Use "all" for all default fitting params.',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Compute device: "cpu" or "cuda"',
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "lbfgs"],
        help="Optimizer for batch fitting",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "l1", "poisson_nll", "poisson_deviance", "chi2"],
        help="Loss function used for tile fitting",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=200,
        help="Optimization iterations per tile",
    )
    parser.add_argument(
        "--no_init_amp",
        action="store_true",
        default=False,
        help="Disable heuristic amplitude initialization",
    )
    parser.add_argument(
        "--no_snip",
        action="store_true",
        default=False,
        help="Disable SNIP background subtraction",
    )
    parser.add_argument(
        "--detector_material",
        type=str,
        default="Si",
        help='Fallback detector material if not provided by the override file',
    )
    parser.add_argument(
        "--escape_factor",
        type=float,
        default=0.0,
        help="Fallback escape factor if not provided by the override file",
    )
    parser.add_argument(
        "--max_tiles",
        type=int,
        default=None,
        help="Optional tile limit for pilot runs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the output file if it already exists",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Show a tile-level progress bar",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging inside fit_spec_batch",
    )
    main(parser.parse_args())
