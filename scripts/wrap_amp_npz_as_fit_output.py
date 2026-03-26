import argparse
from pathlib import Path

import h5py
import numpy as np

from mapstorch.batchfit import decode_names, decode_text, normalize_tile_shape, tile_grid_shape
from mapstorch.default import default_fitting_params, default_param_vals
from mapstorch.io import parse_override_params_file


def _base_param_dict(override: dict) -> dict[str, float]:
    params = dict(default_param_vals)
    for key in default_param_vals:
        if key in override:
            params[key] = float(override[key])
    return params


def main(args):
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(input_path, allow_pickle=True)
    amp_maps = np.asarray(data["amp_maps"], dtype=np.float32)
    if amp_maps.ndim < 2:
        raise ValueError("amp_maps must have at least one map dimension and one component axis")
    amp_names = decode_names(data["amp_names"])
    dataset = decode_text(data["dataset"])
    spec_vol_key = decode_text(data["spec_vol_key"])
    energy_dim = int(data["energy_dim"])
    energy_range = tuple(int(x) for x in data["energy_range"])
    return_log10 = bool(data["return_log10"])

    if not return_log10:
        amp_maps = np.log10(np.clip(amp_maps, float(args.amp_eps), None)).astype(np.float32, copy=False)

    override = parse_override_params_file(args.override_params_file)
    base_params = _base_param_dict(override)
    detector_material = str(override.get("detector_material", args.detector_material))
    escape_factor = float(override.get("escape_factor", args.escape_factor))

    map_shape = amp_maps.shape[:-1]
    tile_shape = normalize_tile_shape(map_shape, args.tile_shape)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output_path}. Pass --overwrite to replace it.")

    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(output_path, "w") as h5:
        h5.attrs["dataset"] = dataset
        h5.attrs["spec_vol_key"] = spec_vol_key
        h5.attrs["override_params_file"] = str(args.override_params_file)
        h5.attrs["energy_dim"] = int(energy_dim)
        h5.attrs["energy_range_start"] = int(energy_range[0])
        h5.attrs["energy_range_stop"] = int(energy_range[1])
        h5.attrs["device"] = "fixed_linear_amplitudes"
        h5.attrs["optimizer"] = "linear_solve"
        h5.attrs["loss"] = "n/a"
        h5.attrs["n_iter"] = 0
        h5.attrs["use_snip"] = bool(not args.no_snip)
        h5.attrs["init_amp"] = False
        h5.attrs["detector_material"] = detector_material
        h5.attrs["escape_factor"] = escape_factor
        h5.attrs["map_shape"] = map_shape
        h5.attrs["tile_shape"] = tile_shape
        h5.attrs["fit_params"] = ""
        h5.attrs["per_spectrum_params"] = ""

        h5.create_dataset("amp_names", data=np.asarray(amp_names, dtype=object), dtype=string_dtype)
        h5.create_dataset("fit_params", data=np.asarray([], dtype=object), dtype=string_dtype)
        h5.create_dataset("per_spectrum_params", data=np.asarray([], dtype=object), dtype=string_dtype)
        h5.create_dataset(
            "amp_maps_log10",
            data=amp_maps.astype(np.float32, copy=False),
            compression="gzip",
        )

        param_group = h5.create_group("param_maps")
        for name in default_fitting_params:
            param_group.create_dataset(
                name,
                data=np.full(map_shape, float(base_params[name]), dtype=np.float32),
                compression="gzip",
            )

        tile_group = h5.create_group("tile_metrics")
        grid_shape = tile_grid_shape(map_shape, tile_shape)
        tile_group.create_dataset("final_loss", data=np.full(grid_shape, np.nan, dtype=np.float32), compression="gzip")
        tile_group.create_dataset("n_iter_ran", data=np.zeros(grid_shape, dtype=np.int32), compression="gzip")

    print(f"Saved wrapped fit-style output to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrap an amplitude-only NPZ plus override params into the batch-fit HDF5 layout."
    )
    parser.add_argument("input_path", type=str, help="Amplitude NPZ from fit_spec_vol.py")
    parser.add_argument(
        "--override_params_file",
        type=str,
        required=True,
        help="Override file providing the fixed parameter values",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Wrapped HDF5 output path",
    )
    parser.add_argument(
        "--tile_shape",
        type=int,
        nargs="+",
        default=None,
        help="Tile shape metadata for downstream diagnostics",
    )
    parser.add_argument(
        "--detector_material",
        type=str,
        default="Si",
        help="Fallback detector material if not present in the override file",
    )
    parser.add_argument(
        "--escape_factor",
        type=float,
        default=0.0,
        help="Fallback escape factor if not present in the override file",
    )
    parser.add_argument(
        "--no_snip",
        action="store_true",
        default=False,
        help="Mark the wrapped output as having no SNIP background subtraction",
    )
    parser.add_argument(
        "--amp_eps",
        type=float,
        default=1e-12,
        help="Clamp used when converting linear amplitudes to log10",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite the wrapped output file if it already exists",
    )
    main(parser.parse_args())
