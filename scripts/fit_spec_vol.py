import sys
from copy import copy
import argparse
import warnings
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from mapstorch.bkg import snip_bkg
from mapstorch.constant import read_constants
from mapstorch.default import (
    default_elem_info,
    default_energy_consts,
    default_fitting_elems,
    default_fitting_params,
    default_param_vals,
)
from mapstorch.io import parse_override_params_file, write_override_params_file
from mapstorch.map import build_model_basis
from mapstorch.opt import fit_spec
from mapstorch.plot import (
    plot_amp_map,
    plot_amp_maps_grid,
    plot_elem_amp_rank,
    plot_elem_peak_pos,
    plot_elem_peak_ranges,
    plot_elem_spec_contribs,
    plot_specs,
)
from mapstorch.util import estimate_and_update_params

PLOT_NAME_ALIASES = {
    "amp_rank": "amp_rank",
    "specs": "specs",
    "elem_peak_ranges": "elem_peak_ranges",
    "elem_spec_contribs": "elem_spec_contribs",
    "elem_peak_pos": "elem_peak_pos",
    "maps_grid": "maps_grid",
    "maps_individual": "maps_individual",
}
CANONICAL_PLOTS = tuple(sorted(set(PLOT_NAME_ALIASES.values())))
AVAILABLE_PLOT_CHOICES = tuple(
    sorted(set(PLOT_NAME_ALIASES.keys()).union({"all"}))
)


def _resolve_h5_key(
    h5: h5py.File,
    preferred: str | None,
    fallbacks: list[str],
    *,
    required: bool,
    label: str,
) -> str | None:
    if preferred is not None:
        if preferred in h5:
            return preferred
        raise KeyError(f"Could not find {label} dataset key '{preferred}' in the HDF5 file")
    for key in fallbacks:
        if key in h5:
            return key
    if required:
        msg = f"Could not find {label} dataset in the HDF5 file. Tried: {fallbacks}"
        raise KeyError(msg)
    return None


def _read_elems(h5: h5py.File, elem_key: str | None) -> tuple[str | None, list[str] | None]:
    key = _resolve_h5_key(
        h5,
        elem_key,
        ["MAPS/channel_names", "MAPS/XRF_Analyzed/Fitted/Channel_Names"],
        required=False,
        label="element names",
    )
    if key is None:
        return None, None
    raw = h5[key][:]
    elems = []
    for item in raw:
        if isinstance(item, bytes):
            elems.append(item.decode("utf-8"))
        else:
            elems.append(str(item))
    if "COHERENT_SCT_AMPLITUDE" not in elems:
        elems.append("COHERENT_SCT_AMPLITUDE")
    if "COMPTON_AMPLITUDE" not in elems:
        elems.append("COMPTON_AMPLITUDE")
    return key, elems


def _read_int_spec(
    h5: h5py.File, int_spec_key: str | None, dtype=np.float32
) -> tuple[str | None, np.ndarray | None]:
    key = _resolve_h5_key(
        h5,
        int_spec_key,
        [
            "MAPS/int_spec",
            "MAPS/Spectra/Integrated_Spectra/Spectra",
            "MAPS/Spectra/Integrated_Spectra/Max_Channels_Integrated_Spectra",
        ],
        required=False,
        label="integrated spectrum",
    )
    if key is None:
        return None, None
    return key, h5[key][:].astype(dtype)


def _coerce_energy_dim(energy_dim: int, ndim: int) -> int:
    energy_dim = int(energy_dim)
    if energy_dim < 0:
        energy_dim = ndim + energy_dim
    if energy_dim < 0 or energy_dim >= ndim:
        raise ValueError(f"energy_dim={energy_dim} is out of bounds for ndim={ndim}")
    return energy_dim


def _default_tile_shape(map_shape: tuple[int, ...]) -> tuple[int, ...]:
    if len(map_shape) == 1:
        return (min(8192, int(map_shape[0])),)
    if len(map_shape) >= 2:
        lead = (1,) * max(0, len(map_shape) - 2)
        h = min(64, int(map_shape[-2]))
        w = min(64, int(map_shape[-1]))
        return lead + (h, w)
    raise ValueError("map_shape must have at least 1 dimension")


def _iter_tiles(map_shape: tuple[int, ...], tile_shape: tuple[int, ...]):
    from itertools import product

    if len(map_shape) != len(tile_shape):
        raise ValueError("tile_shape must match the number of non-energy dimensions")
    starts = [
        range(0, int(map_shape[d]), max(1, int(tile_shape[d])))
        for d in range(len(map_shape))
    ]
    for start_idx in product(*starts):
        slices = tuple(
            slice(s, min(int(map_shape[d]), s + max(1, int(tile_shape[d]))))
            for d, s in enumerate(start_idx)
        )
        yield slices


def _prepare_numeric_params(override_params: dict | None, incident_energy: float | None, *, heuristics: bool, int_spec: np.ndarray | None, energy_range: tuple[int, int], verbose: bool):
    params = copy(default_param_vals)
    if override_params:
        for k in default_param_vals:
            if k in override_params:
                params[k] = override_params[k]
        if verbose:
            src = str(override_params.get("_source", "override file"))
            print(f"Using init params from {src}")
        return params

    if incident_energy is not None and incident_energy > 0:
        params["COHERENT_SCT_ENERGY"] = float(incident_energy)
        if heuristics and int_spec is not None:
            updates = estimate_and_update_params(
                int_spec, energy_range, float(incident_energy), params
            )
            params.update(updates)
            if verbose and updates:
                print("Heuristic param updates:")
                for k, v in updates.items():
                    print(f"  {k}: {v}")
    return params


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


def _solve_mca_amplitudes_h5(
    h5_path: str,
    spec_vol_key: str,
    *,
    energy_dim: int,
    energy_range: tuple[int, int],
    params: dict,
    elements_to_fit: list[str],
    use_snip: bool,
    amp_method: str,
    ridge_lambda: float,
    nonneg: bool,
    nnls_max_iter: int,
    nnls_step: float | None,
    nnls_tol: float,
    return_log10: bool,
    eps: float,
    chunk_size: int,
    tile_shape: tuple[int, ...],
    device: str,
    e_consts: dict,
    detector_type: str,
    escape_factor: float,
    progress_bar: bool,
    verbose: bool,
):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        if verbose:
            print("CUDA is not available, using CPU instead")
    device_t = torch.device(device)
    chunk_size = _validate_positive_int(chunk_size, "chunk_size")
    tile_shape = tuple(int(x) for x in tile_shape)
    if any(x <= 0 for x in tile_shape):
        raise ValueError("tile_shape entries must be > 0")

    # Make sure params are scalar float tensors on the solve device.
    params_device: dict[str, torch.Tensor] = {}
    for k in default_param_vals:
        v = params.get(k, default_param_vals[k])
        if isinstance(v, torch.Tensor):
            params_device[k] = v.detach().to(device=device_t, dtype=torch.float32)
        else:
            params_device[k] = torch.tensor(float(v), device=device_t, dtype=torch.float32)

    basis, basis_names = build_model_basis(
        params_device,
        energy_range,
        elements_to_fit=elements_to_fit,
        device=device_t,
        e_consts=e_consts,
        elem_info=default_elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
        verbose=verbose,
    )
    basis = basis.to(device=device_t, dtype=torch.float32)
    n_comp = int(basis.shape[0])

    if amp_method not in ("ridge", "lstsq", "nnls_pgd"):
        raise ValueError('amp_method must be "ridge", "lstsq", or "nnls_pgd"')

    if amp_method in ("ridge", "nnls_pgd"):
        reg = float(ridge_lambda)
        eye = torch.eye(n_comp, device=device_t, dtype=torch.float32)
        gram = basis @ basis.T
        system = gram + reg * eye
        if amp_method == "nnls_pgd":
            if nnls_step is None:
                with torch.no_grad():
                    L = torch.linalg.eigvalsh(system).max().clamp_min(1e-12)
                    step = (1.0 / L).item()
            else:
                step = float(nnls_step)

    with h5py.File(h5_path, "r") as h5:
        if spec_vol_key not in h5:
            raise KeyError(f"Could not find spectra volume dataset key '{spec_vol_key}' in the HDF5 file")
        dset = h5[spec_vol_key]
        ndim = int(dset.ndim)
        energy_dim = _coerce_energy_dim(energy_dim, ndim)
        n_ch_full = int(dset.shape[energy_dim])
        ch0, ch1 = int(energy_range[0]), int(energy_range[1])
        if ch1 < ch0:
            raise ValueError("energy_range stop must be >= start")
        if ch0 < 0 or ch1 >= n_ch_full:
            raise ValueError(
                f"energy_range={energy_range} is out of bounds for n_channels={n_ch_full}"
            )

        map_dims = [i for i in range(ndim) if i != energy_dim]
        map_shape = tuple(int(dset.shape[i]) for i in map_dims)
        if len(tile_shape) != len(map_shape):
            raise ValueError(
                f"tile_shape must have length {len(map_shape)} (got {len(tile_shape)})"
            )

        amp_maps = np.empty((*map_shape, n_comp), dtype=np.float32)

        n_tiles = int(
            np.prod(
                [
                    (map_shape[d] + max(1, int(tile_shape[d])) - 1) // max(1, int(tile_shape[d]))
                    for d in range(len(map_shape))
                ]
            )
        )
        iterator = _iter_tiles(map_shape, tile_shape)
        if progress_bar:
            iterator = tqdm(iterator, total=n_tiles, desc="Solving amplitude tiles")

        for map_slices in iterator:
            # Build slicing tuple in the original dataset dimension order.
            slicer = [slice(None)] * ndim
            slicer[energy_dim] = slice(ch0, ch1 + 1)
            for md, sl in zip(map_dims, map_slices):
                slicer[md] = sl

            block = dset[tuple(slicer)]
            block = np.moveaxis(block, energy_dim, -1)
            block_map_shape = block.shape[:-1]
            n_spec = int(np.prod(block_map_shape)) if block_map_shape else 1
            spectra = block.reshape(n_spec, -1)

            amps_block = np.empty((n_spec, n_comp), dtype=np.float32)
            n_chunks = (n_spec + chunk_size - 1) // chunk_size
            for ci in range(n_chunks):
                start = ci * chunk_size
                stop = min(n_spec, start + chunk_size)
                y = torch.as_tensor(spectra[start:stop], device=device_t, dtype=torch.float32)

                if use_snip:
                    with torch.no_grad():
                        bkg = snip_bkg(
                            y,
                            energy_range,
                            params_device["ENERGY_OFFSET"],
                            params_device["ENERGY_SLOPE"],
                            params_device["ENERGY_QUADRATIC"],
                            params_device["SNIP_WIDTH"],
                            params_device["FWHM_OFFSET"],
                            params_device["FWHM_FANOPRIME"],
                            device=device_t,
                        )
                    y = y - bkg

                if amp_method == "ridge":
                    ybt = y @ basis.T
                    a = torch.linalg.solve(system, ybt.T).T
                elif amp_method == "nnls_pgd":
                    ybt = y @ basis.T
                    try:
                        a = torch.linalg.solve(system, ybt.T).T
                    except RuntimeError:
                        a = torch.zeros((y.shape[0], n_comp), device=device_t, dtype=torch.float32)
                    a = torch.clamp(a, min=0.0)

                    if nnls_tol and float(nnls_tol) > 0:
                        tol = float(nnls_tol)
                        denom = torch.linalg.norm(ybt).clamp_min(1e-12)
                    else:
                        tol = 0.0
                        denom = None

                    for _ in range(int(nnls_max_iter)):
                        grad = a @ system - ybt
                        next_a = torch.clamp(a - step * grad, min=0.0)
                        if tol:
                            diff = torch.linalg.norm(next_a - a) / denom
                            if diff.item() < tol:
                                a = next_a
                                break
                        a = next_a
                else:
                    sol = torch.linalg.lstsq(basis.T, y.T).solution
                    a = sol.T

                if nonneg:
                    a = torch.clamp(a, min=0.0)
                if return_log10:
                    a = torch.log10(torch.clamp(a, min=float(eps)))

                amps_block[start:stop] = a.detach().cpu().numpy().astype(np.float32, copy=False)

            amps_block = amps_block.reshape(*block_map_shape, n_comp)
            amp_maps[map_slices + (slice(None),)] = amps_block

    return amp_maps, basis_names


def _mean_component_values(amp_maps, amp_names, *, is_log10: bool):
    arr = np.asarray(amp_maps)
    flat = arr.reshape(-1, arr.shape[-1])
    if is_log10:
        with np.errstate(over="ignore", invalid="ignore"):
            lin = np.power(10.0, flat)
        lin = np.nan_to_num(lin, nan=0.0, posinf=0.0, neginf=0.0)
        mean_vals = np.mean(lin, axis=0)
    else:
        flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
        mean_vals = np.mean(np.clip(flat, 0.0, None), axis=0)
    return {name: float(v) for name, v in zip(amp_names, mean_vals)}


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"error: {message}\n")
        self.print_help()
        sys.exit(2)


def main(args):
    dataset = args.dataset
    spec_vol_path = args.spec_vol_path
    int_spec_path = args.int_spec_path
    elem_path = args.elem_path

    energy_dim = args.energy_dim
    energy_range = (int(args.energy_range[0]), int(args.energy_range[1]))
    incident_energy = args.incident_energy
    verbose = bool(args.verbose)

    device_selection = args.device_selection
    n_iter = int(args.n_iter)
    init_amp = not args.no_init_amp
    use_snip = not args.no_snip
    loss_selection = args.loss_selection
    optimizer = args.optimizer
    tune_globals = not args.no_tune_globals
    skip_global_fit = bool(args.skip_global_fit)
    heuristics_enabled = not args.disable_heuristics

    amp_method = args.amp_method
    ridge_lambda = float(args.ridge_lambda)
    nonneg = bool(args.nonneg)
    return_log10 = not args.linear_amps
    chunk_size = _validate_positive_int(args.chunk_size, "--chunk_size")
    nnls_max_iter = int(args.nnls_max_iter)
    nnls_step = args.nnls_step
    nnls_tol = float(args.nnls_tol)

    detector_material = args.detector_material
    escape_factor = float(args.escape_factor)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or Path(dataset).stem

    plots_enabled = not args.disable_plots
    plot_output_dir = Path(args.plot_output_dir)
    if plots_enabled:
        plot_output_dir.mkdir(parents=True, exist_ok=True)
    raw_plot_choices = args.plots or []
    if "all" in raw_plot_choices:
        plot_selections = set(CANONICAL_PLOTS)
    else:
        plot_selections = {PLOT_NAME_ALIASES.get(c, c) for c in raw_plot_choices}
        plot_selections &= set(CANONICAL_PLOTS)

    # Read lightweight metadata from HDF5.
    with h5py.File(dataset, "r") as h5:
        spec_vol_key = _resolve_h5_key(
            h5,
            spec_vol_path,
            ["MAPS/Spectra/mca_arr", "MAPS/mca_arr"],
            required=True,
            label="spectra volume",
        )
        elem_key_resolved, elems = _read_elems(h5, elem_path)
        elems = elems or []
        int_spec_key_resolved, int_spec_og = _read_int_spec(
            h5, int_spec_path, dtype=np.float32
        )

        dset = h5[spec_vol_key]
        energy_dim_resolved = _coerce_energy_dim(int(energy_dim), int(dset.ndim))
        n_ch = int(dset.shape[energy_dim_resolved])
        if energy_range[0] < 0 or energy_range[1] >= n_ch:
            raise ValueError(
                f"energy_range={energy_range} is out of bounds for n_channels={n_ch}"
            )

        if verbose:
            print(f"Spectra volume key: {spec_vol_key}")
            print(f"Spectra volume shape: {tuple(int(x) for x in dset.shape)} (energy_dim={energy_dim_resolved})")
            if int_spec_og is not None:
                print(
                    f"Integrated spectrum key: {int_spec_key_resolved} "
                    f"(n_channels={int(int_spec_og.size)})"
                )
            if elem_key_resolved is not None:
                print(f"Elements key: {elem_key_resolved} (n_elems={len(elems)})")

    override_params = None
    e_consts = default_energy_consts
    elements_with_pileup = []

    if args.override_params_file:
        override_params = parse_override_params_file(args.override_params_file)
        override_params["_source"] = args.override_params_file
        e_consts = read_constants(args.override_params_file, default_elem_info)
        detector_material = override_params.get("detector_material", detector_material)
        escape_factor = float(override_params.get("escape_factor", escape_factor))
        elements_with_pileup = list(override_params.get("elements_with_pileup", []))
        override_elements = list(override_params.get("elements_to_fit", []))
        elements_to_fit = override_elements + elements_with_pileup or elems
        if verbose:
            print(f"Override parameters loaded from {args.override_params_file}")
            print(f"Detector material: {detector_material}")
            print(f"Escape factor: {escape_factor}")
            print(f"Elements to fit: {elements_to_fit}")
    else:
        elements_to_fit = elems
        if verbose:
            print(f"Detector material: {detector_material}")
            print(f"Escape factor: {escape_factor}")

    if skip_global_fit and not override_params:
        raise ValueError("--skip_global_fit requires --override_params_file")

    params_init = _prepare_numeric_params(
        override_params,
        incident_energy,
        heuristics=heuristics_enabled,
        int_spec=int_spec_og,
        energy_range=energy_range,
        verbose=verbose,
    )

    fitted_tensors = None
    fitted_spec = None
    fitted_bkg = None
    loss_trace = None

    progress_bar = verbose and not args.no_progress

    if not skip_global_fit:
        if int_spec_og is None:
            raise KeyError(
                "Integrated spectrum not found in dataset. Provide --int_spec_path or store MAPS/int_spec."
            )

        fitted_tensors, fitted_spec, fitted_bkg, loss_trace = fit_spec(
            int_spec_og,
            energy_range,
            elements_to_fit=elements_to_fit,
            fitting_params=default_fitting_params,
            init_param_vals=params_init if override_params is None else override_params,
            fixed_param_vals={},
            indices=None,
            tune_params=tune_globals,
            init_amp=init_amp,
            use_snip=use_snip,
            loss=loss_selection,
            optimizer=optimizer,
            n_iter=n_iter,
            progress_bar=progress_bar,
            device=device_selection,
            verbose=verbose,
            e_consts=e_consts,
            detector_type=detector_material,
            escape_factor=escape_factor,
        )

    # Choose params for the amplitude solve.
    if fitted_tensors is not None:
        params_for_solve = fitted_tensors
    else:
        params_for_solve = params_init

    # Tile shape for streaming solve.
    with h5py.File(dataset, "r") as h5:
        dset = h5[spec_vol_key]
        map_shape = tuple(int(dset.shape[i]) for i in range(dset.ndim) if i != energy_dim_resolved)
    if args.tile_shape is None:
        tile_shape = _default_tile_shape(map_shape)
    else:
        tile_shape = tuple(int(x) for x in args.tile_shape)
        if any(x <= 0 for x in tile_shape):
            raise ValueError("--tile_shape values must be > 0")
    if verbose:
        print(f"Tile shape (non-energy dims): {tile_shape}")

    amp_maps, amp_names = _solve_mca_amplitudes_h5(
        dataset,
        spec_vol_key,
        energy_dim=energy_dim_resolved,
        energy_range=energy_range,
        params=params_for_solve,
        elements_to_fit=elements_to_fit,
        use_snip=use_snip,
        amp_method=amp_method,
        ridge_lambda=ridge_lambda,
        nonneg=nonneg,
        nnls_max_iter=nnls_max_iter,
        nnls_step=nnls_step,
        nnls_tol=nnls_tol,
        return_log10=return_log10,
        eps=float(args.amp_eps),
        chunk_size=chunk_size,
        tile_shape=tile_shape,
        device=device_selection,
        e_consts=e_consts,
        detector_type=detector_material,
        escape_factor=escape_factor,
        progress_bar=progress_bar,
        verbose=verbose,
    )

    # Save outputs.
    amp_npz_path = Path(args.amp_output_path) if args.amp_output_path else (output_dir / f"{output_prefix}_amp_maps.npz")
    np.savez_compressed(
        amp_npz_path,
        amp_maps=amp_maps.astype(np.float32, copy=False),
        amp_names=np.array(amp_names, dtype=object),
        dataset=str(dataset),
        spec_vol_key=str(spec_vol_key),
        energy_dim=int(energy_dim_resolved),
        energy_range=np.array(energy_range, dtype=int),
        return_log10=bool(return_log10),
    )
    print(f"Saved amplitude maps to {amp_npz_path}")

    if args.save_params:
        if fitted_tensors is not None:
            param_values = {
                p: float(fitted_tensors[p].item())
                for p in default_fitting_params
                if p in fitted_tensors
            }
        else:
            param_values = {p: float(params_init[p]) for p in default_fitting_params if p in params_init}

        params_path = Path(args.params_output_path) if args.params_output_path else (output_dir / f"{output_prefix}_maps_fit_parameters_override_mapstorch.txt")
        elements_to_save = [e for e in amp_names if e not in set(elements_with_pileup)]
        invalid = [e for e in elements_to_save if e not in default_fitting_elems]
        if invalid:
            print(
                "Warning: dropping unsupported elements from override output: "
                + ", ".join(invalid)
            )
        elements_to_save = [e for e in elements_to_save if e in default_fitting_elems]
        write_override_params_file(
            str(params_path),
            param_values=param_values,
            elements=elements_to_save,
            elements_with_pileup=elements_with_pileup,
            detector_material=detector_material,
            escape_factor=escape_factor,
        )
        print(f"Saved override parameters to {params_path}")

    should_render_plots = plots_enabled and plot_selections
    if should_render_plots:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            def plot_path(default_name: str) -> str:
                return str(plot_output_dir / f"{output_prefix}_{default_name}")

            if "maps_grid" in plot_selections:
                plot_amp_maps_grid(
                    amp_maps,
                    amp_names,
                    target_elems=args.map_elems,
                    n_elem=args.map_n_elem,
                    reverse=True,
                    map_index=args.map_index,
                    is_log10=return_log10,
                    display=args.map_display,
                    percentile=tuple(args.map_percentile),
                    n_cols=args.map_n_cols,
                    cmap=args.map_cmap,
                    colorbar=args.map_colorbar,
                    save=True,
                    show=False,
                    filename=plot_path("amp_maps_grid.png"),
                )

            if "maps_individual" in plot_selections:
                # Determine which components to plot (same selection logic as grid).
                if args.map_elems:
                    selected = list(args.map_elems)
                else:
                    mean_vals = _mean_component_values(amp_maps, amp_names, is_log10=return_log10)
                    order = sorted(mean_vals.items(), key=lambda kv: kv[1], reverse=True)
                    selected = [k for k, _ in order[: int(args.map_n_elem)]]
                name_to_idx = {n: i for i, n in enumerate(amp_names)}
                for name in selected:
                    if name not in name_to_idx:
                        continue
                    idx = name_to_idx[name]
                    view = amp_maps
                    if view.ndim > 3:
                        lead_ndim = view.ndim - 3
                        map_index = args.map_index or [0] * lead_ndim
                        if len(map_index) != lead_ndim:
                            raise ValueError(
                                f"--map_index must have length {lead_ndim} for amp_maps.ndim={view.ndim}"
                            )
                        slicer = tuple(int(i) for i in map_index) + (
                            slice(None),
                            slice(None),
                            slice(None),
                        )
                        view = view[slicer]
                    img = view[..., idx]
                    display = args.map_display
                    if display == "auto":
                        display = "log10" if return_log10 else "linear"
                    if display == "linear":
                        if return_log10:
                            with np.errstate(over="ignore", invalid="ignore"):
                                img = np.power(10.0, img)
                        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
                        img = np.clip(img, 0.0, None)
                    else:
                        if not return_log10:
                            img = np.log10(np.clip(img, float(args.amp_eps), None))
                    plot_amp_map(
                        img,
                        name=name,
                        save=True,
                        show=False,
                        filename=plot_path(f"map_{name}.png"),
                        cmap=args.map_cmap,
                        percentile=tuple(args.map_percentile),
                        colorbar=True,
                    )

            # Integrated-spectrum diagnostics (only when global fit is run).
            if fitted_tensors is not None:
                int_spec = int_spec_og[energy_range[0] : energy_range[1] + 1]

                if "specs" in plot_selections:
                    spectra = [int_spec, fitted_bkg, fitted_spec + fitted_bkg]
                    plot_specs(
                        spectra,
                        labels=["experiment", "background", "fitted"],
                        save=True,
                        show=False,
                        filename=plot_path("fitting_res.png"),
                    )

                if "amp_rank" in plot_selections:
                    plot_elem_amp_rank(
                        fitted_tensors,
                        target_elems=[e for e in elements_to_fit if e in fitted_tensors],
                        save=True,
                        show=False,
                        filename=plot_path("element_rank.png"),
                    )

                if "elem_peak_ranges" in plot_selections:
                    plot_elem_peak_ranges(
                        fitted_tensors,
                        int_spec=int_spec_og,
                        energy_range=energy_range,
                        target_elems=list(elements_to_fit),
                        save=True,
                        show=False,
                        filename=plot_path("element_peaks.png"),
                    )

                if "elem_spec_contribs" in plot_selections:
                    amps = {p: float(fitted_tensors[p].item()) for p in elements_to_fit if p in fitted_tensors}
                    plot_elem_spec_contribs(
                        fitted_tensors,
                        int_spec=int_spec_og,
                        energy_range=energy_range,
                        target_elems=list(elements_to_fit),
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
                        target_elems=list(elements_to_fit),
                        save=True,
                        show=False,
                        filename=plot_path("element_peak_pos.png"),
                    )

            # Amplitude ranking by mean map signal (works even when skip_global_fit).
            if fitted_tensors is None and "amp_rank" in plot_selections:
                mean_vals = _mean_component_values(amp_maps, amp_names, is_log10=return_log10)
                pseudo = {k: torch.tensor(float(v)) for k, v in mean_vals.items()}
                plot_elem_amp_rank(
                    pseudo,
                    target_elems=list(mean_vals.keys()),
                    save=True,
                    show=False,
                    filename=plot_path("element_rank_mapmean.png"),
                )

            print(f"Plot outputs saved to {plot_output_dir}")


if __name__ == "__main__":
    parser = CustomArgumentParser(
        description="Fit a spectra volume (MCA cube) using MapsTorch: global fit + per-pixel amplitude solve."
    )

    parser.add_argument("dataset", type=str, help="Path to the dataset HDF5 file")
    parser.add_argument(
        "-e",
        "--incident_energy",
        type=float,
        default=None,
        help="Incident energy (keV). Used for heuristics and as COHERENT_SCT_ENERGY seed when no override file is provided.",
    )
    parser.add_argument(
        "--spec_vol_path",
        type=str,
        default=None,
        help="HDF5 key for spectra volume (default: auto MAPS/Spectra/mca_arr or MAPS/mca_arr).",
    )
    parser.add_argument(
        "--int_spec_path",
        type=str,
        default=None,
        help="HDF5 key for integrated spectrum (default: auto MAPS/int_spec).",
    )
    parser.add_argument(
        "--elem_path",
        type=str,
        default=None,
        help="HDF5 key for element/channel names (default: auto MAPS/channel_names).",
    )
    parser.add_argument(
        "--energy_dim",
        type=int,
        default=0,
        help="Which dimension in spectra volume is the channel axis (default: 0).",
    )
    parser.add_argument(
        "--energy_range",
        type=int,
        nargs=2,
        default=[50, 1450],
        help="Channel range (start stop) inclusive.",
    )
    parser.add_argument(
        "--device_selection",
        type=str,
        default="cpu",
        help='Device selection: "cpu" or "cuda".',
    )
    parser.add_argument(
        "-n",
        "--n_iter",
        type=int,
        default=500,
        help="Number of iterations for global (integrated spectrum) fitting.",
    )
    parser.add_argument(
        "--loss_selection",
        type=str,
        default="mse",
        help="Loss function for global fitting.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "lbfgs"],
        help="Optimizer for global fitting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logs and progress bars.",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        default=False,
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--no_init_amp",
        action="store_true",
        default=False,
        help="Disable heuristic element amplitude initialization in the global fit.",
    )
    parser.add_argument(
        "--no_snip",
        action="store_true",
        default=False,
        help="Disable SNIP background subtraction.",
    )
    parser.add_argument(
        "--no_tune_globals",
        action="store_true",
        default=False,
        help="Only fit element amplitudes in the global fit (keep global params fixed).",
    )
    parser.add_argument(
        "--disable_heuristics",
        action="store_true",
        default=False,
        help="Disable scatter-peak heuristics (ENERGY_SLOPE / COMPTON_ANGLE seeds).",
    )
    parser.add_argument(
        "--skip_global_fit",
        action="store_true",
        default=False,
        help="Skip global fit and only solve amplitudes (requires --override_params_file).",
    )
    parser.add_argument(
        "--override_params_file",
        type=str,
        default=None,
        help="MAPS-style override parameters file used to seed global fit or to run solve-only mode.",
    )
    parser.add_argument(
        "--detector_material",
        type=str,
        default="Si",
        help='Detector material: "Si" or "Ge". Overridden by override file if provided.',
    )
    parser.add_argument(
        "--escape_factor",
        type=float,
        default=0.0,
        help="Escape peak factor. Overridden by override file if provided.",
    )

    # Amplitude solve settings
    parser.add_argument(
        "--amp_method",
        type=str,
        default="ridge",
        choices=["ridge", "lstsq", "nnls_pgd"],
        help="Amplitude solver method.",
    )
    parser.add_argument(
        "--ridge_lambda",
        type=float,
        default=1e-3,
        help="Ridge regularization strength (amp_method=ridge/nnls_pgd).",
    )
    parser.add_argument(
        "--nonneg",
        action="store_true",
        default=False,
        help="Clamp negative amplitudes to 0 after solving.",
    )
    parser.add_argument(
        "--linear_amps",
        action="store_true",
        default=False,
        help="Return/save linear amplitudes (default: log10 amplitudes).",
    )
    parser.add_argument(
        "--amp_eps",
        type=float,
        default=1e-12,
        help="Clamp epsilon when converting linear amps to log10.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=8192,
        help="Number of spectra per solve chunk (inside each tile).",
    )
    parser.add_argument(
        "--tile_shape",
        type=int,
        nargs="+",
        default=None,
        help="Tile shape for streaming HDF5 reads (non-energy dims, in stored order).",
    )
    parser.add_argument(
        "--nnls_max_iter",
        type=int,
        default=200,
        help="Max iterations for NNLS-PGD (amp_method=nnls_pgd).",
    )
    parser.add_argument(
        "--nnls_step",
        type=float,
        default=None,
        help="Optional fixed PGD step for NNLS-PGD.",
    )
    parser.add_argument(
        "--nnls_tol",
        type=float,
        default=0.0,
        help="Optional relative convergence tolerance for NNLS-PGD (0 disables).",
    )

    # Outputs
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to store results (.npz and optional override params).",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Filename prefix for outputs (default: dataset stem).",
    )
    parser.add_argument(
        "--amp_output_path",
        type=str,
        default=None,
        help="Path to save amplitude maps (.npz). Default: {output_dir}/{prefix}_amp_maps.npz",
    )
    parser.add_argument(
        "--save_params",
        action="store_true",
        default=False,
        help="Save global fitted parameters to an override file.",
    )
    parser.add_argument(
        "--params_output_path",
        type=str,
        default=None,
        help="Path to save the override parameters file.",
    )

    # Plotting
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        default="plots",
        help="Directory to store generated plots.",
    )
    parser.add_argument(
        "--disable_plots",
        action="store_true",
        default=False,
        help="Skip plot generation entirely.",
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=AVAILABLE_PLOT_CHOICES,
        default=["maps_grid", "specs", "amp_rank", "elem_peak_pos"],
        help="Select which plots to generate. Use 'all' for every available plot.",
    )
    parser.add_argument(
        "--map_elems",
        nargs="+",
        default=None,
        help="Optional list of component names to plot for maps.",
    )
    parser.add_argument(
        "--map_n_elem",
        type=int,
        default=12,
        help="Number of top components (by mean amplitude) to plot when --map_elems is not provided.",
    )
    parser.add_argument(
        "--map_index",
        type=int,
        nargs="+",
        default=None,
        help="Indices for leading map dims when amp_maps has >2 spatial dims (e.g., time index).",
    )
    parser.add_argument(
        "--map_display",
        type=str,
        default="auto",
        choices=["auto", "log10", "linear"],
        help='Map display mode: "log10" or "linear". "auto" uses log10 when saving log10 amps.',
    )
    parser.add_argument(
        "--map_percentile",
        type=float,
        nargs=2,
        default=[1, 99],
        help="Percentile clip (low high) for map display scaling.",
    )
    parser.add_argument(
        "--map_n_cols",
        type=int,
        default=4,
        help="Number of columns in the map grid plot.",
    )
    parser.add_argument(
        "--map_cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for elemental maps.",
    )
    parser.add_argument(
        "--map_colorbar",
        action="store_true",
        default=False,
        help="Add a colorbar to each subplot in the map grid (can be cluttered).",
    )

    args = parser.parse_args()
    main(args)
