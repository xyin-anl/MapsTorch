import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from mapstorch.batchfit import (
    coerce_energy_dim,
    decode_names,
    decode_text,
    display_component_name,
    iter_tiles,
    resolve_h5_key,
    sanitize_component_name,
)
from mapstorch.bkg import snip_bkg
from mapstorch.constant import read_constants
from mapstorch.default import default_elem_info, default_param_vals
from mapstorch.io import parse_override_params_file
from mapstorch.map import compton_peak, elastic_peak, model_elem_spec, model_spec
from mapstorch.plot import plot_amp_map


def _base_param_dict(override: dict) -> dict[str, float]:
    params = dict(default_param_vals)
    for key in default_param_vals:
        if key in override:
            params[key] = float(override[key])
    return params


def _tile_tensors(
    base_params: dict[str, float],
    param_maps: dict[str, np.ndarray],
    amp_maps: np.ndarray,
    amp_names: list[str],
    map_slices: tuple[slice, ...],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in base_params.items():
        if key in param_maps:
            arr = param_maps[key][map_slices].astype(np.float32, copy=False)
            tensors[key] = torch.as_tensor(arr, device=device, dtype=torch.float32).unsqueeze(-1)
        else:
            tensors[key] = torch.tensor(float(value), device=device, dtype=torch.float32)
    for idx, name in enumerate(amp_names):
        arr = amp_maps[map_slices + (idx,)].astype(np.float32, copy=False)
        tensors[name] = torch.as_tensor(arr, device=device, dtype=torch.float32).unsqueeze(-1)
    return tensors


def _pixel_tensors(
    base_params: dict[str, float],
    param_maps: dict[str, np.ndarray],
    amp_maps: np.ndarray,
    amp_names: list[str],
    y: int,
    x: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, value in base_params.items():
        if key in param_maps:
            tensors[key] = torch.tensor(float(param_maps[key][y, x]), device=device, dtype=torch.float32)
        else:
            tensors[key] = torch.tensor(float(value), device=device, dtype=torch.float32)
    for idx, name in enumerate(amp_names):
        tensors[name] = torch.tensor(float(amp_maps[y, x, idx]), device=device, dtype=torch.float32)
    return tensors


def _compute_background(
    obs_t: torch.Tensor,
    energy_range: tuple[int, int],
    tensors: dict[str, torch.Tensor],
    *,
    use_snip: bool,
    device: torch.device,
) -> torch.Tensor:
    if not use_snip:
        return torch.zeros_like(obs_t, device=device)
    return snip_bkg(
        obs_t,
        energy_range,
        tensors["ENERGY_OFFSET"],
        tensors["ENERGY_SLOPE"],
        tensors["ENERGY_QUADRATIC"],
        tensors["SNIP_WIDTH"],
        tensors["FWHM_OFFSET"],
        tensors["FWHM_FANOPRIME"],
        device=device,
        use_single_log=tensors["SNIP_USE_SINGLE_LOG"],
        use_fwhm_width=tensors["SNIP_USE_FWHM_WIDTH"],
    )


def _plot_spectra_grid(pixel_records: list[dict], output_path: Path):
    n = len(pixel_records)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.1 * n), squeeze=False)

    for row, rec in enumerate(pixel_records):
        obs = rec["obs"]
        bkg = rec["bkg"]
        total = rec["total"]
        norm_resid = rec["norm_resid"]
        x = np.arange(obs.size)
        title = (
            f"{rec['label']} (y={rec['y']}, x={rec['x']}), "
            f"SSE={rec['sse']:.3g}, MAE={rec['mae']:.3g}, "
            f"NRMSE={rec['nrmse_frac']:.3g}"
        )

        ax = axes[row][0]
        ax.plot(x, obs, label="experiment", color="0.35", lw=1.2)
        ax.plot(x, bkg, label="background", color="silver", lw=1.2)
        ax.plot(x, total, label="bkg + fitted", color="tab:blue", lw=1.2)
        ax.plot(x, obs - total, label="residual", color="tab:red", lw=1.0, alpha=0.8)
        ax.set_title(title)
        ax.set_ylabel("Counts")
        if row == n - 1:
            ax.set_xlabel("ROI Channel Index")
        ax.legend(loc="upper right", fontsize=8)

        ax = axes[row][1]
        ax.plot(x, obs, label="experiment", color="0.35", lw=1.2)
        ax.plot(x, bkg, label="background", color="silver", lw=1.2)
        ax.plot(x, total, label="bkg + fitted", color="tab:blue", lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1, None)
        if row == n - 1:
            ax.set_xlabel("ROI Channel Index")
        ax.set_ylabel("Counts")

        ax = axes[row][2]
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.plot(x, norm_resid, color="tab:red", lw=1.1)
        ax.set_ylabel("Residual / Total Counts")
        if row == n - 1:
            ax.set_xlabel("ROI Channel Index")
        finite = norm_resid[np.isfinite(norm_resid)]
        if finite.size:
            lim = np.percentile(np.abs(finite), 99)
            lim = max(lim, 1e-6)
            ax.set_ylim(-1.1 * lim, 1.1 * lim)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _plot_pixel_locations(metric_map: np.ndarray, pixel_records: list[dict], output_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    finite = metric_map[np.isfinite(metric_map)]
    if finite.size:
        vmin, vmax = np.percentile(finite, [1, 99]).astype(float)
    else:
        vmin, vmax = 0.0, 1.0
    im = ax.imshow(metric_map, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    for rec in pixel_records:
        ax.scatter(rec["x"], rec["y"], s=40, facecolors="none", edgecolors="cyan", linewidths=1.5)
        ax.text(rec["x"] + 2, rec["y"] + 2, rec["label"], color="white", fontsize=9)
    ax.set_title("Selected Pixels on Fit-Metric Map")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.85, label="Per-pixel SSE")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def _far_enough(
    seen: set[tuple[int, int]],
    y: int,
    x: int,
    *,
    min_separation: float,
) -> bool:
    if min_separation <= 0:
        return True
    for sy, sx in seen:
        if (float(y) - float(sy)) ** 2 + (float(x) - float(sx)) ** 2 < float(min_separation) ** 2:
            return False
    return True


def _unique_append(
    records: list[dict],
    seen: set[tuple[int, int]],
    label: str,
    y: int,
    x: int,
    *,
    min_separation: float = 0.0,
):
    coord = (int(y), int(x))
    if coord in seen:
        return
    if not _far_enough(seen, int(y), int(x), min_separation=min_separation):
        return
    seen.add(coord)
    records.append({"label": label, "y": int(y), "x": int(x)})


def _fill_unique_from_ranked(
    records: list[dict],
    seen: set[tuple[int, int]],
    ranked_indices: np.ndarray,
    map_shape: tuple[int, int],
    *,
    prefix: str,
    limit: int,
    min_separation: float,
):
    counter = 1
    for flat_idx in ranked_indices:
        if len(records) >= limit:
            break
        y, x = np.unravel_index(int(flat_idx), map_shape)
        coord = (int(y), int(x))
        if coord in seen:
            continue
        prev_len = len(records)
        _unique_append(
            records,
            seen,
            f"{prefix}_{counter:02d}",
            y,
            x,
            min_separation=min_separation,
        )
        if len(records) > prev_len:
            counter += 1


def _component_spec(
    tensors: dict[str, torch.Tensor],
    name: str,
    ev: torch.Tensor,
    energy_slope: torch.Tensor,
    *,
    device: torch.device,
    e_consts,
) -> torch.Tensor:
    if name == "COHERENT_SCT_AMPLITUDE":
        return elastic_peak(tensors, ev, energy_slope)
    if name == "COMPTON_AMPLITUDE":
        return compton_peak(tensors, ev, energy_slope, True, False)
    try:
        return model_elem_spec(
            tensors,
            name,
            ev,
            True,
            False,
            device=device,
            e_consts=e_consts,
            elem_info=default_elem_info,
        )
    except Exception:
        return torch.zeros_like(ev, device=device)


def _map_cmap(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "viridis"
    lo, hi = np.percentile(finite, [1, 99]).astype(float)
    if lo < 0.0 < hi:
        return "RdBu_r"
    return "magma"


def _plot_named_grid(
    maps: dict[str, np.ndarray],
    *,
    title: str,
    filename: Path,
    percentile: tuple[float, float] = (1.0, 99.0),
    n_cols: int = 4,
    display_names: dict[str, str] | None = None,
):
    names = list(maps.keys())
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.3 * n_rows), squeeze=False)
    fig.suptitle(title, y=0.995, fontsize=14)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    for idx, name in enumerate(names):
        ax = axes[idx // n_cols][idx % n_cols]
        img = np.asarray(maps[name], dtype=np.float32)
        finite = img[np.isfinite(img)]
        if finite.size:
            vmin, vmax = np.percentile(finite, percentile).astype(float)
            if vmax <= vmin:
                pad = max(1e-12, 0.05 * max(abs(vmin), 1.0))
                vmin -= pad
                vmax += pad
        else:
            vmin, vmax = 0.0, 1.0
        aspect = "auto" if 1 in img.shape else None
        im = ax.imshow(
            img,
            origin="lower",
            cmap=_map_cmap(img),
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
        )
        ax.set_title(display_names.get(name, name) if display_names is not None else name)
        fig.colorbar(im, ax=ax, shrink=0.85)

    for idx in range(len(names), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def main(args):
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir = output_dir / "spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        dataset_path = decode_text(h5.attrs["dataset"])
        override_path = decode_text(h5.attrs["override_params_file"])
        spec_vol_key_attr = decode_text(h5.attrs["spec_vol_key"])
        energy_dim = int(h5.attrs["energy_dim"])
        energy_range = (
            int(h5.attrs["energy_range_start"]),
            int(h5.attrs["energy_range_stop"]),
        )
        use_snip = bool(h5.attrs.get("use_snip", True))
        detector_material = decode_text(h5.attrs.get("detector_material", "Si"))
        escape_factor = float(h5.attrs.get("escape_factor", 0.0))
        tile_shape = tuple(int(x) for x in h5.attrs["tile_shape"])
        amp_maps = h5["amp_maps_log10"][:]
        amp_names = decode_names(h5["amp_names"][:])
        amp_display_names = [display_component_name(name) for name in amp_names]
        amp_display_map = dict(zip(amp_names, amp_display_names, strict=True))
        param_names = sorted(h5["param_maps"].keys())
        param_maps = {name: h5["param_maps"][name][:] for name in param_names}

    override = parse_override_params_file(override_path)
    e_consts = read_constants(override_path, default_elem_info)
    base_params = _base_param_dict(override)
    map_shape = amp_maps.shape[:-1]
    if len(map_shape) != 2:
        raise ValueError(
            f"{input_path} contains map_shape={map_shape}. "
            "plot_batchfit_diagnostics.py currently supports 2D maps only."
        )

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    n_rows = map_shape[0]
    n_cols = map_shape[1]
    sse_map = np.zeros(map_shape, dtype=np.float32)
    mae_map = np.zeros(map_shape, dtype=np.float32)
    mean_residual_map = np.zeros(map_shape, dtype=np.float32)
    total_counts_map = np.zeros(map_shape, dtype=np.float32)
    rmse_frac_map = np.zeros(map_shape, dtype=np.float32)
    l1_frac_map = np.zeros(map_shape, dtype=np.float32)
    bias_frac_map = np.zeros(map_shape, dtype=np.float32)
    sse_over_counts2_map = np.zeros(map_shape, dtype=np.float32)
    sse_delta_maps = np.zeros(map_shape + (len(amp_names),), dtype=np.float32)
    sse_delta_over_counts2_maps = np.zeros(map_shape + (len(amp_names),), dtype=np.float32)

    spec_vol_path = None if spec_vol_key_attr == "None" else spec_vol_key_attr

    with h5py.File(dataset_path, "r") as h5:
        spec_vol_key = resolve_h5_key(
            h5,
            spec_vol_path,
            ["MAPS/Spectra/mca_arr", "MAPS/mca_arr"],
            required=True,
            label="spectra volume",
        )
        dset = h5[spec_vol_key]
        energy_dim_resolved = coerce_energy_dim(energy_dim, dset.ndim)
        map_dims = [i for i in range(dset.ndim) if i != energy_dim_resolved]

        iterator = iter_tiles(map_shape, tile_shape)
        if args.progress:
            n_tiles = int(np.prod([(map_shape[d] + tile_shape[d] - 1) // tile_shape[d] for d in range(len(map_shape))]))
            iterator = tqdm(iterator, total=n_tiles, desc="Computing diagnostics")

        for map_slices in iterator:
            slicer = [slice(None)] * dset.ndim
            slicer[energy_dim_resolved] = slice(energy_range[0], energy_range[1] + 1)
            for md, sl in zip(map_dims, map_slices):
                slicer[md] = sl
            obs_block = dset[tuple(slicer)]
            obs_block = np.moveaxis(obs_block, energy_dim_resolved, -1).astype(np.float32, copy=False)
            obs_t = torch.as_tensor(obs_block, device=device, dtype=torch.float32)

            tensors = _tile_tensors(
                base_params,
                param_maps,
                amp_maps,
                amp_names,
                map_slices,
                device,
            )

            with torch.no_grad():
                energy = torch.arange(
                    energy_range[0],
                    energy_range[1] + 1,
                    device=device,
                    dtype=torch.float32,
                )
                energy = energy.view((1,) * len(map_slices) + (energy.numel(),))
                ev = (
                    tensors["ENERGY_OFFSET"]
                    + tensors["ENERGY_SLOPE"] * energy
                    + tensors["ENERGY_QUADRATIC"] * (energy**2)
                )
                spec_fit = model_spec(
                    tensors,
                    energy_range,
                    elements_to_fit=amp_names,
                    device=device,
                    e_consts=e_consts,
                    elem_info=default_elem_info,
                    detector_type=detector_material,
                    escape_factor=escape_factor,
                )
                bkg = _compute_background(
                    obs_t,
                    energy_range,
                    tensors,
                    use_snip=use_snip,
                    device=device,
                )
                total = spec_fit + bkg
                resid = obs_t - total
                sse = torch.sum(resid * resid, dim=-1)
                mae = torch.mean(torch.abs(resid), dim=-1)
                mean_resid = torch.mean(resid, dim=-1)
                total_counts = torch.sum(obs_t, dim=-1)
                total_counts_safe = total_counts.clamp_min(1.0)
                total_counts_safe_sq = total_counts_safe * total_counts_safe
                sse_over_counts2 = sse / total_counts_safe_sq
                rmse_frac = torch.sqrt(sse) / total_counts_safe
                l1_frac = torch.sum(torch.abs(resid), dim=-1) / total_counts_safe
                bias_frac = torch.sum(resid, dim=-1) / total_counts_safe

                for comp_idx, comp_name in enumerate(amp_names):
                    comp_spec = _component_spec(
                        tensors,
                        comp_name,
                        ev,
                        tensors["ENERGY_SLOPE"],
                        device=device,
                        e_consts=e_consts,
                    )
                    delta_sse = torch.sum((resid + comp_spec) ** 2 - resid**2, dim=-1)
                    sse_delta_maps[map_slices + (comp_idx,)] = (
                        delta_sse.detach().cpu().numpy().astype(np.float32, copy=False)
                    )
                    sse_delta_over_counts2_maps[map_slices + (comp_idx,)] = (
                        (delta_sse / total_counts_safe_sq).detach().cpu().numpy().astype(np.float32, copy=False)
                    )

            sse_map[map_slices] = sse.detach().cpu().numpy().astype(np.float32, copy=False)
            mae_map[map_slices] = mae.detach().cpu().numpy().astype(np.float32, copy=False)
            mean_residual_map[map_slices] = mean_resid.detach().cpu().numpy().astype(np.float32, copy=False)
            total_counts_map[map_slices] = total_counts.detach().cpu().numpy().astype(np.float32, copy=False)
            sse_over_counts2_map[map_slices] = sse_over_counts2.detach().cpu().numpy().astype(np.float32, copy=False)
            rmse_frac_map[map_slices] = rmse_frac.detach().cpu().numpy().astype(np.float32, copy=False)
            l1_frac_map[map_slices] = l1_frac.detach().cpu().numpy().astype(np.float32, copy=False)
            bias_frac_map[map_slices] = bias_frac.detach().cpu().numpy().astype(np.float32, copy=False)

            del tensors, obs_t, energy, ev, spec_fit, bkg, total, resid, sse, mae, mean_resid, total_counts, total_counts_safe, total_counts_safe_sq, sse_over_counts2, rmse_frac, l1_frac, bias_frac
            if device.type == "cuda":
                torch.cuda.empty_cache()

    np.savez_compressed(
        output_dir / "diagnostics_metrics.npz",
        sse_map=sse_map,
        mae_map=mae_map,
        mean_residual_map=mean_residual_map,
        total_counts_map=total_counts_map,
        sse_over_counts2_map=sse_over_counts2_map,
        rmse_frac_map=rmse_frac_map,
        l1_frac_map=l1_frac_map,
        bias_frac_map=bias_frac_map,
        sse_delta_maps=sse_delta_maps,
        sse_delta_over_counts2_maps=sse_delta_over_counts2_maps,
        amp_names=np.asarray(amp_names, dtype=object),
        amp_display_names=np.asarray(amp_display_names, dtype=object),
        energy_range=np.asarray(energy_range, dtype=np.int32),
    )

    plot_amp_map(
        sse_map,
        name="fit_metric_sse",
        title="Per-pixel Fit Metric (SSE)",
        filename=str(output_dir / "fit_metric_sse_map.png"),
        cmap="magma",
        percentile=(1, 99),
        save=True,
        show=False,
    )
    plot_amp_map(
        mae_map,
        name="residual_mae",
        title="Per-pixel Residual Magnitude (MAE)",
        filename=str(output_dir / "residual_mae_map.png"),
        cmap="inferno",
        percentile=(1, 99),
        save=True,
        show=False,
    )
    plot_amp_map(
        mean_residual_map,
        name="residual_mean",
        title="Per-pixel Signed Residual Mean",
        filename=str(output_dir / "residual_mean_map.png"),
        cmap="RdBu_r",
        percentile=(1, 99),
        save=True,
        show=False,
    )
    plot_amp_map(
        sse_over_counts2_map,
        name="fit_metric_sse_over_counts2",
        title="Per-pixel Fit Metric (SSE / Total Counts^2)",
        filename=str(output_dir / "fit_metric_sse_over_counts2_map.png"),
        cmap="magma",
        percentile=(1, 99),
        save=True,
        show=False,
    )
    plot_amp_map(
        rmse_frac_map,
        name="residual_rmse_over_counts",
        title="Residual RMSE / Total Counts",
        filename=str(output_dir / "residual_rmse_over_counts_map.png"),
        cmap="magma",
        percentile=(1, 99),
        save=True,
        show=False,
    )
    plot_amp_map(
        l1_frac_map,
        name="residual_l1_over_counts",
        title="Residual L1 / Total Counts",
        filename=str(output_dir / "residual_l1_over_counts_map.png"),
        cmap="magma",
        percentile=(1, 99),
        save=True,
        show=False,
    )
    plot_amp_map(
        bias_frac_map,
        name="residual_bias_over_counts",
        title="Residual Bias / Total Counts",
        filename=str(output_dir / "residual_bias_over_counts_map.png"),
        cmap="RdBu_r",
        percentile=(1, 99),
        save=True,
        show=False,
    )

    sse_delta_dir = output_dir / "sse_delta"
    sse_delta_dir.mkdir(parents=True, exist_ok=True)
    sse_delta_dict = {name: sse_delta_maps[..., idx] for idx, name in enumerate(amp_names)}
    _plot_named_grid(
        sse_delta_dict,
        title="Leave-One-Component-Out ΔSSE Maps",
        filename=output_dir / "sse_component_delta_grid.png",
        percentile=(1.0, 99.0),
        n_cols=5,
        display_names=amp_display_map,
    )
    for idx, name in enumerate(amp_names):
        display_name = amp_display_map[name]
        plot_amp_map(
            sse_delta_maps[..., idx],
            name=display_name,
            title=f"ΔSSE if {display_name} Removed",
            filename=str(
                sse_delta_dir / f"{idx:02d}_{sanitize_component_name(name)}.png"
            ),
            cmap=_map_cmap(sse_delta_maps[..., idx]),
            percentile=(1, 99),
            save=True,
            show=False,
        )

    sse_delta_over_counts2_dir = output_dir / "sse_delta_over_counts2"
    sse_delta_over_counts2_dir.mkdir(parents=True, exist_ok=True)
    sse_delta_over_counts2_dict = {
        name: sse_delta_over_counts2_maps[..., idx] for idx, name in enumerate(amp_names)
    }
    _plot_named_grid(
        sse_delta_over_counts2_dict,
        title="Leave-One-Component-Out ΔSSE / Total Counts^2 Maps",
        filename=output_dir / "sse_component_delta_over_counts2_grid.png",
        percentile=(1.0, 99.0),
        n_cols=5,
        display_names=amp_display_map,
    )
    for idx, name in enumerate(amp_names):
        display_name = amp_display_map[name]
        plot_amp_map(
            sse_delta_over_counts2_maps[..., idx],
            name=display_name,
            title=f"ΔSSE / Total Counts^2 if {display_name} Removed",
            filename=str(
                sse_delta_over_counts2_dir
                / f"{idx:02d}_{sanitize_component_name(name)}.png"
            ),
            cmap=_map_cmap(sse_delta_over_counts2_maps[..., idx]),
            percentile=(1, 99),
            save=True,
            show=False,
        )

    elemental_candidates = [
        name
        for name in amp_names
        if name not in ("COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE") and "_" not in name
    ]
    dominant_elem = None
    if elemental_candidates:
        means = {}
        for name in elemental_candidates:
            idx = amp_names.index(name)
            means[name] = float(np.nanmean(amp_maps[..., idx]))
        dominant_elem = max(means.items(), key=lambda kv: kv[1])[0]
    if dominant_elem is None:
        dominant_elem = amp_names[0]
    dominant_idx = amp_names.index(dominant_elem)

    pixel_records: list[dict] = []
    seen: set[tuple[int, int]] = set()
    min_sep = float(args.min_pixel_separation)
    _unique_append(pixel_records, seen, "center", n_rows // 2, n_cols // 2, min_separation=min_sep)
    max_counts_y, max_counts_x = np.unravel_index(np.argmax(total_counts_map), total_counts_map.shape)
    _unique_append(pixel_records, seen, "max_counts", max_counts_y, max_counts_x, min_separation=min_sep)
    max_elem_y, max_elem_x = np.unravel_index(np.argmax(amp_maps[..., dominant_idx]), map_shape)
    _unique_append(pixel_records, seen, f"max_{dominant_elem}", max_elem_y, max_elem_x, min_separation=min_sep)
    worst_y, worst_x = np.unravel_index(np.argmax(sse_map), sse_map.shape)
    _unique_append(pixel_records, seen, "worst_fit", worst_y, worst_x, min_separation=min_sep)

    total_thresh = np.percentile(total_counts_map, 50.0)
    valid = total_counts_map >= total_thresh
    masked = np.where(valid, sse_map, np.inf)
    best_y, best_x = np.unravel_index(np.argmin(masked), masked.shape)
    _unique_append(pixel_records, seen, "best_fit_highcount", best_y, best_x, min_separation=min_sep)

    target_n = int(args.n_pixel_spectra)
    if len(pixel_records) < target_n:
        ranked_sse = np.argsort(sse_map.reshape(-1))[::-1]
        _fill_unique_from_ranked(
            pixel_records,
            seen,
            ranked_sse,
            map_shape,
            prefix="high_sse",
            limit=target_n,
            min_separation=min_sep,
        )
    if len(pixel_records) < target_n:
        ranked_counts = np.argsort(total_counts_map.reshape(-1))[::-1]
        _fill_unique_from_ranked(
            pixel_records,
            seen,
            ranked_counts,
            map_shape,
            prefix="high_counts",
            limit=target_n,
            min_separation=min_sep,
        )
    if len(pixel_records) < target_n:
        ranked_best = np.argsort(np.where(valid, sse_map, np.inf).reshape(-1))
        _fill_unique_from_ranked(
            pixel_records,
            seen,
            ranked_best,
            map_shape,
            prefix="low_sse",
            limit=target_n,
            min_separation=min_sep,
        )

    with h5py.File(dataset_path, "r") as h5:
        dset = h5[spec_vol_key]
        for rec in pixel_records[: int(args.n_pixel_spectra)]:
            y, x = rec["y"], rec["x"]
            slicer = [slice(None)] * dset.ndim
            slicer[energy_dim_resolved] = slice(energy_range[0], energy_range[1] + 1)
            for md, idx in zip(map_dims, (y, x)):
                slicer[md] = int(idx)
            obs = dset[tuple(slicer)].astype(np.float32, copy=False)
            obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
            tensors = _pixel_tensors(base_params, param_maps, amp_maps, amp_names, y, x, device)

            with torch.no_grad():
                spec_fit = model_spec(
                    tensors,
                    energy_range,
                    elements_to_fit=amp_names,
                    device=device,
                    e_consts=e_consts,
                    elem_info=default_elem_info,
                    detector_type=detector_material,
                    escape_factor=escape_factor,
                )
                bkg = _compute_background(
                    obs_t,
                    energy_range,
                    tensors,
                    use_snip=use_snip,
                    device=device,
                )
                total = spec_fit + bkg

            rec["obs"] = obs_t.detach().cpu().numpy().astype(np.float32, copy=False)
            rec["bkg"] = bkg.detach().cpu().numpy().astype(np.float32, copy=False)
            rec["total"] = total.detach().cpu().numpy().astype(np.float32, copy=False)
            rec["sse"] = float(sse_map[y, x])
            rec["mae"] = float(mae_map[y, x])
            rec["nrmse_frac"] = float(rmse_frac_map[y, x])

            row = rec["obs"].reshape(-1)
            bkg_row = rec["bkg"].reshape(-1)
            total_row = rec["total"].reshape(-1)
            rec["obs"] = row
            rec["bkg"] = bkg_row
            rec["total"] = total_row
            denom = max(float(np.sum(row)), 1.0)
            rec["norm_resid"] = (row - total_row) / denom

    _plot_pixel_locations(sse_map, pixel_records[: int(args.n_pixel_spectra)], output_dir / "selected_pixels_on_fit_metric.png")
    _plot_spectra_grid(pixel_records[: int(args.n_pixel_spectra)], output_dir / "selected_pixel_spectra.png")

    for rec in pixel_records[: int(args.n_pixel_spectra)]:
        single_path = spectra_dir / f"{sanitize_component_name(rec['label'])}_y{rec['y']}_x{rec['x']}.png"
        _plot_spectra_grid([rec], single_path)

    print(f"Saved diagnostics to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot residual/fit-metric maps and selected pixel spectra for a batch-fit output."
    )
    parser.add_argument("input_path", type=str, help="Path to batch-fit output HDF5")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write diagnostics outputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Compute device for reconstruction: "cpu" or "cuda"',
    )
    parser.add_argument(
        "--n_pixel_spectra",
        type=int,
        default=12,
        help="How many representative pixels to plot",
    )
    parser.add_argument(
        "--min_pixel_separation",
        type=float,
        default=12.0,
        help="Minimum pixel separation between selected example spectra",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Show tile-level progress while computing metrics",
    )
    main(parser.parse_args())
