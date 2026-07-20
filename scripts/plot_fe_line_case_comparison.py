import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch

from mapstorch.batchfit import coerce_energy_dim, decode_names, decode_text, resolve_h5_key
from mapstorch.bkg import snip_bkg
from mapstorch.constant import read_constants
from mapstorch.default import default_elem_info, default_param_vals
from mapstorch.io import parse_override_params_file
from mapstorch.map import model_spec


CASE_LABELS = {
    "amponly": "Amplitude only",
    "fwhmoffsetonly": "FWHM offset only",
    "fwhmfanoprimeonly": "FWHM fano only",
    "fwhmoffset_fwhmfanoprime": "FWHM offset + fano",
}

CASE_COLORS = {
    "amponly": "#4c78a8",
    "fwhmoffsetonly": "#f58518",
    "fwhmfanoprimeonly": "#54a24b",
    "fwhmoffset_fwhmfanoprime": "#e45756",
}


def _log1p10(values: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(values, dtype=np.float32) + 1.0)


def _line_colors(n: int) -> np.ndarray:
    return plt.cm.viridis(np.linspace(0.12, 0.92, int(n)))


def _energy_suffix(energy_min: float, energy_max: float) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")

    return f"{fmt(float(energy_min))}_{fmt(float(energy_max))}kev"


def _legend_columns(n_items: int) -> int:
    if n_items <= 8:
        return 1
    if n_items <= 16:
        return 2
    return 3


@dataclass
class CaseData:
    key: str
    label: str
    color: str
    h5_path: Path
    dataset_path: Path
    override_path: Path
    spec_vol_key: str | None
    energy_dim: int
    fit_energy_range: tuple[int, int]
    use_snip: bool
    detector_material: str
    escape_factor: float
    amp_maps: np.ndarray
    amp_names: list[str]
    param_maps: dict[str, np.ndarray]


def _base_param_dict(override: dict) -> dict[str, float]:
    params = dict(default_param_vals)
    for key in default_param_vals:
        if key in override:
            params[key] = float(override[key])
    return params


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


def load_case(case_key: str, h5_path: Path) -> CaseData:
    with h5py.File(h5_path, "r") as h5:
        dataset_path = Path(decode_text(h5.attrs["dataset"]))
        override_path = Path(decode_text(h5.attrs["override_params_file"]))
        spec_vol_key_attr = decode_text(h5.attrs["spec_vol_key"])
        energy_dim = int(h5.attrs["energy_dim"])
        fit_energy_range = (
            int(h5.attrs["energy_range_start"]),
            int(h5.attrs["energy_range_stop"]),
        )
        use_snip = bool(h5.attrs.get("use_snip", True))
        detector_material = decode_text(h5.attrs.get("detector_material", "Si"))
        escape_factor = float(h5.attrs.get("escape_factor", 0.0))
        amp_maps = h5["amp_maps_log10"][:]
        amp_names = decode_names(h5["amp_names"][:])
        param_names = sorted(h5["param_maps"].keys())
        param_maps = {name: h5["param_maps"][name][:] for name in param_names}

    return CaseData(
        key=case_key,
        label=CASE_LABELS.get(case_key, case_key),
        color=CASE_COLORS.get(case_key, "#444444"),
        h5_path=h5_path,
        dataset_path=dataset_path,
        override_path=override_path,
        spec_vol_key=None if spec_vol_key_attr == "None" else spec_vol_key_attr,
        energy_dim=energy_dim,
        fit_energy_range=fit_energy_range,
        use_snip=use_snip,
        detector_material=detector_material,
        escape_factor=escape_factor,
        amp_maps=amp_maps,
        amp_names=amp_names,
        param_maps=param_maps,
    )


def validate_cases(cases: list[CaseData]):
    first = cases[0]
    for case in cases[1:]:
        if case.dataset_path != first.dataset_path:
            raise ValueError("Dataset mismatch across cases")
        if case.override_path != first.override_path:
            raise ValueError("Override mismatch across cases")
        if case.fit_energy_range != first.fit_energy_range:
            raise ValueError("Fit energy range mismatch across cases")


def load_energy_axis(dataset_path: Path) -> np.ndarray:
    with h5py.File(dataset_path, "r") as h5:
        for key in ("MAPS/energy", "MAPS/Spectra/Energy"):
            if key in h5:
                return h5[key][:].astype(np.float32, copy=False)
    raise KeyError(f"Energy axis not found in {dataset_path}")


def load_observed_window(
    case: CaseData,
    y: int,
    xs: list[int],
    channel_lo: int,
    channel_hi: int,
) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with h5py.File(case.dataset_path, "r") as h5:
        spec_vol_key = resolve_h5_key(
            h5,
            case.spec_vol_key,
            ["MAPS/Spectra/mca_arr", "MAPS/mca_arr"],
            required=True,
            label="spectra volume",
        )
        dset = h5[spec_vol_key]
        energy_dim = coerce_energy_dim(case.energy_dim, dset.ndim)
        map_dims = [i for i in range(dset.ndim) if i != energy_dim]
        for x in xs:
            slicer = [slice(None)] * dset.ndim
            slicer[energy_dim] = slice(channel_lo, channel_hi + 1)
            for md, idx in zip(map_dims, (y, x)):
                slicer[md] = int(idx)
            out[int(x)] = dset[tuple(slicer)].astype(np.float32, copy=False).reshape(-1)
    return out


def reconstruct_case_window(
    case: CaseData,
    y: int,
    xs: list[int],
    observed_full_roi: dict[int, np.ndarray],
    channel_lo: int,
    channel_hi: int,
    device: torch.device,
) -> dict[int, np.ndarray]:
    override = parse_override_params_file(str(case.override_path))
    e_consts = read_constants(str(case.override_path), default_elem_info)
    base_params = _base_param_dict(override)
    out: dict[int, np.ndarray] = {}

    fit_lo, fit_hi = case.fit_energy_range
    rel_lo = int(channel_lo - fit_lo)
    rel_hi = int(channel_hi - fit_lo)
    if rel_lo < 0 or rel_hi >= (fit_hi - fit_lo + 1):
        raise ValueError("Requested plotting window is outside the fit ROI")

    for x in xs:
        obs_roi = observed_full_roi[int(x)]
        obs_t = torch.as_tensor(obs_roi, device=device, dtype=torch.float32)
        tensors = _pixel_tensors(base_params, case.param_maps, case.amp_maps, case.amp_names, y, int(x), device)
        with torch.no_grad():
            spec_fit = model_spec(
                tensors,
                case.fit_energy_range,
                elements_to_fit=case.amp_names,
                device=device,
                e_consts=e_consts,
                elem_info=default_elem_info,
                detector_type=case.detector_material,
                escape_factor=case.escape_factor,
            )
            bkg = _compute_background(
                obs_t,
                case.fit_energy_range,
                tensors,
                use_snip=case.use_snip,
                device=device,
            )
            total = spec_fit + bkg
        total_np = total.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        out[int(x)] = total_np[rel_lo : rel_hi + 1]
    return out


def plot_element_map_with_line(
    element_map: np.ndarray,
    element_name: str,
    y: int,
    xs: list[int],
    output_path: Path,
):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 5.9),
        gridspec_kw={"width_ratios": [1.0, 1.08, 0.045]},
    )
    ax_full, ax_zoom, cax = axes
    finite = element_map[np.isfinite(element_map)]
    vmin, vmax = np.percentile(finite, [1, 99]).astype(float)
    im = ax_full.imshow(element_map, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    x_pad = max(4, int(np.ceil(len(xs) * 0.4)))
    y_pad = 7
    x0 = min(xs) - x_pad
    x1 = max(xs) + x_pad
    y0 = y - y_pad
    y1 = y + y_pad
    colors = _line_colors(len(xs))

    ax_full.plot(xs, [y] * len(xs), color="cyan", lw=1.2, alpha=0.8)
    for idx, (x, color) in enumerate(zip(xs, colors), start=1):
        ax_full.scatter(x, y, s=26, facecolors="none", edgecolors=color, linewidths=1.1)
    ax_full.add_patch(
        Rectangle(
            (x0 - 0.5, y0 - 0.5),
            (x1 - x0) + 1.0,
            (y1 - y0) + 1.0,
            fill=False,
            edgecolor="cyan",
            linewidth=2.0,
        )
    )
    ax_full.set_title(f"{element_name} map: full field with focus box")
    ax_full.set_xticks([])
    ax_full.set_yticks([])

    ax_zoom.imshow(element_map, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    ax_zoom.plot(xs, [y] * len(xs), color="cyan", lw=1.5, alpha=0.85)
    x_pad = max(4, int(np.ceil(len(xs) * 0.55)))
    y_pad = 8
    for idx, (x, color) in enumerate(zip(xs, colors), start=1):
        ax_zoom.scatter(x, y, s=44, facecolors="none", edgecolors=color, linewidths=1.5)
        dx = (-12 if idx % 2 else 12)
        dy_levels = [18, 10, -10, -18]
        dy = dy_levels[(idx - 1) % len(dy_levels)]
        ax_zoom.annotate(
            f"L{idx:02d}",
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            color="white",
            fontsize=8,
            ha="center",
            va="center",
            weight="bold",
            bbox={"facecolor": "black", "alpha": 0.42, "pad": 1.3, "edgecolor": "none"},
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "white", "alpha": 0.75},
        )
    ax_zoom.scatter(xs[-1], y, s=68, facecolors="none", edgecolors="lime", linewidths=2.0)
    ax_zoom.set_title(f"Zoomed {element_name} map with selected line points")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax_zoom.set_ylim(y - y_pad, y + y_pad)
    fig.colorbar(im, cax=cax, label=f"log10 {element_name} amplitude")
    fig.subplots_adjust(left=0.035, right=0.965, top=0.9, bottom=0.05, wspace=0.12)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_stacked(
    energy_window: np.ndarray,
    observed_window: dict[int, np.ndarray],
    element_name: str,
    y: int,
    xs: list[int],
    output_path: Path,
    *,
    scale_mode: str,
):
    fig, ax = plt.subplots(figsize=(11.2, 7.9))
    colors = _line_colors(len(xs))
    offset = 0.0
    transformed_traces: list[np.ndarray] = []
    max_peak = 0.0
    if scale_mode == "normalized":
        step = 1.22
    elif scale_mode == "log1p":
        step = 0.72
    else:
        step = None

    for idx, (x, color) in enumerate(zip(xs, colors), start=1):
        obs = np.asarray(observed_window[int(x)], dtype=np.float32)
        if scale_mode == "normalized":
            yvals = obs / max(float(np.max(obs)), 1.0)
        elif scale_mode == "log1p":
            yvals = _log1p10(obs)
        else:
            yvals = obs.copy()
        transformed_traces.append(yvals)
        max_peak = max(max_peak, float(np.max(yvals)))

    for idx, (x, color, yvals) in enumerate(zip(xs, colors, transformed_traces), start=1):
        if scale_mode in {"normalized", "log1p"}:
            local_offset = (len(xs) - idx) * step
            ax.plot(
                energy_window,
                yvals + local_offset,
                color=color,
                lw=1.1,
                label=f"L{idx:02d} x={x}",
            )
        else:
            ax.plot(
                energy_window,
                yvals + offset,
                color=color,
                lw=1.1,
                label=f"L{idx:02d} x={x}",
            )
            offset += max(float(np.max(yvals)) * 1.18, 36.0)
    ax.set_xlim(float(energy_window[0]), float(energy_window[-1]))
    ax.set_xlabel("Energy (keV)")
    if scale_mode == "normalized":
        ax.set_ylabel("Normalized intensity + offset")
        suffix = "normalized"
    elif scale_mode == "log1p":
        ax.set_ylabel("log10(count + 1) + offset")
        suffix = "log10(count + 1)"
        ax.set_ylim(bottom=0.0)
    else:
        ax.set_ylabel("Counts + offset")
        suffix = "raw"
    ax.set_title(
        f"Experimental {element_name} peak-window spectra along selected line ({suffix})\n"
        f"y={y}, x from {xs[0]} to {xs[-1]}"
    )
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Line points",
        fontsize=8,
        title_fontsize=9,
        ncol=_legend_columns(len(xs)),
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_experiment_overlay(
    energy_window: np.ndarray,
    observed_window: dict[int, np.ndarray],
    element_name: str,
    y: int,
    xs: list[int],
    output_path: Path,
    *,
    scale_mode: str,
):
    fig, ax = plt.subplots(figsize=(11.0, 7.2))
    colors = _line_colors(len(xs))

    ymax = 0.0
    for idx, (x, color) in enumerate(zip(xs, colors), start=1):
        obs = np.asarray(observed_window[int(x)], dtype=np.float32)
        if scale_mode == "normalized":
            yvals = obs / max(float(np.max(obs)), 1.0)
        elif scale_mode == "log1p":
            yvals = _log1p10(obs)
        else:
            yvals = obs.copy()
        ymax = max(ymax, float(np.max(yvals)))
        ax.plot(energy_window, yvals, color=color, lw=1.1, alpha=0.95, label=f"L{idx:02d} x={x}")

    ax.set_xlim(float(energy_window[0]), float(energy_window[-1]))
    ax.set_xlabel("Energy (keV)")
    if scale_mode == "normalized":
        ax.set_ylabel("Normalized counts")
        ax.set_ylim(0.0, 1.18)
        suffix = "normalized"
    elif scale_mode == "log1p":
        ax.set_ylabel("log10(count + 1)")
        ax.set_ylim(0.0, ymax * 1.06)
        suffix = "log10(count + 1)"
    else:
        ax.set_ylabel("Counts")
        ax.set_ylim(0.0, ymax * 1.06)
        suffix = "raw"
    ax.set_title(
        f"Experimental {element_name} peak-window spectra without offset ({suffix})\n"
        f"y={y}, x from {xs[0]} to {xs[-1]}"
    )
    ax.grid(alpha=0.18, linewidth=0.5)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        title="Line points",
        fontsize=8,
        title_fontsize=9,
        ncol=_legend_columns(len(xs)),
    )
    plt.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fit_grid(
    energy_window: np.ndarray,
    observed_window: dict[int, np.ndarray],
    fitted_window: dict[str, dict[int, np.ndarray]],
    element_name: str,
    y: int,
    xs: list[int],
    output_path: Path,
    *,
    scale_mode: str,
):
    n = len(xs)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.6 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, x in enumerate(xs, start=1):
        ax = axes[(idx - 1) // n_cols][(idx - 1) % n_cols]
        ax.axis("on")
        obs = np.asarray(observed_window[int(x)], dtype=np.float32)
        if scale_mode == "normalized":
            scale = max(float(np.max(obs)), 1.0)
            obs_plot = obs / scale
            ax.plot(energy_window, obs_plot, color="0.15", lw=1.1, marker="o", ms=2.0, label="Experiment")
            for case_key, curves in fitted_window.items():
                ax.plot(
                    energy_window,
                    np.asarray(curves[int(x)], dtype=np.float32) / scale,
                    color=CASE_COLORS.get(case_key, "#444444"),
                    lw=1.0,
                    label=CASE_LABELS.get(case_key, case_key),
                )
            ax.set_ylim(0.0, 1.18)
            ax.set_ylabel("Normalized counts")
        elif scale_mode == "log1p":
            obs_plot = _log1p10(obs)
            ax.plot(energy_window, obs_plot, color="0.15", lw=1.1, marker="o", ms=2.0, label="Experiment")
            ymax = float(np.max(obs_plot))
            for case_key, curves in fitted_window.items():
                fit = _log1p10(np.asarray(curves[int(x)], dtype=np.float32))
                ymax = max(ymax, float(np.max(fit)))
                ax.plot(
                    energy_window,
                    fit,
                    color=CASE_COLORS.get(case_key, "#444444"),
                    lw=1.0,
                    label=CASE_LABELS.get(case_key, case_key),
                )
            ax.set_ylim(0.0, ymax * 1.08)
            ax.set_ylabel("log10(count + 1)")
        else:
            ax.plot(energy_window, obs, color="0.15", lw=1.1, marker="o", ms=2.0, label="Experiment")
            ymax = float(np.max(obs))
            for case_key, curves in fitted_window.items():
                fit = np.asarray(curves[int(x)], dtype=np.float32)
                ymax = max(ymax, float(np.max(fit)))
                ax.plot(
                    energy_window,
                    fit,
                    color=CASE_COLORS.get(case_key, "#444444"),
                    lw=1.0,
                    label=CASE_LABELS.get(case_key, case_key),
                )
            ax.set_ylim(0.0, ymax * 1.1)
            ax.set_ylabel("Counts")
        ax.set_title(f"L{idx:02d}: y={y}, x={x}")
        ax.set_xlim(float(energy_window[0]), float(energy_window[-1]))
        ax.grid(alpha=0.18, linewidth=0.5)
        if (idx - 1) // n_cols == n_rows - 1:
            ax.set_xlabel("Energy (keV)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    legend_ncol = 3 if len(labels) <= 5 else 4
    fig.legend(handles, labels, loc="lower center", ncol=legend_ncol, bbox_to_anchor=(0.5, 0.01), frameon=False)
    if scale_mode == "normalized":
        suffix = "normalized"
    elif scale_mode == "log1p":
        suffix = "log10(count + 1)"
    else:
        suffix = "raw"
    fig.suptitle(
        f"{element_name} peak-window experiment vs fitted spectrum (+bkg) ({suffix})\n"
        f"y={y}, selected line toward G01",
        y=0.985,
        fontsize=14,
    )
    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.92))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_grid(
    energy_window: np.ndarray,
    observed_window: dict[int, np.ndarray],
    fitted_window: dict[str, dict[int, np.ndarray]],
    element_name: str,
    y: int,
    xs: list[int],
    output_path: Path,
    *,
    scale_mode: str,
):
    n = len(xs)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.1 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, x in enumerate(xs, start=1):
        ax = axes[(idx - 1) // n_cols][(idx - 1) % n_cols]
        ax.axis("on")
        obs = np.asarray(observed_window[int(x)], dtype=np.float32)
        ax.axhline(0.0, color="0.6", lw=0.8)
        lim = 0.0
        for case_key, curves in fitted_window.items():
            resid = obs - np.asarray(curves[int(x)], dtype=np.float32)
            finite = resid[np.isfinite(resid)]
            if finite.size:
                lim = max(lim, float(np.percentile(np.abs(finite), 99)))
            ax.plot(
                energy_window,
                resid,
                color=CASE_COLORS.get(case_key, "#444444"),
                lw=1.0,
                label=CASE_LABELS.get(case_key, case_key),
            )
        lim = max(lim, 1.0)
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
        if scale_mode == "symlog":
            ax.set_yscale("symlog", linthresh=max(1.0, 0.08 * lim), linscale=1.0)
        ax.set_title(f"L{idx:02d}: y={y}, x={x}")
        ax.set_xlim(float(energy_window[0]), float(energy_window[-1]))
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_ylabel("Obs - fit" if scale_mode == "linear" else "Obs - fit (symlog)")
        if (idx - 1) // n_cols == n_rows - 1:
            ax.set_xlabel("Energy (keV)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False)
    mode_label = "linear" if scale_mode == "linear" else "symlog"
    fig.suptitle(
        f"{element_name} peak-window residuals by case ({mode_label})\n"
        f"y={y}, selected line toward G01",
        y=0.985,
        fontsize=14,
    )
    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.92))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(
    output_path: Path,
    xs: list[int],
    y: int,
    element_map: np.ndarray,
    element_name: str,
    observed_window: dict[int, np.ndarray],
):
    amp_col = f"{element_name.lower()}_log10_amp"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "label",
                "y",
                "x",
                amp_col,
                "window_total_counts",
                "window_peak_counts",
            ],
        )
        writer.writeheader()
        for idx, x in enumerate(xs, start=1):
            obs = np.asarray(observed_window[int(x)], dtype=np.float32)
            writer.writerow(
                {
                    "label": f"L{idx:02d}",
                    "y": int(y),
                    "x": int(x),
                    amp_col: float(element_map[int(y), int(x)]),
                    "window_total_counts": float(np.sum(obs)),
                    "window_peak_counts": float(np.max(obs)),
                }
            )


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [load_case(case_key, Path(h5_path)) for case_key, h5_path in args.case]
    validate_cases(cases)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    reference_case = next((c for c in cases if c.key == args.reference_case), cases[0])
    element_name = str(args.map_element)
    if element_name not in reference_case.amp_names:
        raise KeyError(f"{element_name} not present in reference case amp_names: {reference_case.h5_path}")
    elem_idx = reference_case.amp_names.index(element_name)
    element_map = np.asarray(reference_case.amp_maps[..., elem_idx], dtype=np.float32)

    energy_axis = load_energy_axis(reference_case.dataset_path)
    channel_lo = int(np.argmin(np.abs(energy_axis - float(args.energy_min))))
    channel_hi = int(np.argmin(np.abs(energy_axis - float(args.energy_max))))
    if channel_hi < channel_lo:
        channel_lo, channel_hi = channel_hi, channel_lo
    energy_window = energy_axis[channel_lo : channel_hi + 1]
    suffix = _energy_suffix(float(args.energy_min), float(args.energy_max))

    y = int(args.y)
    xs = [int(v) for v in args.xs]

    observed_window = load_observed_window(reference_case, y, xs, channel_lo, channel_hi)
    fit_lo, fit_hi = reference_case.fit_energy_range
    observed_full_roi = load_observed_window(reference_case, y, xs, fit_lo, fit_hi)

    fitted_window: dict[str, dict[int, np.ndarray]] = {}
    for case in cases:
        fitted_window[case.key] = reconstruct_case_window(
            case,
            y,
            xs,
            observed_full_roi,
            channel_lo,
            channel_hi,
            device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "selected_line_points.csv", xs, y, element_map, element_name, observed_window)
    plot_element_map_with_line(element_map, element_name, y, xs, output_dir / f"{element_name.lower()}_map_selected_line.png")
    plot_experiment_stacked(
        energy_window,
        observed_window,
        element_name,
        y,
        xs,
        output_dir / f"experiment_stacked_raw_{suffix}.png",
        scale_mode="raw",
    )
    plot_experiment_stacked(
        energy_window,
        observed_window,
        element_name,
        y,
        xs,
        output_dir / f"experiment_stacked_normalized_{suffix}.png",
        scale_mode="normalized",
    )
    plot_experiment_stacked(
        energy_window,
        observed_window,
        element_name,
        y,
        xs,
        output_dir / f"experiment_stacked_logcountp1_{suffix}.png",
        scale_mode="log1p",
    )
    plot_experiment_overlay(
        energy_window,
        observed_window,
        element_name,
        y,
        xs,
        output_dir / f"experiment_overlay_raw_{suffix}.png",
        scale_mode="raw",
    )
    plot_experiment_overlay(
        energy_window,
        observed_window,
        element_name,
        y,
        xs,
        output_dir / f"experiment_overlay_normalized_{suffix}.png",
        scale_mode="normalized",
    )
    plot_experiment_overlay(
        energy_window,
        observed_window,
        element_name,
        y,
        xs,
        output_dir / f"experiment_overlay_logcountp1_{suffix}.png",
        scale_mode="log1p",
    )
    plot_fit_grid(
        energy_window,
        observed_window,
        fitted_window,
        element_name,
        y,
        xs,
        output_dir / f"fit_grid_raw_{suffix}.png",
        scale_mode="raw",
    )
    plot_fit_grid(
        energy_window,
        observed_window,
        fitted_window,
        element_name,
        y,
        xs,
        output_dir / f"fit_grid_normalized_{suffix}.png",
        scale_mode="normalized",
    )
    plot_fit_grid(
        energy_window,
        observed_window,
        fitted_window,
        element_name,
        y,
        xs,
        output_dir / f"fit_grid_logcountp1_{suffix}.png",
        scale_mode="log1p",
    )
    plot_residual_grid(
        energy_window,
        observed_window,
        fitted_window,
        element_name,
        y,
        xs,
        output_dir / f"fit_grid_residuals_{suffix}.png",
        scale_mode="linear",
    )
    plot_residual_grid(
        energy_window,
        observed_window,
        fitted_window,
        element_name,
        y,
        xs,
        output_dir / f"fit_grid_residuals_symlog_{suffix}.png",
        scale_mode="symlog",
    )

    print(
        f"Saved {element_name}-line comparison plots to {output_dir} "
        f"(channels {channel_lo}-{channel_hi}, energy {float(energy_window[0]):.3f}-{float(energy_window[-1]):.3f} keV)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot a line of Fe-window spectra around a hotspot and compare fitted spectra (+bkg) across cases."
    )
    parser.add_argument(
        "--case",
        nargs=2,
        metavar=("CASE_KEY", "H5_PATH"),
        action="append",
        required=True,
        help="Case key and HDF5 path. Use the same keys as prior comparison runs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write plots.",
    )
    parser.add_argument(
        "--reference_case",
        type=str,
        default="fwhmoffset_fwhmfanoprime",
        help="Case key used for the Fe map and raw spectra lookup.",
    )
    parser.add_argument(
        "--map_element",
        type=str,
        default="Fe",
        help="Element amplitude map to show in the two-panel map figure and to use in titles.",
    )
    parser.add_argument("--y", type=int, required=True, help="Fixed row for the selected line.")
    parser.add_argument("--xs", type=int, nargs="+", required=True, help="Ordered x coordinates along the selected line.")
    parser.add_argument("--energy_min", type=float, default=6.0, help="Lower plot energy bound in keV.")
    parser.add_argument("--energy_max", type=float, default=7.2, help="Upper plot energy bound in keV.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device used for reconstruction.",
    )
    main(parser.parse_args())
