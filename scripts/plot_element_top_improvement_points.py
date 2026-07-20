import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
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

DEFAULT_REFIT_CASE = "fwhmoffset_fwhmfanoprime"


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
    metrics: dict[str, np.ndarray]


def _log1p10(values: np.ndarray) -> np.ndarray:
    return np.log10(np.asarray(values, dtype=np.float32) + 1.0)


def _energy_suffix(energy_min: float, energy_max: float) -> str:
    def fmt(value: float) -> str:
        text = f"{value:.2f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")

    return f"{fmt(float(energy_min))}_{fmt(float(energy_max))}kev"


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


def _finite_percentile_limits(values: list[np.ndarray], lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    finite_parts = []
    for arr in values:
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_parts.append(finite.reshape(-1))
    if not finite_parts:
        return 0.0, 1.0
    merged = np.concatenate(finite_parts, axis=0)
    vmin, vmax = np.percentile(merged, [lo, hi]).astype(float)
    if vmax <= vmin:
        pad = max(1e-12, 0.05 * max(abs(vmin), 1.0))
        vmin -= pad
        vmax += pad
    return vmin, vmax


def _element_info(name: str):
    target = name.strip()
    for info in default_elem_info:
        if info.name.strip() == target:
            return info
    raise KeyError(f"Element info not found for {name}")


def load_case(case_key: str, h5_path: Path) -> CaseData:
    metrics_path = h5_path.parent / "plots" / case_key / "diagnostics" / "diagnostics_metrics.npz"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics metrics for {case_key}: {metrics_path}")

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

    metrics_npz = np.load(metrics_path, allow_pickle=True)
    metrics = {name: metrics_npz[name] for name in metrics_npz.files}

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
        metrics=metrics,
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


def load_observed_range(case: CaseData, coords: list[tuple[int, int]], channel_lo: int, channel_hi: int) -> dict[tuple[int, int], np.ndarray]:
    out: dict[tuple[int, int], np.ndarray] = {}
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
        for y, x in coords:
            slicer = [slice(None)] * dset.ndim
            slicer[energy_dim] = slice(channel_lo, channel_hi + 1)
            for md, idx in zip(map_dims, (y, x)):
                slicer[md] = int(idx)
            out[(int(y), int(x))] = dset[tuple(slicer)].astype(np.float32, copy=False).reshape(-1)
    return out


def reconstruct_case_range(
    case: CaseData,
    coords: list[tuple[int, int]],
    observed_full_roi: dict[tuple[int, int], np.ndarray],
    channel_lo: int,
    channel_hi: int,
    device: torch.device,
) -> dict[tuple[int, int], np.ndarray]:
    override = parse_override_params_file(str(case.override_path))
    e_consts = read_constants(str(case.override_path), default_elem_info)
    base_params = _base_param_dict(override)
    out: dict[tuple[int, int], np.ndarray] = {}

    fit_lo, fit_hi = case.fit_energy_range
    rel_lo = int(channel_lo - fit_lo)
    rel_hi = int(channel_hi - fit_lo)
    if rel_lo < 0 or rel_hi >= (fit_hi - fit_lo + 1):
        raise ValueError("Requested plotting window is outside the fit ROI")

    for y, x in coords:
        obs_roi = observed_full_roi[(y, x)]
        obs_t = torch.as_tensor(obs_roi, device=device, dtype=torch.float32)
        tensors = _pixel_tensors(base_params, case.param_maps, case.amp_maps, case.amp_names, y, x, device)
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
        out[(y, x)] = total_np[rel_lo : rel_hi + 1]
    return out


def select_points(
    element_map: np.ndarray,
    score_map: np.ndarray,
    total_counts_map: np.ndarray,
    *,
    n_points: int,
    amp_percentile: float,
    min_pixel_separation: float,
    label_prefix: str,
    extra_fields: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    valid = np.isfinite(element_map) & np.isfinite(score_map) & np.isfinite(total_counts_map) & (score_map > 0.0)
    if not np.any(valid):
        raise ValueError("No valid pixels with positive improvement")

    amp_values = element_map[valid]
    flat_score = score_map.reshape(-1)
    flat_amp = element_map.reshape(-1)
    flat_counts = total_counts_map.reshape(-1)
    ranked = np.lexsort((-flat_counts, -flat_amp, -flat_score))

    thresholds = []
    for q in [amp_percentile, 90.0, 85.0, 80.0, 75.0, 70.0, 60.0, 50.0]:
        if q not in thresholds:
            thresholds.append(float(q))

    records: list[dict] = []
    seen: set[tuple[int, int]] = set()
    current_sep = float(min_pixel_separation)
    while len(records) < n_points and current_sep >= 0.0:
        records.clear()
        seen.clear()
        counter = 1
        for q in thresholds + [-np.inf]:
            if len(records) >= n_points:
                break
            thr = -np.inf if q == -np.inf else float(np.nanpercentile(amp_values, q))
            for flat_idx in ranked:
                if len(records) >= n_points:
                    break
                y, x = np.unravel_index(int(flat_idx), element_map.shape)
                if not valid[y, x]:
                    continue
                if float(element_map[y, x]) < thr:
                    continue
                coord = (int(y), int(x))
                if coord in seen:
                    continue
                if not _far_enough(seen, int(y), int(x), min_separation=current_sep):
                    continue
                seen.add(coord)
                rec = {
                    "label": f"{label_prefix}{counter:02d}",
                    "y": int(y),
                    "x": int(x),
                    "element_log10_amp": float(element_map[y, x]),
                    "selection_score": float(score_map[y, x]),
                    "total_counts": float(total_counts_map[y, x]),
                }
                if extra_fields:
                    for key, arr in extra_fields.items():
                        rec[key] = float(arr[y, x])
                records.append(rec)
                counter += 1
        current_sep = current_sep - 2.0

    if len(records) < n_points:
        raise RuntimeError(f"Only selected {len(records)} points, expected {n_points}")

    return records[:n_points]


def plot_selected_maps(
    element_map: np.ndarray,
    score_map: np.ndarray,
    records: list[dict],
    *,
    element_name: str,
    score_title: str,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.3))
    map_specs = [
        (axes[0], element_map, "magma", f"{element_name} map"),
        (axes[1], score_map, "magma", score_title),
    ]
    for ax, img, cmap, title in map_specs:
        finite = img[np.isfinite(img)]
        vmin, vmax = np.percentile(finite, [1, 99]).astype(float)
        if vmax <= vmin:
            vmax = vmin + 1.0
        im = ax.imshow(img, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        for rec in records:
            ax.scatter(rec["x"], rec["y"], s=44, facecolors="none", edgecolors="cyan", linewidths=1.4)
            ax.text(
                rec["x"] + 2,
                rec["y"] + 2,
                rec["label"],
                color="white",
                fontsize=8,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.2, "edgecolor": "none"},
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.84)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_shared_amp_maps(
    case_maps: list[tuple[str, np.ndarray]],
    records: list[dict],
    *,
    element_name: str,
    output_path: Path,
):
    n = len(case_maps)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 4.4), squeeze=False)
    axes = axes[0]
    vmin, vmax = _finite_percentile_limits([img for _, img in case_maps], 1.0, 99.0)
    for ax, (case_key, img) in zip(axes, case_maps):
        im = ax.imshow(img, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        for rec in records:
            ax.scatter(rec["x"], rec["y"], s=42, facecolors="none", edgecolors="cyan", linewidths=1.2)
            ax.text(
                rec["x"] + 2,
                rec["y"] + 2,
                rec["label"],
                color="white",
                fontsize=7.5,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.0, "edgecolor": "none"},
            )
        ax.set_title(CASE_LABELS.get(case_key, case_key))
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.84)
    fig.suptitle(f"{element_name}: shared-scale log10 amplitude maps", y=0.98, fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_maps(
    delta_maps: list[tuple[str, np.ndarray]],
    records: list[dict],
    *,
    element_name: str,
    output_path: Path,
):
    n = len(delta_maps)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.4), squeeze=False)
    axes = axes[0]
    _, vmax = _finite_percentile_limits([np.abs(img) for _, img in delta_maps], 1.0, 99.0)
    vmax = max(vmax, 1e-6)
    for ax, (case_key, img) in zip(axes, delta_maps):
        im = ax.imshow(img, origin="lower", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        for rec in records:
            ax.scatter(rec["x"], rec["y"], s=42, facecolors="none", edgecolors="black", linewidths=1.0)
            ax.text(
                rec["x"] + 2,
                rec["y"] + 2,
                rec["label"],
                color="white",
                fontsize=7.5,
                weight="bold",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1.0, "edgecolor": "none"},
            )
        ax.set_title(f"{CASE_LABELS.get(case_key, case_key)} - amponly")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.84)
    fig.suptitle(f"{element_name}: Δlog10 amplitude vs amponly", y=0.98, fontsize=14)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_fit_grid(
    energy_axis: np.ndarray,
    observed: dict[tuple[int, int], np.ndarray],
    fitted: dict[str, dict[tuple[int, int], np.ndarray]],
    records: list[dict],
    *,
    title: str,
    output_path: Path,
    scale_mode: str,
):
    n = len(records)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.6 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, rec in enumerate(records):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axis("on")
        coord = (rec["y"], rec["x"])
        obs = np.asarray(observed[coord], dtype=np.float32)

        if scale_mode == "log1p":
            obs_plot = _log1p10(obs)
            ymax = float(np.max(obs_plot))
            ax.plot(energy_axis, obs_plot, color="0.15", lw=1.1, marker="o", ms=1.8, label="Experiment")
            for case_key, curves in fitted.items():
                fit_plot = _log1p10(np.asarray(curves[coord], dtype=np.float32))
                ymax = max(ymax, float(np.max(fit_plot)))
                ax.plot(energy_axis, fit_plot, color=CASE_COLORS.get(case_key, "#444444"), lw=1.0, label=CASE_LABELS.get(case_key, case_key))
            ax.set_ylim(0.0, ymax * 1.08)
            ax.set_ylabel("log10(count + 1)")
        elif scale_mode == "normalized":
            scale = max(float(np.max(obs)), 1.0)
            obs_plot = obs / scale
            ax.plot(energy_axis, obs_plot, color="0.15", lw=1.1, marker="o", ms=1.8, label="Experiment")
            for case_key, curves in fitted.items():
                fit_plot = np.asarray(curves[coord], dtype=np.float32) / scale
                ax.plot(energy_axis, fit_plot, color=CASE_COLORS.get(case_key, "#444444"), lw=1.0, label=CASE_LABELS.get(case_key, case_key))
            ax.set_ylim(0.0, 1.18)
            ax.set_ylabel("Normalized counts")
        else:
            ymax = float(np.max(obs))
            ax.plot(energy_axis, obs, color="0.15", lw=1.1, marker="o", ms=1.8, label="Experiment")
            for case_key, curves in fitted.items():
                fit_plot = np.asarray(curves[coord], dtype=np.float32)
                ymax = max(ymax, float(np.max(fit_plot)))
                ax.plot(energy_axis, fit_plot, color=CASE_COLORS.get(case_key, "#444444"), lw=1.0, label=CASE_LABELS.get(case_key, case_key))
            ax.set_ylim(0.0, ymax * 1.08)
            ax.set_ylabel("Counts")

            ax.set_title(
                f"{rec['label']} y={rec['y']} x={rec['x']}\n"
            f"score={rec['selection_score']:.2e}, amp={rec['element_log10_amp']:.2f}"
        )
        ax.grid(alpha=0.18, linewidth=0.5)
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Energy (keV)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.suptitle(title, y=0.985, fontsize=14)
    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_residual_grid(
    energy_axis: np.ndarray,
    observed: dict[tuple[int, int], np.ndarray],
    fitted: dict[str, dict[tuple[int, int], np.ndarray]],
    records: list[dict],
    *,
    title: str,
    output_path: Path,
):
    n = len(records)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 3.1 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    for idx, rec in enumerate(records):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axis("on")
        coord = (rec["y"], rec["x"])
        obs = np.asarray(observed[coord], dtype=np.float32)
        ax.axhline(0.0, color="0.6", lw=0.8)
        lim = 0.0
        for case_key, curves in fitted.items():
            resid = obs - np.asarray(curves[coord], dtype=np.float32)
            finite = resid[np.isfinite(resid)]
            if finite.size:
                lim = max(lim, float(np.percentile(np.abs(finite), 99)))
            ax.plot(energy_axis, resid, color=CASE_COLORS.get(case_key, "#444444"), lw=1.0, label=CASE_LABELS.get(case_key, case_key))
        lim = max(lim, 1.0)
        ax.set_yscale("symlog", linthresh=max(1.0, 0.08 * lim), linscale=1.0)
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
        ax.set_title(f"{rec['label']} y={rec['y']} x={rec['x']}")
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_ylabel("Obs - fit (symlog)")
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Energy (keV)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.01), frameon=False)
    fig.suptitle(title, y=0.985, fontsize=14)
    plt.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_csv(output_path: Path, records: list[dict], *, element_name: str):
    amp_col = f"{element_name.lower()}_log10_amp"
    fieldnames = ["label", "y", "x", amp_col, "selection_score", "total_counts"]
    for key in ["amponly_log10_amp", "compare_log10_amp", "delta_log10_amp"]:
        if records and key in records[0]:
            fieldnames.append(key)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        for rec in records:
            row = {
                "label": rec["label"],
                "y": rec["y"],
                "x": rec["x"],
                amp_col: rec["element_log10_amp"],
                "selection_score": rec["selection_score"],
                "total_counts": rec["total_counts"],
            }
            for key in ["amponly_log10_amp", "compare_log10_amp", "delta_log10_amp"]:
                if key in rec:
                    row[key] = rec[key]
            writer.writerow(row)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [load_case(case_key, Path(h5_path)) for case_key, h5_path in args.case]
    validate_cases(cases)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    case_by_key = {case.key: case for case in cases}
    if "amponly" not in case_by_key:
        raise KeyError("Selection requires an 'amponly' case")
    reference_case = next((c for c in cases if c.key == args.reference_case), cases[0])
    compare_case = next((c for c in cases if c.key == args.compare_case), None)
    if compare_case is None:
        raise KeyError(f"compare_case {args.compare_case!r} not present in cases")
    if compare_case.key == "amponly":
        raise ValueError("compare_case must be a refit case, not amponly")

    element_name = str(args.element)
    if element_name not in reference_case.amp_names:
        raise KeyError(f"{element_name} not present in reference case amp_names")
    elem_idx = reference_case.amp_names.index(element_name)
    element_map = np.asarray(reference_case.amp_maps[..., elem_idx], dtype=np.float32)
    amponly_element_map = np.asarray(case_by_key["amponly"].amp_maps[..., elem_idx], dtype=np.float32)
    compare_element_map = np.asarray(compare_case.amp_maps[..., elem_idx], dtype=np.float32)

    energy_axis_all = load_energy_axis(reference_case.dataset_path)
    fit_lo, fit_hi = reference_case.fit_energy_range
    full_energy_axis = energy_axis_all[fit_lo : fit_hi + 1]

    info = _element_info(element_name)
    ka1 = float(info.xrf.get("ka1"))
    focus_min = float(args.focus_energy_min) if args.focus_energy_min is not None else ka1 - float(args.focus_half_width_kev)
    focus_max = float(args.focus_energy_max) if args.focus_energy_max is not None else ka1 + float(args.focus_half_width_kev)
    focus_suffix = _energy_suffix(focus_min, focus_max)

    focus_lo = int(np.argmin(np.abs(energy_axis_all - focus_min)))
    focus_hi = int(np.argmin(np.abs(energy_axis_all - focus_max)))
    if focus_hi < focus_lo:
        focus_lo, focus_hi = focus_hi, focus_lo
    focus_energy_axis = energy_axis_all[focus_lo : focus_hi + 1]

    sse_improve_map = np.asarray(case_by_key["amponly"].metrics["sse_over_counts2_map"], dtype=np.float32) - np.minimum.reduce(
        [np.asarray(case.metrics["sse_over_counts2_map"], dtype=np.float32) for case in cases if case.key != "amponly"]
    )
    delta_amp_map = compare_element_map - amponly_element_map
    total_counts_map = np.asarray(reference_case.metrics["total_counts_map"], dtype=np.float32)

    if args.selection_mode == "delta_amp":
        score_map = delta_amp_map
        score_title = f"Δlog10 {element_name} amplitude: {compare_case.label} - amponly"
        extra_fields = {
            "amponly_log10_amp": amponly_element_map,
            "compare_log10_amp": compare_element_map,
            "delta_log10_amp": delta_amp_map,
        }
    else:
        score_map = sse_improve_map
        score_title = "Improvement over amponly"
        extra_fields = None

    records = select_points(
        element_map,
        score_map,
        total_counts_map,
        n_points=int(args.n_points),
        amp_percentile=float(args.amplitude_percentile),
        min_pixel_separation=float(args.min_pixel_separation),
        label_prefix=element_name[:1].upper(),
        extra_fields=extra_fields,
    )
    coords = [(rec["y"], rec["x"]) for rec in records]

    observed_full = load_observed_range(reference_case, coords, fit_lo, fit_hi)
    observed_focus = load_observed_range(reference_case, coords, focus_lo, focus_hi)

    fitted_full: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    fitted_focus: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for case in cases:
        fitted_full[case.key] = reconstruct_case_range(case, coords, observed_full, fit_lo, fit_hi, device)
        fitted_focus[case.key] = reconstruct_case_range(case, coords, observed_full, focus_lo, focus_hi, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(output_dir / "selected_points.csv", records, element_name=element_name)
    plot_shared_amp_maps(
        [(case.key, np.asarray(case.amp_maps[..., elem_idx], dtype=np.float32)) for case in cases],
        records,
        element_name=element_name,
        output_path=output_dir / f"{element_name.lower()}_shared_scale_maps.png",
    )
    plot_delta_maps(
        [
            (
                case.key,
                np.asarray(case.amp_maps[..., elem_idx], dtype=np.float32) - amponly_element_map,
            )
            for case in cases
            if case.key != "amponly"
        ],
        records,
        element_name=element_name,
        output_path=output_dir / f"{element_name.lower()}_delta_vs_amponly_maps.png",
    )
    plot_selected_maps(
        element_map,
        score_map,
        records,
        element_name=element_name,
        score_title=score_title,
        output_path=output_dir / f"{element_name.lower()}_selected_points_maps.png",
    )

    plot_fit_grid(
        full_energy_axis,
        observed_full,
        fitted_full,
        records,
        title=f"{element_name}: full-spectrum experiment vs fitted (+bkg), log10(count + 1)",
        output_path=output_dir / "full_spectrum_fit_grid_logcountp1.png",
        scale_mode="log1p",
    )
    plot_residual_grid(
        full_energy_axis,
        observed_full,
        fitted_full,
        records,
        title=f"{element_name}: full-spectrum residuals by case (symlog)",
        output_path=output_dir / "full_spectrum_residuals_symlog.png",
    )
    plot_fit_grid(
        focus_energy_axis,
        observed_focus,
        fitted_focus,
        records,
        title=f"{element_name}: focus peak experiment vs fitted (+bkg), raw\n{focus_energy_axis[0]:.3f}-{focus_energy_axis[-1]:.3f} keV",
        output_path=output_dir / f"focus_peak_fit_grid_raw_{focus_suffix}.png",
        scale_mode="raw",
    )
    plot_fit_grid(
        focus_energy_axis,
        observed_focus,
        fitted_focus,
        records,
        title=f"{element_name}: focus peak experiment vs fitted (+bkg), normalized\n{focus_energy_axis[0]:.3f}-{focus_energy_axis[-1]:.3f} keV",
        output_path=output_dir / f"focus_peak_fit_grid_normalized_{focus_suffix}.png",
        scale_mode="normalized",
    )
    plot_residual_grid(
        focus_energy_axis,
        observed_focus,
        fitted_focus,
        records,
        title=f"{element_name}: focus peak residuals by case (symlog)\n{focus_energy_axis[0]:.3f}-{focus_energy_axis[-1]:.3f} keV",
        output_path=output_dir / f"focus_peak_residuals_symlog_{focus_suffix}.png",
    )

    print(
        f"Saved {element_name} improvement-point plots to {output_dir} "
        f"(focus window {float(focus_energy_axis[0]):.3f}-{float(focus_energy_axis[-1]):.3f} keV)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select top-improvement pixels for an element and plot full-spectrum and focus-peak comparisons."
    )
    parser.add_argument(
        "--case",
        nargs=2,
        metavar=("CASE_KEY", "H5_PATH"),
        action="append",
        required=True,
        help="Case key and HDF5 path.",
    )
    parser.add_argument("--element", type=str, required=True, help="Element name, e.g. P, S, Cl, Ca.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to write plots.")
    parser.add_argument(
        "--reference_case",
        type=str,
        default=DEFAULT_REFIT_CASE,
        help="Case key used for the element map and total-count map.",
    )
    parser.add_argument(
        "--compare_case",
        type=str,
        default=DEFAULT_REFIT_CASE,
        help="Refit case used for amplitude-delta selection and delta-map outputs.",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        default="sse_improvement",
        choices=["sse_improvement", "delta_amp"],
        help="Select points by full-spectrum SSE improvement or by Δlog10 element amplitude.",
    )
    parser.add_argument("--n_points", type=int, default=16, help="Number of points to select.")
    parser.add_argument(
        "--amplitude_percentile",
        type=float,
        default=85.0,
        help="Initial element-map percentile threshold used to keep selection channel-specific.",
    )
    parser.add_argument(
        "--min_pixel_separation",
        type=float,
        default=8.0,
        help="Minimum Euclidean separation between selected pixels.",
    )
    parser.add_argument(
        "--focus_half_width_kev",
        type=float,
        default=0.15,
        help="Half-width around Ka1 used when focus_energy_min/max are not provided.",
    )
    parser.add_argument("--focus_energy_min", type=float, default=None, help="Optional manual focus-window lower bound in keV.")
    parser.add_argument("--focus_energy_max", type=float, default=None, help="Optional manual focus-window upper bound in keV.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device used for reconstruction.",
    )
    main(parser.parse_args())
