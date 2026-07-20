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

CATEGORY_STYLES = {
    "good": {"prefix": "G", "color": "#1b9e77"},
    "average": {"prefix": "A", "color": "#7570b3"},
    "bad": {"prefix": "B", "color": "#d95f02"},
    "improved": {"prefix": "I", "color": "#e7298a"},
}


@dataclass
class CaseData:
    key: str
    label: str
    color: str
    h5_path: Path
    metrics_path: Path
    dataset_path: Path
    override_path: Path
    spec_vol_key: str | None
    energy_dim: int
    energy_range: tuple[int, int]
    use_snip: bool
    detector_material: str
    escape_factor: float
    map_shape: tuple[int, int]
    amp_maps: np.ndarray
    amp_names: list[str]
    param_maps: dict[str, np.ndarray]
    metrics: dict[str, np.ndarray]


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


def _append_ranked_pixels(
    records: list[dict],
    seen: set[tuple[int, int]],
    ranked_indices: np.ndarray,
    map_shape: tuple[int, int],
    *,
    category: str,
    start_counter: int,
    needed: int,
    min_separation: float,
):
    counter = start_counter
    for flat_idx in ranked_indices:
        if needed <= 0:
            break
        y, x = np.unravel_index(int(flat_idx), map_shape)
        coord = (int(y), int(x))
        if coord in seen:
            continue
        if not _far_enough(seen, int(y), int(x), min_separation=min_separation):
            continue
        seen.add(coord)
        records.append(
            {
                "category": category,
                "label": f"{CATEGORY_STYLES[category]['prefix']}{counter:02d}",
                "y": int(y),
                "x": int(x),
            }
        )
        counter += 1
        needed -= 1
    return counter, needed


def _case_metrics_path(case_key: str, h5_path: Path) -> Path:
    return h5_path.parent / "plots" / case_key / "diagnostics" / "diagnostics_metrics.npz"


def load_case(case_key: str, h5_path: Path) -> CaseData:
    metrics_path = _case_metrics_path(case_key, h5_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics metrics for {case_key}: {metrics_path}")

    with h5py.File(h5_path, "r") as h5:
        dataset_path = Path(decode_text(h5.attrs["dataset"]))
        override_path = Path(decode_text(h5.attrs["override_params_file"]))
        spec_vol_key_attr = decode_text(h5.attrs["spec_vol_key"])
        energy_dim = int(h5.attrs["energy_dim"])
        energy_range = (
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
        color=CASE_COLORS.get(case_key, None) or "#444444",
        h5_path=h5_path,
        metrics_path=metrics_path,
        dataset_path=dataset_path,
        override_path=override_path,
        spec_vol_key=None if spec_vol_key_attr == "None" else spec_vol_key_attr,
        energy_dim=energy_dim,
        energy_range=energy_range,
        use_snip=use_snip,
        detector_material=detector_material,
        escape_factor=escape_factor,
        map_shape=tuple(int(v) for v in amp_maps.shape[:-1]),
        amp_maps=amp_maps,
        amp_names=amp_names,
        param_maps=param_maps,
        metrics=metrics,
    )


def validate_cases(cases: list[CaseData]):
    first = cases[0]
    for case in cases[1:]:
        if case.dataset_path != first.dataset_path:
            raise ValueError(f"Dataset mismatch: {case.dataset_path} != {first.dataset_path}")
        if case.override_path != first.override_path:
            raise ValueError(f"Override mismatch: {case.override_path} != {first.override_path}")
        if case.map_shape != first.map_shape:
            raise ValueError(f"Map shape mismatch: {case.map_shape} != {first.map_shape}")
        if case.energy_range != first.energy_range:
            raise ValueError(f"Energy range mismatch: {case.energy_range} != {first.energy_range}")


def select_pixels(cases: list[CaseData], n_pixels: int, min_separation: float) -> tuple[list[dict], dict[str, np.ndarray]]:
    if n_pixels % 4 != 0:
        raise ValueError("n_pixels must be divisible by 4 so categories are balanced.")

    metrics_stack = np.stack([case.metrics["sse_over_counts2_map"] for case in cases], axis=0)
    counts_map = np.asarray(cases[0].metrics["total_counts_map"], dtype=np.float32)
    valid = np.all(np.isfinite(metrics_stack), axis=0) & np.isfinite(counts_map) & (counts_map > 0)
    log_metrics = np.log10(np.clip(metrics_stack, 1e-12, None))
    combined_log_metric = np.mean(log_metrics, axis=0)
    best_alt_metric = np.min(metrics_stack[1:], axis=0)
    improvement_map = metrics_stack[0] - best_alt_metric

    count_q25 = float(np.percentile(counts_map[valid], 25))
    count_q50 = float(np.percentile(counts_map[valid], 50))
    count_q75 = float(np.percentile(counts_map[valid], 75))
    combined_median = float(np.median(combined_log_metric[valid]))

    high_count_mask = valid & (counts_map >= count_q50)
    mid_count_mask = valid & (counts_map >= count_q25) & (counts_map <= count_q75)
    broad_count_mask = valid & (counts_map >= count_q25)

    ranked = {
        "good": np.argsort(np.where(high_count_mask, combined_log_metric, np.inf).reshape(-1)),
        "average": np.argsort(np.where(mid_count_mask, np.abs(combined_log_metric - combined_median), np.inf).reshape(-1)),
        "bad": np.argsort(np.where(broad_count_mask, combined_log_metric, -np.inf).reshape(-1))[::-1],
        "improved": np.argsort(np.where(high_count_mask, improvement_map, -np.inf).reshape(-1))[::-1],
    }

    records: list[dict] = []
    seen: set[tuple[int, int]] = set()
    per_category = n_pixels // 4

    for category in ("good", "average", "bad", "improved"):
        counter, needed = 1, per_category
        counter, needed = _append_ranked_pixels(
            records,
            seen,
            ranked[category],
            cases[0].map_shape,
            category=category,
            start_counter=counter,
            needed=needed,
            min_separation=min_separation,
        )
        if needed > 0:
            fallback = np.argsort(np.where(valid, combined_log_metric, np.inf).reshape(-1))
            _append_ranked_pixels(
                records,
                seen,
                fallback,
                cases[0].map_shape,
                category=category,
                start_counter=counter,
                needed=needed,
                min_separation=min_separation,
            )

    for rec in records:
        y, x = rec["y"], rec["x"]
        rec["total_counts"] = float(counts_map[y, x])
        rec["combined_log_sse_over_counts2"] = float(combined_log_metric[y, x])
        rec["improvement_over_amponly"] = float(improvement_map[y, x])

    return records, {
        "counts_map": counts_map,
        "combined_log_metric": combined_log_metric,
        "improvement_map": improvement_map,
    }


def load_observed_spectra(case: CaseData, pixels: list[dict]) -> dict[tuple[int, int], np.ndarray]:
    obs_by_coord: dict[tuple[int, int], np.ndarray] = {}
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
        for rec in pixels:
            y, x = rec["y"], rec["x"]
            slicer = [slice(None)] * dset.ndim
            slicer[energy_dim] = slice(case.energy_range[0], case.energy_range[1] + 1)
            for md, idx in zip(map_dims, (y, x)):
                slicer[md] = int(idx)
            obs = dset[tuple(slicer)].astype(np.float32, copy=False).reshape(-1)
            obs_by_coord[(y, x)] = obs
    return obs_by_coord


def reconstruct_case_pixel_spectra(
    case: CaseData,
    pixels: list[dict],
    obs_by_coord: dict[tuple[int, int], np.ndarray],
    device: torch.device,
) -> dict[tuple[int, int], dict[str, np.ndarray | float]]:
    override = parse_override_params_file(str(case.override_path))
    e_consts = read_constants(str(case.override_path), default_elem_info)
    base_params = _base_param_dict(override)

    out: dict[tuple[int, int], dict[str, np.ndarray | float]] = {}
    for rec in pixels:
        y, x = rec["y"], rec["x"]
        obs = obs_by_coord[(y, x)]
        obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
        tensors = _pixel_tensors(base_params, case.param_maps, case.amp_maps, case.amp_names, y, x, device)
        with torch.no_grad():
            spec_fit = model_spec(
                tensors,
                case.energy_range,
                elements_to_fit=case.amp_names,
                device=device,
                e_consts=e_consts,
                elem_info=default_elem_info,
                detector_type=case.detector_material,
                escape_factor=case.escape_factor,
            )
            bkg = _compute_background(
                obs_t,
                case.energy_range,
                tensors,
                use_snip=case.use_snip,
                device=device,
            )
            total = spec_fit + bkg
        total_np = total.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
        resid_np = obs.astype(np.float32, copy=False) - total_np
        denom = max(float(np.sum(obs)), 1.0)
        out[(y, x)] = {
            "total": total_np,
            "residual_frac": resid_np / denom,
            "sse_over_counts2": float(case.metrics["sse_over_counts2_map"][y, x]),
            "sse": float(case.metrics["sse_map"][y, x]),
        }
    return out


def _metric_image_limits(img: np.ndarray) -> tuple[float, float]:
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1.0, 99.0]).astype(float)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def plot_selected_pixels_map(
    maps: dict[str, np.ndarray],
    records: list[dict],
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    for ax, img, title in (
        (axes[0], maps["combined_log_metric"], "Selection Map: Mean log10(SSE / Counts^2)"),
        (axes[1], maps["improvement_map"], "Selection Map: Amplitude-only Improvement"),
    ):
        vmin, vmax = _metric_image_limits(img)
        im = ax.imshow(img, origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
        for rec in records:
            style = CATEGORY_STYLES[rec["category"]]
            ax.scatter(rec["x"], rec["y"], s=48, facecolors="none", edgecolors=style["color"], linewidths=1.6)
            ax.text(rec["x"] + 2, rec["y"] + 2, rec["label"], color=style["color"], fontsize=8, weight="bold")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.82)

    handles = []
    labels = []
    for category in ("good", "average", "bad", "improved"):
        style = CATEGORY_STYLES[category]
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor="none",
                markeredgecolor=style["color"],
                markeredgewidth=1.8,
            )
        )
        labels.append(category)
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary_overlays(
    records: list[dict],
    obs_by_coord: dict[tuple[int, int], np.ndarray],
    per_case: dict[str, dict[tuple[int, int], dict[str, np.ndarray | float]]],
    cases: list[CaseData],
    output_path: Path,
    *,
    yscale: str,
):
    n = len(records)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.3 * n_rows), squeeze=False)
    channel_axis = np.arange(cases[0].energy_range[0], cases[0].energy_range[1] + 1, dtype=np.int32)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, rec in enumerate(records):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axis("on")
        obs = obs_by_coord[(rec["y"], rec["x"])]
        ax.plot(channel_axis, obs, color="0.2", lw=1.2, label="Experiment")
        metric_parts = []
        best_case = None
        best_metric = float("inf")
        for case in cases:
            fit = per_case[case.key][(rec["y"], rec["x"])]
            metric = float(fit["sse_over_counts2"])
            metric_parts.append(f"{case.key}={metric:.3e}")
            if metric < best_metric:
                best_metric = metric
                best_case = case.label
            ax.plot(channel_axis, fit["total"], color=case.color, lw=1.0, label=case.label)
        if yscale == "log":
            ax.set_yscale("log")
            ax.set_ylim(1, None)
        ax.set_title(
            f"{rec['label']} {rec['category']} y={rec['y']} x={rec['x']}\n"
            f"counts={rec['total_counts']:.0f}, best={best_case}"
        )
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Detector channel")
        ax.set_ylabel("Counts")
        ax.grid(alpha=0.18, linewidth=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_summary_residuals(
    records: list[dict],
    per_case: dict[str, dict[tuple[int, int], dict[str, np.ndarray | float]]],
    cases: list[CaseData],
    output_path: Path,
):
    n = len(records)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.4 * n_cols, 3.0 * n_rows), squeeze=False)
    channel_axis = np.arange(cases[0].energy_range[0], cases[0].energy_range[1] + 1, dtype=np.int32)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, rec in enumerate(records):
        ax = axes[idx // n_cols][idx % n_cols]
        ax.axis("on")
        ax.axhline(0.0, color="0.6", lw=0.8)
        lim = 0.0
        for case in cases:
            resid = per_case[case.key][(rec["y"], rec["x"])]["residual_frac"]
            resid = np.asarray(resid, dtype=np.float32)
            finite = resid[np.isfinite(resid)]
            if finite.size:
                lim = max(lim, float(np.percentile(np.abs(finite), 99)))
            ax.plot(channel_axis, resid, color=case.color, lw=0.95, label=case.label)
        lim = max(lim, 1e-4)
        ax.set_ylim(-1.1 * lim, 1.1 * lim)
        ax.set_title(f"{rec['label']} {rec['category']} y={rec['y']} x={rec['x']}")
        if idx // n_cols == n_rows - 1:
            ax.set_xlabel("Detector channel")
        ax.set_ylabel("Residual / total counts")
        ax.grid(alpha=0.18, linewidth=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.01))
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_category_pages(
    records: list[dict],
    obs_by_coord: dict[tuple[int, int], np.ndarray],
    per_case: dict[str, dict[tuple[int, int], dict[str, np.ndarray | float]]],
    cases: list[CaseData],
    output_dir: Path,
):
    channel_axis = np.arange(cases[0].energy_range[0], cases[0].energy_range[1] + 1, dtype=np.int32)
    for category in ("good", "average", "bad", "improved"):
        subset = [rec for rec in records if rec["category"] == category]
        fig, axes = plt.subplots(len(subset), 2, figsize=(13, 3.6 * len(subset)), squeeze=False)
        for row, rec in enumerate(subset):
            obs = obs_by_coord[(rec["y"], rec["x"])]
            ax = axes[row][0]
            ax.plot(channel_axis, obs, color="0.2", lw=1.2, label="Experiment")
            for case in cases:
                fit = per_case[case.key][(rec["y"], rec["x"])]
                ax.plot(channel_axis, fit["total"], color=case.color, lw=1.0, label=case.label)
            ax.set_title(
                f"{rec['label']} y={rec['y']} x={rec['x']} counts={rec['total_counts']:.0f} "
                f"improvement={rec['improvement_over_amponly']:.3e}"
            )
            ax.set_ylabel("Counts")
            ax.grid(alpha=0.18, linewidth=0.5)
            if row == len(subset) - 1:
                ax.set_xlabel("Detector channel")

            ax = axes[row][1]
            ax.plot(channel_axis, obs, color="0.2", lw=1.2, label="Experiment")
            for case in cases:
                fit = per_case[case.key][(rec["y"], rec["x"])]
                metric = float(fit["sse_over_counts2"])
                ax.plot(channel_axis, fit["total"], color=case.color, lw=1.0, label=f"{case.label} ({metric:.3e})")
            ax.set_yscale("log")
            ax.set_ylim(1, None)
            ax.grid(alpha=0.18, linewidth=0.5)
            if row == len(subset) - 1:
                ax.set_xlabel("Detector channel")
            ax.set_ylabel("Counts")

        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
        fig.suptitle(f"Representative {category.capitalize()} Pixels", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{category}_pixels_overlay.png", bbox_inches="tight")
        plt.close(fig)


def find_peak_centers(
    obs: np.ndarray,
    *,
    n_peaks: int,
    min_separation: int,
    smooth_window: int,
    fallback_centers: list[int] | None = None,
) -> list[int]:
    if smooth_window < 1:
        smooth_window = 1
    if smooth_window % 2 == 0:
        smooth_window += 1
    kernel = np.ones(smooth_window, dtype=np.float32) / float(smooth_window)
    smoothed = np.convolve(obs.astype(np.float32, copy=False), kernel, mode="same")
    candidate_indices = []
    threshold = max(float(np.percentile(smoothed, 80.0)), 0.12 * float(np.max(smoothed)))
    for idx in range(1, smoothed.size - 1):
        if smoothed[idx] >= smoothed[idx - 1] and smoothed[idx] >= smoothed[idx + 1] and smoothed[idx] >= threshold:
            candidate_indices.append(idx)
    if not candidate_indices:
        candidate_indices = list(np.argsort(smoothed)[::-1][: max(12, n_peaks * 4)])

    selected: list[int] = []
    for idx in sorted(candidate_indices, key=lambda i: float(smoothed[i]), reverse=True):
        if all(abs(int(idx) - int(prev)) >= int(min_separation) for prev in selected):
            selected.append(int(idx))
        if len(selected) >= n_peaks:
            break

    if fallback_centers:
        for idx in fallback_centers:
            if all(abs(int(idx) - int(prev)) >= int(min_separation) for prev in selected):
                selected.append(int(idx))
            if len(selected) >= n_peaks:
                break

    if len(selected) < n_peaks:
        ranked = np.argsort(smoothed)[::-1]
        for idx in ranked:
            if all(abs(int(idx) - int(prev)) >= int(min_separation) for prev in selected):
                selected.append(int(idx))
            if len(selected) >= n_peaks:
                break

    return sorted(int(idx) for idx in selected[:n_peaks])


def plot_peak_zoom_pages(
    records: list[dict],
    obs_by_coord: dict[tuple[int, int], np.ndarray],
    per_case: dict[str, dict[tuple[int, int], dict[str, np.ndarray | float]]],
    cases: list[CaseData],
    output_dir: Path,
    *,
    n_peak_windows: int,
    peak_half_width: int,
    peak_min_separation: int,
    peak_smooth_window: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    global_obs = np.mean(np.stack([obs_by_coord[(rec["y"], rec["x"])] for rec in records], axis=0), axis=0)
    fallback_centers = find_peak_centers(
        global_obs,
        n_peaks=n_peak_windows,
        min_separation=peak_min_separation,
        smooth_window=peak_smooth_window,
        fallback_centers=None,
    )
    channel_axis = np.arange(cases[0].energy_range[0], cases[0].energy_range[1] + 1, dtype=np.int32)

    for category in ("good", "average", "bad", "improved"):
        subset = [rec for rec in records if rec["category"] == category]
        if not subset:
            continue

        fig_overlay, axes_overlay = plt.subplots(
            len(subset),
            n_peak_windows,
            figsize=(4.2 * n_peak_windows, 3.0 * len(subset)),
            squeeze=False,
        )
        fig_resid, axes_resid = plt.subplots(
            len(subset),
            n_peak_windows,
            figsize=(4.2 * n_peak_windows, 2.7 * len(subset)),
            squeeze=False,
        )

        for row, rec in enumerate(subset):
            coord = (rec["y"], rec["x"])
            obs = obs_by_coord[coord]
            centers = find_peak_centers(
                obs,
                n_peaks=n_peak_windows,
                min_separation=peak_min_separation,
                smooth_window=peak_smooth_window,
                fallback_centers=fallback_centers,
            )
            for col, center in enumerate(centers):
                lo = max(0, int(center) - int(peak_half_width))
                hi = min(obs.size - 1, int(center) + int(peak_half_width))
                x = channel_axis[lo : hi + 1]
                obs_slice = obs[lo : hi + 1]

                ax = axes_overlay[row][col]
                ax.plot(x, obs_slice, color="0.15", lw=0.8, marker="o", ms=1.8, label="Experiment")
                y_min = float(np.min(obs_slice))
                y_max = float(np.max(obs_slice))
                title = f"{rec['label']} ch {channel_axis[int(center)]}"
                for case in cases:
                    fit_slice = np.asarray(per_case[case.key][coord]["total"], dtype=np.float32)[lo : hi + 1]
                    y_min = min(y_min, float(np.min(fit_slice)))
                    y_max = max(y_max, float(np.max(fit_slice)))
                    ax.plot(x, fit_slice, color=case.color, lw=1.1, label=case.label)
                pad = max(1.0, 0.08 * (y_max - y_min))
                ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
                ax.set_title(title)
                ax.grid(alpha=0.18, linewidth=0.5)
                if row == len(subset) - 1:
                    ax.set_xlabel("Detector channel")
                if col == 0:
                    ax.set_ylabel(f"{rec['label']}\nCounts")

                ax = axes_resid[row][col]
                ax.axhline(0.0, color="0.6", lw=0.8)
                resid_lim = 0.0
                for case in cases:
                    fit_slice = np.asarray(per_case[case.key][coord]["total"], dtype=np.float32)[lo : hi + 1]
                    resid_slice = obs_slice - fit_slice
                    finite = resid_slice[np.isfinite(resid_slice)]
                    if finite.size:
                        resid_lim = max(resid_lim, float(np.percentile(np.abs(finite), 99)))
                    ax.plot(x, resid_slice, color=case.color, lw=1.0, label=case.label)
                resid_lim = max(resid_lim, 1.0)
                ax.set_ylim(-1.1 * resid_lim, 1.1 * resid_lim)
                ax.set_title(title)
                ax.grid(alpha=0.18, linewidth=0.5)
                if row == len(subset) - 1:
                    ax.set_xlabel("Detector channel")
                if col == 0:
                    ax.set_ylabel(f"{rec['label']}\nObs - fit")

        handles, labels = axes_overlay[0][0].get_legend_handles_labels()
        fig_overlay.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.01))
        fig_overlay.suptitle(f"Peak Zoom Overlays: {category.capitalize()} Pixels", y=1.03, fontsize=14)
        plt.figure(fig_overlay.number)
        plt.tight_layout()
        plt.savefig(output_dir / f"{category}_peak_windows_overlay.png", bbox_inches="tight")
        plt.close(fig_overlay)

        handles, labels = axes_resid[0][0].get_legend_handles_labels()
        fig_resid.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.01))
        fig_resid.suptitle(f"Peak Zoom Residuals: {category.capitalize()} Pixels", y=1.03, fontsize=14)
        plt.figure(fig_resid.number)
        plt.tight_layout()
        plt.savefig(output_dir / f"{category}_peak_windows_residuals.png", bbox_inches="tight")
        plt.close(fig_resid)


def write_pixel_csv(
    records: list[dict],
    cases: list[CaseData],
    per_case: dict[str, dict[tuple[int, int], dict[str, np.ndarray | float]]],
    output_path: Path,
):
    fieldnames = [
        "label",
        "category",
        "y",
        "x",
        "total_counts",
        "combined_log_sse_over_counts2",
        "improvement_over_amponly",
        "best_case",
    ]
    for case in cases:
        fieldnames.append(f"{case.key}_sse_over_counts2")
        fieldnames.append(f"{case.key}_sse")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = {k: rec[k] for k in fieldnames if k in rec}
            best_case = None
            best_metric = float("inf")
            for case in cases:
                fit = per_case[case.key][(rec["y"], rec["x"])]
                row[f"{case.key}_sse_over_counts2"] = float(fit["sse_over_counts2"])
                row[f"{case.key}_sse"] = float(fit["sse"])
                if float(fit["sse_over_counts2"]) < best_metric:
                    best_metric = float(fit["sse_over_counts2"])
                    best_case = case.key
            row["best_case"] = best_case
            writer.writerow(row)


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_pixel_dir = output_dir / "category_pages"
    per_pixel_dir.mkdir(parents=True, exist_ok=True)

    cases = [load_case(case_key, Path(h5_path)) for case_key, h5_path in args.case]
    if len(cases) < 2:
        raise ValueError("Need at least two cases.")
    validate_cases(cases)

    device_name = args.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    records, selection_maps = select_pixels(cases, int(args.n_pixels), float(args.min_pixel_separation))
    obs_by_coord = load_observed_spectra(cases[0], records)

    per_case = {}
    for case in cases:
        per_case[case.key] = reconstruct_case_pixel_spectra(case, records, obs_by_coord, device)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_pixel_csv(records, cases, per_case, output_dir / "selected_pixels.csv")
    plot_selected_pixels_map(selection_maps, records, output_dir / "selected_pixels_map.png")
    plot_summary_overlays(
        records,
        obs_by_coord,
        per_case,
        cases,
        output_dir / "selected_pixels_overlay_linear.png",
        yscale="linear",
    )
    plot_summary_overlays(
        records,
        obs_by_coord,
        per_case,
        cases,
        output_dir / "selected_pixels_overlay_log.png",
        yscale="log",
    )
    plot_summary_residuals(
        records,
        per_case,
        cases,
        output_dir / "selected_pixels_residuals.png",
    )
    plot_category_pages(records, obs_by_coord, per_case, cases, per_pixel_dir)
    plot_peak_zoom_pages(
        records,
        obs_by_coord,
        per_case,
        cases,
        output_dir / "peak_zoom_pages",
        n_peak_windows=int(args.n_peak_windows),
        peak_half_width=int(args.peak_half_width),
        peak_min_separation=int(args.peak_min_separation),
        peak_smooth_window=int(args.peak_smooth_window),
    )

    print(f"Saved representative pixel comparisons to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare experimental and fitted spectra for representative pixels across multiple batch-fit cases."
    )
    parser.add_argument(
        "--case",
        nargs=2,
        metavar=("CASE_KEY", "H5_PATH"),
        action="append",
        required=True,
        help="Case key and corresponding HDF5 path. The first case is treated as the reference for improvement ranking.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write comparison plots and CSV outputs.",
    )
    parser.add_argument(
        "--n_pixels",
        type=int,
        default=20,
        help="Number of representative pixels to select. Must be divisible by 4.",
    )
    parser.add_argument(
        "--min_pixel_separation",
        type=float,
        default=18.0,
        help="Minimum Euclidean separation, in pixels, between selected coordinates.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device used for spectrum reconstruction.",
    )
    parser.add_argument(
        "--n_peak_windows",
        type=int,
        default=4,
        help="Number of zoomed peak windows to plot per selected pixel.",
    )
    parser.add_argument(
        "--peak_half_width",
        type=int,
        default=22,
        help="Half-width of each zoomed peak window, in detector channels.",
    )
    parser.add_argument(
        "--peak_min_separation",
        type=int,
        default=90,
        help="Minimum separation between selected peak centers, in detector channels.",
    )
    parser.add_argument(
        "--peak_smooth_window",
        type=int,
        default=7,
        help="Smoothing window used to detect peak centers.",
    )
    main(parser.parse_args())
