import argparse
import csv
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mapstorch.batchfit import decode_names
from mapstorch.batchfit import display_component_name
from mapstorch.batchfit import sanitize_component_name


def _decode_case_label(text: str) -> str:
    return str(text).replace("_", " ")


def _load_case_maps(h5_path: Path) -> tuple[list[str], np.ndarray]:
    with h5py.File(h5_path, "r") as h5:
        amp_maps_log10 = h5["amp_maps_log10"][:].astype(np.float32, copy=False)
        amp_names = decode_names(h5["amp_names"][:])
    with np.errstate(over="ignore", invalid="ignore"):
        amp_maps_linear = np.power(10.0, amp_maps_log10, dtype=np.float32)
    amp_maps_linear = np.nan_to_num(amp_maps_linear, nan=0.0, posinf=0.0, neginf=0.0)
    amp_maps_linear = np.clip(amp_maps_linear, 0.0, None)
    return amp_names, amp_maps_linear


def _shared_vmax(values: list[np.ndarray], percentile: float) -> float:
    finite_parts = []
    for arr in values:
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            finite_parts.append(finite.reshape(-1))
    if not finite_parts:
        return 1.0
    merged = np.concatenate(finite_parts, axis=0)
    vmax = float(np.percentile(merged, percentile))
    return max(vmax, 1e-12)


def _plot_element_panel(
    element_name: str,
    case_maps: list[tuple[str, np.ndarray]],
    *,
    vmax: float,
    output_path: Path,
):
    n_cols = len(case_maps)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 4.3), squeeze=False)
    axes = axes[0]
    mappable = None
    for ax, (case_label, img) in zip(axes, case_maps):
        mappable = ax.imshow(img, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
        ax.set_title(case_label)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{element_name}: shared linear scale", y=0.98, fontsize=14)
    fig.colorbar(mappable, ax=axes, shrink=0.85, pad=0.02, label="Linear amplitude")
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_grid(
    elements: list[str],
    per_element_case_maps: dict[str, list[tuple[str, np.ndarray]]],
    limits: dict[str, float],
    *,
    output_path: Path,
):
    n_rows = len(elements)
    n_cols = len(next(iter(per_element_case_maps.values())))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    for row, element_name in enumerate(elements):
        case_maps = per_element_case_maps[element_name]
        vmax = limits[element_name]
        mappable = None
        for col, (case_label, img) in enumerate(case_maps):
            ax = axes[row][col]
            mappable = ax.imshow(img, origin="lower", cmap="viridis", vmin=0.0, vmax=vmax)
            if row == 0:
                ax.set_title(case_label)
            if col == 0:
                ax.set_ylabel(display_component_name(element_name))
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(
            mappable,
            ax=axes[row, :],
            shrink=0.84,
            pad=0.01,
            label=f"{display_component_name(element_name)} linear amplitude",
        )

    fig.suptitle("Shared-scale linear element maps across cases", y=0.995, fontsize=15)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_limits_csv(output_path: Path, limits: dict[str, float]):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["element", "vmin", "vmax"])
        for element_name, vmax in limits.items():
            writer.writerow([element_name, 0.0, float(vmax)])


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    element_dir = output_dir / "elements"
    element_dir.mkdir(parents=True, exist_ok=True)

    case_defs: list[tuple[str, Path]] = [(label, Path(h5_path)) for label, h5_path in args.case]
    loaded_cases: list[tuple[str, list[str], np.ndarray]] = []
    for case_label, h5_path in case_defs:
        amp_names, amp_maps_linear = _load_case_maps(h5_path)
        loaded_cases.append((_decode_case_label(case_label), amp_names, amp_maps_linear))

    ref_names = loaded_cases[0][1]
    ref_name_set = set(ref_names)
    for case_label, amp_names, _ in loaded_cases[1:]:
        if set(amp_names) != ref_name_set:
            raise ValueError(f"amp_names mismatch for case {case_label}")

    if args.elements:
        elements = [str(name) for name in args.elements]
    else:
        elements = list(ref_names)

    per_element_case_maps: dict[str, list[tuple[str, np.ndarray]]] = {}
    limits: dict[str, float] = {}
    for element_name in elements:
        if element_name not in ref_name_set:
            raise KeyError(f"Element {element_name!r} not found in amp_names")
        case_maps = []
        values = []
        for case_label, amp_names, amp_maps_linear in loaded_cases:
            elem_idx = amp_names.index(element_name)
            img = np.asarray(amp_maps_linear[..., elem_idx], dtype=np.float32)
            case_maps.append((case_label, img))
            values.append(img)
        vmax = _shared_vmax(values, float(args.percentile))
        per_element_case_maps[element_name] = case_maps
        limits[element_name] = vmax

        output_path = element_dir / f"{sanitize_component_name(element_name)}_shared_linear_cases.png"
        _plot_element_panel(
            display_component_name(element_name),
            case_maps,
            vmax=vmax,
            output_path=output_path,
        )

    _plot_grid(
        elements,
        per_element_case_maps,
        limits,
        output_path=output_dir / "selected_elements_shared_linear_grid.png",
    )
    _write_limits_csv(output_dir / "shared_linear_limits.csv", limits)

    print(f"Saved shared-scale linear case maps to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot shared-scale linear amplitude maps for selected elements across multiple HDF5 cases."
    )
    parser.add_argument(
        "--case",
        nargs=2,
        metavar=("CASE_LABEL", "H5_PATH"),
        action="append",
        required=True,
        help="Case label and HDF5 path. Repeat for each case.",
    )
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="Optional element/component names to compare. Default: all.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.0,
        help="Upper percentile used to define the shared vmax across cases for each element.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write outputs.",
    )
    main(parser.parse_args())
