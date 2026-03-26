import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from mapstorch.batchfit import (
    decode_names,
    display_component_name,
    sanitize_component_name,
)
from mapstorch.plot import plot_amp_map, plot_amp_maps_grid


def _param_cmap(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "viridis"
    lo, hi = np.percentile(finite, [1, 99]).astype(float)
    if lo < 0.0 < hi:
        return "RdBu_r"
    return "viridis"


def _plot_param_grid(
    param_maps: dict[str, np.ndarray],
    *,
    filename: str,
    percentile: tuple[float, float],
    n_cols: int = 4,
):
    names = list(param_maps.keys())
    n_cols = max(1, int(n_cols))
    n_rows = int(np.ceil(len(names) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.3 * n_cols, 3.3 * n_rows), squeeze=False)

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])

    for idx, name in enumerate(names):
        ax = axes[idx // n_cols][idx % n_cols]
        img = np.asarray(param_maps[name], dtype=np.float32)
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
            cmap=_param_cmap(img),
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            aspect=aspect,
        )
        ax.set_title(name)
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
    amp_dir = output_dir / "elements"
    param_dir = output_dir / "parameters"
    amp_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as h5:
        amp_maps = h5["amp_maps_log10"][:]
        amp_names = decode_names(h5["amp_names"][:])
        amp_display_names = [display_component_name(name) for name in amp_names]
        param_names = sorted(h5["param_maps"].keys())
        param_maps = {name: h5["param_maps"][name][:] for name in param_names}

    map_shape = amp_maps.shape[:-1]
    if len(map_shape) > 2:
        raise ValueError(
            f"{input_path} contains map_shape={map_shape}. "
            "plot_batchfit_maps.py currently supports 1D or 2D maps only."
        )

    amp_percentile = (float(args.amp_percentile[0]), float(args.amp_percentile[1]))
    param_percentile = (float(args.param_percentile[0]), float(args.param_percentile[1]))

    plot_amp_maps_grid(
        amp_maps,
        amp_display_names,
        target_elems=amp_display_names,
        n_cols=int(args.amp_n_cols),
        is_log10=True,
        display=args.amp_display,
        cmap=args.amp_cmap,
        percentile=amp_percentile,
        colorbar=True,
        filename=str(output_dir / "elements_grid.png"),
        save=True,
        show=False,
    )

    for idx, name in enumerate(amp_names):
        display_name = amp_display_names[idx]
        img = amp_maps[..., idx]
        if args.amp_display == "linear":
            with np.errstate(over="ignore", invalid="ignore"):
                img = np.power(10.0, img)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img, 0.0, None)
        plot_amp_map(
            img,
            name=display_name,
            title=display_name,
            filename=str(amp_dir / f"{idx:02d}_{sanitize_component_name(name)}.png"),
            cmap=args.amp_cmap,
            percentile=amp_percentile,
            save=True,
            show=False,
        )

    _plot_param_grid(
        param_maps,
        filename=str(output_dir / "parameters_grid.png"),
        percentile=param_percentile,
        n_cols=int(args.param_n_cols),
    )

    for idx, name in enumerate(param_names):
        plot_amp_map(
            param_maps[name],
            name=name,
            title=name,
            filename=str(param_dir / f"{idx:02d}_{sanitize_component_name(name)}.png"),
            cmap=_param_cmap(param_maps[name]),
            percentile=param_percentile,
            save=True,
            show=False,
        )

    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot element amplitude maps and parameter maps from a tiled batch-fit HDF5 output."
    )
    parser.add_argument("input_path", type=str, help="Path to batch-fit output HDF5")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write PNG outputs",
    )
    parser.add_argument(
        "--amp_display",
        type=str,
        default="log10",
        choices=["log10", "linear"],
        help="Display element amplitude maps in log10 or linear units",
    )
    parser.add_argument(
        "--amp_cmap",
        type=str,
        default="viridis",
        help="Colormap for element amplitude maps",
    )
    parser.add_argument(
        "--amp_percentile",
        type=float,
        nargs=2,
        default=[1, 99],
        help="Percentile range for amplitude map contrast",
    )
    parser.add_argument(
        "--param_percentile",
        type=float,
        nargs=2,
        default=[1, 99],
        help="Percentile range for parameter map contrast",
    )
    parser.add_argument(
        "--amp_n_cols",
        type=int,
        default=5,
        help="Column count for the element summary grid",
    )
    parser.add_argument(
        "--param_n_cols",
        type=int,
        default=4,
        help="Column count for the parameter summary grid",
    )
    main(parser.parse_args())
