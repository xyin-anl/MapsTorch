from __future__ import annotations

from itertools import product
from typing import Iterable

import h5py
import numpy as np


def decode_text(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8").strip("\x00")
    return str(value).strip("\x00")


def decode_names(values: Iterable) -> list[str]:
    return [decode_text(value) for value in values]


def sanitize_component_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def display_component_name(name: str) -> str:
    if name in {"COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"}:
        return name
    if "_" not in name:
        return name
    parts = [part for part in name.split("_") if part]
    if not parts:
        return name
    if len(parts) == 2 and parts[0] == parts[1]:
        return f"{parts[0]} pileup"
    return f"{' + '.join(parts)} pileup"


def unique_in_order(values: Iterable[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def resolve_h5_key(
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
        raise KeyError(f"Could not find {label} dataset in the HDF5 file. Tried: {fallbacks}")
    return None


def coerce_energy_dim(energy_dim: int, ndim: int) -> int:
    energy_dim = int(energy_dim)
    if energy_dim < 0:
        energy_dim = ndim + energy_dim
    if energy_dim < 0 or energy_dim >= ndim:
        raise ValueError(f"energy_dim={energy_dim} is out of bounds for ndim={ndim}")
    return energy_dim


def default_tile_shape(
    map_shape: tuple[int, ...],
    *,
    one_d_limit: int = 2048,
    spatial_limit: int = 32,
) -> tuple[int, ...]:
    if len(map_shape) == 1:
        return (min(int(one_d_limit), int(map_shape[0])),)
    if len(map_shape) >= 2:
        lead = (1,) * max(0, len(map_shape) - 2)
        h = min(int(spatial_limit), int(map_shape[-2]))
        w = min(int(spatial_limit), int(map_shape[-1]))
        return lead + (h, w)
    raise ValueError("map_shape must have at least 1 dimension")


def normalize_tile_shape(
    map_shape: tuple[int, ...],
    tile_shape: Iterable[int] | None,
    *,
    one_d_limit: int = 2048,
    spatial_limit: int = 32,
) -> tuple[int, ...]:
    if tile_shape is None:
        return default_tile_shape(
            map_shape,
            one_d_limit=one_d_limit,
            spatial_limit=spatial_limit,
        )
    normalized = tuple(int(x) for x in tile_shape)
    if len(normalized) != len(map_shape):
        raise ValueError(
            f"tile_shape rank {len(normalized)} does not match map rank {len(map_shape)}"
        )
    if any(x <= 0 for x in normalized):
        raise ValueError("tile_shape values must be > 0")
    return normalized


def iter_tiles(map_shape: tuple[int, ...], tile_shape: tuple[int, ...]):
    if len(map_shape) != len(tile_shape):
        raise ValueError("tile_shape must match map_shape rank")
    starts = [
        range(0, int(map_shape[d]), max(1, int(tile_shape[d])))
        for d in range(len(map_shape))
    ]
    for start_idx in product(*starts):
        yield tuple(
            slice(s, min(int(map_shape[d]), s + max(1, int(tile_shape[d]))))
            for d, s in enumerate(start_idx)
        )


def tile_grid_shape(map_shape: tuple[int, ...], tile_shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(
        (int(map_shape[d]) + int(tile_shape[d]) - 1) // int(tile_shape[d])
        for d in range(len(map_shape))
    )
