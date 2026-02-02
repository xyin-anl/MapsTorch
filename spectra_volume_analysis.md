# Spectra Volume Fitting & Analysis in MapsTorch (`mapstorch`)

This document describes the new **spectra volume** (MCA cube / hyperspectral map) fitting capability in `mapstorch` and how it extends the existing single 1D (integrated) spectrum fitting workflow.

The design supports two complementary modes:

1. **Default / fast**: **Global shared parameters** (energy calibration, peak shapes, scatter params, etc.) + **per-pixel amplitudes** solved via a fast **linear** solver.
2. **Advanced / rare**: **Per-pixel / per-tile / per-ROI tuning** of *all or a subset* of `default_fitting_params`, batched so it can run efficiently on GPU when available.

---

## Data Conventions

### Spectrum volume shape (`mca_arr`)

`mca_arr` does **not** have a fixed shape. The only requirement is that one axis corresponds to the **energy/channel** dimension.

Typical examples:

- `(n_channels, height, width)` e.g. `(2048, H, W)` or `(4096, H, W)`
- `(height, width, n_channels)` if your stack is stored channel-last
- Any higher-dimensional layout is supported (e.g. tiles, time frames), as long as you specify which axis is the channel axis.

In the new APIs, you tell `mapstorch` which axis is the channel axis using `energy_dim`.

### Channel range (`energy_range`)

`energy_range = (ch0, ch1)` is always expressed in **channel indices** (inclusive), consistent with the existing `fit_spec` API and the physics model in `mapstorch.map.model_spec`.

### Parameterization: amplitudes are `log10(A)`

Element/scatter “amplitudes” in the physics model are stored as **log10 amplitudes**:

- The spectrum model uses `10 ** logA` as the linear amplitude scale.
- The linear “volume solve” solves for linear amplitudes by default and can optionally return `log10` amplitudes so they match existing `mapstorch` conventions.

---

## Workflow A (Recommended): Global Params + Per-Pixel Amplitudes (Fast Linear Solve)

This is the default workflow for fitting a whole MCA volume:

1. Fit a **single integrated spectrum** to estimate global parameters.
2. With those parameters fixed, solve per-pixel amplitudes using a **linear** solver.

This is typically orders of magnitude faster than optimizing all parameters per pixel.

### Convenience API: `fit_mca_arr`

Entry point: `mapstorch.opt.fit_mca_arr` (`mapstorch/opt.py:827`)

What it does:

- Moves `energy_dim` to the last axis internally.
- Computes `int_spec = sum(mca_arr over all non-energy axes)`.
- Calls `fit_spec(...)` on `int_spec` to fit global parameters.
- Calls `solve_mca_amplitudes(...)` to compute amplitude maps.

Minimal example:

```python
import numpy as np
from mapstorch.opt import fit_mca_arr
from mapstorch.default import default_fitting_elems, default_fitting_params

# mca_arr: shape (n_ch, H, W)
mca_arr = np.load("mca.npy")
energy_range = (0, 2047)

fitted_params, amp_maps, amp_names, loss_trace = fit_mca_arr(
    mca_arr,
    energy_range,
    elements_to_fit=list(default_fitting_elems),
    fitting_params=list(default_fitting_params),
    energy_dim=0,
    use_snip=True,
    optimizer="adamw",
    n_iter=500,
    amp_method="ridge",     # "ridge" | "lstsq" | "nnls_pgd"
    nonneg=False,           # optional clamp after solve
    return_log10=True,      # keep mapstorch convention (log10 amplitudes)
    device="cuda",          # falls back to CPU if unavailable
)

# amp_maps has shape (H, W, n_components)
# amp_names defines the last-axis component order
```

Recommended configuration knobs:

- `amp_method="ridge"` with a small `ridge_lambda` is usually the most stable.
- Set `use_snip=False` if you want raw least-squares amplitudes without background subtraction.
- If you want nonnegativity: either `amp_method="nnls_pgd"` or `nonneg=True`.

### Amplitude solver API: `solve_mca_amplitudes`

Entry point: `mapstorch.opt.solve_mca_amplitudes` (`mapstorch/opt.py:636`)

This solves amplitudes from a fixed global parameter set:

- Builds a **linear basis** of unit-amplitude component spectra using `build_model_basis(...)`.
- Optionally subtracts SNIP background per spectrum (same logic as `fit_spec`).
- Solves `y ≈ A @ basis` in chunks to control memory.

Key parameters:

- `amp_method`:
  - `"ridge"`: stable, uses `(B Bᵀ + λI)^{-1}` with `torch.linalg.solve`
  - `"lstsq"`: least-squares via `torch.linalg.lstsq`
  - `"nnls_pgd"`: NNLS via projected gradient descent (nonnegative by construction)
- `chunk_size`: number of spectra per chunk (flattened pixels)
- `return_log10`: return amplitudes as `log10(A)` (to match model parameterization)

Output:

- `amp_maps`: numpy array shaped `(*map_shape, n_components)` where `map_shape = mca_arr.shape without energy axis`
- `basis_names`: list of component names aligned with the last axis

### GPU acceleration in this workflow

If a CUDA GPU is available, `solve_mca_amplitudes` can accelerate:

- Large matrix multiplications (`y @ basis.T`)
- Linear solves / least-squares (often, but depends on your GPU + PyTorch build)

Even on GPU, `chunk_size` is still useful to avoid running out of VRAM.

---

## Workflow B (Advanced): Per-Pixel / Per-Tile / ROI Parameter Tuning (Batched Autograd)

Sometimes you want to optimize some or all of the `default_fitting_params` per pixel (or per spectrum in a tile/ROI). This is much heavier than Workflow A.

Entry point: `mapstorch.opt.fit_spec_batch` (`mapstorch/opt.py:479`)

### When to use this

Use batched parameter optimization when:

- Energy calibration varies across the scan (e.g. drift) and you want to correct it per-pixel or per-region.
- Peak broadening parameters vary spatially.
- You want a refinement pass after the global + linear solve workflow.

### How it works

`fit_spec_batch` fits a batch of spectra in one optimization run:

- `spectra` can be shaped like `(tile_h, tile_w, n_ch)` or `(n_spec, n_ch)` or any `batch_shape + (n_ch,)`.
- The user specifies which *global* parameters become per-spectrum via `per_spectrum_params`.
- All element/scatter amplitudes are per-spectrum by default (created with shape `batch_shape + (1,)`).
- The model forward pass is fully batched and differentiable (except escape peaks; see below).

Example: per-tile energy calibration refinement

```python
import numpy as np
from mapstorch.opt import fit_spec_batch

tile = np.load("tile.npy")  # shape (tile_h, tile_w, n_ch)
energy_range = (0, tile.shape[-1] - 1)

tensors, spec_fit, bkg, loss_trace = fit_spec_batch(
    tile,
    energy_range,
    per_spectrum_params=["ENERGY_OFFSET", "ENERGY_SLOPE"],
    optimizer="adamw",   # "adam" | "adamw" | "lbfgs"
    n_iter=200,
    use_snip=True,
    device="cuda",       # falls back to CPU if unavailable
    energy_dim=-1,
    progress_bar=True,
)

# Example: tensors["ENERGY_OFFSET"].shape == (tile_h, tile_w, 1)
# spec_fit.shape == tile.shape
```

### Tiling strategy for full-map per-pixel tuning

Optimizing per-pixel parameters for the entire map can be very large in memory.
Instead, process in tiles:

1. choose a tile size (e.g. `64×64` or `128×128`)
2. run `fit_spec_batch(tile, ...)` per tile
3. stitch the resulting per-spectrum parameters and/or amplitudes back into full images

This keeps the batch manageable and improves GPU utilization when available.

### Notes on optimizers

- `adam` / `adamw`: most robust for large batches; good default for per-tile tuning
- `lbfgs`: can converge in fewer steps for small tiles/ROIs, but scales poorly with very large batches

### Escape peaks and gradients

Escape peaks are added under `torch.no_grad()` (same as before), because they depend on a rounded channel shift. This means:

- forward model includes escape peaks when `escape_factor > 0`
- gradients do **not** propagate through the escape-peak term

This matches the prior single-spectrum behavior and keeps the batched model stable.

---

## “GPU-enabled” in practice

All new APIs accept a `device` argument:

- `device="cpu"`: always works
- `device="cuda"`: uses GPU if available; otherwise falls back to CPU

The biggest GPU wins are typically:

- batched autograd optimization (`fit_spec_batch`) when the batch is big enough
- dense linear algebra (matrix multiplications and solves) in `solve_mca_amplitudes`

Whether linear solves benefit from GPU depends on problem size and hardware; the safest approach is to benchmark your typical `n_channels`, `n_components`, and tile sizes.

---

## Developer Notes: What Changed in `map.py` and `opt.py`

This section is intended for developers extending or reviewing the implementation.

### `mapstorch/map.py`

Key goal: make the physics model **batch-safe** without changing its forward behavior for the 1D case.

Changes:

- **Batched broadcasting support**
  - `elastic_peak(...)` and `model_elem_spec(...)` now broadcast per-spectrum scalars over the channel axis when tensors have batch shape.
  - `model_spec(...)` now infers a `batch_shape` from any batched parameter and ensures `ev` broadcasts correctly.

- **Removed tensor-dependent Python branching**
  - `compton_peak(...)` previously used `if params["COMPTON_F_STEP"] > 0:` style branching, which breaks for batched tensors.
  - It now uses `relu`-based scales (`step_scale`, `tail_scale`, etc.) so the same forward behavior is achieved without Python control flow.

- **Escape peak refactor**
  - `escape_peak(...)` now treats the channel axis as the **last** axis and uses `torch.gather` so it works on batched spectra.

- **Binding energy check refactor**
  - Added `_binding_energy_mask(...)` to replace `EnergyStruct.check_binding_energy(...)` when `COHERENT_SCT_ENERGY` is batched.
  - This avoids Python boolean logic and supports vectorized comparisons.

- **New helper for linear basis**
  - Added `build_model_basis(...)` to build unit-amplitude component spectra for fast linear amplitude solving.

Physics preservation:

- For scalar (single-spectrum) parameters, the formulas used in peak/step/tail/scatter modeling are unchanged.
- The changes are primarily about broadcasting and removing Python branching.

### `mapstorch/opt.py`

Key goal: provide two new fitting paths: (1) fast volume amplitudes, (2) batched per-spectrum param tuning.

Changes:

- **New batched parameter construction**
  - Added `create_batched_tensors(...)` to create per-spectrum tensors with shape `batch_shape + (1,)` for clean broadcasting.
  - Note: if you want to pass per-spectrum `fixed_param_vals`, pass them as `torch.Tensor` (array-like non-tensors are not currently supported by the helper).

- **New batched fitter**
  - Added `fit_spec_batch(...)` which runs `fit_spec`-like optimization over an arbitrary batch of spectra, with `per_spectrum_params` selecting which global parameters vary per spectrum.

- **Fast amplitude map solver**
  - Added `solve_mca_amplitudes(...)` which uses `build_model_basis(...)` and solves for per-spectrum amplitudes using ridge / lstsq / NNLS-PGD.

- **End-to-end wrapper**
  - Added `fit_mca_arr(...)` that fits globals on the integrated spectrum and then computes amplitude maps.

- **Optimizer support**
  - Added `optimizer="lbfgs"` option to `fit_spec(...)` and `fit_spec_batch(...)` for small problems where LBFGS is appropriate.

- **Bugfix for batched indexing**
  - `_calculate_loss(...)` now applies `indices` along the **channel axis** (`[..., indices]`) so partial-channel fitting works with batched spectra.

---

## Compatibility: Single 1D Spectrum Behavior

The changes were written to preserve existing behavior for single-spectrum fitting:

- `model_spec(...)` and `model_elem_spec(...)` match the previous outputs for scalar parameters.
- `fit_spec(...)` with Adam/AdamW behaves the same as before; LBFGS is purely additive.

If you want to validate this in your environment, run comparisons of:

- `mapstorch.map.model_spec(...)` old vs new for the same parameter dict
- `mapstorch.opt.fit_spec(...)` old vs new on the same 1D spectrum and seed

---

## Practical Tips

- For full maps, use **Workflow A** to get amplitude maps quickly, then refine only where needed with **Workflow B**.
- If per-spectrum optimization is needed everywhere, prefer **tile-based** `fit_spec_batch` runs over a full-map batch to keep memory and runtime manageable.
- Start with `use_snip=False` when debugging (fewer moving parts), then enable SNIP background subtraction once the workflow is stable.

