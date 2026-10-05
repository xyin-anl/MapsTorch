#!/usr/bin/env python
"""
Comprehensive test and benchmark script for MapsTorch spectra volume fitting.

Tests:
1. fit_mca_arr - end-to-end workflow (global params + amplitude maps)
2. solve_mca_amplitudes - fast linear amplitude solver (ridge, lstsq, nnls_pgd)
3. fit_spec_batch - batched per-spectrum parameter tuning
4. GPU vs CPU performance comparison
"""

import os
import time
import numpy as np
import torch
import h5py

# Set CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from mapstorch.opt import fit_mca_arr, solve_mca_amplitudes, fit_spec_batch, fit_spec
from mapstorch.map import build_model_basis
from mapstorch.default import default_fitting_elems, default_fitting_params


def load_mca_data(h5_path, subsample=None):
    """Load MCA data from HDF5 file."""
    with h5py.File(h5_path, "r") as f:
        mca_arr = f["MAPS/Spectra/mca_arr"][:]
        energy_calib = f["MAPS/Spectra/Energy_Calibration"][:]
        int_spec = f["MAPS/Spectra/Integrated_Spectra/Spectra"][:]

    print(f"Loaded mca_arr shape: {mca_arr.shape}")
    print(f"Energy calibration: {energy_calib}")

    if subsample is not None:
        # Subsample spatial dimensions for faster testing
        mca_arr = mca_arr[:, ::subsample, ::subsample]
        print(f"Subsampled mca_arr shape: {mca_arr.shape}")

    return mca_arr, energy_calib, int_spec


def benchmark_fit_mca_arr(mca_arr, energy_range, device, elements, n_iter=200):
    """Benchmark fit_mca_arr (global params + amplitude maps)."""
    print(f"\n{'='*60}")
    print(f"Benchmarking fit_mca_arr on {device.upper()}")
    print(f"{'='*60}")

    # Warm-up GPU if needed
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    fitted_params, amp_maps, amp_names, loss_trace = fit_mca_arr(
        mca_arr,
        energy_range,
        elements_to_fit=elements,
        fitting_params=list(default_fitting_params),
        energy_dim=0,
        use_snip=True,
        optimizer="adamw",
        n_iter=n_iter,
        amp_method="ridge",
        ridge_lambda=1e-3,
        return_log10=True,
        device=device,
        chunk_size=8192,
        progress_bar=True,
        verbose=True,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  amp_maps shape: {amp_maps.shape}")
    print(f"  Components: {amp_names}")
    print(f"  Final loss: {loss_trace[-1]:.2e}" if loss_trace else "  No loss trace")

    # Print some fitted parameters
    print("\n  Fitted parameters (sample):")
    for key in ["ENERGY_OFFSET", "ENERGY_SLOPE", "COHERENT_SCT_ENERGY", "FWHM_OFFSET"]:
        if key in fitted_params:
            val = fitted_params[key]
            if isinstance(val, torch.Tensor):
                val = val.item()
            print(f"    {key}: {val:.6f}")

    return fitted_params, amp_maps, amp_names, elapsed


def benchmark_solve_mca_amplitudes(mca_arr, energy_range, params, elements, device, method="ridge"):
    """Benchmark solve_mca_amplitudes with different methods."""
    print(f"\n{'-'*60}")
    print(f"Benchmarking solve_mca_amplitudes (method={method}) on {device.upper()}")
    print(f"{'-'*60}")

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    amp_maps, amp_names = solve_mca_amplitudes(
        mca_arr,
        energy_range,
        params,
        elements_to_fit=elements,
        energy_dim=0,
        use_snip=True,
        amp_method=method,
        ridge_lambda=1e-3,
        nonneg=(method != "nnls_pgd"),  # nnls_pgd is already nonneg
        nnls_max_iter=100,
        return_log10=True,
        chunk_size=8192,
        device=device,
        progress_bar=True,
        verbose=True,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  amp_maps shape: {amp_maps.shape}")
    print(f"  amp_maps stats: min={amp_maps.min():.2f}, max={amp_maps.max():.2f}, mean={amp_maps.mean():.2f}")

    return amp_maps, amp_names, elapsed


def benchmark_fit_spec_batch(mca_arr, energy_range, elements, device, tile_size=64, n_iter=100):
    """Benchmark fit_spec_batch for per-spectrum parameter tuning."""
    print(f"\n{'-'*60}")
    print(f"Benchmarking fit_spec_batch (tile_size={tile_size}x{tile_size}) on {device.upper()}")
    print(f"{'-'*60}")

    # Extract a tile from the MCA array
    n_ch, h, w = mca_arr.shape
    tile_h = min(tile_size, h)
    tile_w = min(tile_size, w)

    # Get center tile
    start_h = (h - tile_h) // 2
    start_w = (w - tile_w) // 2
    tile = mca_arr[:, start_h:start_h+tile_h, start_w:start_w+tile_w]

    # Move channel axis to last
    tile = np.moveaxis(tile, 0, -1)
    print(f"Tile shape: {tile.shape}")

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    tensors, spec_fit, bkg, loss_trace = fit_spec_batch(
        tile,
        energy_range,
        elements_to_fit=elements,
        fitting_params=list(default_fitting_params),
        per_spectrum_params=["ENERGY_OFFSET", "ENERGY_SLOPE"],  # Per-pixel energy calibration
        use_snip=True,
        optimizer="adamw",
        n_iter=n_iter,
        device=device,
        energy_dim=-1,
        progress_bar=True,
        verbose=True,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  spec_fit shape: {spec_fit.shape}")
    print(f"  Final loss: {loss_trace[-1]:.2e}" if loss_trace else "  No loss trace")

    # Check per-spectrum parameters
    for key in ["ENERGY_OFFSET", "ENERGY_SLOPE"]:
        if key in tensors:
            t = tensors[key]
            if isinstance(t, torch.Tensor) and t.ndim > 0:
                print(f"  {key} shape: {tuple(t.shape)}, range: [{t.min().item():.6f}, {t.max().item():.6f}]")

    return tensors, spec_fit, elapsed


def benchmark_build_model_basis(params, energy_range, elements, device):
    """Benchmark build_model_basis."""
    print(f"\n{'-'*60}")
    print(f"Benchmarking build_model_basis on {device.upper()}")
    print(f"{'-'*60}")

    # Convert params to device
    params_device = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_device[k] = v.to(device=device)
        else:
            params_device[k] = torch.tensor(float(v), device=device, dtype=torch.float32)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    basis, basis_names = build_model_basis(
        params_device,
        energy_range,
        elements_to_fit=elements,
        device=device,
        verbose=True,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed*1000:.2f}ms")
    print(f"  Basis shape: {tuple(basis.shape)}")
    print(f"  Components: {basis_names}")

    return basis, basis_names, elapsed


def benchmark_single_spectrum_fit(int_spec, energy_range, elements, device, n_iter=500):
    """Test single spectrum fitting (baseline comparison)."""
    print(f"\n{'-'*60}")
    print(f"Testing single spectrum fit on {device.upper()}")
    print(f"{'-'*60}")

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    tensors, spec_fit, bkg, loss_trace = fit_spec(
        int_spec,
        energy_range,
        elements_to_fit=elements,
        fitting_params=list(default_fitting_params),
        use_snip=True,
        optimizer="adamw",
        n_iter=n_iter,
        device=device,
        progress_bar=True,
        verbose=True,
    )

    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  spec_fit shape: {spec_fit.shape}")
    print(f"  Final loss: {loss_trace[-1]:.2e}" if loss_trace else "  No loss trace")

    return tensors, spec_fit, elapsed


def test_single_spectrum_fit():
    """Recover known amplitudes through the public model and fitting pipeline."""
    import mapstorch
    from mapstorch import default, map as model

    energy_range = (0, 1400)
    components = ["Fe", "COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
    expected = np.array([1.0, 0.6, 0.3])  # Base-ten logarithmic amplitudes.
    params = {
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in default.default_param_vals.items()
    }
    for name, value in zip(components, expected):
        params[name] = torch.tensor(value, dtype=torch.float32)
    spectrum = model.model_spec(params, energy_range, elements_to_fit=["Fe"]).numpy()
    assert spectrum.shape == (1401,)
    assert np.isfinite(spectrum).all() and spectrum.max() > 0

    fitted, prediction, background, losses = fit_spec(
        spectrum,
        energy_range,
        elements_to_fit=["Fe"],
        fitting_params=[],
        tune_params=False,
        init_amp=False,
        use_snip=False,
        optimizer="lbfgs",
        n_iter=8,
        progress_bar=False,
        device="cpu",
    )
    assert len(losses) == 8 and np.isfinite(losses).all()
    assert losses[-1] < losses[0]
    np.testing.assert_allclose(
        [fitted[name].item() for name in components], expected, rtol=0, atol=2e-4
    )
    np.testing.assert_allclose(prediction, spectrum, rtol=5e-4, atol=1e-5)
    np.testing.assert_array_equal(background, np.zeros_like(spectrum))


def main():
    print("="*60)
    print("MapsTorch Spectra Volume Fitting - Test & Benchmark")
    print("="*60)

    # Check GPU availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")

    # Load data
    h5_path = "examples/2xfm_0034.mda.h5"
    print(f"\nLoading data from: {h5_path}")

    # Use subsample for faster testing (set to None for full data)
    mca_arr, energy_calib, int_spec = load_mca_data(h5_path, subsample=4)

    # Define fitting parameters
    energy_range = (0, mca_arr.shape[0] - 1)  # Full channel range
    elements = ["Ca", "Fe", "Cu", "Zn", "Cl", "S", "K", "Ti", "Mn", "Ni"]

    print(f"\nEnergy range: {energy_range}")
    print(f"Elements to fit: {elements}")

    results = {}

    # Test 1: Single spectrum fit (baseline)
    print("\n" + "="*60)
    print("TEST 1: Single Spectrum Fit (Baseline)")
    print("="*60)

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            print(f"Skipping {device} - not available")
            continue

        params, spec_fit, elapsed = benchmark_single_spectrum_fit(
            int_spec, energy_range, elements, device, n_iter=300
        )
        results[f"single_spec_{device}"] = {"params": params, "time": elapsed}

    # Use CPU fitted params for subsequent tests
    base_params = results.get("single_spec_cuda", results.get("single_spec_cpu"))["params"]

    # Test 2: build_model_basis
    print("\n" + "="*60)
    print("TEST 2: Build Model Basis")
    print("="*60)

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue

        basis, basis_names, elapsed = benchmark_build_model_basis(
            base_params, energy_range, elements, device
        )
        results[f"basis_{device}"] = {"time": elapsed}

    # Test 3: solve_mca_amplitudes with different methods
    print("\n" + "="*60)
    print("TEST 3: Solve MCA Amplitudes (Linear Solver)")
    print("="*60)

    for method in ["ridge", "lstsq", "nnls_pgd"]:
        for device in ["cpu", "cuda"]:
            if device == "cuda" and not torch.cuda.is_available():
                continue

            amp_maps, amp_names, elapsed = benchmark_solve_mca_amplitudes(
                mca_arr, energy_range, base_params, elements, device, method=method
            )
            results[f"solve_amps_{method}_{device}"] = {"time": elapsed, "shape": amp_maps.shape}

    # Test 4: fit_mca_arr (end-to-end)
    print("\n" + "="*60)
    print("TEST 4: fit_mca_arr (End-to-End)")
    print("="*60)

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue

        params, amp_maps, amp_names, elapsed = benchmark_fit_mca_arr(
            mca_arr, energy_range, device, elements, n_iter=200
        )
        results[f"fit_mca_arr_{device}"] = {"time": elapsed, "shape": amp_maps.shape}

    # Test 5: fit_spec_batch (per-spectrum tuning)
    print("\n" + "="*60)
    print("TEST 5: fit_spec_batch (Per-Spectrum Tuning)")
    print("="*60)

    for device in ["cpu", "cuda"]:
        if device == "cuda" and not torch.cuda.is_available():
            continue

        tensors, spec_fit, elapsed = benchmark_fit_spec_batch(
            mca_arr, energy_range, elements, device, tile_size=32, n_iter=100
        )
        results[f"fit_batch_{device}"] = {"time": elapsed}

    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)

    print(f"\n{'Test':<40} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10}")
    print("-"*74)

    test_pairs = [
        ("single_spec", "Single Spectrum Fit"),
        ("basis", "Build Model Basis"),
        ("solve_amps_ridge", "Solve Amplitudes (Ridge)"),
        ("solve_amps_lstsq", "Solve Amplitudes (LSTSQ)"),
        ("solve_amps_nnls_pgd", "Solve Amplitudes (NNLS-PGD)"),
        ("fit_mca_arr", "fit_mca_arr (End-to-End)"),
        ("fit_batch", "fit_spec_batch (32x32 tile)"),
    ]

    for key, name in test_pairs:
        cpu_key = f"{key}_cpu"
        gpu_key = f"{key}_cuda"

        cpu_time = results.get(cpu_key, {}).get("time", float("nan"))
        gpu_time = results.get(gpu_key, {}).get("time", float("nan"))

        if not np.isnan(cpu_time) and not np.isnan(gpu_time) and gpu_time > 0:
            speedup = cpu_time / gpu_time
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"

        cpu_str = f"{cpu_time:.2f}" if not np.isnan(cpu_time) else "N/A"
        gpu_str = f"{gpu_time:.2f}" if not np.isnan(gpu_time) else "N/A"

        print(f"{name:<40} {cpu_str:<12} {gpu_str:<12} {speedup_str:<10}")

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
