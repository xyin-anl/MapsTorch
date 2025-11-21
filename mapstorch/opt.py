"""
Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.

Copyright 2024. UChicago Argonne, LLC. This software was produced
under U.S. Government contract DE-AC02-06CH11357 for Argonne National
Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
modified to produce derivative works, such modified software should
be clearly marked, so as not to confuse it with the version available
from ANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

### Initial Author <2024>: Xiangyu Yin

import math
import numpy as np
from tqdm import trange, tqdm

import torch
from torch.cuda.amp import autocast, GradScaler

from mapstorch.util import estimate_gpu_tile_size
from mapstorch.bkg import snip_bkg
from mapstorch.map import model_spec, model_spec_vol
from mapstorch.default import (
    default_fitting_elems,
    default_fitting_params,
    default_param_vals,
    default_learning_rates,
    default_energy_consts,
    default_elem_info,
)


def init_elem_amps(
    elem, spec, e_sct, e_offset, e_slope, e_consts=default_energy_consts, verbose=False
):
    try:
        this_factor = 8.0
        if elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
            e_energy = e_sct
            min_e = e_energy - 0.4
            max_e = e_energy + 0.4
        else:
            e_energy = e_consts[elem][0].energy
            min_e = e_energy - 0.1
            max_e = e_energy + 0.1
        er_min = max(0, round((min_e - e_offset) / e_slope))
        er_max = min(spec.shape[-1] - 1, round((max_e - e_offset) / e_slope))
        er_size = (er_max + 1) - er_min
        e_sum = np.sum(spec[..., er_min:er_max], axis=-1) / er_size
        e_guess = np.log10(np.maximum(e_sum * this_factor + 0.01, 1.0))
        return e_guess
    except Exception:
        if verbose:
            print("Cannot initialize {} amplitude".format(elem))
        return 0.0


def create_tensors(
    elems_to_fit=default_fitting_elems,
    fitting_params=default_fitting_params,
    init_param_vals=default_param_vals,
    fixed_param_vals={},
    map_shape=None,
    tune_params=True,
    init_amp=False,
    spec=None,
    device="cpu",
    verbose=False,
):
    if init_amp:
        assert (
            spec is not None
        ), "Initial amplitude initialization requires the spectrum to be provided."
    tensors, opt_configs = {}, []
    for p, v in default_param_vals.items():
        init_val = init_param_vals.get(p, v)
        if p in fitting_params and tune_params and p not in fixed_param_vals:
            tensors[p] = torch.tensor(
                float(init_val), requires_grad=True, device=device
            )
            opt_configs.append(
                {
                    "params": tensors[p],
                    "lr": default_learning_rates.get(
                        p, default_learning_rates["default_lr"]
                    ),
                }
            )
        else:
            tensors[p] = torch.tensor(
                fixed_param_vals.get(p, init_val), requires_grad=False, device=device
            )
    for e in elems_to_fit:
        if not init_amp:
            if map_shape is not None:
                init_val = np.zeros(map_shape)
            else:
                init_val = 0.0
        else:
            init_val = init_elem_amps(
                e,
                spec,
                tensors["COHERENT_SCT_ENERGY"].item(),
                tensors["ENERGY_OFFSET"].item(),
                tensors["ENERGY_SLOPE"].item(),
                verbose=verbose,
            )
        tensors[e] = torch.tensor(init_val, requires_grad=True, device=device)
        opt_configs.append(
            {
                "params": tensors[e],
                "lr": default_learning_rates.get(
                    p, default_learning_rates["default_lr"]
                ),
            }
        )
    return tensors, opt_configs


def fit_spec(
    int_spec,
    energy_range,
    elements_to_fit=default_fitting_elems,
    fitting_params=default_fitting_params,
    init_param_vals=default_param_vals,
    fixed_param_vals={},
    indices=None,
    tune_params=True,
    init_amp=True,
    use_snip=True,
    use_step=True,
    use_tail=False,
    loss="mse",
    optimizer="adam",
    n_iter=500,
    l1_lambda=0.0,
    progress_bar=True,
    device="cpu",
    status_updator=None,
    verbose=False,
    use_finite_diff=False,
    finite_diff_epsilon=1e-8,
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
):
    assert loss in ["mse", "l1"], "Loss function not supported"
    assert optimizer in ["adam", "adamw"], "Optimizer not supported"

    elements = [
        elem
        for elem in set(
            elements_to_fit + ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
        )
        if (elem in e_consts) or (elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"])
    ]
    params = [param for param in set(fitting_params) if param in default_fitting_params]

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA is not available, using CPU instead")

    torch.cuda.empty_cache()

    tensors, opt_configs = create_tensors(
        elems_to_fit=elements,
        fitting_params=params,
        init_param_vals=init_param_vals,
        fixed_param_vals=fixed_param_vals,
        tune_params=tune_params,
        init_amp=init_amp,
        spec=int_spec,
        device=device,
        verbose=verbose,
    )
    # Check if int_spec is already a tensor
    if isinstance(int_spec, torch.Tensor):
        int_spec_tensor = int_spec[energy_range[0] : energy_range[1] + 1]
        # Ensure it's on the correct device and dtype
        int_spec_tensor = int_spec_tensor.to(device=device, dtype=torch.float32)
    else:
        int_spec_tensor = torch.tensor(
            int_spec[energy_range[0] : energy_range[1] + 1],
            requires_grad=False,
            dtype=torch.float32,
            device=device,
        )

    if loss == "mse":
        loss_fn = torch.nn.MSELoss(reduction="sum")
    elif loss == "l1":
        loss_fn = torch.nn.L1Loss(reduction="sum")
    else:
        raise ValueError("Loss function not supported")

    if optimizer == "adam":
        optimizer = torch.optim.Adam(opt_configs)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(opt_configs)

    loss_trace = []
    spec_fit = None
    bkg = None

    def closure():
        nonlocal spec_fit, bkg  # Declare as nonlocal to modify them inside the closure
        optimizer.zero_grad()
        if use_finite_diff:
            with torch.no_grad():
                loss_val, spec_fit, bkg = _calculate_loss(
                    tensors,
                    int_spec_tensor,
                    energy_range,
                    elements,
                    use_snip,
                    use_step,
                    use_tail,
                    indices,
                    loss_fn,
                    l1_lambda,
                    e_consts,
                    elem_info,
                )
                for param_name, param in tensors.items():
                    if param.requires_grad:
                        param_grad = torch.zeros_like(param)
                        original_param = param.data.clone()
                        for i in range(param.numel()):
                            param.data.flatten()[i] += finite_diff_epsilon
                            perturbed_loss, _, _ = _calculate_loss(
                                tensors,
                                int_spec_tensor,
                                energy_range,
                                elements,
                                use_snip,
                                use_step,
                                use_tail,
                                indices,
                                loss_fn,
                                l1_lambda,
                                e_consts,
                                elem_info,
                            )
                            param_grad.flatten()[i] = (
                                perturbed_loss - loss_val
                            ) / finite_diff_epsilon
                            param.data.flatten()[i] = original_param.flatten()[i]
                        param.grad = param_grad
        else:
            loss_val, spec_fit, bkg = _calculate_loss(
                tensors,
                int_spec_tensor,
                energy_range,
                elements,
                use_snip,
                use_step,
                use_tail,
                indices,
                loss_fn,
                l1_lambda,
                e_consts,
                elem_info,
            )
            loss_val.backward()
        return loss_val

    for _ in trange(n_iter, disable=not progress_bar):
        try:
            loss_val = optimizer.step(closure)
            loss_trace.append(loss_val.item())
        except RuntimeError as e:
            print(f"Optimization step failed: {e}")
            break

        if status_updator is not None:
            status_updator.update()

    return (
        tensors,
        spec_fit.detach().cpu().numpy(),
        bkg.detach().cpu().numpy(),
        loss_trace,
    )


def _calculate_loss(
    tensors,
    int_spec_tensor,
    energy_range,
    elements,
    use_snip,
    use_step,
    use_tail,
    indices,
    loss_fn,
    l1_lambda,
    e_consts,
    elem_info,
):
    bkg = (
        snip_bkg(
            int_spec_tensor,
            energy_range,
            tensors["ENERGY_OFFSET"],
            tensors["ENERGY_SLOPE"],
            tensors["ENERGY_QUADRATIC"],
            tensors["SNIP_WIDTH"],
            device=int_spec_tensor.device,
        )
        if use_snip
        else torch.zeros_like(int_spec_tensor, device=int_spec_tensor.device)
    )
    spec_fit = model_spec(
        tensors,
        energy_range,
        elements_to_fit=elements,
        use_step=use_step,
        use_tail=use_tail,
        device=int_spec_tensor.device,
        e_consts=e_consts,
        elem_info=elem_info,
    )

    # Calculate main loss
    main_loss = (
        loss_fn(spec_fit + bkg, int_spec_tensor)
        if indices is None
        else loss_fn(spec_fit[indices] + bkg[indices], int_spec_tensor[indices])
    )

    # Only calculate L1 regularization if l1_lambda is non-zero
    if l1_lambda > 0:
        # Calculate the scaling factor based on spectrum intensity
        with torch.no_grad():
            scale_factor = (
                loss_fn(bkg, int_spec_tensor)
                if indices is None
                else loss_fn(bkg[indices], int_spec_tensor[indices])
            ) / len(elements)

        # Calculate L1 regularization
        l1_reg = sum(
            torch.norm(tensors[elem], p=1)
            for elem in elements
            if elem in tensors and tensors[elem].requires_grad
        )

        # Apply scaled L1 regularization
        return main_loss + (l1_lambda * scale_factor) * l1_reg, spec_fit, bkg

    return main_loss, spec_fit, bkg


# This function is not meant to be called directly by common users, but rather to be used by other functions
# It is a helper function for fit_spec_vol_amps and fit_spec_vol_params
# Inproper use of this function may lead to out of memory error and other issues
def _fit_spec_vol(
    spec_vol,
    energy_range,
    elements_to_fit=default_fitting_elems,
    fitting_params=default_fitting_params,
    init_param_vals=default_param_vals,
    fixed_param_vals={},
    indices=None,
    tune_params=True,
    init_amp=True,
    use_snip=True,
    use_step=True,
    use_tail=False,
    loss="mse",
    return_loss_trace=False,
    optimizer="sgd",
    use_scheduler=False,
    n_iter=1000,
    progress_bar=True,
    progress_bar_kwargs={},
    device="cuda",
    status_updator=None,
    use_mixed_precision=True,
    verbose=False,
):
    assert loss in ["mse", "l1"], "Loss function not supported"
    assert optimizer in ["adam", "sgd"], "Optimizer not supported"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA is not available, using CPU instead")

    torch.cuda.empty_cache()

    tensors, opt_configs = create_tensors(
        elems_to_fit=elements_to_fit,
        fitting_params=fitting_params,
        init_param_vals=init_param_vals,
        fixed_param_vals=fixed_param_vals,
        map_shape=spec_vol.shape[:-1],
        tune_params=tune_params,
        init_amp=init_amp,
        spec=spec_vol,
        device=device,
        verbose=verbose,
    )
    # Check if spec_vol is already a tensor
    if isinstance(spec_vol, torch.Tensor):
        spec_vol_tensor = spec_vol[..., energy_range[0] : energy_range[1] + 1]
        # Ensure it's on the correct device and dtype
        spec_vol_tensor = spec_vol_tensor.to(device=device, dtype=torch.float32)
    else:
        spec_vol_tensor = torch.tensor(
            spec_vol[..., energy_range[0] : energy_range[1] + 1],
            requires_grad=False,
            dtype=torch.float32,
            device=device,
        )

    if loss == "mse":
        loss_fn = torch.nn.MSELoss(reduction="sum")
    elif loss == "l1":
        loss_fn = torch.nn.L1Loss(reduction="sum")

    if optimizer == "adam":
        optimizer = torch.optim.Adam(opt_configs, lr=1e-6)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(opt_configs, lr=1e-6)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=20, verbose=False
        )

    scaler = GradScaler(enabled=use_mixed_precision)

    loss_trace = []

    # Calculate background outside the loop if tune_params is False
    with torch.no_grad():
        if use_snip:
            bkg = snip_bkg(
                spec_vol_tensor,
                energy_range,
                tensors["ENERGY_OFFSET"],
                tensors["ENERGY_SLOPE"],
                tensors["ENERGY_QUADRATIC"],
                tensors["SNIP_WIDTH"],
                device=device,
            )
        else:
            bkg = torch.zeros_like(spec_vol_tensor, device=device)

    for _ in trange(n_iter, disable=not progress_bar, **progress_bar_kwargs):
        optimizer.zero_grad()

        # Recalculate background if tune_params is True
        if tune_params and use_snip:
            with torch.no_grad():
                bkg = snip_bkg(
                    spec_vol_tensor,
                    energy_range,
                    tensors["ENERGY_OFFSET"],
                    tensors["ENERGY_SLOPE"],
                    tensors["ENERGY_QUADRATIC"],
                    tensors["SNIP_WIDTH"],
                    device=device,
                )

        with autocast(enabled=use_mixed_precision):
            spec_fit = model_spec_vol(
                tensors,
                energy_range,
                spec_vol_shape=spec_vol_tensor.shape,
                elements_to_fit=elements_to_fit,
                use_step=use_step,
                use_tail=use_tail,
                device=device,
            )
            if indices is None:
                loss_val = loss_fn(spec_fit + bkg, spec_vol_tensor)
            else:
                loss_val = loss_fn(
                    spec_fit[..., indices] + bkg[..., indices],
                    spec_vol_tensor[..., indices],
                )

        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()

        if use_scheduler:
            scheduler.step(loss_val)

        loss_trace.append(loss_val.item())

        if status_updator is not None:
            status_updator.update()

    loss_res = loss_trace if return_loss_trace else loss_val.detach().cpu().numpy()

    return (
        tensors,
        spec_fit.detach().cpu().numpy(),
        bkg.detach().cpu().numpy(),
        loss_res,
    )


def fit_spec_vol_amps(
    spec_vol,
    energy_range,
    elements_to_fit,
    param_vals,
    init_amp=True,
    use_snip=True,
    use_step=True,
    use_tail=False,
    tile_size=None,
    n_iter=200,
    progress_bar=True,
    save_fitted_spec=False,
    save_bkg=False,
    save_loss=False,
    status_updator=None,
    e_consts=default_energy_consts,
):
    assert torch.cuda.is_available(), "CUDA is not available"

    elements = [
        elem
        for elem in set(
            elements_to_fit + ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
        )
        if elem in e_consts or (elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"])
    ]
    params = {
        param: param_vals[param]
        for param in param_vals.keys()
        if param in default_param_vals
    }
    progress_bar = progress_bar if status_updator is None else False

    if tile_size is None:
        tile_size = estimate_gpu_tile_size(spec_vol.shape)

    # Calculate total number of tiles
    x_tiles = math.ceil(spec_vol.shape[0] / tile_size)
    y_tiles = math.ceil(spec_vol.shape[1] / tile_size)
    total_tiles = x_tiles * y_tiles
    print(f"Total number of tiles: {total_tiles} = {x_tiles} * {y_tiles}")

    # Prepare the output arrays
    output_shape = list(spec_vol.shape)
    output_shape[-1] = len(elements)
    amp_vol = np.zeros(output_shape, dtype=np.float32)
    output_shape[-1] = energy_range[1] - energy_range[0] + 1
    if save_fitted_spec:
        fitted_spec = np.zeros(output_shape, dtype=np.float32)
    if save_bkg:
        bkg_vol = np.zeros(output_shape, dtype=np.float32)
    if save_loss:
        loss_vol = np.zeros((x_tiles, y_tiles, n_iter), dtype=np.float32)

    # Process tiles sequentially
    with tqdm(
        total=total_tiles * n_iter, disable=not progress_bar, desc="Processing tiles"
    ) as pbar:
        for i in range(0, spec_vol.shape[0], tile_size):
            for j in range(0, spec_vol.shape[1], tile_size):
                tile = spec_vol[i : i + tile_size, j : j + tile_size, :]

                tensors, spec_fit, bkg, loss = _fit_spec_vol(
                    tile,
                    energy_range,
                    elements_to_fit=elements,
                    init_param_vals=params,
                    fixed_param_vals=params,  # Use param_vals as fixed values
                    tune_params=False,
                    init_amp=init_amp,
                    use_snip=use_snip,
                    use_step=use_step,
                    use_tail=use_tail,
                    n_iter=n_iter,
                    device="cuda",
                    progress_bar=False,
                    return_loss_trace=True,
                    status_updator=pbar if status_updator is None else status_updator,
                )

                # Process results
                for k, elem in enumerate(elements):
                    amp_vol[i : i + tile_size, j : j + tile_size, k] = (
                        tensors[elem].detach().cpu().numpy()
                    )
                if save_fitted_spec:
                    fitted_spec[i : i + tile_size, j : j + tile_size, :] = spec_fit
                if save_bkg:
                    bkg_vol[i : i + tile_size, j : j + tile_size, :] = bkg
                if save_loss:
                    loss_vol[i : i + tile_size, j : j + tile_size, :] = loss

                # Clear CUDA cache after each tile
                torch.cuda.empty_cache()

    amp_dict = {elem: amp_vol[..., i] for i, elem in enumerate(elements)}
    tile_info = {
        "x_tiles": x_tiles,
        "y_tiles": y_tiles,
        "tile_size": tile_size,
    }

    if save_fitted_spec or save_bkg or save_loss:
        return (
            amp_dict,
            tile_info,
            fitted_spec if save_fitted_spec else None,
            bkg_vol if save_bkg else None,
            loss_vol if save_loss else None,
        )
    else:
        return amp_dict, tile_info, None, None, None


def fit_spec_vol_params(
    spec_vol,
    energy_range,
    elements_to_fit=default_fitting_elems,
    fitting_params=default_fitting_params,
    init_param_vals=default_param_vals,
    fixed_param_vals={},
    init_amp=True,
    use_snip=True,
    use_step=True,
    use_tail=False,
    tile_size=None,
    max_n_tile_side=5,
    n_iter=500,
    progress_bar=True,
    save_fitted_spec=False,
    save_bkg=False,
    save_loss=False,
    verbose=False,
    status_updator=None,
    e_consts=default_energy_consts,
):
    elements = [
        elem
        for elem in set(
            elements_to_fit + ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
        )
        if elem in e_consts or (elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"])
    ]
    params = [param for param in set(fitting_params) if param in default_fitting_params]
    progress_bar = progress_bar if status_updator is None else False

    min_tile_size = max(
        spec_vol.shape[0] // max_n_tile_side, spec_vol.shape[1] // max_n_tile_side
    )
    tile_size = (
        max(min_tile_size, tile_size) if tile_size is not None else min_tile_size
    )
    print(f"Adjusted tile size: {tile_size}")

    # Calculate total number of tiles
    x_tiles = math.ceil(spec_vol.shape[0] / tile_size)
    y_tiles = math.ceil(spec_vol.shape[1] / tile_size)
    total_tiles = x_tiles * y_tiles
    print(f"Total number of tiles: {total_tiles} = {x_tiles} * {y_tiles}")

    # Prepare the output arrays
    param_vol = np.zeros(
        (*spec_vol.shape[:2], len(params) + len(elements)),
        dtype=np.float32,
    )

    if save_fitted_spec:
        fitted_spec = np.zeros(
            (x_tiles, y_tiles, energy_range[1] - energy_range[0] + 1), dtype=np.float32
        )
    if save_bkg:
        bkg_vol = np.zeros(
            (x_tiles, y_tiles, energy_range[1] - energy_range[0] + 1), dtype=np.float32
        )
    if save_loss:
        loss_vol = np.zeros((x_tiles, y_tiles, n_iter), dtype=np.float32)

    # Process tiles sequentially
    with tqdm(
        total=total_tiles * n_iter, disable=not progress_bar, desc="Processing tiles"
    ) as pbar:
        for i in range(0, spec_vol.shape[0], tile_size):
            for j in range(0, spec_vol.shape[1], tile_size):
                # Integrate spectrum for the tile
                int_spec = np.sum(
                    spec_vol[i : i + tile_size, j : j + tile_size, :], axis=(0, 1)
                )

                tensors, spec_fit, bkg, loss_trace = fit_spec(
                    int_spec,
                    energy_range,
                    elements_to_fit=elements,
                    fitting_params=params,
                    init_param_vals=init_param_vals,
                    fixed_param_vals=fixed_param_vals,
                    tune_params=True,
                    init_amp=init_amp,
                    use_snip=use_snip,
                    use_step=use_step,
                    use_tail=use_tail,
                    n_iter=n_iter,
                    device="cpu",
                    progress_bar=False,
                    verbose=verbose,
                    status_updator=pbar if status_updator is None else status_updator,
                )

                # Process results
                for k, param in enumerate(params + elements):
                    param_vol[i : i + tile_size, j : j + tile_size, k] = (
                        tensors[param].detach().cpu().numpy()
                    )
                if save_fitted_spec:
                    fitted_spec[i // tile_size, j // tile_size, :] = spec_fit
                if save_bkg:
                    bkg_vol[i // tile_size, j // tile_size, :] = bkg
                if save_loss:
                    loss_vol[i // tile_size, j // tile_size, :] = loss_trace

    param_dict = {param: param_vol[..., i] for i, param in enumerate(params + elements)}
    tile_info = {
        "x_tiles": x_tiles,
        "y_tiles": y_tiles,
        "tile_size": tile_size,
    }

    if save_fitted_spec or save_bkg or save_loss:
        return (
            param_dict,
            tile_info,
            fitted_spec if save_fitted_spec else None,
            bkg_vol if save_bkg else None,
            loss_vol if save_loss else None,
        )
    else:
        return param_dict, tile_info, None, None, None
