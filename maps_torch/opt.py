'''
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
'''

### Initial Author <2024>: Xiangyu Yin

import torch
import numpy as np
from tqdm import trange
from tqdm.contrib.concurrent import thread_map
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

from maps_torch.util import split_into_tiles, optimize_parallelization
from maps_torch.bkg import snip_bkg
from maps_torch.map import model_spec, model_spec_vol
from maps_torch.default import (
    default_fitting_elems,
    default_fitting_params,
    default_param_vals,
    default_learning_rates,
    default_energy_consts,
)

def init_elem_amps(elem, spec, e_sct, e_offset, e_slope, e_consts=default_energy_consts):
    try:
        this_factor = 8.0
        if elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
            e_energy = e_sct
            min_e = e_energy - 0.4
            max_e = e_energy + 0.4
        else:
            e_energy = (
                e_consts[elem][0].energy
            )
            min_e = e_energy - 0.1
            max_e = e_energy + 0.1
        er_min = max(0, round((min_e - e_offset) / e_slope))
        er_max = min(spec.shape[-1] - 1, round((max_e - e_offset) / e_slope))
        er_size = (er_max + 1) - er_min
        e_sum = np.sum(spec[..., er_min:er_max], axis=-1) / er_size
        e_guess = np.log10(np.maximum(e_sum * this_factor + 0.01, 1.0))
        return e_guess
    except Exception:
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
):
    if init_amp:
        assert (
            spec is not None
        ), "Initial amplitude initialization requires the spectrum to be provided."
    tensors, opt_configs = {}, []
    for p, v in default_param_vals.items():
        init_val = init_param_vals.get(p, v)
        if p in fitting_params and tune_params and p not in fixed_param_vals:
            tensors[p] = torch.tensor(init_val, requires_grad=True, device=device)
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
    progress_bar=True,
    device="cpu",
    status_updator=None,
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
        tune_params=tune_params,
        init_amp=init_amp,
        spec=int_spec,
        device=device,
    )
    int_spec_tensor = torch.tensor(
        int_spec[energy_range[0] : energy_range[1] + 1],
        requires_grad=False,
        dtype=torch.float32,
        device=device,
    )

    if loss == "mse":
        loss = torch.nn.MSELoss(reduction="sum")
    elif loss == "l1":
        loss = torch.nn.L1Loss(reduction="sum")
    else:
        raise ValueError("Loss function not supported")

    if optimizer == "adam":
        optimizer = torch.optim.Adam(opt_configs, lr=1e-6)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(opt_configs, lr=1e-6)
    else:
        raise ValueError("Optimizer not supported")

    loss_trace = []
    for _ in trange(n_iter, disable=not progress_bar):
        optimizer.zero_grad()
        bkg = (
            snip_bkg(
                int_spec_tensor,
                energy_range,
                tensors["ENERGY_OFFSET"],
                tensors["ENERGY_SLOPE"],
                tensors["ENERGY_QUADRATIC"],
                tensors["SNIP_WIDTH"],
                device=device,
            )
            if use_snip
            else torch.zeros_like(int_spec_tensor, device=device)
        )
        spec_fit = model_spec(
            tensors,
            energy_range,
            elements_to_fit=elements_to_fit,
            use_step=use_step,
            use_tail=use_tail,
            device=device,
        )
        loss_val = (
            loss(spec_fit + bkg, int_spec_tensor)
            if indices is None
            else loss(spec_fit[indices] + bkg[indices], int_spec_tensor[indices])
        )
        loss_trace.append(loss_val.item())
        loss_val.backward()
        optimizer.step()
        if status_updator is not None:
            status_updator.update()

    return tensors, spec_fit.detach().cpu().numpy(), bkg.detach().cpu().numpy(), loss_trace


def fit_spec_vol(
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
    optimizer="adam",
    use_scheduler=False,
    n_iter=1000,
    progress_bar=True,
    progress_bar_kwargs={},
    device="cuda",
    status_updator=None,
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
    )
    spec_vol_tensor = torch.tensor(
        spec_vol[..., energy_range[0] : energy_range[1] + 1],
        requires_grad=False,
        dtype=torch.float32,
        device=device,
    )

    if loss == "mse":
        loss = torch.nn.MSELoss(reduction="none")
    elif loss == "l1":
        loss = torch.nn.L1Loss(reduction="none")
    else:
        raise ValueError("Loss function not supported")

    if optimizer == "adam":
        optimizer = torch.optim.Adam(opt_configs, lr=1e-6)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(opt_configs, lr=1e-6)
    else:
        raise ValueError("Optimizer not supported")

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.8, patience=20, verbose=False
        )

    loss_trace = []
    for _ in trange(n_iter, disable=not progress_bar, **progress_bar_kwargs):
        optimizer.zero_grad()
        bkg = (
            snip_bkg(
                spec_vol_tensor,
                energy_range,
                tensors["ENERGY_OFFSET"],
                tensors["ENERGY_SLOPE"],
                tensors["ENERGY_QUADRATIC"],
                tensors["SNIP_WIDTH"],
                device=device,
            )
            if use_snip
            else torch.zeros_like(spec_vol_tensor, device=device)
        )
        spec_fit = model_spec_vol(
            tensors,
            energy_range,
            spec_vol_shape=spec_vol_tensor.shape,
            elements_to_fit=elements_to_fit,
            use_step=use_step,
            use_tail=use_tail,
            device=device,
        )
        loss_vol = (
            loss(spec_fit + bkg, spec_vol_tensor)
            if indices is None
            else loss(
                spec_fit[..., indices] + bkg[..., indices],
                spec_vol_tensor[..., indices]
            )
        )
        loss_val = loss_vol.sum()
        loss_trace.append(loss_val.item())
        loss_val.backward()
        optimizer.step()
        if use_scheduler:
            scheduler.step(loss_val)
        if status_updator is not None:
            status_updator.update()
        loss_res = loss_trace if return_loss_trace else loss_vol.detach().cpu().numpy()

    return tensors, spec_fit.detach().cpu().numpy(), bkg.detach().cpu().numpy(), loss_res


def process_tile(tile_data, energy_range, elements_to_fit, fitting_params, init_param_vals, 
                 fixed_param_vals, indices, tune_params, init_amp, use_snip, use_step, 
                 use_tail, loss, optimizer, use_scheduler, n_iter, device, progress_bar):
    """Process a single tile."""
    tile, (i, j) = tile_data
    tensors, spec_fit, bkg, loss_vol = fit_spec_vol(
        tile, energy_range, elements_to_fit=elements_to_fit, fitting_params=fitting_params, 
        init_param_vals=init_param_vals, fixed_param_vals=fixed_param_vals, indices=indices,
        tune_params=tune_params, init_amp=init_amp, use_snip=use_snip, use_step=use_step,
        use_tail=use_tail, loss=loss, optimizer=optimizer, use_scheduler=use_scheduler,
        n_iter=n_iter, progress_bar=progress_bar, device=device,
        progress_bar_kwargs={'desc': f"Tile ({i}, {j})", 'miniters':20}
    )
    return (i, j), tensors, spec_fit, bkg, loss_vol


def fit_spec_vol_parallel(
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
    optimizer="adam",
    use_scheduler=False,
    n_iter=1000,
    device="cuda",
    tile_size=None,
    n_workers=None,
    return_tiles=False,
    progress_bar=True,
    inner_progress_bar=False,
):
    if tile_size is None or n_workers is None:
        tile_size, n_workers = optimize_parallelization(spec_vol.shape, device)
    
    # Split the spec_vol into tiles
    tiles = split_into_tiles(spec_vol, tile_size)

    # Create a partial function with all the fixed arguments
    process_tile_partial = partial(
        process_tile, 
        energy_range=energy_range, 
        elements_to_fit=elements_to_fit,
        fitting_params=fitting_params,
        init_param_vals=init_param_vals,
        fixed_param_vals=fixed_param_vals,
        indices=indices,
        tune_params=tune_params,
        init_amp=init_amp,
        use_snip=use_snip,
        use_step=use_step,
        use_tail=use_tail,
        loss=loss,
        optimizer=optimizer,
        use_scheduler=use_scheduler,
        n_iter=n_iter,
        device=device,
        progress_bar=inner_progress_bar
    )

    # Process tiles in parallel
    total_tiles = len(tiles)
    if progress_bar:
        results = thread_map(process_tile_partial, tiles, max_workers=n_workers, 
                             desc="Processing tiles", total=total_tiles)
    else:
        if device == "cuda":
            # Use ThreadPoolExecutor for GPU to avoid forking issues
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(process_tile_partial, tiles))
        else:
            # Use ProcessPoolExecutor for CPU to leverage multiple cores
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results = list(executor.map(process_tile_partial, tiles))
    if return_tiles:
        return results
    else:
        # Reconstruct the full output arrays
        h, w, _ = spec_vol.shape
        tensor_keys = results[0][1].keys()
        full_spec_fit = np.zeros((h, w, energy_range[1] - energy_range[0] + 1))
        full_bkg = np.zeros_like(full_spec_fit)
        full_loss_vol = np.zeros_like(full_spec_fit)
        tensor_maps = {}
        for key in tensor_keys:
            if key in fitting_params and tune_params and key not in fixed_param_vals:
                tensor_maps[key] = np.zeros((h, w))
            if key in elements_to_fit:
                tensor_maps[key] = np.zeros((h, w))

        for (i, j), tensors, spec_fit, bkg, loss_vol in results:
            th, tw = spec_fit.shape[:2]
            full_spec_fit[i:i+th, j:j+tw] = spec_fit
            full_bkg[i:i+th, j:j+tw] = bkg
            full_loss_vol[i:i+th, j:j+tw] = loss_vol
            for key in tensor_keys:
                if key in fitting_params and tune_params and key not in fixed_param_vals:
                    tensor_maps[key][i:i+th, j:j+tw] = tensors[key].item()
                if key in elements_to_fit:
                    tensor_maps[key][i:i+th, j:j+tw] = tensors[key].detach().cpu().numpy()

        return tensor_maps, full_spec_fit, full_bkg, full_loss_vol