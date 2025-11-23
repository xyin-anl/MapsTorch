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

from tqdm import trange
import numpy as np
import torch
from mapstorch.bkg import snip_bkg
from mapstorch.map import model_spec
from mapstorch.default import (
    default_fitting_elems,
    default_fitting_params,
    default_param_vals,
    default_learning_rates,
    default_energy_consts,
    default_elem_info,
)


def init_elem_amps(
    elem,
    spec,
    e_sct,
    e_offset,
    e_slope,
    e_consts=default_energy_consts,
    verbose=False,
    *,
    compton_angle=None,
):
    try:
        this_factor = 8.0
        if elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
            if elem == "COMPTON_AMPLITUDE" and compton_angle is not None:
                theta_rad = np.deg2rad(compton_angle)
                e_energy = e_sct / (1.0 + (e_sct / 511.0) * (1.0 - np.cos(theta_rad)))
            else:
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
        e_sum = np.sum(spec[..., er_min : er_max + 1], axis=-1) / er_size
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
            compton_angle_val = (
                tensors["COMPTON_ANGLE"].item() if "COMPTON_ANGLE" in tensors else None
            )
            init_val = init_elem_amps(
                e,
                spec,
                tensors["COHERENT_SCT_ENERGY"].item(),
                tensors["ENERGY_OFFSET"].item(),
                tensors["ENERGY_SLOPE"].item(),
                verbose=verbose,
                compton_angle=compton_angle_val,
            )
        tensors[e] = torch.tensor(init_val, requires_grad=True, device=device)
        elem_lr = default_learning_rates.get(e, default_learning_rates["default_lr"])
        opt_configs.append(
            {
                "params": tensors[e],
                "lr": elem_lr,
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
    detector_type="Si", # "Si" or "Ge"
    escape_factor=0.0,
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
                    indices,
                    loss_fn,
                    l1_lambda,
                    e_consts,
                    elem_info,
                    detector_type,
                    escape_factor,
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
                                indices,
                                loss_fn,
                                l1_lambda,
                                e_consts,
                                elem_info,
                                detector_type,
                                escape_factor,
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
                indices,
                loss_fn,
                l1_lambda,
                e_consts,
                elem_info,
                detector_type,
                escape_factor,
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
    indices,
    loss_fn,
    l1_lambda,
    e_consts,
    elem_info,
    detector_type,
    escape_factor,
):
    bkg = (
        snip_bkg(
            int_spec_tensor,
            energy_range,
            tensors["ENERGY_OFFSET"],
            tensors["ENERGY_SLOPE"],
            tensors["ENERGY_QUADRATIC"],
            tensors["SNIP_WIDTH"],
            tensors["FWHM_OFFSET"],
            tensors["FWHM_FANOPRIME"],
            device=int_spec_tensor.device,
        )
        if use_snip
        else torch.zeros_like(int_spec_tensor, device=int_spec_tensor.device)
    )
    spec_fit = model_spec(
        tensors,
        energy_range,
        elements_to_fit=elements,
        device=int_spec_tensor.device,
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
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

        # Calculate L1 regularization on amplitudes (A = 10 ** logA)
        l1_reg = torch.tensor(0.0, device=int_spec_tensor.device)
        for elem in elements:
            if elem in tensors and tensors[elem].requires_grad:
                logA = tensors[elem]
                A = torch.pow(torch.tensor(10.0, device=logA.device), logA)
                l1_reg = l1_reg + torch.norm(A, p=1)

        # Apply scaled L1 regularization
        return main_loss + (l1_lambda * scale_factor) * l1_reg, spec_fit, bkg

    return main_loss, spec_fit, bkg
