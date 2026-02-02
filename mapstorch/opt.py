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
from mapstorch.map import model_spec, build_model_basis
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


def create_batched_tensors(
    elems_to_fit=default_fitting_elems,
    fitting_params=default_fitting_params,
    init_param_vals=default_param_vals,
    fixed_param_vals={},
    batch_shape=(),
    tune_params=True,
    init_amp=False,
    spec=None,
    per_spectrum_params=None,
    per_spectrum_elems=True,
    device="cpu",
    verbose=False,
):
    """Create tensors for batched/tiled spectrum fitting.

    Per-spectrum parameters are created with shape `batch_shape + (1,)` so they broadcast
    cleanly over the channel axis during modeling.
    """
    if init_amp:
        assert (
            spec is not None
        ), "Initial amplitude initialization requires the spectra to be provided."

    if batch_shape is None:
        batch_shape = ()
    batch_shape = tuple(int(x) for x in batch_shape)
    has_batch = len(batch_shape) > 0
    param_shape = batch_shape + (1,) if has_batch else ()

    if per_spectrum_params is None:
        per_spectrum_params = []
    per_spectrum_params = set(per_spectrum_params)

    def _as_tensor(value, shape, requires_grad):
        if isinstance(value, torch.Tensor):
            t = value.to(device=device, dtype=torch.float32)
        else:
            t = torch.tensor(float(value), device=device, dtype=torch.float32)

        if shape:
            if tuple(t.shape) == shape:
                return t.detach().clone().requires_grad_(requires_grad)
            if tuple(t.shape) == shape[:-1] and shape[-1] == 1:
                t = t.unsqueeze(-1)
                return t.detach().clone().requires_grad_(requires_grad)
            if t.numel() == 1:
                return torch.full(
                    shape,
                    float(t.item()),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=requires_grad,
                )
            raise ValueError(
                f"Cannot broadcast init value with shape {tuple(t.shape)} to {shape}"
            )

        return t.detach().clone().requires_grad_(requires_grad)

    tensors, opt_configs = {}, []

    for p, v in default_param_vals.items():
        init_val = init_param_vals.get(p, v)
        is_per_spec = p in per_spectrum_params
        this_shape = param_shape if is_per_spec else ()
        requires_grad = p in fitting_params and tune_params and p not in fixed_param_vals
        value = fixed_param_vals.get(p, init_val)
        tensors[p] = _as_tensor(value, this_shape, requires_grad=requires_grad)
        if requires_grad:
            opt_configs.append(
                {
                    "params": tensors[p],
                    "lr": default_learning_rates.get(
                        p, default_learning_rates["default_lr"]
                    ),
                }
            )

    if init_amp and spec is not None:
        spec_np = (
            spec.detach().cpu().numpy()
            if isinstance(spec, torch.Tensor)
            else np.asarray(spec)
        )
    else:
        spec_np = None

    for e in elems_to_fit:
        if not init_amp:
            if per_spectrum_elems and has_batch:
                tensors[e] = torch.zeros(
                    param_shape, device=device, dtype=torch.float32, requires_grad=True
                )
            else:
                tensors[e] = torch.tensor(
                    0.0, device=device, dtype=torch.float32, requires_grad=True
                )
        else:
            compton_angle_val = tensors.get("COMPTON_ANGLE", None)
            compton_angle_val = (
                float(compton_angle_val.detach().mean().item())
                if isinstance(compton_angle_val, torch.Tensor)
                else None
            )
            init_guess = init_elem_amps(
                e,
                spec_np,
                float(tensors["COHERENT_SCT_ENERGY"].detach().mean().item()),
                float(tensors["ENERGY_OFFSET"].detach().mean().item()),
                float(tensors["ENERGY_SLOPE"].detach().mean().item()),
                verbose=verbose,
                compton_angle=compton_angle_val,
            )
            if per_spectrum_elems and has_batch:
                init_guess = np.asarray(init_guess, dtype=float)
                if init_guess.shape != batch_shape:
                    init_guess = np.full(batch_shape, float(init_guess), dtype=float)
                t = torch.as_tensor(init_guess, device=device, dtype=torch.float32).unsqueeze(
                    -1
                )
                tensors[e] = t.detach().clone().requires_grad_(True)
            else:
                tensors[e] = torch.tensor(
                    float(np.asarray(init_guess).mean()),
                    device=device,
                    dtype=torch.float32,
                    requires_grad=True,
                )

        elem_lr = default_learning_rates.get(e, default_learning_rates["default_lr"])
        opt_configs.append({"params": tensors[e], "lr": elem_lr})

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
    lbfgs_lr=1.0,
    lbfgs_max_iter=20,
    lbfgs_history_size=10,
    lbfgs_line_search_fn="strong_wolfe",
):
    assert loss in ["mse", "l1"], "Loss function not supported"
    assert optimizer in ["adam", "adamw", "lbfgs"], "Optimizer not supported"

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
    elif optimizer == "lbfgs":
        lbfgs_params = [cfg["params"] for cfg in opt_configs]
        optimizer = torch.optim.LBFGS(
            lbfgs_params,
            lr=float(lbfgs_lr),
            max_iter=int(lbfgs_max_iter),
            history_size=int(lbfgs_history_size),
            line_search_fn=lbfgs_line_search_fn,
        )

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


def fit_spec_batch(
    spectra,
    energy_range,
    *,
    elements_to_fit=default_fitting_elems,
    fitting_params=default_fitting_params,
    per_spectrum_params=None,
    init_param_vals=default_param_vals,
    fixed_param_vals={},
    tune_params=True,
    init_amp=False,
    use_snip=True,
    loss="mse",
    optimizer="adamw",
    n_iter=200,
    l1_lambda=0.0,
    progress_bar=True,
    device="cpu",
    status_updator=None,
    verbose=False,
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    detector_type="Si",
    escape_factor=0.0,
    energy_dim=-1,
    lbfgs_lr=1.0,
    lbfgs_max_iter=20,
    lbfgs_history_size=10,
    lbfgs_line_search_fn="strong_wolfe",
):
    """Fit a batch/tile/ROI of spectra with optional per-spectrum tuning.

    `spectra` can have any batch shape as long as `energy_dim` identifies the
    energy/channel axis.
    """
    assert loss in ["mse", "l1"], "Loss function not supported"
    assert optimizer in ["adam", "adamw", "lbfgs"], "Optimizer not supported"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        if verbose:
            print("CUDA is not available, using CPU instead")

    device = torch.device(device)

    if isinstance(spectra, torch.Tensor):
        spec = spectra.to(device=device, dtype=torch.float32)
    else:
        spec = torch.as_tensor(spectra, device=device, dtype=torch.float32)

    if spec.ndim < 1:
        raise ValueError("spectra must have at least 1 dimension")
    if energy_dim < 0:
        energy_dim = spec.ndim + int(energy_dim)
    if energy_dim < 0 or energy_dim >= spec.ndim:
        raise ValueError("energy_dim is out of bounds for spectra")
    if energy_dim != spec.ndim - 1:
        spec = spec.movedim(energy_dim, -1)

    batch_shape = spec.shape[:-1]

    elements = [
        elem
        for elem in set(
            elements_to_fit + ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
        )
        if (elem in e_consts) or (elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"])
    ]
    params = [param for param in set(fitting_params) if param in default_fitting_params]

    tensors, opt_configs = create_batched_tensors(
        elems_to_fit=elements,
        fitting_params=params,
        init_param_vals=init_param_vals,
        fixed_param_vals=fixed_param_vals,
        batch_shape=batch_shape,
        tune_params=tune_params,
        init_amp=init_amp,
        spec=spec,
        per_spectrum_params=per_spectrum_params or [],
        per_spectrum_elems=True,
        device=device,
        verbose=verbose,
    )

    ch0, ch1 = int(energy_range[0]), int(energy_range[1])
    if ch1 < ch0:
        raise ValueError("energy_range stop must be >= start")
    if ch0 < 0 or ch1 >= spec.shape[-1]:
        raise ValueError(
            f"energy_range={energy_range} is out of bounds for n_channels={spec.shape[-1]}"
        )

    spec_roi = spec[..., ch0 : ch1 + 1]

    if loss == "mse":
        loss_fn = torch.nn.MSELoss(reduction="sum")
    else:
        loss_fn = torch.nn.L1Loss(reduction="sum")

    if optimizer == "adam":
        optim = torch.optim.Adam(opt_configs)
    elif optimizer == "adamw":
        optim = torch.optim.AdamW(opt_configs)
    else:
        lbfgs_params = [cfg["params"] for cfg in opt_configs]
        optim = torch.optim.LBFGS(
            lbfgs_params,
            lr=float(lbfgs_lr),
            max_iter=int(lbfgs_max_iter),
            history_size=int(lbfgs_history_size),
            line_search_fn=lbfgs_line_search_fn,
        )

    loss_trace = []
    spec_fit = None
    bkg = None

    def closure():
        nonlocal spec_fit, bkg
        optim.zero_grad()
        loss_val, spec_fit, bkg = _calculate_loss(
            tensors,
            spec_roi,
            energy_range,
            elements,
            use_snip,
            indices=None,
            loss_fn=loss_fn,
            l1_lambda=l1_lambda,
            e_consts=e_consts,
            elem_info=elem_info,
            detector_type=detector_type,
            escape_factor=escape_factor,
        )
        loss_val.backward()
        return loss_val

    for _ in trange(n_iter, disable=not progress_bar):
        try:
            loss_val = optim.step(closure)
            loss_trace.append(float(loss_val.item()))
        except RuntimeError as exc:
            if verbose:
                print(f"Optimization step failed: {exc}")
            break
        if status_updator is not None:
            status_updator.update()

    return (
        tensors,
        spec_fit.detach().cpu().numpy(),
        bkg.detach().cpu().numpy(),
        loss_trace,
    )


def solve_mca_amplitudes(
    mca_arr,
    energy_range,
    params,
    *,
    elements_to_fit=default_fitting_elems,
    energy_dim=0,
    use_snip=True,
    amp_method="ridge",
    ridge_lambda=1e-3,
    nonneg=False,
    nnls_max_iter=200,
    nnls_step=None,
    nnls_tol=0.0,
    return_log10=True,
    eps=1e-12,
    chunk_size=8192,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    detector_type="Si",
    escape_factor=0.0,
    progress_bar=True,
    verbose=False,
):
    """Solve per-pixel component amplitudes for an MCA volume given fixed global params.

    Args:
        mca_arr: Spectrum volume as numpy array or torch tensor. Can be any shape as long
            as one axis is the energy/channel axis.
        energy_range: (start, stop) channel indices (inclusive), consistent with `fit_spec`.
        params: Dict of global parameters (e.g. output `tensors` from `fit_spec`).
        elements_to_fit: Elements/components to include; scatter components are added.
        energy_dim: Which axis of mca_arr is energy/channel (default: 0).
        use_snip: Subtract SNIP background per spectrum before solving amplitudes.
        amp_method: "ridge", "lstsq", or "nnls_pgd" (projected gradient NNLS).
        ridge_lambda: L2 regularization for ridge solve (stabilizes nearly-collinear bases).
        nonneg: If True, clamp negative amplitudes to 0 (simple projection).
        nnls_max_iter: Max PGD iterations when amp_method="nnls_pgd".
        nnls_step: Optional fixed PGD step size; if None uses 1 / max_eig(BB^T + λI).
        nnls_tol: Optional relative convergence tolerance (0 disables early stopping).
        return_log10: If True, return log10 amplitudes (matches existing parameterization).
        chunk_size: Number of spectra (flattened pixels) per solve chunk.
    """
    if energy_range is None or len(energy_range) != 2:
        raise ValueError("energy_range must be a (start, stop) tuple of channel indices")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        if verbose:
            print("CUDA is not available, using CPU instead")

    device = torch.device(device)

    # Ensure parameters live on the same device (and are tensors).
    params_device = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_device[k] = v.to(device=device)
        else:
            params_device[k] = torch.tensor(float(v), device=device, dtype=torch.float32)

    if isinstance(mca_arr, torch.Tensor):
        mca = mca_arr.to(device=device, dtype=torch.float32)
    else:
        mca = torch.as_tensor(mca_arr, device=device, dtype=torch.float32)

    if mca.ndim < 2:
        raise ValueError("mca_arr must have at least 2 dimensions")
    if energy_dim < 0:
        energy_dim = mca.ndim + int(energy_dim)
    if energy_dim < 0 or energy_dim >= mca.ndim:
        raise ValueError("energy_dim is out of bounds for mca_arr")
    if energy_dim != mca.ndim - 1:
        mca = mca.movedim(energy_dim, -1)

    ch0, ch1 = int(energy_range[0]), int(energy_range[1])
    if ch1 < ch0:
        raise ValueError("energy_range stop must be >= start")
    if ch0 < 0 or ch1 >= mca.shape[-1]:
        raise ValueError(
            f"energy_range={energy_range} is out of bounds for n_channels={mca.shape[-1]}"
        )

    mca_roi = mca[..., ch0 : ch1 + 1]
    map_shape = mca_roi.shape[:-1]
    n_spec = int(np.prod(map_shape)) if map_shape else 1
    n_ch = mca_roi.shape[-1]

    spectra = mca_roi.reshape(n_spec, n_ch)

    basis, basis_names = build_model_basis(
        params_device,
        energy_range,
        elements_to_fit=elements_to_fit,
        device=device,
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
        verbose=verbose,
    )
    basis = basis.to(device=device, dtype=torch.float32)
    n_comp = basis.shape[0]

    if amp_method not in ("ridge", "lstsq", "nnls_pgd"):
        raise ValueError('amp_method must be "ridge", "lstsq", or "nnls_pgd"')

    if amp_method in ("ridge", "nnls_pgd"):
        reg = float(ridge_lambda)
        eye = torch.eye(n_comp, device=device, dtype=torch.float32)
        gram = basis @ basis.T
        system = gram + reg * eye
        if amp_method == "nnls_pgd":
            if nnls_step is None:
                # system is symmetric PSD; use max eigenvalue as Lipschitz constant.
                with torch.no_grad():
                    L = torch.linalg.eigvalsh(system).max().clamp_min(1e-12)
                    step = (1.0 / L).item()
            else:
                step = float(nnls_step)

    amps = torch.empty((n_spec, n_comp), device=device, dtype=torch.float32)

    n_chunks = (n_spec + int(chunk_size) - 1) // int(chunk_size)
    iterator = trange(n_chunks, disable=not progress_bar, desc="Solving amplitudes")
    for ci in iterator:
        start = ci * int(chunk_size)
        stop = min(n_spec, start + int(chunk_size))
        y = spectra[start:stop]

        if use_snip:
            with torch.no_grad():
                bkg = snip_bkg(
                    y,
                    energy_range,
                    params_device["ENERGY_OFFSET"],
                    params_device["ENERGY_SLOPE"],
                    params_device["ENERGY_QUADRATIC"],
                    params_device["SNIP_WIDTH"],
                    params_device["FWHM_OFFSET"],
                    params_device["FWHM_FANOPRIME"],
                    device=device,
                )
            y = y - bkg

        if amp_method == "ridge":
            ybt = y @ basis.T
            a = torch.linalg.solve(system, ybt.T).T
        elif amp_method == "nnls_pgd":
            ybt = y @ basis.T
            # Warm start from the unconstrained ridge solution when possible.
            try:
                a = torch.linalg.solve(system, ybt.T).T
            except RuntimeError:
                a = torch.zeros((y.shape[0], n_comp), device=device, dtype=torch.float32)
            a = torch.clamp(a, min=0.0)

            if nnls_tol and float(nnls_tol) > 0:
                tol = float(nnls_tol)
                denom = torch.linalg.norm(ybt).clamp_min(1e-12)
            else:
                tol = 0.0
                denom = None

            for _ in range(int(nnls_max_iter)):
                grad = a @ system - ybt
                next_a = torch.clamp(a - step * grad, min=0.0)
                if tol:
                    diff = torch.linalg.norm(next_a - a) / denom
                    if diff.item() < tol:
                        a = next_a
                        break
                a = next_a
        else:
            # Solve basis.T @ x = y.T  =>  y ≈ x.T @ basis
            sol = torch.linalg.lstsq(basis.T, y.T).solution
            a = sol.T

        if nonneg:
            a = torch.clamp(a, min=0.0)

        amps[start:stop] = a

    if return_log10:
        amps = torch.log10(torch.clamp(amps, min=eps))

    amps = amps.reshape(*map_shape, n_comp)
    return amps.detach().cpu().numpy(), basis_names


def fit_mca_arr(
    mca_arr,
    energy_range,
    *,
    elements_to_fit=default_fitting_elems,
    fitting_params=default_fitting_params,
    init_param_vals=default_param_vals,
    fixed_param_vals={},
    energy_dim=0,
    tune_params=True,
    init_amp=True,
    use_snip=True,
    loss="mse",
    optimizer="adamw",
    n_iter=500,
    amp_method="ridge",
    ridge_lambda=1e-3,
    nonneg=False,
    nnls_max_iter=200,
    nnls_step=None,
    nnls_tol=0.0,
    return_log10=True,
    chunk_size=8192,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    detector_type="Si",
    escape_factor=0.0,
    progress_bar=True,
    verbose=False,
):
    """Convenience wrapper: fit global params on integrated spectrum, then solve amplitude maps."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        if verbose:
            print("CUDA is not available, using CPU instead")

    device = torch.device(device)

    if isinstance(mca_arr, torch.Tensor):
        mca = mca_arr.to(device=device, dtype=torch.float32)
    else:
        mca = torch.as_tensor(mca_arr, device=device, dtype=torch.float32)
    if mca.ndim < 2:
        raise ValueError("mca_arr must have at least 2 dimensions")
    if energy_dim < 0:
        energy_dim = mca.ndim + int(energy_dim)
    if energy_dim != mca.ndim - 1:
        mca = mca.movedim(energy_dim, -1)

    int_spec = torch.sum(mca, dim=tuple(range(mca.ndim - 1)))
    fitted_params, _, _, loss_trace = fit_spec(
        int_spec,
        energy_range,
        elements_to_fit=elements_to_fit,
        fitting_params=fitting_params,
        init_param_vals=init_param_vals,
        fixed_param_vals=fixed_param_vals,
        tune_params=tune_params,
        init_amp=init_amp,
        use_snip=use_snip,
        loss=loss,
        optimizer=optimizer,
        n_iter=n_iter,
        progress_bar=progress_bar,
        device=str(device),
        verbose=verbose,
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
    )

    amp_maps, amp_names = solve_mca_amplitudes(
        mca,
        energy_range,
        fitted_params,
        elements_to_fit=elements_to_fit,
        energy_dim=-1,
        use_snip=use_snip,
        amp_method=amp_method,
        ridge_lambda=ridge_lambda,
        nonneg=nonneg,
        nnls_max_iter=nnls_max_iter,
        nnls_step=nnls_step,
        nnls_tol=nnls_tol,
        return_log10=return_log10,
        chunk_size=chunk_size,
        device=str(device),
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
        progress_bar=progress_bar,
        verbose=verbose,
    )

    return fitted_params, amp_maps, amp_names, loss_trace


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
        else loss_fn(
            spec_fit[..., indices] + bkg[..., indices], int_spec_tensor[..., indices]
        )
    )

    # Only calculate L1 regularization if l1_lambda is non-zero
    if l1_lambda > 0:
        # Calculate the scaling factor based on spectrum intensity
        with torch.no_grad():
            scale_factor = (
                loss_fn(bkg, int_spec_tensor)
                if indices is None
                else loss_fn(bkg[..., indices], int_spec_tensor[..., indices])
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
