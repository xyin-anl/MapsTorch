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
from mapstorch.losses import compute_loss
from mapstorch.metrics import (
    build_element_roi_masks,
    build_roi_weights,
    mmd_rbf_multiscale,
    shift_blur_1d,
    to_distribution,
    wasserstein1_shift,
)
from mapstorch.map import model_spec, build_model_basis
from mapstorch.default import (
    default_fitting_elems,
    default_fitting_params,
    default_param_vals,
    default_learning_rates,
    default_energy_consts,
    default_elem_info,
)


def _resolve_bool_param(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, torch.Tensor):
        value = value.detach()
        if value.numel() != 1:
            value = value.reshape(-1)[0]
        return bool(value.item() >= 0.5)
    return bool(value)


def _ordered_unique(items):
    return list(dict.fromkeys(items))


# The squared-error loss is summed in counts², so on a bright spectrum gradients pass
# 1e19 (ENERGY_QUADRATIC first) and Adam's running average of squared gradients
# overflows float32, freezing that parameter for the rest of the fit. CUDA reaches it
# first because its kernel squares before it scales. Adam and AdamW therefore step on
# the loss scaled so the observed squared counts sum to ADAM_OBJECTIVE_POWER, which
# keeps gradients far from both the overflow and Adam's eps; their steps do not
# depend on the scale otherwise, and reported losses stay unscaled.
ADAM_OBJECTIVE_POWER = 1e10


def _adam_objective_scale(obs, loss, optimizer):
    if loss != "mse" or optimizer not in ("adam", "adamw"):
        return 1.0
    power = float(torch.sum(obs.detach().double() ** 2))
    return ADAM_OBJECTIVE_POWER / power if power > 0 else 1.0


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
    loss_params=None,
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
    escape_model="constant",
    lbfgs_lr=1.0,
    lbfgs_max_iter=20,
    lbfgs_history_size=10,
    lbfgs_line_search_fn="strong_wolfe",
    project=None,
):
    """Fit one spectrum's settings and amplitudes.

    `project`, when given, is called as project(tensors) without gradients after
    every optimizer step; it may change the tensors in place, for example to keep
    refined settings within physical limits. L-BFGS evaluates points inside a
    step that the projection has not seen; the returned settings are projected.
    """
    assert loss in ["mse", "l1", "poisson_nll", "poisson_deviance", "chi2"], (
        "Loss function not supported"
    )
    assert optimizer in ["adam", "adamw", "lbfgs"], "Optimizer not supported"

    elements = [
        elem
        for elem in _ordered_unique(
            elements_to_fit + ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
        )
        if (elem in e_consts) or (elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"])
    ]
    params = [
        param for param in _ordered_unique(fitting_params) if param in default_fitting_params
    ]

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
        int_spec_full_tensor = int_spec.to(device=device, dtype=torch.float32)
        int_spec_tensor = int_spec_full_tensor[energy_range[0] : energy_range[1] + 1]
        # Ensure it's on the correct device and dtype
    else:
        int_spec_full_tensor = torch.tensor(
            int_spec,
            requires_grad=False,
            dtype=torch.float32,
            device=device,
        )
        int_spec_tensor = int_spec_full_tensor[energy_range[0] : energy_range[1] + 1]

    if loss_params is None:
        loss_params = {}

    objective_scale = _adam_objective_scale(int_spec_tensor, loss, optimizer)

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
                    int_spec_full_tensor,
                    energy_range,
                    elements,
                    use_snip,
                    indices,
                    loss,
                    loss_params,
                    l1_lambda,
                    e_consts,
                    elem_info,
                    detector_type,
                    escape_factor,
                    escape_model,
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
                                int_spec_full_tensor,
                                energy_range,
                                elements,
                                use_snip,
                                indices,
                                loss,
                                loss_params,
                                l1_lambda,
                                e_consts,
                                elem_info,
                                detector_type,
                                escape_factor,
                                escape_model,
                            )
                            param_grad.flatten()[i] = (
                                perturbed_loss - loss_val
                            ) / finite_diff_epsilon
                            param.data.flatten()[i] = original_param.flatten()[i]
                        param.grad = param_grad * objective_scale
        else:
            loss_val, spec_fit, bkg = _calculate_loss(
                tensors,
                int_spec_tensor,
                int_spec_full_tensor,
                energy_range,
                elements,
                use_snip,
                indices,
                loss,
                loss_params,
                l1_lambda,
                e_consts,
                elem_info,
                detector_type,
                escape_factor,
                escape_model,
            )
            (loss_val * objective_scale).backward()
        return loss_val

    for _ in trange(n_iter, disable=not progress_bar):
        failed = None
        try:
            loss_val = optimizer.step(closure)
            loss_trace.append(loss_val.item())
        except RuntimeError as e:
            failed = e
        # a failed step may still have moved the settings (an L-BFGS line search)
        if project is not None:
            with torch.no_grad():
                project(tensors)
        if failed is not None:
            print(f"Optimization step failed: {failed}")
            break

        if status_updator is not None:
            status_updator.update()

    # with a projection the last step's model predates the final projected settings
    if spec_fit is None or bkg is None or project is not None:
        with torch.no_grad():
            _, spec_fit, bkg = _calculate_loss(
                tensors,
                int_spec_tensor,
                int_spec_full_tensor,
                energy_range,
                elements,
                use_snip,
                indices,
                loss,
                loss_params,
                l1_lambda,
                e_consts,
                elem_info,
                detector_type,
                escape_factor,
                escape_model,
            )

    return (tensors, spec_fit.detach().cpu().numpy(), bkg.detach().cpu().numpy(), loss_trace)


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
    loss_params=None,
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
    escape_model="constant",
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
    assert loss in ["mse", "l1", "poisson_nll", "poisson_deviance", "chi2"], (
        "Loss function not supported"
    )
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
        for elem in _ordered_unique(
            elements_to_fit + ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]
        )
        if (elem in e_consts) or (elem in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"])
    ]
    params = [
        param for param in _ordered_unique(fitting_params) if param in default_fitting_params
    ]

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

    if loss_params is None:
        loss_params = {}

    objective_scale = _adam_objective_scale(spec_roi, loss, optimizer)

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
            spec,
            energy_range,
            elements,
            use_snip,
            indices=None,
            loss=loss,
            loss_params=loss_params,
            l1_lambda=l1_lambda,
            e_consts=e_consts,
            elem_info=elem_info,
            detector_type=detector_type,
            escape_factor=escape_factor,
            escape_model=escape_model,
        )
        (loss_val * objective_scale).backward()
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

    if spec_fit is None or bkg is None:
        with torch.no_grad():
            _, spec_fit, bkg = _calculate_loss(
                tensors,
                spec_roi,
                spec,
                energy_range,
                elements,
                use_snip,
                indices=None,
                loss=loss,
                loss_params=loss_params,
                l1_lambda=l1_lambda,
                e_consts=e_consts,
                elem_info=elem_info,
                detector_type=detector_type,
                escape_factor=escape_factor,
                escape_model=escape_model,
            )

    return (tensors, spec_fit.detach().cpu().numpy(), bkg.detach().cpu().numpy(), loss_trace)


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
    escape_model="constant",
    pixel_background=None,
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
        pixel_background: Optional (component, shape) from `tied_pixel_background`: each
            spectrum's background is `shape` (one value per fit channel) times that scatter
            component's amplitude, fitted with the lines on the raw counts; overrides
            `use_snip`. For pixels of a few hundred counts, where SNIP of one spectrum finds
            almost no background and the continuum under each line is counted as the line.
        amp_method: "ridge", "lstsq", or "nnls_pgd" (projected gradient NNLS).
        ridge_lambda: L2 regularization for ridge solve (stabilizes nearly-collinear bases).
        nonneg: If True, clamp negative amplitudes to 0 (simple projection).
        nnls_max_iter: Max PGD iterations when amp_method="nnls_pgd".
            Unregularized NNLS keeps XRF-Maps' initial value of 0.5 in numerically
            unconstrained directions and warm-starts the identifiable amplitudes.
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

    map_shape = mca.shape[:-1]
    n_spec = int(np.prod(map_shape)) if map_shape else 1
    spectra_full = mca.reshape(n_spec, mca.shape[-1])
    spectra = spectra_full[:, ch0 : ch1 + 1]
    n_ch = spectra.shape[-1]

    basis, basis_names = build_model_basis(
        params_device,
        energy_range,
        elements_to_fit=elements_to_fit,
        device=device,
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
        escape_model=escape_model,
        verbose=verbose,
    )
    basis = basis.to(device=device, dtype=torch.float32)
    n_comp = basis.shape[0]
    if pixel_background is not None:
        carrier, shape = pixel_background
        if carrier not in basis_names:
            raise ValueError(f"pixel_background carrier {carrier!r} is not a fitted component")
        shape = torch.as_tensor(np.asarray(shape), device=device, dtype=torch.float32)
        if shape.shape != (n_ch,):
            raise ValueError(
                f"pixel_background shape has {tuple(shape.shape)} values for {n_ch} fit channels"
            )
        basis = basis.clone()
        basis[list(basis_names).index(carrier)] += shape
        use_snip = False  # each spectrum's background is fitted with its lines

    if amp_method not in ("ridge", "lstsq", "nnls_pgd"):
        raise ValueError('amp_method must be "ridge", "lstsq", or "nnls_pgd"')

    if amp_method in ("ridge", "nnls_pgd"):
        reg = float(ridge_lambda)
        eye = torch.eye(n_comp, device=device, dtype=torch.float32)
        gram = basis @ basis.T
        system = gram + reg * eye
        if amp_method == "nnls_pgd":
            if reg == 0.0:
                # Keep the finite XRF-Maps start in the numerical null space.
                # The small K x K pseudoinverse is taken in double precision on
                # the CPU (some devices have no float64), so eigenvector roundoff
                # does not contaminate tiny columns; rank reflects the float32 input.
                system_double = system.detach().to("cpu", torch.float64)
                system_pinv = torch.linalg.pinv(
                    system_double,
                    rtol=n_comp * torch.finfo(system.dtype).eps,
                    hermitian=True,
                ).to(device=device, dtype=torch.float32)
                initial_ybt = (0.5 * system_double.sum(dim=0)).to(
                    device=device, dtype=torch.float32
                )
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
            y = y - _snip_fit_range(spectra_full[start:stop], energy_range, params_device, device)

        if amp_method == "ridge":
            ybt = y @ basis.T
            a = torch.linalg.solve(system, ybt.T).T
        elif amp_method == "nnls_pgd":
            ybt = y @ basis.T
            if reg == 0.0:
                a = 0.5 + (ybt - initial_ybt) @ system_pinv
            else:
                # Keep the ridge warm start when regularization is requested.
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


def _snip_fit_range(spectra_full, energy_range, params_device, device):
    """SNIP background of each full-length spectrum (N, channels) over the fit range."""
    use_full = _resolve_bool_param(params_device.get("SNIP_USE_FULL_SPECTRUM", 1.0), default=True)
    lo, hi = int(energy_range[0]), int(energy_range[1])
    with torch.no_grad():
        return snip_bkg(
            spectra_full if use_full else spectra_full[:, lo : hi + 1],
            energy_range,
            params_device["ENERGY_OFFSET"],
            params_device["ENERGY_SLOPE"],
            params_device["ENERGY_QUADRATIC"],
            params_device["SNIP_WIDTH"],
            params_device["FWHM_OFFSET"],
            params_device["FWHM_FANOPRIME"],
            device=device,
            use_single_log=params_device["SNIP_USE_SINGLE_LOG"],
            use_fwhm_width=params_device["SNIP_USE_FWHM_WIDTH"],
            use_full_spectrum=use_full,
            fwhm_fano_coeff=params_device.get(
                "FWHM_FANO_COEFF", default_param_vals["FWHM_FANO_COEFF"]
            ),
        )


def _params_on(params, device):
    return {
        k: (v.to(device=device) if isinstance(v, torch.Tensor)
            else torch.tensor(float(v), device=device, dtype=torch.float32))
        for k, v in params.items()
    }


# Scatter components that may carry a tied background, in order of preference.
TIED_BACKGROUND_CARRIERS = ("COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE")
# How much worse (Poisson deviance below 0.9 of the beam energy) the tied background may fit a
# scan's sampled spectra than SNIP of each spectrum before the scan keeps SNIP of each spectrum.
# The tie is for faint pixels, where SNIP of one pixel finds no background; at many counts per
# pixel SNIP finds it, and one shape for the whole scan becomes the larger error. Synthetic scan,
# two continuum shapes, Zn summed over pixels over planted, tie against SNIP of each pixel, and
# the deviance ratio: 300 counts per pixel 0.99 against 1.90, ratio 0.77; 50,000 counts 0.94
# against 1.10, 1.14; 100,000 counts 0.94 against 1.04, 1.60. On measured scans the ratio stays
# near or below 1 where the tie helps.
TIED_BACKGROUND_GUARD = 1.15
# How much more the carrier may hold, summed over the sampled spectra, fitted with the tie than with
# SNIP of each spectrum. With the tie the carrier's component is its peak plus the background, and
# the fit is least squares on raw counts, so a peak that no line in the list explains is covered
# by scaling that component up; the background under every line grows with it. Synthetic film with
# weak scatter and La L lines missing from the list: carrier 1.39 times, Zn 0.15 of planted, Ca
# 0.89, while the deviance ratio (0.69) alone would keep the tie. On 55 development scans from
# seven instruments, 47 with a tie were at or below 1.15 and the next six at 1.35-8.8: the threshold
# sits in that gap.
TIED_CARRIER_GUARD = 1.2


def _peak_inside(row, margin=3):
    return row.max() > 0 and margin <= int(row.argmax()) < len(row) - margin


def tied_pixel_background(
    spectrum,
    energy_range,
    params,
    *,
    elements_to_fit=default_fitting_elems,
    amp_method="nnls_pgd",
    ridge_lambda=0.0,
    nnls_max_iter=2000,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    detector_type="Si",
    escape_factor=0.0,
    escape_model="constant",
):
    """A scan's background for its per-pixel fits, from its summed spectrum, for
    `solve_mca_amplitudes(..., pixel_background=...)`.

    A pixel of a few hundred counts spread over about a thousand channels has most channels
    empty: SNIP of that one spectrum finds next to no background, and the continuum under each
    line is counted as the line (on low-concentration tissue scans, roughly a tenth of a trace
    element's map; on a synthetic scan at 300 counts per pixel, Zn 1.8 times planted). The
    continuum comes mostly from scattered photons, so it is tied to a scatter peak: the summed
    spectrum's SNIP background per unit amplitude of that peak in the summed fit is added to the
    peak's component, and each pixel's background is that shape times its own scatter
    amplitude, with no new free amplitude per pixel.

    The carrier is the Compton peak, or the elastic peak when no Compton peak with counts has
    its maximum inside the fit range (elastic scatter also holds diffraction from crystalline
    grains, which changes pixel to pixel without the continuum, so it is the second choice).
    One shape holds for the whole scan, so check it with `guard_tied_background` before use.

    Returns (component name, shape over the fit range), or None when the summed spectrum has
    no background or no scatter peak can carry it.
    """
    spec = torch.as_tensor(np.asarray(spectrum, dtype=np.float32)).reshape(1, -1)
    common = dict(
        elements_to_fit=elements_to_fit,
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
        escape_model=escape_model,
    )
    amps, names = solve_mca_amplitudes(
        spec, energy_range, params, energy_dim=-1, use_snip=True, amp_method=amp_method,
        ridge_lambda=ridge_lambda, nnls_max_iter=nnls_max_iter, return_log10=False,
        device=device, progress_bar=False, **common,
    )
    amps, names = amps[0].astype(np.float64), list(names)
    params_device = _params_on(params, torch.device("cpu"))
    with torch.no_grad():
        basis, _ = build_model_basis(params_device, energy_range, device="cpu", **common)
        background = _snip_fit_range(spec, energy_range, params_device, "cpu")[0]
    basis = basis.numpy().astype(np.float64)
    background = background.numpy().astype(np.float64)
    peaks = {
        name: amps[k] * basis[k].sum()
        for k, name in enumerate(names)
        if name in TIED_BACKGROUND_CARRIERS and _peak_inside(basis[k])
    }
    carrier = next((n for n in TIED_BACKGROUND_CARRIERS if peaks.get(n, 0.0) > 0), None)
    if carrier is None or not background.sum() > 0:
        return None
    return carrier, background / amps[names.index(carrier)]


def _poisson_deviance(counts, mu):
    mu = np.maximum(mu, 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(counts > 0, counts * np.log(np.where(counts > 0, counts, 1.0) / mu), 0.0)
    return 2.0 * (term - (counts - mu))


def guard_tied_background(
    pixel_background,
    spectra,
    energy_range,
    params,
    *,
    elements_to_fit=default_fitting_elems,
    amp_method="nnls_pgd",
    ridge_lambda=0.0,
    nnls_max_iter=200,
    threshold=TIED_BACKGROUND_GUARD,
    carrier_threshold=TIED_CARRIER_GUARD,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    detector_type="Si",
    escape_factor=0.0,
    escape_model="constant",
):
    """Keep a tied background (`tied_pixel_background`) only where it fits the scan's own
    spectra about as well as SNIP of each spectrum and its carrier still holds only its peak.

    `spectra` (N, channels) holds raw counts of a sample of the scan's pixels, such as a few
    thousand from evenly spaced rows. Both models are fitted to them with the same solver and
    scored by Poisson deviance over the fit range below 0.9 of the beam energy (above it the
    scatter peaks dominate), and the carrier's amplitude summed over them is compared both ways.
    Returns the tie or None (keep SNIP of each spectrum), the tie's deviance over SNIP's, and that
    carrier ratio. A tie that fits worse than `threshold` does not describe that scan's continuum,
    as on bright scans where SNIP of each spectrum already finds it; a carrier above
    `carrier_threshold` has taken in counts that are not its peak's, such as a peak missing from
    the element list, and would subtract that much more background under every line. A comparison
    that cannot be scored (no channel below that energy, or a non-finite score) keeps SNIP, with
    ratios None; a sample with no finite spectrum is an error.
    """
    if pixel_background is None:
        return None, None, None
    spectra = np.asarray(spectra, dtype=np.float32)
    spectra = spectra[np.isfinite(spectra).all(axis=1)]
    if not len(spectra):
        raise ValueError("guard_tied_background needs at least one finite spectrum to score")
    lo, hi = int(energy_range[0]), int(energy_range[1])
    value = {
        k: float(np.asarray(params[k].detach().cpu() if isinstance(params[k], torch.Tensor)
                            else params[k]).reshape(-1)[0])
        for k in ("ENERGY_OFFSET", "ENERGY_SLOPE", "ENERGY_QUADRATIC", "COHERENT_SCT_ENERGY")
    }
    ch = np.arange(lo, hi + 1, dtype=np.float64)
    energy = value["ENERGY_OFFSET"] + value["ENERGY_SLOPE"] * ch + value["ENERGY_QUADRATIC"] * ch**2
    window = energy < 0.9 * value["COHERENT_SCT_ENERGY"]
    if not window.any():
        return None, None, None
    common = dict(
        elements_to_fit=elements_to_fit,
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
        escape_model=escape_model,
    )
    solve = dict(
        energy_dim=-1, amp_method=amp_method, ridge_lambda=ridge_lambda,
        nnls_max_iter=nnls_max_iter, return_log10=False, device=device, progress_bar=False,
    )
    a, names = solve_mca_amplitudes(spectra, energy_range, params, use_snip=True, **solve, **common)
    b, _ = solve_mca_amplitudes(
        spectra, energy_range, params, pixel_background=pixel_background, **solve, **common
    )
    params_device = _params_on(params, torch.device("cpu"))
    with torch.no_grad():
        basis, _ = build_model_basis(params_device, energy_range, device="cpu", **common)
        snip = _snip_fit_range(torch.as_tensor(spectra), energy_range, params_device, "cpu")
    basis = basis.numpy().astype(np.float64)
    k = list(names).index(pixel_background[0])
    carried = basis.copy()
    carried[k] += np.asarray(pixel_background[1])
    y = spectra[:, lo : hi + 1].astype(np.float64)
    today = _poisson_deviance(y, a.astype(np.float64) @ basis + snip.numpy())[:, window].sum()
    tied = _poisson_deviance(y, b.astype(np.float64) @ carried)[:, window].sum()
    carrier_today, carrier_tied = float(a[:, k].sum()), float(b[:, k].sum())
    if not all(np.isfinite([today, tied, carrier_today, carrier_tied])):
        return None, None, None
    ratio, carrier = _ratio(tied, today), _ratio(carrier_tied, carrier_today)
    keep = ratio <= threshold and carrier <= carrier_threshold
    return (pixel_background if keep else None), ratio, carrier


def _ratio(num, den):
    return float(num / den) if den > 0 else (float("inf") if num > 0 else 1.0)


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
    loss_params=None,
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
    escape_model="constant",
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
        loss_params=loss_params,
        optimizer=optimizer,
        n_iter=n_iter,
        progress_bar=progress_bar,
        device=str(device),
        verbose=verbose,
        e_consts=e_consts,
        elem_info=elem_info,
        detector_type=detector_type,
        escape_factor=escape_factor,
        escape_model=escape_model,
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
        escape_model=escape_model,
        progress_bar=progress_bar,
        verbose=verbose,
    )

    return fitted_params, amp_maps, amp_names, loss_trace


def _calculate_loss(
    tensors,
    int_spec_tensor,
    full_spec_tensor,
    energy_range,
    elements,
    use_snip,
    indices,
    loss,
    loss_params,
    l1_lambda,
    e_consts,
    elem_info,
    detector_type,
    escape_factor,
    escape_model="constant",
):
    use_full_spectrum_snip = _resolve_bool_param(
        tensors.get("SNIP_USE_FULL_SPECTRUM", 1.0), default=True
    )
    bkg = (
        snip_bkg(
            full_spec_tensor if use_full_spectrum_snip else int_spec_tensor,
            energy_range,
            tensors["ENERGY_OFFSET"],
            tensors["ENERGY_SLOPE"],
            tensors["ENERGY_QUADRATIC"],
            tensors["SNIP_WIDTH"],
            tensors["FWHM_OFFSET"],
            tensors["FWHM_FANOPRIME"],
            device=int_spec_tensor.device,
            use_single_log=tensors["SNIP_USE_SINGLE_LOG"],
            use_fwhm_width=tensors["SNIP_USE_FWHM_WIDTH"],
            use_full_spectrum=use_full_spectrum_snip,
            fwhm_fano_coeff=tensors.get(
                "FWHM_FANO_COEFF", default_param_vals["FWHM_FANO_COEFF"]
            ),
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
        escape_model=escape_model,
    )

    if loss_params is None:
        loss_params = {}
    channel_weights = loss_params.get("channel_weights", None)
    if channel_weights is not None:
        channel_weights = torch.as_tensor(
            channel_weights, device=int_spec_tensor.device, dtype=int_spec_tensor.dtype
        )

    total_loss = compute_loss(
        spec_fit + bkg,
        int_spec_tensor,
        loss=loss,
        indices=indices,
        weights=channel_weights,
        eps=float(loss_params.get("eps", 1e-6)),
        sigma2_floor=float(loss_params.get("sigma2_floor", 0.0)),
        chi2_variant=str(loss_params.get("chi2_variant", "pearson")),
        reduction=str(loss_params.get("reduction", "sum")),
    )

    # Optional ROI-focused auxiliary loss (to emphasize weak peaks / chemistry-relevant regions).
    roi_lambda = float(loss_params.get("roi_lambda", 0.0))
    if roi_lambda != 0.0:
        roi_profile = str(loss_params.get("roi_profile", "box"))
        roi_n_sigma = float(loss_params.get("roi_n_sigma", 3.0))
        roi_on_peaknet = bool(loss_params.get("roi_on_peaknet", False))
        roi_reduction = str(loss_params.get("roi_reduction", "weighted_mean"))

        roi_weights = build_roi_weights(
            tensors,
            energy_range,
            elements_to_fit=elements,
            roi_n_sigma=roi_n_sigma,
            profile=roi_profile,  # type: ignore[arg-type]
            include_scatter=bool(loss_params.get("roi_include_scatter", True)),
            allowed_lines_only=bool(loss_params.get("roi_allowed_lines_only", True)),
            e_consts=e_consts,
            elem_info=elem_info,
            eps=float(loss_params.get("eps", 1e-6)),
        )

        if roi_on_peaknet:
            pred_roi = spec_fit
            obs_roi = torch.relu(int_spec_tensor - bkg)
        else:
            pred_roi = spec_fit + bkg
            obs_roi = int_spec_tensor

        roi_loss = compute_loss(
            pred_roi,
            obs_roi,
            loss=loss,
            indices=indices,
            weights=roi_weights,
            eps=float(loss_params.get("eps", 1e-6)),
            sigma2_floor=float(loss_params.get("sigma2_floor", 0.0)),
            chi2_variant=str(loss_params.get("chi2_variant", "pearson")),
            reduction=roi_reduction,  # type: ignore[arg-type]
        )
        total_loss = total_loss + roi_lambda * roi_loss

    # Optional distribution/shape loss (shift-tolerant peak-structure matching).
    shape_lambda = float(loss_params.get("shape_lambda", 0.0))
    if shape_lambda != 0.0:
        shape_metric = str(loss_params.get("shape_metric", "w1"))
        shape_on_peaknet = bool(loss_params.get("shape_on_peaknet", True))
        shape_reduction = str(loss_params.get("shape_reduction", "mean"))
        eps_shape = float(loss_params.get("shape_eps", loss_params.get("eps", 1e-6)))

        if shape_on_peaknet:
            obs_mass = torch.relu(int_spec_tensor - bkg)
            pred_mass = torch.relu(spec_fit)
        else:
            obs_mass = torch.relu(int_spec_tensor)
            pred_mass = torch.relu(spec_fit + bkg)

        p = to_distribution(obs_mass, eps=eps_shape)
        q = to_distribution(pred_mass, eps=eps_shape)

        shift_mode = str(loss_params.get("shape_shift_mode", "none"))
        max_shift = int(loss_params.get("shape_shift_max", 0))
        blur_sigma = float(loss_params.get("shape_shift_sigma", 1.0))
        softmin_tau = float(loss_params.get("shape_shift_tau", 0.05))

        if shape_metric == "w1":
            shape_loss = wasserstein1_shift(
                p,
                q,
                dx=float(loss_params.get("shape_dx", 1.0)),
                shift_mode=shift_mode,  # type: ignore[arg-type]
                max_shift=max_shift,
                blur_sigma=blur_sigma,
                softmin_tau=softmin_tau,
                reduction=shape_reduction,  # type: ignore[arg-type]
                eps=eps_shape,
            )
        elif shape_metric == "mmd_rbf":
            if shift_mode == "blur" and max_shift > 0:
                q = to_distribution(
                    shift_blur_1d(q, max_shift=max_shift, sigma=blur_sigma, eps=eps_shape),
                    eps=eps_shape,
                )
            sigmas = loss_params.get("shape_mmd_sigmas", [1.0, 2.0, 4.0, 8.0])
            shape_loss = mmd_rbf_multiscale(
                p,
                q,
                sigmas=[float(s) for s in sigmas],
                truncate=float(loss_params.get("shape_mmd_truncate", 3.0)),
                reduction=shape_reduction,  # type: ignore[arg-type]
            )
        else:
            raise ValueError(f"Unsupported shape_metric={shape_metric!r}")

        total_loss = total_loss + shape_lambda * shape_loss

    # Optional peak-area loss (matches integrated net areas in per-element ROIs).
    area_lambda = float(loss_params.get("area_lambda", 0.0))
    if area_lambda != 0.0:
        area_roi_n_sigma = float(loss_params.get("area_roi_n_sigma", 3.0))
        area_mode = str(loss_params.get("area_mode", "relative_l1"))
        area_on_peaknet = bool(loss_params.get("area_on_peaknet", True))
        area_include_scatter = bool(loss_params.get("area_include_scatter", False))
        area_allowed_lines_only = bool(loss_params.get("area_allowed_lines_only", True))
        eps_area = float(loss_params.get("area_eps", loss_params.get("eps", 1e-6)))

        if area_on_peaknet:
            obs_net = torch.relu(int_spec_tensor - bkg)
            pred_net = torch.relu(spec_fit)
        else:
            obs_net = torch.relu(int_spec_tensor)
            pred_net = torch.relu(spec_fit + bkg)

        masks = build_element_roi_masks(
            tensors,
            energy_range,
            elements_to_fit=elements,
            roi_n_sigma=area_roi_n_sigma,
            include_scatter=area_include_scatter,
            allowed_lines_only=area_allowed_lines_only,
            e_consts=e_consts,
            elem_info=elem_info,
            eps=eps_area,
        )

        per_elem_losses = []
        for name, mask in masks.items():
            if not area_include_scatter and name in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
                continue
            area_obs = torch.sum(obs_net * mask, dim=-1)
            area_pred = torch.sum(pred_net * mask, dim=-1)
            if area_mode == "relative_l1":
                denom = torch.abs(area_obs).clamp_min(eps_area)
                per = torch.abs(area_pred - area_obs) / denom
            elif area_mode == "l2":
                per = (area_pred - area_obs) ** 2
            else:
                raise ValueError(f"Unsupported area_mode={area_mode!r}")
            per_elem_losses.append(per)

        if per_elem_losses:
            area_loss = torch.mean(torch.stack(per_elem_losses, dim=0))
            total_loss = total_loss + area_lambda * area_loss

    # Only calculate L1 regularization if l1_lambda is non-zero
    if l1_lambda > 0:
        # Calculate the scaling factor based on spectrum intensity
        with torch.no_grad():
            ref = int_spec_tensor if indices is None else int_spec_tensor[..., indices]
            scale_factor = torch.sum(ref) / max(len(elements), 1)

        # Calculate L1 regularization on amplitudes (A = 10 ** logA)
        l1_reg = torch.tensor(0.0, device=int_spec_tensor.device)
        for elem in elements:
            if elem in tensors and tensors[elem].requires_grad:
                logA = tensors[elem]
                A = torch.pow(torch.tensor(10.0, device=logA.device), logA)
                l1_reg = l1_reg + torch.norm(A, p=1)

        # Apply scaled L1 regularization
        return total_loss + (l1_lambda * scale_factor) * l1_reg, spec_fit, bkg

    return total_loss, spec_fit, bkg
