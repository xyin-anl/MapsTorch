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

import torch
from mapstorch.constant import (
    M_PI,
    SQRT_2XPI,
    M_SQRT2,
    binding_threshold_for_line,
    parse_pileup_descriptor,
)
from mapstorch.default import (
    default_fitting_elems,
    default_energy_consts,
    default_elem_info,
    default_param_vals,
)

_MODEL_SPEC_FAILURE_CACHE: set[tuple[str, str, str]] = set()


def peak(gain, sigma, delta_energy):
    return gain / (sigma * SQRT_2XPI) * torch.exp(-0.5 * (delta_energy / sigma) ** 2)


def step(gain, sigma, delta_energy, peak_E):
    return gain / 2.0 / peak_E * torch.erfc(delta_energy / (M_SQRT2 * sigma))


def _positive_clamp(value, minimum):
    if isinstance(value, torch.Tensor):
        return torch.clamp(value, min=float(minimum))
    return max(float(value), float(minimum))


def _resolution_fano_coeff(params):
    coeff = params.get("FWHM_FANO_COEFF", default_param_vals["FWHM_FANO_COEFF"])
    if isinstance(coeff, torch.Tensor):
        return torch.clamp(coeff, min=0.0)
    return max(float(coeff), 0.0)


def tail(gain, sigma, delta_energy, gamma):
    sigma = _positive_clamp(sigma, 1e-6)
    gamma = _positive_clamp(gamma, 0.1)
    val1 = torch.erfc(delta_energy / (M_SQRT2 * sigma) + (1.0 / (gamma * M_SQRT2)))
    val2 = torch.exp(torch.clamp(delta_energy / (gamma * sigma), max=0.0)) * val1
    d_e = torch.where(delta_energy < 0, val2, val1)
    exp_term = 0.5 / gamma**2
    if not isinstance(exp_term, torch.Tensor):
        exp_term = torch.tensor(
            exp_term, device=delta_energy.device, dtype=delta_energy.dtype
        )
    counts = gain * torch.exp(exp_term) / (2.0 * gamma * sigma) * d_e
    return torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)


def elastic_peak(params, ev, gain):
    fano_coeff = _resolution_fano_coeff(params)
    sigma = torch.sqrt(
        (params["FWHM_OFFSET"] / 2.3548) ** 2
        + params["COHERENT_SCT_ENERGY"] * fano_coeff * params["FWHM_FANOPRIME"]
    )
    coherent = params["COHERENT_SCT_ENERGY"]
    amp = 10 ** params["COHERENT_SCT_AMPLITUDE"]

    if isinstance(coherent, torch.Tensor) and coherent.ndim == ev.ndim - 1:
        coherent = coherent.unsqueeze(-1)
    if isinstance(sigma, torch.Tensor) and sigma.ndim == ev.ndim - 1:
        sigma = sigma.unsqueeze(-1)
    if isinstance(gain, torch.Tensor) and gain.ndim == ev.ndim - 1:
        gain = gain.unsqueeze(-1)
    if isinstance(amp, torch.Tensor) and amp.ndim == ev.ndim - 1:
        amp = amp.unsqueeze(-1)

    delta_energy = ev - coherent
    counts = amp * peak(gain, sigma, delta_energy)
    return counts


def compton_peak(params, ev, gain, use_step=True, use_tail=False):
    compton_E = params["COHERENT_SCT_ENERGY"] / (
        1
        + (params["COHERENT_SCT_ENERGY"] / 511)
        * (1 - torch.cos(params["COMPTON_ANGLE"] * 2 * M_PI / 360.0))
    )
    fano_coeff = _resolution_fano_coeff(params)
    sigma = torch.sqrt(
        (params["FWHM_OFFSET"] / 2.3548) ** 2
        + compton_E * fano_coeff * params["FWHM_FANOPRIME"]
    )
    # Broadcast per-spectrum scalars over the channel axis if needed.
    if isinstance(compton_E, torch.Tensor) and compton_E.ndim == ev.ndim - 1:
        compton_E = compton_E.unsqueeze(-1)
    if isinstance(sigma, torch.Tensor) and sigma.ndim == ev.ndim - 1:
        sigma = sigma.unsqueeze(-1)
    if isinstance(gain, torch.Tensor) and gain.ndim == ev.ndim - 1:
        gain = gain.unsqueeze(-1)

    delta_energy = ev - compton_E

    def _relu_param(name: str):
        val = params.get(name, 0.0)
        if isinstance(val, torch.Tensor):
            return torch.relu(val)
        return max(float(val), 0.0)

    step_scale = _relu_param("COMPTON_F_STEP") if use_step else 0.0
    tail_scale = _relu_param("COMPTON_F_TAIL") if use_tail else 0.0
    hi_tail_scale = _relu_param("COMPTON_HI_F_TAIL") if use_tail else 0.0

    if isinstance(step_scale, torch.Tensor) and step_scale.ndim == ev.ndim - 1:
        step_scale = step_scale.unsqueeze(-1)
    if isinstance(tail_scale, torch.Tensor) and tail_scale.ndim == ev.ndim - 1:
        tail_scale = tail_scale.unsqueeze(-1)
    if isinstance(hi_tail_scale, torch.Tensor) and hi_tail_scale.ndim == ev.ndim - 1:
        hi_tail_scale = hi_tail_scale.unsqueeze(-1)

    norm = 1.0 + step_scale + tail_scale + hi_tail_scale
    faktor = 10 ** params["COMPTON_AMPLITUDE"] / norm

    counts = faktor * peak(gain, sigma * params["COMPTON_FWHM_CORR"], delta_energy)
    if use_step:
        counts = counts + faktor * step_scale * step(gain, sigma, delta_energy, compton_E)
    if use_tail:
        counts = counts + faktor * tail_scale * tail(
            gain, sigma, delta_energy, params["COMPTON_GAMMA"]
        )
        counts = counts + faktor * hi_tail_scale * tail(
            gain, sigma, -delta_energy, params["COMPTON_HI_GAMMA"]
        )
    return counts


def escape_peak(spectra, ev, escape_factor, device, detector_type="Si"):
    if detector_type == "Si":
        # Approximate Si Kα1 energy (keV)
        detector_escape_E = 1.73998
    elif detector_type == "Ge":
        # Approximate Ge Kα1 energy (keV)
        detector_escape_E = 9.886
    else:
        detector_escape_E = 1.73998

    # Treat the energy/channel axis as the last dimension; supports batched spectra.
    escape_spec = torch.zeros_like(spectra, device=device)

    # Support batched spectra by assuming the energy/channel axis is the last dimension.
    n_ch = ev.shape[-1]
    if n_ch < 2:
        return escape_spec

    delta_e = torch.mean(ev[..., 1:] - ev[..., :-1], dim=-1)
    valid = delta_e > 0
    # Match historical behavior: if delta_e <= 0, return all zeros for that spectrum.
    # Use a safe delta_e for computing bins, then mask the output.
    delta_e_safe = torch.where(valid, delta_e, torch.ones_like(delta_e))
    bins = torch.round(detector_escape_E / delta_e_safe).to(torch.int64)

    # For each spectrum: escape_spec[..., i] = spectra[..., i + bins] if in range else 0.
    # Expand bins to match spectra leading dims (gather does not broadcast indices).
    bins_view = bins
    if bins_view.ndim < spectra.ndim - 1:
        bins_view = bins_view.view((1,) * (spectra.ndim - 1 - bins_view.ndim) + bins_view.shape)
    bins_exp = bins_view.expand(spectra.shape[:-1]).unsqueeze(-1)

    base = torch.arange(n_ch, device=device).view((1,) * (spectra.ndim - 1) + (n_ch,))
    base = base.expand(spectra.shape[:-1] + (n_ch,))
    idx = base + bins_exp
    mask = idx < n_ch
    idx_safe = torch.clamp(idx, 0, n_ch - 1)
    shifted = torch.gather(spectra, -1, idx_safe) * escape_factor

    out = shifted * mask.to(shifted.dtype)
    if valid.ndim > 0:
        # Broadcast per-spectrum validity over channel axis.
        valid_view = valid
        if valid_view.ndim < spectra.ndim - 1:
            valid_view = valid_view.view((1,) * (spectra.ndim - 1 - valid_view.ndim) + valid_view.shape)
        out = out * valid_view.expand(spectra.shape[:-1]).unsqueeze(-1).to(out.dtype)
    elif not bool(valid.item()):
        out = out * 0.0
    return out


def build_model_basis(
    params,
    energy_range,
    elements_to_fit=default_fitting_elems,
    *,
    include_scatter=True,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    detector_type="Si",
    escape_factor=0.0,
    verbose=False,
):
    """Build a linear basis (K, n_channels) for fitting per-spectrum amplitudes.

    Each basis vector corresponds to a single component at *unit linear amplitude*.
    The modeled spectrum for amplitudes A (n_spectra, K) is: `A @ basis`.

    Notes:
        - `energy_range` uses absolute channel indices (inclusive), consistent with `model_spec`.
        - Energy/channel axis is the last dimension of the returned basis.
        - Scatter components (Compton + elastic) are optionally appended.
        - Escape peaks are incorporated as a linear operator when `escape_factor > 0`.
    """
    if energy_range is None or len(energy_range) != 2:
        raise ValueError("energy_range must be a (start, stop) tuple of channel indices")
    ch0, ch1 = int(energy_range[0]), int(energy_range[1])
    if ch1 < ch0:
        raise ValueError("energy_range stop must be >= start")

    device = torch.device(device)
    energy = torch.arange(ch0, ch1 + 1, device=device, dtype=torch.float32)
    ev = (
        params["ENERGY_OFFSET"]
        + params["ENERGY_SLOPE"] * energy
        + params["ENERGY_QUADRATIC"] * (energy**2)
    )

    # Preserve order but drop duplicates.
    components = []
    seen = set()
    for elem in elements_to_fit:
        if elem in seen:
            continue
        seen.add(elem)
        components.append(elem)
    if include_scatter:
        for scatter in ("COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"):
            if scatter not in seen:
                seen.add(scatter)
                components.append(scatter)

    basis_specs = []
    basis_names = []
    for comp in components:
        try:
            unit_params = dict(params)
            if comp == "COHERENT_SCT_AMPLITUDE":
                unit_params[comp] = torch.tensor(0.0, device=device, dtype=ev.dtype)
                spec = elastic_peak(unit_params, ev, unit_params["ENERGY_SLOPE"])
            elif comp == "COMPTON_AMPLITUDE":
                unit_params[comp] = torch.tensor(0.0, device=device, dtype=ev.dtype)
                spec = compton_peak(unit_params, ev, unit_params["ENERGY_SLOPE"], True, True)
            else:
                unit_params[comp] = torch.tensor(0.0, device=device, dtype=ev.dtype)
                spec = model_elem_spec(
                    unit_params,
                    comp,
                    ev,
                    True,
                    True,
                    device=device,
                    e_consts=e_consts,
                    elem_info=elem_info,
                )
            basis_specs.append(spec)
            basis_names.append(comp)
        except Exception as exc:
            if verbose:
                print(f"Failed to build basis for {comp}: {exc}")

    if not basis_specs:
        raise RuntimeError("No basis components could be built; check elements_to_fit")

    basis = torch.stack(basis_specs, dim=0)
    if escape_factor > 0.0:
        with torch.no_grad():
            basis = basis + escape_peak(basis, ev, escape_factor, device, detector_type)
    return basis, basis_names


def _binding_energy_mask(er_struct, coherent_sct_energy, elem_info):
    """Return a bool mask indicating whether a line is physically allowed.

    Mirrors `EnergyStruct.check_binding_energy`, but supports tensor `coherent_sct_energy`
    (e.g. per-spectrum or per-pixel tuning).
    """
    if getattr(er_struct, "type", 0) == 0:
        return torch.zeros_like(coherent_sct_energy, dtype=torch.bool)

    if getattr(er_struct, "is_pileup", False):
        components = getattr(er_struct, "pileup_components", ()) or parse_pileup_descriptor(
            er_struct.name
        )
        if components is None:
            return torch.ones_like(coherent_sct_energy, dtype=torch.bool)

        component_ptypes = getattr(er_struct, "pileup_ptypes", ()) or ()
        mask = torch.ones_like(coherent_sct_energy, dtype=torch.bool)
        for idx, (elem_name, shell) in enumerate(components):
            e_info = next((e for e in elem_info if e.name == elem_name), None)
            if e_info is None:
                return torch.ones_like(coherent_sct_energy, dtype=torch.bool)
            ptype = component_ptypes[idx] if idx < len(component_ptypes) else ""
            thr = binding_threshold_for_line(e_info, ptype, shell)
            if thr > 0.0:
                mask = mask & (coherent_sct_energy > thr)
        return mask

    elname = str(er_struct.name).split("_")[0]
    e_info = next((e for e in elem_info if e.name == elname), None)
    if e_info is None:
        return torch.ones_like(coherent_sct_energy, dtype=torch.bool)

    ptype = str(getattr(er_struct, "ptype", ""))
    thr = binding_threshold_for_line(e_info, ptype)
    if thr > 0.0:
        return coherent_sct_energy > thr

    return torch.zeros_like(coherent_sct_energy, dtype=torch.bool)


def model_elem_spec(
    params,
    element_to_fit,
    ev,
    use_step=True,
    use_tail=True,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
):
    spec = torch.zeros_like(ev, device=device)
    pre_faktor = 10 ** params[element_to_fit]
    if isinstance(pre_faktor, torch.Tensor) and pre_faktor.ndim == ev.ndim - 1:
        pre_faktor = pre_faktor.unsqueeze(-1)

    coherent = params["COHERENT_SCT_ENERGY"]
    if isinstance(coherent, torch.Tensor) and coherent.ndim == ev.ndim - 1:
        coherent = coherent.unsqueeze(-1)

    energy_slope = params["ENERGY_SLOPE"]
    if isinstance(energy_slope, torch.Tensor) and energy_slope.ndim == ev.ndim - 1:
        energy_slope = energy_slope.unsqueeze(-1)
    fano_coeff = _resolution_fano_coeff(params)
    for er_struct in e_consts[element_to_fit]:
        sigma = torch.sqrt(
            (params["FWHM_OFFSET"] / 2.3548) ** 2
            + er_struct.energy * fano_coeff * params["FWHM_FANOPRIME"]
        )
        if isinstance(sigma, torch.Tensor) and sigma.ndim == ev.ndim - 1:
            sigma = sigma.unsqueeze(-1)
        f_step = torch.abs(
            er_struct.mu_fraction
            * (params["F_STEP_OFFSET"] + params["F_STEP_LINEAR"] * er_struct.energy)
        )
        if isinstance(f_step, torch.Tensor) and f_step.ndim == ev.ndim - 1:
            f_step = f_step.unsqueeze(-1)
        f_tail = torch.abs(
            params["F_TAIL_OFFSET"] + params["F_TAIL_LINEAR"] * er_struct.mu_fraction
        )
        if isinstance(f_tail, torch.Tensor) and f_tail.ndim == ev.ndim - 1:
            f_tail = f_tail.unsqueeze(-1)
        kb_f_tail = torch.abs(
            params["KB_F_TAIL_OFFSET"]
            + params["KB_F_TAIL_LINEAR"] * er_struct.mu_fraction
        )
        if isinstance(kb_f_tail, torch.Tensor) and kb_f_tail.ndim == ev.ndim - 1:
            kb_f_tail = kb_f_tail.unsqueeze(-1)
        if er_struct.ratio == 0 or er_struct.energy <= 0:
            continue
        delta_energy = ev - er_struct.energy
        faktor = er_struct.ratio * pre_faktor

        binding_ok = _binding_energy_mask(er_struct, coherent, elem_info).to(dtype=spec.dtype)
        if er_struct.ptype.startswith("Ka") or er_struct.ptype.startswith("L"):
            denom = 1.0 + f_step + (f_tail if use_tail else 0.0)
        elif er_struct.ptype.startswith("Kb"):
            denom = 1.0 + f_step + (kb_f_tail if use_tail else 0.0)
        else:
            denom = 1.0
        faktor = (faktor / denom) * binding_ok

        spec.add_(faktor * peak(energy_slope, sigma, delta_energy))
        if use_step:
            spec.add_(faktor * f_step * step(energy_slope, sigma, delta_energy, er_struct.energy))
        if use_tail:
            if er_struct.ptype in ["Kb1", "Kb2"]:
                gamma = (
                    torch.abs(
                        params["GAMMA_OFFSET"]
                        + params["GAMMA_LINEAR"] * er_struct.energy
                    )
                    * er_struct.width_multi
                )
                if isinstance(gamma, torch.Tensor) and gamma.ndim == ev.ndim - 1:
                    gamma = gamma.unsqueeze(-1)
                spec.add_(
                    faktor
                    * kb_f_tail
                    * tail(energy_slope, sigma, delta_energy, gamma)
                )
            if er_struct.ptype in [
                "Ka1",
                "Ka2",
                "La1",
                "La2",
                "Lb1",
                "Lb2",
                "Lb3",
                "Lb4",
                "Lg1",
                "Lg2",
                "Lg3",
                "Lg4",
                "Ll",
                "Ln",
            ]:
                gamma = (
                    torch.abs(
                        params["GAMMA_OFFSET"]
                        + params["GAMMA_LINEAR"] * er_struct.energy
                    )
                    * er_struct.width_multi
                )
                if isinstance(gamma, torch.Tensor) and gamma.ndim == ev.ndim - 1:
                    gamma = gamma.unsqueeze(-1)
                spec.add_(
                    faktor
                    * f_tail
                    * tail(energy_slope, sigma, delta_energy, gamma)
                )
    return spec


def model_spec(
    params,
    energy_range,
    elements_to_fit=default_fitting_elems,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
    detector_type="Si", # "Si" or "Ge"
    escape_factor=0.0,
):
    # Determine batch shape (if any) from provided parameters. Batched parameters are
    # expected to be shaped like `batch_shape` or `batch_shape + (1,)`.
    batch_shape = ()
    for v in params.values():
        if isinstance(v, torch.Tensor) and v.ndim > 0:
            batch_shape = v.shape[:-1] if v.shape[-1] == 1 else v.shape
            break

    energy = torch.arange(
        energy_range[0],
        energy_range[1] + 1,
        device=device,
        dtype=torch.float32,
    )
    if batch_shape:
        energy = energy.view((1,) * len(batch_shape) + (energy.numel(),))

    def _as_channel_broadcast(val):
        if not isinstance(val, torch.Tensor) or val.ndim == 0:
            return val
        if batch_shape and tuple(val.shape) == tuple(batch_shape):
            return val.unsqueeze(-1)
        return val

    energy_offset = _as_channel_broadcast(params["ENERGY_OFFSET"])
    energy_slope = _as_channel_broadcast(params["ENERGY_SLOPE"])
    energy_quadratic = _as_channel_broadcast(params["ENERGY_QUADRATIC"])

    ev = energy_offset + energy_slope * energy + energy_quadratic * (energy**2)
    if batch_shape and tuple(ev.shape[:-1]) != tuple(batch_shape):
        ev = ev.expand(batch_shape + (ev.shape[-1],))
    agr_spec = torch.zeros_like(ev)
    for e in elements_to_fit:
        if e in e_consts and not e in [
            "COMPTON_AMPLITUDE",
            "COHERENT_SCT_AMPLITUDE",
        ]:
            try:
                element_spec = model_elem_spec(
                    params,
                    e,
                    ev,
                    True,
                    True,
                    device=device,
                    e_consts=e_consts,
                    elem_info=elem_info,
                )
                agr_spec.add_(element_spec)
            except Exception as exc:
                err_key = (str(e), type(exc).__name__, str(exc))
                if err_key not in _MODEL_SPEC_FAILURE_CACHE:
                    _MODEL_SPEC_FAILURE_CACHE.add(err_key)
                    print(
                        f"Failed to model spectrum for {e}: "
                        f"{type(exc).__name__}: {exc}"
                    )
    elastic_spec = elastic_peak(params, ev, energy_slope)
    compton_spec = compton_peak(params, ev, energy_slope, True, True)
    agr_spec.add_(elastic_spec)
    agr_spec.add_(compton_spec)
    if escape_factor > 0.0:
        with torch.no_grad():
            # Escape reindexes channels via round/gather, so we keep it outside the
            # autograd path until a differentiable surrogate is introduced.
            escape_spec = escape_peak(agr_spec, ev, escape_factor, device, detector_type)
            agr_spec.add_(escape_spec)
    return agr_spec
