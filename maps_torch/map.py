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

from maps_torch.constant import M_PI, SQRT_2XPI, M_SQRT2
from maps_torch.default import default_fitting_elems, default_energy_consts, default_elem_info

def peak(gain, sigma, delta_energy):
    return gain / (sigma * SQRT_2XPI) * torch.exp(-0.5 * (delta_energy / sigma) ** 2)


def step(gain, sigma, delta_energy, peak_E):
    return gain / 2.0 / peak_E * torch.erfc(delta_energy / (M_SQRT2 * sigma))


def tail(gain, sigma, delta_energy, gamma):
    val1 = torch.erfc(delta_energy / (M_SQRT2 * sigma) + (1.0 / (gamma * M_SQRT2)))
    val2 = torch.exp(delta_energy / (gamma * sigma)) * val1
    d_e = torch.where(delta_energy < 0, val2, val1)
    return gain / 2.0 / gamma / sigma / torch.exp(-0.5 / gamma**2) * d_e


def elastic_peak(params, ev, gain):
    sigma = torch.sqrt(
        (params["FWHM_OFFSET"] / 2.3548) ** 2
        + params["COHERENT_SCT_ENERGY"] * 2.96 * params["FWHM_FANOPRIME"]
    )
    delta_energy = ev - params["COHERENT_SCT_ENERGY"]
    counts = 10 ** params["COHERENT_SCT_AMPLITUDE"] * peak(gain, sigma, delta_energy)
    return counts


def compton_peak(params, ev, gain, use_step=True, use_tail=False):
    compton_E = params["COHERENT_SCT_ENERGY"] / (
        1
        + (params["COHERENT_SCT_ENERGY"] / 511)
        * (1 - torch.cos(params["COMPTON_ANGLE"] * 2 * M_PI / 360.0))
    )
    sigma = torch.sqrt(
        (params["FWHM_OFFSET"] / 2.3548) ** 62
        + compton_E * 2.96 * params["FWHM_FANOPRIME"]
    )
    delta_energy = ev - compton_E
    faktor = 1.0 / (
        1.0
        + params["COMPTON_F_STEP"]
        + params["COMPTON_F_TAIL"]
        + params["COMPTON_HI_F_TAIL"]
    )
    faktor *= 10 ** params["COMPTON_AMPLITUDE"]
    counts = faktor * peak(gain, sigma * params["COMPTON_FWHM_CORR"], delta_energy)
    if params["COMPTON_F_STEP"] > 0 and use_step:
        counts += (faktor * params["COMPTON_F_STEP"]) * step(
            gain, sigma, delta_energy, compton_E
        )
    if use_tail:
        counts += (faktor * params["COMPTON_F_TAIL"]) * tail(
            gain, sigma, delta_energy, params["COMPTON_GAMMA"]
        )
        counts += (faktor * params["COMPTON_HI_F_TAIL"]) * tail(
            gain, sigma, -delta_energy, params["COMPTON_HI_GAMMA"]
        )
    return counts


def escape_peak(spectra, ev, escape_factor, device):
    Si_K_edge = 1.73998
    bins = int(
        Si_K_edge / (ev[1] - ev[0])
    )  # todo: ev[1] - ev[0] can be zero and cause error...
    escape_spec = torch.zeros_like(spectra, device=device)
    # for i in range(len(ev) - bins):
    #     escape_spec[i] = spectra[i + bins] * escape_factor
    escape_spec[: len(ev) - bins] = spectra[bins : len(ev)] * escape_factor
    return escape_spec


def model_elem_spec(
    params, element_to_fit, ev, use_step=True, use_tail=True, device="cpu", e_consts=default_energy_consts, elem_info=default_elem_info
):
    spec = torch.zeros_like(ev, device=device)
    pre_faktor = 10 ** params[element_to_fit]
    for er_struct in e_consts[element_to_fit]:
        sigma = torch.sqrt(
            (params["FWHM_OFFSET"] / 2.3548) ** 2
            + er_struct.energy * 2.96 * params["FWHM_FANOPRIME"]
        )
        f_step = torch.abs(
            er_struct.mu_fraction
            * (params["F_STEP_OFFSET"] + params["F_STEP_LINEAR"] * er_struct.energy)
        )
        f_tail = torch.abs(
            params["F_TAIL_OFFSET"] + params["F_TAIL_LINEAR"] * er_struct.mu_fraction
        )
        kb_f_tail = torch.abs(
            params["KB_F_TAIL_OFFSET"]
            + params["KB_F_TAIL_LINEAR"] * er_struct.mu_fraction
        )
        if er_struct.ratio == 0 or er_struct.energy <= 0:
            continue
        delta_energy = ev - er_struct.energy
        faktor = er_struct.ratio * pre_faktor
        label = ""
        if er_struct.check_binding_energy(
            elem_info, params["COHERENT_SCT_ENERGY"]
        ):
            if er_struct.ptype.startswith("Ka"):
                label = "K Alpha"
                faktor = faktor / (1.0 + f_tail + f_step)
            elif er_struct.ptype.startswith("Kb"):
                label = "K Beta"
                faktor = faktor / (1.0 + kb_f_tail + f_step)
            elif er_struct.ptype.startswith("L"):
                label = "L Lines"
                faktor = faktor / (1.0 + f_tail + f_step)
            else:
                pass
        else:
            faktor = 0
        spec += faktor * peak(params["ENERGY_SLOPE"], sigma, delta_energy)
        if f_step > 0 and use_step:
            spec += (
                faktor
                * f_step
                * step(params["ENERGY_SLOPE"], sigma, delta_energy, er_struct.energy)
            )
        if use_tail:
            if er_struct.ptype in ["Kb1", "Kb2"]:
                gamma = (
                    torch.abs(
                        params["GAMMA_OFFSET"]
                        + params["GAMMA_LINEAR"] * er_struct.energy
                    )
                    * er_struct.width_multi
                )
                spec += (
                    faktor
                    * kb_f_tail
                    * tail(params["ENERGY_SLOPE"], sigma, delta_energy, gamma)
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
                spec += (
                    faktor
                    * f_tail
                    * tail(params["ENERGY_SLOPE"], sigma, delta_energy, gamma)
                )
    return spec


def model_spec(
    params,
    energy_range,
    elements_to_fit=default_fitting_elems,
    extra_info=False,
    use_step=True,
    use_tail=False,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
):
    if extra_info:
        agr_specs = []
    agr_spec = torch.zeros(energy_range[1] - energy_range[0] + 1, device=device)
    energy = torch.linspace(
        energy_range[0],
        energy_range[1] + 1,
        energy_range[1] - energy_range[0] + 1,
        device=device,
    )
    ev = (
        params["ENERGY_OFFSET"]
        + params["ENERGY_SLOPE"] * energy
        + params["ENERGY_QUADRATIC"] * (energy**2)
    )
    if extra_info:
        agr_specs.append(ev)
    for e in elements_to_fit:
        if e in default_fitting_elems and not e in ["COMPTON_AMPLITUDE", "COHERENT_SCT_AMPLITUDE"]:
            try:
                element_spec = model_elem_spec(
                    params, e, ev, use_step, use_tail, device=device, e_consts=e_consts, elem_info=elem_info
                )
                agr_spec += element_spec
            except Exception:
                print("Failed to model spectrum for", e)
            if extra_info:
                agr_specs.append(element_spec)
    elastic_spec = elastic_peak(params, ev, params["ENERGY_SLOPE"])
    compton_spec = compton_peak(params, ev, params["ENERGY_SLOPE"], use_step, use_tail)
    if extra_info:
        agr_specs.append(elastic_spec)
        agr_specs.append(compton_spec)
    agr_spec += elastic_spec + compton_spec
    if params["ENERGY_OFFSET"] > 0:
        escape_spec = escape_peak(agr_spec, ev, params["SI_ESCAPE"], device)
        if extra_info:
            agr_specs.append(escape_spec)
        agr_spec += escape_spec
    if extra_info:
        agr_specs.append(agr_spec)
        return agr_specs
    else:
        return agr_spec


def elastic_peak_vol(params, ev, gain):
    sigma = torch.sqrt(
        (params["FWHM_OFFSET"] / 2.3548) ** 2
        + params["COHERENT_SCT_ENERGY"] * 2.96 * params["FWHM_FANOPRIME"]
    )
    delta_energy = ev - params["COHERENT_SCT_ENERGY"]
    counts = (
        peak(gain, sigma, delta_energy)
        * (10 ** params["COHERENT_SCT_AMPLITUDE"])[..., None]
    )
    return counts


def compton_peak_vol(params, ev, gain, use_step=True, use_tail=True):
    compton_E = params["COHERENT_SCT_ENERGY"] / (
        1
        + (params["COHERENT_SCT_ENERGY"] / 511)
        * (1 - torch.cos(params["COMPTON_ANGLE"] * 2 * M_PI / 360.0))
    )
    sigma = torch.sqrt(
        (params["FWHM_OFFSET"] / 2.3548) ** 62
        + compton_E * 2.96 * params["FWHM_FANOPRIME"]
    )
    delta_energy = ev - compton_E
    faktor = 1.0 / (
        1.0
        + params["COMPTON_F_STEP"]
        + params["COMPTON_F_TAIL"]
        + params["COMPTON_HI_F_TAIL"]
    )
    faktor = faktor * 10 ** params["COMPTON_AMPLITUDE"]
    counts = (
        peak(gain, sigma * params["COMPTON_FWHM_CORR"], delta_energy)
        * faktor[..., None]
    )
    if params["COMPTON_F_STEP"] > 0 and use_step:
        counts += (
            step(gain, sigma, delta_energy, compton_E)
            * (faktor * params["COMPTON_F_STEP"])[..., None]
        )
    if use_tail:
        counts += (
            tail(gain, sigma, delta_energy, params["COMPTON_GAMMA"])
            * (faktor * params["COMPTON_F_TAIL"])[..., None]
        )
        counts += (
            tail(gain, sigma, -delta_energy, params["COMPTON_HI_GAMMA"])
            * (faktor * params["COMPTON_HI_F_TAIL"])[..., None]
        )
    return counts


def escape_peak_vol(spectra, ev, escape_factor, device):
    Si_K_edge = 1.73998
    bins = (Si_K_edge / (ev[..., 1] - ev[..., 0])).to(torch.int32)
    escape_spec = torch.zeros_like(spectra, device=device)
    if bins.dim() == 0:
        escape_spec[: ev.shape[-1] - bins] = (
            spectra[bins : ev.shape[-1]] * escape_factor
        )
    elif bins.dim() == 1:
        for i in range(bins.shape[0]):
            escape_spec[i, 0 : ev.shape[-1] - bins[i]] = (
                spectra[i, bins[i] : ev.shape[-1]] * escape_factor
            )
    elif bins.dim() == 2:
        for i in range(bins.shape[0]):
            for j in range(bins.shape[1]):
                escape_spec[i, j, 0 : ev.shape[-1] - bins[i, j]] = (
                    spectra[i, j, bins[i, j] : ev.shape[-1]] * escape_factor
                )
    else:
        raise ValueError("escape_peak_vol: bins dim > 2")
    return escape_spec


def model_elem_spec_vol(
    params, element_to_fit, ev, use_step=True, use_tail=True, device="cpu", e_consts=default_energy_consts, elem_info=default_elem_info
):
    spec = torch.zeros_like(ev, device=device)
    pre_faktor = 10 ** params[element_to_fit]
    for er_struct in e_consts[element_to_fit]:
        sigma = torch.sqrt(
            (params["FWHM_OFFSET"] / 2.3548) ** 2
            + er_struct.energy * 2.96 * params["FWHM_FANOPRIME"]
        )
        f_step = torch.abs(
            er_struct.mu_fraction
            * (params["F_STEP_OFFSET"] + params["F_STEP_LINEAR"] * er_struct.energy)
        )
        f_tail = torch.abs(
            params["F_TAIL_OFFSET"] + params["F_TAIL_LINEAR"] * er_struct.mu_fraction
        )
        kb_f_tail = torch.abs(
            params["KB_F_TAIL_OFFSET"]
            + params["KB_F_TAIL_LINEAR"] * er_struct.mu_fraction
        )
        if er_struct.ratio == 0 or er_struct.energy <= 0:
            continue
        delta_energy = ev - er_struct.energy
        faktor = er_struct.ratio * pre_faktor
        label = ""
        if er_struct.check_binding_energy(
            elem_info, params["COHERENT_SCT_ENERGY"]
        ):
            if er_struct.ptype.startswith("Ka"):
                label = "K Alpha"
                faktor = faktor / (1.0 + f_tail + f_step)
            elif er_struct.ptype.startswith("Kb"):
                label = "K Beta"
                faktor = faktor / (1.0 + kb_f_tail + f_step)
            elif er_struct.ptype.startswith("L"):
                label = "L Lines"
                faktor = faktor / (1.0 + f_tail + f_step)
            else:
                pass
        else:
            faktor = torch.zeros_like(faktor, device=device)
        spec += peak(params["ENERGY_SLOPE"], sigma, delta_energy) * faktor[..., None]
        if f_step > 0 and use_step:
            spec += f_step * (
                step(params["ENERGY_SLOPE"], sigma, delta_energy, er_struct.energy)
                * faktor[..., None]
            )
        if use_tail:
            if er_struct.ptype in ["Kb1", "Kb2"]:
                gamma = (
                    torch.abs(
                        params["GAMMA_OFFSET"]
                        + params["GAMMA_LINEAR"] * er_struct.energy
                    )
                    * er_struct.width_multi
                )
                spec += kb_f_tail * (
                    tail(params["ENERGY_SLOPE"], sigma, delta_energy, gamma)
                    * faktor[..., None]
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
                spec += f_tail * (
                    tail(params["ENERGY_SLOPE"], sigma, delta_energy, gamma)
                    * faktor[..., None]
                )
    return spec


def model_spec_vol(
    params,
    energy_range,
    spec_vol_shape,
    elements_to_fit=default_fitting_elems,
    extra_info=False,
    use_step=True,
    use_tail=False,
    device="cpu",
    e_consts=default_energy_consts,
    elem_info=default_elem_info,
):
    if extra_info:
        agr_specs = []
    agr_spec = torch.zeros(spec_vol_shape, device=device)
    energy = (
        torch.linspace(
            energy_range[0], energy_range[1] + 1, energy_range[1] - energy_range[0] + 1
        )
        .expand(spec_vol_shape)
        .to(device=device)
    )
    ev = (
        params["ENERGY_OFFSET"]
        + params["ENERGY_SLOPE"] * energy
        + params["ENERGY_QUADRATIC"] * (energy**2)
    )
    if extra_info:
        agr_specs.append(ev)
    for e in elements_to_fit:
        if e in default_fitting_elems:
            element_spec = model_elem_spec_vol(
                params, e, ev, use_step, use_tail, device=device, e_consts=e_consts, elem_info=elem_info
            )
            if extra_info:
                agr_specs.append(element_spec)
            agr_spec += element_spec
    elastic_spec = elastic_peak_vol(params, ev, params["ENERGY_SLOPE"])
    compton_spec = compton_peak_vol(
        params, ev, params["ENERGY_SLOPE"], use_step, use_tail
    )
    if extra_info:
        agr_specs.append(elastic_spec)
        agr_specs.append(compton_spec)
    agr_spec += elastic_spec + compton_spec
    if params["ENERGY_OFFSET"] > 0:
        escape_spec = escape_peak_vol(agr_spec, ev, params["SI_ESCAPE"], device)
        if extra_info:
            agr_specs.append(escape_spec)
        agr_spec += escape_spec
    if extra_info:
        agr_specs.append(agr_spec)
        return agr_specs
    else:
        return agr_spec

