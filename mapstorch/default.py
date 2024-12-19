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

from mapstorch.constant import MapsElements, define_constants

# Lists of fitting parameters
default_fitting_params = [
    "COHERENT_SCT_ENERGY",
    "ENERGY_OFFSET",
    "ENERGY_SLOPE",
    "ENERGY_QUADRATIC",
    "COMPTON_ANGLE",
    "COMPTON_FWHM_CORR",
    "COMPTON_HI_F_TAIL",
    "COMPTON_F_TAIL",
    "FWHM_FANOPRIME",
    "FWHM_OFFSET",
    "F_TAIL_OFFSET",
    "KB_F_TAIL_OFFSET",
]

# Default values for fitting parameters
default_param_vals = {
    "GE_ESCAPE": 0.0,
    "COMPTON_F_TAIL": 0.783374011516571,
    "COMPTON_ANGLE": 109.85600280761719,
    "KB_F_TAIL_QUADRATIC": 0.0,
    "COHERENT_SCT_ENERGY": 12.270400047302246,
    "F_TAIL_OFFSET": 0.03593109920620918,
    "F_STEP_LINEAR": 0.0,
    "GAMMA_QUADRATIC": 0.0,
    "FWHM_OFFSET": -0.08414760231971741,
    "ENERGY_QUADRATIC": -2.2838600344243787e-08,
    "COMPTON_FWHM_CORR": 2.7478699684143066,
    "F_STEP_QUADRATIC": 0.0,
    "COMPTON_F_STEP": 0.0,
    "SI_ESCAPE": 0.0,
    "FWHM_FANOPRIME": 0.00017760999617166817,
    "COMPTON_HI_GAMMA": 3.0,
    "ENERGY_OFFSET": -0.0020250400993973017,
    "KB_F_TAIL_OFFSET": 0.16684700548648834,
    "ENERGY_SLOPE": 0.011979999952018261,
    "F_TAIL_QUADRATIC": 0.0,
    "COMPTON_GAMMA": 3.0,
    "F_STEP_OFFSET": 0.0,
    "COMPTON_HI_F_TAIL": 0.08728790283203125,
    "F_TAIL_LINEAR": 1.6940699334764354e-21,
    "GAMMA_OFFSET": 2.2101199626922607,
    "KB_F_TAIL_LINEAR": 0.0,
    "MAX_ENERGY_TO_FIT": 14.800000190734863,
    "GAMMA_LINEAR": 0.0,
    "MIN_ENERGY_TO_FIT": 1.0,
    "SNIP_WIDTH": 0.5,
}

# Dictionary of learning rates for each fitting parameter
# around 1 percent of expected change
default_learning_rates = {
    "default_lr": 1e-2,
    "COHERENT_SCT_ENERGY": 1e-2,
    "ENERGY_OFFSET": 1e-3,
    "ENERGY_SLOPE": 1e-6,
    "ENERGY_QUADRATIC": 1e-7,
    "COMPTON_ANGLE": 1,
    "COMPTON_FWHM_CORR": 1e-2,
    "COMPTON_HI_F_TAIL": 1e-6,
    "COMPTON_F_TAIL": 1e-4,
    "FWHM_FANOPRIME": 1e-8,
    "FWHM_OFFSET": 1e-4,
    "F_TAIL_OFFSET": 1e-4,
    "KB_F_TAIL_OFFSET": 1e-4,
}

param_name_map = {
    "CAL_OFFSET_[E_OFFSET]": "ENERGY_OFFSET",
    "CAL_SLOPE_[E_LINEAR]": "ENERGY_SLOPE",
    "CAL_QUAD_[E_QUADRATIC]": "ENERGY_QUADRATIC",
    "STEP_OFFSET": "F_STEP_OFFSET",
    "STEP_LINEAR": "F_STEP_LINEAR",
    "STEP_QUADRATIC": "F_STEP_QUADRATIC",
    "COMPTON_OFFSET": "COMPTON_F_STEP",
    "COMPTON_LINEAR": "COMPTON_F_LINEAR",
    "COMPTON_QUADRATIC": "COMPTON_F_QUADRATIC",
    "COMPTON_STEP": "COMPTON_F_STEP",
    "SI_ESCAPE_ENABLE": "SI_ESCAPE",
}

# Default fitting elements
default_K_lines = [
    "Ag",
    "Al",
    "Ar",
    "As",
    "Br",
    "Ca",
    "Cd",
    "Cl",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Ga",
    "Ge",
    "Hf",
    "I",
    "K",
    "Mg",
    "Mn",
    "Mo",
    "Ni",
    "P",
    "Pb",
    "Pd",
    "Rb",
    "S",
    "Sc",
    "Se",
    "Si",
    "Sr",
    "Ti",
    "V",
    "Y",
    "Zn",
    "Zr",
]
default_L_lines = [
    "Ag_L",
    "As_L",
    "Au_L",
    "Ba_L",
    "Bi_L",
    "Br_L",
    "Cd_L",
    "Ce_L",
    "Cs_L",
    "Dy_L",
    "Er_L",
    "Eu_L",
    "Gd_L",
    "Hf_L",
    "Hg_L",
    "Ho_L",
    "I_L",
    "Kr_L",
    "La_L",
    "Lu_L",
    "Mo_L",
    "Nd_L",
    "Os_L",
    "Pa_L",
    "Pb_L",
    "Pm_L",
    "Pr_L",
    "Pt_L",
    "Rb_L",
    "Se_L",
    "Sm_L",
    "Sn_L",
    "Sr_L",
    "Ta_L",
    "Tb_L",
    "Th_L",
    "Tl_L",
    "Tm_L",
    "U_L",
    "W_L",
    "Xe_L",
    "Y_L",
    "Yb_L",
    "Zn_L",
    "Zr_L",
]
default_M_lines = [
    "Au_M",
    "Hf_M",
    "Hg_M",
    "Os_M",
    "Pt_M",
    "Th_M",
    "U_M",
    "W_M",
]
default_pileups = [
    "Si_Si",
    "Si_Cl",
    "Cl_Cl",
]
default_auxiliary_comps = [
    "COMPTON_AMPLITUDE",
    "COHERENT_SCT_AMPLITUDE",
]
default_fitting_elems = (
    default_K_lines
    + default_pileups
    + default_auxiliary_comps
    + default_L_lines
    + default_M_lines
)

unsupported_elements = [
    "Ac",
    "Am",
    "At",
    "B",
    "Be",
    "Bh",
    "Bk",
    "C",
    "Cf",
    "Cm",
    "Cn",
    "Db",
    "Ds",
    "Es",
    "F",
    "Fl",
    "Fm",
    "Fr",
    "H",
    "He",
    "Hs",
    "In",
    "Ir",
    "Li",
    "Lr",
    "Lv",
    "Mc",
    "Md",
    "Mt",
    "N",
    "Na",
    "Nb",
    "Ne",
    "Nh",
    "No",
    "Np",
    "O",
    "Og",
    "Po",
    "Pu",
    "Ra",
    "Re",
    "Rf",
    "Rg",
    "Rh",
    "Rn",
    "Ru",
    "Sb",
    "Sg",
    "Tc",
    "Te",
    "Ts",
]

supported_elements_mapping = {
    "Ag": ["K", "L"],
    "Al": ["K"],
    "Ar": ["K"],
    "As": ["K", "L"],
    "Au": ["L", "M"],
    "Ba": ["L"],
    "Bi": ["L"],
    "Br": ["K", "L"],
    "Ca": ["K"],
    "Cd": ["K", "L"],
    "Ce": ["L"],
    "Cl": ["K"],
    "Co": ["K"],
    "Cr": ["K"],
    "Cs": ["L"],
    "Cu": ["K"],
    "Dy": ["L"],
    "Er": ["L"],
    "Eu": ["L"],
    "Fe": ["K"],
    "Ga": ["K"],
    "Gd": ["L"],
    "Ge": ["K"],
    "Hf": ["K", "L", "M"],
    "Hg": ["L", "M"],
    "Ho": ["L"],
    "I": ["K", "L"],
    "K": ["K"],
    "Kr": ["L"],
    "La": ["L"],
    "Lu": ["L"],
    "Mg": ["K"],
    "Mn": ["K"],
    "Mo": ["K", "L"],
    "Nd": ["L"],
    "Ni": ["K"],
    "Os": ["L", "M"],
    "P": ["K"],
    "Pa": ["L"],
    "Pb": ["K", "L"],
    "Pd": ["K"],
    "Pm": ["L"],
    "Pr": ["L"],
    "Pt": ["L", "M"],
    "Rb": ["K", "L"],
    "S": ["K"],
    "Sc": ["K"],
    "Se": ["K", "L"],
    "Si": ["K"],
    "Sm": ["L"],
    "Sn": ["L"],
    "Sr": ["K", "L"],
    "Ta": ["L"],
    "Tb": ["L"],
    "Th": ["L", "M"],
    "Ti": ["K"],
    "Tl": ["L"],
    "Tm": ["L"],
    "U": ["L", "M"],
    "V": ["K"],
    "W": ["L", "M"],
    "Xe": ["L"],
    "Y": ["K", "L"],
    "Yb": ["L"],
    "Zn": ["K", "L"],
    "Zr": ["K", "L"],
}

# Default element information
default_elem_info = MapsElements().get_element_info()

# Default element constants
default_energy_consts = define_constants(default_elem_info, default_pileups)
