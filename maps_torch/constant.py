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

import time, csv, math
import numpy as np
from xdrlib import *
from pathlib import Path

reference_directory = str(Path(__file__).parent.parent.absolute()) + "/reference/"
element_henke_filename = "henke.xdr"
element_csv_filename = "xrf_library.csv"

kele = [
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Hf",
    "Pb",
]

lele = [
    "Mo_L",
    "Tc_L",
    "Ru_L",
    "Rh_L",
    "Pd_L",
    "Ag_L",
    "Cd_L",
    "In_L",
    "Sn_L",
    "Sb_L",
    "Te_L",
    "I_L",
    "Xe_L,",
    "Cs_L",
    "Ba_L",
    "La_L",
    "Ce_L",
    "Pr_L",
    "Nd_L",
    "Pm_L",
    "Sm_L",
    "Eu_L",
    "Gd_L",
    "Tb_L",
    "Dy_L",
    "Ho_L",
    "Er_L",
    "Tm_L",
    "Yb_L",
    "Lu_L",
    "Hf_L",
    "Ta_L",
    "W_L",
    "Re_L",
    "Os_L",
    "Ir_L",
    "Pt_L",
    "Au_L",
    "Hg_L",
    "Tl_L",
    "Pb_L",
    "Bi_L",
    "Po_L",
    "At_L",
    "Rn_L",
    "Fr_L",
    "Ac_L",
    "Th_L",
    "Pa_L",
    "U_L",
    "Np_L",
    "Pu_L",
    "Am_L",
    "Zn_L",
    "Zr_L",
    "Y_L",
    "Sr_L",
    "Se_L",
    "As_L",
    "Xe_L",
    "Br_L",
    "Kr_L",
    "Rb_L",
]

mele = ["Au_M", "Pb_M", "U_M", "Pt_M", "Th_M", "Hf_M", "W_M", "Os_M", "Hg_M"]

kele_pos = np.arange(len(kele)) + 1
lele_pos = np.arange(len(lele)) + np.amax(kele_pos) + 1
mele_pos = np.arange(len(mele)) + np.amax(lele_pos) + 1

AVOGADRO = 6.02204531e23
HC_ANGSTROMS = 12398.52
RE = 2.817938070e-13  # in cm
M_PI = math.pi
M_SQRT2 = math.sqrt(2)
SQRT_2XPI = math.sqrt(2 * math.pi)
ENERGY_RES_OFFSET = 150.0
ENERGY_RES_SQRT = 12.0


def open_file_with_retry(
    filename, open_attr, retry_amt=5, retry_sleep=0.1, retry_sleep_inc=1.1
):
    file = None
    for i in range(retry_amt):
        try:
            file = open(filename, open_attr)
            break
        except:
            time.sleep(retry_sleep)
            retry_sleep += retry_sleep_inc
    return file


class Henke:
    def __init__(self, filepath=None):
        self.filepath = (
            reference_directory + element_henke_filename
            if filepath is None
            else filepath
        )

        self.compound_name = [
            "water",
            "protein",
            "lipid",
            "nucleosome",
            "dna",
            "helium",
            "chromatin",
            "air",
            "pmma",
            "nitride",
            "graphite",
            "nickel",
            "beryl",
            "copper",
            "quartz",
            "aluminum",
            "gold",
            "ice",
            "carbon",
            "polystyrene",
            "silicon",
            "germanium",
        ]

        self.compound_forumula = [
            "H2O",
            "H48.6C32.9N8.9O8.9S0.6",
            "H62.5C31.5O6.3",
            "H42.1C31.9N10.3O13.9P1.6S0.3",
            "H35.5C30.8N11.7O18.9P3.1",
            "He",
            "H49.95C24.64N8.66O15.57P1.07S0.03",
            "N78.08O20.95Ar0.93",
            "C5H8O2",
            "Si3N4",
            "C",
            "Ni",
            "Be",
            "Cu",
            "SiO2",
            "Al",
            "Au",
            "H2O",
            "C",
            "C8H8",
            "Si",
            "Ge",
        ]
        self.compound_density = [
            1.0,
            1.35,
            1.0,
            1.5,
            1.7,
            1.66e-04,
            1.527,
            1.20e-03,
            1.18,
            3.44,
            2.26,
            8.876,
            1.845,
            8.96,
            2.2,
            2.7,
            19.3,
            0.92,
            1,
            1.06,
            2.33,
            5.323,
        ]

    def compound(self, compound_string):
        z_array = []
        atwt = 0
        if compound_string in self.compound_name:
            compound_string = self.compound_forumula[
                self.compound_name.index(compound_string)
            ]

        if compound_string in self.compound_forumula:
            z_array = self.zcompound(compound_string, z_array)
            atwt = self.zatwt(z_array)

        return z_array, atwt

    def zcompound(self, compound_string, z_array, paren_multiplier=False):

        verbose = False
        if verbose:
            print("compound_string: %s", compound_string)
        if paren_multiplier == False:
            z_array = np.zeros(92)
            paren_multiplier = 1.0

        max_z_index = 93

        last_char_index = len(compound_string) - 1

        if compound_string[0] != "(":
            first_char = compound_string[0]
            if len(compound_string) > 1:
                second_char = compound_string[1]
            else:
                second_char = ""
            this_element_name = first_char

            if second_char >= "a" and second_char <= "z":
                this_element_name = this_element_name + second_char
                num_start_index = 2
            else:
                this_element_name = this_element_name + " "
                num_start_index = 1

        if verbose:
            print(
                "this_element_name: %s num_start_index: %s",
                this_element_name,
                num_start_index,
            )

        this_z = 0
        if this_element_name == "H ":
            this_z = 1
        elif this_element_name == "He":
            this_z = 2
        elif this_element_name == "Li":
            this_z = 3
        elif this_element_name == "Be":
            this_z = 4
        elif this_element_name == "B ":
            this_z = 5
        elif this_element_name == "C ":
            this_z = 6
        elif this_element_name == "N ":
            this_z = 7
        elif this_element_name == "O ":
            this_z = 8
        elif this_element_name == "F ":
            this_z = 9
        elif this_element_name == "Ne":
            this_z = 10
        elif this_element_name == "Na":
            this_z = 11
        elif this_element_name == "Mg":
            this_z = 12
        elif this_element_name == "Al":
            this_z = 13
        elif this_element_name == "Si":
            this_z = 14
        elif this_element_name == "P ":
            this_z = 15
        elif this_element_name == "S ":
            this_z = 16
        elif this_element_name == "Cl":
            this_z = 17
        elif this_element_name == "Ar":
            this_z = 18
        elif this_element_name == "K ":
            this_z = 19
        elif this_element_name == "Ca":
            this_z = 20
        elif this_element_name == "Sc":
            this_z = 21
        elif this_element_name == "Ti":
            this_z = 22
        elif this_element_name == "V ":
            this_z = 23
        elif this_element_name == "Cr":
            this_z = 24
        elif this_element_name == "Mn":
            this_z = 25
        elif this_element_name == "Fe":
            this_z = 26
        elif this_element_name == "Co":
            this_z = 27
        elif this_element_name == "Ni":
            this_z = 28
        elif this_element_name == "Cu":
            this_z = 29
        elif this_element_name == "Zn":
            this_z = 30
        elif this_element_name == "Ga":
            this_z = 31
        elif this_element_name == "Ge":
            this_z = 32
        elif this_element_name == "As":
            this_z = 33
        elif this_element_name == "Se":
            this_z = 34
        elif this_element_name == "Br":
            this_z = 35
        elif this_element_name == "Kr":
            this_z = 36
        elif this_element_name == "Rb":
            this_z = 37
        elif this_element_name == "Sr":
            this_z = 38
        elif this_element_name == "Y ":
            this_z = 39
        elif this_element_name == "Zr":
            this_z = 40
        elif this_element_name == "Nb":
            this_z = 41
        elif this_element_name == "Mo":
            this_z = 42
        elif this_element_name == "Tc":
            this_z = 43
        elif this_element_name == "Ru":
            this_z = 44
        elif this_element_name == "Rh":
            this_z = 45
        elif this_element_name == "Pd":
            this_z = 46
        elif this_element_name == "Ag":
            this_z = 47
        elif this_element_name == "Cd":
            this_z = 48
        elif this_element_name == "In":
            this_z = 49
        elif this_element_name == "Sn":
            this_z = 50
        elif this_element_name == "Sb":
            this_z = 51
        elif this_element_name == "Te":
            this_z = 52
        elif this_element_name == "I ":
            this_z = 53
        elif this_element_name == "Xe":
            this_z = 54
        elif this_element_name == "Cs":
            this_z = 55
        elif this_element_name == "Ba":
            this_z = 56
        elif this_element_name == "La":
            this_z = 57
        elif this_element_name == "Ce":
            this_z = 58
        elif this_element_name == "Pr":
            this_z = 59
        elif this_element_name == "Nd":
            this_z = 60
        elif this_element_name == "Pm":
            this_z = 61
        elif this_element_name == "Sm":
            this_z = 62
        elif this_element_name == "Eu":
            this_z = 63
        elif this_element_name == "Gd":
            this_z = 64
        elif this_element_name == "Tb":
            this_z = 65
        elif this_element_name == "Dy":
            this_z = 66
        elif this_element_name == "Ho":
            this_z = 67
        elif this_element_name == "Er":
            this_z = 68
        elif this_element_name == "Tm":
            this_z = 69
        elif this_element_name == "Yb":
            this_z = 70
        elif this_element_name == "Lu":
            this_z = 71
        elif this_element_name == "Hf":
            this_z = 72
        elif this_element_name == "Ta":
            this_z = 73
        elif this_element_name == "W ":
            this_z = 74
        elif this_element_name == "Re":
            this_z = 75
        elif this_element_name == "Os":
            this_z = 76
        elif this_element_name == "Ir":
            this_z = 77
        elif this_element_name == "Pt":
            this_z = 78
        elif this_element_name == "Au":
            this_z = 79
        elif this_element_name == "Hg":
            this_z = 80
        elif this_element_name == "Tl":
            this_z = 81
        elif this_element_name == "Pb":
            this_z = 82
        elif this_element_name == "Bi":
            this_z = 83
        elif this_element_name == "Po":
            this_z = 84
        elif this_element_name == "At":
            this_z = 85
        elif this_element_name == "Rn":
            this_z = 86
        elif this_element_name == "Fr":
            this_z = 87
        elif this_element_name == "Ra":
            this_z = 88
        elif this_element_name == "Ac":
            this_z = 89
        elif this_element_name == "Th":
            this_z = 90
        elif this_element_name == "Pa":
            this_z = 91
        elif this_element_name == "U ":
            this_z = 92
        else:
            this_z = 0

        if this_z == 0:
            print("zcompound is confused: %s", compound_string)
            compound_string = ""
            return np.zeros(0)

        postnum_index = num_start_index
        if len(compound_string) > num_start_index + 1:
            test_char = compound_string[postnum_index]
        else:
            test_char = ""
        while (
            (test_char == "0")
            or (test_char == "1")
            or (test_char == "2")
            or (test_char == "2")
            or (test_char == "3")
            or (test_char == "4")
            or (test_char == "5")
            or (test_char == "6")
            or (test_char == "7")
            or (test_char == "8")
            or (test_char == "9")
            or (test_char == ".")
        ) and (postnum_index <= last_char_index):
            postnum_index = postnum_index + 1
            if postnum_index <= last_char_index:
                test_char = compound_string[postnum_index]
            else:
                test_char = ""

        if num_start_index != postnum_index:
            number_string = compound_string[num_start_index:postnum_index]
            num_multiplier = 1.0
            if verbose:
                print("Trying to interpret %s as a number.", number_string)
            if len(number_string) != 0:
                num_multiplier = float(number_string)
        else:
            num_multiplier = 1.0

        if this_z <= max_z_index:
            z_array[this_z - 1] = z_array[this_z - 1] + num_multiplier
        else:
            print("zcompound: z_array smaller than %s", max_z_index)
            return np.zeros(0)

        remaining_string = compound_string[postnum_index : last_char_index + 1]

        if len(remaining_string) > 0:
            z_array = self.zcompound(remaining_string, z_array, paren_multiplier=True)

        return z_array

    def zatwt(self, z_array):

        maxz = z_array.size
        atwt = 0.0

        for i in range(maxz):
            if z_array[i] != 0.0:
                if i + 1 == 1:
                    this_atwt = 1.00794
                elif i + 1 == 2:
                    this_atwt = 4.0026
                elif i + 1 == 3:
                    this_atwt = 6.941
                elif i + 1 == 4:
                    this_atwt = 9.01218
                elif i + 1 == 5:
                    this_atwt = 10.81
                elif i + 1 == 6:
                    this_atwt = 12.011
                elif i + 1 == 7:
                    this_atwt = 14.0067
                elif i + 1 == 8:
                    this_atwt = 15.9994
                elif i + 1 == 9:
                    this_atwt = 18.9984
                elif i + 1 == 10:
                    this_atwt = 21.179
                elif i + 1 == 11:
                    this_atwt = 22.98977
                elif i + 1 == 12:
                    this_atwt = 24.305
                elif i + 1 == 13:
                    this_atwt = 26.98154
                elif i + 1 == 14:
                    this_atwt = 28.0855
                elif i + 1 == 15:
                    this_atwt = 30.97376
                elif i + 1 == 16:
                    this_atwt = 32.06
                elif i + 1 == 17:
                    this_atwt = 35.453
                elif i + 1 == 18:
                    this_atwt = 39.948
                elif i + 1 == 19:
                    this_atwt = 39.0983
                elif i + 1 == 20:
                    this_atwt = 40.08
                elif i + 1 == 21:
                    this_atwt = 44.9559
                elif i + 1 == 22:
                    this_atwt = 47.88
                elif i + 1 == 23:
                    this_atwt = 50.9415
                elif i + 1 == 24:
                    this_atwt = 51.996
                elif i + 1 == 25:
                    this_atwt = 54.9380
                elif i + 1 == 26:
                    this_atwt = 55.847
                elif i + 1 == 27:
                    this_atwt = 58.9332
                elif i + 1 == 28:
                    this_atwt = 58.69
                elif i + 1 == 29:
                    this_atwt = 63.546
                elif i + 1 == 30:
                    this_atwt = 65.38
                elif i + 1 == 31:
                    this_atwt = 69.72
                elif i + 1 == 32:
                    this_atwt = 72.59
                elif i + 1 == 33:
                    this_atwt = 74.9216
                elif i + 1 == 34:
                    this_atwt = 78.96
                elif i + 1 == 35:
                    this_atwt = 79.904
                elif i + 1 == 36:
                    this_atwt = 83.80
                elif i + 1 == 37:
                    this_atwt = 85.4678
                elif i + 1 == 38:
                    this_atwt = 87.62
                elif i + 1 == 39:
                    this_atwt = 88.9059
                elif i + 1 == 40:
                    this_atwt = 91.22
                elif i + 1 == 41:
                    this_atwt = 92.9064
                elif i + 1 == 42:
                    this_atwt = 95.94
                elif i + 1 == 43:
                    this_atwt = 98.0
                elif i + 1 == 44:
                    this_atwt = 101.07
                elif i + 1 == 45:
                    this_atwt = 102.9055
                elif i + 1 == 46:
                    this_atwt = 106.42
                elif i + 1 == 47:
                    this_atwt = 107.8682
                elif i + 1 == 48:
                    this_atwt = 112.41
                elif i + 1 == 49:
                    this_atwt = 114.82
                elif i + 1 == 50:
                    this_atwt = 118.69
                elif i + 1 == 51:
                    this_atwt = 121.75
                elif i + 1 == 52:
                    this_atwt = 127.60
                elif i + 1 == 53:
                    this_atwt = 126.9054
                elif i + 1 == 54:
                    this_atwt = 131.29
                elif i + 1 == 55:
                    this_atwt = 132.9054
                elif i + 1 == 56:
                    this_atwt = 137.33
                elif i + 1 == 57:
                    this_atwt = 138.9055
                elif i + 1 == 58:
                    this_atwt = 140.12
                elif i + 1 == 59:
                    this_atwt = 140.9077
                elif i + 1 == 60:
                    this_atwt = 144.24
                elif i + 1 == 61:
                    this_atwt = 145.0
                elif i + 1 == 62:
                    this_atwt = 150.36
                elif i + 1 == 63:
                    this_atwt = 151.96
                elif i + 1 == 64:
                    this_atwt = 157.25
                elif i + 1 == 65:
                    this_atwt = 158.9254
                elif i + 1 == 66:
                    this_atwt = 162.5
                elif i + 1 == 67:
                    this_atwt = 164.9304
                elif i + 1 == 68:
                    this_atwt = 167.26
                elif i + 1 == 69:
                    this_atwt = 168.9342
                elif i + 1 == 70:
                    this_atwt = 173.04
                elif i + 1 == 71:
                    this_atwt = 174.967
                elif i + 1 == 72:
                    this_atwt = 178.49
                elif i + 1 == 73:
                    this_atwt = 180.9479
                elif i + 1 == 74:
                    this_atwt = 183.85
                elif i + 1 == 75:
                    this_atwt = 186.207
                elif i + 1 == 76:
                    this_atwt = 190.2
                elif i + 1 == 77:
                    this_atwt = 192.22
                elif i + 1 == 78:
                    this_atwt = 195.08
                elif i + 1 == 79:
                    this_atwt = 196.9665
                elif i + 1 == 80:
                    this_atwt = 200.59
                elif i + 1 == 81:
                    this_atwt = 204.383
                elif i + 1 == 82:
                    this_atwt = 207.2
                elif i + 1 == 83:
                    this_atwt = 208.9804
                elif i + 1 == 84:
                    this_atwt = 209.0
                elif i + 1 == 85:
                    this_atwt = 210.0
                elif i + 1 == 86:
                    this_atwt = 222.0
                elif i + 1 == 87:
                    this_atwt = 223.0
                elif i + 1 == 88:
                    this_atwt = 226.0254
                elif i + 1 == 89:
                    this_atwt = 227.0278
                elif i + 1 == 90:
                    this_atwt = 232.0381
                elif i + 1 == 91:
                    this_atwt = 231.0359
                elif i + 1 == 92:
                    this_atwt = 238.0289
                else:
                    this_atwt = 0.0

                atwt = atwt + z_array[i] * this_atwt

        return atwt

    def extra(self, ielement=-1):

        energies, f1, f2, n_extra, energies_extra, f1_extra, f2_extra = self.read(
            ielement, all=False
        )
        if not n_extra == None and n_extra != 0:
            energies_all = np.concatenate((energies, energies_extra), axis=0)
            f1_all = np.concatenate((f1, f1_extra), axis=0)
            f2_all = np.concatenate((f2, f2_extra), axis=0)
            sort_order = energies_all.argsort()
            energies_all = energies_all[sort_order]
            f1_all = f1_all[sort_order]
            f2_all = f2_all[sort_order]
        else:
            energies_all = energies
            f1_all = f1
            f2_all = f2

        return energies, f1, f2, energies_extra, f1_extra, f2_extra

    def read(self, ielement=-1, all=True):
        if ielement == -1:
            all = True

        verbose = False
        expected_pos = 0

        filename = self.filepath

        try:
            file = open(str(filename), "rb")
        except:
            try:
                filename = reference_directory + element_henke_filename
                file = open(str(filename), "rb")
            except:
                print("Could not open file %s", filename)
                return None, None, None, None, None, None, None

        if verbose:
            print("File: %s", filename)

        buf = file.read()
        u = Unpacker(buf)

        if all:
            n_elements = u.unpack_int()
            n_energies = u.unpack_int()

            if verbose:
                print("n_energies: %s", n_energies)
                print("n_elements: %s", n_elements)
                expected_pos = expected_pos + 2 * 4
                print(
                    "Actual, expected file position before reading energies: %s %s",
                    u.get_position(),
                    expected_pos,
                )

            energies = u.unpack_farray(n_energies, u.unpack_float)
            energies = np.array(energies)
            if verbose:
                print("energies: %s", energies)

            f1 = np.zeros((n_elements, n_energies))
            f2 = np.zeros((n_elements, n_energies))
            this_f1 = np.zeros((n_energies))
            this_f2 = np.zeros((n_energies))

            if verbose:
                expected_pos = expected_pos + 4 * n_energies
                print(
                    "Actual, expected file position before reading elements: %s %s",
                    u.get_position(),
                    expected_pos,
                )

            for i_element in range(n_elements):
                this_f1 = u.unpack_farray(n_energies, u.unpack_float)
                this_f2 = u.unpack_farray(n_energies, u.unpack_float)
                f1[i_element, :] = this_f1
                f2[i_element, :] = this_f2

            if verbose:
                expected_pos = expected_pos + n_elements * n_energies * 2 * 4
                print(
                    "Actual, expected file position before reading n_extra_energies: %s %s",
                    u.get_position(),
                    expected_pos,
                )

            n_extra_energies = u.unpack_int()
            if verbose:
                print("n_extra_energies: %s", n_extra_energies)

            if verbose:
                expected_pos = expected_pos + 4
                print(
                    "Actual, expected file position before reading extras: %s %s",
                    u.get_position(),
                    expected_pos,
                )

            n_extra = np.zeros((n_elements), dtype=int)
            extra_energies = np.zeros((n_elements, n_extra_energies))
            extra_f1 = np.zeros((n_elements, n_extra_energies))
            extra_f2 = np.zeros((n_elements, n_extra_energies))
            this_n_extra = 0
            this_extra_energies = np.zeros((n_extra_energies))
            this_extra_f1 = np.zeros((n_extra_energies))
            this_extra_f2 = np.zeros((n_extra_energies))

            for i_element in range(n_elements):
                this_n_extra = u.unpack_int()
                this_extra_energies = u.unpack_farray(n_extra_energies, u.unpack_float)
                this_extra_f1 = u.unpack_farray(n_extra_energies, u.unpack_float)
                this_extra_f2 = u.unpack_farray(n_extra_energies, u.unpack_float)
                n_extra[i_element] = this_n_extra
                extra_energies[i_element, :] = this_extra_energies
                extra_f1[i_element, :] = this_extra_f1
                extra_f2[i_element, :] = this_extra_f2

        else:
            n_elements = u.unpack_int()
            n_energies = u.unpack_int()

            energies = u.unpack_farray(n_energies, u.unpack_float)
            energies = np.array(energies)
            if verbose:
                print("energies: %s", energies)

            byte_offset = 4 + 4 + 4 * n_energies + 8 * ielement * n_energies
            u.set_position(byte_offset)

            f1 = u.unpack_farray(n_energies, u.unpack_float)
            f2 = u.unpack_farray(n_energies, u.unpack_float)

            byte_offset = 4 + 4 + 4 * n_energies + 8 * n_elements * n_energies
            u.set_position(byte_offset)

            n_extra_energies = u.unpack_int()
            if verbose:
                print("n_extra_energies %s", n_extra_energies)

            byte_offset = (
                4
                + 4
                + 4 * n_energies
                + 8 * n_elements * n_energies
                + 4
                + ielement * (4 + 12 * n_extra_energies)
            )
            u.set_position(byte_offset)

            n_extra = u.unpack_int()
            this_extra_energies = u.unpack_farray(n_extra_energies, u.unpack_float)
            this_extra_f1 = u.unpack_farray(n_extra_energies, u.unpack_float)
            this_extra_f2 = u.unpack_farray(n_extra_energies, u.unpack_float)

            extra_energies = this_extra_energies[0:n_extra]
            extra_f1 = this_extra_f1[0:n_extra]
            extra_f2 = this_extra_f2[0:n_extra]

        file.close()

        return energies, f1, f2, n_extra, extra_energies, extra_f1, extra_f2

    def array(self, compound_name, density, graze_mrad=0):

        z_array = []
        z_array, atwt = self.compound(compound_name)
        if len(z_array) == 0:
            z_array = self.zcompound(compound_name, z_array)
            atwt = self.zatwt(z_array)

        maxz = 92
        first_time = 1
        for i in range(maxz):
            if z_array[i] != 0.0:
                (
                    energies,
                    this_f1,
                    this_f2,
                    n_extra,
                    extra_energies,
                    extra_f1,
                    extra_f2,
                ) = self.read(ielement=i)
                if energies == None:
                    continue
                print("this_f1.shape: %s", this_f1.shape)
                if first_time == 1:
                    f1 = z_array[i] * this_f1
                    f2 = z_array[i] * this_f2
                    first_time = 0
                else:
                    f1 = f1 + z_array[i] * this_f1
                    f2 = f2 + z_array[i] * this_f2

        num_energies = len(energies)
        AVOGADRO = 6.02204531e23
        HC_ANGSTROMS = 12398.52
        RE = 2.817938070e-13  # in cm

        if atwt != 0.0:
            molecules_per_cc = density * AVOGADRO / atwt
        else:
            molecules_per_cc = 0.0

        wavelength_angstroms = HC_ANGSTROMS / energies
        constant = (
            RE
            * (1.0e-16 * wavelength_angstroms * wavelength_angstroms)
            * molecules_per_cc
            / (2.0 * np.pi)
        )
        delta = constant * f1
        beta = constant * f2
        alpha = 1.0e4 * density * AVOGADRO * RE / (2.0 * np.pi * atwt)

        if graze_mrad == 0.0:
            reflect = np.ones((num_energies))
        else:
            theta = 1.0e-3 * graze_mrad
            sinth = np.sin(theta)
            sinth2 = sinth * sinth
            coscot = np.cos(theta)
            coscot = coscot * coscot / sinth
            alpha = 2.0 * delta - delta * delta + beta * beta
            gamma = 2.0 * (1.0 - delta) * beta
            rhosq = 0.5 * (
                sinth2
                - alpha
                + np.sqrt((sinth2 - alpha) * (sinth2 - alpha) + gamma * gamma)
            )
            rho = np.sqrt(rhosq)
            i_sigma = (4.0 * rhosq * (sinth - rho) * (sinth - rho) + gamma * gamma) / (
                4.0 * rhosq * (sinth + rho) * (sinth + rho) + gamma * gamma
            )
            piosig = (4.0 * rhosq * (rho - coscot) * (rho - coscot) + gamma * gamma) / (
                4.0 * rhosq * (rho + coscot) * (rho + coscot) + gamma * gamma
            )
            reflect = 50.0 * i_sigma * (1 + piosig)

        denom = energies * 4.0 * np.pi * beta

        zeroes = np.where(denom == 0.0)
        nonzeroes = np.where(denom != 0.0)
        denom[zeroes] = 1e-8

        inverse_mu = np.array((len(energies)))

        inverse_mu = 1.239852 / denom
        if len(zeroes) > 0:
            inverse_mu[zeroes] = np.inf

        return (
            energies,
            f1,
            f2,
            delta,
            beta,
            graze_mrad,
            reflect,
            inverse_mu,
            atwt,
            alpha,
        )

    def get_henke(self, compound_name, density, energy):
        if len(compound_name) == 0:
            print(
                "henke, compound_name, density, energy, f1, f2, delta, beta, graze_mrad, reflect, inverse_mu=inverse_mu inverse_mu is 1/e absorption length in microns. atwt is the atom-averaged atomic weight for the compound"
            )
            return None, None, None, None, None, None, None, None

        (
            enarr,
            f1arr,
            f2arr,
            deltaarr,
            betaarr,
            graze_mrad,
            reflect_arr,
            inverse_mu,
            atwt,
            alpha,
        ) = self.array(compound_name, density)

        num_energies = len(enarr)

        high_index = 0
        while (energy > enarr[high_index]) and (high_index < (num_energies - 1)):
            high_index = high_index + 1

        if high_index == 0:
            high_index = 1
        low_index = high_index - 1

        ln_lower_energy = np.log(enarr[low_index])
        ln_higher_energy = np.log(enarr[high_index])
        fraction = (np.log(energy) - ln_lower_energy) / (
            ln_higher_energy - ln_lower_energy
        )

        f1_lower = f1arr[low_index]
        f1_higher = f1arr[high_index]
        f1 = f1_lower + fraction * (f1_higher - f1_lower)

        ln_f2_lower = np.log(np.abs(f2arr(low_index)))
        ln_f2_higher = np.log(np.abs(f2arr(high_index)))
        f2 = np.exp(ln_f2_lower + fraction * (ln_f2_higher - ln_f2_lower))

        delta_lower = deltaarr[low_index]
        delta_higher = deltaarr[high_index]
        delta = delta_lower + fraction * (delta_higher - delta_lower)

        ln_beta_lower = np.log(np.abs(betaarr(low_index)))
        ln_beta_higher = np.log(np.abs(betaarr(high_index)))
        beta = np.exp(ln_beta_lower + fraction * (ln_beta_higher - ln_beta_lower))

        reflect_lower = reflect_arr[low_index]
        reflect_higher = reflect_arr[high_index]
        reflect = reflect_lower + fraction * (reflect_higher - reflect_lower)

        if beta != 0.0:
            inverse_mu = 1.239852 / (energy * 4.0 * np.pi * beta)
        else:
            inverse_mu = np.Inf

        return f1, f2, delta, beta, graze_mrad, reflect, inverse_mu, atwt

    def get_henke_single(self, name, density, energy_array):
        AVOGADRO = 6.02204531e23
        HC_ANGSTROMS = 12398.52
        RE = 2.817938070e-13  # in cm

        z_array, atwt = self.compound(name.strip(), density)
        if len(z_array) == 0:
            z_array = self.zcompound(name, z_array)
            atwt = self.zatwt(z_array)

        wo = np.where(z_array > 0)[0]

        if len(wo) == 0:
            print(
                "Warning: get_henke_single() name=%s encountered error, will return",
                name,
            )
            return 0, 0, 0, 0

        z = wo + 1
        if atwt != 0.0:
            molecules_per_cc = density * AVOGADRO / atwt
        else:
            molecules_per_cc = 0.0

        if len(wo) > 1:
            energies_all, f1_all, f2_all, energies_extra, f1_extra, f2_extra = (
                self.extra(ielement=z[0])
            )
        else:
            energies_all, f1_all, f2_all, energies_extra, f1_extra, f2_extra = (
                self.extra(ielement=z[0] - 1)
            )

        if isinstance(energy_array, float):
            n_array = 1
        else:
            n_array = len(energy_array)
        f1_array = np.zeros((n_array))
        f2_array = np.zeros((n_array))
        delta_array = np.zeros((n_array))
        beta_array = np.zeros((n_array))

        for i in range(n_array):
            energy = energy_array
            wavelength_angstroms = HC_ANGSTROMS / energy
            constant = (
                RE
                * (1.0e-16 * wavelength_angstroms * wavelength_angstroms)
                * molecules_per_cc
                / (2.0 * np.pi)
            )

            wo = np.where(energies_all > energy)[0]
            if len(wo) == 0:
                hi_e_ind = 0
            else:
                hi_e_ind = wo[0]

            wo = np.where(energies_all < energy)[0]
            if len(wo) == 0:
                lo_e_ind = len(energies_all) - 1
            else:
                lo_e_ind = wo[-1]

            ln_lower_energy = np.log(energies_all[lo_e_ind])
            ln_higher_energy = np.log(energies_all[hi_e_ind])
            fraction = (np.log(energy) - ln_lower_energy) / (
                ln_higher_energy - ln_lower_energy
            )

            f1_lower = f1_all[lo_e_ind]
            f1_higher = f1_all[hi_e_ind]
            f1_array[i] = f1_lower + fraction * (f1_higher - f1_lower)

            ln_f2_lower = np.log(np.abs(f2_all[lo_e_ind]))
            ln_f2_higher = np.log(np.abs(f2_all[hi_e_ind]))
            f2_array[i] = np.exp(ln_f2_lower + fraction * (ln_f2_higher - ln_f2_lower))

            delta_array[i] = constant * f1_array[i]
            beta_array[i] = constant * f2_array[i]

        return f1_array, f2_array, delta_array, beta_array


henkedata = Henke()


class ElementInfo:
    def __init__(self):
        self.z = 0
        self.name = ""
        self.xrf = {
            "Ka1": 0.0,
            "Ka2": 0.0,
            "Kb1": 0.0,
            "Kb2": 0.0,
            "La1": 0.0,
            "La2": 0.0,
            "Lb1": 0.0,
            "Lb2": 0.0,
            "Lb3": 0.0,
            "Lb4": 0.0,
            "Lb5": 0.0,
            "Lg1": 0.0,
            "Lg2": 0.0,
            "Lg3": 0.0,
            "Lg4": 0.0,
            "Ll": 0.0,
            "Ln": 0.0,
            "Ma1": 0.0,
            "Ma2": 0.0,
            "Mb": 0.0,
            "Mg": 0.0,
        }
        self.xrf_abs_yield = {
            "Ka1": 0.0,
            "Ka2": 0.0,
            "Kb1": 0.0,
            "Kb2": 0.0,
            "La1": 0.0,
            "La2": 0.0,
            "Lb1": 0.0,
            "Lb2": 0.0,
            "Lb3": 0.0,
            "Lb4": 0.0,
            "Lb5": 0.0,
            "Lg1": 0.0,
            "Lg2": 0.0,
            "Lg3": 0.0,
            "Lg4": 0.0,
            "Ll": 0.0,
            "Ln": 0.0,
            "Ma1": 0.0,
            "Ma2": 0.0,
            "Mb": 0.0,
            "Mg": 0.0,
        }
        self.yieldD = {"k": 0.0, "l1": 0.0, "l2": 0.0, "l3": 0.0, "m": 0.0}
        self.density = 1.0
        self.mass = 1.0
        self.bindingE = {
            "K": 0.0,
            "L1": 0.0,
            "L2": 0.0,
            "L3": 0.0,
            "M1": 0.0,
            "M2": 0.0,
            "M3": 0.0,
            "M4": 0.0,
            "M5": 0.0,
            "N1": 0.0,
            "N2": 0.0,
            "N3": 0.0,
            "N4": 0.0,
            "N5": 0.0,
            "N6": 0.0,
            "N7": 0.0,
            "O1": 0.0,
            "O2": 0.0,
            "O3": 0.0,
            "O4": 0.0,
            "O5": 0.0,
            "P1": 0.0,
            "P2": 0.0,
            "P3": 0.0,
        }
        self.jump = {
            "K": 0.0,
            "L1": 0.0,
            "L2": 0.0,
            "L3": 0.0,
            "M1": 0.0,
            "M2": 0.0,
            "M3": 0.0,
            "M4": 0.0,
            "M5": 0.0,
            "N1": 0.0,
            "N2": 0.0,
            "N3": 0.0,
            "N4": 0.0,
            "N5": 0.0,
            "O1": 0.0,
            "O2": 0.0,
            "O3": 0.0,
        }

    def __repr__(self) -> str:
        return f"Element {self.name} with Z={self.z} and mass={self.mass} g/mol"


class MapsElements:
    def __init__(self):
        pass

    def get_element_info(self, filepath=None):
        nels = 100

        els_file = (
            reference_directory + element_csv_filename if filepath is None else filepath
        )

        try:
            f = open(els_file, "r")
            csvf = csv.reader(f, delimiter=",")
        except:
            try:
                els_file = "../reference/xrf_library.csv"
                f = open(els_file, "r")
                csvf = csv.reader(f, delimiter=",")
            except:
                print(
                    "Error: Could not find xrf_library.csv file! Please get the library file (e.g., from runtime maps at http://www.stefan.vogt.net/downloads.html) and make sure it is in the Python path"
                )
                return None

        version = 0.0
        for row in csvf:
            if row[0] == "version:":
                version = float(row[1])
                break
        if version != 1.2:
            print(
                "Warning: the only xrf_library.csv file that was found is out of date.  Please use the latest file. A copy can be downloaded, e.g., as part of the runtime maps release available at http://www.stefan.vogt.net/downloads.html"
            )

        element = []
        for i in range(nels):
            element.append(ElementInfo())

        for row in csvf:
            if (
                (row[0] == "version:")
                or (row[0] == "")
                or (row[0] == "aprrox intensity")
                or (row[0] == "transition")
                or (row[0] == "Z")
            ):
                continue

            i = int(row[0]) - 1

            element[i].z = int(float(row[0]))
            element[i].name = row[1]
            element[i].xrf["ka1"] = float(row[2])
            element[i].xrf["ka2"] = float(row[3])
            element[i].xrf["kb1"] = float(row[4])
            element[i].xrf["kb2"] = float(row[5])
            element[i].xrf["la1"] = float(row[6])
            element[i].xrf["la2"] = float(row[7])
            element[i].xrf["lb1"] = float(row[8])
            element[i].xrf["lb2"] = float(row[9])
            element[i].xrf["lb3"] = float(row[10])
            element[i].xrf["lb4"] = float(row[11])
            element[i].xrf["lg1"] = float(row[12])
            element[i].xrf["lg2"] = float(row[13])
            element[i].xrf["lg3"] = float(row[14])
            element[i].xrf["lg4"] = float(row[15])
            element[i].xrf["ll"] = float(row[16])
            element[i].xrf["ln"] = float(row[17])
            element[i].xrf["ma1"] = float(row[18])
            element[i].xrf["ma2"] = float(row[19])
            element[i].xrf["mb"] = float(row[20])
            element[i].xrf["mg"] = float(row[21])
            element[i].yieldD["k"] = float(row[22])
            element[i].yieldD["l1"] = float(row[23])
            element[i].yieldD["l2"] = float(row[24])
            element[i].yieldD["l3"] = float(row[25])
            element[i].yieldD["m"] = float(row[26])
            element[i].xrf_abs_yield["ka1"] = float(row[27])
            element[i].xrf_abs_yield["ka2"] = float(row[28])
            element[i].xrf_abs_yield["kb1"] = float(row[29])
            element[i].xrf_abs_yield["kb2"] = float(row[30])
            element[i].xrf_abs_yield["la1"] = float(row[31])
            element[i].xrf_abs_yield["la2"] = float(row[32])
            element[i].xrf_abs_yield["lb1"] = float(row[33])
            element[i].xrf_abs_yield["lb2"] = float(row[34])
            element[i].xrf_abs_yield["lb3"] = float(row[35])
            element[i].xrf_abs_yield["lb4"] = float(row[36])
            element[i].xrf_abs_yield["lg1"] = float(row[37])
            element[i].xrf_abs_yield["lg2"] = float(row[38])
            element[i].xrf_abs_yield["lg3"] = float(row[39])
            element[i].xrf_abs_yield["lg4"] = float(row[40])
            element[i].xrf_abs_yield["ll"] = float(row[41])
            element[i].xrf_abs_yield["ln"] = float(row[42])
            element[i].xrf_abs_yield["ma1"] = float(row[43])
            element[i].xrf_abs_yield["ma2"] = float(row[44])
            element[i].xrf_abs_yield["mb"] = float(row[45])
            element[i].xrf_abs_yield["mg"] = float(row[46])

            if len(row) > 46:
                element[i].density = float(row[47])
                element[i].mass = float(row[48])

                element[i].bindingE["K"] = float(row[49])

                element[i].bindingE["L1"] = float(row[50])
                element[i].bindingE["L2"] = float(row[51])
                element[i].bindingE["L3"] = float(row[52])

                element[i].bindingE["M1"] = float(row[53])
                element[i].bindingE["M2"] = float(row[54])
                element[i].bindingE["M3"] = float(row[55])
                element[i].bindingE["M4"] = float(row[56])
                element[i].bindingE["M5"] = float(row[57])

                element[i].bindingE["N1"] = float(row[58])
                element[i].bindingE["N2"] = float(row[59])
                element[i].bindingE["N3"] = float(row[60])
                element[i].bindingE["N4"] = float(row[61])
                element[i].bindingE["N5"] = float(row[62])
                element[i].bindingE["N6"] = float(row[63])
                element[i].bindingE["N7"] = float(row[64])

                element[i].bindingE["O1"] = float(row[65])
                element[i].bindingE["O2"] = float(row[66])
                element[i].bindingE["O3"] = float(row[67])
                element[i].bindingE["O4"] = float(row[68])
                element[i].bindingE["O5"] = float(row[69])

                element[i].bindingE["P1"] = float(row[70])
                element[i].bindingE["P2"] = float(row[71])
                element[i].bindingE["P3"] = float(row[72])

                element[i].jump["K"] = float(row[73])

                element[i].jump["L1"] = float(row[74])
                element[i].jump["L2"] = float(row[75])
                element[i].jump["L3"] = float(row[76])

                element[i].jump["M1"] = float(row[77])
                element[i].jump["M2"] = float(row[78])
                element[i].jump["M3"] = float(row[79])
                element[i].jump["M4"] = float(row[80])
                element[i].jump["M5"] = float(row[81])

                element[i].jump["N1"] = float(row[82])
                element[i].jump["N2"] = float(row[83])
                element[i].jump["N3"] = float(row[84])
                element[i].jump["N4"] = float(row[85])
                element[i].jump["N5"] = float(row[86])

                element[i].jump["O1"] = float(row[87])
                element[i].jump["O2"] = float(row[88])
                element[i].jump["O3"] = float(row[89])

        f.close()

        return element


class EnergyStruct:
    def __init__(self):
        self.name = ""
        self.energy = 0.0
        self.ratio = 0.0
        self.Ge_mu = 0.0
        self.Si_mu = 0.0
        self.mu_fraction = 0.0
        self.width_multi = 1.0
        self.type = 0
        self.ptype = ""
        self.is_pileup = False

    def check_binding_energy(self, info_elements, coherent_sct_energy):
        if self.type == 0:
            return False
        elif self.is_pileup:
            return True
        else:
            elname = self.name.split("_")[0]
            e_info = next((e for e in info_elements if e.name == elname), None)
            if e_info is None:
                print(f"Error: Could not find element {elname}")
                return False
            else:
                if self.ptype.startswith("K"):
                    return e_info.bindingE["K"] < coherent_sct_energy
                elif self.ptype.startswith("L"):
                    if self.ptype in ["Lb3", "Lb4", "Lg2", "Lg3", "Lg4"]:
                        return e_info.bindingE["L1"] < coherent_sct_energy
                    elif self.ptype in ["Lb1", "Lg1", "Ln"]:
                        return e_info.bindingE["L2"] < coherent_sct_energy
                    elif self.ptype in ["La1", "La2", "Lb2", "Ll"]:
                        return e_info.bindingE["L3"] < coherent_sct_energy
                    else:
                        print(f"Error: Unknown L shell type {self.ptype}")
                elif self.ptype.startswith("M"):
                    return e_info.bindingE["M1"] < coherent_sct_energy
                return False

    def __repr__(self) -> str:
        return f"{self.name} {self.ptype} {self.energy} {self.ratio} {self.mu_fraction} {self.width_multi}"


def calculate_mu_fractions(pars, info_elements, calc_range):
    for k in range(2):
        if k == 0:
            name = "Ge"
            density = 5.323

        if k == 1:
            name = "Si"
            density = 2.33

        z_array, _ = henkedata.compound(name)
        wo = np.where(z_array == 1.0)
        if 1.0 not in z_array:
            print("encountered error, will return")

        z = wo[0][0] + 1

        energies_all, _, f2_all, _, _, _ = henkedata.extra(ielement=z - 1)

        iter_range = (
            range(calc_range)
            if isinstance(calc_range, int)
            or isinstance(calc_range, np.int64)
            or isinstance(calc_range, np.int32)
            or isinstance(calc_range, np.int16)
            or isinstance(calc_range, np.int8)
            else range(calc_range[0], calc_range[1])
        )
        for i in iter_range:
            for j in range(12):
                energy = pars[i, j].energy * 1000.0
                if energy == 0.0:
                    continue
                wavelength_angstroms = HC_ANGSTROMS / energy
                element_mass = 0.0
                elname = pars[i, 0].name.split("_")[0]
                for e in range(len(info_elements)):
                    if info_elements[e].name == elname:
                        element_mass = info_elements[e].mass
                if element_mass == 0.0:
                    pass
                else:
                    molecules_per_cc = density * AVOGADRO / element_mass
                constant = (
                    RE
                    * (1.0e-16 * wavelength_angstroms * wavelength_angstroms)
                    * molecules_per_cc
                    / (2.0 * np.pi)
                )

                wo = np.where(energies_all > energy)
                if wo[0].size > 0:
                    hi_e_ind = wo[0][0]
                else:
                    hi_e_ind = 0

                wo = np.where(energies_all < energy)
                if wo[0].size > 0:
                    lo_e_ind = wo[0][-1]
                else:
                    lo_e_ind = len(energies_all) - 1
                ln_lower_energy = np.log(energies_all[lo_e_ind])
                ln_higher_energy = np.log(energies_all[hi_e_ind])
                fraction = (np.log(energy) - ln_lower_energy) / (
                    ln_higher_energy - ln_lower_energy
                )
                ln_f2_lower = np.log(np.abs(f2_all[lo_e_ind]))
                ln_f2_higher = np.log(np.abs(f2_all[hi_e_ind]))
                f2 = np.exp(ln_f2_lower + fraction * (ln_f2_higher - ln_f2_lower))
                beta = constant * f2
                if k == 0:
                    pars[i, j].Ge_mu = (
                        (energy * 4.0 * np.pi * beta) / (5.323 * 1.239852) * 10000.0
                    )
                if k == 1:
                    pars[i, j].Si_mu = (
                        (energy * 4.0 * np.pi * beta) / (2.33 * 1.239852) * 10000.0
                    )
                # by default use Silicon detector, if not, need to change in override file
                pars[i, j].mu_fraction = pars[i, j].Si_mu


def define_pileup_constants(info_elements, pileups, return_dict=False):
    pileup_pars = np.empty((len(pileups), 12), dtype=object)
    for i in range(len(pileups)):
        for j in range(12):
            pileup_pars[i, j] = EnergyStruct()
    st = "".join([" "] * 32)
    pileup_name = np.array([st] * len(pileups))

    for i, pileup in enumerate(pileups):
        pileup_name[i] = pileup
        str_list = pileup.split("_")
        e_1, e_2 = "", ""
        if len(str_list) == 2:
            e_1, e_2 = str_list
        elif len(str_list) == 3:
            if str_list[1] == "L" or str_list[1] == "M":
                e_1, e_2 = str_list[0], str_list[2]
            elif str_list[2] == "L" or str_list[2] == "M":
                e_1, e_2 = str_list[0], str_list[1]
            else:
                print(
                    f"Pileup {pileup} skipped. Currently only A_B pileups are supported."
                )
                continue
        elif len(str_list) == 4:
            if (str_list[1] == "L" or str_list[1] == "M") and (
                str_list[3] == "L" or str_list[3] == "M"
            ):
                e_1, e_2 = str_list[0], str_list[2]
            else:
                print(
                    f"Pileup {pileup} skipped. Currently only A_B pileups are supported."
                )
                continue
        else:
            print(f"Pileup {pileup} skipped. Currently only A_B pileups are supported.")
            continue

        if len(e_1) > 0 and len(e_2) > 0:
            for j in range(len(info_elements)):
                if info_elements[j].name == e_1:
                    for k in range(len(info_elements)):
                        if info_elements[k].name == e_2:
                            for l in range(12):
                                pileup_pars[i, l].name = pileup

                            pileup_pars[i, 0].energy = (
                                info_elements[j].xrf["ka1"]
                                + info_elements[k].xrf["ka1"]
                            )
                            pileup_pars[i, 1].energy = (
                                info_elements[j].xrf["ka1"]
                                + info_elements[k].xrf["kb1"]
                            )
                            pileup_pars[i, 2].energy = (
                                info_elements[j].xrf["kb1"]
                                + info_elements[k].xrf["kb1"]
                            )
                            pileup_pars[i, 0].ratio = 1.0
                            pileup_pars[i, 1].ratio = (
                                info_elements[k].xrf_abs_yield["kb1"]
                                / info_elements[k].xrf_abs_yield["ka1"]
                            )
                            pileup_pars[i, 2].ratio = (
                                info_elements[j].xrf_abs_yield["kb1"]
                                / info_elements[j].xrf_abs_yield["ka1"]
                            ) * pileup_pars[i, 1].ratio
                            pileup_pars[i, 0].type = 1
                            pileup_pars[i, 1].type = 2
                            pileup_pars[i, 2].type = 2
                            pileup_pars[i, 0].is_pileup = True
                            pileup_pars[i, 1].is_pileup = True
                            pileup_pars[i, 2].is_pileup = True
                            if e_1 != e_2:
                                pileup_pars[i, 3].energy = (
                                    info_elements[j].xrf["kb1"]
                                    + info_elements[k].xrf["ka1"]
                                )
                                pileup_pars[i, 3].ratio = (
                                    info_elements[j].xrf_abs_yield["kb1"]
                                    / info_elements[j].xrf_abs_yield["ka1"]
                                )
                                pileup_pars[i, 3].type = 2
                                pileup_pars[i, 3].is_pileup = True
                            break
                    else:
                        continue
                    break
    calculate_mu_fractions(pileup_pars, info_elements, len(pileups))
    if return_dict:
        return {
            pileup_pars[i, 0].name: pileup_pars[i] for i in range(pileup_pars.shape[0])
        }
    else:
        return pileup_pars, pileup_name


def define_constants(info_elements, pileups=None, return_dict=True):
    npars = np.amax(mele_pos) - np.amin(kele_pos) + 1
    e_pars = np.empty((npars, 12), dtype=object)
    for i in range(npars):
        for j in range(12):
            e_pars[i, j] = EnergyStruct()

    st = "".join([" "] * 32)
    s_name = np.array([st] * (npars + 1))

    s_name[kele_pos] = kele
    for i in range(len(kele_pos)):
        for j in range(len(info_elements)):
            if info_elements[j].name == kele[i]:
                for l in range(12):
                    e_pars[i, l].name = kele[i]
                e_pars[i, 0].energy = info_elements[j].xrf["ka1"]
                e_pars[i, 1].energy = info_elements[j].xrf["ka2"]
                e_pars[i, 2].energy = info_elements[j].xrf["kb1"]
                e_pars[i, 3].energy = info_elements[j].xrf["kb2"]

                e_pars[i, 0].ratio = 1.0
                e_pars[i, 1].ratio = (
                    info_elements[j].xrf_abs_yield["ka2"]
                    / info_elements[j].xrf_abs_yield["ka1"]
                )
                e_pars[i, 2].ratio = (
                    info_elements[j].xrf_abs_yield["kb1"]
                    / info_elements[j].xrf_abs_yield["ka1"]
                )
                e_pars[i, 3].ratio = (
                    info_elements[j].xrf_abs_yield["kb2"]
                    / info_elements[j].xrf_abs_yield["ka1"]
                )

                e_pars[i, 0].type = 1
                e_pars[i, 1].type = 1
                e_pars[i, 2].type = 2
                e_pars[i, 3].type = 2

                e_pars[i, 0].ptype = "Ka1"
                e_pars[i, 1].ptype = "Ka2"
                e_pars[i, 2].ptype = "Kb1"
                e_pars[i, 3].ptype = "Kb2"

    s_name[lele_pos] = lele
    for i in range(len(lele_pos)):
        ii = i + np.amax(kele_pos) - np.amin(kele_pos) + 1
        for j in range(len(info_elements)):
            elname = lele[i]
            elname = elname[:-2]
            if info_elements[j].name == elname:
                for l in range(12):
                    e_pars[ii, l].name = lele[i]

                e_pars[ii, 0].energy = info_elements[j].xrf["la1"]
                e_pars[ii, 1].energy = info_elements[j].xrf["la2"]
                e_pars[ii, 2].energy = info_elements[j].xrf["lb1"]
                e_pars[ii, 3].energy = info_elements[j].xrf["lb2"]
                e_pars[ii, 4].energy = info_elements[j].xrf["lb3"]
                e_pars[ii, 5].energy = info_elements[j].xrf["lb4"]
                e_pars[ii, 6].energy = info_elements[j].xrf["lg1"]
                e_pars[ii, 7].energy = info_elements[j].xrf["lg2"]
                e_pars[ii, 8].energy = info_elements[j].xrf["lg3"]
                e_pars[ii, 9].energy = info_elements[j].xrf["lg4"]
                e_pars[ii, 10].energy = info_elements[j].xrf["ll"]
                e_pars[ii, 11].energy = info_elements[j].xrf["ln"]

                e_pars[ii, 0].ratio = 1.0
                e_pars[ii, 1].ratio = (
                    info_elements[j].xrf_abs_yield["la2"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 2].ratio = (
                    info_elements[j].xrf_abs_yield["lb1"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 3].ratio = (
                    info_elements[j].xrf_abs_yield["lb2"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 4].ratio = (
                    info_elements[j].xrf_abs_yield["lb3"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 5].ratio = (
                    info_elements[j].xrf_abs_yield["lb4"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 6].ratio = (
                    info_elements[j].xrf_abs_yield["lg1"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 7].ratio = (
                    info_elements[j].xrf_abs_yield["lg2"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 8].ratio = (
                    info_elements[j].xrf_abs_yield["lg3"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 9].ratio = (
                    info_elements[j].xrf_abs_yield["lg4"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 10].ratio = (
                    info_elements[j].xrf_abs_yield["ll"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )
                e_pars[ii, 11].ratio = (
                    info_elements[j].xrf_abs_yield["ln"]
                    / info_elements[j].xrf_abs_yield["la1"]
                )

                for t in range(12):
                    e_pars[ii, t].type = 3

                e_pars[ii, 0].ptype = "La1"
                e_pars[ii, 1].ptype = "La2"
                e_pars[ii, 2].ptype = "Lb1"
                e_pars[ii, 3].ptype = "Lb2"
                e_pars[ii, 4].ptype = "Lb3"
                e_pars[ii, 5].ptype = "Lb4"
                e_pars[ii, 6].ptype = "Lg1"
                e_pars[ii, 7].ptype = "Lg2"
                e_pars[ii, 8].ptype = "Lg3"
                e_pars[ii, 9].ptype = "Lg4"
                e_pars[ii, 10].ptype = "Ll"
                e_pars[ii, 11].ptype = "Ln"

    s_name[mele_pos] = mele
    for i in range(len(mele_pos)):
        ii = i + np.amax(lele_pos) - np.amin(kele_pos) + 1
        for j in range(len(info_elements)):
            elname = mele[i]
            elname = elname[:-2]
            if info_elements[j].name == elname:
                for l in range(12):
                    e_pars[ii, l].name = mele[i]
                e_pars[ii, 0].energy = info_elements[j].xrf["ma1"]
                e_pars[ii, 1].energy = info_elements[j].xrf["ma2"]
                e_pars[ii, 2].energy = info_elements[j].xrf["mb"]
                e_pars[ii, 3].energy = info_elements[j].xrf["mg"]

                e_pars[ii, 0].ratio = 1.0
                e_pars[ii, 1].ratio = (
                    info_elements[j].xrf_abs_yield["ma2"]
                    / info_elements[j].xrf_abs_yield["ma1"]
                    if info_elements[j].xrf_abs_yield["ma2"] != 0.0
                    else 1.0
                )
                e_pars[ii, 2].ratio = (
                    info_elements[j].xrf_abs_yield["mb"]
                    / info_elements[j].xrf_abs_yield["ma1"]
                    if info_elements[j].xrf_abs_yield["mb"] != 0.0
                    else 1.0
                )
                e_pars[ii, 3].ratio = (
                    info_elements[j].xrf_abs_yield["mg"]
                    / info_elements[j].xrf_abs_yield["ma1"]
                    if info_elements[j].xrf_abs_yield["mg"] != 0.0
                    else 1.0
                )

                e_pars[ii, 0].type = 7
                e_pars[ii, 1].type = 7
                e_pars[ii, 2].type = 7
                e_pars[ii, 3].type = 7

                e_pars[ii, 0].ptype = "Ma1"
                e_pars[ii, 1].ptype = "Ma2"
                e_pars[ii, 2].ptype = "Mb"
                e_pars[ii, 3].ptype = "Mg"

    calculate_mu_fractions(
        e_pars, info_elements, np.amax(mele_pos) - np.amin(kele_pos) + 1
    )

    if pileups is not None:
        pileup_pars, pileup_name = define_pileup_constants(info_elements, pileups)
        e_pars = np.vstack((e_pars, pileup_pars))
        s_name = np.append(s_name, pileup_name)

    if return_dict:
        return {e_pars[i, 0].name: e_pars[i] for i in range(e_pars.shape[0])}
    else:
        return e_pars, s_name[1:]


def read_constants(filename, info_elements, pileups=None, return_dict=True):
    e_pars, s_name = define_constants(info_elements, pileups, False)
    f = open(str(filename), "rt")
    if f == None:
        return None, None, None
    for line in f:
        if ":" in line:
            slist = line.split(":")
            tag = slist[0]
            value = "".join(slist[1:])

            if tag == "ELEMENTS_WITH_PILEUP":
                pileups = value.split(",")
                pileups = [x.strip() for x in pileups]
                pileup_pars, pileup_name = define_pileup_constants(
                    info_elements, pileups
                )
                e_pars = np.vstack((e_pars, pileup_pars))
                s_name = np.append(s_name, pileup_name)

            elif tag == "DETECTOR_MATERIAL":
                temp = int(value)
                parsdims = e_pars.shape
                if temp == 1:
                    for i in range(parsdims[0]):
                        for j in range(parsdims[1]):
                            e_pars[i, j].mu_fraction = e_pars[i, j].Si_mu
                else:
                    for i in range(parsdims[0]):
                        for j in range(parsdims[1]):
                            e_pars[i, j].mu_fraction = e_pars[i, j].Ge_mu

            elif tag == "BRANCHING_FAMILY_ADJUSTMENT_L":
                temp_string = value.split(",")
                temp_string = [x.strip() for x in temp_string]
                wo = np.where(s_name == temp_string[0])

                if (wo[0].size > 0) and (len(temp_string) == 4):
                    ii = wo[0][0] - np.amin(kele_pos) + 1
                    factor_l1 = float(temp_string[1])
                    factor_l2 = float(temp_string[2])
                    factor_l3 = float(temp_string[3])
                    name = temp_string[0].strip()
                    el_names = [x.name for x in info_elements]
                    if name[:-2] in el_names:
                        j = el_names.index(name[:-2])
                    else:
                        j = -1

                    if j > 0:
                        e_pars[ii, 0].ratio = 1.0
                        e_pars[ii, 1].ratio = (
                            info_elements[j].xrf_abs_yield["la2"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 2].ratio = (
                            info_elements[j].xrf_abs_yield["lb1"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 3].ratio = (
                            info_elements[j].xrf_abs_yield["lb2"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 4].ratio = (
                            info_elements[j].xrf_abs_yield["lb3"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 5].ratio = (
                            info_elements[j].xrf_abs_yield["lb4"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 6].ratio = (
                            info_elements[j].xrf_abs_yield["lg1"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 7].ratio = (
                            info_elements[j].xrf_abs_yield["lg2"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 8].ratio = (
                            info_elements[j].xrf_abs_yield["lg3"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 9].ratio = (
                            info_elements[j].xrf_abs_yield["lg4"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 10].ratio = (
                            info_elements[j].xrf_abs_yield["ll"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 11].ratio = (
                            info_elements[j].xrf_abs_yield["ln"]
                            / info_elements[j].xrf_abs_yield["la1"]
                        )
                        e_pars[ii, 0].ratio *= factor_l3
                        e_pars[ii, 1].ratio *= factor_l3
                        e_pars[ii, 3].ratio *= factor_l3
                        e_pars[ii, 10].ratio *= factor_l3
                        e_pars[ii, 2].ratio *= factor_l2
                        e_pars[ii, 6].ratio *= factor_l2
                        e_pars[ii, 11].ratio *= factor_l2
                        e_pars[ii, 4].ratio *= factor_l1
                        e_pars[ii, 5].ratio *= factor_l1
                        e_pars[ii, 7].ratio *= factor_l1
                        e_pars[ii, 8].ratio *= factor_l1
                        e_pars[ii, 9].ratio *= factor_l1

            elif tag == "BRANCHING_RATIO_ADJUSTMENT_L":
                temp_string = value.split(",")
                temp_string = [x.strip() for x in temp_string]
                wo = np.where(s_name == temp_string[0])

                if wo[0].size > 0 and len(temp_string) == 13:
                    ii = wo[0][0] - np.amin(kele_pos) + 1
                    name = temp_string[0].strip()
                    el_names = [x.name for x in info_elements]

                    if len(temp_string) >= 13:
                        for jj in range(12):
                            e_pars[ii, jj].ratio *= float(temp_string[(jj + 1)])

            elif tag == "BRANCHING_RATIO_ADJUSTMENT_K":
                temp_string = value.split(",")
                temp_string = [x.strip() for x in temp_string]
                wo = np.where(s_name == temp_string[0])

                if (wo[0].size > 0) and (len(temp_string) == 5):
                    ii = wo[0][0] - np.amin(kele_pos) + 1
                    name = temp_string[0].strip()
                    el_names = [x.name for x in info_elements]
                    if name[:-2] in el_names:
                        j = el_names.index(name[:-2])
                    else:
                        j = -1

                    if j > 0:
                        e_pars[ii, 0].ratio = 1.0
                        e_pars[ii, 1].ratio = (
                            info_elements[j].xrf_abs_yield["ka2"]
                            / info_elements[j].xrf_abs_yield["ka1"]
                        )
                        e_pars[ii, 2].ratio = (
                            info_elements[j].xrf_abs_yield["kb1"]
                            / info_elements[j].xrf_abs_yield["ka1"]
                        )
                        e_pars[ii, 3].ratio = (
                            info_elements[j].xrf_abs_yield["kb2"]
                            / info_elements[j].xrf_abs_yield["ka1"]
                        )

                    if len(temp_string) >= 5:
                        for jj in range(4):
                            e_pars[ii, jj].ratio *= float(temp_string[(jj + 1)])

            elif tag == "TAIL_FRACTION_ADJUST_SI":
                temp = float(value)
                wo = np.where(s_name == "Si")
                ii = wo[0][0] - np.amin(kele_pos) + 1
                e_pars[ii, :].mu_fraction = value * e_pars[ii, :].mu_fraction

            elif tag == "TAIL_WIDTH_ADJUST_SI":
                temp = float(value)
                wo = np.where(s_name == "Si")
                ii = wo[0][0] - np.amin(kele_pos) + 1
                e_pars[ii, 0:3].width_multi = value

    if return_dict:
        return {e_pars[i, 0].name: e_pars[i] for i in range(e_pars.shape[0])}
    else:
        return e_pars, s_name
