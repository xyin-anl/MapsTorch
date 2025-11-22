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
import warnings
import h5py
import numpy as np
from datetime import datetime
from mapstorch.default import (
    default_param_vals,
    param_name_map,
    default_fitting_elems,
)


def read_dataset(
    file_path, spec_vol_key=None, fit_elem_key=None, int_spec_key=None, dtype=np.float32
):
    spec_vol, fit_elems, int_spec = None, None, None
    with h5py.File(file_path, "r") as f:
        if spec_vol_key is not None:
            try:
                spec_vol = f[spec_vol_key][:].astype(dtype)
            except:
                raise KeyError(
                    "Could not find spectra volume in the dataset with the given key {}".format(
                        spec_vol_key
                    )
                )
        else:
            try:
                spec_vol = f["MAPS/Spectra/mca_arr"][:].astype(dtype)
            except:
                try:
                    spec_vol = f["MAPS/mca_arr"][:].astype(dtype)
                except:
                    warnings.warn("Could not find spectra volume in the dataset")
                    spec_vol = None
        if fit_elem_key is not None:
            try:
                fit_elems = f[fit_elem_key][:]
            except:
                raise KeyError(
                    "Could not find fitting elements in the dataset with the given key {}".format(
                        fit_elem_key
                    )
                )
        else:
            try:
                fit_elems = f["MAPS/channel_names"][:]
            except:
                try:
                    fit_elems = f["MAPS/XRF_Analyzed/Fitted/Channel_Names"][:]
                except:
                    warnings.warn("Could not find fitting elements in the dataset")
                    fit_elems = None
        if int_spec_key is not None:
            try:
                int_spec = f[int_spec_key][:].astype(dtype)
            except:
                raise KeyError(
                    "Could not find integrated spectra in the dataset with the given key {}".format(
                        int_spec_key
                    )
                )
        else:
            try:
                int_spec = f["MAPS/int_spec"][:].astype(dtype)
            except:
                warnings.warn("Could not find integrated spectra in the dataset")
                int_spec = None
    if fit_elems is not None:
        fit_elems = [elem.decode("utf-8") for elem in fit_elems]
        if "COHERENT_SCT_AMPLITUDE" not in fit_elems:
            fit_elems.append("COHERENT_SCT_AMPLITUDE")
        if "COMPTON_AMPLITUDE" not in fit_elems:
            fit_elems.append("COMPTON_AMPLITUDE")
    return {"spec_vol": spec_vol, "elems": fit_elems, "int_spec": int_spec}


def create_dataset(
    spec_vol,
    energy_dim,
    output_path,
    fit_elems=None,
    dtype=np.float32,
    compression="gzip",
    compression_opts=4,
):
    """Create an HDF5 file compatible with read_dataset function.

    Args:
        spec_vol: 3D numpy array or path to .npy file containing spectral volume
        energy_dim: Which dimension (0,1,2) contains the energy channels
        output_path: Path where to save the HDF5 file
        fit_elems: Optional list of element names
        dtype: numpy dtype for data storage (e.g. np.float32, np.float16). Default: np.float32
        compression: Compression filter to use. Options: 'gzip', 'lzf', None. Default: 'gzip'
        compression_opts: Compression settings. For 'gzip', this is the compression level (0-9). Default: 4
    """
    if energy_dim not in [0, 1, 2]:
        raise ValueError("energy_dim must be 0, 1, or 2")

    if isinstance(spec_vol, str):
        try:
            spec_vol = np.load(spec_vol)
        except:
            raise ValueError("Could not load .npy file")
    elif not isinstance(spec_vol, np.ndarray) or spec_vol.ndim != 3:
        raise ValueError("spec_vol must be a 3D numpy array")

    # Move energy channels to last dimension if needed
    if energy_dim != 2:
        spec_vol = np.moveaxis(spec_vol, energy_dim, 2)

    # Calculate integrated spectrum
    int_spec = np.sum(spec_vol, axis=(0, 1))

    # Create HDF5 file
    with h5py.File(output_path, "w") as f:
        # Save spectral volume with compression
        f.create_dataset(
            "MAPS/mca_arr",
            data=spec_vol.astype(dtype),
            compression=compression,
            compression_opts=compression_opts,
        )

        # Save integrated spectrum with compression
        f.create_dataset(
            "MAPS/int_spec",
            data=int_spec.astype(dtype),
            compression=compression,
            compression_opts=compression_opts,
        )

        # Save element names if provided
        if fit_elems is not None:
            # Convert strings to bytes for HDF5 compatibility
            fit_elems_bytes = [elem.encode("utf-8") for elem in fit_elems]
            if "COHERENT_SCT_AMPLITUDE" not in fit_elems:
                fit_elems_bytes.append("COHERENT_SCT_AMPLITUDE".encode("utf-8"))
            if "COMPTON_AMPLITUDE" not in fit_elems:
                fit_elems_bytes.append("COMPTON_AMPLITUDE".encode("utf-8"))
            f.create_dataset("MAPS/channel_names", data=fit_elems_bytes)

    return output_path


def write_override_params_file(
    output_path, param_values=None, elements=None, detector_material="Si", escape_factor=0.0
):
    """Write a maps_fit_parameters_override.txt file with specified parameters and elements.

    Args:
        output_path: Path where to save the override file
        param_values: Dictionary of parameter values to override defaults
        elements: List of element names to fit
        detector_material: Detector material (default: "Si")

    The function will use default values from mapstorch.default for any parameters
    not specified in param_values.
    """
    if detector_material not in ["Si", "Ge"]:
        raise ValueError(f"Detector material {detector_material} not supported")
    detector_material_id = 0 if detector_material == "Ge" else 1
    si_escape_enable = 1 if detector_material == "Si" else 0
    si_escape_factor = escape_factor if detector_material == "Si" else 0.0
    ge_escape_enable = 1 if detector_material == "Ge" else 0
    ge_escape_factor = escape_factor if detector_material == "Ge" else 0.0
    
    # Validate and prepare elements list
    if elements is not None:
        invalid_elements = []
        for elem in elements:
            if elem not in default_fitting_elems:
                invalid_elements.append(elem)
        if invalid_elements:
            raise ValueError(f"Unsupported elements: {invalid_elements}")
    else:
        elements = default_fitting_elems

    # Merge provided params with defaults
    params = default_param_vals.copy()
    if param_values:
        mapped_params = {}
        for key, value in param_values.items():
            mapped_key = param_name_map.get(key, key)
            mapped_params[mapped_key] = value
        params.update(mapped_params)

    with open(output_path, "w") as f:
        # Header section
        f.write(
            "   This file will override default fit settings for the maps program for a 3 element detector\n"
        )
        f.write("   note, the filename MUST be maps_fit_parameters_override.txt\n")
        f.write("VERSION: 5.0\n")
        f.write(f"DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')}\n")
        f.write(
            "   put below the number of detectors that were used to acquire spectra. IMPORTANT:\n"
        )
        f.write("   this MUST come after VERSION, and before all other options!\n")
        f.write("DETECTOR_ELEMENTS:       1\n")
        f.write("   give this file an internal name, whatever you like\n")
        f.write("IDENTIFYING_NAME_[WHATEVERE_YOU_LIKE]: automatic\n")

        # Elements section
        f.write(
            "   list the elements that you want to be fit. For K lines, just use the element\n"
        )
        f.write("   name, for L lines add _L, e.g., Au_L, for M lines add _M\n")
        f.write(f"ELEMENTS_TO_FIT: {', '.join(elements)}\n")
        f.write(
            "   list the element combinations you want to fit for pileup, e.g., Si_Si, Si_Si_Si, Si_Cl, etc\n"
        )
        f.write("ELEMENTS_WITH_PILEUP: \n")

        # Calibration parameters
        f.write("   offset of energy calibration, in kev\n")
        f.write(
            f"CAL_OFFSET_[E_OFFSET]: {params.get('ENERGY_OFFSET', -0.0041840752)}\n"
        )
        f.write("CAL_OFFSET_[E_OFFSET]_MAX: 0.5\n")
        f.write("CAL_OFFSET_[E_OFFSET]_MIN: -0.5\n")
        f.write("   slope of energy calibration, in leV / channel\n")
        f.write(f"CAL_SLOPE_[E_LINEAR]: {params.get('ENERGY_SLOPE', 0.0095216077)}\n")
        f.write("CAL_SLOPE_[E_LINEAR]_MAX: 0.015\n")
        f.write("CAL_SLOPE_[E_LINEAR]_MIN: 0.0079999994\n")
        f.write(
            "   quadratic correction for energy calibration, unless you know exactly what you are doing, please leave it at 0.\n"
        )
        f.write(f"CAL_QUAD_[E_QUADRATIC]: {params.get('ENERGY_QUADRATIC', 0.0)}\n")
        f.write("CAL_QUAD_[E_QUADRATIC]_MAX: 9.9999997e-05\n")
        f.write("CAL_QUAD_[E_QUADRATIC]_MIN: -9.9999997e-05\n")

        # FWHM parameters
        f.write("    energy_resolution at 0keV\n")
        f.write(f"FWHM_OFFSET: {params.get('FWHM_OFFSET', 0.09721764)}\n")
        f.write("    energy dependence of the energy resolution\n")
        f.write(f"FWHM_FANOPRIME: {params.get('FWHM_FANOPRIME', 0.00022440446)}\n")

        # Coherent scattering parameters
        f.write("    incident energy\n")
        f.write(f"COHERENT_SCT_ENERGY: {params.get('COHERENT_SCT_ENERGY', 10.0)}\n")
        f.write("    upper constraint for the incident energy\n")
        f.write("COHERENT_SCT_ENERGY_MAX: 14.599999\n")
        f.write("    lower constraint for the incident energy\n")
        f.write("COHERENT_SCT_ENERGY_MIN: 9.699999\n")

        # Compton parameters
        f.write("    angle for the compton scatter (in degrees)\n")
        f.write(f"COMPTON_ANGLE: {params.get('COMPTON_ANGLE', 87.274598)}\n")
        f.write("COMPTON_ANGLE_MAX: 170.0\n")
        f.write("COMPTON_ANGLE_MIN: 70.0\n")
        f.write("    additional width of the compton\n")
        f.write(f"COMPTON_FWHM_CORR: {params.get('COMPTON_FWHM_CORR', 1.5730172)}\n")
        f.write(f"COMPTON_STEP: {params.get('COMPTON_F_STEP', 0.0)}\n")
        f.write(f"COMPTON_F_TAIL: {params.get('COMPTON_F_TAIL', 0.13308163)}\n")
        f.write(f"COMPTON_GAMMA: {params.get('COMPTON_GAMMA', 3.0)}\n")
        f.write(f"COMPTON_HI_F_TAIL: {params.get('COMPTON_HI_F_TAIL', 0.0039171793)}\n")
        f.write(f"COMPTON_HI_GAMMA: {params.get('COMPTON_HI_GAMMA', 3.0)}\n")

        # Tailing parameters
        f.write(
            "    tailing parameters, see also Grieken, Markowicz, Handbook of X-ray spectrometry\n"
        )
        f.write(
            "    2nd ed, van Espen spectrum evaluation page 287.	_A corresponds to f_S, _B to\n"
        )
        f.write("    f_T and _C to gamma\n")
        f.write(f"STEP_OFFSET: {params.get('F_STEP_OFFSET', 0.0)}\n")
        f.write(f"STEP_LINEAR: {params.get('F_STEP_LINEAR', 0.0)}\n")
        f.write(f"STEP_QUADRATIC: {params.get('F_STEP_QUADRATIC', 0.0)}\n")
        f.write(f"F_TAIL_OFFSET: {params.get('F_TAIL_OFFSET', 0.003)}\n")
        f.write(f"F_TAIL_LINEAR: {params.get('F_TAIL_LINEAR', 1.6940659e-21)}\n")
        f.write(f"F_TAIL_QUADRATIC: {params.get('F_TAIL_QUADRATIC', 0.0)}\n")
        f.write(f"KB_F_TAIL_OFFSET: {params.get('KB_F_TAIL_OFFSET', 0.05)}\n")
        f.write(f"KB_F_TAIL_LINEAR: {params.get('KB_F_TAIL_LINEAR', 0.0)}\n")
        f.write(f"KB_F_TAIL_QUADRATIC: {params.get('KB_F_TAIL_QUADRATIC', 0.0)}\n")
        f.write(f"GAMMA_OFFSET: {params.get('GAMMA_OFFSET', 2.2101209)}\n")
        f.write(f"GAMMA_LINEAR: {params.get('GAMMA_LINEAR', 0.0)}\n")
        f.write(f"GAMMA_QUADRATIC: {params.get('GAMMA_QUADRATIC', 0.0)}\n")

        # Background parameters
        f.write(
            "    snip width is the width used for estimating background. 0.5 is typically a good start \n"
        )
        f.write(f"SNIP_WIDTH: {params.get('SNIP_WIDTH', 0.5)}\n")
        f.write(
            "    set FIT_SNIP_WIDTH to 1 to fit the width of the snipping for background estimate, set to 0 not to. Only use if you know what it is doing!\n"
        )
        f.write("FIT_SNIP_WIDTH: 0\n")

        # Detector parameters
        f.write("    detector material: 0= Germanium, 1 = Si\n")
        f.write(f"DETECTOR_MATERIAL: {detector_material_id}\n")
        f.write("    beryllium window thickness, in micrometers, typically 8 or 24\n")
        f.write("BE_WINDOW_THICKNESS: 0.0\n")
        f.write("thickness of the detector chip, e.g., 350 microns for an SDD\n")
        f.write("DET_CHIP_THICKNESS: 0.0\n")
        f.write(
            "thickness of the Germanium detector dead layer, in microns, for the purposes of the NBS calibration\n"
        )
        f.write("GE_DEAD_LAYER: 0.0\n")

        # Energy range parameters
        f.write("    maximum energy value to fit up to [keV]\n")
        f.write(f"MAX_ENERGY_TO_FIT: {params.get('MAX_ENERGY_TO_FIT', 11.0)}\n")
        f.write("    minimum energy value [keV]\n")
        f.write(f"MIN_ENERGY_TO_FIT: {params.get('MIN_ENERGY_TO_FIT', 1.0)}\n")

        branching_ratio_section = """    this allows manual adjustment of the branhcing ratios between the different lines of L1, L2, and L3.
    note, the numbers that are put in should be RELATIVE modifications, i.e., a 1 will correspond to exactly the literature value,
    0.8 will correspond to to 80% of that, etc.
BRANCHING_FAMILY_ADJUSTMENT_L: Pt_L, 0., 1., 1.
BRANCHING_FAMILY_ADJUSTMENT_L: Gd_L, 1., 1., 1.
BRANCHING_FAMILY_ADJUSTMENT_L: Sn_L, 0., 0., 1.
BRANCHING_FAMILY_ADJUSTMENT_L: I_L, 1., 1., 1.
    this allows manual adjustment of the branhcing ratios between the different L lines, such as La 1, la2, etc.
    Please note, these are all RELATIVE RELATIVE modifications, i.e., a 1 will correspond to exactly the literature value, etc.
    all will be normalized to the La1 line, and the values need to be in the following order:
    La1, La2, Lb1, Lb2, Lb3, Lb4, Lg1, Lg2, Lg3, Lg4, Ll, Ln
    please note, the first value (la1) MUST BE A 1. !!!
BRANCHING_RATIO_ADJUSTMENT_L: Pb_L, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.
BRANCHING_RATIO_ADJUSTMENT_L: I_L, 1., 1., 0.45, 1.0, 0.45, 0.45, 0.6, 1., 0.3, 1., 1., 1.
BRANCHING_RATIO_ADJUSTMENT_L: Gd_L, 1., 0.48, 0.59, 0.98, 0.31, 0.08, 0.636, 1., 0.3, 1., 1., 1.
BRANCHING_RATIO_ADJUSTMENT_L: Sn_L, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    this allows manual adjustment of the branhcing ratios between the different K lines, such as Ka1, Ka2, Kb1, Kb2
    Please note, these are all RELATIVE RELATIVE modifications, i.e., a 1 will correspond to exactly the literature value, etc.
    all will be normalized to the Ka1 line, and the values need to be in the following order:
    Ka1, Ka2, Kb1(+3), Kb2
    please note, the first value (Ka1) MUST BE A 1. !!!
BRANCHING_RATIO_ADJUSTMENT_K: Na, 1., 1., 4.0, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Mg, 1., 1., 3.6, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Al, 1., 1., 3.3, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Si, 1., 1., 2.9, 1.
BRANCHING_RATIO_ADJUSTMENT_K: P, 1., 1., 2.75, 1.
BRANCHING_RATIO_ADJUSTMENT_K: S, 1., 1., 2.6, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Cl, 1., 1., 2.5, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Ar, 1., 1., 2.2, 1.
BRANCHING_RATIO_ADJUSTMENT_K: K, 1., 1., 1.9, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Ca, 1., 1., 1.7, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Ti, 1., 1., 1.6, 1.
BRANCHING_RATIO_ADJUSTMENT_K: V, 1., 1., 1.4, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Cr, 1., 1., 1.35, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Mn, 1., 1., 1.3, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Fe, 1., 1., 1.2, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Co, 1., 1., 1.1, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Ni, 1., 1., 1.05, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Cu, 1., 1., 1.0, 1.
BRANCHING_RATIO_ADJUSTMENT_K: Zn, 1., 1., 1.0, 1.
    the parameter adds the escape peaks (offset) to the fit if larger than 0. You should not enable Si and Ge at the same time, ie, one of these two values should be zero
"""
        f.write(branching_ratio_section)
        f.write(f"SI_ESCAPE_FACTOR: {si_escape_factor}\n")
        f.write(f"GE_ESCAPE_FACTOR: {ge_escape_factor}\n")
        f.write("    this parameter adds a component to the escape peak that depends linear on energy\n")
        f.write("LINEAR_ESCAPE_FACTOR: 0.0\n")
        f.write("    the parameter enables fitting of the escape peak strengths. set 1 to enable, set to 0 to disable. (in matrix fitting always disabled)\n")
        f.write(f"SI_ESCAPE_ENABLE: {si_escape_enable}\n")
        f.write(f"GE_ESCAPE_ENABLE: {ge_escape_enable}\n")
        final_section = """    the lines (if any) below will override the detector names built in to maps. please modify only if you are sure you understand the effect
SRCURRENT: S:SRcurrentAI
US_IC: 2xfm:scaler3_cts1.B
DS_IC: 2xfm:scaler3_cts1.C
DPC1_IC: 2xfm:scaler3_cts2.A
DPC2_IC: 2xfm:scaler3_cts2.B
CFG_1: 2xfm:scaler3_cts3.B
CFG_2: 2xfm:scaler3_cts3.C
CFG_3: 2xfm:scaler3_cts3.D
CFG_4: 2xfm:scaler3_cts4.A
CFG_5: 2xfm:scaler3_cts4.B
CFG_6: 2xfm:scaler3_cts4.C
CFG_7: 2xfm:scaler3_cts4.D
CFG_8: 2xfm:scaler3_cts5.A
CFG_9: 2xfm:scaler3_cts5.B
    These scalers are for fly scans.  Tag: Name;PV
TIME_SCALER_PV: 2xfm:mcs:mca1.VAL
TIME_SCALER_CLOCK: 25000000.0
TIME_NORMALIZED_SCALER: US_IC;2xfm:mcs:mca2.VAL
TIME_NORMALIZED_SCALER: DS_IC;2xfm:mcs:mca3.VAL
TIME_NORMALIZED_SCALER: DPC1_IC;2xfm:mcs:mca5.VAL
TIME_NORMALIZED_SCALER: DPC2_IC;2xfm:mcs:mca6.VAL
TIME_NORMALIZED_SCALER: CFG_1;2xfm:mcs:mca10.VAL
TIME_NORMALIZED_SCALER: CFG_2;2xfm:mcs:mca11.VAL
TIME_NORMALIZED_SCALER: CFG_3;2xfm:mcs:mca12.VAL
TIME_NORMALIZED_SCALER: CFG_4;2xfm:mcs:mca13.VAL
TIME_NORMALIZED_SCALER: CFG_5;2xfm:mcs:mca14.VAL
TIME_NORMALIZED_SCALER: CFG_6;2xfm:mcs:mca15.VAL
TIME_NORMALIZED_SCALER: CFG_7;2xfm:mcs:mca16.VAL
TIME_NORMALIZED_SCALER: CFG_8;2xfm:mcs:mca17.VAL
TIME_NORMALIZED_SCALER: CFG_9;2xfm:mcs:mca18.VAL
    Spectra scalers
ELT1: dxpXMAP2xfm3:mca1.ELTM
ERT1: dxpXMAP2xfm3:mca1.ERTM
ICR1: dxpXMAP2xfm3:dxp1:InputCountRate
OCR1: dxpXMAP2xfm3:dxp1:OutputCountRate
    the lines below (if any) give backup description of IC amplifier sensitivity, in case it cannot be found in the mda file
		 for the amps, the _NUM value should be between 0 and 8 where 0=1, 1=2, 2=5, 3=10, 4=20, 5=50, 6=100, 7=200, 8=500
		 for the amps, the _UNIT value should be between 0 and 3 where 0=pa/v, 1=na/v, 2=ua/v 3=ma/v
    search for the PV first, if not found then use the values below
US_AMP_SENS_NUM_PV: 2xfm:A1sens_num.VAL
US_AMP_SENS_UNIT_PV: 2xfm:A1sens_unit.VAL
DS_AMP_SENS_NUM_PV: 2xfm:A2sens_num.VAL
DS_AMP_SENS_UNIT_PV: 2xfm:A2sens_unit.VAL

US_AMP_SENS_NUM:        5
US_AMP_SENS_UNIT:        1
DS_AMP_SENS_NUM:        1
DS_AMP_SENS_UNIT:        1"""
        f.write(final_section)

    return output_path


def parse_override_params_file(file_path):
    """Parse a maps_fit_parameters_override.txt file and return a dictionary of parameter values.

    Args:
        file_path: Path to the override parameters file

    Returns:
        dict: Dictionary of parameter values in the format used by default_param_vals
    """
    # Initialize dictionary for storing parameter values
    params = {}

    # Parameters to extract with their file keys and corresponding internal names
    param_keys = {k: k for k in default_param_vals.keys()} | param_name_map

    # Additional data that might be extracted
    elements_to_fit = []
    elements_with_pileup = []
    detector_material = "Si"
    si_escape_enable = 0
    ge_escape_enable = 0
    si_escape_factor = 0.0
    ge_escape_factor = 0.0

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                # Skip empty lines and comment lines
                if not line or line.startswith("   "):
                    continue

                # Extract parameter name and value
                if ":" in line:
                    parts = line.split(":", 1)
                    param = parts[0].strip()
                    value = parts[1].strip()

                    # Handle specific parameters
                    if param == "DETECTOR_MATERIAL":
                        detector_material = "Ge" if int(value) == 0 else "Si"
                    elif param == "ELEMENTS_TO_FIT":
                        elements_to_fit = [
                            e.strip() for e in value.split(",") if e.strip()
                        ]
                    elif param == "ELEMENTS_WITH_PILEUP":
                        elements_with_pileup = [
                            e.strip() for e in value.split(",") if e.strip()
                        ]
                    elif param == "SI_ESCAPE_FACTOR":
                        si_escape_factor = float(value)
                    elif param == "GE_ESCAPE_FACTOR":
                        ge_escape_factor = float(value)
                    elif param == "SI_ESCAPE_ENABLE":
                        si_escape_enable = int(value)
                    elif param == "GE_ESCAPE_ENABLE":
                        ge_escape_enable = int(value)

                    # Extract numerical parameters
                    elif param in param_keys:
                        internal_param = param_keys[param]
                        try:
                            # Convert to float
                            params[internal_param] = float(value)
                        except ValueError:
                            # Skip if conversion fails
                            pass
    except Exception as e:
        import warnings

        warnings.warn(f"Error parsing override parameter file: {e}")

    # Add additional extracted data to the returned dictionary
    params["elements_to_fit"] = elements_to_fit
    params["elements_with_pileup"] = elements_with_pileup
    params["detector_material"] = detector_material

    if si_escape_enable == 1 and detector_material == "Si":
        escape_factor = si_escape_factor
    elif ge_escape_enable == 1 and detector_material == "Ge":
        escape_factor = ge_escape_factor
    else:
        escape_factor = 0.0
    params["escape_factor"] = escape_factor

    return params
