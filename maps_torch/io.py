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
