> [!TIP]
> Play with a demo dataset on [Hugging Space](https://huggingface.co/spaces/shawnyin/MapsTorch)

<p align="center">
  <img src="assets/logo.png" alt="MapsTorch Logo" width="400"/>
</p>

## MapsTorch

MapsTorch is a differentiable modeling package for automating X-ray fluorescence (XRF) analysis. It combines the physics-based [MAPS](https://www.aps.anl.gov/Microscopy/Software-and-Tools-MAPS) model with PyTorch's automatic differentiation framework.

Key features:

- Automated parameter optimization for XRF spectrum fitting
- Automatic element detection and decision support
- GPU acceleration support for spectrum volume fitting
- Integration with existing [XRF-Maps](https://github.com/xyin-anl/XRF-Maps) workflows

See [extended abstract](https://academic.oup.com/mam/article/30/Supplement_1/ozae044.1017/7720325) from M&M 2024.

## Get Started

Latest development version:

```
pip install git+https://github.com/xyin-anl/MapsTorch.git
```

Latest release version:

```
pip install mapstorch
```

_Optional:_ For GPU acceleration of spectra volume fitting, install PyTorch with CUDA support following https://pytorch.org/get-started/locally/

## Prepare Data

MapsTorch works with HDF5 files produced by [XRFMaps](https://github.com/xyin-anl/XRF-Maps). The HDF5 file must contain:

- Spectra volume data in `MAPS/mca_arr` group
- Integrated spectra array in `MAPS/int_spec` group

For numpy array data, use the `create_dataset` function in `mapstorch.io`:

```python
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
```

## Interactive Apps

We provide marimo notebooks in the `apps` folder that can be run as web apps (download the notebooks and run them in a Python environment with `mapstorch` installed):

### Guess elements:

```
marimo run guess_elements.py
```

[![Guessing elements](https://img.youtube.com/vi/dJQiLpy4r-Q/0.jpg)](https://www.youtube.com/watch?v=dJQiLpy4r-Q)

### Optimize parameters:

```
marimo run optimize_parameters.py
```

[![Optimizing parameters](https://img.youtube.com/vi/d8Z2n-97f9Q/0.jpg)](https://www.youtube.com/watch?v=d8Z2n-97f9Q)

## Example Scripts

The `scripts` folder contains example scripts like `fit_spec.py` for fitting integrated spectra. Usage:

```
python fit_spec.py DATASET.h5 -e 12.0
```

For all options:

```
python scripts/fit_spec.py -h
```

## Contact

- Bug reports and feature requests: Open GitHub Issues
- Contributions: Fork repository, create branch, submit pull request
- XRF analysis help: Contact xyin@anl.gov or aglowacki@anl.gov

## Acknowledgement

This research used resources of the Advanced Photon Source, an U.S. Department of Energy (DOE) Office of Science User Facility operated for the DOE Office of Science by Argonne National Laboratory under Contract No. DE-AC02-06CH11357. The authors acknowledge funding support from Argonne LDRD 2023-0049.
