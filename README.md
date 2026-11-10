> [!TIP]
> Play with a demo dataset on [Hugging Space](https://huggingface.co/spaces/shawnyin/MapsTorch)

<p align="center">
  <img src="assets/logo.png" alt="MapsTorch Logo" width="400"/>
</p>

## MapsTorch

MapsTorch is a differentiable modeling package for automating X-ray fluorescence (XRF) analysis. It combines the physics-based [MAPS](https://www.aps.anl.gov/Microscopy/Software-and-Tools-MAPS) model with PyTorch's automatic differentiation framework.

Key features:

- Parameter optimization for XRF spectrum fitting
- Element selection decision support
- Integration with existing [XRF-Maps](https://github.com/xyin-anl/XRF-Maps) workflows

See [paper](https://doi.org/10.1107/S160057752501032X) for technical details.

## Get Started

Python 3.10 or newer is required. The core engine supports NumPy 2 and needs
only NumPy, PyTorch, h5py, and tqdm:

```sh
pip install mapstorch
```

Optional tools are available as extras:

```sh
pip install 'mapstorch[plot]'    # Matplotlib plots and SciencePlots styles
pip install 'mapstorch[widget]'  # Notebook widgets
pip install 'mapstorch[apps]'    # Apps and scripts, including plots and widgets
pip install 'mapstorch[all]'     # All optional tools
```

**Develop with Pixi**

We keep the development environment reproducible with [Pixi](https://pixi.sh/). Once Pixi is installed:

1. Resolve the environment: `pixi install`
2. Enter the shell: `pixi shell`
3. Install MapsTorch in editable mode: `pixi run install`

After that you can run examples or scripts in the Pixi shell, e.g. `python scripts/fit_spec.py -h` or `marimo run apps/guess_elements.py`.

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

## Line Ratios

Each element's lines enter the model with fixed ratios to its strongest line. By default these come
from the built-in line table, the one XRF-Maps uses, times the multipliers in the settings file
(`BRANCHING_RATIO_ADJUSTMENT_K`, `_L`, `_M` and `BRANCHING_FAMILY_ADJUSTMENT_L`), as XRF-Maps
applies them. The table's L-line ratios can be far from measured ones. To take every L family's
ratios from xraylib at the beam energy instead, pass the energy to `read_constants`:

```python
from mapstorch.constant import read_constants
from mapstorch.default import default_elem_info

e_consts = read_constants("maps_fit_parameters_override.txt", default_elem_info, incident_energy=13.6)
```

The settings file's L multipliers were tuned against the table, so they are skipped on xraylib's
ratios (with a warning) unless `keep_l_adjustments=True`; K and M lines keep the table and the
multipliers. Lines the beam cannot excite get ratio 0, and each excited L family also gets the
L3-N1 (Lb6) line.

To reproduce a MAPS fit, use the settings MAPS used. XRF-Maps records them in the analysed file
(`MAPS/Fit_Parameters_Override`), but some versions record no line-ratio group there; those fits
used the multipliers in the folder's `maps_fit_parameters_override.txt`, which then belong in the
settings you pass.

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

If you have an override parameters file:
```
python fit_spec.py DATASET.h5 -e 12.0 --override_params_file parameters_override.txt
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
