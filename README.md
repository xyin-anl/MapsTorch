# MapsTorch: Differentiable XRF Spectrum Fitting

X-ray fluorescence (XRF) has been widely utilized to analyze chemical composition and elemental distribution within samples. The process of fitting raw XRF spectra and obtaining quantified insights (i.e., mapping) is at the core of XRF analysis [1]. At the Advanced Photon Source (APS), there are large amount of diverse XRF datasets from different samples, instruments, and experimental conditions. Extensive parameter tuning is needed to obtain best results for each specific dataset. Typically, the fitting parameters such as energy calibration coefficients or gaussian parameters are tuned by beamline scientists based on their past experiences. This reduces overall throughput and potentially introduces subjective biases. We anticipate parameter tuning will become a major bottleneck especially after the APS upgrade, when the data collection rate will be ~100 times higher [2]. Therefore, it is important to develop automated XRF parameter tuning workflows. 

The MAPS algorithm [3] and the XRF-Maps software [4] are currently utilized at the APS to analyze XRF data, requiring users to specify element types and fitting parameters for each experiment. To automatically search for optimal parameters, we present a physics-based and data-efficient differentiable modeling (DM) approach by leveraging automatic differentiation (AD) – a well-known technique in the field of machine learning [5] and has been applied to computational imaging techniques such as ptychography [6]. When solving optimization problems with complex physical models, AD can produce accurate numerical gradients for all inputs efficiently in a programmatic fashion, eliminating the need for deriving closed forms gradients or approximating finite differences. We implemented a differentiable version of the robust physics-based MAPS algorithm using PyTorch [7]. In the forward pass, each potential element’s contribution is calculated and summed to the model spectrum. Then elastic, Compton and escape amplitudes are calculated to produce the model spectrum. The resulting model spectrum is then compared to the experimental spectrum to calculate a loss, which can be backpropagated via AD. Since the amplitudes and all fitting parameters are all free parameters in the DM approach, we can optimize them simultaneously during the fitting process, thus eliminating the need of manual tuning. Besides, the framework can automatically determine which elements present in the sample (i.e., amplitudes are non-zero) from a set of potential candidates. Moreover, with the flexibility of the DM framework, it is possible to design advanced optimization routine to overcome various issues encountered in the current fitting routine and provide better fit to the experimental data. Overall, combining the solid physics foundation of the MAPS algorithm and powerful AD, the DM framework represents a promising route to lift the burden of parameter tuning and can be extended to other spectrum fitting problems such as energy-dispersive X-ray spectroscopy and electron energy loss spectroscopy.

- [1] T Nietzold et al., J. Vis. Exp. 132 (2018), p. e56042. https://doi.org/10.3791/56042
- [2] The APS Upgrade: Building a Brighter Future, https://www.aps.anl.gov/APS-Upgrade (Feb 13, 2024)
- [3] S Vogt. et al., J. Phys. IV France, 104 (2023), p. 617-622, https://doi.org/10.1051/jp4:20030156
- [4] A Glowacki, XRF-Maps, https://doi.org/10.11578/dc.20210824.5.
- [5] A Baydin et al., J. Mach. Learn. Res., 18 (2018), p. 1-43, http://jmlr.org/papers/v18/17-468.html
- [6] M Du et al., Optics Express (2021), p. 10000-100035. https://doi.org/10.1364/OE.418296
- [7] A Paszke et al., NeurIPS (2019), p. 8024–8035. https://doi.org/10.5555/3454287.3455008


## Setup environment
Please download/clone this repository and enter the directory
```
git clone https://github.com/xyin-anl/MapsTorch.git
cd MapsTorch
```

A Python environment is needed. `conda` is recommended

```conda env create -f env.yml ```

## Prepare data
The input files are HDF5 files produced by XRFMaps, it should at least contain the spectra volume information.

## Run apps
* Explore spectra volume: ```marimo run apps/explore_dataset.py```
* Guess elements: ```marimo run apps/guess_elements.py```
* Optimize parameters: ```marimo run apps/optimize_parameters.py```

## Roadmap
- [ ] Advaned optimization example
- [ ] Creating elemental maps example
- [ ] Creating parameter maps example
- [ ] Creating scripting examples

## Contact
If you want to report bug or suggest features/examples, please open new github Issues. If you want to contribute to the package, please fork the repository, create a new branch and then open a pull request. If you need help using the software or analyzing challenging XRF data, please reach out to xyin@anl.gov or aglowacki@anl.gov

## Acknowledgement
This research used resources of the Advanced Photon Source, an U.S. Department of Energy (DOE) Office of Science User Facility operated for the DOE Office of Science by Argonne National Laboratory under Contract No. DE-AC02-06CH11357. The authors acknowledge funding support from Argonne LDRD 2023-0049.