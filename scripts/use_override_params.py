from mapstorch.io import read_dataset, parse_override_params_file
from mapstorch.default import default_elem_info
from mapstorch.constant import read_constants
from mapstorch.opt import fit_spec


dataset = "img.dat.20250505/2idd_0038.mda.h50"
override_params_file = "maps_fit_parameters_override.txt"
energy_range = (50, 1000)

# Read spectrum
dataset_dict = read_dataset(dataset)
int_spec_og = dataset_dict["int_spec"]

# Preprocess spectrum
int_spec = int_spec_og[energy_range[0] : energy_range[1] + 1]

# Read params
override_params = parse_override_params_file(override_params_file)

# Read constants
e_consts = read_constants(override_params_file, default_elem_info)

# Fit the spectrum using specified parameters
fitted_tensors, fitted_spec, fitted_bkg, _ = fit_spec(
    int_spec_og,
    energy_range,
    elements_to_fit=override_params["elements_to_fit"]
    + override_params["elements_with_pileup"],
    init_param_vals=override_params,
    tune_params=True,
    e_consts=e_consts,
)