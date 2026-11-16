import math
import unittest

import torch

from mapstorch.default import default_param_vals
from mapstorch.map import build_model_basis, escape_probability, model_spec

CA_KA, BEAM = 3.6917, 10.52
TRUE = {"P": 1.0, "Ca": 300.0, "Zn": 0.5, "COHERENT_SCT_AMPLITUDE": 400.0, "COMPTON_AMPLITUDE": 600.0}


def _params():
    p = {k: torch.tensor(v, dtype=torch.float64) for k, v in default_param_vals.items()}
    p["ENERGY_OFFSET"] = torch.tensor(0.0, dtype=torch.float64)
    p["ENERGY_SLOPE"] = torch.tensor(0.01, dtype=torch.float64)
    p["ENERGY_QUADRATIC"] = torch.tensor(0.0, dtype=torch.float64)
    p["COHERENT_SCT_ENERGY"] = torch.tensor(BEAM, dtype=torch.float64)
    for name, amp in TRUE.items():  # amplitudes are log10 of the linear ones
        p[name] = torch.tensor(math.log10(amp), dtype=torch.float64)
    return p


class PhysicalEscapeTests(unittest.TestCase):
    def test_a_constant_escape_fitted_at_ca_takes_zn_from_under_the_scatter_peaks(self):
        # The curve: Reed and Ware with xraylib gives 0.907 % at Ca Ka and 0.087 % at 10.52 keV.
        self.assertAlmostEqual(escape_probability(CA_KA), 0.00907, delta=0.0003)
        self.assertAlmostEqual(escape_probability(BEAM), 0.00087, delta=0.00006)

        # A spectrum with escape copies planted by the physics at both energies (model_spec adds them
        # to the summed spectrum) comes back exactly from the per-component basis with the same model...
        params, window, elems = _params(), (100, 1100), ["P", "Ca", "Zn"]
        spec = model_spec(params, window, elements_to_fit=elems, escape_factor=1.0,
                          escape_model="physical").double()
        basis, names = build_model_basis(params, window, elements_to_fit=elems, escape_factor=1.0,
                                         escape_model="physical")
        basis = basis.double()
        amp = torch.linalg.lstsq(basis.T, spec.unsqueeze(1)).solution.squeeze(1)
        for k, name in enumerate(names):
            self.assertAlmostEqual(amp[k].item() / TRUE[name], 1.0, delta=1e-4, msg=name)

        # ...while one fraction at every energy, right at Ca, overstates the scatter peaks' copies on Zn.
        const, _ = build_model_basis(params, window, elements_to_fit=elems,
                                     escape_factor=escape_probability(CA_KA), escape_model="constant")
        amp_c = torch.linalg.lstsq(const.double().T, spec.unsqueeze(1)).solution.squeeze(1)
        self.assertLess(amp_c[names.index("Zn")].item() / TRUE["Zn"], 0.9)


if __name__ == "__main__":
    unittest.main()
