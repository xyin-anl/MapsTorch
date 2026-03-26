import unittest

import torch

from mapstorch.default import default_param_vals
from mapstorch.map import build_model_basis, model_spec


def _make_params(dtype=torch.float64):
    params = {}
    for key, value in default_param_vals.items():
        params[key] = torch.tensor(value, dtype=dtype)
    params["ENERGY_OFFSET"] = torch.tensor(0.0, dtype=dtype)
    params["ENERGY_SLOPE"] = torch.tensor(0.01, dtype=dtype)
    params["ENERGY_QUADRATIC"] = torch.tensor(0.0, dtype=dtype)
    params["COHERENT_SCT_ENERGY"] = torch.tensor(12.0, dtype=dtype)
    params["COHERENT_SCT_AMPLITUDE"] = torch.tensor(2.0, dtype=dtype)
    params["COMPTON_AMPLITUDE"] = torch.tensor(2.0, dtype=dtype)
    params["Fe"] = torch.tensor(3.0, dtype=dtype)
    return params


class ModelParityTests(unittest.TestCase):
    def test_model_spec_and_basis_include_tails(self):
        base = _make_params()
        tail_heavy = _make_params()
        tail_heavy["F_TAIL_OFFSET"] = torch.tensor(0.2, dtype=torch.float64)
        tail_heavy["KB_F_TAIL_OFFSET"] = torch.tensor(0.3, dtype=torch.float64)
        tail_heavy["COMPTON_F_TAIL"] = torch.tensor(0.9, dtype=torch.float64)
        tail_heavy["COMPTON_HI_F_TAIL"] = torch.tensor(0.2, dtype=torch.float64)

        spec_base = model_spec(base, (50, 600), elements_to_fit=["Fe"])
        spec_tail = model_spec(tail_heavy, (50, 600), elements_to_fit=["Fe"])
        self.assertGreater(torch.max(torch.abs(spec_tail - spec_base)).item(), 0.0)

        basis_base, _ = build_model_basis(base, (50, 600), elements_to_fit=["Fe"])
        basis_tail, _ = build_model_basis(tail_heavy, (50, 600), elements_to_fit=["Fe"])
        self.assertGreater(torch.max(torch.abs(basis_tail - basis_base)).item(), 0.0)

    def test_tail_gradients_and_resolution_coeff_are_finite(self):
        params = _make_params()
        params["F_TAIL_OFFSET"] = torch.tensor(0.05, dtype=torch.float64, requires_grad=True)
        params["KB_F_TAIL_OFFSET"] = torch.tensor(
            0.15, dtype=torch.float64, requires_grad=True
        )
        params["COMPTON_F_TAIL"] = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        params["COMPTON_HI_F_TAIL"] = torch.tensor(
            0.1, dtype=torch.float64, requires_grad=True
        )
        params["FWHM_FANO_COEFF"] = torch.tensor(
            3.58, dtype=torch.float64, requires_grad=True
        )

        spec = model_spec(params, (50, 600), elements_to_fit=["Fe"])
        loss = spec.sum()
        loss.backward()

        for key in [
            "F_TAIL_OFFSET",
            "KB_F_TAIL_OFFSET",
            "COMPTON_F_TAIL",
            "COMPTON_HI_F_TAIL",
            "FWHM_FANO_COEFF",
        ]:
            grad = params[key].grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.all(torch.isfinite(grad)).item(), key)
            self.assertGreater(torch.max(torch.abs(grad)).item(), 0.0, key)

        coeff_358 = _make_params()
        coeff_296 = _make_params()
        coeff_296["FWHM_FANO_COEFF"] = torch.tensor(2.96, dtype=torch.float64)
        spec_358 = model_spec(coeff_358, (50, 600), elements_to_fit=["Fe"])
        spec_296 = model_spec(coeff_296, (50, 600), elements_to_fit=["Fe"])
        self.assertGreater(torch.max(torch.abs(spec_358 - spec_296)).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
