import unittest

import numpy as np
import torch

from mapstorch import default, map as model
from mapstorch.opt import _ordered_unique, fit_spec


class OptHelperTests(unittest.TestCase):
    def test_ordered_unique_preserves_first_seen_order(self):
        self.assertEqual(
            _ordered_unique(["Fe", "Pb_L", "Fe", "COMPTON_AMPLITUDE", "Pb_L", "Au_M"]),
            ["Fe", "Pb_L", "COMPTON_AMPLITUDE", "Au_M"],
        )


def _fit_quadratic(log_amp, n_iter=30):
    energy_range = (50, 1400)
    params = {
        key: torch.tensor(value, dtype=torch.float32)
        for key, value in default.default_param_vals.items()
    }
    for name, offset in [("Fe", 0.0), ("COMPTON_AMPLITUDE", -0.4), ("COHERENT_SCT_AMPLITUDE", -0.7)]:
        params[name] = torch.tensor(log_amp + offset)
    spectrum = np.zeros(2048, dtype=np.float32)
    spectrum[: energy_range[1] + 1] = model.model_spec(
        params, (0, energy_range[1]), elements_to_fit=["Fe"]
    ).numpy()
    fitted, _, _, losses = fit_spec(
        spectrum,
        energy_range,
        elements_to_fit=["Fe"],
        fitting_params=["ENERGY_QUADRATIC"],
        init_param_vals=dict(default.default_param_vals, ENERGY_QUADRATIC=0.0),
        init_amp=True,
        use_snip=False,
        optimizer="adam",
        n_iter=n_iter,
        progress_bar=False,
        device="cpu",
    )
    return fitted["ENERGY_QUADRATIC"].item(), losses


class AdamBrightSpectrumTests(unittest.TestCase):
    def test_bright_spectrum_does_not_freeze_parameters(self):
        """A 100x brighter spectrum takes the same Adam steps.

        Fails when the squared-gradient average overflows float32 (gradients
        near 1e21 here), which freezes ENERGY_QUADRATIC at its start value.
        """
        dim_quad, dim_losses = _fit_quadratic(7.0)
        bright_quad, bright_losses = _fit_quadratic(9.0)
        self.assertNotEqual(bright_quad, 0.0)
        np.testing.assert_allclose(bright_quad, dim_quad, rtol=1e-3)
        np.testing.assert_allclose(
            np.asarray(bright_losses) / 1e4, dim_losses, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main()


class ProjectionTests(unittest.TestCase):
    def test_projection_holds_a_refined_setting_inside_its_limits(self):
        """A spectrum made with the energy offset 0.05 keV above the start,
        refined with the offset limited to +-0.01 keV, ends on the limit and the
        returned model is the one at the projected setting.

        Fails when the hook is not called after each step (the offset runs to
        0.05) or when the returned model predates the last projection."""
        energy_range = (50, 1400)
        true = dict(default.default_param_vals)
        start = true["ENERGY_OFFSET"]
        true["ENERGY_OFFSET"] = start + 0.05
        params = {k: torch.tensor(v, dtype=torch.float32) for k, v in true.items()}
        for name, log_amp in [("Fe", 6.0), ("COMPTON_AMPLITUDE", 5.6), ("COHERENT_SCT_AMPLITUDE", 5.3)]:
            params[name] = torch.tensor(log_amp)
        spectrum = np.zeros(2048, dtype=np.float32)
        spectrum[: energy_range[1] + 1] = model.model_spec(
            params, (0, energy_range[1]), elements_to_fit=["Fe"]
        ).numpy()

        def project(tensors):
            tensors["ENERGY_OFFSET"].clamp_(start - 0.01, start + 0.01)

        fitted, spec_fit, _, _ = fit_spec(
            spectrum,
            energy_range,
            elements_to_fit=["Fe"],
            fitting_params=["ENERGY_OFFSET"],
            init_param_vals=dict(default.default_param_vals),
            init_amp=True,
            use_snip=False,
            loss="poisson_nll",
            n_iter=200,
            progress_bar=False,
            device="cpu",
            project=project,
        )
        self.assertAlmostEqual(fitted["ENERGY_OFFSET"].item(), start + 0.01, places=6)
        with torch.no_grad():
            expected = model.model_spec(
                fitted, energy_range, elements_to_fit=["Fe"]
            ).numpy()
        np.testing.assert_allclose(spec_fit, expected, rtol=1e-4, atol=1e-3)
