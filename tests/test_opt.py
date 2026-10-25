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
