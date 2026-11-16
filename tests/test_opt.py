import unittest

import numpy as np
import torch

from mapstorch import default, map as model
from mapstorch.constant import define_pileup_constants
from mapstorch.opt import _ordered_unique, fit_spec, solve_mca_amplitudes


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

        Fails when the hook is not called (the offset runs to 0.05) or the
        returned model is far from the one at the projected setting."""
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


class NNLSAmplitudeTests(unittest.TestCase):
    def test_out_of_window_tail_does_not_inflate_nnls_or_shift_in_window_lines(self):
        params = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in default.default_param_vals.items()
        }
        params.update({
            "ENERGY_OFFSET": torch.tensor(0.0),
            "ENERGY_SLOPE": torch.tensor(0.01),
            "ENERGY_QUADRATIC": torch.tensor(0.0),
            "COHERENT_SCT_ENERGY": torch.tensor(10.5),
        })
        consts = dict(default.default_energy_consts)
        consts.update(define_pileup_constants(
            default.default_elem_info, ["Zn_Zn"], return_dict=True
        ))
        elements = ["Zn", "Zn_Zn", "Fe", "Ca"]
        names = elements + ["COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"]
        expected = np.array([
            [10000, 0.5, 20000, 30000, 1000, 2000],
            [30000, 0.5, 10000, 20000, 2000, 1000],
        ], dtype=np.float32)
        volume = np.empty((1, 2, 2048), dtype=np.float32)
        for pixel, amplitudes in enumerate(expected):
            pixel_params = dict(params)
            pixel_params.update({
                name: torch.tensor(float(np.log10(amplitude)))
                for name, amplitude in zip(names, amplitudes)
            })
            volume[0, pixel] = model.model_spec(
                pixel_params, (0, 2047), elements, e_consts=consts
            ).numpy()

        options = dict(
            energy_dim=-1, use_snip=False, amp_method="nnls_pgd",
            ridge_lambda=0.0, return_log10=False, chunk_size=1,
            device="cpu", e_consts=consts, progress_bar=False,
        )
        fitted, fitted_names = solve_mca_amplitudes(
            volume, (100, 1100), params, elements_to_fit=elements, **options
        )
        self.assertEqual(fitted_names, names)
        self.assertEqual(fitted.shape, (1, 2, len(names)))
        self.assertTrue(np.isfinite(fitted).all())
        # XRF-Maps retains this negligible tail at its NNLS starting value,
        # not zero. An inverse-based start instead amplifies roundoff to billions.
        np.testing.assert_allclose(fitted[0], expected, rtol=1e-5, atol=1e-5)

        without_pileup, reduced_names = solve_mca_amplitudes(
            volume, (100, 1100), params,
            elements_to_fit=["Zn", "Fe", "Ca"], **options
        )
        retained = [names.index(name) for name in reduced_names]
        np.testing.assert_allclose(
            fitted[..., retained], without_pileup, rtol=1e-5, atol=1e-5
        )
