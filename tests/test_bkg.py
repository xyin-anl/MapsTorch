import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mapstorch.bkg import snip_bkg
from mapstorch.io import (
    parse_override_params_file,
    resolve_energy_range,
    write_override_params_file,
)


class SnipBackgroundTests(unittest.TestCase):
    def test_default_matches_legacy_width_and_single_log_is_opt_in(self):
        n_ch = 256
        x = torch.arange(n_ch, dtype=torch.float64)
        baseline = 20.0 + 0.03 * x + 8.0 * torch.exp(-((x - 48.0) / 28.0) ** 2)
        peak = 900.0 * torch.exp(-0.5 * ((x - 160.0) / 18.0) ** 2)
        spec = baseline + peak

        kwargs = {
            "spec": spec,
            "er": (0, n_ch - 1),
            "e_offset": torch.tensor(0.0, dtype=torch.float64),
            "e_slope": torch.tensor(0.01, dtype=torch.float64),
            "e_quad": torch.tensor(0.0, dtype=torch.float64),
            "snip_width": torch.tensor(0.5, dtype=torch.float64),
            "fwhm_offset": torch.tensor(0.08, dtype=torch.float64),
            "fwhm_fanoprime": torch.tensor(1.8e-4, dtype=torch.float64),
            "device": "cpu",
        }

        default_bkg = snip_bkg(**kwargs)
        double_log_bkg = snip_bkg(**kwargs, use_single_log=False)
        single_log_bkg = snip_bkg(**kwargs, use_single_log=True)
        legacy_bkg = snip_bkg(**kwargs, use_fwhm_width=False)
        fwhm_width_bkg = snip_bkg(**kwargs, use_fwhm_width=True)
        tensor_single_log_bkg = snip_bkg(
            **kwargs, use_single_log=torch.tensor(1.0, dtype=torch.float64)
        )

        torch.testing.assert_close(default_bkg, double_log_bkg)
        torch.testing.assert_close(default_bkg, legacy_bkg)
        torch.testing.assert_close(single_log_bkg, tensor_single_log_bkg)
        self.assertLess(default_bkg[160].item(), single_log_bkg[160].item())
        self.assertGreater(fwhm_width_bkg[160].item(), default_bkg[160].item())

    def test_full_spectrum_mode_matches_full_background_slice(self):
        n_ch = 256
        x = torch.arange(n_ch, dtype=torch.float64)
        spec = (
            30.0
            + 0.04 * x
            + 700.0 * torch.exp(-0.5 * ((x - 48.0) / 11.0) ** 2)
            + 500.0 * torch.exp(-0.5 * ((x - 205.0) / 17.0) ** 2)
        )
        er = (60, 180)
        kwargs = {
            "er": er,
            "e_offset": torch.tensor(0.0, dtype=torch.float64),
            "e_slope": torch.tensor(0.01, dtype=torch.float64),
            "e_quad": torch.tensor(0.0, dtype=torch.float64),
            "snip_width": torch.tensor(0.5, dtype=torch.float64),
            "fwhm_offset": torch.tensor(0.08, dtype=torch.float64),
            "fwhm_fanoprime": torch.tensor(1.8e-4, dtype=torch.float64),
            "device": "cpu",
        }

        roi_bkg = snip_bkg(spec=spec[er[0] : er[1] + 1], **kwargs)
        full_bkg = snip_bkg(
            spec=spec,
            use_full_spectrum=True,
            return_full=True,
            **kwargs,
        )
        full_slice_bkg = snip_bkg(
            spec=spec,
            use_full_spectrum=True,
            **kwargs,
        )

        torch.testing.assert_close(full_bkg[er[0] : er[1] + 1], full_slice_bkg)
        self.assertGreater(
            torch.max(torch.abs(full_slice_bkg - roi_bkg)).item(),
            1e-3,
        )


class OverrideParamsTests(unittest.TestCase):
    def test_snip_options_round_trip_through_override_file(self):
        with tempfile.TemporaryDirectory(prefix="mapstorch_bkg_tests_") as tmpdir:
            override_path = Path(tmpdir) / "override.txt"
            write_override_params_file(
                str(override_path),
                param_values={
                    "SNIP_SINGLE_LOG": 1,
                    "SNIP_FWHM_WIDTH": 1,
                    "SNIP_FULL_SPECTRUM": 0,
                    "FWHM_FANO_COEFF": 3.58,
                },
                elements=["Fe"],
            )

            contents = override_path.read_text(encoding="utf-8")
            self.assertIn("SNIP_USE_SINGLE_LOG: 1", contents)
            self.assertIn("SNIP_USE_FWHM_WIDTH: 1", contents)
            self.assertIn("SNIP_USE_FULL_SPECTRUM: 0", contents)
            self.assertIn("FWHM_FANO_COEFF: 3.58", contents)

            parsed = parse_override_params_file(str(override_path))
            self.assertEqual(parsed["SNIP_USE_SINGLE_LOG"], 1.0)
            self.assertEqual(parsed["SNIP_USE_FWHM_WIDTH"], 1.0)
            self.assertEqual(parsed["SNIP_USE_FULL_SPECTRUM"], 0.0)
            self.assertEqual(parsed["FWHM_FANO_COEFF"], 3.58)


class EnergyRangeTests(unittest.TestCase):
    def test_override_energy_range_uses_energy_axis(self):
        energy_axis = np.linspace(0.0, 20.0, 201)
        override = {"MIN_ENERGY_TO_FIT": 1.5, "MAX_ENERGY_TO_FIT": 10.2}
        self.assertEqual(
            resolve_energy_range(
                None,
                n_channels=energy_axis.size,
                energy_values=energy_axis,
                override_params=override,
            ),
            (15, 101),
        )

    def test_explicit_energy_range_takes_precedence(self):
        energy_axis = np.linspace(0.0, 20.0, 201)
        override = {"MIN_ENERGY_TO_FIT": 1.5, "MAX_ENERGY_TO_FIT": 10.2}
        self.assertEqual(
            resolve_energy_range(
                (7, 21),
                n_channels=energy_axis.size,
                energy_values=energy_axis,
                override_params=override,
            ),
            (7, 21),
        )


if __name__ == "__main__":
    unittest.main()
