import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

os.environ.setdefault("MPLBACKEND", "Agg")

from mapstorch.default import default_param_vals
from mapstorch.io import create_dataset, write_override_params_file
from scripts.fit_spec import main as fit_spec_main


class FitSpecScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="mapstorch_fit_spec_tests_")
        self.tmp_path = Path(self.tmpdir.name)
        self.dataset_path = self.tmp_path / "sample.mda.h50"
        self.override_path = self.tmp_path / "override.txt"

        spectra = np.random.default_rng(0).poisson(5, size=(1, 1, 64)).astype(np.float32)
        create_dataset(spectra, energy_dim=2, output_path=str(self.dataset_path), fit_elems=["Fe"])
        write_override_params_file(str(self.override_path), elements=["Fe"])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_output_prefix_and_peak_pos_include_scatter_components(self):
        captured = {}

        def fake_fit_spec(*args, **kwargs):
            tensors = {
                "Fe": torch.tensor(3.0),
                "COHERENT_SCT_AMPLITUDE": torch.tensor(2.0),
                "COMPTON_AMPLITUDE": torch.tensor(1.5),
            }
            for key, value in default_param_vals.items():
                tensors.setdefault(key, torch.tensor(value, dtype=torch.float32))
            n_ch = int(kwargs.get("energy_range", args[1])[1] - kwargs.get("energy_range", args[1])[0] + 1)
            fitted_spec = np.zeros(n_ch, dtype=np.float32)
            fitted_bkg = np.zeros(n_ch, dtype=np.float32)
            return tensors, fitted_spec, fitted_bkg, [0.0]

        def fake_plot_elem_peak_pos(*args, **kwargs):
            captured["target_elems"] = list(kwargs["target_elems"])
            captured["filename"] = kwargs["filename"]
            Path(kwargs["filename"]).touch()
            return None, {}

        plot_dir = self.tmp_path / "plots"
        args = SimpleNamespace(
            dataset=str(self.dataset_path),
            incident_energy=12.0,
            detector_material="Si",
            escape_factor=0.0,
            n_iter=0,
            verbose=False,
            output_prefix=None,
            int_spec_path="MAPS/int_spec",
            elem_path="MAPS/channel_names",
            energy_range=[0, 63],
            init_amp=False,
            use_snip=False,
            loss_selection="mse",
            device_selection="cpu",
            save_params=True,
            params_output_path=None,
            plot_output_dir=str(plot_dir),
            disable_plots=False,
            disable_heuristics=True,
            plots=["elem_peak_pos"],
            override_params_file=str(self.override_path),
        )

        cwd = os.getcwd()
        os.chdir(self.tmp_path)
        try:
            with patch("scripts.fit_spec.fit_spec", side_effect=fake_fit_spec):
                with patch(
                    "scripts.fit_spec.plot_elem_peak_pos",
                    side_effect=fake_plot_elem_peak_pos,
                ):
                    fit_spec_main(args)
        finally:
            os.chdir(cwd)

        self.assertEqual(
            captured["target_elems"],
            ["Fe", "COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"],
        )
        self.assertEqual(
            Path(captured["filename"]).name,
            "sample.mda_element_peak_pos.png",
        )
        self.assertTrue(
            (self.tmp_path / "sample.mda_maps_fit_parameters_override_mapstorch.txt").exists()
        )


if __name__ == "__main__":
    unittest.main()
