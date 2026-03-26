import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from mapstorch.batchfit import decode_names
from mapstorch.io import create_dataset, write_override_params_file
from scripts.fit_spec_vol_batch import main as fit_spec_vol_batch_main
from scripts.plot_batchfit_diagnostics import main as plot_batchfit_diagnostics_main
from scripts.wrap_amp_npz_as_fit_output import main as wrap_amp_npz_as_fit_output_main


class BatchFitScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="mapstorch_batchfit_tests_")
        self.tmp_path = Path(self.tmpdir.name)
        self.override_path = self.tmp_path / "override.txt"
        write_override_params_file(str(self.override_path), elements=["Fe"])

    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_dataset(self, spectra: np.ndarray, *, fit_elems: list[str] | None = None) -> Path:
        dataset_path = self.tmp_path / "input.h5"
        create_dataset(
            spectra.astype(np.float32, copy=False),
            energy_dim=0,
            output_path=str(dataset_path),
            fit_elems=fit_elems or ["Fe"],
        )
        return dataset_path

    def test_fit_spec_vol_batch_writes_expected_layout(self):
        spectra = np.random.default_rng(0).poisson(5, size=(48, 2, 3)).astype(np.float32)
        dataset_path = self._write_dataset(spectra)
        output_path = self.tmp_path / "batch_fit.h5"

        fit_spec_vol_batch_main(
            SimpleNamespace(
                dataset=str(dataset_path),
                override_params_file=str(self.override_path),
                output_path=str(output_path),
                spec_vol_path=None,
                elem_path=None,
                energy_dim=-1,
                energy_range=[5, 24],
                tile_shape=[1, 2],
                per_spectrum_params=["ENERGY_OFFSET", "ENERGY_SLOPE"],
                fit_params=["ENERGY_OFFSET", "ENERGY_SLOPE"],
                device="cpu",
                optimizer="adamw",
                loss="mse",
                n_iter=0,
                no_init_amp=False,
                no_snip=False,
                detector_material="Si",
                escape_factor=0.0,
                max_tiles=None,
                overwrite=True,
                progress=False,
                verbose=False,
            )
        )

        with h5py.File(output_path, "r") as h5:
            amp_names = decode_names(h5["amp_names"][:])
            self.assertEqual(h5["amp_maps_log10"].shape, (2, 3, len(amp_names)))
            self.assertEqual(h5["param_maps"]["ENERGY_OFFSET"].shape, (2, 3))
            self.assertEqual(h5["tile_metrics"]["final_loss"].shape, (2, 2))
            self.assertIn("Fe", amp_names)
            self.assertIn("COMPTON_AMPLITUDE", amp_names)
            self.assertIn("COHERENT_SCT_AMPLITUDE", amp_names)
            self.assertTrue(bool(h5.attrs["use_snip"]))
            self.assertEqual(tuple(int(x) for x in h5.attrs["tile_shape"]), (1, 2))

    def test_wrap_decodes_byte_strings_and_diagnostics_respects_no_snip(self):
        spectra = np.arange(1, 1 + 32 * 2 * 3, dtype=np.float32).reshape(32, 2, 3)
        dataset_path = self._write_dataset(spectra)
        amp_npz_path = self.tmp_path / "amps.npz"
        wrapped_output_path = self.tmp_path / "wrapped.h5"
        diagnostics_dir = self.tmp_path / "diagnostics"
        energy_range = np.array([4, 11], dtype=np.int32)

        np.savez_compressed(
            amp_npz_path,
            amp_maps=np.full((2, 3, 3), -30.0, dtype=np.float32),
            amp_names=np.array(
                [b"Fe", b"COHERENT_SCT_AMPLITUDE", b"COMPTON_AMPLITUDE"],
                dtype=object,
            ),
            dataset=np.bytes_(str(dataset_path)),
            spec_vol_key=np.bytes_("MAPS/mca_arr"),
            energy_dim=np.array(-1, dtype=np.int32),
            energy_range=energy_range,
            return_log10=np.array(True),
        )

        wrap_amp_npz_as_fit_output_main(
            SimpleNamespace(
                input_path=str(amp_npz_path),
                override_params_file=str(self.override_path),
                output_path=str(wrapped_output_path),
                tile_shape=[1, 3],
                detector_material="Si",
                escape_factor=0.0,
                no_snip=True,
                amp_eps=1e-30,
                overwrite=True,
            )
        )

        with h5py.File(wrapped_output_path, "r") as h5:
            amp_names = decode_names(h5["amp_names"][:])
            self.assertEqual(
                amp_names,
                ["Fe", "COHERENT_SCT_AMPLITUDE", "COMPTON_AMPLITUDE"],
            )
            self.assertFalse(bool(h5.attrs["use_snip"]))

        plot_batchfit_diagnostics_main(
            SimpleNamespace(
                input_path=str(wrapped_output_path),
                output_dir=str(diagnostics_dir),
                device="cpu",
                n_pixel_spectra=2,
                min_pixel_separation=0.0,
                progress=False,
            )
        )

        metrics = np.load(diagnostics_dir / "diagnostics_metrics.npz", allow_pickle=True)
        expected_sse_map = np.sum(spectra[energy_range[0] : energy_range[1] + 1] ** 2, axis=0)
        np.testing.assert_allclose(
            metrics["sse_map"],
            expected_sse_map.astype(np.float32),
            rtol=1e-5,
            atol=1e-3,
        )


if __name__ == "__main__":
    unittest.main()
