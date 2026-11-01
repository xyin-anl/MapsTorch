import copy
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from mapstorch.constant import binding_threshold_for_line, read_constants
from mapstorch.default import default_elem_info, default_param_vals
from mapstorch.map import model_elem_spec

try:
    import xraylib  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    xraylib = None

# Fixed-table spectra written before the L3-N1 line was added
# (python tests/test_l3n1_line.py --write-reference rewrites them).
REFERENCE = os.path.join(os.path.dirname(__file__), "data", "fixed_table_spectra.csv")
ELEMENTS = ["Fe", "La_L", "Pb_L", "Pb_M"]
# Germanium detector, so the test also covers DETECTOR_MATERIAL.
SETTINGS = "ELEMENTS_TO_FIT: " + ", ".join(ELEMENTS) + "\nDETECTOR_MATERIAL: 0\n"


def _constants(**kwargs):
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/maps_fit_parameters_override.txt"
        with open(path, "w") as f:
            f.write(SETTINGS)
        return read_constants(path, default_elem_info, **kwargs)


def _spectrum(e_consts, element):
    params = {k: torch.tensor(v, dtype=torch.float64) for k, v in default_param_vals.items()}
    params["COHERENT_SCT_ENERGY"] = torch.tensor(16.5, dtype=torch.float64)
    # Nonzero step and tail slopes, so the lines' detector absorption shows.
    params["F_TAIL_OFFSET"] = torch.tensor(0.05, dtype=torch.float64)
    params["F_TAIL_LINEAR"] = torch.tensor(1e-4, dtype=torch.float64)
    params["F_STEP_OFFSET"] = torch.tensor(1e-4, dtype=torch.float64)
    params["F_STEP_LINEAR"] = torch.tensor(1e-5, dtype=torch.float64)
    params[element] = torch.tensor(3.0, dtype=torch.float64)
    ev = 0.01 * torch.arange(100, 2000, dtype=torch.float64)
    return model_elem_spec(params, element, ev, e_consts=e_consts, elem_info=default_elem_info)


def _fixed_table_spectra():
    e_consts = _constants()
    return np.stack([_spectrum(e_consts, name).numpy() for name in ELEMENTS], axis=1)


def _write_reference(spectra):
    # 17 significant digits read back as the same float64.
    np.savetxt(REFERENCE, spectra, fmt="%.17g", delimiter=",", header=",".join(ELEMENTS))


class FixedTableTests(unittest.TestCase):
    def test_fixed_table_model_is_unchanged(self):
        """Without a beam energy the L ratios come from the fixed table and no
        L3-N1 line is added. Fails when the change leaks into fixed-table mode:
        an extra line, a moved column, or a changed absorption or tail."""
        ref = np.loadtxt(REFERENCE, delimiter=",")
        spectra = _fixed_table_spectra()
        for k, name in enumerate(ELEMENTS):
            self.assertTrue(np.array_equal(spectra[:, k], ref[:, k]), name)


@unittest.skipIf(xraylib is None, "xraylib not installed")
class L3N1LineTests(unittest.TestCase):
    def test_lb6_mis_built_as_lb4_doubles_lb4(self):
        """With Lb4's energy and ratio, the appended L3-N1 line is a second copy
        of Lb4, so the model equals the one with Lb4's ratio doubled. Fails when
        the appended line's detector absorption is not computed at its energy or
        ignores DETECTOR_MATERIAL, or when the line gets no tail. The beam must
        clear the L1 edge for Lb4, so the L3 threshold is checked directly."""
        beam = 16.5  # above lead's L1 edge (15.86 keV), so Lb4 is excited
        e_consts = _constants(incident_energy=beam)
        pb = {s.ptype: s for s in e_consts["Pb_L"]}
        self.assertAlmostEqual(pb["Lb6"].energy, 12.14, places=1)
        self.assertTrue(0.01 < pb["Lb6"].ratio < 0.02, pb["Lb6"].ratio)
        pb_info = next(e for e in default_elem_info if e.name == "Pb")
        self.assertEqual(
            binding_threshold_for_line(pb_info, "Lb6"), pb_info.bindingE["L3"]
        )

        lb4 = {
            name: next((s.energy, s.ratio) for s in row if s.ptype == "Lb4")
            for name, row in e_consts.items()
            if any(s.ptype == "Lb6" for s in row)
        }
        with mock.patch(
            "mapstorch.constant.l3n1_line", lambda name, energy: lb4.get(name)
        ):
            mis_built = _constants(incident_energy=beam)

        doubled = [copy.copy(s) for s in e_consts["Pb_L"] if s.ptype != "Lb6"]
        for s in doubled:
            if s.ptype == "Lb4":
                s.ratio *= 2.0
        self.assertGreater(pb["Lb4"].ratio, 0.0)
        torch.testing.assert_close(
            _spectrum(mis_built, "Pb_L"),
            _spectrum({"Pb_L": doubled}, "Pb_L"),
            rtol=1e-12,
            atol=1e-9,
        )


if __name__ == "__main__":
    import sys

    if "--write-reference" in sys.argv:
        os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
        _write_reference(_fixed_table_spectra())
    else:
        unittest.main()
