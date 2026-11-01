import unittest
import warnings

from mapstorch.constant import read_constants
from mapstorch.default import default_elem_info

try:
    import xraylib  # noqa: F401
except ImportError:  # pragma: no cover - optional dependency
    xraylib = None


def _ratios(e_consts, name):
    return {s.ptype: s.ratio for s in e_consts[name] if s.ptype}


@unittest.skipIf(xraylib is None, "xraylib not installed")
class XraylibLineRatioTests(unittest.TestCase):
    def _write(self, tmp, text):
        path = f"{tmp}/maps_fit_parameters_override.txt"
        with open(path, "w") as f:
            f.write(text)
        return path

    def test_lanthanum_lbeta_follows_xraylib_not_the_table(self):
        """La L beta1+3+4 over L alpha1 at 10.27 keV is about 0.77 (xraylib, and
        0.76 measured on a thin-film standard); the shared table gives 2.0.
        Fails when the table's ratios are kept."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "ELEMENTS_TO_FIT: La_L\n")
            table = _ratios(read_constants(path, default_elem_info), "La_L")
            beam = _ratios(
                read_constants(path, default_elem_info, incident_energy=10.27), "La_L"
            )
        lb_table = table["Lb1"] + table["Lb3"] + table["Lb4"]
        lb_beam = beam["Lb1"] + beam["Lb3"] + beam["Lb4"]
        self.assertGreater(lb_table, 1.5)
        self.assertGreater(lb_beam, 0.6)
        self.assertLess(lb_beam, 0.9)

    def test_lines_above_the_beam_energy_are_zero(self):
        """At 13.5 keV only lead's L3 subshell is ionised (L2 edge 15.2 keV), so
        every L1 and L2 line has ratio 0. Fails when a line xraylib gives no cross
        section for keeps its table ratio."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, "ELEMENTS_TO_FIT: Pb_L\n")
            beam = _ratios(
                read_constants(path, default_elem_info, incident_energy=13.5), "Pb_L"
            )
        for line in ("Lb1", "Lg1", "Ln", "Lb3", "Lb4", "Lg2", "Lg3", "Lg4"):
            self.assertEqual(beam[line], 0.0, line)
        self.assertGreater(beam["Lb2"], 0.1)

    def test_settings_file_l_adjustments_are_not_applied_twice(self):
        """A settings file's L multipliers correct the table; on xraylib's ratios
        they are skipped, with a warning. Fails when iodine's L beta comes out
        scaled by the file's 0.45."""
        import tempfile

        text = (
            "ELEMENTS_TO_FIT: I_L\n"
            "BRANCHING_RATIO_ADJUSTMENT_L: I_L, 1., 1., 0.45, 1., 0.45, 0.45, 1., 1., 1., 1., 1., 1.\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            plain = self._write(tmp, "ELEMENTS_TO_FIT: I_L\n")
            reference = _ratios(
                read_constants(plain, default_elem_info, incident_energy=13.6), "I_L"
            )
            path = self._write(tmp, text)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                adjusted = _ratios(
                    read_constants(path, default_elem_info, incident_energy=13.6), "I_L"
                )
            kept = _ratios(
                read_constants(
                    path, default_elem_info, incident_energy=13.6, keep_l_adjustments=True
                ),
                "I_L",
            )
        self.assertEqual(adjusted, reference)
        self.assertTrue(any("I_L" in str(w.message) for w in caught))
        self.assertAlmostEqual(kept["Lb1"], 0.45 * reference["Lb1"])


if __name__ == "__main__":
    unittest.main()
