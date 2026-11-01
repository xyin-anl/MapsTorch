import tempfile
import unittest
import warnings

from mapstorch.constant import read_constants, settings_float
from mapstorch.default import default_elem_info
from mapstorch.io import parse_override_params_file


def _write(tmp, text):
    path = f"{tmp}/maps_fit_parameters_override.txt"
    with open(path, "w") as f:
        f.write(text)
    return path


class SettingsFileParsingTests(unittest.TestCase):
    def test_number_with_a_trailing_dot_reads_as_xrf_maps_reads_it(self):
        """A beamline settings file writes "0.8." in a K adjustment; XRF-Maps
        reads it as 0.8. Fails when the line stops every fit of those scans
        (float("0.8.") raises) or when it is read as anything but 0.8."""
        with tempfile.TemporaryDirectory() as tmp:
            plain = read_constants(_write(tmp, "ELEMENTS_TO_FIT: K\n"), default_elem_info)
            path = _write(tmp, "ELEMENTS_TO_FIT: K\nBRANCHING_RATIO_ADJUSTMENT_K: K, 1., 1., 0.8., 1.\n")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                adjusted = read_constants(path, default_elem_info)
        kb = {s.ptype: s.ratio for s in adjusted["K"]}["Kb1"]
        kb0 = {s.ptype: s.ratio for s in plain["K"]}["Kb1"]
        self.assertAlmostEqual(kb, 0.8 * kb0, places=12)
        self.assertTrue(any("0.8." in str(w.message) for w in caught))

    def test_tag_written_twice_keeps_the_first_element(self):
        """A settings file writes "ELEMENTS_TO_FIT: ELEMENTS_TO_FIT: Al, Si".
        Fails when Al is lost as an element named "ELEMENTS_TO_FIT: Al"."""
        with tempfile.TemporaryDirectory() as tmp:
            params = parse_override_params_file(
                _write(tmp, "ELEMENTS_TO_FIT: ELEMENTS_TO_FIT: Al, Si, Fe\n")
            )
        self.assertEqual(params["elements_to_fit"], ["Al", "Si", "Fe"])

    def test_settings_float_needs_a_number(self):
        """Fails when text with no number is read as some value instead of raising."""
        self.assertEqual(settings_float("9.9999997e-005"), 9.9999997e-05)
        with self.assertRaises(ValueError):
            settings_float("abc")


if __name__ == "__main__":
    unittest.main()
