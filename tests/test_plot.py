import unittest

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

from mapstorch.constant import define_constants
from mapstorch.default import default_elem_info, default_param_vals
from mapstorch.plot import _format_elem_label, plot_elem_peak_pos, plot_elem_spec_contribs


def _make_params(dtype=torch.float32):
    params = {}
    for key, value in default_param_vals.items():
        params[key] = torch.tensor(value, dtype=dtype)
    params["ENERGY_OFFSET"] = torch.tensor(0.0, dtype=dtype)
    params["ENERGY_SLOPE"] = torch.tensor(0.01, dtype=dtype)
    params["ENERGY_QUADRATIC"] = torch.tensor(0.0, dtype=dtype)
    params["COHERENT_SCT_ENERGY"] = torch.tensor(20.0, dtype=dtype)
    params["COMPTON_ANGLE"] = torch.tensor(90.0, dtype=dtype)
    params["Fe_Fe"] = torch.tensor(3.0, dtype=dtype)
    return params


class PlotPileupTests(unittest.TestCase):
    def test_pileup_labels_are_human_readable(self):
        self.assertEqual(_format_elem_label("Fe_Fe"), "Fe pileup")
        self.assertEqual(_format_elem_label("Fe_L_Fe"), "Fe L + Fe pileup")

    def test_pileup_markers_and_contribution_plots_include_pileups(self):
        e_consts = define_constants(default_elem_info, pileups=["Fe_Fe"], return_dict=True)
        params = _make_params()
        int_spec = np.zeros(2001, dtype=np.float32)

        _, elem_ranges = plot_elem_peak_pos(
            params,
            int_spec=int_spec,
            energy_range=(0, 2000),
            target_elems=["Fe_Fe"],
            e_consts=e_consts,
            save=False,
            show=False,
        )
        self.assertIn("Fe_Fe", elem_ranges)

        _, contribs = plot_elem_spec_contribs(
            params,
            int_spec=int_spec,
            energy_range=(0, 2000),
            target_elems=["Fe_Fe"],
            elem_amps={"Fe_Fe": float(params["Fe_Fe"].item())},
            e_consts=e_consts,
            save=False,
            show=False,
        )
        self.assertIn("Fe_Fe", contribs)
        self.assertGreater(float(np.max(contribs["Fe_Fe"])), 0.0)


if __name__ == "__main__":
    unittest.main()
