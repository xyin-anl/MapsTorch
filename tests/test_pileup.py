import tempfile
import unittest
from pathlib import Path

import torch

from mapstorch.constant import binding_threshold_for_line, define_pileup_constants, read_constants
from mapstorch.default import default_elem_info
from mapstorch.io import write_override_params_file
from mapstorch.map import _binding_energy_mask


def _get_elem(name):
    return next(info for info in default_elem_info if info.name == name)


class PileupConstantTests(unittest.TestCase):
    def test_same_element_binary_k_pileups_are_generated(self):
        consts = define_pileup_constants(
            default_elem_info, ["Fe_Fe", "K_K"], return_dict=True
        )
        fe = _get_elem("Fe")
        potassium = _get_elem("K")

        self.assertIn("Fe_Fe", consts)
        self.assertIn("K_K", consts)
        self.assertAlmostEqual(
            float(consts["Fe_Fe"][0].energy),
            float(fe.xrf["ka1"] + fe.xrf["ka1"]),
            places=6,
        )
        self.assertAlmostEqual(
            float(consts["K_K"][0].energy),
            float(potassium.xrf["ka1"] + potassium.xrf["ka1"]),
            places=6,
        )

    def test_shell_suffixes_change_pileup_energies(self):
        consts = define_pileup_constants(
            default_elem_info,
            ["Fe_Fe", "Fe_L_Fe", "Fe_Fe_L", "Fe_L_Fe_L"],
            return_dict=True,
        )
        fe = _get_elem("Fe")
        fe_fe_energies = [float(r.energy) for r in consts["Fe_Fe"] if float(r.energy) > 0.0]
        fe_l_fe_energies = [
            float(r.energy) for r in consts["Fe_L_Fe"] if float(r.energy) > 0.0
        ]
        fe_fe_l_energies = [
            float(r.energy) for r in consts["Fe_Fe_L"] if float(r.energy) > 0.0
        ]
        fe_l_fe_l_energies = [
            float(r.energy) for r in consts["Fe_L_Fe_L"] if float(r.energy) > 0.0
        ]

        self.assertTrue(
            any(
                abs(energy - float(fe.xrf["la1"] + fe.xrf["ka1"])) < 1e-6
                for energy in fe_l_fe_energies
            )
        )
        self.assertTrue(
            any(
                abs(energy - float(fe.xrf["ka1"] + fe.xrf["la1"])) < 1e-6
                for energy in fe_fe_l_energies
            )
        )
        self.assertTrue(
            any(
                abs(energy - float(fe.xrf["la1"] + fe.xrf["la1"])) < 1e-6
                for energy in fe_l_fe_l_energies
            )
        )
        self.assertFalse(
            set(round(energy, 6) for energy in fe_l_fe_energies).issubset(
                set(round(energy, 6) for energy in fe_fe_energies)
            )
        )

    def test_l_shell_pileup_binding_threshold_is_shell_aware(self):
        consts = define_pileup_constants(
            default_elem_info, ["Fe_L_Fe_L"], return_dict=True
        )
        low_energy = torch.tensor(0.5, dtype=torch.float64)
        high_energy = torch.tensor(1.0, dtype=torch.float64)

        self.assertFalse(
            bool(_binding_energy_mask(consts["Fe_L_Fe_L"][0], low_energy, default_elem_info).item())
        )
        self.assertTrue(
            bool(_binding_energy_mask(consts["Fe_L_Fe_L"][0], high_energy, default_elem_info).item())
        )

    def test_non_pileup_l_and_m_lines_keep_their_shell_thresholds(self):
        pb = _get_elem("Pb")
        au = _get_elem("Au")

        self.assertAlmostEqual(
            binding_threshold_for_line(pb, "La1"),
            float(pb.bindingE["L3"]),
            places=6,
        )
        self.assertAlmostEqual(
            binding_threshold_for_line(au, "Ma1"),
            float(au.bindingE["M1"]),
            places=6,
        )
        self.assertLess(binding_threshold_for_line(pb, "La1"), float(pb.bindingE["K"]))
        self.assertLess(binding_threshold_for_line(au, "Ma1"), float(au.bindingE["K"]))

    def test_empty_pileup_override_entry_does_not_create_empty_key(self):
        with tempfile.TemporaryDirectory(prefix="mapstorch_pileup_tests_") as tmpdir:
            override_path = Path(tmpdir) / "override.txt"
            write_override_params_file(str(override_path), elements=["Fe"])
            consts = read_constants(override_path, default_elem_info)

        self.assertNotIn("", consts)


if __name__ == "__main__":
    unittest.main()
