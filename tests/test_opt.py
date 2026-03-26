import unittest

from mapstorch.opt import _ordered_unique


class OptHelperTests(unittest.TestCase):
    def test_ordered_unique_preserves_first_seen_order(self):
        self.assertEqual(
            _ordered_unique(["Fe", "Pb_L", "Fe", "COMPTON_AMPLITUDE", "Pb_L", "Au_M"]),
            ["Fe", "Pb_L", "COMPTON_AMPLITUDE", "Au_M"],
        )


if __name__ == "__main__":
    unittest.main()
