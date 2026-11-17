"""Known answer for the background in per-pixel fits.

Fails if a faint pixel's continuum goes into its element maps, or if a bright scan whose
continuum changes shape is given one tied shape anyway. At a few hundred counts per pixel most
channels of one pixel are empty, so SNIP of that pixel alone is near zero and the continuum
under a line is counted as the line (Zn about 1.8 times planted on this synthetic scan).
"""

import unittest

import numpy as np
import torch

from mapstorch.default import default_param_vals
from mapstorch.map import build_model_basis
from mapstorch.opt import (
    TIED_BACKGROUND_GUARD,
    TIED_CARRIER_GUARD,
    guard_tied_background,
    solve_mca_amplitudes,
    tied_pixel_background,
)

E0 = 10.57  # keV
LO, HI = 100, 1100  # 1-11 keV at 10 eV per channel
ELEMENTS = ["P", "S", "Cl", "Ar", "K", "Ca", "Fe", "Cu", "Zn"]
# mean counts per pixel of each component in a tissue-like pixel
MEAN = {"P": 6, "S": 8, "Cl": 10, "Ar": 6, "K": 12, "Ca": 15, "Fe": 6, "Cu": 1.5, "Zn": 3,
        "COHERENT_SCT_AMPLITUDE": 60, "COMPTON_AMPLITUDE": 90}
SOLVER = dict(elements_to_fit=ELEMENTS, amp_method="nnls_pgd", ridge_lambda=0.0)


def _params():
    values = {**default_param_vals, "ENERGY_OFFSET": 0.0, "ENERGY_SLOPE": 0.01,
              "ENERGY_QUADRATIC": 0.0, "COHERENT_SCT_ENERGY": E0}
    return {k: torch.tensor(float(v)) for k, v in values.items()}


def _scan(params, n_pixels, counts, seed, flat_half=False, scatter=1.0, missing=None):
    """Poisson spectra (pixels, 2048) and the planted counts per component. The continuum rises
    toward the Compton peak with a low-energy shoulder (a shape SNIP does not make), is about a
    fifth of the counts, and follows each pixel's density within +-40 %. With `flat_half`, half
    the pixels have a flatter continuum (another material). `scatter` scales both scatter peaks;
    `missing` (element, strength) adds that element's lines, `strength` times all other lines'
    counts, to the data but not to the fit's element list."""
    with torch.no_grad():
        basis, names = build_model_basis(params, (LO, HI), elements_to_fit=ELEMENTS)
    basis, names = basis.numpy().astype(np.float64), [str(n) for n in names]
    rng = np.random.default_rng(seed)
    e = np.arange(LO, HI + 1) * 0.01
    shape = 0.4 * np.exp((e - 10.0) / 1.2) + 0.03 * np.exp(-(((e - 2.0) / 1.5) ** 2)) + 0.01
    density = rng.gamma(4, 0.25, n_pixels)
    weight = {"COMPTON_AMPLITUDE": scatter, "COHERENT_SCT_AMPLITUDE": scatter}
    unit = np.array([MEAN.get(n, 0.0) * weight.get(n, 1.0) for n in names])
    unit = unit / np.where(basis.sum(axis=1) > 0, basis.sum(axis=1), 1.0)
    amplitudes = density[:, None] * unit[None, :]
    lines = amplitudes @ basis
    if missing is not None:
        with torch.no_grad():
            extra, extra_names = build_model_basis(params, (LO, HI), elements_to_fit=[missing[0]])
        row = extra.numpy()[[str(n) for n in extra_names].index(missing[0])].astype(np.float64)
        per_pixel = density * missing[1] * lines.sum() / density.sum()
        lines = lines + np.outer(per_pixel, row / row.sum())
    continuum = (density * rng.uniform(0.6, 1.4, n_pixels))[:, None] * shape[None, :]
    if flat_half:
        flat = 0.2 * np.exp((e - 10.0) / 2.5) + 0.02
        continuum[: n_pixels // 2] *= (flat * shape.sum() / flat.sum() / shape)[None, :]
    continuum *= 0.25 * lines.sum() / continuum.sum()
    scale = counts / (lines + continuum).sum(axis=1).mean()
    spectra = np.zeros((n_pixels, 2048), np.float32)
    spectra[:, LO : HI + 1] = rng.poisson((lines + continuum) * scale)
    planted = dict(zip(names, (amplitudes * scale).sum(axis=0) * basis.sum(axis=1), strict=True))
    return spectra, planted, basis, names


def _summed(params, spectra, basis, names, pixel_background):
    amps, _ = solve_mca_amplitudes(
        spectra, (LO, HI), params, energy_dim=-1, return_log10=False, progress_bar=False,
        pixel_background=pixel_background, **SOLVER,
    )
    return dict(zip(names, amps.sum(axis=0) * basis.sum(axis=1), strict=True))


class TiedBackgroundTests(unittest.TestCase):
    def test_faint_pixels_keep_the_continuum_out_and_bright_ones_keep_snip(self):
        params = _params()

        # Faint: the tie is kept and gives the planted counts; SNIP of each pixel does not.
        spectra, planted, basis, names = _scan(params, 3000, 300, seed=0)
        tie = tied_pixel_background(spectra.sum(axis=0), (LO, HI), params, **SOLVER)
        self.assertIsNotNone(tie)
        self.assertEqual(tie[0], "COMPTON_AMPLITUDE")
        kept, ratio, _ = guard_tied_background(tie, spectra[::3], (LO, HI), params, **SOLVER)
        self.assertIs(kept, tie, ratio)
        with self.assertRaises(ValueError):  # nothing to score is no licence to keep the tie
            guard_tied_background(tie, np.full_like(spectra[:5], np.nan), (LO, HI), params)
        tied = _summed(params, spectra, basis, names, kept)
        for element in ("Ca", "Fe", "Zn"):
            self.assertAlmostEqual(tied[element] / planted[element], 1.0, delta=0.06, msg=element)
        snip = _summed(params, spectra, basis, names, None)
        self.assertGreater(snip["Zn"] / planted["Zn"], 1.4)

        # Bright, two continuum shapes: SNIP of each pixel finds the background, the scan's one
        # shape does not, and the guard drops the tie.
        spectra, planted, basis, names = _scan(params, 2000, 300_000, seed=2, flat_half=True)
        tie = tied_pixel_background(spectra.sum(axis=0), (LO, HI), params, **SOLVER)
        self.assertIsNotNone(tie)
        kept, ratio, _ = guard_tied_background(tie, spectra[::4], (LO, HI), params, **SOLVER)
        self.assertIsNone(kept, ratio)
        snip = _summed(params, spectra, basis, names, kept)
        self.assertAlmostEqual(snip["Zn"] / planted["Zn"], 1.0, delta=0.04)

    def test_a_line_missing_from_the_element_list_does_not_swell_the_background(self):
        """Fails if a peak the fit has no line for is taken into the tied background: least
        squares scales the carrier's row (its peak plus the background shape) up to cover it, and
        the background under every line grows with it, while the deviance ratio alone would keep
        the tie. The guard must keep SNIP of each spectrum, and the trace Zn must survive."""
        params = _params()
        spectra, planted, basis, names = _scan(
            params, 2000, 3000, seed=4, scatter=0.2, missing=("La_L", 3.0)
        )
        tie = tied_pixel_background(spectra.sum(axis=0), (LO, HI), params, **SOLVER)
        self.assertIsNotNone(tie)
        kept, ratio, carrier = guard_tied_background(tie, spectra[::4], (LO, HI), params, **SOLVER)
        self.assertLessEqual(ratio, TIED_BACKGROUND_GUARD)
        self.assertGreater(carrier, TIED_CARRIER_GUARD)
        self.assertIsNone(kept)
        self.assertGreater(_summed(params, spectra, basis, names, kept)["Zn"] / planted["Zn"], 0.8)


if __name__ == "__main__":
    unittest.main()
