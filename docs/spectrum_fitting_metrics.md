# Spectrum Fitting Metrics in MapsTorch (`mapstorch`)

This document describes **objective functions (“losses”)** for XRF spectrum fitting in MapsTorch (`mapstorch`), and how to use them in the fitting APIs.

---

## 1) Why MSE is often not a great default for XRF

XRF spectra have several properties that make vanilla MSE behave poorly in common edge cases:

- **Counting statistics**: counts are Poisson-like, so the variance scales with intensity. MSE assumes constant-variance Gaussian noise.
- **Weak peaks matter**: MSE can be dominated by strong peaks + continuum/background and ignore weak lines.
- **Small calibration drift**: tiny peak shifts (gain/offset drift) can produce a large MSE even if the fit is chemically correct.
- **Peak-shape mismatch**: incorrect resolution/tails/escape behavior can look “OK” in MSE but is obviously wrong to humans.

---

## 2) What the loss is scoring in MapsTorch

MapsTorch’s forward model is physics-inspired, so many parameters change spectra in structured ways:

- **Energy calibration**: `ENERGY_OFFSET`, `ENERGY_SLOPE`, `ENERGY_QUADRATIC` map channel → energy.
- **Resolution**: `FWHM_OFFSET`, `FWHM_FANOPRIME` drive the energy-dependent peak width.
- **Line families**: characteristic lines (K/L/M) are summed using tabulated line energies and branching ratios (with binding-energy gates).
- **Scatter**: elastic (`COHERENT_SCT_AMPLITUDE`) and Compton (`COMPTON_AMPLITUDE`) components.
- **Background**: SNIP is used during fitting/evaluation (`SNIP_WIDTH`).

Because calibration/resolution errors shift/broaden features, **pointwise channel errors** can be a misleading proxy for “spectral correctness”.

---

## 3) Base losses available for spectrum fitting

All base losses are available through `fit_spec()` and `fit_spec_batch()` via the `loss` argument.

### 3.1 `poisson_deviance` (recommended baseline)

Use when you want a statistically meaningful objective under Poisson-like noise. It naturally upweights relative errors on weak peaks compared to strong peaks.

### 3.2 `poisson_nll`

Poisson negative log-likelihood (up to constants). Similar behavior to deviance; deviance is often nicer for diagnostics.

### 3.3 `chi2` (weighted least squares)

Useful approximation when counts are high and you want a WLS-like objective.

- Pearson: variance ≈ prediction (`chi2_variant="pearson"`)
- Neyman: variance ≈ observation (`chi2_variant="neyman"`)

Use `sigma2_floor` to avoid instability at tiny counts.

### 3.4 `mse`, `l1`

Kept mainly as baselines and for some regimes where they can work well, but they are generally less aligned with XRF counting noise.

---

## 4) Hybrid (structure-aware) loss add-ons

MapsTorch supports **hybrid objectives**: a base loss plus optional auxiliary terms.
These are configured via `loss_params` and implemented inside `mapstorch/opt.py` (via `_calculate_loss()`).

### 4.1 ROI loss (weak-peak emphasis)

Adds an auxiliary term focusing on energy windows around expected line energies (defined using the *current* calibration and resolution):

- Enable with `roi_lambda != 0`
- Use `roi_on_peaknet=true` to apply it to background-subtracted peaks (`y - bkg`)

### 4.2 Shape loss (shift-tolerant structure matching)

Adds a distribution-matching term on non-negative “mass” (typically net peaks). This is useful when you want robustness to small calibration shifts.

- Enable with `shape_lambda != 0`
- `shape_metric="w1"` uses a 1D Wasserstein-1 distance (with optional shift marginalization)
- `shape_metric="mmd_rbf"` uses multi-scale RBF MMD

### 4.3 Area loss (quantification-aware)

Adds a per-element ROI-integrated area matching term (typically on net peaks):

- Enable with `area_lambda != 0`
- Use small weights first; area terms can over-constrain if set too high.

---

## 5) How to use these losses in the fitting APIs

### 5.1 `fit_spec` (single spectrum)

```python
from mapstorch.opt import fit_spec

fitted, spec_fit, bkg, loss_trace = fit_spec(
    int_spec,
    energy_range,
    elements_to_fit=elements_to_fit,
    loss="poisson_deviance",
    loss_params={
        "eps": 1e-6,
        "reduction": "mean",
        "roi_lambda": 0.5,
        "roi_profile": "gaussian",
        "roi_n_sigma": 3.0,
        "roi_on_peaknet": True,
        "roi_reduction": "weighted_mean",
        "shape_lambda": 0.1,
        "shape_metric": "w1",
        "shape_on_peaknet": True,
        "shape_shift_mode": "blur",
        "shape_shift_max": 2,
        "shape_shift_sigma": 1.0,
    },
)
```

### 5.2 `fit_spec_batch` (many spectra at once)

`fit_spec_batch(..., loss=..., loss_params=...)` uses the same loss stack (base + optional ROI/shape/area).

### 5.3 `fit_mca_arr` (spectra volume)

`fit_mca_arr(..., loss=..., loss_params=...)` applies the loss stack to the **integrated-spectrum global fit** step.
The per-pixel amplitude solving step (`solve_mca_amplitudes`) is a separate linear solve and does not use the hybrid loss terms.

---

## 6) `loss_params` reference (what `mapstorch/opt.py` consumes)

Keys not listed below are ignored.

| Feature | Key(s) | Typical values | Meaning |
| --- | --- | --- | --- |
| Base loss | `reduction` | `sum`, `mean`, `weighted_mean` | How to reduce per-channel loss |
| Base loss | `eps` | `1e-6` | Clamp for logs / denominators |
| Chi-square | `chi2_variant` | `pearson` / `neyman` | Var≈pred vs Var≈obs |
| Chi-square | `sigma2_floor` | `0–10` | Adds a variance floor to stabilize tiny counts |
| Channel weighting | `channel_weights` | array-like or tensor | Optional per-channel weights |
| ROI add-on | `roi_lambda` | `0–2` | Weight of ROI auxiliary loss |
| ROI add-on | `roi_profile` | `box` / `gaussian` | ROI mask shape |
| ROI add-on | `roi_n_sigma` | `2–4` | ROI half-width in σ (resolution units) |
| ROI add-on | `roi_on_peaknet` | `true/false` | Apply ROI to net peaks (`y-bkg`) vs raw counts |
| ROI add-on | `roi_reduction` | `weighted_mean` / `mean` | Reduction for ROI term |
| ROI add-on | `roi_include_scatter` | `true/false` | Include Compton/elastic in ROI definition |
| ROI add-on | `roi_allowed_lines_only` | `true/false` | Exclude lines forbidden by binding-energy gates |
| Shape add-on | `shape_lambda` | `0–1` | Weight of shape auxiliary loss |
| Shape add-on | `shape_metric` | `w1` / `mmd_rbf` | Distribution metric |
| Shape add-on | `shape_on_peaknet` | `true/false` | Shape on net peaks vs raw |
| Shape add-on | `shape_reduction` | `mean` / `sum` | Reduction for shape term |
| Shape add-on | `shape_eps` | `1e-6` | Clamp for distribution normalization |
| Shape add-on | `shape_shift_mode` | `none`, `blur`, `mean`, `min`, `softmin` | How to marginalize over small shifts |
| Shape add-on | `shape_shift_max` | `0–4` | Max shift in channels (integer) |
| Shape add-on | `shape_shift_sigma` | `0.5–2` | Shift blur width (when `blur`) |
| Shape add-on | `shape_shift_tau` | `0.01–0.2` | Softmin temperature (when `softmin`) |
| Shape (W1) | `shape_dx` | `1.0` | Channel spacing for W1 distance |
| Shape (MMD) | `shape_mmd_sigmas` | e.g. `[1,2,4,8]` | Multi-scale kernel widths (channels) |
| Shape (MMD) | `shape_mmd_truncate` | `3` | Kernel support truncation in σ units |
| Area add-on | `area_lambda` | `0–0.5` | Weight of area auxiliary loss |
| Area add-on | `area_mode` | `relative_l1` / `l2` | Per-element area error function |
| Area add-on | `area_roi_n_sigma` | `2–4` | ROI width used for per-element area integration |
| Area add-on | `area_on_peaknet` | `true/false` | Area on net peaks vs raw |
| Area add-on | `area_include_scatter` | `true/false` | Include scatter “areas” as elements |
| Area add-on | `area_allowed_lines_only` | `true/false` | Apply binding-energy line gating |
| Area add-on | `area_eps` | `1e-6` | Stabilizer for relative error denominators |

---

## 7) Practical starting points

If you want a simple default that is usually more aligned with XRF than MSE:

- Base: `loss="poisson_deviance"`, `reduction="mean"`, `eps=1e-6`
- Weak peaks: start with `roi_lambda≈0.5`, `roi_on_peaknet=true`
- Small drift robustness: add `shape_lambda≈0.1`, `shape_metric="w1"`, `shape_shift_mode="blur"`, `shape_shift_max=2`
- Quantification stability: consider `area_lambda≈0.05–0.2` (start small)

