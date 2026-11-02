# SpectralView 2.1.12 Validation Report

## Release decision

**Decision:** Stable candidate for controlled research testing.

**Not approved as:** externally validated, certified grain grader, calibrated sensor replacement, or publication-ready prediction model without dataset-specific validation.

## Executed checks

### Python syntax

```text
spectral_core.py: PASS
spectral_gui.py: PASS
compileall package: PASS
```

### Deterministic and integration tests

Executed with `pytest` in the build environment:

```text
12 passed
```

Covered contracts:

1. exact raw-dark-white reflectance arithmetic;
2. dark-only output is not mislabelled reflectance;
3. explicit already-reflectance mode;
4. genuine SAM identity and scale invariance;
5. spectral resampling on a compatible grid;
6. wavelength-based RGB band selection;
7. synthetic foreground-object segmentation;
8. deterministic raw/dark/white header discovery;
9. micrometre-to-nanometre metadata conversion;
10. real synthetic ENVI BIL document loading and calibration;
11. governed batch terminal-status and evidence publication;
12. offscreen PyQt GUI open, render and close smoke test.

### Preprocessing smoke check

The following methods returned finite arrays with preserved spectral length:

- none;
- SNV;
- Savitzky–Golay smoothing;
- first derivative;
- second derivative;
- absorbance.

## Executed runtime smoke tests

`PyQt5` and `spectral` were installed in the build environment for integration validation. The package successfully:

- created and read synthetic ENVI files;
- opened a raw/dark/white dataset in the PyQt interface using the offscreen Qt platform;
- rendered the document and closed cleanly;
- completed the command-line `--smoke-test` route;
- completed the governed batch CLI and wrote a `SUCCESS` terminal status.

Native Windows desktop execution remains a required release gate because this build environment is not Windows.

## Required external release evidence

A certified release still requires:

- Windows native GUI start, open, close, batch, and export tests;
- real ENVI BIL/BIP/BSQ datasets;
- raw/dark/white acquisitions from the intended camera;
- calibration comparison with trusted reference software;
- annotated-mask segmentation benchmarks;
- independently held-out SAM/classification evaluation;
- memory and cancellation testing on the largest expected cube;
- exact source, dependency lock, and packaged-executable fingerprints.
