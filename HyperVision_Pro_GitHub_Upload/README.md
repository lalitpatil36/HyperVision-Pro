# SpectralView 2.1.12 — Scientific Stable Candidate

**Release status:** `STABLE_CANDIDATE_NOT_EXTERNALLY_VALIDATED`

This package is a full scientific upgrade of the original single-file prototype. It removes unsupported grading claims, replaces the mislabelled Euclidean classifier with genuine Spectral Angle Mapper, and makes GUI and batch processing use the same scientific core.

## Start on Windows

1. Extract the ZIP to a normal writable folder.
2. Run `INSTALL_DEPENDENCIES.bat` once.
3. Run `START_SPECTRALVIEW.bat`.
4. Use **Open folder** and select a folder containing the sample ENVI header and, where available, matching dark and white-reference headers.

To validate the numerical and integration contracts, run `RUN_CORE_TESTS.bat`.

For automated GUI startup validation, run for example:

```text
python spectral_gui.py --folder "C:\path\to\sample" --smoke-test
```

For command-line batch processing:

```text
python spectral_batch.py --parent "C:\datasets" --output "C:\results" --extraction auto_foreground
```

## Input contract

The folder discovery engine recognises:

- sample/data headers containing terms such as `data`, `raw`, `sample`, `capture`, `image`, or `cube`;
- dark references containing `dark`, `black`, or `shutter`;
- white references containing `white`, `reference`, `ref_white`, or `panel`.

Ambiguous files are reported. The application does not silently call dark-subtracted counts “reflectance.”

### Calibration modes

| Available inputs | Output status |
|---|---|
| Explicit already-reflectance declaration | `DECLARED_REFLECTANCE_INPUT` |
| Raw + dark + white | `RAW_DARK_WHITE_REFLECTANCE` |
| Raw + white, no dark | `RAW_WHITE_REFLECTANCE_NO_DARK` with warning |
| Raw + dark, no white | `DARK_CORRECTED_COUNTS`, not reflectance |
| Raw only | `RAW_COUNTS_UNCALIBRATED` |

Reflectance is calculated as:

```text
(raw - dark) / (white - dark)
```

Low denominators, non-finite values, wavelength mismatches, incompatible reference shapes, wavelength units, and exposure/integration-time mismatches are explicitly governed. Reflectance clipping is optional and disabled by default.

## Major upgrades

### Scientific truth

- Removed fabricated `Grade A`, `Grade B`, and `Reject` assignments.
- Replaced “SAM” Euclidean distance with genuine angular SAM.
- Replaced band-fraction “true colour” with wavelength-based 650/550/450 nm rendering.
- Exposes the real calibration mode and units in the GUI and all exports.
- Uses an explicit `stable candidate` banner instead of a certification claim.

### Unified engine

`spectral_core.py` is independent of Qt. The same functions are used by:

- interactive preview;
- point and ROI extraction;
- calibration;
- spectral preprocessing;
- SAM classification;
- batch extraction;
- evidence exports;
- automated tests.

### ENVI and wavelength governance

- Memory-mapped ENVI loading when supported by Spectral Python.
- Physical wavelength parsing and monotonicity validation.
- Dark/white wavelength compatibility checks.
- Safe spectral resampling for cross-cube reference libraries.
- Band-index fallback is clearly labelled when wavelengths are absent.

### Visualisation

- PCA composite with deterministic sampling.
- Wavelength-based true colour and false colour.
- Single-band heatmap.
- NDVI, GNDVI, NDRE, PRI, PSRI, WBI, MSI, CIRE, MTCI, and MCARI maps.
- Index wavelength-tolerance gate.
- Percentile contrast and gamma controls.

### Spectral extraction and preprocessing

- Exact full-resolution point extraction from a downsampled display.
- ROI object extraction with PCA/Otsu, auto polarity, morphology, connected components, area filtering, and QC.
- No silent whole-ROI fallback; the user must explicitly accept it.
- SNV, Savitzky–Golay smoothing, first derivative, second derivative, and absorbance transforms.
- Long-format exports that remain valid across different wavelength grids.

### Genuine Spectral Angle Mapper

- Replicate references are averaged by user-supplied class label.
- Cross-document references are wavelength-aligned before use.
- Pixel classification is processed in row chunks.
- User-defined angular acceptance threshold.
- Unknown/unclassified pixels are retained as `-1`.
- Exports class map, minimum-angle map, legend, overlay, and provenance manifest.

### Governed batch processing

- Recursive dataset discovery.
- Auto-foreground mode that fails explicitly when segmentation fails.
- Explicit whole-frame mode when that is scientifically intended.
- Chunked calibrated mean spectra.
- Atomic publication of:
  - `sample_status.csv`
  - `spectra_long.csv`
  - `run_manifest.json`
- Every discovered sample receives a terminal status.
- Input header SHA-256 fingerprints and processing configuration are stored.

## Files

| File | Purpose |
|---|---|
| `spectral_gui.py` | PyQt5 desktop application |
| `spectral_batch.py` | governed command-line batch runner |
| `spectral_core.py` | Qt-independent scientific engine |
| `tests/test_spectral_core.py` | deterministic numerical tests |
| `requirements.txt` | Python dependencies |
| `INSTALL_DEPENDENCIES.bat` | Windows dependency installer |
| `START_SPECTRALVIEW.bat` | Windows startup launcher |
| `RUN_CORE_TESTS.bat` | Windows core validation launcher |
| `VALIDATION_REPORT.md` | executed validation evidence and limitations |

## Important limitations

This package is not yet an externally validated grain-quality or biochemical-prediction system. SAM maps similarity to labelled reference spectra; it does not prove grain grade, composition, disease status, or varietal identity.

The ROI segmentation is a transparent heuristic and reports QC, but it requires validation against manually annotated masks for each acquisition setup.

Before a certified release, execute:

- real Headwall VNIR/SWIR raw-dark-white calibration fixtures;
- sensor-specific saturation and exposure checks;
- Windows GUI startup and long-run tests;
- segmentation accuracy against expert masks;
- external classification/chemometric validation;
- installer and frozen-build testing against the exact release fingerprint.
