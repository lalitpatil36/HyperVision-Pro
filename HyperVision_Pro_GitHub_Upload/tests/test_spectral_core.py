import tempfile
from pathlib import Path

import numpy as np

from spectral_core import (
    CalibrationConfig,
    calibrate_array,
    compose_rgb,
    discover_envi_folder,
    nearest_band,
    resample_spectrum,
    sam_classify_array,
    segment_objects,
)


def test_raw_dark_white_reflectance_exact():
    raw = np.array([[[30.0, 50.0]]], dtype=np.float32)
    dark = np.array([[[10.0, 10.0]]], dtype=np.float32)
    white = np.array([[[50.0, 90.0]]], dtype=np.float32)
    calibrated, qc = calibrate_array(raw, dark, white, CalibrationConfig())
    np.testing.assert_allclose(calibrated, [[[0.5, 0.5]]], atol=1e-7)
    assert qc.mode == "RAW_DARK_WHITE_REFLECTANCE"
    assert qc.units == "reflectance"


def test_dark_only_is_not_called_reflectance():
    raw = np.array([[[30.0]]], dtype=np.float32)
    dark = np.array([[[10.0]]], dtype=np.float32)
    calibrated, qc = calibrate_array(raw, dark, None, CalibrationConfig())
    np.testing.assert_allclose(calibrated, [[[20.0]]])
    assert qc.mode == "DARK_CORRECTED_COUNTS"
    assert qc.units == "dark-corrected counts"
    assert any("NOT_REFLECTANCE" in warning for warning in qc.warnings)


def test_declared_reflectance_is_preserved():
    raw = np.array([[[0.2, 0.8]]], dtype=np.float32)
    calibrated, qc = calibrate_array(raw, None, None, CalibrationConfig(input_is_reflectance=True))
    np.testing.assert_allclose(calibrated, raw)
    assert qc.mode == "DECLARED_REFLECTANCE_INPUT"


def test_sam_identical_and_proportional_spectra_match():
    spectra = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    refs = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    classes, angles = sam_classify_array(spectra, refs, threshold_deg=1.0)
    assert classes[0] == 0
    assert classes[1] == 0
    assert classes[2] == -1
    assert angles[0] < 0.05
    assert angles[1] < 0.05
    assert angles[2] > 1.0


def test_resample_requires_full_overlap():
    source_wavelengths = np.array([400.0, 500.0, 600.0])
    spectrum = np.array([1.0, 2.0, 3.0])
    target = np.array([450.0, 550.0])
    result = resample_spectrum(source_wavelengths, spectrum, target)
    np.testing.assert_allclose(result, [1.5, 2.5])


def test_true_colour_uses_nearest_wavelengths():
    cube = np.zeros((2, 2, 4), dtype=np.float32)
    cube[:, :, 0] = 1.0
    cube[:, :, 1] = 2.0
    cube[:, :, 2] = 3.0
    cube[:, :, 3] = 4.0
    wavelengths = np.array([450.0, 550.0, 650.0, 800.0])
    _image, indices = compose_rgb(cube, wavelengths, (650.0, 550.0, 450.0))
    assert indices == (2, 1, 0)
    assert nearest_band(wavelengths, 799.0) == 3


def test_segmentation_finds_bright_object():
    cube = np.zeros((40, 40, 5), dtype=np.float32)
    cube[10:30, 12:28, :] = np.linspace(0.2, 1.0, 5)
    result = segment_objects(cube, min_area_px=20, polarity="auto")
    assert len(result.objects) >= 1
    assert 0.05 < result.foreground_fraction < 0.8


def test_discovery_finds_data_dark_white():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for filename in ("data.hdr", "darkReference.hdr", "whiteReference.hdr"):
            (root / filename).write_text("ENVI\n", encoding="utf-8")
        result = discover_envi_folder(root)
        assert result.data_header.endswith("data.hdr")
        assert result.dark_header.endswith("darkReference.hdr")
        assert result.white_header.endswith("whiteReference.hdr")


def test_wavelength_micrometers_are_converted_to_nm():
    from spectral_core import parse_wavelengths

    wavelengths, physical, warnings = parse_wavelengths(
        {"wavelength": [0.45, 0.55, 0.65], "wavelength units": "micrometers"}, 3
    )
    np.testing.assert_allclose(wavelengths, [450.0, 550.0, 650.0])
    assert physical is True
    assert any("CONVERTED_FROM_MICROMETERS" in warning for warning in warnings)
