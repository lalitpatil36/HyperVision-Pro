from pathlib import Path

import numpy as np
from spectral.io import envi

from spectral_core import CalibrationConfig, EnviDocument, ProcessingConfig, run_batch


def _write_dataset(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    wavelengths = [450, 550, 650, 800, 900]
    metadata = {"wavelength": wavelengths, "wavelength units": "nm"}
    dark = np.full((20, 30, 5), 10, dtype=np.float32)
    white = np.full((20, 30, 5), 110, dtype=np.float32)
    raw = np.full((20, 30, 5), 60, dtype=np.float32)
    raw[5:15, 8:22, :] = np.linspace(40, 90, 5, dtype=np.float32)
    envi.save_image(str(folder / "data.hdr"), raw, metadata=metadata, dtype=np.float32, interleave="bil", force=True)
    envi.save_image(
        str(folder / "darkReference.hdr"), dark, metadata=metadata, dtype=np.float32, interleave="bil", force=True
    )
    envi.save_image(
        str(folder / "whiteReference.hdr"), white, metadata=metadata, dtype=np.float32, interleave="bil", force=True
    )


def test_envi_document_raw_dark_white(tmp_path):
    sample = tmp_path / "sample_01"
    _write_dataset(sample)
    document = EnviDocument(
        str(sample / "data.hdr"),
        str(sample / "darkReference.hdr"),
        str(sample / "whiteReference.hdr"),
        CalibrationConfig(),
    )
    preview, qc, scale_x, scale_y = document.calibrated_preview()
    assert preview.shape == (20, 30, 5)
    assert qc.mode == "RAW_DARK_WHITE_REFLECTANCE"
    assert qc.units == "reflectance"
    np.testing.assert_allclose(preview[0, 0], 0.5, atol=1e-7)
    assert scale_x == 1.0
    assert scale_y == 1.0


def test_governed_batch_writes_terminal_evidence(tmp_path):
    parent = tmp_path / "datasets"
    _write_dataset(parent / "sample_01")
    output = tmp_path / "output"
    manifest = run_batch(
        str(parent),
        str(output),
        ProcessingConfig(calibration=CalibrationConfig(), preprocessing="none", chunk_rows=4),
        extraction_mode="whole_frame",
    )
    assert manifest["dataset_count"] == 1
    assert manifest["terminal_status_counts"].get("SUCCESS", 0) + manifest["terminal_status_counts"].get(
        "SUCCESS_WITH_WARNING", 0
    ) == 1
    assert (output / "sample_status.csv").is_file()
    assert (output / "spectra_long.csv").is_file()
    assert (output / "run_manifest.json").is_file()
