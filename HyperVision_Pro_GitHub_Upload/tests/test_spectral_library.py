# -*- coding: utf-8 -*-
import os
import tempfile
import numpy as np
import pytest
from spectral_library import (
    SpectralLibrary,
    SpectralLibraryEntry,
    compute_sam_angle,
    compute_scm_correlation,
    compute_sid_divergence,
    compute_rmse
)

def test_similarity_metrics():
    s1 = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    s2 = s1.copy()
    assert abs(compute_sam_angle(s1, s2)) < 1e-4
    assert abs(compute_scm_correlation(s1, s2) - 1.0) < 1e-4
    assert abs(compute_sid_divergence(s1, s2)) < 1e-4
    assert abs(compute_rmse(s1, s2)) < 1e-4

    s3 = 2.5 * s1
    assert abs(compute_sam_angle(s1, s3)) < 1e-4

def test_library_envelopes():
    wv = np.linspace(400, 1000, 50)
    samples = np.array([
        np.ones(50) * 0.2,
        np.ones(50) * 0.4,
        np.ones(50) * 0.6
    ])
    entry = SpectralLibraryEntry.from_samples("TestClass", wv, samples, color="#2ca02c")
    assert entry.sample_count == 3
    assert np.allclose(entry.mean, 0.4)
    assert np.allclose(entry.min_spec, 0.2)
    assert np.allclose(entry.max_spec, 0.6)
    assert np.allclose(entry.median, 0.4)

def test_envi_sli_export_roundtrip():
    lib = SpectralLibrary("EnviTest")
    wv = np.linspace(400, 900, 25)
    s1 = np.random.rand(5, 25)
    s2 = np.random.rand(8, 25)
    lib.add_entry(SpectralLibraryEntry.from_samples("Endmember_A", wv, s1))
    lib.add_entry(SpectralLibraryEntry.from_samples("Endmember_B", wv, s2))

    with tempfile.TemporaryDirectory() as td:
        sli_path = os.path.join(td, "lib_export.sli")
        hdr_path = os.path.join(td, "lib_export.hdr")
        lib.export_envi_sli(sli_path)

        assert os.path.exists(sli_path)
        assert os.path.exists(hdr_path)
        assert os.path.getsize(sli_path) == 2 * 25 * 4

        raw = np.fromfile(sli_path, dtype=np.float32).reshape((2, 25))
        assert np.allclose(raw[0], lib.entries[0].mean.astype(np.float32), atol=1e-5)
        assert np.allclose(raw[1], lib.entries[1].mean.astype(np.float32), atol=1e-5)

def test_json_and_csv_export():
    lib = SpectralLibrary("JsonCsvTest")
    wv = np.linspace(450, 850, 15)
    lib.add_entry(SpectralLibraryEntry.from_samples("Veg", wv, np.random.rand(4, 15)))

    with tempfile.TemporaryDirectory() as td:
        j_p = os.path.join(td, "lib.json")
        c_p = os.path.join(td, "lib.csv")
        lib.export_json(j_p)
        lib.export_csv(c_p)

        assert os.path.exists(j_p)
        assert os.path.exists(c_p)

        lib2 = SpectralLibrary()
        lib2.import_json(j_p)
        assert len(lib2.entries) == 1
        assert lib2.entries[0].name == "Veg"
        assert np.allclose(lib2.entries[0].mean, lib.entries[0].mean)


def test_spectral_library_get_entry_and_container_protocols():
    lib = SpectralLibrary("TestLib")
    wv = np.linspace(400, 1000, 30)
    e1 = SpectralLibraryEntry.from_samples("Grain_Wheat_01", wv, np.random.rand(4, 30), "#e11d48")
    e2 = SpectralLibraryEntry.from_samples("Plant_Canopy_Leaf", np.linspace(400, 950, 40), np.random.rand(3, 40), "#22c55e")
    lib.add_entry(e1)
    lib.add_entry(e2)

    assert hasattr(lib, "get_entry")
    assert lib.get_entry("Grain_Wheat_01") is e1
    assert lib.get_entry("grain_wheat_01 ") is e1
    assert lib.get_entry("Canopy") is e2
    assert lib.get_entry(0) is e1
    assert lib.get_entry(1) is e2
    assert lib.get_entry("missing_entry") is None
    assert hasattr(lib, "get_entry_by_name")
    assert hasattr(lib, "get_entries")
    assert len(lib.get_entries()) == 2
    assert len(lib) == 2
    assert "Grain_Wheat_01" in lib
    assert lib["Grain_Wheat_01"] is e1
    assert [e.name for e in lib] == ["Grain_Wheat_01", "Plant_Canopy_Leaf"]


def test_spatial_studio_with_populated_spectral_library():
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from spatial_classification_studio import SpatialClassificationStudioDialog
    from PyQt5.QtWidgets import QApplication

    _app = QApplication.instance() or QApplication([''])
    lib = SpectralLibrary("PopulatedLib")
    wv = np.linspace(400, 1000, 20)
    cube = np.random.rand(15, 15, 20).astype(np.float32)

    e1 = SpectralLibraryEntry.from_samples("Grain_01", wv, np.random.rand(3, 20))
    e2 = SpectralLibraryEntry.from_samples("Foreign_Ref", np.linspace(400, 1000, 35), np.random.rand(2, 35))
    lib.add_entry(e1)
    lib.add_entry(e2)

    dlg = SpatialClassificationStudioDialog(
        datacube=cube,
        wavelengths=wv,
        spectral_library=lib,
        classes=["Class_A", "Class_B"]
    )
    assert dlg is not None
    assert dlg.sam_rule_map is not None
    assert dlg.sam_rule_map.shape == (15, 15)

    # Change to foreign library reference with mismatching bands
    for i in range(dlg.cmb_sam_target.count()):
        if "Foreign_Ref" in dlg.cmb_sam_target.itemText(i):
            dlg.cmb_sam_target.setCurrentIndex(i)
            break
    assert dlg.sam_rule_map.shape == (15, 15)
    dlg.close()
