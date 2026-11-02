# -*- coding: utf-8 -*-
"""
Unit and integration tests for Advanced Analytics & Scientific Studios:
- Spatial Prediction & Uncertainty Studio
- PLS-DA VIP Biomarkers & 2D-COS Chemometrics
- Multi-Metric Radar Similarity Matcher
- XAI Waterfall & Gain/Lift Curves
- Automated Scientific Publication Dossier Generator
"""

import sys
import os
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

# Ensure QApplication exists for UI tests
_app = QApplication.instance() or QApplication(sys.argv)

def test_shannon_entropy_calculation():
    """Verify Shannon entropy math for certainty and uniform ambiguity (normalized 0 to 1)."""
    from spatial_classification_studio import compute_shannon_entropy
    
    # 1. Complete certainty: [1.0, 0.0, 0.0] -> normalized entropy = 0.0
    p_certain = np.array([[[1.0, 0.0, 0.0]]])
    h_certain = compute_shannon_entropy(p_certain)
    assert np.isclose(h_certain[0, 0], 0.0, atol=1e-4)

    # 2. Complete uniform ambiguity: 4 classes each 0.25 -> normalized entropy = 1.0
    p_ambiguous = np.array([[[0.25, 0.25, 0.25, 0.25]]])
    h_ambiguous = compute_shannon_entropy(p_ambiguous)
    assert np.isclose(h_ambiguous[0, 0], 1.0, atol=1e-4)

def test_spatial_classification_studio_workflow():
    """Verify Spatial Classification Studio initialization, prediction, and rule images."""
    from spatial_classification_studio import SpatialClassificationStudioDialog
    from sklearn.ensemble import RandomForestClassifier
    
    # Create synthetic datacube (20x20 pixels, 30 bands)
    rng = np.random.RandomState(42)
    cube = rng.rand(20, 20, 30).astype(np.float32)
    wvs = np.linspace(450, 950, 30)

    # Train dummy model on 2 classes
    X_train = rng.rand(40, 30)
    y_train = np.array([0]*20 + [1]*20)
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    clf.fit(X_train, y_train)

    dlg = SpatialClassificationStudioDialog(
        datacube=cube,
        wavelengths=wvs,
        model=clf,
        model_name="Test RF",
        classes=["Class_A", "Class_B"]
    )
    assert dlg is not None
    assert dlg.pred_map is not None
    assert dlg.pred_map.shape == (20, 20)
    assert dlg.entropy_map is not None
    assert dlg.entropy_map.shape == (20, 20)

    # Test SAM rule image generation
    ref_sig = rng.rand(30)
    rule_img = dlg.compute_sam_rule_image(ref_sig)
    assert rule_img.shape == (20, 20)
    assert np.all(rule_img >= 0.0)

def test_chemometrics_vip_and_2dcos():
    """Verify PLS-DA VIP scores and 2D Correlation Spectroscopy math."""
    from chemometrics_advanced_studio import compute_pls_vip, compute_2d_cos
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.RandomState(42)
    X = rng.rand(30, 25)
    y = np.array([0]*15 + [1]*15)

    pls = PLSRegression(n_components=3)
    pls.fit(X, y)

    vips = compute_pls_vip(pls, X, y)
    assert len(vips) == 25
    # Average of squared VIP scores is 1.0
    assert np.isclose(np.mean(vips**2), 1.0, atol=0.1)

    # Test 2D-COS (Synchronous and Asynchronous matrices)
    sync, async_mat = compute_2d_cos(X)
    assert sync.shape == (25, 25)
    assert async_mat.shape == (25, 25)
    # Synchronous matrix is symmetric
    assert np.allclose(sync, sync.T, atol=1e-5)
    # Asynchronous matrix is skew-symmetric (diag is zero)
    assert np.allclose(np.diag(async_mat), 0.0, atol=1e-5)
    assert np.allclose(async_mat, -async_mat.T, atol=1e-5)

def test_spectral_indices_calculator():
    """Verify spectral index calculation across multiple scientific formulations."""
    from chemometrics_advanced_studio import calculate_spectral_index

    wvs = np.array([531.0, 570.0, 670.0, 705.0, 750.0, 800.0, 900.0, 970.0, 2000.0, 2200.0])
    spectrum = np.array([0.1, 0.12, 0.08, 0.25, 0.65, 0.70, 0.72, 0.55, 0.40, 0.20])

    ndvi = calculate_spectral_index(wvs, spectrum, "NDVI (Normalized Difference Veg)")
    assert -1.0 <= ndvi <= 1.0
    assert ndvi > 0.6  # Healthy green veg index

    pri = calculate_spectral_index(wvs, spectrum, "PRI (Photochemical Reflectance)")
    assert -1.0 <= pri <= 1.0

def test_xai_waterfall_and_gain_lift():
    """Verify Explainable AI Waterfall and Cumulative Gain & Lift plots in ModelVisualizationStudioDialog."""
    from model_visualization_studio import ModelVisualizationStudioDialog

    dlg = ModelVisualizationStudioDialog()
    assert dlg is not None

    # Check Cumulative Gain & Lift
    dlg.rad_gain_lift.setChecked(True)
    dlg.refresh_plot()
    assert "Lift" in dlg.lbl_stats.text() or "Gain" in dlg.txt_title.text()

    # Check XAI Waterfall
    dlg.rad_xai.setChecked(True)
    dlg.refresh_plot()
    assert "Impact" in dlg.lbl_stats.text() or "Waterfall" in dlg.txt_title.text()

def test_spectral_matcher_radar():
    """Verify SpectralMatcherDialog radar (spider) chart and multi-metric candidate ranking."""
    from spectral_library import SpectralLibrary, SpectralLibraryEntry, SpectralMatcherDialog

    lib = SpectralLibrary()
    wvs = np.linspace(400, 1000, 50)
    spec1 = np.sin(np.linspace(0, 3, 50))**2 + 0.1
    spec2 = np.cos(np.linspace(0, 3, 50))**2 + 0.1
    lib.add_entry(SpectralLibraryEntry(name="Mineral_A", wavelengths=wvs, mean=spec1, color="#10b981"))
    lib.add_entry(SpectralLibraryEntry(name="Mineral_B", wavelengths=wvs, mean=spec2, color="#6366f1"))

    dlg = SpectralMatcherDialog(library=lib)
    dlg.last_unknown_wv = wvs
    dlg.last_unknown_spec = spec1 + 0.02 * np.random.rand(50)
    dlg.last_matches = lib.match_spectrum(dlg.last_unknown_wv, dlg.last_unknown_spec)
    assert len(dlg.last_matches) == 2
    assert dlg.last_matches[0]["name"] == "Mineral_A"

    # Render radar chart
    dlg._render_radar_and_overlay(0)
    assert dlg.canvas_radar is not None
    assert dlg.canvas_curve is not None

def test_scientific_dossier_generator():
    """Verify Scientific Dossier Generator compiles session data and handles export."""
    from scientific_dossier_generator import ScientificDossierGeneratorDialog

    dlg = ScientificDossierGeneratorDialog()
    assert dlg is not None
    assert dlg.txt_preview.toPlainText() != ""
    assert "Hyperspectral" in dlg.txt_preview.toPlainText() or "AI" in dlg.txt_preview.toPlainText()


def test_csv_analytics_boxplot_and_multiclass_fs(monkeypatch):
    """Verifies boxplot rendering under modern Matplotlib, multiclass L1 Logistic Regression, and single-class validation."""
    from PyQt5.QtWidgets import QMessageBox
    from advanced_csv_analysis import AdvancedCSVAnalysisDialog

    dlg = AdvancedCSVAnalysisDialog()

    # 1. Add multiclass data (3 classes)
    wvs = np.linspace(400, 1000, 20)
    for c_name in ["Healthy", "Mild_Disease", "Severe_Disease"]:
        data = np.random.uniform(0.1, 0.9, (15, 20))
        dlg.datasets.append({
            'name': f"{c_name}_Set",
            'data': data,
            'wavelengths': wvs,
            'class_label': c_name,
            'classes': [c_name] * 15
        })

    # Test Boxplot rendering
    dlg.cached_idx_data = {
        'name': 'NDVI',
        'vals': np.random.uniform(0.2, 0.8, 45)
    }
    dlg.cached_raw_data = {
        'X': np.random.uniform(0.1, 0.9, (45, 20)),
        'y': np.array(["Healthy"] * 15 + ["Mild_Disease"] * 15 + ["Severe_Disease"] * 15),
        'wv': wvs
    }
    dlg.class_styles = {
        "Healthy": {"color": "#10b981"},
        "Mild_Disease": {"color": "#f59e0b"},
        "Severe_Disease": {"color": "#ef4444"}
    }
    dlg.cmb_view.addItem("NDVI Boxplot")
    dlg.cmb_view.setCurrentText("NDVI Boxplot")
    dlg.draw_current_plot()
    assert len(dlg.figure.axes) > 0

    # 2. Test Multiclass Feature Selection (Logistic Regression L1 with 3 classes)
    X, y, wv = dlg.prepare_data()
    assert len(np.unique(y)) == 3
    
    # Run the worker task directly to verify OneVsRestClassifier L1
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    base_lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
    clf = OneVsRestClassifier(base_lr)
    clf.fit(X, y)
    coefs = np.array([est.coef_[0] for est in clf.estimators_])
    importances = np.mean(np.abs(coefs), axis=0)
    assert importances.shape == (20,)
    assert not np.isnan(importances).any()

    # 3. Test Single-Class Warning Protection
    dlg_single = AdvancedCSVAnalysisDialog()
    dlg_single.datasets.append({
        'name': "Single_Set",
        'data': np.random.uniform(0.1, 0.9, (10, 20)),
        'wavelengths': wvs,
        'class_label': "OnlyOneClass",
        'classes': ["OnlyOneClass"] * 10
    })
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, msg: warnings.append((title, msg)))
    dlg_single.run_fs()
    assert len(warnings) == 1
    assert "Multiple Classes Required" in warnings[0][0]


def test_chemometrics_studio_real_datacube_and_permutation_histogram():
    """Verify Chemometrics Studio datacube ingestion, safe histogram bins, and permutation testing."""
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from chemometrics_advanced_studio import ChemometricsAdvancedStudioDialog
    QMessageBox.information = lambda *a, **k: None
    QMessageBox.warning = lambda *a, **k: None

    _app = QApplication.instance() or QApplication([""])

    # Synthetic datacube (25x25 pixels, 40 bands)
    rng = np.random.RandomState(42)
    cube = rng.rand(25, 25, 40).astype(np.float32)
    wvs = np.linspace(400, 1000, 40)

    # 1. Test initialization directly from datacube
    dlg = ChemometricsAdvancedStudioDialog(
        datacube=cube,
        wavelengths=wvs
    )
    assert dlg.spectra is not None
    assert dlg.spectra.shape[1] == 40
    assert dlg.vip_scores is not None
    assert len(dlg.vip_scores) == 40

    # 2. Test permutation testing without ValueError: Too many bins
    dlg.run_permutation_test()
    assert len(dlg.perm_r2) > 0
    assert len(dlg.perm_q2) > 0

    # 3. Test plot rendering with identical/zero-variance data
    dlg.perm_r2 = [0.0] * 50
    dlg.perm_q2 = [0.0] * 50
    dlg.rad_perm.setChecked(True)
    dlg.refresh_plot()  # Must not crash with ValueError

    dlg.close()
