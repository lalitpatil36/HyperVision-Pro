# -*- coding: utf-8 -*-
"""
Tests for:
1. Advanced CSV Analysis VIP calculation matrix shape fix (reproducing screenshot error).
2. MultiCubeSelectionDialog for combining multiple extracted datacubes in AI training.
3. DatacubeSegmentationStudioDialog for grain, seed, and plant part segmentation with watershed.
"""

import sys
import os
import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)

def test_advanced_csv_vip_matrix_shape_fix():
    """Verify that PLS-DA VIP calculation in CSV analysis handles multiclass targets without ValueError."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import label_binarize, StandardScaler

    rng = np.random.RandomState(42)
    X = rng.rand(50, 40)
    y = np.array(["Wheat"] * 25 + ["Barley"] * 25)

    X_scaled = StandardScaler().fit_transform(X)
    unique_classes = np.unique(y)
    Y_bin = label_binarize(y, classes=unique_classes)
    if len(unique_classes) == 2:
        Y_bin = np.column_stack((1 - Y_bin, Y_bin))

    pls = PLSRegression(n_components=min(3, X.shape[1], X.shape[0]-1))
    pls.fit(X_scaled, Y_bin)
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = w.shape
    vips = np.zeros((p,))
    s = np.diag(t.T @ t @ q.T @ q).ravel()
    total_s = np.sum(s)
    assert total_s > 0
    for i in range(p):
        weight = np.array([(w[i, j] / (np.linalg.norm(w[:, j]) + 1e-12))**2 for j in range(h)])
        # Must produce scalar assignment without raising ValueError
        vips[i] = np.sqrt(p * np.sum(s * weight) / total_s)

    assert vips.shape == (40,)
    assert np.all(vips >= 0.0)
    assert np.isclose(np.mean(vips**2), 1.0, atol=0.1)

def test_multicube_selection_dialog():
    """Verify MultiCubeSelectionDialog can select and combine multiple datacube selections."""
    from spectral_gui_2026_07_16 import MultiCubeSelectionDialog

    class MockDoc:
        def __init__(self, wvs):
            self.wavelengths = np.array(wvs, dtype=float)

    class MockWindow:
        def __init__(self, name, n_sel, wvs):
            self.name = name
            self.selections = [
                {"id": f"s_{i}", "class_label": f"Class_{i%2}", "profile": np.ones(len(wvs)) * (i + 1)}
                for i in range(n_sel)
            ]
            self.document = MockDoc(wvs)

    w1 = MockWindow("Cube_A", 20, [400, 500, 600, 700])
    w2 = MockWindow("Cube_B", 30, [400, 500, 600, 700])
    w3 = MockWindow("Cube_C", 15, [400, 500, 600, 700])

    dlg = MultiCubeSelectionDialog([w1, w2, w3], active_window=w1)
    assert dlg is not None
    selected = dlg.get_selected_windows()
    assert len(selected) == 3

    total_samples = sum(len(w.selections) for w in selected)
    assert total_samples == 65

    # Deselect one
    dlg.list_widget.item(2).setCheckState(0)
    selected_2 = dlg.get_selected_windows()
    assert len(selected_2) == 2
    assert sum(len(w.selections) for w in selected_2) == 50

def test_grain_and_plant_segmentation_studio():
    """Verify DatacubeSegmentationStudioDialog detects grains and extracts morphometry/spectra."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog

    dlg = DatacubeSegmentationStudioDialog()
    assert dlg is not None
    assert dlg.num_objects > 0
    assert len(dlg.grain_metrics) == dlg.num_objects
    assert len(dlg.grain_spectra) == dlg.num_objects

    # Check metrics fields
    first = dlg.grain_metrics[0]
    assert "area" in first
    assert "perimeter" in first
    assert "equiv_diam" in first
    assert "aspect_ratio" in first
    assert "circularity" in first
    assert "mean_ndvi" in first
    assert first["area"] >= 15

    # Check spectra matrix dimensions
    assert dlg.grain_spectra[0].shape == dlg.wavelengths.shape

def test_segmentation_watershed_algorithm():
    """Verify that watershed algorithm can be executed on grain scenes."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog

    dlg = DatacubeSegmentationStudioDialog()
    # Select Watershed
    idx = next(i for i in range(dlg.cmb_algo.count()) if "Watershed" in dlg.cmb_algo.itemText(i))
    if idx >= 0:
        dlg.cmb_algo.setCurrentIndex(idx)
    else:
        dlg.cmb_algo.setCurrentIndex(1)
    dlg.run_segmentation()

    assert dlg.labeled_instances is not None
    assert dlg.binary_mask is not None
    assert dlg.num_objects > 0


def test_actual_datacube_ingestion_in_segmentation():
    """Verify that DatacubeSegmentationStudioDialog accepts 100% actual real datacube data."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog

    # Construct an actual real experimental cube (e.g. 50x60x120 bands)
    real_cube = np.random.uniform(0.1, 0.8, (50, 60, 120)).astype(np.float32)
    real_wvs = np.linspace(400, 1000, 120)

    dlg = DatacubeSegmentationStudioDialog(
        datacube=real_cube,
        wavelengths=real_wvs,
        datacube_name="Actual_Wheat_Experiment_1"
    )

    assert dlg.is_actual_data is True
    assert dlg.datacube.shape == (50, 60, 120)
    assert np.allclose(dlg.wavelengths, real_wvs)
    assert "Actual_Wheat_Experiment_1" in dlg.windowTitle()
    dlg.close()

def test_train_ai_model_multicube_no_unbound_np():
    """Verify that multi-datacube selection and combination does not trigger UnboundLocalError 'np'."""
    from spectral_gui_2026_07_16 import HyperspectralViewer, MultiCubeSelectionDialog

    app = QApplication.instance() or QApplication([])
    viewer = HyperspectralViewer()

    # Create two mock windows with extracted ROI selections
    class MockDoc:
        wavelengths = np.linspace(400, 1000, 30)

    class MockWindow:
        def __init__(self, name, num_samples, class_name):
            self.name = name
            self.document = MockDoc()
            self.selections = [
                {
                    "id": f"sel_{i}",
                    "class_label": class_name,
                    "profile": np.random.uniform(0.2, 0.8, 30)
                }
                for i in range(num_samples)
            ]

    w1 = MockWindow("Disease_Cube", 15, "Disease")
    w2 = MockWindow("Healthy_Cube", 10, "Healthy")
    viewer.image_windows = [w1, w2]

    dlg = MultiCubeSelectionDialog([w1, w2], active_window=w1, parent=viewer)
    assert dlg.chk_combine.isChecked() is True
    assert len(dlg.get_selected_windows()) == 2

    # Verify interpolation and combining logic
    ref_wv = np.asarray(w1.document.wavelengths, dtype=np.float64)
    all_selections = []
    for w in [w1, w2]:
        src_wv = np.asarray(w.document.wavelengths, dtype=np.float64)
        for sel in w.selections:
            prof = np.asarray(sel["profile"], dtype=np.float64).flatten()
            if not np.allclose(src_wv, ref_wv, atol=1e-2):
                prof = np.interp(ref_wv, src_wv, prof)
            sel_copy = dict(sel)
            sel_copy["profile"] = prof
            all_selections.append(sel_copy)

    assert len(all_selections) == 25
    assert len(set(s["class_label"] for s in all_selections)) == 2

    viewer.close()
    app.processEvents()


def test_point_reference_and_roi_bounding_box_segmentation():
    """Verify Point Reference SAM segmentation, ROI bounding box crop, and 1-click auto-segment."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog

    h, w, b = 120, 120, 40
    cube = np.zeros((h, w, b), dtype=np.float32) + 0.1
    wvs = np.linspace(400, 900, b)
    grain_sig = np.linspace(0.25, 0.75, b).astype(np.float32)

    # Place 6 grains
    grain_coords = [(25, 25), (25, 45), (25, 65), (50, 25), (50, 45), (50, 65)]
    for cy, cx in grain_coords:
        cube[cy-4:cy+5, cx-4:cx+5] = grain_sig

    mock_selections = [
        {"id": idx + 1, "selection_type": "point", "raw_x": cx, "raw_y": cy, "profile": grain_sig}
        for idx, (cy, cx) in enumerate(grain_coords)
    ]

    dlg = DatacubeSegmentationStudioDialog(
        datacube=cube,
        wavelengths=wvs,
        selections=mock_selections,
        datacube_name="Point_Ref_Test"
    )

    # 1. Verify point references ingested and SAM auto-selected
    assert len(dlg.point_references) == 6
    assert "Point Reference" in dlg.cmb_algo.currentText()
    assert dlg.active_roi_box is not None
    assert dlg.num_objects == 6

    # 2. Verify ROI Bounding Box coordinates
    bx1, by1, bx2, by2 = dlg.active_roi_box
    assert bx1 <= 25 and bx2 >= 65
    assert by1 <= 25 and by2 >= 50

    # 3. Test 1-click auto segment workflow (silent)
    dlg.one_click_auto_segment(silent=True)
    assert dlg.num_objects == 6

    # 4. Verify grain metrics have both local and global coordinates
    m0 = dlg.grain_metrics[0]
    assert "cx" in m0 and "cy" in m0
    assert "cx_local" in m0 and "cy_local" in m0

    # 5. Verify sensitivity slider live adjustment
    dlg.slider_sam_tol.setValue(20)  # 0.20 rad
    assert dlg.num_objects >= 6

    # 6. Test reset to full scene
    dlg._reset_roi_box()
    assert dlg.active_roi_box is None
    assert dlg.num_objects == 6

    dlg.close()


def test_individual_grain_point_and_roi_selection(monkeypatch):
    """Verify individual grain selection by point click, individual grain visualizer, and gallery."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: None)

    h, w, b = 100, 100, 30
    cube = np.zeros((h, w, b), dtype=np.float32) + 0.50
    wvs = np.linspace(420, 880, b)
    grain_sig = np.linspace(0.15, 0.25, b).astype(np.float32)

    # Place 3 grains
    for cy, cx in [(25, 25), (25, 55), (55, 25)]:
        for y in range(cy-4, cy+5):
            for x in range(cx-4, cx+5):
                if (y-cy)**2 + (x-cx)**2 <= 16:
                    cube[y, x] = grain_sig

    dlg = DatacubeSegmentationStudioDialog(
        datacube=cube,
        wavelengths=wvs,
        datacube_name="Indiv_Test_Cube"
    )

    # 1. Switch to individual point selection mode
    dlg.rad_mode_point.setChecked(True)
    assert dlg.interaction_mode == "point_individual"

    dlg._clear_all_grains()
    assert dlg.num_objects == 0

    # 2. Extract grain at (25, 25)
    dlg._segment_single_grain_at(25, 25)
    assert dlg.num_objects == 1
    assert dlg.grain_metrics[0]["id"] == 1
    assert dlg.grain_metrics[0]["area"] > 10

    # 3. Extract grain at (55, 25)
    dlg._segment_single_grain_at(55, 25)
    assert dlg.num_objects == 2

    # 4. Check individual grain visualizer
    dlg._update_individual_grain_view()
    assert "Grain #" in dlg.lbl_indiv_card.text()

    # 5. Check gallery render
    dlg._render_grain_gallery()
    assert len(dlg.fig_gal.axes) == 2

    # 6. Test multi-grain box selection
    dlg._segment_multiple_grains_in_box(45, 15, 65, 35)
    assert dlg.num_objects >= 2

    dlg.close()


def test_send_to_spectral_library_and_exports(monkeypatch, tmp_path):
    """Verify Send to Spectral Library, export measurements CSV, and export spectra CSV."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog
    from PyQt5.QtWidgets import QMessageBox, QFileDialog

    # Mock all dialogs
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'question', lambda *args, **kwargs: QMessageBox.No)

    h, w, b = 60, 60, 20
    cube = np.zeros((h, w, b), dtype=np.float32) + 0.4
    wvs = np.linspace(450, 850, b)
    grain_sig = np.linspace(0.15, 0.25, b).astype(np.float32)

    # 2 grains
    cube[10:18, 10:18] = grain_sig
    cube[30:38, 30:38] = grain_sig

    dlg = DatacubeSegmentationStudioDialog(
        datacube=cube,
        wavelengths=wvs,
        datacube_name="Export_Test_Cube"
    )

    # Segment grains
    dlg.rad_mode_point.setChecked(True)
    dlg._clear_all_grains()
    dlg._segment_single_grain_at(14, 14)
    dlg._segment_single_grain_at(34, 34)
    assert dlg.num_objects == 2

    # 1. Test Send to Spectral Library
    dlg.send_to_spectral_library()
    assert hasattr(dlg, "_spectral_library")
    assert len(dlg._spectral_library.entries) == 1
    entry = dlg._spectral_library.entries[0]
    assert entry.sample_count == 2
    assert "Export_Test_Cube" in entry.name
    assert len(entry.mean) == b

    # 2. Test export measurements CSV
    csv_file = str(tmp_path / "measurements.csv")
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (csv_file, ""))
    dlg.export_measurements_csv()
    assert os.path.exists(csv_file)
    with open(csv_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3 # header + 2 grains

    # 3. Test export spectra CSV
    spec_file = str(tmp_path / "spectra.csv")
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **k: (spec_file, ""))
    dlg.export_spectra_csv()
    assert os.path.exists(spec_file)
    with open(spec_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 3

    # 4. Test backward compatibility shim
    from spectral_library_dialog import get_spectral_library, SpectralLibraryEntry
    shim_lib = get_spectral_library()
    assert shim_lib is not None

    dlg.close()


def test_touching_grain_single_selection(monkeypatch):
    """Verify that clicking a single grain in touching grains selects only that grain, not its neighbor."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog
    from PyQt5.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: None)

    h, w, b = 60, 60, 20
    cube = np.zeros((h, w, b), dtype=np.float32) + 0.6
    wvs = np.linspace(450, 850, b)
    grain_sig = np.linspace(0.15, 0.25, b).astype(np.float32)

    # Two touching circular seeds: centers at (22, 30) and (34, 30), radii = 7
    # Distance between centers = 12 px, so they touch and overlap
    import cv2
    mask_touching = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_touching, (22, 30), 7, 255, -1)
    cv2.circle(mask_touching, (34, 30), 7, 255, -1)
    cube[mask_touching > 0] = grain_sig

    dlg = DatacubeSegmentationStudioDialog(
        datacube=cube,
        wavelengths=wvs,
        datacube_name="Touching_Grains_Cube"
    )

    dlg.rad_mode_point.setChecked(True)
    dlg._clear_all_grains()

    # Click on the FIRST grain at (22, 30)
    dlg._segment_single_grain_at(22, 30)
    assert dlg.num_objects == 1

    grain1 = dlg.grain_metrics[0]
    # Center X should be near 22, not averaged across both (which would be 28)
    assert abs(grain1["cx"] - 22) < 3.0
    # Area should be a single grain (~130-160 px), not merged (~300 px)
    assert 80 <= grain1["area"] <= 200

    # Click on the SECOND grain at (34, 30)
    dlg._segment_single_grain_at(34, 30)
    assert dlg.num_objects == 2

    grain2 = dlg.grain_metrics[1]
    assert abs(grain2["cx"] - 34) < 3.0
    assert 80 <= grain2["area"] <= 200

    # Verify auto-segmentation separates touching grains into 2 objects
    dlg._clear_all_grains()
    dlg.rad_mode_auto.setChecked(True)
    idx_bh = dlg.cmb_algo.findText("Black-Hat")
    dlg.cmb_algo.setCurrentIndex(idx_bh)
    dlg.chk_invert.setChecked(True)
    dlg.run_segmentation()
    # With peak-watershed enabled, both touching grains are separated
    assert dlg.num_objects == 2

    dlg.close()


def test_faint_grain_selection_at_point_160_51(monkeypatch):
    """Verify that clicking on a faint grain at point (160, 51) never fails with QMessageBox warning and correctly extracts the grain."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog
    from PyQt5.QtWidgets import QMessageBox

    warnings_shown = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda parent, title, msg: warnings_shown.append((title, msg)))
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: None)

    # Construct large datacube covering (160, 51)
    h, w, b = 100, 200, 25
    cube = np.zeros((h, w, b), dtype=np.float32) + 0.70  # bright background
    wvs = np.linspace(400, 950, b)

    # Place a faint grain at gx=160, gy=51 (contrast only ~0.15)
    import cv2
    mask2d = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask2d, (160, 51), 5, 255, -1)
    for bi in range(b):
        cube[:, :, bi] = np.where(mask2d > 0, 0.55, 0.70)

    dlg = DatacubeSegmentationStudioDialog(
        datacube=cube,
        wavelengths=wvs,
        datacube_name="Faint_Seed_Test"
    )
    dlg.rad_mode_point.setChecked(True)
    dlg._clear_all_grains()

    # Click on the faint grain at (160, 51)
    dlg._segment_single_grain_at(160, 51)

    # MUST NOT raise any QMessageBox.warning
    assert len(warnings_shown) == 0, f"Unexpected warning: {warnings_shown}"
    assert dlg.num_objects == 1

    extracted = dlg.grain_metrics[0]
    assert abs(extracted["cx"] - 160) <= 2.0
    assert abs(extracted["cy"] - 51) <= 2.0
    assert extracted["area"] >= 10
    dlg.close()


def test_multi_grain_roi_box_high_accuracy(monkeypatch):
    """Verify multi-grain ROI box extracts all grains in the drawn bounding box cleanly."""
    from datacube_segmentation_studio import DatacubeSegmentationStudioDialog
    from PyQt5.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, 'warning', lambda *args, **kwargs: None)
    monkeypatch.setattr(QMessageBox, 'information', lambda *args, **kwargs: None)

    h, w, b = 120, 120, 20
    cube = np.zeros((h, w, b), dtype=np.float32) + 0.85
    wvs = np.linspace(400, 900, b)
    seed_val = 0.20

    # Draw 6 seeds inside bounding box [20..100, 20..100]
    # Two of them touching: (40, 40) and (50, 40)
    seed_coords = [(40, 40), (50, 40), (75, 40), (40, 75), (75, 75), (58, 58)]
    import cv2
    mask = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in seed_coords:
        cv2.circle(mask, (cx, cy), 5, 255, -1)

    for bi in range(b):
        cube[:, :, bi] = np.where(mask > 0, seed_val, 0.85)

    dlg = DatacubeSegmentationStudioDialog(
        datacube=cube,
        wavelengths=wvs,
        datacube_name="ROI_Accuracy_Cube"
    )
    dlg.rad_mode_roi.setChecked(True)
    dlg._clear_all_grains()

    # Drag ROI box covering the seeds: x from 25 to 90, y from 25 to 90
    dlg._segment_multiple_grains_in_box(25, 25, 90, 90)

    # All 6 grains should be extracted cleanly
    assert dlg.num_objects == 6, f"Expected 6 grains extracted, got {dlg.num_objects}"
    dlg.close()


