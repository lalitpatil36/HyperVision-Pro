import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
from PyQt5.QtWidgets import QApplication
from spectral.io import envi

from spectral_gui import HyperspectralViewer


def test_gui_opens_calibrated_envi_dataset(tmp_path):
    wavelengths = [450, 550, 650, 800, 900]
    metadata = {"wavelength": wavelengths, "wavelength units": "nm"}
    dark = np.full((10, 12, 5), 10, dtype=np.float32)
    white = np.full((10, 12, 5), 110, dtype=np.float32)
    raw = np.full((10, 12, 5), 60, dtype=np.float32)
    envi.save_image(str(tmp_path / "data.hdr"), raw, metadata=metadata, dtype=np.float32, force=True)
    envi.save_image(str(tmp_path / "darkReference.hdr"), dark, metadata=metadata, dtype=np.float32, force=True)
    envi.save_image(str(tmp_path / "whiteReference.hdr"), white, metadata=metadata, dtype=np.float32, force=True)

    app = QApplication.instance() or QApplication([])
    viewer = HyperspectralViewer()
    viewer.load_folder(str(tmp_path))
    app.processEvents()
    assert len(viewer.image_windows) == 1
    window = viewer.image_windows[0]
    assert window.preview_cube is not None
    assert window.preview_qc.mode == "RAW_DARK_WHITE_REFLECTANCE"
    viewer.close()
    app.processEvents()


def test_hyperspectral_viewer_2026_actions_and_exports(monkeypatch):
    """Verifies that the updated HyperspectralViewer (2026 version) has all export and analysis methods and starts without error."""
    from PyQt5.QtWidgets import QMessageBox
    from spectral_gui_2026_07_16 import HyperspectralViewer, MultiCubeSelectionDialog

    # Ensure all required attributes exist on class
    assert hasattr(HyperspectralViewer, "export_profiles")
    assert hasattr(HyperspectralViewer, "export_profiles_wide")
    assert hasattr(HyperspectralViewer, "export_roi_1000")
    assert hasattr(HyperspectralViewer, "export_plot")
    assert hasattr(HyperspectralViewer, "train_ai_model")
    assert hasattr(HyperspectralViewer, "open_datacube_segmentation_studio")

    # Prevent blocking message boxes in headless testing
    warnings_issued = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, msg: warnings_issued.append((title, msg)))

    app = QApplication.instance() or QApplication([])
    viewer = HyperspectralViewer()
    app.processEvents()

    # Test export methods with 0 selections (should show warning and return safely)
    viewer.export_profiles()
    viewer.export_profiles_wide()
    assert len(warnings_issued) >= 1

    viewer.close()
    app.processEvents()


def test_hyperspectral_pipeline_tutorial_dialog():
    """Verifies that the interactive Guided Pipeline Tutorial dialog functions cleanly through all 8 stages."""
    from pipeline_tutorial_dialog import HyperspectralPipelineTutorialDialog, PIPELINE_STEPS
    from spectral_gui_2026_07_16 import HyperspectralViewer

    app = QApplication.instance() or QApplication([])
    viewer = HyperspectralViewer()
    app.processEvents()

    assert hasattr(viewer, "open_pipeline_tutorial")
    assert len(PIPELINE_STEPS) == 8

    dlg = HyperspectralPipelineTutorialDialog(parent_viewer=viewer, initial_step=0)
    assert dlg.current_step == 0
    assert "Calibration" in dlg.lbl_card_title.text()

    # Step through all 8 pipeline stages
    for idx in range(8):
        dlg._load_step(idx)
        assert dlg.current_step == idx
        assert dlg.lbl_stage_badge.text() != ""
        assert dlg.lbl_rationale.text() != ""
        assert dlg.lbl_howto.text() != ""
        assert dlg.lbl_tip.text() != ""

    # Test startup preference toggle
    dlg.chk_startup.setChecked(False)
    assert viewer.user_profile.show_tutorial_on_startup is False
    dlg.chk_startup.setChecked(True)
    assert viewer.user_profile.show_tutorial_on_startup is True

    dlg.close()
    viewer.close()
    app.processEvents()
