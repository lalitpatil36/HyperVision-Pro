# -*- coding: utf-8 -*-
"""
HyperVision Pro - Grain, Seed & Plant Part Segmentation Studio
Advanced spatial-spectral segmentation for grains, seeds, foliage, and plant components.

Features:
- Individual Grain Point Click Selection: Click grains one-by-one to segment and extract them.
- Individual Grain ROI Box Selection: Draw a box around a grain to isolate it.
- Auto-Detect All Grains: Fast SAM or Morphological Black-Hat separation for seed grids on cards/trays.
- Dedicated Individual Grain Segmentation Visualizer: High-magnification zoomed RGB crop, binary mask, single-grain spectrum, and morphology card.
- Grain Gallery Visualizer: Thumbnail grid of all segmented grains.
- 100% actual calibrated data ingestion.
- Full export: Binary/labeled masks, CSV morphometry table, CSV spectral matrix, Spectral Library, Datacube ROIs.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QMessageBox, QGroupBox, QScrollArea, QWidget, QRadioButton, QButtonGroup,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QSlider, QProgressBar, QApplication, QFrame, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QCursor

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.widgets import RectangleSelector
from matplotlib.patches import Rectangle
import cv2
from scipy import ndimage
from sklearn.decomposition import PCA


class DatacubeSegmentationStudioDialog(QDialog):
    """Interactive Grain, Seed, and Plant Part Segmentation Studio."""

    def __init__(
        self,
        datacube=None,
        wavelengths=None,
        parent_viewer=None,
        parent=None,
        datacube_name=None,
        selections=None,
        active_window=None,
        *args,
        **kwargs
    ):
        super().__init__(parent or parent_viewer)
        self.parent_viewer = parent_viewer or parent
        self.active_window = active_window or (
            self.parent_viewer.get_active_window()
            if self.parent_viewer and hasattr(self.parent_viewer, "get_active_window")
            else None
        )
        self.datacube_name = datacube_name or kwargs.get("datacube_name", "Active Datacube")
        self.setWindowTitle(f"HyperVision Pro — Grain, Seed & Plant Part Segmentation Studio [{self.datacube_name}]")
        self.resize(1440, 900)

        self.wavelengths = (
            np.asarray(wavelengths, dtype=np.float64)
            if wavelengths is not None
            else np.linspace(400, 1000, 100)
        )
        self.is_actual_data = (datacube is not None)
        self.datacube = datacube if datacube is not None else self._build_synthetic_grain_scene()

        # Ingest selections from viewer
        if selections is not None:
            self.selections = selections
        elif self.active_window and hasattr(self.active_window, "selections"):
            self.selections = self.active_window.selections
        else:
            self.selections = []

        self.point_references: List[Dict[str, Any]] = []
        self.viewer_roi_boxes: List[Tuple[int, int, int, int]] = []
        self._parse_selections()

        # State variables
        self.active_roi_box: Optional[Tuple[int, int, int, int]] = None
        self.reference_spectrum: Optional[np.ndarray] = None
        self.reference_point_coord: Optional[Tuple[int, int]] = None
        self.picking_reference: bool = False
        self.interaction_mode: str = "auto"  # "auto", "point_individual", "roi_individual"
        self.roi_selector: Optional[RectangleSelector] = None

        self.feature_map: Optional[np.ndarray] = None
        self.binary_mask: Optional[np.ndarray] = None
        self.labeled_instances: Optional[np.ndarray] = None
        self.sub_labeled_instances: Optional[np.ndarray] = None
        self.num_objects: int = 0
        self.grain_metrics: List[Dict[str, Any]] = []
        self.grain_spectra: List[np.ndarray] = []
        self.selected_object_id: Optional[int] = None
        self._updating: bool = False

        # Viewport zoom & pan state
        self.ax_seg = None
        self.view_xlim = None
        self.view_ylim = None
        self._pan_active: bool = False
        self._pan_start = None
        self._pan_xlim = None
        self._pan_ylim = None

        # If point references exist, initialize reference spectrum
        if self.point_references:
            all_profs = [p["profile"] for p in self.point_references if "profile" in p and len(p["profile"]) == len(self.wavelengths)]
            if all_profs:
                self.reference_spectrum = np.mean(all_profs, axis=0).astype(np.float32)

        self._init_ui()
        self._compute_feature_map()

        # Auto-configure if points exist from viewer
        if self.point_references:
            idx_sam = self.cmb_algo.findText("Point Reference Matcher (SAM)", Qt.MatchContains)
            if idx_sam >= 0:
                self.cmb_algo.setCurrentIndex(idx_sam)
            self._auto_fit_box_from_points(silent=True)

        self.run_segmentation()

    def _parse_selections(self):
        """Extract point probes and ROI boxes from viewer selections."""
        for sel in self.selections:
            stype = sel.get("selection_type", "")
            raw_x = sel.get("raw_x")
            raw_y = sel.get("raw_y")
            prof = sel.get("profile")

            cx, cy = raw_x, raw_y
            if self.active_window and hasattr(self.active_window, "raw_to_scene") and raw_x is not None and raw_y is not None:
                try:
                    sx, sy = self.active_window.raw_to_scene(raw_x, raw_y)
                    cx, cy = int(round(sx)), int(round(sy))
                except Exception:
                    pass

            if (stype == "point" or stype == "" or "point" in stype.lower()) and prof is not None:
                self.point_references.append({
                    "id": sel.get("id", len(self.point_references) + 1),
                    "label": sel.get("class_label", f"Point_{len(self.point_references) + 1}"),
                    "raw_x": cx,
                    "raw_y": cy,
                    "profile": np.asarray(prof, dtype=np.float32)
                })
            elif "roi" in stype.lower() or "segmented_object" in stype.lower() or "explicit_whole_roi" in stype.lower():
                marker = sel.get("marker")
                if marker and hasattr(marker, "rect"):
                    r = marker.rect()
                    x1 = int(round(r.left()))
                    y1 = int(round(r.top()))
                    x2 = int(round(r.right()))
                    y2 = int(round(r.bottom()))
                    self.viewer_roi_boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))

    def _build_synthetic_grain_scene(self) -> np.ndarray:
        """Construct a realistic synthetic agricultural scene with scattered grains and plant parts."""
        h, w, b = 160, 200, len(self.wavelengths)
        cube = np.zeros((h, w, b), dtype=np.float32)
        cube += 0.08 + np.random.normal(0, 0.005, (h, w, b)).astype(np.float32)
        wvs = self.wavelengths
        grain_sig = 0.25 + 0.45 * (1.0 / (1.0 + np.exp(-(wvs - 680) / 40))) + 0.1 * np.exp(-((wvs - 850) / 60)**2)
        grain_sig = grain_sig.astype(np.float32)
        leaf_sig = 0.10 + 0.15 * np.exp(-((wvs - 550) / 35)**2) - 0.08 * np.exp(-((wvs - 680) / 25)**2) + 0.60 * (1.0 / (1.0 + np.exp(-(wvs - 710) / 30)))
        leaf_sig = leaf_sig.astype(np.float32)

        np.random.seed(42)
        for i in range(28):
            cy = np.random.randint(20, h - 25)
            cx = np.random.randint(20, w - 25)
            if i % 4 == 0:
                cx += 6
                cy += 4
            ry = np.random.randint(5, 9)
            rx = np.random.randint(8, 14)
            angle = np.random.uniform(0, np.pi)
            yy, xx = np.ogrid[:h, :w]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = (xx - cx) * cos_a + (yy - cy) * sin_a
            y_rot = -(xx - cx) * sin_a + (yy - cy) * cos_a
            mask = ((x_rot / rx)**2 + (y_rot / ry)**2) <= 1.0
            cube[mask] = grain_sig + np.random.normal(0, 0.02, (np.sum(mask), b)).astype(np.float32)

        for _ in range(5):
            cy = np.random.randint(25, h - 25)
            cx = np.random.randint(25, w - 25)
            ry = np.random.randint(12, 20)
            rx = np.random.randint(14, 26)
            angle = np.random.uniform(0, np.pi)
            yy, xx = np.ogrid[:h, :w]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            x_rot = (xx - cx) * cos_a + (yy - cy) * sin_a
            y_rot = -(xx - cx) * sin_a + (yy - cy) * cos_a
            mask = ((x_rot / rx)**2 + (y_rot / ry)**2) <= 1.0
            cube[mask] = leaf_sig + np.random.normal(0, 0.02, (np.sum(mask), b)).astype(np.float32)

        return np.clip(cube, 0.0, 1.2)

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # ----------------------------------------------------------------------
        # LEFT CONTROL PANEL (Scrollable)
        # ----------------------------------------------------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(420)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(10)

        # 0. QUICK AUTO-SEGMENT BANNER & STATUS
        grp_quick = QGroupBox("⚡ Quick Auto-Segmentation")
        grp_quick.setStyleSheet("QGroupBox { font-weight: bold; color: #047857; }")
        l_quick = QVBoxLayout(grp_quick)

        if self.is_actual_data:
            lbl_data_badge = QLabel(f"🟢 100% Actual Calibrated Data: {self.datacube_name}")
            lbl_data_badge.setStyleSheet("color: #059669; font-weight: bold; background: #ecfdf5; border: 1px solid #a7f3d0; padding: 4px; border-radius: 4px;")
            l_quick.addWidget(lbl_data_badge)

        self.btn_one_click = QPushButton("⚡ 1-Click Auto-Segment Grains")
        self.btn_one_click.setStyleSheet("""
            QPushButton {
                background-color: #059669;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 8px;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #047857; }
        """)
        self.btn_one_click.clicked.connect(lambda: self.one_click_auto_segment(silent=False))
        l_quick.addWidget(self.btn_one_click)

        lbl_guide = QLabel("<b>💡 Workflow:</b> 1. Pick reference grain (or click grains individually)  •  2. Drag an ROI box around the grain plate  •  3. Adjust sensitivity slider.")
        lbl_guide.setWordWrap(True)
        lbl_guide.setStyleSheet("color: #475569; font-size: 11px; padding: 2px;")
        l_quick.addWidget(lbl_guide)

        left_layout.addWidget(grp_quick)

        # 1. INDIVIDUAL GRAIN SELECTION TOOLBAR
        grp_indiv = QGroupBox("1. Individual Grain Selection Mode")
        grp_indiv.setStyleSheet("QGroupBox { font-weight: bold; color: #b45309; }")
        l_indiv = QVBoxLayout(grp_indiv)

        self.btn_group_mode = QButtonGroup(self)
        self.rad_mode_auto = QRadioButton("🌐 Auto-Detect All Grains in ROI / Scene")
        self.rad_mode_point = QRadioButton("🎯 Point Click: Select Grains Individually")
        self.rad_mode_roi = QRadioButton("📐 ROI Box: Select Individual Grain / Group")
        self.rad_mode_auto.setChecked(True)

        self.btn_group_mode.addButton(self.rad_mode_auto)
        self.btn_group_mode.addButton(self.rad_mode_point)
        self.btn_group_mode.addButton(self.rad_mode_roi)

        self.rad_mode_auto.toggled.connect(self._on_mode_toggled)
        self.rad_mode_point.toggled.connect(self._on_mode_toggled)
        self.rad_mode_roi.toggled.connect(self._on_mode_toggled)

        l_indiv.addWidget(self.rad_mode_auto)
        l_indiv.addWidget(self.rad_mode_point)
        l_indiv.addWidget(self.rad_mode_roi)

        self.lbl_indiv_status = QLabel("Mode: Auto-Detect All Grains")
        self.lbl_indiv_status.setStyleSheet("color: #0369a1; font-weight: bold; font-size: 11px;")
        l_indiv.addWidget(self.lbl_indiv_status)

        h_indiv_acts = QHBoxLayout()
        self.btn_del_last = QPushButton("🗑️ Remove Last")
        self.btn_del_last.setStyleSheet("background-color: #f1f5f9; padding: 3px;")
        self.btn_del_last.clicked.connect(self._remove_last_grain)
        h_indiv_acts.addWidget(self.btn_del_last)

        self.btn_clear_grains = QPushButton("🗑️ Clear All Grains")
        self.btn_clear_grains.setStyleSheet("background-color: #f1f5f9; padding: 3px;")
        self.btn_clear_grains.clicked.connect(self._clear_all_grains)
        h_indiv_acts.addWidget(self.btn_clear_grains)
        l_indiv.addLayout(h_indiv_acts)

        left_layout.addWidget(grp_indiv)

        # 2. REGION OF INTEREST (ROI) & CROP SCOPE
        grp_roi = QGroupBox("2. Analysis Region & ROI Scope")
        grp_roi.setStyleSheet("QGroupBox { font-weight: bold; color: #1e3a8a; }")
        l_roi = QVBoxLayout(grp_roi)

        l_roi.addWidget(QLabel("<b>Analysis Scope / Bounding Box:</b>"))
        self.cmb_region_scope = QComboBox()
        self.cmb_region_scope.addItems([
            "🌐 Entire Datacube Scene",
            "🎯 Active Viewer ROI",
            "📐 Auto-Bounding Box from Reference Points",
            "✏️ Custom ROI Box (Interactive Drag)"
        ])
        self.cmb_region_scope.currentIndexChanged.connect(self._on_region_scope_changed)
        l_roi.addWidget(self.cmb_region_scope)

        h_roi_btns = QHBoxLayout()
        self.btn_drag_roi = QPushButton("📐 Drag ROI Box on Image")
        self.btn_drag_roi.setStyleSheet("background-color: #f1f5f9; font-weight: bold; padding: 4px;")
        self.btn_drag_roi.clicked.connect(self._toggle_drag_roi_box)
        h_roi_btns.addWidget(self.btn_drag_roi)

        self.btn_reset_roi = QPushButton("🔄 Reset Full Scene")
        self.btn_reset_roi.setStyleSheet("background-color: #f1f5f9; padding: 4px;")
        self.btn_reset_roi.clicked.connect(self._reset_roi_box)
        h_roi_btns.addWidget(self.btn_reset_roi)
        l_roi.addLayout(h_roi_btns)

        self.lbl_roi_box_info = QLabel("Active Box: None (Full Scene)")
        self.lbl_roi_box_info.setStyleSheet("color: #64748b; font-size: 11px; font-weight: bold;")
        l_roi.addWidget(self.lbl_roi_box_info)

        self.chk_zoom_roi = QCheckBox("🔍 Zoom In to Active ROI (Focus on Grains)")
        self.chk_zoom_roi.setChecked(False)
        self.chk_zoom_roi.toggled.connect(lambda chk: self._zoom_to_roi() if chk else self._reset_zoom())
        l_roi.addWidget(self.chk_zoom_roi)

        left_layout.addWidget(grp_roi)

        # 3. POINT REFERENCE SPECTRAL MATCHER
        grp_ref = QGroupBox("3. Point Reference Spectral Matching")
        grp_ref.setStyleSheet("QGroupBox { font-weight: bold; color: #1e3a8a; }")
        l_ref = QVBoxLayout(grp_ref)

        l_ref.addWidget(QLabel("<b>Reference Spectrum Source:</b>"))
        self.cmb_point_ref = QComboBox()
        self._populate_point_ref_combo()
        self.cmb_point_ref.currentIndexChanged.connect(self._on_point_ref_changed)
        l_ref.addWidget(self.cmb_point_ref)

        self.btn_pick_grain = QPushButton("👆 Pick Reference Grain on Canvas")
        self.btn_pick_grain.setStyleSheet("background-color: #f8fafc; font-weight: bold; padding: 5px;")
        self.btn_pick_grain.clicked.connect(self._toggle_pick_grain)
        l_ref.addWidget(self.btn_pick_grain)

        self.lbl_sam_tol = QLabel("<b>Match Sensitivity (SAM Angle):</b> 0.12 rad (~7°)")
        self.lbl_sam_tol.setStyleSheet("color: #0284c7; font-weight: bold;")
        l_ref.addWidget(self.lbl_sam_tol)

        h_sam = QHBoxLayout()
        self.slider_sam_tol = QSlider(Qt.Horizontal)
        self.slider_sam_tol.setRange(2, 40)
        self.slider_sam_tol.setValue(12)
        self.slider_sam_tol.valueChanged.connect(self._on_sam_slider_changed)
        h_sam.addWidget(self.slider_sam_tol)

        self.spin_sam_tol = QDoubleSpinBox()
        self.spin_sam_tol.setRange(0.01, 0.50)
        self.spin_sam_tol.setSingleStep(0.01)
        self.spin_sam_tol.setValue(0.12)
        self.spin_sam_tol.valueChanged.connect(self._on_sam_spin_changed)
        h_sam.addWidget(self.spin_sam_tol)
        l_ref.addLayout(h_sam)

        lbl_sam_hint = QLabel("<small style='color:#64748b;'>Strict (fewer pixels) ◄──────── 0.12 rad ────────► Broad (more pixels)</small>")
        l_ref.addWidget(lbl_sam_hint)

        left_layout.addWidget(grp_ref)

        # 4. SEGMENTATION ENGINE GROUP
        grp_seg = QGroupBox("4. Segmentation Engine")
        grp_seg.setStyleSheet("QGroupBox { font-weight: bold; color: #1e3a8a; }")
        l_seg = QVBoxLayout(grp_seg)

        l_seg.addWidget(QLabel("<b>Segmentation Algorithm:</b>"))
        self.cmb_algo = QComboBox()
        self.cmb_algo.addItems([
            "🎯 Point Reference Matcher (SAM - Spectral Angle Mapper)",
            "🧬 Point Reference Matcher (Spectral Pearson Correlation)",
            "🌾 Black-Hat Contrast Extractor (Grains on Card/Tray)",
            "⚡ Otsu Bimodal Thresholding (Auto)",
            "🌊 Watershed Instance Separation (Touch Grains)",
            "📈 Spectral Index Cutoff (Slider)",
            "🌓 Adaptive Gaussian Local Threshold",
            "🎚️ Dual Manual Threshold (Bandpass)"
        ])
        self.cmb_algo.currentIndexChanged.connect(self._on_algo_changed)
        l_seg.addWidget(self.cmb_algo)

        l_seg.addWidget(QLabel("<b>Feature Map Channel:</b>"))
        self.cmb_channel = QComboBox()
        self.cmb_channel.addItems([
            "🌾 Grain & Seed Brightness Index (GBI)",
            "🌿 NDVI (Normalized Difference Veg)",
            "🌱 NDRE (Red Edge Biomass)",
            "🔬 PRI (Photochemical Reflectance)",
            "💧 Water Absorption Index (WI)",
            "📊 PCA (First Principal Component)",
            "🎯 Single Wavelength Band"
        ])
        self.cmb_channel.currentIndexChanged.connect(self._on_channel_changed)
        l_seg.addWidget(self.cmb_channel)

        self.lbl_band = QLabel("Selected Wavelength:")
        l_seg.addWidget(self.lbl_band)
        self.slider_band = QSlider(Qt.Horizontal)
        self.slider_band.setRange(0, len(self.wavelengths) - 1)
        self.slider_band.setValue(len(self.wavelengths) // 2)
        self.slider_band.valueChanged.connect(self._on_band_slider_changed)
        l_seg.addWidget(self.slider_band)
        self.lbl_band_val = QLabel(f"{self.wavelengths[len(self.wavelengths)//2]:.1f} nm")
        self.lbl_band_val.setStyleSheet("color: #0284c7; font-weight: bold;")
        l_seg.addWidget(self.lbl_band_val)
        self.lbl_band.setVisible(False)
        self.slider_band.setVisible(False)
        self.lbl_band_val.setVisible(False)

        self.lbl_thresh = QLabel("Threshold Cutoff:")
        l_seg.addWidget(self.lbl_thresh)
        self.spin_thresh = QDoubleSpinBox()
        self.spin_thresh.setRange(-1.0, 1.0)
        self.spin_thresh.setSingleStep(0.02)
        self.spin_thresh.setValue(0.30)
        self.spin_thresh.valueChanged.connect(self.run_segmentation)
        l_seg.addWidget(self.spin_thresh)
        self.lbl_thresh.setVisible(False)
        self.spin_thresh.setVisible(False)

        self.chk_invert = QCheckBox("Invert Polarity (Background is Brighter)")
        self.chk_invert.toggled.connect(self.run_segmentation)
        l_seg.addWidget(self.chk_invert)

        left_layout.addWidget(grp_seg)

        # 5. MORPHOLOGICAL CLEANING & FILTERING
        grp_morph = QGroupBox("5. Morphological Post-Processing")
        grp_morph.setStyleSheet("QGroupBox { font-weight: bold; color: #1e3a8a; }")
        l_morph = QGridLayout(grp_morph)

        l_morph.addWidget(QLabel("Noise Blur Kernel:"), 0, 0)
        self.spin_blur = QSpinBox()
        self.spin_blur.setRange(1, 15)
        self.spin_blur.setSingleStep(2)
        self.spin_blur.setValue(3)
        self.spin_blur.valueChanged.connect(self.run_segmentation)
        l_morph.addWidget(self.spin_blur, 0, 1)

        l_morph.addWidget(QLabel("Min Object Area (px):"), 1, 0)
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(1, 50000)
        self.spin_min_area.setValue(12)
        self.spin_min_area.valueChanged.connect(self.run_segmentation)
        l_morph.addWidget(self.spin_min_area, 1, 1)

        l_morph.addWidget(QLabel("Max Object Area (px):"), 2, 0)
        self.spin_max_area = QSpinBox()
        self.spin_max_area.setRange(50, 500000)
        self.spin_max_area.setValue(800)  # Grains are small dots, reject whole card
        self.spin_max_area.valueChanged.connect(self.run_segmentation)
        l_morph.addWidget(self.spin_max_area, 2, 1)

        self.chk_fill_holes = QCheckBox("Fill Internal Holes")
        self.chk_fill_holes.setChecked(True)
        self.chk_fill_holes.toggled.connect(self.run_segmentation)
        l_morph.addWidget(self.chk_fill_holes, 3, 0, 1, 2)

        self.chk_opening = QCheckBox("Morphological Opening (Dust Rejection)")
        self.chk_opening.setChecked(True)
        self.chk_opening.toggled.connect(self.run_segmentation)
        l_morph.addWidget(self.chk_opening, 4, 0, 1, 2)

        self.chk_split_touching = QCheckBox("⚡ Separate Touching Grains (Watershed)")
        self.chk_split_touching.setChecked(True)
        self.chk_split_touching.setStyleSheet("color: #047857; font-weight: bold;")
        self.chk_split_touching.setToolTip("Uses peak distance-transform watershed to separate touching or closely-spaced grains into individual instances")
        self.chk_split_touching.toggled.connect(self.run_segmentation)
        l_morph.addWidget(self.chk_split_touching, 5, 0, 1, 2)

        l_morph.addWidget(QLabel("Grain Separation Distance (px):"), 6, 0)
        self.spin_split_dist = QSpinBox()
        self.spin_split_dist.setRange(3, 40)
        self.spin_split_dist.setValue(7)
        self.spin_split_dist.setToolTip("Minimum distance between grain peaks to split touching seeds (e.g. 5–9 px for closely packed wells)")
        self.spin_split_dist.valueChanged.connect(self.run_segmentation)
        l_morph.addWidget(self.spin_split_dist, 6, 1)

        left_layout.addWidget(grp_morph)

        # 6. ACTION & EXPORT BUTTONS
        grp_act = QGroupBox("6. Export & Pipeline Actions")
        grp_act.setStyleSheet("QGroupBox { font-weight: bold; color: #1e3a8a; }")
        l_act = QVBoxLayout(grp_act)

        btn_mask = QPushButton("💾 Export Binary / Instance Mask")
        btn_mask.clicked.connect(self.export_mask)
        l_act.addWidget(btn_mask)

        btn_csv = QPushButton("📊 Export Grain Measurements (CSV)")
        btn_csv.clicked.connect(self.export_measurements_csv)
        l_act.addWidget(btn_csv)

        btn_spec = QPushButton("🧬 Export Grain Spectra (CSV)")
        btn_spec.clicked.connect(self.export_spectra_csv)
        l_act.addWidget(btn_spec)

        btn_lib = QPushButton("📚 Send to Spectral Library")
        btn_lib.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 6px;")
        btn_lib.clicked.connect(self.send_to_spectral_library)
        l_act.addWidget(btn_lib)

        btn_roi = QPushButton("🎯 Create Datacube ROIs from Grains")
        btn_roi.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px;")
        btn_roi.clicked.connect(self.create_datacube_rois)
        l_act.addWidget(btn_roi)

        left_layout.addWidget(grp_act)
        left_layout.addStretch()

        left_scroll.setWidget(left_widget)
        main_layout.addWidget(left_scroll)

        # ----------------------------------------------------------------------
        # RIGHT DISPLAY TABS & TABLE (Splitter)
        # ----------------------------------------------------------------------
        right_splitter = QSplitter(Qt.Vertical)

        # Tabs for Visualizations
        self.tab_vis = QTabWidget()

        # Tab 1: Full Scene & Overlay Visualizer
        tab_seg_w = QWidget()
        l_tab1 = QVBoxLayout(tab_seg_w)
        l_tab1.setContentsMargins(4, 4, 4, 4)

        # Dedicated Zoom & Focus Control Bar for easy grain selection
        h_zoom_bar = QHBoxLayout()
        h_zoom_bar.setSpacing(6)

        self.btn_zoom_roi = QPushButton("🔍 Zoom to Card / ROI")
        self.btn_zoom_roi.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.btn_zoom_roi.setToolTip("Zoom directly into the grain card or active ROI box for easy, magnified grain selection")
        self.btn_zoom_roi.clicked.connect(self._zoom_to_roi)
        h_zoom_bar.addWidget(self.btn_zoom_roi)

        self.btn_zoom_in = QPushButton("➕ Zoom In")
        self.btn_zoom_in.setStyleSheet("background-color: #f1f5f9; color: #1e293b; font-weight: bold; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 4px;")
        self.btn_zoom_in.setToolTip("Zoom in 1.4x centered on current view")
        self.btn_zoom_in.clicked.connect(lambda: self._zoom_in(1.4))
        h_zoom_bar.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("➖ Zoom Out")
        self.btn_zoom_out.setStyleSheet("background-color: #f1f5f9; color: #1e293b; font-weight: bold; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 4px;")
        self.btn_zoom_out.setToolTip("Zoom out 1.4x centered on current view")
        self.btn_zoom_out.clicked.connect(lambda: self._zoom_out(1.4))
        h_zoom_bar.addWidget(self.btn_zoom_out)

        self.btn_zoom_fit = QPushButton("🔄 Fit Full Scene")
        self.btn_zoom_fit.setStyleSheet("background-color: #f1f5f9; color: #1e293b; font-weight: bold; padding: 4px 8px; border: 1px solid #cbd5e1; border-radius: 4px;")
        self.btn_zoom_fit.setToolTip("Reset zoom to display entire datacube scene (100%)")
        self.btn_zoom_fit.clicked.connect(self._reset_zoom)
        h_zoom_bar.addWidget(self.btn_zoom_fit)

        self.lbl_zoom_level = QLabel("Zoom: 100%")
        self.lbl_zoom_level.setStyleSheet("color: #475569; font-weight: bold; font-size: 11px; padding: 0 4px;")
        h_zoom_bar.addWidget(self.lbl_zoom_level)

        self.chk_auto_zoom_roi = QCheckBox("Auto-Zoom to ROI")
        self.chk_auto_zoom_roi.setChecked(True)
        self.chk_auto_zoom_roi.setToolTip("Automatically magnify when an ROI box is drawn or selected")
        h_zoom_bar.addWidget(self.chk_auto_zoom_roi)

        lbl_zoom_hint = QLabel("💡 <i>Scroll wheel: Zoom at cursor | Right-drag: Pan</i>")
        lbl_zoom_hint.setStyleSheet("color: #64748b; font-size: 10.5px;")
        h_zoom_bar.addWidget(lbl_zoom_hint)
        h_zoom_bar.addStretch(1)

        l_tab1.addLayout(h_zoom_bar)

        self.fig_seg = Figure(figsize=(9, 5), dpi=100)
        self.canvas_seg = FigureCanvasQTAgg(self.fig_seg)
        self.canvas_seg.mpl_connect("button_press_event", self._on_canvas_press)
        self.canvas_seg.mpl_connect("button_release_event", self._on_canvas_release)
        self.canvas_seg.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas_seg.mpl_connect("scroll_event", self._on_canvas_scroll)
        self.toolbar_seg = NavigationToolbar2QT(self.canvas_seg, self)
        l_tab1.addWidget(self.toolbar_seg)
        l_tab1.addWidget(self.canvas_seg)
        self.tab_vis.addTab(tab_seg_w, "🔬 Full Scene & Contours Visualizer")

        # Tab 2: Individual Grain Detail & Mask (NEW!)
        tab_indiv_w = QWidget()
        l_tab2 = QVBoxLayout(tab_indiv_w)
        l_tab2.setContentsMargins(4, 4, 4, 4)

        # Top navigator for individual grains
        h_indiv_nav = QHBoxLayout()
        self.btn_prev_grain = QPushButton("◀ Previous Grain")
        self.btn_prev_grain.clicked.connect(self._prev_individual_grain)
        h_indiv_nav.addWidget(self.btn_prev_grain)

        self.cmb_grain_selector = QComboBox()
        self.cmb_grain_selector.currentIndexChanged.connect(self._on_grain_selector_changed)
        h_indiv_nav.addWidget(self.cmb_grain_selector, 1)

        self.btn_next_grain = QPushButton("Next Grain ▶")
        self.btn_next_grain.clicked.connect(self._next_individual_grain)
        h_indiv_nav.addWidget(self.btn_next_grain)
        l_tab2.addLayout(h_indiv_nav)

        self.fig_indiv = Figure(figsize=(9, 4.5), dpi=100)
        self.canvas_indiv = FigureCanvasQTAgg(self.fig_indiv)
        l_tab2.addWidget(self.canvas_indiv)

        self.lbl_indiv_card = QLabel("Select a grain to view zoomed morphology, binary mask, and individual spectrum.")
        self.lbl_indiv_card.setStyleSheet("background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 4px; padding: 6px; font-weight: bold; color: #1e293b;")
        l_tab2.addWidget(self.lbl_indiv_card)

        self.tab_vis.addTab(tab_indiv_w, "🔍 Individual Grain Detail & Mask")

        # Tab 3: Grain Gallery (Grid of Thumbnails) (NEW!)
        tab_gal_w = QWidget()
        l_tab3 = QVBoxLayout(tab_gal_w)
        l_tab3.setContentsMargins(4, 4, 4, 4)
        self.fig_gal = Figure(figsize=(9, 5), dpi=100)
        self.canvas_gal = FigureCanvasQTAgg(self.fig_gal)
        l_tab3.addWidget(self.canvas_gal)
        self.tab_vis.addTab(tab_gal_w, "🖼️ Grain Gallery (All Seeds)")

        # Tab 4: Morphology Distributions
        tab_dist_w = QWidget()
        l_tab4 = QVBoxLayout(tab_dist_w)
        l_tab4.setContentsMargins(4, 4, 4, 4)
        self.fig_dist = Figure(figsize=(9, 5), dpi=100)
        self.canvas_dist = FigureCanvasQTAgg(self.fig_dist)
        l_tab4.addWidget(self.canvas_dist)
        self.tab_vis.addTab(tab_dist_w, "📊 Particle Size & Shape Distributions")

        # Tab 5: Individual Grain Spectral Curves
        tab_spec_w = QWidget()
        l_tab5 = QVBoxLayout(tab_spec_w)
        l_tab5.setContentsMargins(4, 4, 4, 4)
        self.fig_spec = Figure(figsize=(9, 5), dpi=100)
        self.canvas_spec = FigureCanvasQTAgg(self.fig_spec)
        self.toolbar_spec = NavigationToolbar2QT(self.canvas_spec, self)
        l_tab5.addWidget(self.toolbar_spec)
        l_tab5.addWidget(self.canvas_spec)
        self.tab_vis.addTab(tab_spec_w, "🧬 Individual Grain Spectra & Outliers")

        right_splitter.addWidget(self.tab_vis)

        # Bottom Panel: Grain Measurements Table & Summary
        bottom_widget = QWidget()
        l_bot = QVBoxLayout(bottom_widget)
        l_bot.setContentsMargins(4, 4, 4, 4)

        h_bot = QHBoxLayout()
        self.lbl_stats = QLabel("Detected Grains: 0 | Total Area: 0 px | Coverage: 0.0%")
        self.lbl_stats.setStyleSheet("font-size: 12px; font-weight: bold; color: #1e3a8a; padding: 4px;")
        h_bot.addWidget(self.lbl_stats)
        h_bot.addStretch()

        self.cmb_overlay_mode = QComboBox()
        self.cmb_overlay_mode.addItems(["Contours on RGB", "Labeled Instances Mask", "Binary Mask Only", "Feature Map"])
        self.cmb_overlay_mode.currentIndexChanged.connect(self.render_visualizations)
        h_bot.addWidget(QLabel("<b>Display Overlay:</b>"))
        h_bot.addWidget(self.cmb_overlay_mode)
        l_bot.addLayout(h_bot)

        self.tbl_grains = QTableWidget(0, 8)
        self.tbl_grains.setHorizontalHeaderLabels([
            "Object ID", "Type / Class", "Centroid (X, Y)", "Area (px)",
            "Equiv. Diam.", "Aspect Ratio", "Circularity", "Mean NDVI"
        ])
        self.tbl_grains.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_grains.itemSelectionChanged.connect(self._on_table_row_selected)
        l_bot.addWidget(self.tbl_grains)

        right_splitter.addWidget(bottom_widget)
        right_splitter.setSizes([500, 260])

        main_layout.addWidget(right_splitter, 1)

    # --------------------------------------------------------------------------
    # INTERACTION MODES & GRAIN SELECTION
    # --------------------------------------------------------------------------
    def _on_mode_toggled(self):
        if self.rad_mode_point.isChecked():
            self.interaction_mode = "point_individual"
            self.lbl_indiv_status.setText("Mode: Click grains on image to add them individually")
            self.lbl_indiv_status.setStyleSheet("color: #b45309; font-weight: bold;")
            QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
            if hasattr(self, "roi_selector") and self.roi_selector:
                self.roi_selector.set_active(False)
        elif self.rad_mode_roi.isChecked():
            self.interaction_mode = "roi_individual"
            self.lbl_indiv_status.setText("Mode: Drag box around a grain to segment it")
            self.lbl_indiv_status.setStyleSheet("color: #b45309; font-weight: bold;")
            QApplication.restoreOverrideCursor()
            if hasattr(self, "roi_selector") and self.roi_selector:
                self.roi_selector.set_active(True)
        else:
            self.interaction_mode = "auto"
            self.lbl_indiv_status.setText("Mode: Auto-Detect All Grains")
            self.lbl_indiv_status.setStyleSheet("color: #0369a1; font-weight: bold;")
            QApplication.restoreOverrideCursor()
            if hasattr(self, "roi_selector") and self.roi_selector:
                self.roi_selector.set_active(True)

    def _remove_last_grain(self):
        if self.grain_metrics:
            self.grain_metrics.pop()
            if self.grain_spectra:
                self.grain_spectra.pop()
            self.num_objects = len(self.grain_metrics)
            self._rebuild_labeled_masks()
            self._populate_table()
            self._populate_grain_selector()
            self.render_visualizations()

    def _clear_all_grains(self):
        self.grain_metrics.clear()
        self.grain_spectra.clear()
        self.num_objects = 0
        h, w = self.datacube.shape[:2]
        self.labeled_instances = np.zeros((h, w), dtype=np.int32)
        self.sub_labeled_instances = np.zeros((h, w), dtype=np.int32)
        self.binary_mask = np.zeros((h, w), dtype=np.uint8)
        self._populate_table()
        self._populate_grain_selector()
        self.render_visualizations()

    def _rebuild_labeled_masks(self):
        h, w = self.datacube.shape[:2]
        full_labeled = np.zeros((h, w), dtype=np.int32)
        for idx, m in enumerate(self.grain_metrics, start=1):
            m["id"] = idx
            cnts = m.get("contours_global", [])
            for cnt in cnts:
                cv2.drawContours(full_labeled, [cnt], -1, idx, -1)
        self.labeled_instances = full_labeled
        self.binary_mask = (full_labeled > 0).astype(np.uint8) * 255
        self.num_objects = len(self.grain_metrics)

    # --------------------------------------------------------------------------
    # POINT REFERENCE & REGION SCOPE HELPERS
    # --------------------------------------------------------------------------
    def _populate_point_ref_combo(self):
        self.cmb_point_ref.clear()
        if self.point_references:
            self.cmb_point_ref.addItem(f"⭐ Average of Selected Points ({len(self.point_references)} points)")
            for p in self.point_references:
                rx = p.get("raw_x", "?")
                ry = p.get("raw_y", "?")
                lbl = p.get("label", f"Point_{p.get('id')}")
                lbl_clean = lbl.split("|")[0].strip()
                self.cmb_point_ref.addItem(f"🎯 {lbl_clean} (X={rx}, Y={ry})")
        else:
            self.cmb_point_ref.addItem("👆 No Viewer Points — Click Below to Pick")

    def _get_point_coordinates(self) -> List[Tuple[int, int]]:
        pts = []
        for p in self.point_references:
            rx = p.get("raw_x")
            ry = p.get("raw_y")
            if rx is not None and ry is not None and isinstance(rx, (int, float)) and isinstance(ry, (int, float)):
                pts.append((int(rx), int(ry)))
        return pts

    def _on_point_ref_changed(self):
        txt = self.cmb_point_ref.currentText()
        if "Average of Selected Points" in txt and self.point_references:
            profs = [p["profile"] for p in self.point_references if "profile" in p and len(p["profile"]) == len(self.wavelengths)]
            if profs:
                self.reference_spectrum = np.mean(profs, axis=0).astype(np.float32)
                self.reference_point_coord = None
        elif "🎯" in txt and self.point_references:
            idx = self.cmb_point_ref.currentIndex() - 1
            if 0 <= idx < len(self.point_references):
                p = self.point_references[idx]
                self.reference_spectrum = p["profile"].astype(np.float32)
                if p.get("raw_x") is not None and p.get("raw_y") is not None:
                    self.reference_point_coord = (int(p["raw_x"]), int(p["raw_y"]))

        if "Point Reference" in self.cmb_algo.currentText():
            self.run_segmentation()

    def _toggle_pick_grain(self):
        self.picking_reference = not self.picking_reference
        if self.picking_reference:
            self.btn_pick_grain.setText("👆 Click on Any Grain in Image...")
            self.btn_pick_grain.setStyleSheet("background-color: #ea580c; color: white; font-weight: bold; padding: 5px;")
            QApplication.setOverrideCursor(QCursor(Qt.CrossCursor))
        else:
            self.btn_pick_grain.setText("👆 Pick Reference Grain on Canvas")
            self.btn_pick_grain.setStyleSheet("background-color: #f8fafc; font-weight: bold; padding: 5px;")
            QApplication.restoreOverrideCursor()

    # --------------------------------------------------------------------------
    # INTERACTIVE ZOOM & PAN CONTROLS (FOR EASY GRAIN SELECTION)
    # --------------------------------------------------------------------------
    def _update_zoom_label(self):
        if not hasattr(self, "ax_seg") or self.ax_seg is None or not hasattr(self, "lbl_zoom_level"):
            return
        h, w = self.datacube.shape[:2]
        x1, x2 = self.ax_seg.get_xlim()
        cur_w = abs(x2 - x1)
        if cur_w > 0:
            pct = int(round((w / cur_w) * 100))
            self.lbl_zoom_level.setText(f"Zoom: {pct}%")

    def _zoom_in(self, factor: float = 1.4):
        """Zoom in centered on the current canvas view."""
        if not hasattr(self, "ax_seg") or self.ax_seg is None:
            return
        x1, x2 = self.ax_seg.get_xlim()
        y1, y2 = self.ax_seg.get_ylim()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = (x2 - x1) / (2.0 * factor)
        dy = (y1 - y2) / (2.0 * factor)
        self.view_xlim = (cx - dx, cx + dx)
        self.view_ylim = (cy + dy, cy - dy)
        self.ax_seg.set_xlim(self.view_xlim)
        self.ax_seg.set_ylim(self.view_ylim)
        self._update_zoom_label()
        self.canvas_seg.draw_idle()

    def _zoom_out(self, factor: float = 1.4):
        """Zoom out centered on the current canvas view."""
        if not hasattr(self, "ax_seg") or self.ax_seg is None:
            return
        h, w = self.datacube.shape[:2]
        x1, x2 = self.ax_seg.get_xlim()
        y1, y2 = self.ax_seg.get_ylim()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dx = ((x2 - x1) * factor) / 2.0
        dy = ((y1 - y2) * factor) / 2.0
        new_x1 = max(-0.5, cx - dx)
        new_x2 = min(w - 0.5, cx + dx)
        new_y1 = min(h - 0.5, cy + dy)
        new_y2 = max(-0.5, cy - dy)
        if (new_x2 - new_x1) >= (w - 1) and (new_y1 - new_y2) >= (h - 1):
            self._reset_zoom()
            return
        self.view_xlim = (new_x1, new_x2)
        self.view_ylim = (new_y1, new_y2)
        self.ax_seg.set_xlim(self.view_xlim)
        self.ax_seg.set_ylim(self.view_ylim)
        self._update_zoom_label()
        self.canvas_seg.draw_idle()

    def _zoom_to_roi(self):
        """Zoom view directly to the grain card or active ROI bounding box."""
        if not hasattr(self, "ax_seg") or self.ax_seg is None:
            return
        h, w = self.datacube.shape[:2]
        if self.active_roi_box is not None:
            rx1, ry1, rx2, ry2 = self.active_roi_box
        elif self.viewer_roi_boxes:
            rx1, ry1, rx2, ry2 = self.viewer_roi_boxes[0]
            self.active_roi_box = (rx1, ry1, rx2, ry2)
        elif self.grain_metrics:
            cxs = [m["cx"] for m in self.grain_metrics]
            cys = [m["cy"] for m in self.grain_metrics]
            rx1, rx2 = max(0, min(cxs) - 15), min(w, max(cxs) + 15)
            ry1, ry2 = max(0, min(cys) - 15), min(h, max(cys) + 15)
        else:
            rx1, rx2 = int(w * 0.25), int(w * 0.85)
            ry1, ry2 = 0, int(h * 0.35)

        pad_x = max(10, (rx2 - rx1) * 0.08)
        pad_y = max(10, (ry2 - ry1) * 0.08)
        x1 = max(-0.5, rx1 - pad_x)
        x2 = min(w - 0.5, rx2 + pad_x)
        y1 = min(h - 0.5, ry2 + pad_y)
        y2 = max(-0.5, ry1 - pad_y)

        self.view_xlim = (x1, x2)
        self.view_ylim = (y1, y2)
        self.ax_seg.set_xlim(self.view_xlim)
        self.ax_seg.set_ylim(self.view_ylim)
        self._update_zoom_label()
        if hasattr(self, "chk_zoom_roi"):
            self.chk_zoom_roi.blockSignals(True)
            self.chk_zoom_roi.setChecked(True)
            self.chk_zoom_roi.blockSignals(False)
        self.canvas_seg.draw_idle()

    def _reset_zoom(self):
        """Reset view limits to fit the full scene."""
        if not hasattr(self, "ax_seg") or self.ax_seg is None:
            return
        h, w = self.datacube.shape[:2]
        self.view_xlim = (-0.5, w - 0.5)
        self.view_ylim = (h - 0.5, -0.5)
        self.ax_seg.set_xlim(self.view_xlim)
        self.ax_seg.set_ylim(self.view_ylim)
        self._update_zoom_label()
        if hasattr(self, "chk_zoom_roi"):
            self.chk_zoom_roi.blockSignals(True)
            self.chk_zoom_roi.setChecked(False)
            self.chk_zoom_roi.blockSignals(False)
        self.canvas_seg.draw_idle()

    def _on_canvas_press(self, event):
        """Handle canvas mouse clicks: Right-drag for Pan, Left-click for Selection."""
        if event.button in (2, 3):  # Middle or Right mouse button -> start pan
            if event.xdata is not None and event.ydata is not None and hasattr(self, "ax_seg") and self.ax_seg is not None:
                self._pan_active = True
                self._pan_start = (event.xdata, event.ydata)
                self._pan_xlim = self.ax_seg.get_xlim()
                self._pan_ylim = self.ax_seg.get_ylim()
                QApplication.setOverrideCursor(Qt.ClosedHandCursor)
        elif event.button == 1:  # Left click -> grain selection or picking
            self._on_canvas_clicked(event)

    def _on_canvas_motion(self, event):
        """Pan viewport while dragging with right/middle mouse button."""
        if self._pan_active and self._pan_start is not None and event.xdata is not None and event.ydata is not None:
            if hasattr(self, "ax_seg") and self.ax_seg is not None:
                dx = event.xdata - self._pan_start[0]
                dy = event.ydata - self._pan_start[1]
                x1, x2 = self._pan_xlim
                y1, y2 = self._pan_ylim
                new_x = (x1 - dx, x2 - dx)
                new_y = (y1 - dy, y2 - dy)
                self.ax_seg.set_xlim(new_x)
                self.ax_seg.set_ylim(new_y)
                self.view_xlim = new_x
                self.view_ylim = new_y
                self.canvas_seg.draw_idle()

    def _on_canvas_release(self, event):
        """End panning on mouse release."""
        if event.button in (2, 3) and self._pan_active:
            self._pan_active = False
            self._pan_start = None
            QApplication.restoreOverrideCursor()
            self._update_zoom_label()

    def _on_canvas_scroll(self, event):
        """Zoom in/out smoothly at the mouse cursor position when scrolling the mouse wheel."""
        if event.xdata is None or event.ydata is None or not hasattr(self, "ax_seg") or self.ax_seg is None:
            return
        h, w = self.datacube.shape[:2]
        x, y = event.xdata, event.ydata
        factor = 1.25 if event.button == "up" else 0.80

        x1, x2 = self.ax_seg.get_xlim()
        y1, y2 = self.ax_seg.get_ylim()

        new_x1 = x - (x - x1) / factor
        new_x2 = x + (x2 - x) / factor
        new_y1 = y - (y - y1) / factor
        new_y2 = y + (y2 - y) / factor

        if new_x2 > new_x1 and new_y1 > new_y2:
            self.view_xlim = (new_x1, new_x2)
            self.view_ylim = (new_y1, new_y2)
            self.ax_seg.set_xlim(self.view_xlim)
            self.ax_seg.set_ylim(self.view_ylim)
            self._update_zoom_label()
            self.canvas_seg.draw_idle()

    def _on_canvas_clicked(self, event):
        if event.xdata is None or event.ydata is None:
            return

        gx = int(round(event.xdata))
        gy = int(round(event.ydata))

        h, w = self.datacube.shape[:2]
        if not (0 <= gx < w and 0 <= gy < h):
            return

        # 1. Picking Reference Spectrum Mode
        if self.picking_reference:
            self.reference_spectrum = np.copy(self.datacube[gy, gx, :]).astype(np.float32)
            self.reference_point_coord = (gx, gy)
            self.picking_reference = False
            QApplication.restoreOverrideCursor()
            self.btn_pick_grain.setText("👆 Pick Reference Grain on Canvas")
            self.btn_pick_grain.setStyleSheet("background-color: #f8fafc; font-weight: bold; padding: 5px;")

            picked_label = f"Picked Grain: (X={gx}, Y={gy})"
            idx = self.cmb_point_ref.findText("Picked Grain", Qt.MatchContains)
            if idx >= 0:
                self.cmb_point_ref.setItemText(idx, picked_label)
                self.cmb_point_ref.setCurrentIndex(idx)
            else:
                self.cmb_point_ref.addItem(picked_label)
                self.cmb_point_ref.setCurrentIndex(self.cmb_point_ref.count() - 1)

            idx_sam = self.cmb_algo.findText("Point Reference Matcher (SAM)", Qt.MatchContains)
            if idx_sam >= 0:
                self.cmb_algo.setCurrentIndex(idx_sam)

            self.run_segmentation()
            return

        # 2. Individual Grain Point Selection Mode (User clicked on a single grain)
        if self.interaction_mode == "point_individual":
            self._segment_single_grain_at(gx, gy)
            return

        # 3. Default: Click to Select/Inspect existing grain
        if self.grain_metrics:
            clicked_id = None
            min_dist = 20.0
            for m in self.grain_metrics:
                dist = np.hypot(m["cx"] - gx, m["cy"] - gy)
                if dist < min_dist:
                    min_dist = dist
                    clicked_id = m["id"]
            if clicked_id is not None:
                self.selected_object_id = clicked_id
                self._update_individual_grain_view()
                self.render_visualizations()

    def _segment_single_grain_at(self, gx: int, gy: int):
        """Extract a single grain centered at global coordinates (gx, gy) using seeded peak-watershed isolation."""
        h, w, b = self.datacube.shape
        R = 25
        y1, y2 = max(0, gy - R), min(h, gy + R + 1)
        x1, x2 = max(0, gx - R), min(w, gx + R + 1)

        patch = self.datacube[y1:y2, x1:x2, :]
        if hasattr(self, "feature_map") and self.feature_map is not None and self.feature_map.shape == (h, w):
            patch_base = self.feature_map[y1:y2, x1:x2].astype(np.float32)
        else:
            patch_base = np.mean(patch, axis=-1).astype(np.float32)

        seed_local_x = gx - x1
        seed_local_y = gy - y1

        patch_u8 = cv2.normalize(patch_base, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Polarity check: examine local surrounding pixels (r >= 12 px) to detect background intensity
        yy, xx = np.ogrid[:patch_u8.shape[0], :patch_u8.shape[1]]
        d_from_click = np.hypot(xx - seed_local_x, yy - seed_local_y)
        bg_sample = patch_u8[d_from_click >= 12]
        bg_val = float(np.median(bg_sample)) if bg_sample.size > 0 else float(np.median(patch_u8))
        seed_val = float(patch_u8[seed_local_y, seed_local_x])

        if seed_val < bg_val:
            contrast = np.clip(bg_val - patch_u8.astype(np.float32), 0, None)
        else:
            contrast = np.clip(patch_u8.astype(np.float32) - bg_val, 0, None)

        contrast_u8 = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        blurred = cv2.GaussianBlur(contrast_u8, (3, 3), 0)

        # Refine true seed center ONLY within +/-1 pixel (never jump to another grain)
        sub_r = 1
        sy1, sy2 = max(0, seed_local_y - sub_r), min(blurred.shape[0], seed_local_y + sub_r + 1)
        sx1, sx2 = max(0, seed_local_x - sub_r), min(blurred.shape[1], seed_local_x + sub_r + 1)
        max_loc = np.unravel_index(np.argmax(blurred[sy1:sy2, sx1:sx2]), (sy2 - sy1, sx2 - sx1))
        true_seed_y = sy1 + max_loc[0]
        true_seed_x = sx1 + max_loc[1]

        # Adaptive threshold: guaranteed to keep clicked seed in foreground
        seed_contrast = int(blurred[true_seed_y, true_seed_x])
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if seed_contrast > 0:
            thresh_final = min(otsu_val, max(10, int(0.35 * seed_contrast)))
        else:
            thresh_final = max(8, int(otsu_val * 0.5))

        bin_m = (blurred >= thresh_final).astype(np.uint8)
        bin_m[true_seed_y, true_seed_x] = 1
        bin_m[seed_local_y, seed_local_x] = 1

        # Safe morphological opening (only keep if it preserves the seed)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bin_opened = cv2.morphologyEx(bin_m, cv2.MORPH_OPEN, k_open)
        if bin_opened[true_seed_y, true_seed_x] > 0:
            bin_m = bin_opened

        # Distance transform to detect individual grain peaks
        dist = cv2.distanceTransform(bin_m, cv2.DIST_L2, 5)
        split_dist = max(3, self.spin_split_dist.value() if hasattr(self, "spin_split_dist") else 5)
        local_max = (dist == ndimage.maximum_filter(dist, size=split_dist)) & (dist >= 1.0)
        num_peaks, peak_labels = cv2.connectedComponents(local_max.astype(np.uint8))

        markers = np.zeros(bin_m.shape, dtype=np.int32)
        d_from_seed = np.hypot(xx - true_seed_x, yy - true_seed_y)
        markers[d_from_seed >= 16] = 2  # Background marker outside grain reach

        # Find the peak closest to the clicked seed
        best_peak = 1
        min_d = float('inf')
        for p_id in range(1, num_peaks):
            py, px = np.where(peak_labels == p_id)
            d = np.hypot(np.mean(px) - true_seed_x, np.mean(py) - true_seed_y)
            if d < min_d:
                min_d = d
                best_peak = p_id

        if num_peaks > 2:
            # Separate adjacent peaks into individual markers so touching grains are separated
            for p_id in range(1, num_peaks):
                if p_id == best_peak:
                    markers[peak_labels == p_id] = 1
                else:
                    markers[peak_labels == p_id] = p_id + 2
        else:
            markers[peak_labels == best_peak] = 1

        # Seed coordinates are always part of Marker 1
        markers[true_seed_y, true_seed_x] = 1
        markers[seed_local_y, seed_local_x] = 1
        cv2.circle(markers, (true_seed_x, true_seed_y), 1, 1, -1)

        try:
            from skimage.segmentation import watershed as ski_watershed
            ws_labels = ski_watershed(-dist, markers, mask=(bin_m > 0))
            grain_mask = (ws_labels == 1).astype(np.uint8)
        except Exception:
            _, lbls = cv2.connectedComponents(bin_m)
            seed_lbl = lbls[true_seed_y, true_seed_x]
            grain_mask = (lbls == seed_lbl).astype(np.uint8) if seed_lbl > 0 else np.zeros_like(bin_m)

        # Radial limit safeguard & fill holes
        grain_mask = (grain_mask & (d_from_seed <= 15)).astype(np.uint8)
        grain_mask = ndimage.binary_fill_holes(grain_mask > 0).astype(np.uint8)

        # Fallback circle if mask is unexpectedly small
        if np.sum(grain_mask) < 5:
            cv2.circle(grain_mask, (true_seed_x, true_seed_y), 4, 1, -1)

        area_px = int(np.sum(grain_mask))
        cnts, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or area_px < 5:
            cv2.circle(grain_mask, (true_seed_x, true_seed_y), 4, 1, -1)
            area_px = int(np.sum(grain_mask))
            cnts, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        y_idx, x_idx = np.where(grain_mask)
        cy_local, cx_local = float(np.mean(y_idx)), float(np.mean(x_idx))
        cx_global = cx_local + x1
        cy_global = cy_local + y1

        # Check if this exact grain was already segmented
        existing = next((m for m in self.grain_metrics if np.hypot(m["cx"] - cx_global, m["cy"] - cy_global) < 4), None)
        if existing is not None:
            self.selected_object_id = existing["id"]
            self._update_individual_grain_view()
            self.render_visualizations()
            return

        perimeter = float(cv2.arcLength(cnts[0], True))
        equiv_diam = float(2.0 * np.sqrt(area_px / np.pi))

        if len(cnts[0]) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnts[0])
                (e_x, e_y), (d1, d2), angle = ellipse
                aspect_ratio = max(d1, d2) / (min(d1, d2) + 1e-6)
            except Exception:
                aspect_ratio = 1.0
        else:
            aspect_ratio = 1.0

        circularity = float(4.0 * np.pi * area_px / (perimeter**2 + 1e-6)) if perimeter > 0 else 1.0

        mean_spec = np.mean(patch[grain_mask > 0], axis=0)
        self.grain_spectra.append(mean_spec)

        wvs = self.wavelengths
        idx800 = int(np.argmin(np.abs(wvs - 800)))
        idx670 = int(np.argmin(np.abs(wvs - 670)))
        denom = mean_spec[idx800] + mean_spec[idx670]
        mean_ndvi = float((mean_spec[idx800] - mean_spec[idx670]) / (denom + 1e-6)) if denom != 0 else 0.0

        grain_id = len(self.grain_metrics) + 1

        cnt_global = cnts[0].copy()
        cnt_global[:, 0, 0] += x1
        cnt_global[:, 0, 1] += y1

        self.grain_metrics.append({
            "id": grain_id,
            "type": "Plant Part" if mean_ndvi > 0.40 else "Grain / Seed",
            "cx": cx_global,
            "cy": cy_global,
            "cx_local": cx_local,
            "cy_local": cy_local,
            "area": area_px,
            "perimeter": perimeter,
            "equiv_diam": equiv_diam,
            "aspect_ratio": aspect_ratio,
            "circularity": circularity,
            "mean_ndvi": mean_ndvi,
            "mean_refl": float(np.mean(mean_spec)),
            "contours_local": cnts,
            "contours_global": [cnt_global],
            "crop_box": (x1, y1, x2, y2),
            "local_mask": grain_mask
        })

        self.selected_object_id = grain_id
        self.num_objects = len(self.grain_metrics)
        self._rebuild_labeled_masks()
        self._populate_table()
        self._populate_grain_selector()
        self._update_individual_grain_view()
        self.render_visualizations()

    def _segment_multiple_grains_in_box(self, x1: int, y1: int, x2: int, y2: int):
        """Extract all individual grains inside the drawn ROI bounding box with multi-scale morphology and peak-watershed."""
        h, w, b = self.datacube.shape
        x1, x2 = max(0, min(x1, x2)), min(w, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h, max(y1, y2))
        if x2 - x1 < 6 or y2 - y1 < 6:
            return

        crop = self.datacube[y1:y2, x1:x2, :]
        if hasattr(self, "feature_map") and self.feature_map is not None and self.feature_map.shape == (h, w):
            gray = self.feature_map[y1:y2, x1:x2].astype(np.float32)
        else:
            gray = np.mean(crop, axis=-1).astype(np.float32)

        gray_u8 = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Polarity detection from border of the ROI box
        border = np.concatenate([gray_u8[0, :], gray_u8[-1, :], gray_u8[:, 0], gray_u8[:, -1]])
        border_median = float(np.median(border)) if border.size > 0 else 128.0
        is_bright_bg = (self.chk_invert.isChecked() if hasattr(self, "chk_invert") else False) or (border_median > 128.0)

        # Multi-scale morphological filtering to extract seeds of various diameters
        k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k13 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        if is_bright_bg:
            contrast = np.maximum(cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k7),
                                  cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, k13))
        else:
            contrast = np.maximum(cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k7),
                                  cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, k13))

        blurred = cv2.GaussianBlur(contrast, (3, 3), 0)
        otsu_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (blurred >= max(otsu_val, 18)).astype(np.uint8)

        if np.sum(mask > 0) < 6:
            inv_u8 = (255 - gray_u8) if is_bright_bg else gray_u8
            otsu_inv, _ = cv2.threshold(inv_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            mask = (inv_u8 >= max(otsu_inv, 25)).astype(np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

        # Distance transform watershed peak separation to prevent merging touching grains
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        split_dist = max(3, self.spin_split_dist.value() if hasattr(self, "spin_split_dist") else 5)
        local_max = (dist == ndimage.maximum_filter(dist, size=split_dist)) & (dist >= 1.0)
        num_peaks, peak_labels = cv2.connectedComponents(local_max.astype(np.uint8))

        if num_peaks > 2:
            try:
                from skimage.segmentation import watershed as ski_watershed
                labels = ski_watershed(-dist, peak_labels, mask=(mask > 0))
                num_labels = int(np.max(labels)) + 1
            except Exception:
                num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask)
        else:
            num_labels, labels, _, _ = cv2.connectedComponentsWithStats(mask)

        valid_indices = []
        stats = {}
        cents = {}
        for idx in range(1, num_labels):
            m_i = (labels == idx)
            area_px = int(np.sum(m_i))
            if 6 <= area_px <= 2500:
                valid_indices.append(idx)
                stats[idx] = area_px
                y_idx, x_idx = np.where(m_i)
                cents[idx] = (float(np.mean(x_idx)), float(np.mean(y_idx)))

        if not valid_indices:
            QMessageBox.information(self, "No Grains in Box", "No grains of expected size (6–2500 px) were found in the selected box.")
            return

        wvs = self.wavelengths
        idx800 = int(np.argmin(np.abs(wvs - 800)))
        idx670 = int(np.argmin(np.abs(wvs - 670)))

        added_count = 0
        for idx in valid_indices:
            grain_m = (labels == idx).astype(np.uint8)
            area_px = stats[idx]
            cx_local, cy_local = cents[idx]
            cx_global = cx_local + x1
            cy_global = cy_local + y1

            is_duplicate = any(np.hypot(m["cx"] - cx_global, m["cy"] - cy_global) < 5 for m in self.grain_metrics)
            if is_duplicate:
                continue

            cnts, _ = cv2.findContours(grain_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue

            perimeter = float(cv2.arcLength(cnts[0], True))
            equiv_diam = float(2.0 * np.sqrt(area_px / np.pi))

            if len(cnts[0]) >= 5:
                try:
                    ellipse = cv2.fitEllipse(cnts[0])
                    (e_x, e_y), (d1, d2), angle = ellipse
                    aspect_ratio = max(d1, d2) / (min(d1, d2) + 1e-6)
                except Exception:
                    aspect_ratio = 1.0
            else:
                aspect_ratio = 1.0

            circularity = float(4.0 * np.pi * area_px / (perimeter**2 + 1e-6)) if perimeter > 0 else 1.0

            mean_spec = np.mean(crop[grain_m > 0], axis=0)
            self.grain_spectra.append(mean_spec)

            denom = mean_spec[idx800] + mean_spec[idx670]
            mean_ndvi = float((mean_spec[idx800] - mean_spec[idx670]) / (denom + 1e-6)) if denom != 0 else 0.0

            grain_id = len(self.grain_metrics) + 1

            cnt_global = cnts[0].copy()
            cnt_global[:, 0, 0] += x1
            cnt_global[:, 0, 1] += y1

            self.grain_metrics.append({
                "id": grain_id,
                "type": "Plant Part" if mean_ndvi > 0.40 else "Grain / Seed",
                "cx": cx_global,
                "cy": cy_global,
                "cx_local": cx_local,
                "cy_local": cy_local,
                "area": area_px,
                "perimeter": perimeter,
                "equiv_diam": equiv_diam,
                "aspect_ratio": aspect_ratio,
                "circularity": circularity,
                "mean_ndvi": mean_ndvi,
                "mean_refl": float(np.mean(mean_spec)),
                "contours_local": cnts,
                "contours_global": [cnt_global],
                "crop_box": (x1, y1, x2, y2),
                "local_mask": grain_m
            })
            added_count += 1

        if added_count > 0:
            self.selected_object_id = len(self.grain_metrics)
            self.num_objects = len(self.grain_metrics)
            self._rebuild_labeled_masks()
            self._populate_table()
            self._populate_grain_selector()
            self._update_individual_grain_view()
            self.render_visualizations()


    def _on_region_scope_changed(self):
        txt = self.cmb_region_scope.currentText()
        if "Entire Datacube" in txt:
            self._reset_roi_box()
        elif "Active Viewer ROI" in txt:
            if self.viewer_roi_boxes:
                self.active_roi_box = self.viewer_roi_boxes[0]
                x1, y1, x2, y2 = self.active_roi_box
                self.lbl_roi_box_info.setText(f"Active Box: [X: {x1}–{x2}, Y: {y1}–{y2}] ({x2-x1}×{y2-y1} px)")
                self.lbl_roi_box_info.setStyleSheet("color: #059669; font-weight: bold;")
                self.run_segmentation()
            else:
                QMessageBox.warning(self, "No Viewer ROI", "No ROI box was drawn in the active datacube window.")
        elif "Auto-Bounding Box" in txt:
            self._auto_fit_box_from_points(silent=False)

    def _toggle_drag_roi_box(self):
        if hasattr(self, "toolbar_seg") and self.toolbar_seg.mode:
            self.toolbar_seg.mode = ""
        if hasattr(self, "roi_selector") and self.roi_selector is not None:
            self.roi_selector.set_active(True)
        QMessageBox.information(
            self,
            "Drag ROI Box",
            "Click and drag directly on the image to draw a rectangular bounding box around the grains.\n\n"
            "Segmentation will automatically restrict exclusively to your drawn box."
        )

    def _on_roi_selected(self, eclick, erelease):
        if eclick.xdata is None or erelease.xdata is None:
            return
        x1, x2 = sorted([int(round(eclick.xdata)), int(round(erelease.xdata))])
        y1, y2 = sorted([int(round(eclick.ydata)), int(round(erelease.ydata))])

        h, w = self.datacube.shape[:2]
        x1 = max(0, min(w - 2, x1))
        x2 = max(x1 + 2, min(w, x2))
        y1 = max(0, min(h - 2, y1))
        y2 = max(y1 + 2, min(h, y2))

        # If in individual ROI selection mode: extract all grains inside this box
        if self.interaction_mode == "roi_individual":
            self._segment_multiple_grains_in_box(x1, y1, x2, y2)
            return

        # Default: Set active ROI bounding box
        self.active_roi_box = (x1, y1, x2, y2)
        self.lbl_roi_box_info.setText(f"Active Box: [X: {x1}–{x2}, Y: {y1}–{y2}] ({x2-x1}×{y2-y1} px)")
        if hasattr(self, "chk_auto_zoom_roi") and self.chk_auto_zoom_roi.isChecked():
            self._zoom_to_roi()
        self.lbl_roi_box_info.setStyleSheet("color: #059669; font-weight: bold;")

        idx = self.cmb_region_scope.findText("Custom ROI", Qt.MatchContains)
        if idx >= 0:
            self.cmb_region_scope.blockSignals(True)
            self.cmb_region_scope.setCurrentIndex(idx)
            self.cmb_region_scope.blockSignals(False)

        self.run_segmentation()

    def _auto_fit_box_from_points(self, silent: bool = False):
        pts = self._get_point_coordinates()
        if not pts:
            if not silent:
                QMessageBox.warning(self, "No Points", "No reference points found to fit a bounding box.")
            return

        h, w = self.datacube.shape[:2]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        pad_x_left = 25
        pad_x_right = 160 if (max_x - min_x) < 60 else 35
        pad_y = 25

        x1 = max(0, min_x - pad_x_left)
        x2 = min(w, max_x + pad_x_right)
        y1 = max(0, min_y - pad_y)
        y2 = min(h, max_y + pad_y)

        self.active_roi_box = (x1, y1, x2, y2)
        self.lbl_roi_box_info.setText(f"Active Box: [X: {x1}–{x2}, Y: {y1}–{y2}] ({x2-x1}×{y2-y1} px)")
        self.lbl_roi_box_info.setStyleSheet("color: #059669; font-weight: bold;")

        idx = self.cmb_region_scope.findText("Auto-Bounding Box", Qt.MatchContains)
        if idx >= 0:
            self.cmb_region_scope.blockSignals(True)
            self.cmb_region_scope.setCurrentIndex(idx)
            self.cmb_region_scope.blockSignals(False)

        self.run_segmentation()

    def _reset_roi_box(self):
        self.active_roi_box = None
        self.lbl_roi_box_info.setText("Active Box: None (Full Scene)")
        self.lbl_roi_box_info.setStyleSheet("color: #64748b; font-weight: bold;")
        idx = self.cmb_region_scope.findText("Entire Datacube", Qt.MatchContains)
        if idx >= 0:
            self.cmb_region_scope.blockSignals(True)
            self.cmb_region_scope.setCurrentIndex(idx)
            self.cmb_region_scope.blockSignals(False)
        self.run_segmentation()

    def _on_sam_slider_changed(self, val):
        angle = val / 100.0
        self.spin_sam_tol.blockSignals(True)
        self.spin_sam_tol.setValue(angle)
        self.spin_sam_tol.blockSignals(False)
        deg = angle * 180.0 / np.pi
        self.lbl_sam_tol.setText(f"<b>Match Sensitivity (SAM Angle):</b> {angle:.2f} rad (~{deg:.1f}°)")
        if "Point Reference" in self.cmb_algo.currentText():
            self.run_segmentation()

    def _on_sam_spin_changed(self, val):
        self.slider_sam_tol.blockSignals(True)
        self.slider_sam_tol.setValue(int(round(val * 100)))
        self.slider_sam_tol.blockSignals(False)
        deg = val * 180.0 / np.pi
        self.lbl_sam_tol.setText(f"<b>Match Sensitivity (SAM Angle):</b> {val:.2f} rad (~{deg:.1f}°)")
        if "Point Reference" in self.cmb_algo.currentText():
            self.run_segmentation()

    def one_click_auto_segment(self, silent: bool = False):
        """Intelligently detects points/ROIs and segments grains in 1 click."""
        pts = self._get_point_coordinates()
        if pts and len(pts) > 0:
            self._auto_fit_box_from_points(silent=True)
            idx_sam = self.cmb_algo.findText("Point Reference Matcher (SAM)", Qt.MatchContains)
            if idx_sam >= 0:
                self.cmb_algo.setCurrentIndex(idx_sam)
            self.spin_sam_tol.setValue(0.12)
            self.run_segmentation()
            if not silent:
                QMessageBox.information(
                    self,
                    "Auto-Segmentation Complete",
                    f"Successfully detected {self.num_objects} grains using your selected reference points and auto-fitted ROI!\n\n"
                    f"Grains are circled with red/green contours and measurements are listed in the table below."
                )
        else:
            idx_gbi = self.cmb_channel.findText("GBI", Qt.MatchContains)
            if idx_gbi >= 0:
                self.cmb_channel.setCurrentIndex(idx_gbi)
            idx_bh = self.cmb_algo.findText("Black-Hat", Qt.MatchContains)
            if idx_bh >= 0:
                self.cmb_algo.setCurrentIndex(idx_bh)
            else:
                idx_otsu = self.cmb_algo.findText("Otsu", Qt.MatchContains)
                if idx_otsu >= 0:
                    self.cmb_algo.setCurrentIndex(idx_otsu)
            self.run_segmentation()

    # --------------------------------------------------------------------------
    # FEATURE MAP & SEGMENTATION ENGINES
    # --------------------------------------------------------------------------
    def _on_channel_changed(self):
        txt = self.cmb_channel.currentText()
        is_single = "Single Wavelength" in txt
        self.lbl_band.setVisible(is_single)
        self.slider_band.setVisible(is_single)
        self.lbl_band_val.setVisible(is_single)
        self._compute_feature_map()
        self.run_segmentation()

    def _on_band_slider_changed(self, idx):
        wv = self.wavelengths[idx]
        self.lbl_band_val.setText(f"{wv:.1f} nm")
        self._compute_feature_map()
        self.run_segmentation()

    def _on_algo_changed(self):
        txt = self.cmb_algo.currentText()
        is_slider = "Spectral Index Cutoff" in txt or "Dual Manual" in txt
        self.lbl_thresh.setVisible(is_slider)
        self.spin_thresh.setVisible(is_slider)
        self.run_segmentation()

    def _compute_feature_map(self):
        """Extract the selected 2D scalar feature map from the 3D datacube."""
        txt = self.cmb_channel.currentText()
        cube = self.datacube
        wvs = self.wavelengths

        def get_band(target_nm):
            idx = int(np.argmin(np.abs(wvs - target_nm)))
            return cube[:, :, idx].astype(np.float32)

        if "Single Wavelength" in txt:
            idx = self.slider_band.value()
            self.feature_map = cube[:, :, idx].astype(np.float32)
        elif "NDVI" in txt:
            r800 = get_band(800)
            r670 = get_band(670)
            denom = r800 + r670
            self.feature_map = np.where(denom > 1e-6, (r800 - r670) / denom, 0.0)
        elif "NDRE" in txt:
            r790 = get_band(790)
            r720 = get_band(720)
            denom = r790 + r720
            self.feature_map = np.where(denom > 1e-6, (r790 - r720) / denom, 0.0)
        elif "PRI" in txt:
            r531 = get_band(531)
            r570 = get_band(570)
            denom = r531 + r570
            self.feature_map = np.where(denom > 1e-6, (r531 - r570) / denom, 0.0)
        elif "Water" in txt:
            r900 = get_band(900)
            r970 = get_band(970)
            self.feature_map = np.where(r970 > 1e-6, r900 / r970, 0.0)
        elif "PCA" in txt:
            h, w, b = cube.shape
            flat = cube.reshape(-1, b)
            pca = PCA(n_components=1, random_state=42)
            pc1 = pca.fit_transform(flat).reshape(h, w)
            self.feature_map = pc1.astype(np.float32)
        else:
            # GBI
            nir_mask = (wvs >= 750) & (wvs <= 900)
            if np.any(nir_mask):
                self.feature_map = np.mean(cube[:, :, nir_mask], axis=-1).astype(np.float32)
            else:
                self.feature_map = np.mean(cube, axis=-1).astype(np.float32)

        # Auto-polarity check
        if self.feature_map is not None and self.feature_map.size > 0:
            border_vals = np.concatenate([
                self.feature_map[0, :], self.feature_map[-1, :],
                self.feature_map[:, 0], self.feature_map[:, -1]
            ])
            if np.median(border_vals) > np.median(self.feature_map) + 0.08:
                self.chk_invert.setChecked(True)

    def _get_pseudo_rgb(self) -> np.ndarray:
        """Create a normalized 3-channel pseudo-RGB image for display."""
        cube = self.datacube
        wvs = self.wavelengths

        def get_band_idx(target_nm):
            return int(np.argmin(np.abs(wvs - target_nm)))

        r_idx = get_band_idx(640)
        g_idx = get_band_idx(550)
        b_idx = get_band_idx(460)

        rgb = np.stack([cube[:, :, r_idx], cube[:, :, g_idx], cube[:, :, b_idx]], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb_norm = np.clip((rgb - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
        return rgb_norm

    def run_segmentation(self):
        """Execute thresholding, morphological filtering, and watershed instance separation."""
        if self.feature_map is None or self.datacube is None:
            return

        # If user is in individual selection mode, preserve existing manually clicked grains
        if self.interaction_mode in ["point_individual", "roi_individual"] and len(self.grain_metrics) > 0:
            return

        # Determine sub-cube and feature map based on active ROI box
        if self.active_roi_box is not None:
            rx1, ry1, rx2, ry2 = self.active_roi_box
            sub_cube = self.datacube[ry1:ry2, rx1:rx2, :]
            sub_fmap = self.feature_map[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
        else:
            sub_cube = self.datacube
            sub_fmap = self.feature_map
            offset_x, offset_y = 0, 0

        algo_text = self.cmb_algo.currentText()

        # ----------------------------------------------------------------------
        # ALGORITHM 1: POINT REFERENCE MATCHER (SAM)
        # ----------------------------------------------------------------------
        if "Point Reference Matcher (SAM)" in algo_text:
            # Check if processing a card crop with dark seeds: use Black-Hat seed isolation
            norm_sub = cv2.normalize(sub_fmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            k_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            bh_map = cv2.morphologyEx(norm_sub, cv2.MORPH_BLACKHAT, k_bh)
            _, bin_mask = cv2.threshold(bh_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


        # ----------------------------------------------------------------------
        # ALGORITHM 2: POINT REFERENCE (PEARSON CORRELATION)
        # ----------------------------------------------------------------------
        elif "Pearson Correlation" in algo_text:
            ref = self.reference_spectrum
            if ref is None or len(ref) != sub_cube.shape[-1]:
                mid_y, mid_x = sub_cube.shape[0] // 2, sub_cube.shape[1] // 2
                ref = sub_cube[mid_y, mid_x, :]

            ref_ctr = ref - np.mean(ref)
            ref_norm = float(np.linalg.norm(ref_ctr)) + 1e-12
            ref_unit = (ref_ctr / ref_norm).astype(np.float32)

            cube_ctr = sub_cube - np.mean(sub_cube, axis=-1, keepdims=True)
            cube_norms = np.linalg.norm(cube_ctr, axis=-1) + 1e-12
            r_map = np.dot(cube_ctr, ref_unit) / cube_norms

            r_cutoff = max(0.60, 1.0 - (self.spin_sam_tol.value() * 0.8))
            bin_mask = ((r_map >= r_cutoff).astype(np.uint8)) * 255

        # ----------------------------------------------------------------------
        # ALGORITHM 3: BLACK-HAT CONTRAST EXTRACTOR (Grains on Card/Tray)
        # ----------------------------------------------------------------------
        elif "Black-Hat" in algo_text:
            norm_map = cv2.normalize(sub_fmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            kernel_size = 15
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            if self.chk_invert.isChecked():
                contrast_map = cv2.morphologyEx(norm_map, cv2.MORPH_BLACKHAT, kernel)
            else:
                contrast_map = cv2.morphologyEx(norm_map, cv2.MORPH_TOPHAT, kernel)
            _, bin_mask = cv2.threshold(contrast_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # ----------------------------------------------------------------------
        # ALGORITHM 4: CLASSICAL CV ENGINES
        # ----------------------------------------------------------------------
        else:
            fmap = np.copy(sub_fmap)
            k_blur = self.spin_blur.value()
            if k_blur > 1:
                fmap = cv2.GaussianBlur(fmap, (k_blur, k_blur), 0)

            norm_map = cv2.normalize(fmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            if "Otsu" in algo_text:
                _, bin_mask = cv2.threshold(norm_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif "Adaptive" in algo_text:
                bin_mask = cv2.adaptiveThreshold(norm_map, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
            elif "Spectral Index" in algo_text or "Dual Manual" in algo_text:
                cutoff = self.spin_thresh.value()
                bin_mask = np.where(fmap >= cutoff, 255, 0).astype(np.uint8)
            else:
                _, bin_mask = cv2.threshold(norm_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if self.chk_invert.isChecked():
                bin_mask = cv2.bitwise_not(bin_mask)

        # Morphological post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if self.chk_opening.isChecked():
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if self.chk_fill_holes.isChecked():
            bin_mask = ndimage.binary_fill_holes(bin_mask > 0).astype(np.uint8) * 255

        # Watershed instance separation (separates touching grains into individual instances)
        should_split = (hasattr(self, "chk_split_touching") and self.chk_split_touching.isChecked()) or ("Watershed" in algo_text)
        if should_split and np.any(bin_mask > 0):
            dist_transform = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
            split_dist = max(3, self.spin_split_dist.value() if hasattr(self, "spin_split_dist") else 7)
            local_max = (dist_transform == ndimage.maximum_filter(dist_transform, size=split_dist)) & (dist_transform >= 1.5)
            num_peaks, markers = cv2.connectedComponents(local_max.astype(np.uint8))
            if num_peaks > 1:
                try:
                    from skimage.segmentation import watershed as ski_watershed
                    labeled = ski_watershed(-dist_transform, markers, mask=(bin_mask > 0))
                except Exception:
                    _, labeled = cv2.connectedComponents((bin_mask > 0).astype(np.uint8))
            else:
                _, labeled = cv2.connectedComponents((bin_mask > 0).astype(np.uint8))
        else:
            _, labeled = cv2.connectedComponents((bin_mask > 0).astype(np.uint8))

        min_area = self.spin_min_area.value()
        max_area = self.spin_max_area.value()

        unique_ids = np.unique(labeled)
        sub_final_labeled = np.zeros_like(labeled, dtype=np.int32)
        valid_count = 0

        self.grain_metrics.clear()
        self.grain_spectra.clear()

        wvs = self.wavelengths
        idx800 = int(np.argmin(np.abs(wvs - 800)))
        idx670 = int(np.argmin(np.abs(wvs - 670)))

        for obj_id in unique_ids:
            if obj_id == 0:
                continue
            mask_i = (labeled == obj_id)
            area_px = int(np.sum(mask_i))
            if min_area <= area_px <= max_area:
                valid_count += 1
                sub_final_labeled[mask_i] = valid_count

                y_idx, x_idx = np.where(mask_i)
                cy_local, cx_local = float(np.mean(y_idx)), float(np.mean(x_idx))

                cx_global = cx_local + offset_x
                cy_global = cy_local + offset_y

                contours, _ = cv2.findContours(mask_i.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                perimeter = float(cv2.arcLength(contours[0], True)) if contours else 0.0
                equiv_diam = float(2.0 * np.sqrt(area_px / np.pi))

                if len(contours) > 0 and len(contours[0]) >= 5:
                    try:
                        ellipse = cv2.fitEllipse(contours[0])
                        (e_x, e_y), (d1, d2), angle = ellipse
                        major = max(d1, d2)
                        minor = min(d1, d2)
                        aspect_ratio = major / (minor + 1e-6)
                    except Exception:
                        aspect_ratio = 1.0
                else:
                    aspect_ratio = 1.0

                circularity = float(4.0 * np.pi * area_px / (perimeter**2 + 1e-6)) if perimeter > 0 else 1.0

                mean_spec = np.mean(sub_cube[mask_i], axis=0)
                self.grain_spectra.append(mean_spec)

                denom = mean_spec[idx800] + mean_spec[idx670]
                mean_ndvi = float((mean_spec[idx800] - mean_spec[idx670]) / (denom + 1e-6)) if denom != 0 else 0.0
                obj_type = "Plant Part" if mean_ndvi > 0.40 else "Grain / Seed"

                # Global contour points
                cnt_global = contours[0].copy() if contours else None
                if cnt_global is not None:
                    cnt_global[:, 0, 0] += offset_x
                    cnt_global[:, 0, 1] += offset_y

                # Crop bounding box for individual visualizer
                pad = 12
                y_min, y_max = max(0, int(np.min(y_idx)) - pad), min(sub_cube.shape[0], int(np.max(y_idx)) + pad + 1)
                x_min, x_max = max(0, int(np.min(x_idx)) - pad), min(sub_cube.shape[1], int(np.max(x_idx)) + pad + 1)

                self.grain_metrics.append({
                    "id": valid_count,
                    "type": obj_type,
                    "cx": cx_global,
                    "cy": cy_global,
                    "cx_local": cx_local,
                    "cy_local": cy_local,
                    "area": area_px,
                    "perimeter": perimeter,
                    "equiv_diam": equiv_diam,
                    "aspect_ratio": aspect_ratio,
                    "circularity": circularity,
                    "mean_ndvi": mean_ndvi,
                    "mean_refl": float(np.mean(mean_spec)),
                    "contours_local": contours,
                    "contours_global": [cnt_global] if cnt_global is not None else [],
                    "crop_box": (x_min + offset_x, y_min + offset_y, x_max + offset_x, y_max + offset_y),
                    "local_mask": mask_i[y_min:y_max, x_min:x_max]
                })

        full_h, full_w = self.datacube.shape[:2]
        full_labeled = np.zeros((full_h, full_w), dtype=np.int32)
        full_bin = np.zeros((full_h, full_w), dtype=np.uint8)

        if self.active_roi_box is not None:
            full_labeled[ry1:ry2, rx1:rx2] = sub_final_labeled
            full_bin[ry1:ry2, rx1:rx2] = (sub_final_labeled > 0).astype(np.uint8) * 255
        else:
            full_labeled = sub_final_labeled
            full_bin = (sub_final_labeled > 0).astype(np.uint8) * 255

        self.labeled_instances = full_labeled
        self.sub_labeled_instances = sub_final_labeled
        self.binary_mask = full_bin
        self.num_objects = valid_count

        total_px = full_labeled.size
        fg_px = int(np.sum(self.binary_mask > 0))
        cov_pct = (fg_px / total_px) * 100.0
        self.lbl_stats.setText(f"Detected Grains / Objects: {valid_count:,} | Foreground Area: {fg_px:,} px | Scene Coverage: {cov_pct:.2f}%")

        if self.grain_metrics and self.selected_object_id is None:
            self.selected_object_id = self.grain_metrics[0]["id"]

        self._populate_table()
        self._populate_grain_selector()
        self._update_individual_grain_view()
        self.render_visualizations()

    def _populate_table(self):
        """Populate the bottom table with grain metrics."""
        self.tbl_grains.blockSignals(True)
        self.tbl_grains.setRowCount(len(self.grain_metrics))
        for r, m in enumerate(self.grain_metrics):
            self.tbl_grains.setItem(r, 0, QTableWidgetItem(f"#{m['id']}"))
            self.tbl_grains.setItem(r, 1, QTableWidgetItem(str(m["type"])))
            self.tbl_grains.setItem(r, 2, QTableWidgetItem(f"({m['cx']:.1f}, {m['cy']:.1f})"))
            self.tbl_grains.setItem(r, 3, QTableWidgetItem(f"{m['area']}"))
            self.tbl_grains.setItem(r, 4, QTableWidgetItem(f"{m['equiv_diam']:.2f}"))
            self.tbl_grains.setItem(r, 5, QTableWidgetItem(f"{m['aspect_ratio']:.2f}"))
            self.tbl_grains.setItem(r, 6, QTableWidgetItem(f"{m['circularity']:.2f}"))
            self.tbl_grains.setItem(r, 7, QTableWidgetItem(f"{m['mean_ndvi']:.3f}"))
        self.tbl_grains.blockSignals(False)

    def _populate_grain_selector(self):
        self.cmb_grain_selector.blockSignals(True)
        self.cmb_grain_selector.clear()
        for m in self.grain_metrics:
            self.cmb_grain_selector.addItem(f"Grain #{m['id']} — Area: {m['area']} px, NDVI: {m['mean_ndvi']:.2f}, Circ: {m['circularity']:.2f}")
        if self.selected_object_id is not None:
            idx = next((i for i, m in enumerate(self.grain_metrics) if m["id"] == self.selected_object_id), 0)
            self.cmb_grain_selector.setCurrentIndex(idx)
        self.cmb_grain_selector.blockSignals(False)

    def _on_table_row_selected(self):
        selected_rows = self.tbl_grains.selectionModel().selectedRows()
        if selected_rows:
            idx = selected_rows[0].row()
            if 0 <= idx < len(self.grain_metrics):
                self.selected_object_id = self.grain_metrics[idx]["id"]
                self.cmb_grain_selector.blockSignals(True)
                self.cmb_grain_selector.setCurrentIndex(idx)
                self.cmb_grain_selector.blockSignals(False)
                self._update_individual_grain_view()
                self.render_visualizations()

    def _on_grain_selector_changed(self, idx):
        if 0 <= idx < len(self.grain_metrics):
            self.selected_object_id = self.grain_metrics[idx]["id"]
            self._update_individual_grain_view()
            self.render_visualizations()

    def _prev_individual_grain(self):
        if not self.grain_metrics:
            return
        idx = next((i for i, m in enumerate(self.grain_metrics) if m["id"] == self.selected_object_id), 0)
        new_idx = max(0, idx - 1)
        self.selected_object_id = self.grain_metrics[new_idx]["id"]
        self.cmb_grain_selector.setCurrentIndex(new_idx)

    def _next_individual_grain(self):
        if not self.grain_metrics:
            return
        idx = next((i for i, m in enumerate(self.grain_metrics) if m["id"] == self.selected_object_id), 0)
        new_idx = min(len(self.grain_metrics) - 1, idx + 1)
        self.selected_object_id = self.grain_metrics[new_idx]["id"]
        self.cmb_grain_selector.setCurrentIndex(new_idx)

    # --------------------------------------------------------------------------
    # INDIVIDUAL GRAIN DETAIL VISUALIZATION (TAB 2)
    # --------------------------------------------------------------------------
    def _update_individual_grain_view(self):
        """Render high-magnification zoomed view, mask, spectrum, and card of the selected grain."""
        if not self.grain_metrics:
            self.fig_indiv.clear()
            self.canvas_indiv.draw()
            self.lbl_indiv_card.setText("No grains detected.")
            return

        sel_idx = next((i for i, m in enumerate(self.grain_metrics) if m["id"] == self.selected_object_id), 0)
        m = self.grain_metrics[sel_idx]
        spec = self.grain_spectra[sel_idx]

        rgb = self._get_pseudo_rgb()
        h, w = rgb.shape[:2]

        # Crop box around this grain
        cx, cy = int(round(m["cx"])), int(round(m["cy"]))
        R = 24
        y1, y2 = max(0, cy - R), min(h, cy + R + 1)
        x1, x2 = max(0, cx - R), min(w, cx + R + 1)

        crop_rgb = rgb[y1:y2, x1:x2]

        # Local mask for this grain in the crop box
        crop_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cnts = m.get("contours_global", [])
        if cnts:
            cnt_local = cnts[0].copy()
            cnt_local[:, 0, 0] -= x1
            cnt_local[:, 0, 1] -= y1
            cv2.drawContours(crop_mask, [cnt_local], -1, 255, -1)

        self.fig_indiv.clear()
        ax1 = self.fig_indiv.add_subplot(131)
        ax2 = self.fig_indiv.add_subplot(132)
        ax3 = self.fig_indiv.add_subplot(133)

        # 1. Zoomed-In Grain RGB with Contour
        ax1.imshow(crop_rgb)
        if cnts:
            pts = cnt_local.squeeze(1)
            if pts.ndim == 2:
                ax1.plot(pts[:, 0], pts[:, 1], color="#dc2626", linewidth=2.5)
        ax1.plot(cx - x1, cy - y1, "g+", markersize=8, markeredgewidth=2)
        ax1.set_title(f"Grain #{m['id']} (Zoomed 6×)", fontsize=10, fontweight="bold")
        ax1.axis("off")

        # 2. Individual Binary Mask
        ax2.imshow(crop_mask, cmap="gray")
        ax2.set_title(f"Binary Mask ({m['area']} px)", fontsize=10, fontweight="bold")
        ax2.axis("off")

        # 3. Individual Reflectance Spectrum
        wvs = self.wavelengths
        ax3.plot(wvs, spec, color="#dc2626", linewidth=2.2, label=f"Grain #{m['id']}")
        if self.reference_spectrum is not None and len(self.reference_spectrum) == len(wvs):
            ax3.plot(wvs, self.reference_spectrum, color="#94a3b8", linestyle="--", linewidth=1.5, label="Target Reference")
        ax3.set_xlabel("Wavelength (nm)", fontsize=8, fontweight="bold")
        ax3.set_ylabel("Reflectance", fontsize=8, fontweight="bold")
        ax3.set_title(f"Grain #{m['id']} Spectral Profile", fontsize=10, fontweight="bold")
        ax3.legend(fontsize=7.5, loc="upper left")
        ax3.grid(True, linestyle=":", alpha=0.6)

        self.fig_indiv.tight_layout()
        self.canvas_indiv.draw()

        # Update text card
        self.lbl_indiv_card.setText(
            f"🔍 Grain #{m['id']} Details:  "
            f"Centroid: ({m['cx']:.1f}, {m['cy']:.1f})  |  "
            f"Area: {m['area']} px  |  "
            f"Equiv. Diam: {m['equiv_diam']:.2f} px  |  "
            f"Aspect Ratio: {m['aspect_ratio']:.2f}  |  "
            f"Circularity: {m['circularity']:.2f}  |  "
            f"Mean NDVI: {m['mean_ndvi']:.3f}  |  "
            f"Class: {m['type']}"
        )

    # --------------------------------------------------------------------------
    # GRAIN GALLERY (TAB 3)
    # --------------------------------------------------------------------------
    def _render_grain_gallery(self):
        """Render a thumbnail grid of all segmented grains."""
        self.fig_gal.clear()
        if not self.grain_metrics:
            self.canvas_gal.draw()
            return

        rgb = self._get_pseudo_rgb()
        h, w = rgb.shape[:2]
        n = min(36, len(self.grain_metrics))
        cols = min(6, n)
        rows = int(np.ceil(n / cols))

        R = 18
        for i in range(n):
            m = self.grain_metrics[i]
            cx, cy = int(round(m["cx"])), int(round(m["cy"]))
            y1, y2 = max(0, cy - R), min(h, cy + R + 1)
            x1, x2 = max(0, cx - R), min(w, cx + R + 1)
            crop = rgb[y1:y2, x1:x2]

            ax = self.fig_gal.add_subplot(rows, cols, i + 1)
            ax.imshow(crop)

            cnts = m.get("contours_global", [])
            if cnts:
                cnt_local = cnts[0].copy()
                cnt_local[:, 0, 0] -= x1
                cnt_local[:, 0, 1] -= y1
                pts = cnt_local.squeeze(1)
                if pts.ndim == 2:
                    color = "#dc2626" if m["id"] == self.selected_object_id else "#10b981"
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5)

            ax.set_title(f"#{m['id']} ({m['area']}px)", fontsize=7.5, fontweight="bold")
            ax.axis("off")

        self.fig_gal.tight_layout()
        self.canvas_gal.draw()

    # --------------------------------------------------------------------------
    # VISUALIZATION RENDERING & INTERACTIVE CANVASES
    # --------------------------------------------------------------------------
    def render_visualizations(self):
        """Render the full scene, individual grain details, gallery, and distributions."""
        if self.labeled_instances is None:
            return

        mode = self.cmb_overlay_mode.currentText()
        rgb = self._get_pseudo_rgb()

        rgb_disp = rgb
        fmap_disp = self.feature_map
        bin_disp = self.binary_mask
        labeled_disp = self.labeled_instances

        # ----------------------------------------------------------------------
        # TAB 1: FULL SCENE & ZOOMABLE GRAIN VISUALIZER
        # ----------------------------------------------------------------------
        self.fig_seg.clear()
        ax_seg = self.fig_seg.add_subplot(111)
        self.ax_seg = ax_seg

        if "Contours" in mode:
            ax_seg.imshow(rgb_disp)
            for m in self.grain_metrics:
                obj_id = m["id"]
                color = "#dc2626" if obj_id == self.selected_object_id else ("#10b981" if m["type"] == "Plant Part" else "#3b82f6")
                lw = 2.5 if obj_id == self.selected_object_id else 1.3

                cnts = m.get("contours_global", [])
                for cnt in cnts:
                    pts = cnt.squeeze(1)
                    if pts.ndim == 2:
                        ax_seg.plot(pts[:, 0], pts[:, 1], color=color, linewidth=lw)

                cx_p = m["cx"]
                cy_p = m["cy"]
                ax_seg.plot(cx_p, cy_p, "r+", markersize=5)
                ax_seg.text(cx_p + 2, cy_p - 2, str(obj_id), color="yellow", fontsize=8, fontweight="bold")

            if self.active_roi_box is not None:
                bx1, by1, bx2, by2 = self.active_roi_box
                rect_patch = Rectangle(
                    (bx1, by1), bx2 - bx1, by2 - by1,
                    fill=False, edgecolor="#fbbf24", linewidth=2.0, linestyle="--"
                )
                ax_seg.add_patch(rect_patch)
                ax_seg.text(bx1 + 3, by1 - 4, "Active ROI Box", color="#fbbf24", fontsize=8.5, fontweight="bold")

            if self.reference_point_coord is not None:
                gx, gy = self.reference_point_coord
                ax_seg.plot(gx, gy, "o", color="#ec4899", markersize=8, markeredgecolor="white", markeredgewidth=1.5)
                ax_seg.text(gx + 4, gy - 4, "Ref Grain", color="#ec4899", fontsize=8, fontweight="bold")

            title_str = f"Grain Contours & Detections (Total: {self.num_objects})"
            ax_seg.set_title(title_str, fontsize=11, fontweight="bold")

        elif "Labeled Instances" in mode:
            if labeled_disp is not None:
                masked_labeled = np.ma.masked_where(labeled_disp == 0, labeled_disp)
                ax_seg.imshow(rgb_disp, alpha=0.35)
                ax_seg.imshow(masked_labeled, cmap="tab20", interpolation="nearest", alpha=0.85)
            ax_seg.set_title(f"Labeled Grain Instances ({self.num_objects} Unique Particles)", fontsize=11, fontweight="bold")
        elif "Binary Mask" in mode:
            if bin_disp is not None:
                ax_seg.imshow(bin_disp, cmap="gray")
            ax_seg.set_title(f"Binary Foreground Mask ({np.sum(self.binary_mask > 0):,} px)", fontsize=11, fontweight="bold")
        else:
            if fmap_disp is not None:
                im = ax_seg.imshow(fmap_disp, cmap="viridis")
                self.fig_seg.colorbar(im, ax=ax_seg, fraction=0.046, pad=0.04)
            ax_seg.set_title(f"Extracted Feature Map: {self.cmb_channel.currentText()}", fontsize=11, fontweight="bold")

        ax_seg.axis("off")

        # Apply persistent or default zoom viewport
        h, w = self.datacube.shape[:2]
        if self.view_xlim is not None and self.view_ylim is not None:
            ax_seg.set_xlim(self.view_xlim)
            ax_seg.set_ylim(self.view_ylim)
        else:
            ax_seg.set_xlim(-0.5, w - 0.5)
            ax_seg.set_ylim(h - 0.5, -0.5)
            self.view_xlim = (-0.5, w - 0.5)
            self.view_ylim = (h - 0.5, -0.5)
        self._update_zoom_label()

        # Initialize RectangleSelector
        self.roi_selector = RectangleSelector(
            ax_seg,
            self._on_roi_selected,
            useblit=False,
            props=dict(facecolor="#f59e0b", edgecolor="#d97706", alpha=0.22, fill=True),
            interactive=True
        )

        self.canvas_seg.draw()

        # Update other tabs
        self._update_individual_grain_view()
        self._render_grain_gallery()

        # ----------------------------------------------------------------------
        # TAB 4: MORPHOLOGY DISTRIBUTIONS
        # ----------------------------------------------------------------------
        self.fig_dist.clear()
        if self.grain_metrics:
            ax1 = self.fig_dist.add_subplot(121)
            ax2 = self.fig_dist.add_subplot(122)

            areas = [m["area"] for m in self.grain_metrics]
            aspects = [m["aspect_ratio"] for m in self.grain_metrics]

            ax1.hist(areas, bins=min(15, max(3, len(areas))), color="#2563eb", edgecolor="#1e293b", alpha=0.75)
            ax1.axvline(np.mean(areas), color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean: {np.mean(areas):.1f} px")
            ax1.set_xlabel("Particle Area (px)", fontsize=9, fontweight="bold")
            ax1.set_ylabel("Frequency", fontsize=9, fontweight="bold")
            ax1.set_title("Grain Size Distribution", fontsize=10, fontweight="bold")
            ax1.legend(fontsize=8)
            ax1.grid(True, linestyle=":", alpha=0.6)

            ax2.hist(aspects, bins=min(15, max(3, len(aspects))), color="#059669", edgecolor="#1e293b", alpha=0.75)
            ax2.axvline(np.mean(aspects), color="#dc2626", linestyle="--", linewidth=1.5, label=f"Mean Ratio: {np.mean(aspects):.2f}")
            ax2.set_xlabel("Aspect Ratio (Major/Minor)", fontsize=9, fontweight="bold")
            ax2.set_ylabel("Frequency", fontsize=9, fontweight="bold")
            ax2.set_title("Grain Elongation & Roundness", fontsize=10, fontweight="bold")
            ax2.legend(fontsize=8)
            ax2.grid(True, linestyle=":", alpha=0.6)

            self.fig_dist.tight_layout()
        self.canvas_dist.draw()

        # ----------------------------------------------------------------------
        # TAB 5: SPECTRAL CURVES & ENVELOPE
        # ----------------------------------------------------------------------
        self.fig_spec.clear()
        ax_sp = self.fig_spec.add_subplot(111)

        wvs = self.wavelengths
        if self.grain_spectra:
            spec_arr = np.array(self.grain_spectra)
            mean_all = np.mean(spec_arr, axis=0)
            std_all = np.std(spec_arr, axis=0)

            for idx, s in enumerate(spec_arr):
                obj_id = self.grain_metrics[idx]["id"]
                if obj_id == self.selected_object_id:
                    continue
                ax_sp.plot(wvs, s, color="#94a3b8", alpha=0.35, linewidth=0.8)

            ax_sp.plot(wvs, mean_all, color="#1e3a8a", linewidth=2.5, label=f"Ensemble Mean (N={len(spec_arr)})")
            ax_sp.fill_between(wvs, mean_all - std_all, mean_all + std_all, color="#1e3a8a", alpha=0.18, label="±1σ Envelope")

            if self.selected_object_id is not None:
                sel_idx = next((i for i, m in enumerate(self.grain_metrics) if m["id"] == self.selected_object_id), None)
                if sel_idx is not None:
                    ax_sp.plot(wvs, spec_arr[sel_idx], color="#dc2626", linewidth=2.5, label=f"Selected Grain #{self.selected_object_id}")

        if self.reference_spectrum is not None and len(self.reference_spectrum) == len(wvs):
            ax_sp.plot(wvs, self.reference_spectrum, color="#ec4899", linestyle="--", linewidth=2.2, label="Target Point Reference Curve")

        ax_sp.set_xlabel("Wavelength (nm)", fontsize=9, fontweight="bold")
        ax_sp.set_ylabel("Reflectance", fontsize=9, fontweight="bold")
        ax_sp.set_title("Grain Spectral Signatures vs. Target Reference", fontsize=11, fontweight="bold")
        if self.grain_spectra or (self.reference_spectrum is not None and len(self.reference_spectrum) == len(wvs)):
            ax_sp.legend(loc="upper left", fontsize=8.5)
        ax_sp.grid(True, linestyle=":", alpha=0.6)

        self.fig_spec.tight_layout()
        self.canvas_spec.draw()

    # --------------------------------------------------------------------------
    # EXPORT & PIPELINE ACTIONS
    # --------------------------------------------------------------------------
    def export_mask(self):
        if self.binary_mask is None:
            QMessageBox.warning(self, "No Mask", "Please run segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Segmentation Mask", "grain_segmentation_mask.png", "Images (*.png *.tif *.jpg)")
        if path:
            try:
                cv2.imwrite(path, self.binary_mask)
                QMessageBox.information(self, "Export Success", f"Mask saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save mask:\n{str(e)}")

    def export_measurements_csv(self):
        if not self.grain_metrics:
            QMessageBox.warning(self, "No Metrics", "Please run segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Grain Measurements", "grain_morphometry.csv", "CSV Files (*.csv)")
        if path:
            try:
                keys = ["id", "type", "cx", "cy", "area", "perimeter", "equiv_diam", "aspect_ratio", "circularity", "mean_ndvi", "mean_refl"]
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                    writer.writeheader()
                    for m in self.grain_metrics:
                        writer.writerow(m)
                QMessageBox.information(self, "Export Success", f"Exported {len(self.grain_metrics)} grains to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export CSV:\n{str(e)}")

    def export_spectra_csv(self):
        if not self.grain_spectra:
            QMessageBox.warning(self, "No Spectra", "Please run segmentation first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Grain Spectra", "grain_spectra.csv", "CSV Files (*.csv)")
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    header = ["grain_id", "type"] + [f"{w:.1f}nm" for w in self.wavelengths]
                    writer.writerow(header)
                    for idx, spec in enumerate(self.grain_spectra):
                        m = self.grain_metrics[idx]
                        row = [m["id"], m["type"]] + list(spec)
                        writer.writerow(row)
                QMessageBox.information(self, "Export Success", f"Exported spectra for {len(self.grain_spectra)} grains to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export spectra:\n{str(e)}")

    def send_to_spectral_library(self):
        """Export segmented grains directly to the HyperVision Master Spectral Library."""
        if not self.grain_spectra:
            QMessageBox.warning(self, "No Grains", "Please run segmentation first to detect grains before sending to Spectral Library.")
            return
        try:
            from spectral_library import SpectralLibrary, SpectralLibraryEntry, SpectralLibraryDialog

            # 1. Retrieve or initialize the master spectral library from parent viewer
            if self.parent_viewer and hasattr(self.parent_viewer, "_spectral_library") and self.parent_viewer._spectral_library is not None:
                lib = self.parent_viewer._spectral_library
            elif hasattr(self, "_spectral_library") and self._spectral_library is not None:
                lib = self._spectral_library
            else:
                lib = SpectralLibrary("HyperVision Master Library")
                if self.parent_viewer:
                    self.parent_viewer._spectral_library = lib
                self._spectral_library = lib

            # 2. Build statistical spectral entry from all extracted grains
            samples_arr = np.asarray(self.grain_spectra, dtype=np.float64)
            entry_name = f"Grains_{self.datacube_name}" if self.datacube_name else "Segmented_Grains"

            entry = SpectralLibraryEntry.from_samples(
                name=entry_name,
                wavelengths=self.wavelengths,
                samples=samples_arr,
                color="#059669"
            )
            entry.metadata = {
                "datacube": self.datacube_name,
                "grain_count": len(self.grain_spectra),
                "source": "Segmentation Studio"
            }
            lib.add_entry(entry)

            # 3. Prompt user with option to open Spectral Library Studio immediately
            reply = QMessageBox.question(
                self,
                "Spectral Library Updated",
                f"Successfully registered '{entry.name}' ({len(self.grain_spectra)} individual grains) into the Spectral Library!\n\n"
                f"Would you like to open the Spectral Library Studio now to inspect the spectra and envelopes?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                dialog = SpectralLibraryDialog(lib, parent_viewer=self.parent_viewer or self, parent=self)
                dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to send to spectral library:\n{str(e)}")

    def create_datacube_rois(self):
        """Send every detected grain or plant object as a distinct ROI to the active datacube window."""
        if not self.parent_viewer:
            QMessageBox.warning(self, "No Viewer", "Viewer instance not available.")
            return
        window = self.active_window or self.parent_viewer.get_active_window()
        if not window:
            QMessageBox.warning(self, "No Active Window", "Please open a datacube in the main viewer first.")
            return

        if not self.grain_metrics:
            QMessageBox.warning(self, "No Grains Segmented", "Please run segmentation first to detect grains before creating ROIs.")
            return

        default_label = "Grain_Seed"
        if window and hasattr(window, "name"):
            wname_lower = window.name.lower()
            if "healthy" in wname_lower:
                default_label = "Healthy"
            elif "disease" in wname_lower:
                default_label = "Disease"
            else:
                default_label = window.name.split("-")[0].replace("_", " ").strip()

        # Call QInputDialog.getText with QLineEdit.Normal as 4th argument (PyQt5 standard)
        class_name, ok = QInputDialog.getText(
            self,
            "Assign Class Label to Detected Grains",
            f"Enter class label for the {len(self.grain_metrics)} detected grains/seeds on '{window.name}':\n(e.g., 'Healthy', 'Disease', 'Variety_A')",
            QLineEdit.Normal,
            default_label
        )
        if not ok or not class_name.strip():
            return
        class_label = class_name.strip()

        try:
            if not hasattr(window, "selections") or window.selections is None:
                window.selections = []

            added = 0
            for idx, m in enumerate(self.grain_metrics):
                obj_id = m["id"]
                spec = self.grain_spectra[idx]
                cx, cy = int(round(m["cx"])), int(round(m["cy"]))

                roi = {
                    "id": f"grain_{obj_id}",
                    "selection_type": "point",
                    "class_label": class_label,
                    "raw_x": cx,
                    "raw_y": cy,
                    "points": [(cx, cy)],
                    "profile": spec,
                    "units": getattr(window.document, "units", "reflectance") if hasattr(window, "document") else "reflectance",
                    "wavelengths": self.wavelengths
                }
                window.selections.append(roi)
                added += 1

            if hasattr(window, "update_selection_table"):
                window.update_selection_table()

            QMessageBox.information(
                self,
                "ROIs Created",
                f"Successfully created {added} segmented grain ROIs under class '{class_label}' on '{window.name}'!\n\n"
                f"You can now train AI models directly using these extracted actual classes."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create datacube ROIs:\n{str(e)}")
