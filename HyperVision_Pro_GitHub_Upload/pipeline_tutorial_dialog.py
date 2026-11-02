# -*- coding: utf-8 -*-
"""
HyperVision Pro - Interactive Guided Hyperspectral Imaging Pipeline Tutorial
Provides a clear, pedagogical, step-by-step walkthrough of the entire scientific
hyperspectral analysis lifecycle with direct interactive action triggers.
"""

import sys
import os
from typing import Optional, Callable

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QDialog, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QStackedWidget, QCheckBox, QFrame,
    QScrollArea, QMessageBox, QSizePolicy
)


PIPELINE_STEPS = [
    {
        "step_num": 1,
        "badge": "STAGE 1: CALIBRATION & INGESTION",
        "title": "📥 Calibration & Reflectance Normalization (RAW → REFLECTANCE)",
        "subtitle": "Convert sensor digital numbers to physical, illumination-independent surface reflectance.",
        "icon": "📥",
        "color": "#2563eb",
        "rationale": (
            "Hyperspectral imaging sensors record raw Digital Numbers (DN) which fluctuate depending on camera exposure, "
            "lens vignetting, and illumination drift from halogen lamps. In order to compare spectra quantitatively across "
            "different experiments, days, or instruments, raw images must be calibrated to physical surface reflectance ($R$) "
            "using standard Lambertian white reference ($W$) and dark current offset ($D$) panels:\n\n"
            "$$\\text{Reflectance}(\\lambda) = \\frac{\\text{Raw}(\\lambda) - \\text{Dark}(\\lambda)}{\\text{White}(\\lambda) - \\text{Dark}(\\lambda)}$$"
        ),
        "how_to": [
            "<b>1. Open Folder:</b> Click <b>'📂 Open Folder'</b> on the top Ribbon bar (or press <code>Ctrl+O</code>).",
            "<b>2. Automatic Reference Discovery:</b> HyperVision automatically scans the folder for <code>data.hdr</code>, <code>whiteReference.hdr</code>, and <code>darkReference.hdr</code>.",
            "<b>3. Direct Reflectance Calculation:</b> Physical reflectance is computed at full floating-point precision ($0.0 \\to 1.0$ or $0\\% - 100\\%$), eliminating illumination artifacts.",
            "<b>4. Radiance & Orthorectification Fallbacks:</b> If white/dark panels are unavailable, use the <b>'Radiance'</b> or <b>'Ortho R.'</b> tools on the Ribbon."
        ],
        "tip": "Always inspect the QC mode badge in the Datacube Window header to verify that <b>RAW_DARK_WHITE_REFLECTANCE</b> is active before conducting quantitative analysis.",
        "action_text": "📂 Open Dataset Folder Now",
        "action_method": "choose_folder"
    },
    {
        "step_num": 2,
        "badge": "STAGE 2: SPATIAL & SPECTRAL EXPLORATION",
        "title": "🔍 Spatial & Spectral Exploration (Cube Inspection & ROIs)",
        "subtitle": "Navigate the 3D datacube, inspect true-color composites, and extract class signatures.",
        "icon": "🔍",
        "color": "#0284c7",
        "rationale": (
            "A hyperspectral image is a 3-dimensional data tensor $(X \\times Y \\times \\lambda)$. Spatial images provide visual context, "
            "while each individual pixel contains a continuous spectral reflectance fingerprint. Before running machine learning, "
            "researchers must visually inspect the scene, identify regions of interest (ROIs), and extract pure class signatures."
        ),
        "how_to": [
            "<b>1. True-Color RGB Preview:</b> HyperVision automatically calculates optimal human perception RGB weights to render true-to-life color images.",
            "<b>2. Single-Band Inspection:</b> In the Left Control Panel, switch Display Mode to <b>'Single band'</b> and use the Wavelength Slider to slice through narrow bands (e.g. 680 nm chlorophyll absorption or 970 nm water absorption).",
            "<b>3. 3D Volumetric Cube View:</b> Click <b>'3D Cube View'</b> on the Ribbon to launch an interactive 3D volumetric visualizer with spatial and spectral slicing planes.",
            "<b>4. Spectral ROI Extraction:</b> In the <b>'Extract Spectra'</b> panel, choose your extraction mode (Point, Polygon ROI, or Freehand) and assign a class label (e.g. <i>Healthy</i>, <i>Diseased</i>, <i>Stressed</i>). Click and drag on the image to sample spectral signatures."
        ],
        "tip": "Collect at least 15–20 distinct ROI regions per target class distributed across different areas of the sample to ensure statistical representation of biological heterogeneity.",
        "action_text": "🧊 Launch 3D Volumetric Cube Viewer",
        "action_method": "generate_3d_viewer"
    },
    {
        "step_num": 3,
        "badge": "STAGE 3: CHEMOMETRIC PREPROCESSING",
        "title": "⚙️ Spectral Preprocessing & Continuum Correction",
        "subtitle": "Eliminate physical light scattering, baseline drift, and surface roughness noise.",
        "icon": "⚙️",
        "color": "#7c3aed",
        "rationale": (
            "In real-world hyperspectral imaging, physical phenomena such as particle size, surface geometry, and light scattering "
            "induce baseline slopes and additive/multiplicative offsets that obscure genuine chemical variations. Chemometric preprocessing "
            "standardizes spectra so that downstream machine learning models focus strictly on chemical absorption features."
        ),
        "how_to": [
            "<b>1. Launch Preprocessing Studio:</b> Click <b>'🔬 Preprocessing Studio'</b> on the Ribbon bar.",
            "<b>2. Standard Normal Variate (SNV):</b> Centers each spectrum to zero mean and unit variance. Highly recommended for grains, seeds, and powdered samples to cancel light scatter differences.",
            "<b>3. Multiplicative Scatter Correction (MSC):</b> Regresses individual spectra against the mean ideal spectrum to eliminate multiplicative pathlength variations.",
            "<b>4. Savitzky-Golay Derivatives (1st & 2nd):</b> Eliminates baseline shifts and resolves closely overlapping absorption peaks into sharp, distinct features.",
            "<b>5. Continuum Removal:</b> Normalizes spectra against an upper convex hull envelope to highlight exact absorption band depths."
        ],
        "tip": "Use SNV as your first choice for seed and leaf phenotyping. For resolving overlapping biochemical bands, apply Savitzky-Golay 2nd derivative with a window length of 11 or 15.",
        "action_text": "🔬 Open Interactive Preprocessing Studio",
        "action_method": "open_preprocessing_studio"
    },
    {
        "step_num": 4,
        "badge": "STAGE 4: SEGMENTATION & PHENOTYPING",
        "title": "🌾 Seed, Grain & Plant Organ Segmentation Studio",
        "subtitle": "Separate foreground targets from imaging trays and resolve touching grains with Watershed.",
        "icon": "🌾",
        "color": "#059669",
        "rationale": (
            "High-throughput phenotyping requires isolating seeds, grains, leaves, or disease lesions from the background tray. "
            "When seeds touch or overlap, standard thresholding merges them into single giant blobs. Marker-controlled Watershed "
            "instance segmentation solves this by using distance transforms to split touching kernels along their true physical contact boundary."
        ),
        "how_to": [
            "<b>1. Launch Segmentation Studio:</b> Click <b>'🌾 Grain & Plant Seg'</b> on the Ribbon bar.",
            "<b>2. Select Target Feature:</b> Choose <b>'🌾 Grain & Seed Brightness Index (GBI)'</b> for seeds on trays, or <b>'🌿 NDVI'</b> for green leaves and seedlings.",
            "<b>3. Enable Touching Grain Separation:</b> Check the box <b>'Separate Touching Grains (Watershed)'</b>. The algorithm calculates the Euclidean distance transform, identifies kernel peaks, and floods boundaries cleanly.",
            "<b>4. Morphological Filtering:</b> Set minimum area (e.g. 20 px) to automatically filter out background dust and chaff.",
            "<b>5. Inspect Morphometric Table:</b> View exact Area, Perimeter, Aspect Ratio (elongation), and Circularity for every individual grain.",
            "<b>6. Create ROIs for AI:</b> Click <b>'🏷️ Create Datacube ROIs'</b> to transfer all detected seeds directly into the main viewer as labeled classes ready for model training!"
        ],
        "tip": "The Segmentation Studio also plots the individual spectral reflectance curves for every segmented grain, instantly highlighting damaged, moldy, or outlier seeds!",
        "action_text": "🌾 Open Grain & Plant Segmentation Studio",
        "action_method": "open_datacube_segmentation_studio"
    },
    {
        "step_num": 5,
        "badge": "STAGE 5: FEATURE SELECTION & BIOMARKERS",
        "title": "📊 Feature Selection, Chemometrics & Biomarkers",
        "subtitle": "Identify key diagnostic wavelengths, eliminate collinearity, and discover biological biomarkers.",
        "icon": "📊",
        "color": "#d97706",
        "rationale": (
            "Hyperspectral sensors record hundreds of contiguous wavelengths. Training models on all bands causes the 'curse of dimensionality' "
            "and increases multicollinearity. Feature selection discovers the most informative diagnostic wavelengths, revealing "
            "specific chemical bonds (e.g., chlorophyll at 680 nm, water at 970 nm, protein at 1200 nm) and enabling the design of cost-effective multispectral cameras."
        ),
        "how_to": [
            "<b>1. Launch Chemometrics Studio:</b> Click <b>'🧬 Biomarkers & 2D-COS'</b> or <b>'CSV Analytics'</b> on the Ribbon.",
            "<b>2. PLS-DA VIP Biomarkers:</b> Variable Importance in Projection (VIP) scores identify bands with scores $> 1.0$, which are statistically significant drivers of class discrimination.",
            "<b>3. Non-Linear Feature Selection:</b> Evaluate Random Forest and Extra Trees Gini importance rankings across all spectral bands.",
            "<b>4. 2D-COS Spectroscopy:</b> Generate Synchronous and Asynchronous 2D Correlation Spectroscopy contour maps to explore dynamic intermolecular band relationships.",
            "<b>5. Export Ranked Wavelengths:</b> Export the top $N$ diagnostic wavelengths directly to CSV for multispectral hardware development."
        ],
        "tip": "Look for consensus wavelengths that appear in both PLS-DA VIP ($VIP > 1.0$) and Random Forest feature rankings for the most robust biological biomarkers.",
        "action_text": "🧬 Open Biomarkers & 2D-COS Studio",
        "action_method": "open_chemometrics_advanced_studio"
    },
    {
        "step_num": 6,
        "badge": "STAGE 6: SUPERVISED AI MODEL TRAINING",
        "title": "🤖 AI Machine Learning & Deep Learning Training (17 Models)",
        "subtitle": "Train, evaluate, and benchmark 17 state-of-the-art classifiers on extracted spectral classes.",
        "icon": "🤖",
        "color": "#dc2626",
        "rationale": (
            "Supervised classification trains artificial intelligence algorithms to accurately distinguish healthy vs diseased samples, "
            "classify seed varieties, or grade crop quality based on their spectral signatures. HyperVision Pro provides 17 distinct machine "
            "learning and deep learning model architectures with automated hyperparameter optimization."
        ),
        "how_to": [
            "<b>1. Launch AI Training:</b> Click <b>'Train AI Model'</b> on the Ribbon bar.",
            "<b>2. Multi-Datacube Training:</b> If multiple datacubes are open, the Multi-Datacube dialog lets you select and combine extracted spectral classes across multiple files into one unified training dataset.",
            "<b>3. Select Model Architectures:</b> Choose from Random Forest, Gradient Boosting, XGBoost, CatBoost, Linear & RBF SVM, Spectral 1D-CNN, Multilayer Perceptron (MLP), PLS-DA, Gaussian Naive Bayes, and more.",
            "<b>4. Data Augmentation & Cross-Validation:</b> Enable Gaussian jitter augmentation to improve generalization and configure test train splits ($20\\% - 30\\%$).",
            "<b>5. Automated Evaluation:</b> The training engine computes Accuracy, Precision, Recall, F1-Score, Cohen's Kappa, Confusion Matrices, and ROC AUC curves."
        ],
        "tip": "Always benchmark a fast linear baseline (such as PLS-DA or Linear SVM) against an ensemble method (Random Forest) and a deep neural network (1D-CNN) to report rigorous scientific comparisons.",
        "action_text": "🤖 Open AI Model Training Studio",
        "action_method": "train_ai_model"
    },
    {
        "step_num": 7,
        "badge": "STAGE 7: SPATIAL MAPPING & EXPLAINABLE AI",
        "title": "🗺️ Spatial Prediction Mapping & Explainable AI (XAI)",
        "subtitle": "Project model predictions across the entire datacube and explain decision rationale.",
        "icon": "🗺️",
        "color": "#0d9488",
        "rationale": (
            "Point and ROI classification only tests extracted samples. The true power of hyperspectral imaging is spatial prediction: "
            "applying the trained model across all $X \\times Y$ pixels to map disease infection spread or purity across an entire plant tray. "
            "Furthermore, Explainable AI (XAI) validates why the model made each decision, ensuring transparency."
        ),
        "how_to": [
            "<b>1. Launch Spatial Prediction:</b> Click <b>'🗺️ Spatial Map'</b> on the Ribbon bar.",
            "<b>2. Full-Cube Inference:</b> The studio applies the trained classifier pixel-by-pixel, rendering a full-resolution classified thematic map overlaid with vibrant class colors.",
            "<b>3. Shannon Entropy Uncertainty Map:</b> Visualizes prediction uncertainty across the scene. High-entropy pixels highlight ambiguous boundary zones requiring researcher attention.",
            "<b>4. XAI Waterfall Plot:</b> In <b>'🎨 Viz Studio'</b>, inspect SHAP-style Waterfall plots showing how individual spectral bands pushed the prediction toward or against a specific class."
        ],
        "tip": "Use the Shannon Entropy overlay to identify early, asymptomatic infection regions where the model detects slight spectral deviations before visible symptoms appear.",
        "action_text": "🗺️ Open Spatial Prediction Studio",
        "action_method": "open_spatial_classification_studio"
    },
    {
        "step_num": 8,
        "badge": "STAGE 8: PUBLICATION DOSSIER & REPORTING",
        "title": "📑 Automated Scientific Dossier & Publication Reporting",
        "subtitle": "Generate publication-grade Word (.docx) and PDF dossiers with embedded figures and audit trails.",
        "icon": "📑",
        "color": "#4f46e5",
        "rationale": (
            "Reproducibility is paramount in scientific research. HyperVision Pro automates report compilation by gathering calibration metadata, "
            "preprocessing parameters, model benchmarks, confusion matrices, ROC curves, and VIP biomarker tables into structured Microsoft Word (.docx) documents."
        ),
        "how_to": [
            "<b>1. Launch Dossier Generator:</b> Click <b>'📑 Research Dossier'</b> or <b>'Word Report'</b> on the Ribbon bar.",
            "<b>2. Fill Study Details:</b> Enter your Research Title, Hypothesis, Lead Author Name, Institution, and Department.",
            "<b>3. Customize Report Sections:</b> Toggle Executive Summary, Spectral Calibration, Preprocessing Method Comparison, Model Performance Benchmarks, Confusion Matrices, and Biomarker Tables.",
            "<b>4. Generate Word Document:</b> Click <b>'Generate Word (.docx) Report'</b> to produce a formatted, justified document complete with 12 pt font, 1.5 line spacing, and embedded high-resolution figures."
        ],
        "tip": "The generated Word document provides a ready-made Materials & Methods and Results section for immediate submission to peer-reviewed scientific journals!",
        "action_text": "📑 Open Scientific Dossier Generator",
        "action_method": "open_scientific_dossier_generator"
    }
]


class HyperspectralPipelineTutorialDialog(QDialog):
    """Interactive, step-by-step guided tutorial dialog for the Hyperspectral Imaging Pipeline."""
    def __init__(self, parent_viewer=None, initial_step=0, parent=None):
        super().__init__(parent or parent_viewer)
        self.parent_viewer = parent_viewer
        self.setWindowTitle("🎓 Guided Hyperspectral Imaging Pipeline Tutorial — HyperVision Pro")
        self.resize(1020, 680)
        self.setMinimumSize(850, 560)
        self.current_step = max(0, min(initial_step, len(PIPELINE_STEPS) - 1))

        self._init_ui()
        self._load_step(self.current_step)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Top Header Banner
        header = QFrame(self)
        header.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e3a8a, stop:1 #312e81);"
            "border-radius: 8px; padding: 12px 18px;"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(0, 0, 0, 0)

        title_box = QVBoxLayout()
        h_title = QLabel("🎓 HyperVision Pro — Guided Hyperspectral Imaging Pipeline", header)
        h_title.setStyleSheet("color: white; font-size: 17px; font-weight: bold;")
        h_subtitle = QLabel(
            "A comprehensive, step-by-step scientific guide to calibration, preprocessing, segmentation, AI training, and publication reporting.",
            header
        )
        h_subtitle.setStyleSheet("color: #93c5fd; font-size: 11px;")
        title_box.addWidget(h_title)
        title_box.addWidget(h_subtitle)
        h_layout.addLayout(title_box)
        h_layout.addStretch()

        badge = QLabel("Interactive Scientific Walkthrough", header)
        badge.setStyleSheet(
            "background-color: rgba(255, 255, 255, 0.15); color: #e0e7ff; font-weight: bold; "
            "padding: 6px 12px; border-radius: 6px; font-size: 11px;"
        )
        h_layout.addWidget(badge)
        main_layout.addWidget(header)

        # 2. Main Content Split: Left Steps List + Right Card Stack
        body_layout = QHBoxLayout()
        body_layout.setSpacing(14)

        # Left Steps Navigator
        left_panel = QFrame(self)
        left_panel.setFixedWidth(280)
        left_panel.setStyleSheet(
            "background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;"
        )
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(8, 10, 8, 10)
        left_layout.setSpacing(6)

        lbl_steps = QLabel("<b>Scientific Pipeline Stages:</b>", left_panel)
        lbl_steps.setStyleSheet("color: #334155; font-size: 12px; padding: 4px;")
        left_layout.addWidget(lbl_steps)

        self.step_list = QListWidget(left_panel)
        self.step_list.setStyleSheet(
            "QListWidget { background-color: transparent; border: none; font-size: 11px; }"
            "QListWidget::item { padding: 8px 10px; border-radius: 6px; margin-bottom: 4px; color: #1e293b; }"
            "QListWidget::item:hover { background-color: #e2e8f0; }"
            "QListWidget::item:selected { background-color: #2563eb; color: white; font-weight: bold; }"
        )
        for s in PIPELINE_STEPS:
            item = QListWidgetItem(f"{s['icon']} Step {s['step_num']}: {s['badge'].split(':')[1].strip()}")
            item.setData(Qt.UserRole, s['step_num'] - 1)
            self.step_list.addItem(item)
        self.step_list.currentRowChanged.connect(self._on_step_list_clicked)
        left_layout.addWidget(self.step_list)
        body_layout.addWidget(left_panel)

        # Right Card Container (Scrollable)
        right_panel = QFrame(self)
        right_panel.setStyleSheet(
            "background-color: white; border: 1px solid #e2e8f0; border-radius: 8px;"
        )
        right_vbox = QVBoxLayout(right_panel)
        right_vbox.setContentsMargins(16, 16, 16, 16)
        right_vbox.setSpacing(12)

        scroll = QScrollArea(right_panel)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.card_content = QWidget()
        self.card_layout = QVBoxLayout(self.card_content)
        self.card_layout.setContentsMargins(4, 4, 12, 4)
        self.card_layout.setSpacing(12)

        # Dynamic widgets in Card:
        self.lbl_stage_badge = QLabel("")
        self.lbl_stage_badge.setStyleSheet(
            "background-color: #dbeafe; color: #1e40af; font-weight: bold; padding: 4px 10px; "
            "border-radius: 4px; font-size: 10px; text-transform: uppercase;"
        )
        self.card_layout.addWidget(self.lbl_stage_badge, alignment=Qt.AlignLeft)

        self.lbl_card_title = QLabel("")
        self.lbl_card_title.setWordWrap(True)
        self.lbl_card_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #0f172a;")
        self.card_layout.addWidget(self.lbl_card_title)

        self.lbl_card_subtitle = QLabel("")
        self.lbl_card_subtitle.setWordWrap(True)
        self.lbl_card_subtitle.setStyleSheet("font-size: 12px; color: #64748b; margin-bottom: 4px;")
        self.card_layout.addWidget(self.lbl_card_subtitle)

        # Scientific Rationale Box
        self.box_rationale = QFrame()
        self.box_rationale.setStyleSheet("background-color: #f1f5f9; border-left: 4px solid #3b82f6; border-radius: 4px; padding: 10px;")
        rat_layout = QVBoxLayout(self.box_rationale)
        rat_title = QLabel("<b>🔬 Scientific Purpose & Theoretical Rationale:</b>")
        rat_title.setStyleSheet("color: #1e3a8a; font-size: 12px;")
        rat_layout.addWidget(rat_title)
        self.lbl_rationale = QLabel("")
        self.lbl_rationale.setWordWrap(True)
        self.lbl_rationale.setStyleSheet("color: #334155; font-size: 11px; line-height: 1.4;")
        rat_layout.addWidget(self.lbl_rationale)
        self.card_layout.addWidget(self.box_rationale)

        # How to execute in HyperVision Box
        self.box_howto = QFrame()
        self.box_howto.setStyleSheet("background-color: #fafaf9; border-left: 4px solid #10b981; border-radius: 4px; padding: 10px;")
        how_layout = QVBoxLayout(self.box_howto)
        how_title = QLabel("<b>🛠️ How to Execute in HyperVision Pro:</b>")
        how_title.setStyleSheet("color: #065f46; font-size: 12px;")
        how_layout.addWidget(how_title)
        self.lbl_howto = QLabel("")
        self.lbl_howto.setWordWrap(True)
        self.lbl_howto.setStyleSheet("color: #1c1917; font-size: 11px; line-height: 1.5;")
        how_layout.addWidget(self.lbl_howto)
        self.card_layout.addWidget(self.box_howto)

        # Best Practice Tip Box
        self.box_tip = QFrame()
        self.box_tip.setStyleSheet("background-color: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 4px; padding: 10px;")
        tip_layout = QVBoxLayout(self.box_tip)
        tip_title = QLabel("<b>💡 Best Practice & Scientific Recommendation:</b>")
        tip_title.setStyleSheet("color: #92400e; font-size: 12px;")
        tip_layout.addWidget(tip_title)
        self.lbl_tip = QLabel("")
        self.lbl_tip.setWordWrap(True)
        self.lbl_tip.setStyleSheet("color: #78350f; font-size: 11px; line-height: 1.4;")
        tip_layout.addWidget(self.lbl_tip)
        self.card_layout.addWidget(self.box_tip)

        # Direct Interactive Action Button
        self.btn_try_action = QPushButton("")
        self.btn_try_action.setStyleSheet(
            "QPushButton { background-color: #2563eb; color: white; font-weight: bold; padding: 10px 20px; border-radius: 6px; font-size: 12px; }"
            "QPushButton:hover { background-color: #1d4ed8; }"
        )
        self.btn_try_action.clicked.connect(self._on_action_clicked)
        self.card_layout.addWidget(self.btn_try_action, alignment=Qt.AlignLeft)
        self.card_layout.addStretch()

        scroll.setWidget(self.card_content)
        right_vbox.addWidget(scroll)
        body_layout.addWidget(right_panel, stretch=1)
        main_layout.addLayout(body_layout, stretch=1)

        # 3. Bottom Control Navigation Bar
        bottom_bar = QHBoxLayout()
        bottom_bar.setContentsMargins(4, 4, 4, 4)

        self.chk_startup = QCheckBox("Show this Guided Pipeline Tutorial on startup", self)
        self.chk_startup.setChecked(True)
        self.chk_startup.setStyleSheet("color: #475569; font-size: 11px; font-weight: 500;")
        self.chk_startup.toggled.connect(self._on_startup_toggled)
        bottom_bar.addWidget(self.chk_startup)
        bottom_bar.addStretch()

        self.lbl_step_indicator = QLabel("Step 1 of 8", self)
        self.lbl_step_indicator.setStyleSheet("color: #64748b; font-weight: bold; font-size: 12px; padding: 0 12px;")
        bottom_bar.addWidget(self.lbl_step_indicator)

        self.btn_prev = QPushButton("⬅️ Previous Step", self)
        self.btn_prev.setStyleSheet("padding: 7px 16px; font-size: 11px; font-weight: bold;")
        self.btn_prev.clicked.connect(self._on_prev_step)
        bottom_bar.addWidget(self.btn_prev)

        self.btn_next = QPushButton("Next Step ➡️", self)
        self.btn_next.setStyleSheet("padding: 7px 16px; font-size: 11px; font-weight: bold; background-color: #0284c7; color: white;")
        self.btn_next.clicked.connect(self._on_next_step)
        bottom_bar.addWidget(self.btn_next)

        btn_close = QPushButton("Start Analyzing", self)
        btn_close.setStyleSheet("padding: 7px 18px; font-size: 11px; font-weight: bold; background-color: #10b981; color: white; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_close)

        main_layout.addLayout(bottom_bar)

        # Load preference
        if self.parent_viewer and hasattr(self.parent_viewer, "user_profile"):
            show = getattr(self.parent_viewer.user_profile, "show_tutorial_on_startup", True)
            self.chk_startup.setChecked(bool(show))

    def _load_step(self, step_idx: int):
        self.current_step = step_idx
        data = PIPELINE_STEPS[step_idx]

        self.step_list.blockSignals(True)
        self.step_list.setCurrentRow(step_idx)
        self.step_list.blockSignals(False)

        self.lbl_stage_badge.setText(data["badge"])
        self.lbl_card_title.setText(f"{data['icon']} {data['title']}")
        self.lbl_card_subtitle.setText(data["subtitle"])
        self.lbl_rationale.setText(data["rationale"])

        howto_html = "<ul style='margin-left: -20px; margin-top: 4px;'>" + "".join(f"<li style='margin-bottom: 6px;'>{item}</li>" for item in data["how_to"]) + "</ul>"
        self.lbl_howto.setText(howto_html)
        self.lbl_tip.setText(data["tip"])
        self.btn_try_action.setText(f"🚀 {data['action_text']}")

        self.lbl_step_indicator.setText(f"Step {step_idx + 1} of {len(PIPELINE_STEPS)}")
        self.btn_prev.setEnabled(step_idx > 0)
        self.btn_next.setEnabled(step_idx < len(PIPELINE_STEPS) - 1)

    def _on_step_list_clicked(self, row: int):
        if 0 <= row < len(PIPELINE_STEPS):
            self._load_step(row)

    def _on_prev_step(self):
        if self.current_step > 0:
            self._load_step(self.current_step - 1)

    def _on_next_step(self):
        if self.current_step < len(PIPELINE_STEPS) - 1:
            self._load_step(self.current_step + 1)

    def _on_startup_toggled(self, checked: bool):
        if self.parent_viewer and hasattr(self.parent_viewer, "user_profile"):
            self.parent_viewer.user_profile.show_tutorial_on_startup = checked
            self.parent_viewer.user_profile.save()

    def _on_action_clicked(self):
        data = PIPELINE_STEPS[self.current_step]
        method_name = data.get("action_method")
        if self.parent_viewer and hasattr(self.parent_viewer, method_name):
            # Close dialog so the user can interact directly with the chosen tool
            self.accept()
            method = getattr(self.parent_viewer, method_name)
            try:
                method()
            except Exception as e:
                QMessageBox.critical(self.parent_viewer, "Tool Error", f"Could not launch {data['title']}: {e}")
        else:
            QMessageBox.information(
                self, "Tool Ready",
                f"To use this tool, click the corresponding button on the HyperVision Pro ribbon once you return to the main workspace."
            )
