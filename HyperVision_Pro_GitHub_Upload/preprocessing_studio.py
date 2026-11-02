# -*- coding: utf-8 -*-
"""
HyperVision Pro - Spectral Preprocessing Comparison Studio
Interactive comparative visualization and diagnostics for Savitzky-Golay,
Continuum Removal, SNV, MSC, Derivatives, and Residual Analysis.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QComboBox,
    QWidget, QCheckBox, QSplitter, QGroupBox, QSpinBox, QSlider, QTabWidget
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from scipy.signal import savgol_filter, find_peaks
from scipy.spatial import ConvexHull


def compute_continuum_removal(wavelengths: np.ndarray, spectrum: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Computes continuum curve (upper convex hull) and continuum-removed band depths.
    Returns (continuum_curve, cr_spectrum, absorption_features).
    """
    wv = np.asarray(wavelengths, dtype=np.float64)
    spec = np.asarray(spectrum, dtype=np.float64)

    pts = np.column_stack((wv, spec))
    try:
        hull = ConvexHull(pts)
        top_verts = sorted([v for v in hull.vertices if pts[v, 1] >= np.percentile(spec, 30)])
        if len(top_verts) < 2:
            top_verts = [0, len(wv) - 1]
        top_wv = wv[top_verts]
        top_y = spec[top_verts]
        continuum = np.interp(wv, top_wv, top_y)
        # Ensure continuum is strictly >= spectrum
        continuum = np.maximum(continuum, spec)
    except Exception:
        # Linear continuum between endpoints fallback
        continuum = np.interp(wv, [wv[0], wv[-1]], [spec[0], spec[-1]])
        continuum = np.maximum(continuum, spec)

    cr_spectrum = np.clip(spec / (continuum + 1e-8), 0.0, 1.0)

    # Detect absorption band depths (inverted peaks in CR spectrum)
    depth = 1.0 - cr_spectrum
    peaks, props = find_peaks(depth, distance=10, prominence=0.03)
    features = []
    for pk in peaks:
        features.append({
            "wavelength": float(wv[pk]),
            "band_depth": float(depth[pk]),
            "prominence": float(props["prominences"][np.where(peaks == pk)[0][0]]) if "prominences" in props else 0.0
        })
    features.sort(key=lambda x: x["band_depth"], reverse=True)
    return continuum, cr_spectrum, features


class PreprocessingStudioDialog(QDialog):
    """Studio for inspecting and comparing preprocessing transformations on spectra."""
    def __init__(self, wavelengths: Optional[np.ndarray] = None, spectra: Optional[np.ndarray] = None, labels: Optional[List[str]] = None, parent_viewer=None, parent=None, *args, **kwargs):
        super().__init__(parent)
        self.setWindowTitle("HyperVision Pro — Spectral Preprocessing Comparison Studio")
        self.resize(1260, 800)

        self.parent_viewer = parent_viewer or parent
        self.wavelengths = np.asarray(wavelengths if wavelengths is not None else np.linspace(400, 1000, 100), dtype=np.float64)
        if spectra is not None:
            self.spectra = np.asarray(spectra, dtype=np.float64)
            if self.spectra.ndim == 1:
                self.spectra = self.spectra.reshape(1, -1)
        else:
            self.spectra = np.zeros((1, len(self.wavelengths)))
        self.labels = labels if labels else [f"Sample {i+1}" for i in range(len(self.spectra))]

        self._build_ui()
        self.populate_spectra_combo()
        self.update_studio_plot()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # Header banner
        header = QLabel(
            "<div style='background-color: #0f172a; color: white; padding: 10px 14px; border-radius: 6px;'>"
            "<table width='100%'><tr>"
            "<td><h3 style='margin: 0; color: #38bdf8;'>🔬 Spectral Preprocessing Comparison Studio</h3>"
            "<span style='font-size: 11px; color: #94a3b8;'>Analyze signal transformations, continuum removal absorption features, Savitzky-Golay noise residuals, and derivatives.</span></td>"
            "</tr></table></div>"
        )
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        # ======================================================================
        # LEFT PANEL: Parameters & Controls
        # ======================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Data source box
        src_box = QGroupBox("Target Spectrum Source")
        src_layout = QVBoxLayout(src_box)

        row_src = QHBoxLayout()
        self.btn_load_roi = QPushButton("📍 Load Active ROI")
        self.btn_load_roi.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px;")
        self.btn_load_roi.clicked.connect(self.load_from_roi)
        row_src.addWidget(self.btn_load_roi)

        self.btn_load_csv = QPushButton("📂 Load CSV")
        self.btn_load_csv.clicked.connect(self.load_from_csv)
        row_src.addWidget(self.btn_load_csv)
        src_layout.addLayout(row_src)

        src_layout.addWidget(QLabel("Select Active Spectrum:"))
        self.cmb_spectrum = QComboBox()
        self.cmb_spectrum.currentIndexChanged.connect(self.update_studio_plot)
        src_layout.addWidget(self.cmb_spectrum)
        left_layout.addWidget(src_box)

        # Transformation selection box
        prep_box = QGroupBox("Preprocessing Method")
        prep_layout = QVBoxLayout(prep_box)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems([
            "Savitzky-Golay Smoothing",
            "Continuum Removal (Convex Hull)",
            "Standard Normal Variate (SNV)",
            "1st Derivative (Inflection / Red Edge)",
            "2nd Derivative (Absorption Minima)",
            "Absorbance [-log10(R)]",
            "Multiplicative Scatter Correction (MSC)"
        ])
        self.cmb_method.currentIndexChanged.connect(self.on_method_changed)
        prep_layout.addWidget(self.cmb_method)

        # Savitzky-Golay parameters widget
        self.sg_widget = QWidget()
        sg_l = QVBoxLayout(self.sg_widget)
        sg_l.setContentsMargins(0, 0, 0, 0)

        row_win = QHBoxLayout()
        row_win.addWidget(QLabel("Window Size (odd):"))
        self.spin_window = QSpinBox()
        self.spin_window.setRange(5, 51)
        self.spin_window.setSingleStep(2)
        self.spin_window.setValue(11)
        self.spin_window.valueChanged.connect(self.update_studio_plot)
        row_win.addWidget(self.spin_window)
        sg_l.addLayout(row_win)

        row_poly = QHBoxLayout()
        row_poly.addWidget(QLabel("Polynomial Order:"))
        self.spin_poly = QSpinBox()
        self.spin_poly.setRange(1, 5)
        self.spin_poly.setValue(2)
        self.spin_poly.valueChanged.connect(self.update_studio_plot)
        row_poly.addWidget(self.spin_poly)
        sg_l.addLayout(row_poly)
        prep_layout.addWidget(self.sg_widget)

        # Options
        self.chk_show_residual = QCheckBox("Show Difference / Residual Curve")
        self.chk_show_residual.setChecked(True)
        self.chk_show_residual.stateChanged.connect(self.update_studio_plot)
        prep_layout.addWidget(self.chk_show_residual)

        self.chk_overlay_all = QCheckBox("Overlay All Loaded Spectra")
        self.chk_overlay_all.stateChanged.connect(self.update_studio_plot)
        prep_layout.addWidget(self.chk_overlay_all)

        left_layout.addWidget(prep_box)

        # Diagnostics & Absorption Features Table
        diag_box = QGroupBox("Signal Metrics & Features")
        diag_layout = QVBoxLayout(diag_box)

        self.table_features = QTableWidget(0, 3)
        self.table_features.setHorizontalHeaderLabels(["Wavelength (nm)", "Feature Type", "Depth / Value"])
        self.table_features.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        diag_layout.addWidget(self.table_features)

        self.lbl_snr = QLabel("Estimated SNR: -- dB")
        self.lbl_snr.setStyleSheet("font-weight: bold; color: #1e293b;")
        diag_layout.addWidget(self.lbl_snr)

        left_layout.addWidget(diag_box)

        # Export Button
        btn_export = QPushButton("💾 Export Preprocessed Spectrum (CSV)...")
        btn_export.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px;")
        btn_export.clicked.connect(self.export_current_preprocessed)
        left_layout.addWidget(btn_export)

        splitter.addWidget(left_widget)

        # ======================================================================
        # RIGHT PANEL: Matplotlib Comparative Figures
        # ======================================================================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.nav_toolbar)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        # Close
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setMaximumWidth(120)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)

    def populate_spectra_combo(self):
        self.cmb_spectrum.blockSignals(True)
        self.cmb_spectrum.clear()
        for i, lbl in enumerate(self.labels):
            self.cmb_spectrum.addItem(f"{i+1}: {lbl}")
        self.cmb_spectrum.blockSignals(False)

    def on_method_changed(self):
        method = self.cmb_method.currentText()
        is_sg = ("Savitzky-Golay" in method or "Derivative" in method)
        self.sg_widget.setVisible(is_sg)
        self.update_studio_plot()

    # --------------------------------------------------------------------------
    # DATA LOADING
    # --------------------------------------------------------------------------
    def load_from_roi(self):
        if not self.parent_viewer or not hasattr(self.parent_viewer, "image_windows"):
            QMessageBox.warning(self, "No Viewer", "Active image viewer is not available.")
            return
        win = self.parent_viewer.get_active_window()
        if not win or not win.selections:
            QMessageBox.warning(self, "No ROIs", "Please create ROI or point selections on the datacube first.")
            return

        wv = win.document.wavelengths
        profiles = []
        labels = []
        for s in win.selections:
            if s.get("profile") is not None and len(s["profile"]) == len(wv):
                profiles.append(s["profile"])
                labels.append(f"{s.get('class_label', 'ROI')} (id:{s.get('id', '')})")

        if profiles:
            self.wavelengths = wv
            self.spectra = np.array(profiles)
            self.labels = labels
            self.populate_spectra_combo()
            self.update_studio_plot()
            QMessageBox.information(self, "Loaded", f"Loaded {len(profiles)} spectra from active datacube.")

    def load_from_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Spectral CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            num_cols = df.select_dtypes(include=[np.number]).columns
            wvs = []
            valid_cols = []
            for c in num_cols:
                try:
                    wvs.append(float(str(c).replace("Wavelength_", "")))
                    valid_cols.append(c)
                except ValueError:
                    pass
            if not wvs:
                QMessageBox.critical(self, "Error", "No numeric wavelength columns found.")
                return

            self.wavelengths = np.array(wvs)
            self.spectra = df[valid_cols].values
            class_col = next((c for c in df.columns if str(c).lower() in ["class_label", "class", "label"]), None)
            if class_col is not None:
                self.labels = df[class_col].astype(str).tolist()
            else:
                self.labels = [f"Sample {i+1}" for i in range(len(self.spectra))]

            self.populate_spectra_combo()
            self.update_studio_plot()
            QMessageBox.information(self, "Loaded", f"Loaded {len(self.spectra)} spectra from CSV.")
        except Exception as e:
            QMessageBox.critical(self, "Failed", f"Could not load CSV:\n{e}")

    # --------------------------------------------------------------------------
    # COMPARATIVE PLOTTING & TRANSFORMS
    # --------------------------------------------------------------------------
    def update_studio_plot(self):
        if len(self.spectra) == 0:
            return

        idx = self.cmb_spectrum.currentIndex()
        if idx < 0 or idx >= len(self.spectra):
            idx = 0

        wv = self.wavelengths
        raw = self.spectra[idx].copy()
        method = self.cmb_method.currentText()
        show_resid = self.chk_show_residual.isChecked()
        overlay_all = self.chk_overlay_all.isChecked()

        win_len = self.spin_window.value()
        if win_len % 2 == 0:
            win_len += 1
        poly_order = min(self.spin_poly.value(), win_len - 1)

        # Compute preprocessed curve
        continuum_curve = None
        cr_features = []
        if method == "Savitzky-Golay Smoothing":
            w = min(win_len, len(raw) if len(raw) % 2 == 1 else len(raw) - 1)
            prep = savgol_filter(raw, window_length=max(5, w), polyorder=poly_order)
            ylab = "Smoothed Reflectance"
        elif method == "Continuum Removal (Convex Hull)":
            continuum_curve, prep, cr_features = compute_continuum_removal(wv, raw)
            ylab = "Continuum-Removed Band Depth"
        elif method == "Standard Normal Variate (SNV)":
            prep = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
            ylab = "SNV Normalized Score (Z)"
        elif method == "1st Derivative (Inflection / Red Edge)":
            w = min(win_len, len(raw) if len(raw) % 2 == 1 else len(raw) - 1)
            prep = savgol_filter(raw, window_length=max(5, w), polyorder=poly_order, deriv=1)
            ylab = "1st Derivative (dR/dλ)"
        elif method == "2nd Derivative (Absorption Minima)":
            w = min(win_len, len(raw) if len(raw) % 2 == 1 else len(raw) - 1)
            prep = savgol_filter(raw, window_length=max(5, w), polyorder=max(poly_order, 3), deriv=2)
            ylab = "2nd Derivative (d²R/dλ²)"
        elif method == "Absorbance [-log10(R)]":
            prep = -np.log10(np.clip(raw, 1e-6, None))
            ylab = "Absorbance (AU)"
        elif method == "Multiplicative Scatter Correction (MSC)":
            ref = np.mean(self.spectra, axis=0) if len(self.spectra) > 1 else raw
            fit = np.polyfit(ref, raw, 1)
            prep = (raw - fit[1]) / (fit[0] + 1e-8)
            ylab = "MSC Corrected Reflectance"
        else:
            prep = raw.copy()
            ylab = "Reflectance"

        self.figure.clear()

        # Decide subplot layout
        if show_resid and continuum_curve is None:
            gs = self.figure.add_gridspec(3, 1, hspace=0.35)
            ax_top = self.figure.add_subplot(gs[0:2, 0])
            ax_bot = self.figure.add_subplot(gs[2, 0], sharex=ax_top)
        elif continuum_curve is not None:
            gs = self.figure.add_gridspec(2, 1, hspace=0.3)
            ax_top = self.figure.add_subplot(gs[0, 0])
            ax_bot = self.figure.add_subplot(gs[1, 0], sharex=ax_top)
        else:
            ax_top = self.figure.add_subplot(111)
            ax_bot = None

        # --- TOP PLOT: RAW & CONTINUUM OVERLAYS ---
        if overlay_all and len(self.spectra) > 1:
            for i in range(min(15, len(self.spectra))):
                alpha = 0.8 if i == idx else 0.25
                lw = 2.2 if i == idx else 1.0
                ax_top.plot(wv, self.spectra[i], alpha=alpha, linewidth=lw, label=self.labels[i] if i == idx else None)
        else:
            ax_top.plot(wv, raw, label="Raw Spectrum", color="#0284c7", linewidth=2.2)

        if continuum_curve is not None:
            ax_top.plot(wv, continuum_curve, label="Fitted Continuum (Convex Hull)", color="#dc2626", linestyle="--", linewidth=2.0)
            ax_top.legend(loc="best", framealpha=0.9)
            ax_top.set_title("Upper Convex Hull Continuum", fontsize=11, fontweight="bold")
            ax_top.set_ylabel("Raw Reflectance", fontsize=9, fontweight="bold")
            ax_top.grid(True, linestyle="--", alpha=0.5)

            # --- BOTTOM PLOT: CONTINUUM REMOVED ---
            ax_bot.plot(wv, prep, label="Continuum Removed (R / Rc)", color="#059669", linewidth=2.2)
            ax_bot.axhline(1.0, color="#64748b", linestyle=":", alpha=0.8)

            # Annotate absorption depths
            self.table_features.setRowCount(0)
            for feat in cr_features[:5]:
                pk_wv = feat["wavelength"]
                pk_depth = feat["band_depth"]
                ax_bot.annotate(f"{pk_wv:.0f}nm (-{pk_depth:.2f})", (pk_wv, 1.0 - pk_depth),
                                textcoords="offset points", xytext=(0, -18), ha="center", fontsize=8, color="#dc2626", fontweight="bold")

                row = self.table_features.rowCount()
                self.table_features.insertRow(row)
                self.table_features.setItem(row, 0, QTableWidgetItem(f"{pk_wv:.1f}"))
                self.table_features.setItem(row, 1, QTableWidgetItem("Absorption Trough"))
                self.table_features.setItem(row, 2, QTableWidgetItem(f"{pk_depth:.4f}"))

            ax_bot.set_title("Normalized Absorption Depth", fontsize=11, fontweight="bold")
            ax_bot.set_ylabel(ylab, fontsize=9, fontweight="bold")
            ax_bot.set_xlabel("Wavelength (nm)", fontsize=9, fontweight="bold")
            ax_bot.grid(True, linestyle="--", alpha=0.5)
            ax_bot.legend(loc="best", framealpha=0.9)

        else:
            # Standard Preprocessing
            ax_top.plot(wv, prep, label=f"Preprocessed: {method}", color="#16a34a", linewidth=2.2)
            ax_top.set_title(f"Comparison: Raw vs {method}", fontsize=11, fontweight="bold")
            ax_top.set_ylabel(ylab, fontsize=9, fontweight="bold")
            ax_top.grid(True, linestyle="--", alpha=0.5)
            ax_top.legend(loc="best", framealpha=0.9)

            # Diagnostics table for derivative peaks
            self.table_features.setRowCount(0)
            if "1st Derivative" in method:
                # Find red-edge inflection peak around 690-740nm
                re_mask = (wv >= 680) & (wv <= 750)
                if np.any(re_mask):
                    re_wvs = wv[re_mask]
                    re_vals = prep[re_mask]
                    max_i = np.argmax(re_vals)
                    row = self.table_features.rowCount()
                    self.table_features.insertRow(row)
                    self.table_features.setItem(row, 0, QTableWidgetItem(f"{re_wvs[max_i]:.1f}"))
                    self.table_features.setItem(row, 1, QTableWidgetItem("Red Edge Inflection (λre)"))
                    self.table_features.setItem(row, 2, QTableWidgetItem(f"{re_vals[max_i]:.6f}"))
                    ax_top.axvline(re_wvs[max_i], color="#ea580c", linestyle="--", alpha=0.8)

            if ax_bot is not None:
                residual = raw - prep if method != "Absorbance [-log10(R)]" else np.zeros_like(raw)
                ax_bot.plot(wv, residual, color="#9333ea", linewidth=1.8, label="Residual (Raw - Preprocessed)")
                ax_bot.axhline(0.0, color="#64748b", linestyle=":", alpha=0.8)
                ax_bot.set_title("Transformation Residual / High-Frequency Noise Removed", fontsize=10, fontweight="bold")
                ax_bot.set_ylabel("Δ Intensity", fontsize=8, fontweight="bold")
                ax_bot.set_xlabel("Wavelength (nm)", fontsize=9, fontweight="bold")
                ax_bot.grid(True, linestyle="--", alpha=0.5)
                ax_bot.legend(loc="best", framealpha=0.9)

                # Estimate SNR
                noise_std = np.std(residual)
                sig_mean = np.mean(raw)
                snr = 20 * np.log10(max(sig_mean / (noise_std + 1e-8), 1.0))
                self.lbl_snr.setText(f"Estimated SNR: {snr:.1f} dB (Noise σ = {noise_std:.5f})")

        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    def export_current_preprocessed(self):
        idx = self.cmb_spectrum.currentIndex()
        if idx < 0 or idx >= len(self.spectra):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Preprocessed Spectrum", "preprocessed_spectrum.csv", "CSV Files (*.csv)")
        if path:
            try:
                # Preprocess current
                method = self.cmb_method.currentText()
                raw = self.spectra[idx]
                win_len = self.spin_window.value()
                poly = self.spin_poly.value()

                if "Savitzky-Golay" in method:
                    prep = savgol_filter(raw, window_length=max(5, win_len if win_len%2==1 else win_len-1), polyorder=poly)
                elif "Continuum Removal" in method:
                    _, prep, _ = compute_continuum_removal(self.wavelengths, raw)
                elif "SNV" in method:
                    prep = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)
                elif "1st Derivative" in method:
                    prep = savgol_filter(raw, window_length=max(5, win_len if win_len%2==1 else win_len-1), polyorder=poly, deriv=1)
                elif "2nd Derivative" in method:
                    prep = savgol_filter(raw, window_length=max(5, win_len if win_len%2==1 else win_len-1), polyorder=max(poly, 3), deriv=2)
                else:
                    prep = raw

                import pandas as pd
                df = pd.DataFrame({
                    "Wavelength_nm": self.wavelengths,
                    "Raw_Spectrum": raw,
                    f"Preprocessed_{method}": prep
                })
                df.to_csv(path, index=False)
                QMessageBox.information(self, "Export Successful", f"Saved preprocessed data to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Failed to export CSV:\n{e}")
