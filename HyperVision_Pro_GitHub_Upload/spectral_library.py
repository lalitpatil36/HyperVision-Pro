# -*- coding: utf-8 -*-
"""
HyperVision Pro - Spectral Library Generator, Manager & Similarity Matcher
Enables curation of reference endmembers, statistical confidence envelopes,
standard ENVI (.sli / .hdr) and CSV/JSON export, and live spectral matching.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QComboBox,
    QWidget, QCheckBox, QSplitter, QGroupBox, QColorDialog, QTextBrowser,
    QTabWidget, QInputDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# ==============================================================================
# SPECTRAL SIMILARITY METRICS
# ==============================================================================

def compute_sam_angle(s1: np.ndarray, s2: np.ndarray) -> float:
    """Computes Spectral Angle Mapper (SAM) in degrees."""
    v1 = np.asarray(s1, dtype=np.float64).flatten()
    v2 = np.asarray(s2, dtype=np.float64).flatten()
    if len(v1) != len(v2):
        v2 = np.interp(np.linspace(0, 1, len(v1)), np.linspace(0, 1, len(v2)), v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 90.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def compute_scm_correlation(s1: np.ndarray, s2: np.ndarray) -> float:
    """Computes Spectral Correlation Measure (Pearson r)."""
    v1 = np.asarray(s1, dtype=np.float64).flatten()
    v2 = np.asarray(s2, dtype=np.float64).flatten()
    if np.std(v1) == 0 or np.std(v2) == 0:
        return 0.0
    corr = np.corrcoef(v1, v2)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0

def compute_sid_divergence(s1: np.ndarray, s2: np.ndarray) -> float:
    """Computes Spectral Information Divergence (SID)."""
    v1 = np.clip(np.asarray(s1, dtype=np.float64).flatten(), 1e-10, None)
    v2 = np.clip(np.asarray(s2, dtype=np.float64).flatten(), 1e-10, None)
    p = v1 / np.sum(v1)
    q = v2 / np.sum(v2)
    d_pq = np.sum(p * np.log(p / q))
    d_qp = np.sum(q * np.log(q / p))
    sid = d_pq + d_qp
    return float(sid) if np.isfinite(sid) else 999.0

def compute_rmse(s1: np.ndarray, s2: np.ndarray) -> float:
    """Computes Root Mean Square Error (Euclidean metric)."""
    v1 = np.asarray(s1, dtype=np.float64).flatten()
    v2 = np.asarray(s2, dtype=np.float64).flatten()
    return float(np.sqrt(np.mean((v1 - v2) ** 2)))

# ==============================================================================
# SPECTRAL LIBRARY CORE DATA STRUCTURES
# ==============================================================================

class SpectralLibraryEntry:
    """Stores statistical spectral metrics for a single endmember/class."""
    def __init__(
        self,
        name: str,
        wavelengths: np.ndarray,
        mean: np.ndarray,
        std: Optional[np.ndarray] = None,
        min_spec: Optional[np.ndarray] = None,
        max_spec: Optional[np.ndarray] = None,
        median: Optional[np.ndarray] = None,
        p25: Optional[np.ndarray] = None,
        p75: Optional[np.ndarray] = None,
        sample_count: int = 1,
        color: str = "#1f77b4",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.name = str(name)
        self.wavelengths = np.asarray(wavelengths, dtype=np.float64)
        self.mean = np.asarray(mean, dtype=np.float64)
        n_bands = len(self.mean)
        self.std = np.asarray(std, dtype=np.float64) if std is not None else np.zeros(n_bands)
        self.min_spec = np.asarray(min_spec, dtype=np.float64) if min_spec is not None else self.mean.copy()
        self.max_spec = np.asarray(max_spec, dtype=np.float64) if max_spec is not None else self.mean.copy()
        self.median = np.asarray(median, dtype=np.float64) if median is not None else self.mean.copy()
        self.p25 = np.asarray(p25, dtype=np.float64) if p25 is not None else (self.mean - 0.5 * self.std)
        self.p75 = np.asarray(p75, dtype=np.float64) if p75 is not None else (self.mean + 0.5 * self.std)
        self.sample_count = int(sample_count)
        self.color = color
        self.metadata = metadata or {}
        self.visible = True

    @classmethod
    def from_samples(cls, name: str, wavelengths: np.ndarray, samples: np.ndarray, color: str = "#1f77b4") -> "SpectralLibraryEntry":
        """Calculates statistical summary from a matrix of (N_samples, N_bands)."""
        samples = np.asarray(samples, dtype=np.float64)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        mean = np.mean(samples, axis=0)
        std = np.std(samples, axis=0) if samples.shape[0] > 1 else np.zeros(samples.shape[1])
        min_s = np.min(samples, axis=0)
        max_s = np.max(samples, axis=0)
        median = np.median(samples, axis=0)
        p25 = np.percentile(samples, 25, axis=0) if samples.shape[0] > 1 else min_s
        p75 = np.percentile(samples, 75, axis=0) if samples.shape[0] > 1 else max_s
        return cls(
            name=name,
            wavelengths=wavelengths,
            mean=mean,
            std=std,
            min_spec=min_s,
            max_spec=max_s,
            median=median,
            p25=p25,
            p75=p75,
            sample_count=samples.shape[0],
            color=color
        )


class SpectralLibrary:
    """Collection of spectral library entries with import/export capabilities."""
    def __init__(self, name: str = "Custom Spectral Library"):
        self.name = name
        self.entries: List[SpectralLibraryEntry] = []

    def add_entry(self, entry: SpectralLibraryEntry):
        self.entries.append(entry)

    def remove_entry(self, index: int):
        if 0 <= index < len(self.entries):
            del self.entries[index]

    def clear(self):
        self.entries.clear()

    def get_common_wavelengths(self) -> Optional[np.ndarray]:
        if not self.entries:
            return None
        return self.entries[0].wavelengths

    def get_entry(self, name_or_idx: Union[str, int]) -> Optional[SpectralLibraryEntry]:
        """Retrieve an entry by its name (exact, case-insensitive, or stripped) or index."""
        if isinstance(name_or_idx, int):
            if 0 <= name_or_idx < len(self.entries):
                return self.entries[name_or_idx]
            return None

        # Try exact match
        for entry in self.entries:
            if entry.name == name_or_idx:
                return entry

        # Try case-insensitive and stripped match
        target = str(name_or_idx).strip().lower()
        for entry in self.entries:
            if entry.name.strip().lower() == target:
                return entry

        # Try substring match
        for entry in self.entries:
            if target in entry.name.strip().lower() or entry.name.strip().lower() in target:
                return entry

        return None

    def get_entry_by_name(self, name: str) -> Optional[SpectralLibraryEntry]:
        """Convenience lookup by entry name."""
        return self.get_entry(name)

    def get_entries(self) -> List[SpectralLibraryEntry]:
        """Returns the list of all spectral library entries."""
        return list(self.entries)

    def __getitem__(self, item: Union[str, int]) -> SpectralLibraryEntry:
        entry = self.get_entry(item)
        if entry is None:
            raise KeyError(f"No entry found for '{item}' in SpectralLibrary '{self.name}'")
        return entry

    def __contains__(self, item: Union[str, int, SpectralLibraryEntry]) -> bool:
        if isinstance(item, SpectralLibraryEntry):
            return item in self.entries
        return self.get_entry(item) is not None

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    # --------------------------------------------------------------------------
    # ENVI SPECTRAL LIBRARY EXPORT (.sli + .hdr)
    # --------------------------------------------------------------------------
    def export_envi_sli(self, file_path: str):
        """Exports library in standard ENVI Spectral Library binary format."""
        if not self.entries:
            raise ValueError("Cannot export an empty spectral library.")

        base_path = Path(file_path).with_suffix("")
        sli_path = base_path.with_suffix(".sli")
        hdr_path = base_path.with_suffix(".hdr")

        ref_wv = self.entries[0].wavelengths
        n_bands = len(ref_wv)
        n_spectra = len(self.entries)

        matrix = np.zeros((n_spectra, n_bands), dtype=np.float32)
        spectra_names = []

        for i, entry in enumerate(self.entries):
            spectra_names.append(entry.name)
            if len(entry.mean) == n_bands:
                matrix[i, :] = entry.mean.astype(np.float32)
            else:
                matrix[i, :] = np.interp(ref_wv, entry.wavelengths, entry.mean).astype(np.float32)

        with open(sli_path, "wb") as f:
            matrix.tofile(f)

        wv_str = ", ".join(f"{w:.4f}" for w in ref_wv)
        names_str = ", ".join(f'"{name}"' for name in spectra_names)
        hdr_content = f"""ENVI
description = {{HyperVision Pro Spectral Library: {self.name}}}
samples = {n_bands}
lines = {n_spectra}
bands = 1
header offset = 0
file type = ENVI Spectral Library
data type = 4
interleave = bsq
sensor type = Unknown
byte order = 0
wavelength units = Nanometers
wavelength = {{
  {wv_str}
}}
spectra names = {{
  {names_str}
}}
"""
        with open(hdr_path, "w", encoding="utf-8") as f:
            f.write(hdr_content)

    # --------------------------------------------------------------------------
    # CSV MULTI-PROFILE EXPORT
    # --------------------------------------------------------------------------
    def export_csv(self, file_path: str):
        """Exports comprehensive CSV with Mean, Std, and Percentiles for all entries."""
        if not self.entries:
            raise ValueError("Library is empty.")

        ref_wv = self.entries[0].wavelengths
        headers = ["Wavelength_nm"]
        for entry in self.entries:
            clean_n = entry.name.replace(",", "_")
            headers.extend([
                f"{clean_n}_Mean",
                f"{clean_n}_Std",
                f"{clean_n}_Min",
                f"{clean_n}_Max",
                f"{clean_n}_Median"
            ])

        rows = [",".join(headers)]
        for i, w in enumerate(ref_wv):
            line = [f"{w:.2f}"]
            for entry in self.entries:
                if i < len(entry.mean):
                    line.extend([
                        f"{entry.mean[i]:.6f}",
                        f"{entry.std[i]:.6f}",
                        f"{entry.min_spec[i]:.6f}",
                        f"{entry.max_spec[i]:.6f}",
                        f"{entry.median[i]:.6f}"
                    ])
                else:
                    line.extend(["0.0", "0.0", "0.0", "0.0", "0.0"])
            rows.append(",".join(line))

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(rows) + "\n")

    # --------------------------------------------------------------------------
    # JSON EXPORT & IMPORT
    # --------------------------------------------------------------------------
    def export_json(self, file_path: str):
        data = {
            "library_name": self.name,
            "entries": []
        }
        for e in self.entries:
            data["entries"].append({
                "name": e.name,
                "wavelengths": e.wavelengths.tolist(),
                "mean": e.mean.tolist(),
                "std": e.std.tolist(),
                "min": e.min_spec.tolist(),
                "max": e.max_spec.tolist(),
                "median": e.median.tolist(),
                "p25": e.p25.tolist(),
                "p75": e.p75.tolist(),
                "sample_count": e.sample_count,
                "color": e.color
            })
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def import_json(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.name = data.get("library_name", "Imported Library")
        self.entries.clear()
        for d in data.get("entries", []):
            entry = SpectralLibraryEntry(
                name=d["name"],
                wavelengths=np.array(d["wavelengths"]),
                mean=np.array(d["mean"]),
                std=np.array(d["std"]),
                min_spec=np.array(d["min"]),
                max_spec=np.array(d["max"]),
                median=np.array(d["median"]),
                p25=np.array(d["p25"]),
                p75=np.array(d["p75"]),
                sample_count=d.get("sample_count", 1),
                color=d.get("color", "#1f77b4")
            )
            self.entries.append(entry)

    # --------------------------------------------------------------------------
    # SIMILARITY MATCHING AGAINST AN UNKNOWN SPECTRUM
    # --------------------------------------------------------------------------
    def match_spectrum(self, unknown_wv: np.ndarray, unknown_spec: np.ndarray) -> List[Dict[str, Any]]:
        """Compares an unknown spectrum against all library entries using SAM, SCM, SID, RMSE."""
        results = []
        unknown_spec = np.asarray(unknown_spec, dtype=np.float64).flatten()
        for entry in self.entries:
            if len(entry.wavelengths) == len(unknown_wv) and np.allclose(entry.wavelengths, unknown_wv, atol=1e-3):
                ref = entry.mean
            else:
                ref = np.interp(unknown_wv, entry.wavelengths, entry.mean)

            sam_deg = compute_sam_angle(unknown_spec, ref)
            scm_r = compute_scm_correlation(unknown_spec, ref)
            sid_val = compute_sid_divergence(unknown_spec, ref)
            rmse_val = compute_rmse(unknown_spec, ref)

            sam_score = max(0.0, 100.0 * (1.0 - (sam_deg / 90.0)))
            scm_score = max(0.0, 100.0 * ((scm_r + 1.0) / 2.0))
            overall_score = (0.6 * sam_score) + (0.4 * scm_score)

            results.append({
                "name": entry.name,
                "color": entry.color,
                "overall_score": overall_score,
                "sam_deg": sam_deg,
                "scm_r": scm_r,
                "sid": sid_val,
                "rmse": rmse_val,
                "sample_count": entry.sample_count
            })

        results.sort(key=lambda x: x["overall_score"], reverse=True)
        return results


# ==============================================================================
# SPECTRAL LIBRARY INTERACTIVE GUI DIALOG
# ==============================================================================

class SpectralLibraryDialog(QDialog):
    """Interactive Spectral Library curation, visualization, and matching studio."""
    def __init__(self, library: Optional[SpectralLibrary] = None, parent_viewer=None, parent=None, *args, **kwargs):
        super().__init__(parent)
        self.setWindowTitle("HyperVision Pro — Spectral Library Generator & Manager")
        self.resize(1250, 780)

        self.library = library or SpectralLibrary()
        self.parent_viewer = parent_viewer or parent
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22']

        self._build_ui()
        self.update_table()
        self.plot_library()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        header = QLabel(
            "<div style='background-color: #0f172a; color: white; padding: 10px 14px; border-radius: 6px;'>"
            "<table width='100%'><tr>"
            "<td><h3 style='margin: 0; color: #38bdf8;'>📚 Spectral Library Studio</h3>"
            "<span style='font-size: 11px; color: #94a3b8;'>Curate endmembers, inspect statistical confidence envelopes, export ENVI .sli/.hdr, and match unknown signatures.</span></td>"
            "</tr></table></div>"
        )
        main_layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        btn_box = QGroupBox("Library Management")
        btn_box_layout = QVBoxLayout(btn_box)

        row1 = QHBoxLayout()
        self.btn_add_rois = QPushButton("➕ Add Active ROIs")
        self.btn_add_rois.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 5px;")
        self.btn_add_rois.clicked.connect(self.add_from_active_rois)
        row1.addWidget(self.btn_add_rois)

        self.btn_import_csv = QPushButton("📂 Import CSV")
        self.btn_import_csv.clicked.connect(self.import_csv_data)
        row1.addWidget(self.btn_import_csv)
        btn_box_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btn_export_envi = QPushButton("💾 Export ENVI .sli")
        self.btn_export_envi.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 5px;")
        self.btn_export_envi.clicked.connect(self.export_envi)
        row2.addWidget(self.btn_export_envi)

        self.btn_export_csv = QPushButton("💾 Export CSV")
        self.btn_export_csv.clicked.connect(self.export_csv)
        row2.addWidget(self.btn_export_csv)

        self.btn_export_json = QPushButton("💾 JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        row2.addWidget(self.btn_export_json)
        btn_box_layout.addLayout(row2)

        self.btn_match = QPushButton("🎯 Match Selected / Unknown Spectrum...")
        self.btn_match.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 6px;")
        self.btn_match.clicked.connect(self.open_matcher_dialog)
        btn_box_layout.addWidget(self.btn_match)

        left_layout.addWidget(btn_box)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Show", "Color", "Signature / Class Name", "N (Samples)"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.cellChanged.connect(self.on_table_edit)
        left_layout.addWidget(self.table)

        table_btn_row = QHBoxLayout()
        btn_del = QPushButton("🗑️ Remove Selected")
        btn_del.clicked.connect(self.remove_selected_entry)
        table_btn_row.addWidget(btn_del)

        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_all)
        table_btn_row.addWidget(btn_clear)
        left_layout.addLayout(table_btn_row)

        splitter.addWidget(left_widget)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        vis_bar = QHBoxLayout()
        vis_bar.addWidget(QLabel("<b>Envelope:</b>"))
        self.cmb_envelope = QComboBox()
        self.cmb_envelope.addItems(["Shaded ±1 Std Dev (68%)", "Shaded ±2 Std Dev (95%)", "IQR (25th-75th Percentile)", "Min / Max Bounds", "Mean Curves Only"])
        self.cmb_envelope.currentIndexChanged.connect(self.plot_library)
        vis_bar.addWidget(self.cmb_envelope)

        vis_bar.addWidget(QLabel("<b>Transform:</b>"))
        self.cmb_transform = QComboBox()
        self.cmb_transform.addItems(["Reflectance / Radiance", "1st Derivative", "2nd Derivative", "Continuum Removed"])
        self.cmb_transform.currentIndexChanged.connect(self.plot_library)
        vis_bar.addWidget(self.cmb_transform)

        self.chk_peaks = QCheckBox("Annotate Major Peaks")
        self.chk_peaks.stateChanged.connect(self.plot_library)
        vis_bar.addWidget(self.chk_peaks)

        vis_bar.addStretch()
        right_layout.addLayout(vis_bar)

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.nav_toolbar)
        right_layout.addWidget(self.canvas)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setMaximumWidth(120)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)

    def update_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for i, entry in enumerate(self.library.entries):
            row = self.table.rowCount()
            self.table.insertRow(row)

            chk = QCheckBox()
            chk.setChecked(entry.visible)
            chk.stateChanged.connect(lambda state, idx=i: self.on_visibility_changed(idx, state))
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.addWidget(chk)
            chk_layout.setAlignment(Qt.AlignCenter)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(row, 0, chk_widget)

            btn_col = QPushButton()
            btn_col.setStyleSheet(f"background-color: {entry.color}; border: 1px solid #64748b; border-radius: 3px;")
            btn_col.clicked.connect(lambda checked, idx=i, btn=btn_col: self.choose_entry_color(idx, btn))
            self.table.setCellWidget(row, 1, btn_col)

            item_name = QTableWidgetItem(entry.name)
            self.table.setItem(row, 2, item_name)

            item_cnt = QTableWidgetItem(str(entry.sample_count))
            item_cnt.setFlags(item_cnt.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 3, item_cnt)

        self.table.blockSignals(False)

    def on_table_edit(self, row, col):
        if col == 2 and 0 <= row < len(self.library.entries):
            new_name = self.table.item(row, col).text().strip()
            if new_name:
                self.library.entries[row].name = new_name
                self.plot_library()

    def on_visibility_changed(self, idx, state):
        if 0 <= idx < len(self.library.entries):
            self.library.entries[idx].visible = (state == Qt.Checked)
            self.plot_library()

    def choose_entry_color(self, idx, btn):
        if 0 <= idx < len(self.library.entries):
            curr = QColor(self.library.entries[idx].color)
            col = QColorDialog.getColor(curr, self, "Select Signature Color")
            if col.isValid():
                new_hex = col.name()
                self.library.entries[idx].color = new_hex
                btn.setStyleSheet(f"background-color: {new_hex}; border: 1px solid #64748b;")
                self.plot_library()

    def remove_selected_entry(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.library.entries):
            self.library.remove_entry(row)
            self.update_table()
            self.plot_library()

    def clear_all(self):
        if QMessageBox.question(self, "Clear Library", "Remove all spectral library entries?") == QMessageBox.Yes:
            self.library.clear()
            self.update_table()
            self.plot_library()

    def add_from_active_rois(self):
        if not self.parent_viewer or not hasattr(self.parent_viewer, "image_windows"):
            QMessageBox.warning(self, "No Active Viewer", "Parent viewer with open datacubes was not found.")
            return

        window = self.parent_viewer.get_active_window()
        if not window or not window.selections:
            QMessageBox.warning(self, "No ROIs Found", "Please create at least one ROI or point selection in the active datacube window.")
            return

        by_class: Dict[str, List[np.ndarray]] = {}
        for sel in window.selections:
            cls_name = sel.get("class_label", "Default")
            prof = sel.get("profile")
            if prof is not None and len(prof) > 0:
                by_class.setdefault(cls_name, []).append(prof)

        wv = window.document.wavelengths
        added_count = 0
        for cls_name, samples_list in by_class.items():
            samples_arr = np.array(samples_list)
            color = self.default_colors[len(self.library.entries) % len(self.default_colors)]
            entry = SpectralLibraryEntry.from_samples(
                name=f"{cls_name} ({window.name})",
                wavelengths=wv,
                samples=samples_arr,
                color=color
            )
            self.library.add_entry(entry)
            added_count += 1

        self.update_table()
        self.plot_library()
        QMessageBox.information(self, "ROIs Imported", f"Successfully imported {added_count} class signature(s) into the Spectral Library.")

    def import_csv_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Spectral CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            class_col = next((c for c in df.columns if str(c).lower() in ["class_label", "class", "label", "target"]), None)
            numeric_cols = [c for c in df.columns if c != class_col]

            wvs = []
            valid_cols = []
            for c in numeric_cols:
                try:
                    wvs.append(float(str(c).replace("Wavelength_", "")))
                    valid_cols.append(c)
                except ValueError:
                    pass

            if not valid_cols:
                QMessageBox.critical(self, "Import Error", "Could not detect numeric wavelength columns in CSV.")
                return

            wvs = np.array(wvs)
            if class_col is not None:
                groups = df.groupby(class_col)
                for name, grp in groups:
                    data = grp[valid_cols].values
                    col = self.default_colors[len(self.library.entries) % len(self.default_colors)]
                    entry = SpectralLibraryEntry.from_samples(str(name), wvs, data, color=col)
                    self.library.add_entry(entry)
            else:
                data = df[valid_cols].values
                col = self.default_colors[len(self.library.entries) % len(self.default_colors)]
                entry = SpectralLibraryEntry.from_samples(Path(path).stem, wvs, data, color=col)
                self.library.add_entry(entry)

            self.update_table()
            self.plot_library()
            QMessageBox.information(self, "Import Success", "Successfully imported CSV data into Spectral Library.")
        except Exception as e:
            QMessageBox.critical(self, "Import Failed", f"Failed to parse CSV file:\n{e}")

    def export_envi(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export ENVI Spectral Library", "spectral_library.sli", "ENVI Spectral Library (*.sli)")
        if path:
            try:
                self.library.export_envi_sli(path)
                QMessageBox.information(self, "Export Complete", f"Saved ENVI Spectral Library:\n{path}\nand accompanying header (.hdr).")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not export ENVI library:\n{e}")

    def export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Spectral Library CSV", "spectral_library.csv", "CSV Files (*.csv)")
        if path:
            try:
                self.library.export_csv(path)
                QMessageBox.information(self, "Export Complete", f"Saved Spectral Library CSV to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not export CSV:\n{e}")

    def export_json(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Spectral Library JSON", "spectral_library.json", "JSON Files (*.json)")
        if path:
            try:
                self.library.export_json(path)
                QMessageBox.information(self, "Export Complete", f"Saved Spectral Library JSON to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Could not export JSON:\n{e}")

    def plot_library(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        visible_entries = [e for e in self.library.entries if e.visible]
        if not visible_entries:
            ax.text(0.5, 0.5, "No visible spectral signatures.\nAdd entries from ROIs or import CSV.", ha="center", va="center", color="#64748b", fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvas.draw()
            return

        envelope_mode = self.cmb_envelope.currentText()
        transform_mode = self.cmb_transform.currentText()
        annotate_peaks = self.chk_peaks.isChecked()

        for entry in visible_entries:
            wv = entry.wavelengths
            y_mean = entry.mean.copy()
            y_std = entry.std.copy()
            y_min = entry.min_spec.copy()
            y_max = entry.max_spec.copy()
            y_p25 = entry.p25.copy()
            y_p75 = entry.p75.copy()

            if transform_mode == "1st Derivative":
                from scipy.signal import savgol_filter
                w_len = min(11, len(y_mean) if len(y_mean) % 2 == 1 else len(y_mean) - 1)
                if w_len >= 5:
                    y_mean = savgol_filter(y_mean, window_length=w_len, polyorder=2, deriv=1)
                    y_p25 = savgol_filter(y_p25, window_length=w_len, polyorder=2, deriv=1)
                    y_p75 = savgol_filter(y_p75, window_length=w_len, polyorder=2, deriv=1)
            elif transform_mode == "2nd Derivative":
                from scipy.signal import savgol_filter
                w_len = min(11, len(y_mean) if len(y_mean) % 2 == 1 else len(y_mean) - 1)
                if w_len >= 5:
                    y_mean = savgol_filter(y_mean, window_length=w_len, polyorder=3, deriv=2)
                    y_p25 = savgol_filter(y_p25, window_length=w_len, polyorder=3, deriv=2)
                    y_p75 = savgol_filter(y_p75, window_length=w_len, polyorder=3, deriv=2)
            elif transform_mode == "Continuum Removed":
                from scipy.spatial import ConvexHull
                pts = np.column_stack((wv, y_mean))
                try:
                    hull = ConvexHull(pts)
                    top_verts = sorted([v for v in hull.vertices if pts[v, 1] >= np.median(y_mean)])
                    top_wv = wv[top_verts]
                    top_y = y_mean[top_verts]
                    continuum = np.interp(wv, top_wv, top_y)
                    y_mean = np.clip(y_mean / (continuum + 1e-8), 0.0, 1.0)
                except Exception:
                    pass

            label = f"{entry.name} (N={entry.sample_count})"
            ax.plot(wv, y_mean, label=label, color=entry.color, linewidth=2.2)

            if "±1 Std Dev" in envelope_mode and entry.sample_count > 1:
                lower = y_mean - y_std
                upper = y_mean + y_std
                ax.fill_between(wv, lower, upper, color=entry.color, alpha=0.22)
            elif "±2 Std Dev" in envelope_mode and entry.sample_count > 1:
                lower = y_mean - 2 * y_std
                upper = y_mean + 2 * y_std
                ax.fill_between(wv, lower, upper, color=entry.color, alpha=0.18)
            elif "IQR" in envelope_mode and entry.sample_count > 1:
                ax.fill_between(wv, y_p25, y_p75, color=entry.color, alpha=0.25)
            elif "Min / Max" in envelope_mode and entry.sample_count > 1:
                ax.fill_between(wv, y_min, y_max, color=entry.color, alpha=0.15, linestyle="--")

            if annotate_peaks and len(y_mean) > 10:
                from scipy.signal import find_peaks
                peaks, _ = find_peaks(y_mean, distance=15, prominence=0.02)
                for pk in peaks[:3]:
                    ax.plot(wv[pk], y_mean[pk], marker="v", color=entry.color, markersize=7)
                    ax.annotate(f"{wv[pk]:.0f}nm", (wv[pk], y_mean[pk]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=entry.color, fontweight="bold")

        ax.set_title(f"Spectral Library: {self.library.name} ({transform_mode})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Wavelength (nm)", fontsize=10, fontweight="bold")
        ax.set_ylabel(transform_mode if transform_mode != "Reflectance / Radiance" else "Intensity / Reflectance", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", framealpha=0.9, fontsize=9)
        self.figure.tight_layout()
        self.canvas.draw()

    def open_matcher_dialog(self):
        if not self.library.entries:
            QMessageBox.warning(self, "Empty Library", "Cannot match without signatures in the library.")
            return

        matcher = SpectralMatcherDialog(self.library, parent_viewer=self.parent_viewer, parent=self)
        matcher.exec_()


class SpectralMatcherDialog(QDialog):
    """Dialog to match an unknown spectrum against library signatures with Radar (Spider) and Overlay plots."""
    def __init__(self, library: SpectralLibrary, parent_viewer=None, parent=None, *args, **kwargs):
        super().__init__(parent)
        self.setWindowTitle("Spectral Similarity Matcher & Diagnostic Radar (SAM / SCM / SID)")
        self.resize(1150, 650)
        self.library = library
        self.parent_viewer = parent_viewer
        self.last_unknown_wv = None
        self.last_unknown_spec = None
        self.last_matches = []

        main_layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("<b>Input Spectrum Source:</b>"))
        self.cmb_source = QComboBox()
        self.cmb_source.addItems(["From Active Datacube ROI / Point", "From CSV File"])
        top_bar.addWidget(self.cmb_source)

        btn_run = QPushButton("⚡ Compute Best Matches")
        btn_run.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px;")
        btn_run.clicked.connect(self.run_matching)
        top_bar.addWidget(btn_run)

        btn_export = QPushButton("💾 Export Matches CSV")
        btn_export.clicked.connect(self.export_csv)
        top_bar.addWidget(btn_export)

        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        splitter = QSplitter(Qt.Horizontal)

        # Left panel: results table & summary
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        left_layout.addWidget(QLabel("<b>Ranked Candidate Signatures:</b> (Click row to inspect)"))
        self.table_results = QTableWidget(0, 6)
        self.table_results.setHorizontalHeaderLabels([
            "Rank", "Library Signature", "Match Score", "SAM Angle", "SCM (r)", "SID Divergence"
        ])
        self.table_results.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_results.itemSelectionChanged.connect(self._on_row_selected)
        left_layout.addWidget(self.table_results)

        self.lbl_verdict = QLabel("Ready to match. Select a source and click '⚡ Compute Best Matches'.")
        self.lbl_verdict.setStyleSheet("font-size: 11px; color: #475569; padding: 6px; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 4px;")
        left_layout.addWidget(self.lbl_verdict)

        splitter.addWidget(left_widget)

        # Right panel: Tabbed Visualizations (Radar Chart & Spectral Overlay)
        self.tab_vis = QTabWidget()

        # Tab 1: Radar (Spider) Chart
        radar_widget = QWidget()
        radar_layout = QVBoxLayout(radar_widget)
        radar_layout.setContentsMargins(4, 4, 4, 4)
        self.fig_radar = Figure(figsize=(5, 5), dpi=100)
        self.canvas_radar = FigureCanvas(self.fig_radar)
        radar_layout.addWidget(self.canvas_radar)
        self.tab_vis.addTab(radar_widget, "🕸️ Multi-Metric Radar (Spider) Chart")

        # Tab 2: Spectral Curve Overlay
        curve_widget = QWidget()
        curve_layout = QVBoxLayout(curve_widget)
        curve_layout.setContentsMargins(4, 4, 4, 4)
        self.fig_curve = Figure(figsize=(5, 5), dpi=100)
        self.canvas_curve = FigureCanvas(self.fig_curve)
        self.toolbar_curve = NavigationToolbar(self.canvas_curve, self)
        curve_layout.addWidget(self.toolbar_curve)
        curve_layout.addWidget(self.canvas_curve)
        self.tab_vis.addTab(curve_widget, "📈 Spectral Curve Overlay & Residuals")

        splitter.addWidget(self.tab_vis)
        splitter.setSizes([520, 600])
        main_layout.addWidget(splitter, 1)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)

        self._init_empty_plots()

    def _init_empty_plots(self):
        self.fig_radar.clear()
        ax = self.fig_radar.add_subplot(111, polar=True)
        ax.set_title("Multi-Metric Similarity Radar", fontsize=11, fontweight='bold', pad=15)
        ax.text(0, 0, "Run matching to generate radar profile", ha='center', va='center', color='#64748b', fontsize=10)
        self.canvas_radar.draw()

        self.fig_curve.clear()
        ax_c = self.fig_curve.add_subplot(111)
        ax_c.set_title("Spectral Profile Alignment", fontsize=11, fontweight='bold')
        ax_c.text(0.5, 0.5, "Unknown spectrum vs matched library signatures", ha='center', va='center', color='#64748b', transform=ax_c.transAxes)
        self.canvas_curve.draw()

    def run_matching(self):
        source = self.cmb_source.currentText()
        unknown_wv = None
        unknown_spec = None

        if "Active Datacube" in source:
            if not self.parent_viewer:
                QMessageBox.warning(self, "No Viewer", "Viewer instance not available.")
                return
            window = self.parent_viewer.get_active_window()
            if not window or not window.selections:
                QMessageBox.warning(self, "No Selection", "Please click a point or ROI on the active datacube to sample an unknown spectrum.")
                return
            unknown_wv = window.document.wavelengths
            unknown_spec = window.selections[-1]["profile"]
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select Unknown CSV", "", "CSV Files (*.csv)")
            if not path:
                return
            import pandas as pd
            df = pd.read_csv(path)
            num_cols = df.select_dtypes(include=[np.number]).columns
            wvs = []
            for c in num_cols:
                try:
                    wvs.append(float(str(c).replace("Wavelength_", "")))
                except ValueError:
                    pass
            if not wvs:
                QMessageBox.critical(self, "Format Error", "CSV does not contain numeric wavelength headers.")
                return
            unknown_wv = np.array(wvs)
            unknown_spec = df[num_cols].iloc[0].values

        self.last_unknown_wv = np.asarray(unknown_wv, dtype=np.float64)
        self.last_unknown_spec = np.asarray(unknown_spec, dtype=np.float64).flatten()
        self.last_matches = self.library.match_spectrum(self.last_unknown_wv, self.last_unknown_spec)

        self.table_results.setRowCount(0)
        for i, m in enumerate(self.last_matches):
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)

            self.table_results.setItem(row, 0, QTableWidgetItem(f"#{i+1}"))
            self.table_results.setItem(row, 1, QTableWidgetItem(m["name"]))

            score_item = QTableWidgetItem(f"{m['overall_score']:.1f} %")
            if i == 0:
                score_item.setForeground(QColor("#059669"))
                score_item.setBackground(QColor("#dcfce7"))
            self.table_results.setItem(row, 2, score_item)

            self.table_results.setItem(row, 3, QTableWidgetItem(f"{m['sam_deg']:.2f}°"))
            self.table_results.setItem(row, 4, QTableWidgetItem(f"{m['scm_r']:.4f}"))
            self.table_results.setItem(row, 5, QTableWidgetItem(f"{m['sid']:.4f}"))

        if self.last_matches:
            best = self.last_matches[0]
            self.lbl_verdict.setText(
                f"<b>Best Match:</b> <span style='color: #059669;'>{best['name']}</span> "
                f"with <b>{best['overall_score']:.1f}% confidence</b> (SAM: {best['sam_deg']:.2f}°, SCM r: {best['scm_r']:.3f})."
            )
            self.table_results.selectRow(0)
            self._render_radar_and_overlay(0)

    def _on_row_selected(self):
        rows = self.table_results.selectionModel().selectedRows()
        if rows:
            self._render_radar_and_overlay(rows[0].row())

    def _render_radar_and_overlay(self, row_idx: int):
        if not self.last_matches or row_idx >= len(self.last_matches):
            return

        m = self.last_matches[row_idx]
        cand_name = m["name"]
        color = m.get("color", "#2563eb")

        # 1. Render Multi-Metric Radar (Spider) Chart
        self.fig_radar.clear()
        ax_r = self.fig_radar.add_subplot(111, polar=True)

        categories = [
            "SAM Proximity",
            "SCM Correlation",
            "Cosine Similarity",
            "SID Inverted",
            "RMSE Proximity",
            "Overall Confidence"
        ]
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        # Calculate normalized metrics (0 - 100%)
        sam_prox = max(0.0, 100.0 * (1.0 - (m["sam_deg"] / 90.0)))
        scm_prox = max(0.0, 100.0 * ((m["scm_r"] + 1.0) / 2.0))
        cos_sim = max(0.0, 100.0 * np.cos(np.radians(m["sam_deg"])))
        sid_prox = max(0.0, 100.0 * np.exp(-max(0.0, m["sid"])))
        rmse_prox = max(0.0, 100.0 * np.exp(-3.0 * max(0.0, m.get("rmse", 0.1))))
        score = float(m["overall_score"])

        values = [sam_prox, scm_prox, cos_sim, sid_prox, rmse_prox, score]
        values += values[:1]

        ax_r.set_theta_offset(np.pi / 2)
        ax_r.set_theta_direction(-1)
        ax_r.set_xticks(angles[:-1])
        ax_r.set_xticklabels(categories, fontsize=9, fontweight='bold')
        ax_r.set_ylim(0, 100)
        ax_r.set_yticks([20, 40, 60, 80, 100])
        ax_r.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], fontsize=8, color="#64748b")
        ax_r.grid(True, linestyle="--", alpha=0.6, color="#cbd5e1")

        # Plot candidate polygon
        ax_r.plot(angles, values, color=color, linewidth=2.2, label=f"Rank #{row_idx+1}: {cand_name}")
        ax_r.fill(angles, values, color=color, alpha=0.28)

        # If best match is different from currently selected row, also plot Rank #1 for benchmark comparison
        if row_idx != 0 and len(self.last_matches) > 0:
            top_m = self.last_matches[0]
            top_vals = [
                max(0.0, 100.0 * (1.0 - (top_m["sam_deg"] / 90.0))),
                max(0.0, 100.0 * ((top_m["scm_r"] + 1.0) / 2.0)),
                max(0.0, 100.0 * np.cos(np.radians(top_m["sam_deg"]))),
                max(0.0, 100.0 * np.exp(-max(0.0, top_m["sid"]))),
                max(0.0, 100.0 * np.exp(-3.0 * max(0.0, top_m.get("rmse", 0.1)))),
                float(top_m["overall_score"])
            ]
            top_vals += top_vals[:1]
            ax_r.plot(angles, top_vals, color="#059669", linewidth=1.5, linestyle="--", label=f"Rank #1: {top_m['name']}")

        ax_r.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)
        ax_r.set_title(f"Similarity Diagnostic Radar — {cand_name} ({score:.1f}%)", fontsize=11, fontweight='bold', pad=15)
        self.canvas_radar.draw()

        # 2. Render Spectral Curve Overlay & Residuals
        self.fig_curve.clear()
        gs = self.fig_curve.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.25)
        ax_top = self.fig_curve.add_subplot(gs[0])
        ax_bot = self.fig_curve.add_subplot(gs[1], sharex=ax_top)

        # Find entry in library
        entry = next((e for e in self.library.entries if e.name == cand_name), None)
        if entry is not None and self.last_unknown_wv is not None:
            if len(entry.wavelengths) == len(self.last_unknown_wv) and np.allclose(entry.wavelengths, self.last_unknown_wv, atol=1e-3):
                ref_spec = entry.mean
            else:
                ref_spec = np.interp(self.last_unknown_wv, entry.wavelengths, entry.mean)

            ax_top.plot(self.last_unknown_wv, self.last_unknown_spec, 'k-', linewidth=2.0, label="Unknown Probe Spectrum")
            ax_top.plot(self.last_unknown_wv, ref_spec, color=color, linewidth=1.8, linestyle="--", label=f"Library: {cand_name}")

            # If entry has std deviation, plot confidence envelope
            if entry.std is not None:
                std_interp = np.interp(self.last_unknown_wv, entry.wavelengths, entry.std) if len(entry.wavelengths) != len(self.last_unknown_wv) else entry.std
                ax_top.fill_between(self.last_unknown_wv, ref_spec - std_interp, ref_spec + std_interp, color=color, alpha=0.18, label="±1σ Envelope")

            ax_top.set_ylabel("Reflectance / Intensity", fontsize=9)
            ax_top.set_title(f"Spectral Comparison: Unknown vs {cand_name}", fontsize=11, fontweight='bold')
            ax_top.legend(loc="upper right", fontsize=8)
            ax_top.grid(True, linestyle=":", alpha=0.6)

            # Residual curve
            residual = self.last_unknown_spec - ref_spec
            ax_bot.plot(self.last_unknown_wv, residual, color="#dc2626", linewidth=1.2, label="Residual (Probe - Ref)")
            ax_bot.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax_bot.fill_between(self.last_unknown_wv, residual, 0, color="#dc2626", alpha=0.2)
            ax_bot.set_xlabel("Wavelength (nm)", fontsize=9)
            ax_bot.set_ylabel("Residual", fontsize=8)
            ax_bot.grid(True, linestyle=":", alpha=0.6)
            ax_bot.legend(loc="upper right", fontsize=8)

        try:
            self.fig_curve.subplots_adjust(top=0.92, bottom=0.12, left=0.12, right=0.95, hspace=0.35)
        except Exception:
            pass
        self.canvas_curve.draw()

    def export_csv(self):
        if not self.last_matches:
            QMessageBox.warning(self, "No Matches", "No match results available to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Matches CSV", "spectral_matches.csv", "CSV Files (*.csv)")
        if not path:
            return
        import csv
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Rank", "Library Signature", "Overall Match Score (%)", "SAM Angle (deg)", "SCM Correlation (r)", "SID Divergence", "RMSE"])
            for i, m in enumerate(self.last_matches):
                writer.writerow([i+1, m["name"], f"{m['overall_score']:.2f}", f"{m['sam_deg']:.3f}", f"{m['scm_r']:.4f}", f"{m['sid']:.4f}", f"{m.get('rmse', 0.0):.4f}"])
        QMessageBox.information(self, "Export Successful", f"Saved {len(self.last_matches)} spectral matches to {os.path.basename(path)}.")
