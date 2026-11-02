"""SpectralView 2.1.12 stable-candidate GUI.

Scientific status: stable candidate, not externally validated. This interface
uses spectral_core.py for GUI and batch calculations so the scientific formulas
cannot silently diverge.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from PyQt5.QtCore import QObject, QRunnable, QRectF, Qt, QThreadPool, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMdiArea,
    QMdiSubWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from matplotlib import colormaps
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from spectral_core import (
    INDEX_DEFINITIONS,
    RELEASE_STATUS,
    VERSION,
    CalibrationConfig,
    EnviDocument,
    InputContractError,
    ProcessingConfig,
    SAMResult,
    SpectralViewError,
    WavelengthCompatibilityError,
    atomic_write_csv,
    atomic_write_text,
    compose_rgb,
    compute_spectral_index,
    discover_envi_folder,
    find_dataset_folders,
    nearest_band,
    pca_rgb,
    preprocess_spectra,
    resample_spectrum,
    robust_uint8,
    run_batch,
    segment_objects,
)

APP_TITLE = f"SpectralView {VERSION} — Scientific Stable Candidate"


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(object)
    result = pyqtSignal(object)
    progress = pyqtSignal(object)


class FunctionWorker(QRunnable):
    """Run a callable that accepts one progress-emitter argument."""

    def __init__(self, function):
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            result = self.function(self.signals.progress.emit)
        except Exception as exc:  # terminal worker error surfaced to the GUI
            self.signals.error.emit((exc, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.reset_labels("Spectral profiles", "Wavelength (nm)", "Value")

    def reset_labels(self, title: str, xlabel: str, ylabel: str) -> None:
        self.axes.set_title(title)
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        self.fig.tight_layout()


class CubeGraphicsView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, cube_window: "DatacubeWindow"):
        super().__init__(scene)
        self.cube_window = cube_window
        self.mode = "Pan"
        self.roi_start = None
        self.current_rect_item = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.setDragMode(QGraphicsView.ScrollHandDrag if mode == "Pan" else QGraphicsView.NoDrag)

    def mousePressEvent(self, event):
        if self.mode == "Pan":
            super().mousePressEvent(event)
            return
        if event.button() != Qt.LeftButton:
            return
        if self.mode == "Point":
            self.cube_window.handle_point_selection(self.mapToScene(event.pos()))
        elif self.mode == "ROI":
            self.roi_start = self.mapToScene(event.pos())
            self.current_rect_item = QGraphicsRectItem()
            self.current_rect_item.setPen(self.cube_window.scaled_pen(QColor(0, 255, 0)))
            self.scene().addItem(self.current_rect_item)

    def mouseMoveEvent(self, event):
        if self.mode == "Pan":
            super().mouseMoveEvent(event)
        elif self.mode == "ROI" and self.roi_start is not None and self.current_rect_item is not None:
            rect = QRectF(self.roi_start, self.mapToScene(event.pos())).normalized()
            self.current_rect_item.setRect(rect)

    def mouseReleaseEvent(self, event):
        if self.mode == "Pan":
            super().mouseReleaseEvent(event)
            return
        if (
            self.mode == "ROI"
            and self.roi_start is not None
            and self.current_rect_item is not None
            and event.button() == Qt.LeftButton
        ):
            rect = QRectF(self.roi_start, self.mapToScene(event.pos())).normalized()
            self.cube_window.handle_roi_selection(rect, self.current_rect_item)
            self.roi_start = None
            self.current_rect_item = None


class DatacubeWindow(QMdiSubWindow):
    def __init__(self, main_app: "HyperspectralViewer", document: EnviDocument, name: str):
        super().__init__()
        self.main_app = main_app
        self.document = document
        self.name = name
        self.setWindowTitle(name)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        self.scale_factor = 1.0
        self.preview_cube: Optional[np.ndarray] = None
        self.preview_scale_x = 1.0
        self.preview_scale_y = 1.0
        self.preview_qc = None
        self.render_cache: Dict[Tuple[Any, ...], np.ndarray] = {}
        self.selections: List[Dict[str, Any]] = []
        self.last_classification: Optional[SAMResult] = None
        self.last_class_names: List[str] = []
        self.last_overlay_rgba: Optional[np.ndarray] = None

        self.scene = QGraphicsScene()
        self.view = CubeGraphicsView(self.scene, self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.class_overlay_item = QGraphicsPixmapItem()
        self.class_overlay_item.setOpacity(0.65)
        self.scene.addItem(self.class_overlay_item)

        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.addWidget(self.view)

        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas(plot_widget)
        self.nav_toolbar = NavigationToolbar(self.canvas, plot_widget)
        plot_layout.addWidget(self.nav_toolbar)
        plot_layout.addWidget(self.canvas)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(image_widget)
        splitter.addWidget(plot_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.setWidget(splitter)

        self.load_preview()

    def load_preview(self) -> None:
        self.main_app.log(f"Loading calibrated preview: {self.name}")
        QApplication.processEvents()
        preview, qc, scale_x, scale_y = self.document.calibrated_preview(max_dimension=1200)
        self.preview_cube = preview
        self.preview_qc = qc
        self.preview_scale_x = scale_x
        self.preview_scale_y = scale_y
        self.render_cache.clear()
        self.update_image_view()
        self.reset_zoom()
        self.main_app.log(
            f"{self.name}: {qc.mode}; units={qc.units}; finite={qc.finite_fraction:.4f}; "
            f"preview={preview.shape[1]}×{preview.shape[0]}"
        )
        for warning in self.document.metadata_warnings + qc.warnings:
            self.main_app.log(f"WARNING {self.name}: {warning}")

    def scaled_pen(self, color: QColor, width: float = 2.0) -> QPen:
        return QPen(color, width / max(self.scale_factor, 1e-8))

    def scene_to_raw(self, x: float, y: float) -> Tuple[int, int]:
        raw_x = int(np.clip(round(x * self.preview_scale_x), 0, self.document.width - 1))
        raw_y = int(np.clip(round(y * self.preview_scale_y), 0, self.document.height - 1))
        return raw_x, raw_y

    def raw_to_scene(self, x: float, y: float) -> Tuple[float, float]:
        return x / self.preview_scale_x, y / self.preview_scale_y

    def _render_key(self) -> Tuple[Any, ...]:
        app = self.main_app
        return (
            app.vis_mode.currentText(),
            app.spin_r_nm.value(),
            app.spin_g_nm.value(),
            app.spin_b_nm.value(),
            app.spin_single_nm.value(),
            app.index_combo.currentText(),
            app.low_percentile.value(),
            app.high_percentile.value(),
            app.gamma_spin.value(),
        )

    def update_image_view(self) -> None:
        if self.preview_cube is None:
            return
        app = self.main_app
        mode = app.vis_mode.currentText()
        key = self._render_key()
        try:
            if key in self.render_cache:
                display = self.render_cache[key]
            elif mode == "PCA composite":
                display = pca_rgb(
                    self.preview_cube,
                    low_percentile=app.low_percentile.value(),
                    high_percentile=app.high_percentile.value(),
                )
                self.render_cache[key] = display
            elif mode == "True colour (wavelength-based)":
                if not self.document.has_physical_wavelengths:
                    raise WavelengthCompatibilityError("True colour requires physical wavelength metadata.")
                display, indices = compose_rgb(
                    self.preview_cube,
                    self.document.wavelengths,
                    (650.0, 550.0, 450.0),
                    app.low_percentile.value(),
                    app.high_percentile.value(),
                    app.gamma_spin.value(),
                    tolerance_nm=40.0,
                )
                actual = [self.document.wavelengths[i] for i in indices]
                app.statusBar().showMessage(
                    f"{self.name}: true-colour bands R/G/B = {actual[0]:.1f}/{actual[1]:.1f}/{actual[2]:.1f} nm"
                )
                self.render_cache[key] = display
            elif mode == "False colour (wavelength-based)":
                if not self.document.has_physical_wavelengths:
                    raise WavelengthCompatibilityError("Wavelength-based false colour requires wavelength metadata.")
                display, indices = compose_rgb(
                    self.preview_cube,
                    self.document.wavelengths,
                    (app.spin_r_nm.value(), app.spin_g_nm.value(), app.spin_b_nm.value()),
                    app.low_percentile.value(),
                    app.high_percentile.value(),
                    app.gamma_spin.value(),
                    tolerance_nm=30.0,
                )
                actual = [self.document.wavelengths[i] for i in indices]
                app.statusBar().showMessage(
                    f"{self.name}: false-colour bands = {actual[0]:.1f}/{actual[1]:.1f}/{actual[2]:.1f} nm"
                )
                self.render_cache[key] = display
            elif mode == "Single-band heatmap":
                idx = nearest_band(self.document.wavelengths, app.spin_single_nm.value(), 30.0)
                channel = self.preview_cube[:, :, idx]
                normalized = robust_uint8(
                    channel,
                    app.low_percentile.value(),
                    app.high_percentile.value(),
                    app.gamma_spin.value(),
                ).astype(np.float32) / 255.0
                display = (colormaps["viridis"](normalized)[:, :, :3] * 255).astype(np.uint8)
                app.statusBar().showMessage(
                    f"{self.name}: band {idx}, {self.document.wavelengths[idx]:.2f} "
                    f"{'nm' if self.document.has_physical_wavelengths else '(band index)'}"
                )
                self.render_cache[key] = display
            elif mode == "Spectral index":
                if not self.document.has_physical_wavelengths:
                    raise WavelengthCompatibilityError("Spectral indices require physical wavelength metadata.")
                index_map, metadata = compute_spectral_index(
                    self.preview_cube,
                    self.document.wavelengths,
                    app.index_combo.currentText(),
                    tolerance_nm=app.index_tolerance.value(),
                )
                normalized = robust_uint8(
                    index_map,
                    app.low_percentile.value(),
                    app.high_percentile.value(),
                    app.gamma_spin.value(),
                ).astype(np.float32) / 255.0
                display = (colormaps["viridis"](normalized)[:, :, :3] * 255).astype(np.uint8)
                app.statusBar().showMessage(
                    f"{self.name}: {metadata['index']} actual wavelengths "
                    + ", ".join(f"{x:.1f}" for x in metadata["actual_wavelengths_nm"])
                    + " nm"
                )
                self.render_cache[key] = display
            else:
                raise SpectralViewError(f"Unknown visualisation mode: {mode}")

            h, w = display.shape[:2]
            qimage = QImage(display.tobytes(), w, h, 3 * w, QImage.Format_RGB888).copy()
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
            self.scene.setSceneRect(0, 0, w, h)
        except Exception as exc:
            self.main_app.log(f"DISPLAY ERROR {self.name}: {exc}")
            self.main_app.statusBar().showMessage(str(exc), 8000)

    def zoom_in(self) -> None:
        factor = 1.25
        self.view.scale(factor, factor)
        self.scale_factor *= factor
        self.update_marker_scales()

    def zoom_out(self) -> None:
        factor = 0.8
        self.view.scale(factor, factor)
        self.scale_factor *= factor
        self.update_marker_scales()

    def reset_zoom(self) -> None:
        self.view.resetTransform()
        if not self.pixmap_item.pixmap().isNull():
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self.scale_factor = max(float(self.view.transform().m11()), 1e-8)
        self.update_marker_scales()

    def update_marker_scales(self) -> None:
        for selection in self.selections:
            marker = selection.get("marker")
            if isinstance(marker, QGraphicsEllipseItem):
                marker.setPen(self.scaled_pen(QColor(255, 0, 0)))
            elif isinstance(marker, QGraphicsRectItem):
                marker.setPen(self.scaled_pen(QColor(0, 255, 0)))
            text_item = selection.get("text_item")
            if text_item:
                text_item.setScale(1.0 / max(self.scale_factor, 1e-8))

    def _next_label(self) -> str:
        supplied = self.main_app.class_label.text().strip()
        return supplied if supplied else f"Reference_{len(self.selections) + 1}"

    def _add_selection(
        self,
        profile: np.ndarray,
        label: str,
        marker,
        text_item,
        selection_type: str,
        raw_x: Optional[int] = None,
        raw_y: Optional[int] = None,
        area_px: Optional[int] = None,
    ) -> None:
        selection_id = len(self.selections) + 1
        self.selections.append(
            {
                "id": selection_id,
                "class_label": label,
                "profile": np.asarray(profile, dtype=np.float32).copy(),
                "wavelengths": self.document.wavelengths.copy(),
                "has_physical_wavelengths": self.document.has_physical_wavelengths,
                "marker": marker,
                "text_item": text_item,
                "selection_type": selection_type,
                "raw_x": raw_x,
                "raw_y": raw_y,
                "area_px": area_px,
                "units": self.preview_qc.units if self.preview_qc else "unknown",
                "calibration_mode": self.preview_qc.mode if self.preview_qc else "unknown",
            }
        )

    def handle_point_selection(self, pos) -> None:
        if self.preview_cube is None:
            return
        if not (0 <= pos.x() < self.preview_cube.shape[1] and 0 <= pos.y() < self.preview_cube.shape[0]):
            return
        raw_x, raw_y = self.scene_to_raw(pos.x(), pos.y())
        try:
            profile, _qc = self.document.calibrated_spectrum(raw_y, raw_x)
        except Exception as exc:
            QMessageBox.critical(self.main_app, "Point extraction failed", str(exc))
            return
        radius = 4.0 / max(self.scale_factor, 1e-8)
        marker = QGraphicsEllipseItem(pos.x() - radius, pos.y() - radius, radius * 2, radius * 2)
        marker.setPen(self.scaled_pen(QColor(255, 0, 0)))
        self.scene.addItem(marker)
        label = self._next_label()
        text_item = QGraphicsTextItem(str(len(self.selections) + 1))
        text_item.setDefaultTextColor(QColor(255, 255, 0))
        text_item.setPos(pos.x() + radius, pos.y() - radius * 3)
        text_item.setScale(1.0 / max(self.scale_factor, 1e-8))
        self.scene.addItem(text_item)
        self._add_selection(profile, label, marker, text_item, "point", raw_x, raw_y)
        self.update_plot()
        self.main_app.log(f"{self.name}: added point reference {label} at x={raw_x}, y={raw_y}")

    def handle_roi_selection(self, rect: QRectF, temporary_item: QGraphicsRectItem) -> None:
        if self.preview_cube is None:
            self.scene.removeItem(temporary_item)
            return
        raw_x1, raw_y1 = self.scene_to_raw(rect.left(), rect.top())
        raw_x2, raw_y2 = self.scene_to_raw(rect.right(), rect.bottom())
        x1, x2 = sorted((raw_x1, raw_x2))
        y1, y2 = sorted((raw_y1, raw_y2))
        x2 = min(self.document.width, x2 + 1)
        y2 = min(self.document.height, y2 + 1)
        self.scene.removeItem(temporary_item)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return
        if (x2 - x1) * (y2 - y1) > 20_000_000:
            QMessageBox.warning(
                self.main_app,
                "ROI too large",
                "The selected ROI exceeds 20 million pixels. Select a smaller region to prevent excessive memory use.",
            )
            return
        try:
            region, _qc = self.document.calibrated_region(y1, y2, x1, x2)
            result = segment_objects(
                region,
                min_area_px=self.main_app.min_area_spin.value(),
                polarity=self.main_app.polarity_combo.currentText().lower(),
                morphology_size=self.main_app.morphology_spin.value(),
            )
        except Exception as exc:
            QMessageBox.critical(self.main_app, "ROI segmentation failed", str(exc))
            return

        if not result.objects:
            answer = QMessageBox.question(
                self.main_app,
                "No segmented objects",
                "No object passed the segmentation and area gates. Add the explicitly selected whole-ROI mean instead?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                profile = np.nanmean(region, axis=(0, 1)).astype(np.float32)
                scene_x1, scene_y1 = self.raw_to_scene(x1, y1)
                scene_x2, scene_y2 = self.raw_to_scene(x2, y2)
                marker = QGraphicsRectItem(QRectF(scene_x1, scene_y1, scene_x2 - scene_x1, scene_y2 - scene_y1))
                marker.setPen(self.scaled_pen(QColor(0, 255, 0)))
                self.scene.addItem(marker)
                text_item = self._make_text_item(len(self.selections) + 1, scene_x1, scene_y1)
                self._add_selection(
                    profile,
                    self._next_label(),
                    marker,
                    text_item,
                    "explicit_whole_roi_mean",
                    x1,
                    y1,
                    (x2 - x1) * (y2 - y1),
                )
                self.update_plot()
            return

        base_label = self._next_label()
        for object_index, obj in enumerate(result.objects, start=1):
            scene_x1, scene_y1 = self.raw_to_scene(x1 + obj.x, y1 + obj.y)
            scene_x2, scene_y2 = self.raw_to_scene(x1 + obj.x + obj.width, y1 + obj.y + obj.height)
            marker = QGraphicsRectItem(QRectF(scene_x1, scene_y1, scene_x2 - scene_x1, scene_y2 - scene_y1))
            marker.setPen(self.scaled_pen(QColor(0, 255, 0)))
            self.scene.addItem(marker)
            cx, cy = self.raw_to_scene(x1 + obj.centroid_x, y1 + obj.centroid_y)
            text_item = self._make_text_item(len(self.selections) + 1, cx, cy)
            label = base_label
            self._add_selection(
                obj.mean_spectrum,
                label,
                marker,
                text_item,
                "segmented_object",
                int(round(x1 + obj.centroid_x)),
                int(round(y1 + obj.centroid_y)),
                obj.area_px,
            )
        self.update_plot()
        self.main_app.log(
            f"{self.name}: ROI accepted {len(result.objects)} objects; foreground={result.foreground_fraction:.4f}; "
            f"border={result.border_fraction:.4f}; polarity={result.polarity}"
        )
        for warning in result.warnings:
            self.main_app.log(f"ROI WARNING {self.name}: {warning}")

    def _make_text_item(self, number: int, x: float, y: float):
        text_item = QGraphicsTextItem(str(number))
        text_item.setDefaultTextColor(QColor(255, 255, 0))
        text_item.setPos(x, y)
        text_item.setScale(1.0 / max(self.scale_factor, 1e-8))
        self.scene.addItem(text_item)
        return text_item

    def update_plot(self) -> None:
        self.canvas.axes.clear()
        app = self.main_app
        method = app.preprocessing_combo.currentData()
        for selection in self.selections:
            try:
                if method == "absorbance" and selection["units"] != "reflectance":
                    raise SpectralViewError("Absorbance preprocessing requires reflectance input.")
                processed = preprocess_spectra(
                    selection["profile"],
                    method,
                    selection["wavelengths"],
                    app.sg_window.value(),
                    app.sg_polyorder.value(),
                )
            except Exception as exc:
                app.log(f"PLOT PREPROCESSING ERROR: {exc}")
                processed = selection["profile"]
            label = f"{selection['id']}: {selection['class_label']}"
            self.canvas.axes.plot(selection["wavelengths"], processed, label=label, linewidth=1.3)
        xlabel = "Wavelength (nm)" if self.document.has_physical_wavelengths else "Band index"
        ylabel = f"{self.preview_qc.units if self.preview_qc else 'value'} [{method}]"
        self.canvas.axes.set_title(f"{self.name} — extracted spectra")
        self.canvas.axes.set_xlabel(xlabel)
        self.canvas.axes.set_ylabel(ylabel)
        self.canvas.axes.grid(True, linestyle="--", alpha=0.4)
        if 0 < len(self.selections) <= 20:
            self.canvas.axes.legend(loc="best", fontsize="x-small", framealpha=0.6)
        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

    def clear_selections(self) -> None:
        for selection in self.selections:
            marker = selection.get("marker")
            text_item = selection.get("text_item")
            if marker and marker.scene() is self.scene:
                self.scene.removeItem(marker)
            if text_item and text_item.scene() is self.scene:
                self.scene.removeItem(text_item)
        self.selections.clear()
        self.class_overlay_item.setPixmap(QPixmap())
        self.last_classification = None
        self.last_class_names = []
        self.last_overlay_rgba = None
        self.update_plot()

    def set_classification_overlay(self, result: SAMResult, class_names: List[str]) -> None:
        if self.preview_cube is None:
            return
        preview_h, preview_w = self.preview_cube.shape[:2]
        reduced = cv2.resize(result.class_map.astype(np.int32), (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
        rgba = np.zeros((preview_h, preview_w, 4), dtype=np.uint8)
        palette = colormaps["tab20"]
        for class_index in range(len(class_names)):
            mask = reduced == class_index
            rgb = (np.asarray(palette(class_index % 20)[:3]) * 255).astype(np.uint8)
            rgba[mask, :3] = rgb
            rgba[mask, 3] = 180
        qimage = QImage(rgba.tobytes(), preview_w, preview_h, 4 * preview_w, QImage.Format_RGBA8888).copy()
        self.class_overlay_item.setPixmap(QPixmap.fromImage(qimage))
        self.last_classification = result
        self.last_class_names = list(class_names)
        self.last_overlay_rgba = rgba.copy()

    def closeEvent(self, event) -> None:
        self.main_app.unregister_window(self)
        super().closeEvent(event)


class HyperspectralViewer(QMainWindow):
    def __init__(self, initial_folder: Optional[str] = None, initial_header: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1680, 980)
        self.image_windows: List[DatacubeWindow] = []
        self.thread_pool = QThreadPool.globalInstance()
        self._busy_tasks = 0

        self.mdi = QMdiArea()
        self.mdi.setBackground(QColor(55, 55, 55))
        self.setCentralWidget(self.mdi)
        self.mdi.subWindowActivated.connect(lambda _window: self.update_active_metadata())

        self._build_actions()
        self._build_controls()
        self.statusBar().showMessage(f"{APP_TITLE} | {RELEASE_STATUS}")
        self.log(f"Started {APP_TITLE}")
        self.log(f"Release status: {RELEASE_STATUS}")

        if initial_folder:
            self.load_folder(initial_folder)
        elif initial_header:
            self.load_header(initial_header)

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_folder = QAction("Open dataset folder...", self)
        open_folder.triggered.connect(self.choose_folder)
        file_menu.addAction(open_folder)
        open_header = QAction("Open ENVI data header...", self)
        open_header.triggered.connect(self.choose_header)
        file_menu.addAction(open_header)
        open_parent = QAction("Open all governed datasets under parent...", self)
        open_parent.triggered.connect(self.choose_parent)
        file_menu.addAction(open_parent)
        file_menu.addSeparator()
        export_profiles = QAction("Export extracted spectra (long CSV)...", self)
        export_profiles.triggered.connect(self.export_profiles)
        file_menu.addAction(export_profiles)
        export_profiles_wide = QAction("Export extracted spectra (wide CSV)...", self)
        export_profiles_wide.triggered.connect(self.export_profiles_wide)
        file_menu.addAction(export_profiles_wide)
        export_roi_1000 = QAction("Export ROI and Wavelengths <= 1000 nm (CSV)...", self)
        export_roi_1000.triggered.connect(self.export_roi_1000)
        file_menu.addAction(export_roi_1000)
        export_plot = QAction("Export active plot...", self)
        export_plot.triggered.connect(self.export_plot)
        file_menu.addAction(export_plot)
        export_class = QAction("Export active SAM result...", self)
        export_class.triggered.connect(self.export_classification)
        file_menu.addAction(export_class)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        batch_menu = self.menuBar().addMenu("Batch")
        run_batch_action = QAction("Run governed batch...", self)
        run_batch_action.triggered.connect(self.batch_process_folders)
        batch_menu.addAction(run_batch_action)

        window_menu = self.menuBar().addMenu("Window")
        tile_action = QAction("Tile", self)
        tile_action.triggered.connect(self.mdi.tileSubWindows)
        window_menu.addAction(tile_action)
        cascade_action = QAction("Cascade", self)
        cascade_action.triggered.connect(self.mdi.cascadeSubWindows)
        window_menu.addAction(cascade_action)

        toolbar = QToolBar("Interaction", self)
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        self.action_pan = QAction("Pan", self, checkable=True)
        self.action_point = QAction("Point reference", self, checkable=True)
        self.action_roi = QAction("ROI objects", self, checkable=True)
        self.action_pan.setChecked(True)
        for action, mode in (
            (self.action_pan, "Pan"),
            (self.action_point, "Point"),
            (self.action_roi, "ROI"),
        ):
            mode_group.addAction(action)
            toolbar.addAction(action)
            action.triggered.connect(lambda _checked, m=mode: self.set_mode_all(m))
        toolbar.addSeparator()
        for text, callback in (
            ("Zoom in", self.zoom_in_active),
            ("Zoom out", self.zoom_out_active),
            ("Reset zoom", self.reset_zoom_active),
            ("Clear active", self.clear_active_selections),
            ("Tile", self.mdi.tileSubWindows),
        ):
            action = QAction(text, self)
            action.triggered.connect(callback)
            toolbar.addAction(action)

    def _build_controls(self) -> None:
        dock = QDockWidget("Scientific controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        release = QLabel(
            f"<b>SpectralView {VERSION}</b><br>"
            "Stable candidate; external validation not claimed.<br>"
            "Calibration and batch processing are fail-explicit."
        )
        release.setWordWrap(True)
        layout.addWidget(release)

        data_group = QGroupBox("1. Input and calibration declaration")
        data_layout = QVBoxLayout(data_group)
        self.input_reflectance = QCheckBox("Input cube is already reflectance (explicit declaration)")
        self.input_reflectance.setToolTip(
            "Enable only when the supplied data are known reflectance. Dark/white files will then be ignored with a warning."
        )
        data_layout.addWidget(self.input_reflectance)
        self.clip_reflectance = QCheckBox("Clip calibrated reflectance (applies to newly opened/batch data)")
        self.clip_reflectance.setChecked(False)
        self.clip_max = QDoubleSpinBox()
        self.clip_max.setRange(0.1, 10.0)
        self.clip_max.setValue(1.5)
        self.clip_max.setEnabled(False)
        self.clip_reflectance.toggled.connect(self.clip_max.setEnabled)
        clip_row = QHBoxLayout()
        clip_row.addWidget(self.clip_reflectance)
        clip_row.addWidget(QLabel("Maximum"))
        clip_row.addWidget(self.clip_max)
        data_layout.addLayout(clip_row)
        open_row = QHBoxLayout()
        button_folder = QPushButton("Open folder")
        button_folder.clicked.connect(self.choose_folder)
        button_header = QPushButton("Open header")
        button_header.clicked.connect(self.choose_header)
        open_row.addWidget(button_folder)
        open_row.addWidget(button_header)
        data_layout.addLayout(open_row)
        button_parent = QPushButton("Open parent datasets")
        button_parent.clicked.connect(self.choose_parent)
        data_layout.addWidget(button_parent)
        layout.addWidget(data_group)

        vis_group = QGroupBox("2. Visualisation")
        vis_layout = QFormLayout(vis_group)
        self.vis_mode = QComboBox()
        self.vis_mode.addItems(
            [
                "PCA composite",
                "True colour (wavelength-based)",
                "False colour (wavelength-based)",
                "Single-band heatmap",
                "Spectral index",
            ]
        )
        vis_layout.addRow("Mode", self.vis_mode)
        self.spin_r_nm = self._wavelength_spin(800.0)
        self.spin_g_nm = self._wavelength_spin(670.0)
        self.spin_b_nm = self._wavelength_spin(550.0)
        self.spin_single_nm = self._wavelength_spin(800.0)
        vis_layout.addRow("False R (nm)", self.spin_r_nm)
        vis_layout.addRow("False G (nm)", self.spin_g_nm)
        vis_layout.addRow("False B (nm)", self.spin_b_nm)
        vis_layout.addRow("Single band (nm)", self.spin_single_nm)
        self.index_combo = QComboBox()
        self.index_combo.addItems(sorted(INDEX_DEFINITIONS))
        vis_layout.addRow("Index", self.index_combo)
        self.index_tolerance = QDoubleSpinBox()
        self.index_tolerance.setRange(0.1, 100.0)
        self.index_tolerance.setValue(15.0)
        self.index_tolerance.setSuffix(" nm")
        vis_layout.addRow("Index tolerance", self.index_tolerance)
        self.low_percentile = QDoubleSpinBox()
        self.low_percentile.setRange(0.0, 49.0)
        self.low_percentile.setValue(2.0)
        self.high_percentile = QDoubleSpinBox()
        self.high_percentile.setRange(51.0, 100.0)
        self.high_percentile.setValue(98.0)
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 5.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        vis_layout.addRow("Low percentile", self.low_percentile)
        vis_layout.addRow("High percentile", self.high_percentile)
        vis_layout.addRow("Gamma", self.gamma_spin)
        layout.addWidget(vis_group)

        preprocessing_group = QGroupBox("3. Spectral preprocessing")
        preprocessing_layout = QFormLayout(preprocessing_group)
        self.preprocessing_combo = QComboBox()
        for label, value in (
            ("None", "none"),
            ("SNV", "snv"),
            ("Savitzky–Golay smooth", "savitzky_golay"),
            ("First derivative", "first_derivative"),
            ("Second derivative", "second_derivative"),
            ("Absorbance −log10(R)", "absorbance"),
        ):
            self.preprocessing_combo.addItem(label, value)
        self.sg_window = QSpinBox()
        self.sg_window.setRange(3, 101)
        self.sg_window.setSingleStep(2)
        self.sg_window.setValue(11)
        self.sg_polyorder = QSpinBox()
        self.sg_polyorder.setRange(1, 5)
        self.sg_polyorder.setValue(2)
        preprocessing_layout.addRow("Method", self.preprocessing_combo)
        preprocessing_layout.addRow("SG window", self.sg_window)
        preprocessing_layout.addRow("SG order", self.sg_polyorder)
        layout.addWidget(preprocessing_group)

        roi_group = QGroupBox("4. Reference extraction and ROI QC")
        roi_layout = QFormLayout(roi_group)
        self.class_label = QLineEdit("Class_1")
        self.class_label.setToolTip("All new point/ROI references receive this class label.")
        self.polarity_combo = QComboBox()
        self.polarity_combo.addItems(["auto", "normal", "inverted"])
        self.min_area_spin = QSpinBox()
        self.min_area_spin.setRange(1, 10_000_000)
        self.min_area_spin.setValue(25)
        self.morphology_spin = QSpinBox()
        self.morphology_spin.setRange(1, 21)
        self.morphology_spin.setSingleStep(2)
        self.morphology_spin.setValue(3)
        roi_layout.addRow("Class label", self.class_label)
        roi_layout.addRow("Mask polarity", self.polarity_combo)
        roi_layout.addRow("Minimum object area", self.min_area_spin)
        roi_layout.addRow("Morphology kernel", self.morphology_spin)
        layout.addWidget(roi_group)

        sam_group = QGroupBox("5. Genuine Spectral Angle Mapper")
        sam_layout = QFormLayout(sam_group)
        self.sam_threshold = QDoubleSpinBox()
        self.sam_threshold.setRange(0.1, 90.0)
        self.sam_threshold.setValue(12.0)
        self.sam_threshold.setSuffix("°")
        self.sam_chunk_rows = QSpinBox()
        self.sam_chunk_rows.setRange(1, 512)
        self.sam_chunk_rows.setValue(32)
        self.classify_button = QPushButton("Classify active cube with SAM")
        self.classify_button.clicked.connect(self.classify_active_image)
        self.export_class_button = QPushButton("Export active SAM evidence")
        self.export_class_button.clicked.connect(self.export_classification)
        sam_layout.addRow("Acceptance angle", self.sam_threshold)
        sam_layout.addRow("Chunk rows", self.sam_chunk_rows)
        sam_layout.addRow(self.classify_button)
        sam_layout.addRow(self.export_class_button)
        layout.addWidget(sam_group)

        batch_group = QGroupBox("6. Governed batch")
        batch_layout = QFormLayout(batch_group)
        self.batch_mode = QComboBox()
        self.batch_mode.addItem("Auto foreground (fail if segmentation fails)", "auto_foreground")
        self.batch_mode.addItem("Whole-frame mean (explicit)", "whole_frame")
        self.batch_button = QPushButton("Run batch to evidence directory")
        self.batch_button.clicked.connect(self.batch_process_folders)
        batch_layout.addRow("Extraction", self.batch_mode)
        batch_layout.addRow(self.batch_button)
        layout.addWidget(batch_group)

        status_group = QGroupBox("Active document truth")
        status_layout = QVBoxLayout(status_group)
        self.active_metadata = QLabel("No active document")
        self.active_metadata.setWordWrap(True)
        status_layout.addWidget(self.active_metadata)
        layout.addWidget(status_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(1000)
        self.log_view.setMinimumHeight(150)
        layout.addWidget(self.log_view)
        layout.addStretch(1)

        for widget in (
            self.vis_mode,
            self.spin_r_nm,
            self.spin_g_nm,
            self.spin_b_nm,
            self.spin_single_nm,
            self.index_combo,
            self.index_tolerance,
            self.low_percentile,
            self.high_percentile,
            self.gamma_spin,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.update_all_image_views)
            else:
                widget.valueChanged.connect(self.update_all_image_views)
        self.preprocessing_combo.currentIndexChanged.connect(self.update_all_plots)
        self.sg_window.valueChanged.connect(self.update_all_plots)
        self.sg_polyorder.valueChanged.connect(self.update_all_plots)

        scroll.setWidget(container)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    @staticmethod
    def _wavelength_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 5000.0)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setSuffix(" nm")
        return spin

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{timestamp}] {message}") if hasattr(self, "log_view") else None

    def set_busy(self, busy: bool) -> None:
        self._busy_tasks += 1 if busy else -1
        self._busy_tasks = max(0, self._busy_tasks)
        active = self._busy_tasks > 0
        self.classify_button.setEnabled(not active)
        self.batch_button.setEnabled(not active)
        if not active:
            self.progress_bar.setValue(0)

    def current_calibration_config(self) -> CalibrationConfig:
        clipping = self.clip_reflectance.isChecked()
        return CalibrationConfig(
            input_is_reflectance=self.input_reflectance.isChecked(),
            clip_min=0.0 if clipping else None,
            clip_max=self.clip_max.value() if clipping else None,
        )

    def current_processing_config(self) -> ProcessingConfig:
        return ProcessingConfig(
            calibration=self.current_calibration_config(),
            preprocessing=self.preprocessing_combo.currentData(),
            sg_window=self.sg_window.value(),
            sg_polyorder=self.sg_polyorder.value(),
            sam_threshold_deg=self.sam_threshold.value(),
            chunk_rows=self.sam_chunk_rows.value(),
            segmentation_min_area_px=self.min_area_spin.value(),
            segmentation_polarity=self.polarity_combo.currentText().lower(),
            segmentation_morphology_size=self.morphology_spin.value(),
            preview_max_dimension=1000,
        )

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select ENVI dataset folder")
        if folder:
            self.load_folder(folder)

    def choose_header(self) -> None:
        header, _filter = QFileDialog.getOpenFileName(self, "Select ENVI data header", "", "ENVI headers (*.hdr)")
        if header:
            self.load_header(header)

    def choose_parent(self) -> None:
        parent = QFileDialog.getExistingDirectory(self, "Select parent directory")
        if not parent:
            return
        folders = find_dataset_folders(parent)
        if not folders:
            QMessageBox.warning(self, "No datasets", "No governed ENVI data headers were found.")
            return
        if len(folders) > 8:
            answer = QMessageBox.question(
                self,
                "Open many datasets",
                f"{len(folders)} datasets were found. Opening all can consume substantial memory. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        for index, folder in enumerate(folders, start=1):
            self.progress_bar.setValue(int(index / len(folders) * 100))
            QApplication.processEvents()
            self.load_folder(folder, show_errors=False)
        self.progress_bar.setValue(0)

    def load_folder(self, folder: str, show_errors: bool = True) -> None:
        try:
            discovery = discover_envi_folder(folder)
            if not discovery.data_header:
                raise InputContractError("No data ENVI header was identified in the selected folder.")
            for warning in discovery.warnings:
                self.log(f"DISCOVERY WARNING {Path(folder).name}: {warning}")
            self._open_document(discovery.data_header, discovery.dark_header, discovery.white_header)
        except Exception as exc:
            self.log(f"OPEN ERROR {folder}: {exc}")
            if show_errors:
                QMessageBox.critical(self, "Open dataset failed", str(exc))

    def load_header(self, data_header: str) -> None:
        try:
            discovery = discover_envi_folder(str(Path(data_header).parent))
            dark = discovery.dark_header
            white = discovery.white_header
            self._open_document(data_header, dark, white)
        except Exception as exc:
            self.log(f"OPEN ERROR {data_header}: {exc}")
            QMessageBox.critical(self, "Open ENVI header failed", str(exc))

    def _open_document(self, data_header: str, dark_header: Optional[str], white_header: Optional[str]) -> None:
        document = EnviDocument(
            data_header,
            dark_header,
            white_header,
            self.current_calibration_config(),
        )
        name = Path(data_header).parent.name or Path(data_header).stem
        window = DatacubeWindow(self, document, name)
        self.image_windows.append(window)
        self.mdi.addSubWindow(window)
        window.show()
        self.set_mode_all("Pan" if self.action_pan.isChecked() else "Point" if self.action_point.isChecked() else "ROI")
        self.update_active_metadata()
        self.log(
            f"Opened {name}: shape={document.height}×{document.width}×{document.n_bands}; "
            f"dark={'yes' if dark_header else 'no'}; white={'yes' if white_header else 'no'}"
        )

    def unregister_window(self, window: DatacubeWindow) -> None:
        if window in self.image_windows:
            self.image_windows.remove(window)
        self.update_active_metadata()

    def get_active_window(self) -> Optional[DatacubeWindow]:
        active = self.mdi.activeSubWindow()
        return active if isinstance(active, DatacubeWindow) else None

    def update_active_metadata(self) -> None:
        active = self.get_active_window()
        if not active:
            self.active_metadata.setText("No active document")
            return
        doc = active.document
        qc = active.preview_qc
        wavelength_label = (
            f"{doc.wavelengths.min():.2f}–{doc.wavelengths.max():.2f} nm"
            if doc.has_physical_wavelengths
            else "band indices only"
        )
        self.active_metadata.setText(
            f"<b>{active.name}</b><br>"
            f"Shape: {doc.height} × {doc.width} × {doc.n_bands}<br>"
            f"Spectral axis: {wavelength_label}<br>"
            f"Calibration: {qc.mode if qc else 'not evaluated'}<br>"
            f"Units: {qc.units if qc else 'unknown'}<br>"
            f"Dark: {'yes' if doc.dark_header else 'no'}; White: {'yes' if doc.white_header else 'no'}<br>"
            f"References selected: {len(active.selections)}"
        )

    def set_mode_all(self, mode: str) -> None:
        for window in self.image_windows:
            window.view.set_mode(mode)

    def zoom_in_active(self) -> None:
        active = self.get_active_window()
        if active:
            active.zoom_in()

    def zoom_out_active(self) -> None:
        active = self.get_active_window()
        if active:
            active.zoom_out()

    def reset_zoom_active(self) -> None:
        active = self.get_active_window()
        if active:
            active.reset_zoom()

    def clear_active_selections(self) -> None:
        active = self.get_active_window()
        if active:
            active.clear_selections()
            self.update_active_metadata()

    def update_all_image_views(self, *_args) -> None:
        for window in self.image_windows:
            window.update_image_view()

    def update_all_plots(self, *_args) -> None:
        for window in self.image_windows:
            window.update_plot()

    def _prepare_reference_library(self, active: DatacubeWindow) -> Tuple[List[str], np.ndarray, List[str]]:
        grouped: Dict[str, List[np.ndarray]] = defaultdict(list)
        warnings: List[str] = []
        target_wavelengths = active.document.wavelengths
        target_physical = active.document.has_physical_wavelengths
        for window in self.image_windows:
            for selection in window.selections:
                try:
                    source_wavelengths = selection["wavelengths"]
                    profile = selection["profile"]
                    source_physical = selection["has_physical_wavelengths"]
                    active_units = active.preview_qc.units if active.preview_qc else "unknown"
                    if selection["units"] != active_units:
                        raise WavelengthCompatibilityError(
                            f"Reference units {selection['units']} do not match active-cube units {active_units}."
                        )
                    if target_physical and source_physical:
                        aligned = resample_spectrum(source_wavelengths, profile, target_wavelengths, True)
                    elif len(profile) == active.document.n_bands and np.allclose(
                        source_wavelengths, target_wavelengths, rtol=0, atol=1e-8
                    ):
                        aligned = np.asarray(profile, dtype=np.float32)
                    else:
                        raise WavelengthCompatibilityError(
                            "Physical wavelengths are missing and band grids are not identical."
                        )
                    if not np.all(np.isfinite(aligned)):
                        raise WavelengthCompatibilityError("Aligned spectrum contains non-finite values.")
                    grouped[selection["class_label"]].append(aligned)
                except Exception as exc:
                    warnings.append(f"Skipped {window.name} reference {selection['id']}: {exc}")
        if not grouped:
            raise SpectralViewError("No wavelength-compatible reference spectra are available.")
        names = sorted(grouped)
        references = np.stack([np.mean(grouped[name], axis=0) for name in names], axis=0).astype(np.float32)
        return names, references, warnings

    def classify_active_image(self) -> None:
        active = self.get_active_window()
        if not active:
            QMessageBox.warning(self, "SAM", "Select an active datacube window.")
            return
        if self.preprocessing_combo.currentData() == "absorbance" and (
            active.preview_qc is None or active.preview_qc.units != "reflectance"
        ):
            QMessageBox.warning(self, "SAM preprocessing", "Absorbance preprocessing requires reflectance input.")
            return
        try:
            class_names, references, warnings = self._prepare_reference_library(active)
        except Exception as exc:
            QMessageBox.warning(self, "SAM reference library", str(exc))
            return
        for warning in warnings:
            self.log(f"SAM REFERENCE WARNING: {warning}")

        processing = self.current_processing_config()
        self.set_busy(True)
        self.progress_bar.setValue(0)
        self.log(
            f"SAM started for {active.name}: classes={class_names}; threshold={processing.sam_threshold_deg:.2f}°; "
            f"preprocessing={processing.preprocessing}"
        )

        def task(progress_emit):
            result = active.document.sam_classify(
                references,
                preprocessing=processing.preprocessing,
                threshold_deg=processing.sam_threshold_deg,
                sg_window=processing.sg_window,
                sg_polyorder=processing.sg_polyorder,
                chunk_rows=processing.chunk_rows,
                progress=lambda value: progress_emit((value, "SAM")),
            )
            return active, class_names, references, result, processing

        worker = FunctionWorker(task)
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.error.connect(self._worker_error)
        worker.signals.result.connect(self._classification_complete)
        worker.signals.finished.connect(lambda: self.set_busy(False))
        self.thread_pool.start(worker)

    def _classification_complete(self, payload) -> None:
        active, class_names, _references, result, processing = payload
        if active not in self.image_windows:
            self.log("SAM completed after the target window was closed; result was not displayed.")
            return
        active.set_classification_overlay(result, class_names)
        self.log(
            f"SAM complete {active.name}: valid={result.valid_fraction:.4f}; "
            f"unknown among valid={result.unknown_fraction:.4f}; classes={class_names}"
        )
        QMessageBox.information(
            self,
            "SAM complete",
            f"Genuine Spectral Angle Mapper completed.\n\n"
            f"Classes: {', '.join(class_names)}\n"
            f"Threshold: {processing.sam_threshold_deg:.2f}°\n"
            f"Valid-pixel fraction: {result.valid_fraction:.4f}\n"
            f"Unknown fraction among valid pixels: {result.unknown_fraction:.4f}",
        )

    def _worker_progress(self, payload) -> None:
        if isinstance(payload, tuple):
            value = int(payload[0])
            detail = str(payload[1]) if len(payload) > 1 else ""
        else:
            value = int(payload)
            detail = ""
        self.progress_bar.setValue(max(0, min(100, value)))
        if detail:
            self.statusBar().showMessage(f"{detail}: {value}%")

    def _worker_error(self, payload) -> None:
        exc, tb = payload
        self.log(f"WORKER ERROR: {type(exc).__name__}: {exc}")
        self.log(tb)
        QMessageBox.critical(self, "Processing failed", f"{type(exc).__name__}: {exc}")

    def export_profiles(self) -> None:
        selections = [(window, selection) for window in self.image_windows for selection in window.selections]
        if not selections:
            QMessageBox.warning(self, "Export", "No extracted spectra are available.")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export extracted spectra", "spectra_long.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        method = self.preprocessing_combo.currentData()
        rows: List[List[Any]] = [[
            "document", "selection_id", "class_label", "selection_type", "raw_x", "raw_y", "area_px",
            "wavelength", "value", "units", "calibration_mode", "preprocessing"
        ]]
        try:
            for window, selection in selections:
                if method == "absorbance" and selection["units"] != "reflectance":
                    raise SpectralViewError(
                        f"Absorbance export blocked for {window.name} selection {selection['id']}: input is {selection['units']}."
                    )
                processed = preprocess_spectra(
                    selection["profile"],
                    method,
                    selection["wavelengths"],
                    self.sg_window.value(),
                    self.sg_polyorder.value(),
                )
                for wavelength, value in zip(selection["wavelengths"], processed):
                    rows.append([
                        window.name,
                        selection["id"],
                        selection["class_label"],
                        selection["selection_type"],
                        selection["raw_x"],
                        selection["raw_y"],
                        selection["area_px"],
                        f"{float(wavelength):.8g}",
                        "" if not np.isfinite(value) else f"{float(value):.10g}",
                        selection["units"],
                        selection["calibration_mode"],
                        method,
                    ])
            atomic_write_csv(path, rows)
            QMessageBox.information(self, "Export complete", f"Extracted spectra saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))


    def export_profiles_wide(self) -> None:
        selections = [(window, selection) for window in self.image_windows for selection in window.selections]
        if not selections:
            QMessageBox.warning(self, "Export", "No extracted spectra are available.")
            return
            
        active = self.get_active_window()
        if not active:
            QMessageBox.warning(
                self, 
                "Export Wide", 
                "Please make one datacube window active to serve as the reference wavelength grid."
            )
            return

        target_wavelengths = active.document.wavelengths
        target_physical = active.document.has_physical_wavelengths
        
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export extracted spectra (wide CSV)", "spectra_wide.csv", "CSV files (*.csv)"
        )
        if not path:
            return
            
        method = self.preprocessing_combo.currentData()
        
        headers = [
            "document", "selection_id", "class_label", "selection_type", "raw_x", "raw_y", "area_px",
            "units", "calibration_mode", "preprocessing"
        ]
        if target_physical:
            headers.extend([f"Wavelength_{w:.2f}" for w in target_wavelengths])
        else:
            headers.extend([f"Band_{int(w)}" for w in target_wavelengths])
            
        rows: List[List[Any]] = [headers]
        
        for window, selection in selections:
            if method == "absorbance" and selection["units"] != "reflectance":
                self.log(f"Skipped {window.name} selection {selection['id']}: Absorbance requires reflectance.")
                continue
                
            try:
                processed = preprocess_spectra(
                    selection["profile"],
                    method,
                    selection["wavelengths"],
                    self.sg_window.value(),
                    self.sg_polyorder.value(),
                )
                
                source_physical = selection["has_physical_wavelengths"]
                if target_physical and source_physical:
                    aligned = resample_spectrum(selection["wavelengths"], processed, target_wavelengths, True)
                elif len(processed) == len(target_wavelengths) and np.allclose(
                    selection["wavelengths"], target_wavelengths, rtol=0, atol=1e-8
                ):
                    aligned = np.asarray(processed, dtype=np.float32)
                else:
                    self.log(f"Skipped {window.name} selection {selection['id']}: Wavelength mismatch.")
                    continue
                    
                if not np.all(np.isfinite(aligned)):
                    self.log(f"Skipped {window.name} selection {selection['id']}: Contains non-finite values.")
                    continue

            except Exception as exc:
                self.log(f"Skipped {window.name} selection {selection['id']}: {exc}")
                continue
                
            row = [
                window.name,
                selection["id"],
                selection["class_label"],
                selection["selection_type"],
                selection["raw_x"],
                selection["raw_y"],
                selection["area_px"],
                selection["units"],
                selection["calibration_mode"],
                method,
            ]
            row.extend(["" if not np.isfinite(val) else f"{float(val):.8g}" for val in aligned])
            rows.append(row)
            
        try:
            atomic_write_csv(path, rows)
            QMessageBox.information(self, "Export complete", f"Wide-format spectra saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Failed to save CSV:\n{exc}")

    def export_roi_1000(self) -> None:
        selections = [(window, selection) for window in self.image_windows for selection in window.selections]
        if not selections:
            QMessageBox.warning(self, "Export", "No extracted spectra are available.")
            return
            
        active = self.get_active_window()
        if not active:
            QMessageBox.warning(
                self, 
                "Export", 
                "Please make one datacube window active to serve as the reference wavelength grid."
            )
            return

        target_wavelengths = active.document.wavelengths
        target_physical = active.document.has_physical_wavelengths
        if not target_physical:
            QMessageBox.warning(self, "Export", "Active document must have physical wavelengths to filter up to 1000 nm.")
            return
            
        path, _filter = QFileDialog.getSaveFileName(
            self, "Export ROI and Wavelengths <= 1000 nm", "roi_wavelengths_1000.csv", "CSV files (*.csv)"
        )
        if not path:
            return
            
        method = self.preprocessing_combo.currentData()
        
        valid_indices = [i for i, w in enumerate(target_wavelengths) if w <= 1000.0]
        filtered_wavelengths = [target_wavelengths[i] for i in valid_indices]
        
        headers = ["ROI_number"]
        headers.extend([f"{w:.2f}" for w in filtered_wavelengths])
            
        rows: List[List[Any]] = [headers]
        
        for window, selection in selections:
            if method == "absorbance" and selection["units"] != "reflectance":
                self.log(f"Skipped {window.name} selection {selection['id']}: Absorbance requires reflectance.")
                continue
                
            try:
                processed = preprocess_spectra(
                    selection["profile"],
                    method,
                    selection["wavelengths"],
                    self.sg_window.value(),
                    self.sg_polyorder.value(),
                )
                
                source_physical = selection["has_physical_wavelengths"]
                if source_physical:
                    aligned = resample_spectrum(selection["wavelengths"], processed, target_wavelengths, True)
                elif len(processed) == len(target_wavelengths) and np.allclose(
                    selection["wavelengths"], target_wavelengths, rtol=0, atol=1e-8
                ):
                    aligned = np.asarray(processed, dtype=np.float32)
                else:
                    self.log(f"Skipped {window.name} selection {selection['id']}: Source lacks physical wavelengths.")
                    continue
                    
                if not np.all(np.isfinite(aligned)):
                    self.log(f"Skipped {window.name} selection {selection['id']}: Contains non-finite values.")
                    continue

            except Exception as exc:
                self.log(f"Skipped {window.name} selection {selection['id']}: {exc}")
                continue
                
            row = [selection["id"]]
            filtered_values = [aligned[i] for i in valid_indices]
            row.extend(["" if not np.isfinite(val) else f"{float(val):.8g}" for val in filtered_values])
            rows.append(row)
            
        try:
            atomic_write_csv(path, rows)
            QMessageBox.information(self, "Export complete", f"Filtered ROI spectra saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Failed to save CSV:\n{exc}")

    def export_plot(self) -> None:
        active = self.get_active_window()
        if not active or not active.selections:
            QMessageBox.warning(self, "Export", "The active document has no plotted selections.")
            return
        path, _filter = QFileDialog.getSaveFileName(
            self,
            "Export active spectral plot",
            f"{active.name}_spectra.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)",
        )
        if not path:
            return
        try:
            active.canvas.fig.savefig(path, dpi=300, bbox_inches="tight")
            QMessageBox.information(self, "Export complete", f"Plot saved to:\n{path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    def export_classification(self) -> None:
        active = self.get_active_window()
        if not active or active.last_classification is None:
            QMessageBox.warning(self, "SAM export", "The active document has no SAM result.")
            return
        parent = QFileDialog.getExistingDirectory(self, "Select SAM evidence output directory")
        if not parent:
            return
        out = Path(parent) / f"{active.name}_SAM_evidence"
        out.mkdir(parents=True, exist_ok=True)
        result = active.last_classification
        try:
            np.save(out / "class_map.npy", result.class_map)
            np.save(out / "minimum_angle_degrees.npy", result.angle_map_deg)
            if active.last_overlay_rgba is not None:
                cv2.imwrite(str(out / "classification_overlay.png"), cv2.cvtColor(active.last_overlay_rgba, cv2.COLOR_RGBA2BGRA))
            class_rows = [["class_index", "class_name"]] + [
                [index, name] for index, name in enumerate(active.last_class_names)
            ]
            atomic_write_csv(out / "class_legend.csv", class_rows)
            metadata = {
                "software": "SpectralView",
                "version": VERSION,
                "release_status": RELEASE_STATUS,
                "document": active.document.provenance(),
                "class_names": active.last_class_names,
                "sam_threshold_degrees": self.sam_threshold.value(),
                "preprocessing": self.preprocessing_combo.currentData(),
                "valid_fraction": result.valid_fraction,
                "unknown_fraction": result.unknown_fraction,
                "note": "Class map indexes are zero-based; -1 denotes unknown/unclassified.",
            }
            atomic_write_text(out / "classification_manifest.json", json.dumps(metadata, indent=2, sort_keys=True))
            QMessageBox.information(self, "SAM export complete", f"Evidence saved to:\n{out}")
        except Exception as exc:
            QMessageBox.critical(self, "SAM export failed", str(exc))

    def batch_process_folders(self) -> None:
        parent = QFileDialog.getExistingDirectory(self, "Select parent directory containing ENVI datasets")
        if not parent:
            return
        output_parent = QFileDialog.getExistingDirectory(self, "Select parent directory for batch evidence")
        if not output_parent:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(Path(output_parent) / f"SpectralView_batch_{timestamp}")
        processing = self.current_processing_config()
        extraction_mode = self.batch_mode.currentData()
        self.set_busy(True)
        self.progress_bar.setValue(0)
        self.log(f"Batch started: parent={parent}; output={output}; extraction={extraction_mode}")

        def task(progress_emit):
            return run_batch(
                parent,
                output,
                processing,
                extraction_mode=extraction_mode,
                progress=lambda value, detail: progress_emit((value, detail)),
            )

        worker = FunctionWorker(task)
        worker.signals.progress.connect(self._worker_progress)
        worker.signals.error.connect(self._worker_error)
        worker.signals.result.connect(lambda manifest: self._batch_complete(output, manifest))
        worker.signals.finished.connect(lambda: self.set_busy(False))
        self.thread_pool.start(worker)

    def _batch_complete(self, output: str, manifest: Dict[str, Any]) -> None:
        counts = manifest.get("terminal_status_counts", {})
        self.log(f"Batch complete: {counts}; output={output}")
        QMessageBox.information(
            self,
            "Batch complete",
            "Governed batch reached terminal status for every discovered dataset.\n\n"
            f"Status counts: {json.dumps(counts, sort_keys=True)}\n"
            f"Evidence directory:\n{output}",
        )


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--folder", help="Open one ENVI dataset folder at startup.")
    parser.add_argument("--header", help="Open one ENVI data header at startup.")
    parser.add_argument("--smoke-test", action="store_true", help="Open the requested data, render briefly, then exit.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    app = QApplication(sys.argv if argv is None else [sys.argv[0], *argv])
    app.setApplicationName("SpectralView")
    app.setApplicationVersion(VERSION)
    viewer = HyperspectralViewer(args.folder, args.header)
    viewer.show()
    if args.smoke_test:
        if not viewer.image_windows:
            return 3
        QTimer.singleShot(500, app.quit)
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
