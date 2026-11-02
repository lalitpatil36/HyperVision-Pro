from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import numpy as np
import traceback
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, 
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QProgressDialog,
    QApplication, QTextEdit, QColorDialog, QSplitter, QComboBox, QWidget, QCheckBox,
    QTabWidget, QGroupBox, QTextBrowser
)
from PyQt5.QtCore import Qt, QRunnable, QThreadPool, pyqtSignal, QObject
from PyQt5.QtGui import QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class WorkerSignals(QObject):
    progress = pyqtSignal(tuple)
    result = pyqtSignal(object)
    error = pyqtSignal(tuple)
    finished = pyqtSignal()

class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(self.signals.progress.emit, *self.args, **self.kwargs)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit((e, traceback.format_exc()))
        finally:
            self.signals.finished.emit()

class NumericMappingDialog(QDialog):
    def __init__(self, unique_classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map Classes to Numeric Values")
        self.resize(400, 300)
        self.mapping = {}
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Regression requires continuous numeric targets.\nPlease assign a numeric value (e.g., severity %) to each class:"))
        
        self.table = QTableWidget(len(unique_classes), 2)
        self.table.setHorizontalHeaderLabels(["Class Label", "Numeric Value"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        for i, cls in enumerate(unique_classes):
            item_cls = QTableWidgetItem(str(cls))
            item_cls.setFlags(item_cls.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, item_cls)
            self.table.setItem(i, 1, QTableWidgetItem("0.0"))
            
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Apply Mapping")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addStretch()
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def get_mapping(self):
        for i in range(self.table.rowCount()):
            cls = self.table.item(i, 0).text()
            val_str = self.table.item(i, 1).text()
            try:
                self.mapping[cls] = float(val_str)
            except ValueError:
                self.mapping[cls] = 0.0
        return self.mapping

class PlotStyleDialog(QDialog):
    def __init__(self, unique_classes, current_styles, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Plot Styles")
        self.resize(550, 400)
        
        self.unique_classes = unique_classes
        self.current_styles = current_styles.copy()
        
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget(len(unique_classes), 3)
        self.table.setHorizontalHeaderLabels(["Original Class", "Legend Name", "Plot Color"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)
        
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for i, cls in enumerate(unique_classes):
            item_orig = QTableWidgetItem(str(cls))
            item_orig.setFlags(item_orig.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, item_orig)
            
            style = self.current_styles.get(cls, {})
            legend_name = style.get('name', str(cls))
            self.table.setItem(i, 1, QTableWidgetItem(legend_name))
            
            color_hex = style.get('color', default_colors[i % len(default_colors)])
            btn_color = QPushButton("Choose Color")
            btn_color.setStyleSheet(f"background-color: {color_hex}; font-weight: bold;")
            btn_color.setProperty('color_hex', color_hex)
            btn_color.clicked.connect(lambda checked, row=i, btn=btn_color: self.choose_color(row, btn))
            self.table.setCellWidget(i, 2, btn_color)
            
        btn_layout = QHBoxLayout()
        btn_save = QPushButton("Save Styles")
        btn_save.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_save)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def choose_color(self, row, btn):
        current_color = btn.property('color_hex')
        color = QColorDialog.getColor(QColor(current_color), self, "Select Plot Color")
        if color.isValid():
            new_hex = color.name()
            btn.setProperty('color_hex', new_hex)
            btn.setStyleSheet(f"background-color: {new_hex}; font-weight: bold;")
            
    def get_styles(self):
        styles = {}
        for i, cls in enumerate(self.unique_classes):
            legend_name = self.table.item(i, 1).text()
            color_hex = self.table.cellWidget(i, 2).property('color_hex')
            styles[cls] = {'name': legend_name, 'color': color_hex}
        return styles

SAMPLE_FORMAT_1_CSV = """class_label,400.0,420.0,440.0,460.0,480.0,500.0,520.0,540.0,560.0,580.0,600.0,620.0,640.0,660.0,680.0,700.0,720.0,740.0,760.0,780.0,800.0,850.0,900.0,950.0,1000.0
Healthy,0.045,0.048,0.052,0.056,0.060,0.065,0.070,0.078,0.082,0.079,0.074,0.068,0.062,0.058,0.055,0.095,0.220,0.380,0.440,0.460,0.470,0.480,0.475,0.460,0.450
Healthy,0.044,0.047,0.051,0.055,0.059,0.064,0.069,0.076,0.081,0.078,0.073,0.067,0.061,0.057,0.054,0.092,0.215,0.375,0.435,0.455,0.465,0.475,0.470,0.455,0.445
Mild,0.055,0.058,0.062,0.066,0.071,0.077,0.083,0.090,0.094,0.090,0.085,0.080,0.075,0.072,0.070,0.110,0.200,0.320,0.370,0.390,0.400,0.410,0.405,0.390,0.380
Mild,0.056,0.059,0.063,0.067,0.072,0.078,0.084,0.091,0.095,0.091,0.086,0.081,0.076,0.073,0.071,0.112,0.202,0.322,0.372,0.392,0.402,0.412,0.407,0.392,0.382
Severe,0.075,0.079,0.083,0.088,0.093,0.100,0.108,0.115,0.120,0.117,0.112,0.108,0.105,0.103,0.102,0.135,0.170,0.240,0.280,0.295,0.305,0.315,0.310,0.300,0.290
Severe,0.076,0.080,0.084,0.089,0.094,0.101,0.109,0.116,0.121,0.118,0.113,0.109,0.106,0.104,0.103,0.136,0.172,0.242,0.282,0.297,0.307,0.317,0.312,0.302,0.292
"""

SAMPLE_FORMAT_2_CSV = """Wavelength,Sample_1,Sample_2,Sample_3,Sample_4,Sample_5
400.0,0.0450,0.0440,0.0550,0.0560,0.0750
420.0,0.0480,0.0470,0.0580,0.0590,0.0790
440.0,0.0520,0.0510,0.0620,0.0630,0.0830
460.0,0.0560,0.0550,0.0660,0.0670,0.0880
480.0,0.0600,0.0590,0.0710,0.0720,0.0930
500.0,0.0650,0.0640,0.0770,0.0780,0.1000
520.0,0.0700,0.0690,0.0830,0.0840,0.1080
540.0,0.0780,0.0760,0.0900,0.0910,0.1150
560.0,0.0820,0.0810,0.0940,0.0950,0.1200
580.0,0.0790,0.0780,0.0900,0.0910,0.1170
600.0,0.0740,0.0730,0.0850,0.0860,0.1120
620.0,0.0680,0.0670,0.0800,0.0810,0.1080
640.0,0.0620,0.0610,0.0750,0.0760,0.1050
660.0,0.0580,0.0570,0.0720,0.0730,0.1030
680.0,0.0550,0.0540,0.0700,0.0710,0.1020
700.0,0.0950,0.0920,0.1100,0.1120,0.1350
720.0,0.2200,0.2150,0.2000,0.2020,0.1700
740.0,0.3800,0.3750,0.3200,0.3220,0.2400
760.0,0.4400,0.4350,0.3700,0.3720,0.2800
780.0,0.4600,0.4550,0.3900,0.3920,0.2950
800.0,0.4700,0.4650,0.4000,0.4020,0.3050
850.0,0.4800,0.4750,0.4100,0.4120,0.3150
900.0,0.4750,0.4700,0.4050,0.4070,0.3100
950.0,0.4600,0.4550,0.3900,0.3920,0.3000
1000.0,0.4500,0.4450,0.3800,0.3820,0.2900
"""

class CSVFormatGuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HyperVision Pro - CSV Format Specifications & Templates")
        self.resize(850, 650)
        
        main_layout = QVBoxLayout(self)
        
        header = QLabel(
            "<div style='background-color: #0f172a; color: white; padding: 12px; border-radius: 6px;'>"
            "<h3 style='margin: 0 0 4px 0; color: #38bdf8;'>📊 Supported CSV Formats for Spectral Analytics</h3>"
            "<p style='margin: 0; font-size: 12px; color: #cbd5e1;'>"
            "HyperVision Pro supports two high-throughput CSV formats for AI classification, dimensionality reduction (PCA/t-SNE), "
            "feature selection, PLSR/SVR regression, and spectral vegetation indices."
            "</p></div>"
        )
        main_layout.addWidget(header)
        
        tabs = QTabWidget(self)
        
        # --- TAB 1: Format 1 (Horizontal / Wide) ---
        tab1 = QWidget()
        l1 = QVBoxLayout(tab1)
        
        t1_desc = QTextBrowser()
        t1_desc.setHtml(
            "<h4 style='color: #1e293b; margin-top: 0;'>Format 1: Horizontal / Wide Table (Recommended for AI & Machine Learning)</h4>"
            "<p>In this format, <b>each row is an individual sample/pixel spectrum</b>, and <b>each numeric column represents a wavelength in nanometers (nm)</b>.</p>"
            "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; border-color: #cbd5e1;'>"
            "<tr style='background-color: #f1f5f9; font-weight: bold;'>"
            "<th>class_label</th><th>400.0</th><th>420.0</th><th>440.0</th><th>...</th><th>1000.0</th>"
            "</tr>"
            "<tr><td>Healthy</td><td>0.045</td><td>0.048</td><td>0.052</td><td>...</td><td>0.450</td></tr>"
            "<tr><td>Mild</td><td>0.055</td><td>0.058</td><td>0.062</td><td>...</td><td>0.380</td></tr>"
            "<tr><td>Severe</td><td>0.075</td><td>0.079</td><td>0.083</td><td>...</td><td>0.290</td></tr>"
            "</table>"
            "<br/><b>Key Requirements:</b>"
            "<ul>"
            "<li><b>Class Column</b>: First column must be named <code>class_label</code>, <code>class</code>, <code>label</code>, or <code>target</code>.</li>"
            "<li><b>Wavelength Columns</b>: Column headers must be numeric wavelengths in nm (e.g. <code>400.0</code>, <code>550.0</code>) or <code>Wavelength_XXX.XX</code>.</li>"
            "<li><b>Data Values</b>: Reflectance, absorbance, or radiance numbers (floats between 0.0 and 1.0 or raw counts).</li>"
            "</ul>"
        )
        l1.addWidget(t1_desc)
        
        l1.addWidget(QLabel("<b>Sample Raw CSV Content (Ready to Copy/Save):</b>"))
        self.txt_f1 = QTextEdit()
        self.txt_f1.setPlainText(SAMPLE_FORMAT_1_CSV.strip())
        self.txt_f1.setReadOnly(True)
        self.txt_f1.setMaximumHeight(130)
        self.txt_f1.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        l1.addWidget(self.txt_f1)
        
        btn_save_f1 = QPushButton("💾 Save Format 1 Template CSV to Disk...")
        btn_save_f1.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px;")
        btn_save_f1.clicked.connect(self.save_format1_template)
        l1.addWidget(btn_save_f1)
        tabs.addTab(tab1, "Format 1 (Horizontal / Wide - AI)")
        
        # --- TAB 2: Format 2 (Vertical / Spectrometer) ---
        tab2 = QWidget()
        l2 = QVBoxLayout(tab2)
        
        t2_desc = QTextBrowser()
        t2_desc.setHtml(
            "<h4 style='color: #1e293b; margin-top: 0;'>Format 2: Vertical / Spectrometer Matrix</h4>"
            "<p>Commonly exported by laboratory spectroradiometers (ASD FieldSpec, Ocean Optics, Spectral Evolution, etc.).</p>"
            "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse; width: 100%; border-color: #cbd5e1;'>"
            "<tr style='background-color: #f1f5f9; font-weight: bold;'>"
            "<th>Wavelength</th><th>Sample_1</th><th>Sample_2</th><th>Sample_3</th><th>...</th>"
            "</tr>"
            "<tr><td>400.0</td><td>0.0450</td><td>0.0440</td><td>0.0550</td><td>...</td></tr>"
            "<tr><td>405.0</td><td>0.0480</td><td>0.0470</td><td>0.0580</td><td>...</td></tr>"
            "<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>"
            "<tr><td>1000.0</td><td>0.4500</td><td>0.4450</td><td>0.3800</td><td>...</td></tr>"
            "</table>"
            "<br/><b>Key Requirements:</b>"
            "<ul>"
            "<li><b>First Column</b>: Must contain monotonic numeric wavelengths (increasing or decreasing nm, > 50 bands).</li>"
            "<li><b>Remaining Columns</b>: Each column represents a distinct spectrum.</li>"
            "<li>HyperVision automatically transposes this matrix into standard analysis orientation upon loading.</li>"
            "</ul>"
        )
        l2.addWidget(t2_desc)
        
        l2.addWidget(QLabel("<b>Sample Raw CSV Content (Ready to Copy/Save):</b>"))
        self.txt_f2 = QTextEdit()
        self.txt_f2.setPlainText(SAMPLE_FORMAT_2_CSV.strip())
        self.txt_f2.setReadOnly(True)
        self.txt_f2.setMaximumHeight(130)
        self.txt_f2.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        l2.addWidget(self.txt_f2)
        
        btn_save_f2 = QPushButton("💾 Save Format 2 Template CSV to Disk...")
        btn_save_f2.setStyleSheet("background-color: #0891b2; color: white; font-weight: bold; padding: 6px;")
        btn_save_f2.clicked.connect(self.save_format2_template)
        l2.addWidget(btn_save_f2)
        tabs.addTab(tab2, "Format 2 (Vertical / Spectrometer)")
        
        # --- TAB 3: Auto-Detection & Class Labels ---
        tab3 = QWidget()
        l3 = QVBoxLayout(tab3)
        t3_desc = QTextBrowser()
        t3_desc.setHtml(
            "<h4 style='color: #1e293b; margin-top: 0;'>Class Label Detection & Workflow Integration</h4>"
            "<p>HyperVision uses a multi-tier resolution system to identify class labels for supervised AI models:</p>"
            "<ol>"
            "<li><b>CSV Column Header</b>: Any column named <code>class_label</code>, <code>class</code>, <code>label</code>, or <code>target</code> is immediately recognized.</li>"
            "<li><b>Filename Keyword Inference</b>: If multiple CSV files are imported without a class column, the software automatically inspects the filename:"
            "<ul>"
            "<li>Files containing <code>health</code> &rarr; Class: <b>Healthy</b></li>"
            "<li>Files containing <code>mild</code> &rarr; Class: <b>Mild</b></li>"
            "<li>Files containing <code>severe</code> &rarr; Class: <b>Severe</b></li>"
            "<li>Files containing <code>coinfect</code> &rarr; Class: <b>Coinfected</b></li>"
            "</ul></li>"
            "<li><b>Interactive Table Override</b>: Double-click any cell under the <b>Class/Value</b> column in the import table to edit or assign classes manually at any time.</li>"
            "<li><b>Direct Export from Datacubes</b>: In the main HyperVision viewer window, click <b>File &rarr; Export extracted spectra (wide CSV)...</b> to export your current ROIs directly into compliant Format 1 files.</li>"
            "</ol>"
        )
        l3.addWidget(t3_desc)
        tabs.addTab(tab3, "Class Labels & Direct Export")
        
        main_layout.addWidget(tabs)
        
        btn_row = QHBoxLayout()
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_close.setMinimumWidth(100)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        main_layout.addLayout(btn_row)
        
    def save_format1_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Format 1 (Horizontal) Template", "spectral_data_format1_horizontal_template.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(SAMPLE_FORMAT_1_CSV.strip() + "\n")
                QMessageBox.information(self, "Template Saved", f"Format 1 template successfully saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Could not save template:\n{e}")
                
    def save_format2_template(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Format 2 (Vertical) Template", "spectral_data_format2_vertical_template.csv", "CSV Files (*.csv)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(SAMPLE_FORMAT_2_CSV.strip() + "\n")
                QMessageBox.information(self, "Template Saved", f"Format 2 template successfully saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Failed", f"Could not save template:\n{e}")

class AdvancedCSVAnalysisDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced CSV Analysis (AI, Registration & Indices)")
        self.resize(1200, 750)
        
        self.datasets = [] 
        self.class_styles = {} 
        self.thread_pool = QThreadPool.globalInstance()
        
        self.cached_dr_data = {}
        self.cached_fs_data = {}
        self.cached_reg_data = {}
        self.cached_idx_data = {}
        self.cached_raw_data = {} 
        
        self.setAcceptDrops(True)
        
        main_layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        # --- LEFT PANEL ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        import_layout = QHBoxLayout()
        self.btn_import = QPushButton("Import Cleaned CSVs")
        self.btn_import.clicked.connect(self.import_csvs)
        import_layout.addWidget(self.btn_import)
        
        self.btn_format_guide = QPushButton("ℹ️ CSV Format Guide")
        self.btn_format_guide.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.btn_format_guide.clicked.connect(self.open_format_guide)
        import_layout.addWidget(self.btn_format_guide)
        
        self.btn_send_library = QPushButton("📚 Send to Spectral Library")
        self.btn_send_library.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        self.btn_send_library.clicked.connect(self.export_to_spectral_library)
        self.btn_send_library.setEnabled(False)
        import_layout.addWidget(self.btn_send_library)
        
        self.lbl_status = QLabel("Ready.")
        import_layout.addWidget(self.lbl_status)
        import_layout.addStretch()
        left_layout.addLayout(import_layout)
        
        lbl_hint = QLabel(
            "<div style='background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 6px; border-radius: 4px; font-size: 11px; color: #475569;'>"
            "💡 <b>Format Tip:</b> Accepts <i>Horizontal (Wide)</i> or <i>Vertical (Spectrometer)</i> CSVs. "
            "Click <b>ℹ️ CSV Format Guide</b> to view rules & save template files."
            "</div>"
        )
        lbl_hint.setWordWrap(True)
        left_layout.addWidget(lbl_hint)
        
        self.table = QTableWidget(0, 3)
        self.table.setAcceptDrops(False)
        self.table.setHorizontalHeaderLabels(["Filename", "Class/Value", "Shape"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.cellChanged.connect(self.on_table_edit)
        left_layout.addWidget(self.table)
        
        self.log_view = QTextEdit()
        self.log_view.setAcceptDrops(False)
        self.log_view.setReadOnly(True)
        left_layout.addWidget(self.log_view)
        
        # --- RIGHT PANEL ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Row 1: Preprocessing & DR
        controls_layout = QHBoxLayout()
        
        self.chk_aug = QCheckBox("Add Noise (Augment)")
        controls_layout.addWidget(self.chk_aug)
        
        self.cmb_prep = QComboBox()
        self.cmb_prep.addItems(["No Preprocessing", "Absorbance", "Savitzky-Golay (11,2)", "1st Derivative", "2nd Derivative", "SNV"])
        controls_layout.addWidget(QLabel("Pre-process:"))
        controls_layout.addWidget(self.cmb_prep)
        
        self.cmb_dr = QComboBox()
        self.cmb_dr.addItems(["PCA (Complete Chemometrics Suite)", "t-SNE", "LDA (Linear Discriminant Analysis)", "Wavelength Correlation Matrix"])
        controls_layout.addWidget(QLabel("Dim Red:"))
        controls_layout.addWidget(self.cmb_dr)
        
        self.btn_dr = QPushButton("Run DR")
        self.btn_dr.clicked.connect(self.run_dr)
        self.btn_dr.setEnabled(False)
        controls_layout.addWidget(self.btn_dr)
        right_layout.addLayout(controls_layout)
        
        # Row 2: FS & Regression
        fs_reg_layout = QHBoxLayout()
        self.cmb_fs = QComboBox()
        self.cmb_fs.addItems(["Random Forest", "Extra Trees", "SVM (Linear)", "Logistic Regression (L1)", "Neural Network (MLP)"])
        fs_reg_layout.addWidget(QLabel("Classification/FS:"))
        fs_reg_layout.addWidget(self.cmb_fs)
        
        self.btn_fs = QPushButton("Run FS")
        self.btn_fs.clicked.connect(self.run_fs)
        self.btn_fs.setEnabled(False)
        fs_reg_layout.addWidget(self.btn_fs)

        self.cmb_reg = QComboBox()
        self.cmb_reg.addItems(["PLSR", "SVR (Linear)"])
        fs_reg_layout.addWidget(QLabel("Regression:"))
        fs_reg_layout.addWidget(self.cmb_reg)
        
        self.btn_reg = QPushButton("Run Regress")
        self.btn_reg.clicked.connect(self.run_reg)
        self.btn_reg.setEnabled(False)
        fs_reg_layout.addWidget(self.btn_reg)
        right_layout.addLayout(fs_reg_layout)
        
        # Row 3: Indices & View
        idx_layout = QHBoxLayout()
        self.cmb_idx = QComboBox()
        self.cmb_idx.addItems(["NDVI", "PRI", "ARI"])
        idx_layout.addWidget(QLabel("Index:"))
        idx_layout.addWidget(self.cmb_idx)
        
        self.btn_idx = QPushButton("Run Index")
        self.btn_idx.clicked.connect(self.run_idx)
        self.btn_idx.setEnabled(False)
        idx_layout.addWidget(self.btn_idx)

        self.btn_styles = QPushButton("Styles")
        self.btn_styles.clicked.connect(self.open_style_dialog)
        self.btn_styles.setEnabled(False)
        idx_layout.addWidget(self.btn_styles)
        
        idx_layout.addSpacing(20)
        idx_layout.addWidget(QLabel("View Plot:"))
        self.cmb_view = QComboBox()
        self.cmb_view.currentIndexChanged.connect(self.draw_current_plot)
        idx_layout.addWidget(self.cmb_view)
        
        right_layout.addLayout(idx_layout)
        
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.nav_toolbar)
        right_layout.addWidget(self.canvas)
        
        self.btn_export = QPushButton("Export All Results & PDF Report")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        right_layout.addWidget(self.btn_export)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter)
        
    def log(self, msg: str):
        self.log_view.append(msg)
        
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            has_csv = False
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith('.csv'):
                    has_csv = True
                    break
            if has_csv:
                event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            has_csv = False
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith('.csv'):
                    has_csv = True
                    break
            if has_csv:
                event.acceptProposedAction()
                
    def dropEvent(self, event):
        files = []
        for url in event.mimeData().urls():
            if url.isLocalFile():
                file_path = url.toLocalFile()
                if file_path.lower().endswith('.csv'):
                    files.append(file_path)
                    
        if files:
            self.process_csv_files(files)

    def import_csvs(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self, "Select Cleaned CSV Files", "", "CSV Files (*.csv)", options=options)
        if files:
            self.process_csv_files(files)


    def export_to_spectral_library(self):
        try:
            from spectral_library import SpectralLibrary, SpectralLibraryEntry, SpectralLibraryDialog
            lib = SpectralLibrary(name="CSV Extracted Library")
            X, y, wv = self.prepare_data()
            if X is None or y is None or wv is None:
                QMessageBox.warning(self, "No Data", "No spectral data loaded to export.")
                return
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#17becf']
            unique_classes = np.unique(y)
            for i, c in enumerate(unique_classes):
                idx = (y == c)
                samples = X[idx]
                col = self.class_styles.get(c, {}).get('color', colors[i % len(colors)])
                entry = SpectralLibraryEntry.from_samples(str(c), wv, samples, color=col)
                lib.add_entry(entry)
            
            dialog = SpectralLibraryDialog(lib, parent_viewer=self, parent=self)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not launch Spectral Library:\n{e}")

    def open_format_guide(self):
        dialog = CSVFormatGuideDialog(self)
        dialog.exec_()
            
    def _parse_csv(self, file_path: str):
        try:
            import pandas as pd
            df_none = pd.read_csv(file_path, header=None)
            
            is_vertical = False
            first_col = pd.to_numeric(df_none.iloc[:, 0], errors='coerce')
            first_col_vals = first_col.dropna().values
            
            if len(first_col_vals) > 50 and len(first_col_vals) >= df_none.shape[0] * 0.8:
                if np.all(np.diff(first_col_vals) > 0) or np.all(np.diff(first_col_vals) < 0):
                    is_vertical = True
                    
            if is_vertical:
                if pd.isna(first_col.iloc[0]):
                    df_none = df_none.iloc[1:].reset_index(drop=True)
                wavelengths = pd.to_numeric(df_none.iloc[:, 0], errors='coerce').values
                df_numeric = df_none.iloc[:, 1:].copy()
                for col in df_numeric.columns:
                    df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                df_numeric = df_numeric.dropna(axis=1, how='all').fillna(0)
                return df_numeric.values.T, np.array(wavelengths), None

            df = pd.read_csv(file_path, header=0)
            classes = None
            class_cols = [c for c in df.columns if str(c).lower().strip() in ['class_label', 'class', 'label', 'target']]
            if class_cols:
                classes = df[class_cols[0]].astype(str).tolist()
            
            df_numeric = df.select_dtypes(include=[np.number])
            if df_numeric.empty:
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df_numeric = df.select_dtypes(include=[np.number])
                
            if df_numeric.empty:
                return None
                
            df_numeric = df_numeric.dropna(axis=1, how='all').fillna(0)
            
            valid_cols = []
            wavelengths = []
            for col in df_numeric.columns:
                try:
                    wavelengths.append(float(col))
                    valid_cols.append(col)
                except ValueError:
                    pass
            
            if valid_cols:
                df_numeric = df_numeric[valid_cols]
                wavelengths = np.array([float(c) for c in valid_cols])
            else:
                wavelengths = None
                
            return df_numeric.values, wavelengths, classes
            
        except Exception as e:
            self.log(f"CSV parsing failed: {e}")
            return None

    def _determine_class(self, filename: str) -> str:
        name_lower = filename.lower()
        if "health" in name_lower: return "Healthy"
        if "mild" in name_lower: return "Mild"
        if "severe" in name_lower: return "Severe"
        if "coinfect" in name_lower: return "Coinfected"
        return "Unknown"

    def process_csv_files(self, files):
        self.log(f"Loading {len(files)} files...")
        for f in files:
            res = self._parse_csv(f)
            if res is not None:
                data, wv, extracted_classes = res
                fname = os.path.basename(f)
                
                class_lbl = None
                if extracted_classes is not None and len(extracted_classes) == data.shape[0]:
                    unique_c = list(set(extracted_classes))
                    if len(unique_c) == 1:
                        class_lbl = unique_c[0]
                    else:
                        class_lbl = f"Mixed ({len(unique_c)})"
                else:
                    class_lbl = self._determine_class(fname)
                    extracted_classes = None
                    
                self.datasets.append({
                    'filename': fname,
                    'class_label': class_lbl,
                    'classes': extracted_classes,
                    'data': data,
                    'wavelengths': wv
                })
                self.log(f"Loaded {fname}: Class -> {class_lbl}")
                
        self.update_table()
        if len(self.datasets) > 0:
            self.btn_dr.setEnabled(True)
            self.btn_fs.setEnabled(True)
            self.btn_reg.setEnabled(True)
            self.btn_idx.setEnabled(True)
            self.btn_styles.setEnabled(True)
            self.btn_send_library.setEnabled(True)
            
    def update_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for ds in self.datasets:
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            item_fname = QTableWidgetItem(ds['filename'])
            item_fname.setFlags(item_fname.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, item_fname)
            self.table.setItem(row, 1, QTableWidgetItem(ds['class_label']))
            
            item_shape = QTableWidgetItem(f"{ds['data'].shape[0]} x {ds['data'].shape[1]}")
            item_shape.setFlags(item_shape.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, item_shape)
        self.table.blockSignals(False)

    def on_table_edit(self, row, col):
        if col == 1: 
            new_label = self.table.item(row, col).text()
            if 0 <= row < len(self.datasets):
                self.datasets[row]['class_label'] = new_label
                self.datasets[row]['classes'] = None 
                self.log(f"Manually updated class for {self.datasets[row]['filename']} to: {new_label}")

    def open_style_dialog(self):
        if not self.datasets: return
        _, y, _ = self.prepare_data()
        if y is None: return
        unique_classes = list(np.unique(y))
        
        dialog = PlotStyleDialog(unique_classes, self.class_styles, self)
        if dialog.exec_() == QDialog.Accepted:
            self.class_styles = dialog.get_styles()
            self.log("Plot styles updated.")
            if self.cmb_view.count() > 0:
                self.draw_current_plot() 

    def prepare_data(self):
        if not self.datasets: return None, None, None
        
        X = []
        y = []
        wv_out = self.datasets[0].get('wavelengths') if self.datasets else None
        
        for ds in self.datasets:
            X.append(ds['data'])
            if ds.get('classes') is not None:
                y.extend(ds['classes'])
            else:
                labels = [ds['class_label']] * ds['data'].shape[0]
                y.extend(labels)
            
        min_bands = min(d.shape[1] for d in X)
        X_trunc = [d[:, :min_bands] for d in X]
        
        X_combined = np.vstack(X_trunc)
        y_combined = np.array(y)
        
        if wv_out is not None:
            wv_out = wv_out[:min_bands]
        else:
            wv_out = np.arange(min_bands)
            
        return X_combined, y_combined, wv_out

    def apply_preprocessing(self, X):
        prep = self.cmb_prep.currentText()
        if prep == "No Preprocessing":
            return X
        elif prep == "Absorbance":
            return -np.log10(np.clip(X, 1e-10, None))
        elif prep == "Savitzky-Golay (11,2)":
            from scipy.signal import savgol_filter
            return savgol_filter(X, window_length=11, polyorder=2, axis=1)
        elif prep == "1st Derivative":
            from scipy.signal import savgol_filter
            return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)
        elif prep == "2nd Derivative":
            from scipy.signal import savgol_filter
            return savgol_filter(X, window_length=11, polyorder=3, deriv=2, axis=1)
        elif prep == "SNV":
            mean = np.mean(X, axis=1, keepdims=True)
            std = np.std(X, axis=1, keepdims=True)
            return (X - mean) / (std + 1e-8)
        return X

    def run_dr(self):
        X, y, wv = self.prepare_data()
        if X is None or len(X) == 0:
            QMessageBox.warning(self, "No Data", "Please load a CSV dataset first.")
            return
        method = self.cmb_dr.currentText()
        if "LDA" in method and len(np.unique(y)) < 2:
            QMessageBox.warning(
                self,
                "Multiple Classes Required",
                f"Linear Discriminant Analysis (LDA) is a supervised method requiring at least 2 distinct classes.\n\n"
                f"Current dataset contains only 1 class: '{np.unique(y)[0]}'.\n\n"
                f"Please load samples from 2 or more classes, or use unsupervised PCA / t-SNE."
            )
            return
        self.btn_dr.setEnabled(False)
        self.log("Starting Dimensionality Reduction...")
        
        if self.chk_aug.isChecked():
            noise = np.random.normal(0, 0.01 * np.std(X), X.shape)
            X = np.vstack([X, X + noise])
            y = np.concatenate([y, y])
            
        X_prep = self.apply_preprocessing(X)
        self.cached_raw_data = {'X': X_prep, 'y': y, 'wv': wv}
        method = self.cmb_dr.currentText()

        def task(emit_prog, X_in, y_in, method_in):
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X_in)
            
            if "PCA" in method_in:
                from sklearn.decomposition import PCA
                n_c = min(5, X_in.shape[0] - 1, X_in.shape[1])
                pca = PCA(n_components=max(2, n_c))
                scores = pca.fit_transform(X_scaled)
                
                # Chemometric Diagnostics: Hotelling T2 & Q-Residuals
                eigenvals = np.maximum(pca.explained_variance_, 1e-8)
                n_eval = min(3, scores.shape[1])
                t2 = np.sum((scores[:, :n_eval] ** 2) / eigenvals[:n_eval], axis=1)
                
                # Q-residuals (unmodeled variance)
                X_recon = pca.inverse_transform(scores)
                q_residuals = np.sum((X_scaled - X_recon) ** 2, axis=1)
                
                t2_95 = float(np.percentile(t2, 95))
                q_95 = float(np.percentile(q_residuals, 95))
                
                extra_pca = {
                    't2': t2,
                    'q_residuals': q_residuals,
                    't2_thresh_95': t2_95,
                    'q_thresh_95': q_95,
                    'eigenvals': eigenvals
                }
                return ("DR", "PCA", scores, pca.explained_variance_ratio_, pca.components_, extra_pca)
                
            elif "LDA" in method_in:
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                unique_c = np.unique(y_in)
                if len(unique_c) > 1:
                    n_c = min(2, len(unique_c) - 1, X_in.shape[1])
                    lda = LinearDiscriminantAnalysis(n_components=max(1, n_c))
                    scores = lda.fit_transform(X_scaled, y_in)
                    if scores.shape[1] == 1:
                        scores = np.column_stack((scores, np.zeros(len(scores))))
                    return ("DR", "LDA", scores, getattr(lda, "explained_variance_ratio_", np.array([0.8, 0.2])), None, {})
                else:
                    raise ValueError("LDA requires at least 2 distinct classes.")
                    
            elif "Correlation" in method_in:
                corr = np.corrcoef(X_in, rowvar=False)
                return ("DR", "Correlation Matrix", corr, None, None, {})
                
            elif "t-SNE" in method_in:
                from sklearn.manifold import TSNE
                perplexity = min(30, max(1, X_scaled.shape[0] - 1))
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                scores = tsne.fit_transform(X_scaled)
                return ("DR", "t-SNE", scores, None, None, {})
                
        worker = Worker(task, X_prep, y, method)
        worker.signals.result.connect(self.on_dr_complete)
        worker.signals.error.connect(lambda e: self.on_error(e))
        self.thread_pool.start(worker)

    def run_fs(self):
        X, y, wv = self.prepare_data()
        if X is None or len(X) == 0:
            QMessageBox.warning(self, "No Data", "Please load a CSV dataset first.")
            return
            
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            QMessageBox.warning(
                self,
                "Multiple Classes Required",
                f"Supervised feature selection requires at least 2 distinct classes (e.g. Healthy vs Disease) to rank discriminative spectral bands.\n\n"
                f"Current dataset contains only 1 class: '{unique_classes[0]}'.\n\n"
                f"Tip: Load two or more CSV datasets with different class labels or ensure your CSV file includes a class column with multiple labels."
            )
            return
            
        self.btn_fs.setEnabled(False)
        self.log("Starting Feature Selection...")
        
        if self.chk_aug.isChecked():
            noise = np.random.normal(0, 0.01 * np.std(X), X.shape)
            X = np.vstack([X, X + noise])
            y = np.concatenate([y, y])
            
        X_prep = self.apply_preprocessing(X)
        self.cached_raw_data = {'X': X_prep, 'y': y, 'wv': wv}
        method = self.cmb_fs.currentText()

        def task(emit_prog, X_in, y_in, method_in):
            from sklearn.preprocessing import StandardScaler
            X_scaled = StandardScaler().fit_transform(X_in)
            unique_classes = np.unique(y_in)
            if len(unique_classes) < 2:
                raise ValueError(
                    f"Supervised Feature Selection requires at least 2 distinct classes. "
                    f"Current dataset has only 1 class ('{unique_classes[0]}'). "
                    f"Please load multiple classes."
                )
            
            if method_in == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X_in, y_in)
                importances = clf.feature_importances_
            elif method_in == "Extra Trees":
                from sklearn.ensemble import ExtraTreesClassifier
                clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
                clf.fit(X_in, y_in)
                importances = clf.feature_importances_
            elif method_in == "SVM (Linear)":
                from sklearn.svm import SVC
                clf = SVC(kernel='linear')
                clf.fit(X_in, y_in)
                importances = np.mean(np.abs(clf.coef_), axis=0)
            elif method_in == "Logistic Regression (L1)":
                from sklearn.linear_model import LogisticRegression
                from sklearn.multiclass import OneVsRestClassifier
                if len(unique_classes) > 2:
                    base_lr = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
                    clf = OneVsRestClassifier(base_lr)
                    clf.fit(X_in, y_in)
                    coefs = np.array([est.coef_[0] for est in clf.estimators_])
                    importances = np.mean(np.abs(coefs), axis=0)
                else:
                    clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
                    clf.fit(X_in, y_in)
                    importances = np.mean(np.abs(clf.coef_), axis=0)
            elif method_in == "Neural Network (MLP)":
                from sklearn.neural_network import MLPClassifier
                clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
                clf.fit(X_scaled, y_in)
                importances = np.mean(np.abs(clf.coefs_[0]), axis=1)
                
            from sklearn.cross_decomposition import PLSRegression
            from sklearn.preprocessing import label_binarize
            unique_classes = np.unique(y_in)
            vips = None
            if len(unique_classes) > 1:
                Y_bin = label_binarize(y_in, classes=unique_classes)
                if len(unique_classes) == 2:
                    Y_bin = np.column_stack((1 - Y_bin, Y_bin))
                pls = PLSRegression(n_components=min(3, X_in.shape[1], X_in.shape[0]-1))
                pls.fit(X_scaled, Y_bin)
                t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
                p, h = w.shape
                vips = np.zeros((p,))
                s = np.diag(t.T @ t @ q.T @ q).ravel()
                total_s = np.sum(s)
                if total_s == 0:
                    vips = np.ones(p)
                else:
                    for i in range(p):
                        weight = np.array([ (w[i, j] / (np.linalg.norm(w[:, j]) + 1e-12))**2 for j in range(h) ])
                        vips[i] = np.sqrt(p * np.sum(s * weight) / total_s)
            
            return ("FS", method_in, importances, vips)
                
        worker = Worker(task, X_prep, y, method)
        worker.signals.result.connect(self.on_fs_complete)
        worker.signals.error.connect(lambda e: self.on_error(e))
        self.thread_pool.start(worker)

    def run_reg(self):
        self.btn_reg.setEnabled(False)
        self.log("Starting Regression Analysis...")
        X, y, wv = self.prepare_data()
        
        try:
            y_numeric = np.array(y, dtype=float)
        except ValueError:
            unique_classes = np.unique(y)
            dialog = NumericMappingDialog(unique_classes, self)
            if dialog.exec_() == QDialog.Accepted:
                mapping = dialog.get_mapping()
                y_numeric = np.array([mapping.get(str(lbl), 0.0) for lbl in y], dtype=float)
            else:
                self.btn_reg.setEnabled(True)
                self.log("Regression cancelled (no numeric mapping provided).")
                return
            
        if self.chk_aug.isChecked():
            noise = np.random.normal(0, 0.01 * np.std(X), X.shape)
            X = np.vstack([X, X + noise])
            y_numeric = np.concatenate([y_numeric, y_numeric])
            
        X_prep = self.apply_preprocessing(X)
        self.cached_raw_data = {'X': X_prep, 'y': y_numeric, 'wv': wv}
        method = self.cmb_reg.currentText()
        
        def task(emit_prog, X_in, y_in, method_in):
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score, mean_squared_error
            X_scaled = StandardScaler().fit_transform(X_in)
            
            if method_in == "PLSR":
                from sklearn.cross_decomposition import PLSRegression
                reg = PLSRegression(n_components=min(5, X_in.shape[1], X_in.shape[0]-1))
                reg.fit(X_scaled, y_in)
                y_pred = reg.predict(X_scaled).flatten()
                coef = reg.coef_.flatten()
            elif method_in == "SVR (Linear)":
                from sklearn.svm import SVR
                reg = SVR(kernel='linear')
                reg.fit(X_scaled, y_in)
                y_pred = reg.predict(X_scaled)
                coef = reg.coef_.flatten()
                
            r2 = r2_score(y_in, y_pred)
            rmse = np.sqrt(mean_squared_error(y_in, y_pred))
            return ("REG", method_in, y_pred, coef, r2, rmse)
            
        worker = Worker(task, X_prep, y_numeric, method)
        worker.signals.result.connect(self.on_reg_complete)
        worker.signals.error.connect(lambda e: self.on_error(e))
        self.thread_pool.start(worker)

    def run_idx(self):
        self.btn_idx.setEnabled(False)
        self.log("Computing Spectral Indices...")
        X, y, wv = self.prepare_data()
        
        if self.chk_aug.isChecked():
            noise = np.random.normal(0, 0.01 * np.std(X), X.shape)
            X = np.vstack([X, X + noise])
            y = np.concatenate([y, y])
            
        X_prep = self.apply_preprocessing(X)
        self.cached_raw_data = {'X': X_prep, 'y': y, 'wv': wv}
        idx_name = self.cmb_idx.currentText()
        
        def task(emit_prog, X_in, y_in, wv_in, index_name):
            def get_band(target):
                return np.argmin(np.abs(wv_in - target))
                
            if index_name == "NDVI":
                nir = X_in[:, get_band(800)]
                red = X_in[:, get_band(680)]
                vals = (nir - red) / (nir + red + 1e-8)
            elif index_name == "PRI":
                b531 = X_in[:, get_band(531)]
                b570 = X_in[:, get_band(570)]
                vals = (b531 - b570) / (b531 + b570 + 1e-8)
            elif index_name == "ARI":
                b550 = X_in[:, get_band(550)]
                b700 = X_in[:, get_band(700)]
                vals = (1 / (b550 + 1e-8)) - (1 / (b700 + 1e-8))
            
            return ("IDX", index_name, vals)
            
        worker = Worker(task, X_prep, y, wv, idx_name)
        worker.signals.result.connect(self.on_idx_complete)
        worker.signals.error.connect(lambda e: self.on_error(e))
        self.thread_pool.start(worker)

    def on_error(self, err):
        self.log(f"Error: {err[0]}\n{err[1]}")
        self.btn_dr.setEnabled(True)
        self.btn_fs.setEnabled(True)
        self.btn_reg.setEnabled(True)
        self.btn_idx.setEnabled(True)

    def on_dr_complete(self, res):
        method = res[1]
        scores = res[2]
        var_ratio = res[3]
        components = res[4]
        extra = res[5] if len(res) > 5 else {}
        
        self.cached_dr_data = {
            'method': method,
            'scores': scores,
            'var_ratio': var_ratio,
            'components': components
        }
        self.cached_dr_data.update(extra)
        
        self.cmb_view.blockSignals(True)
        self.cmb_view.clear()
        
        if method == "PCA":
            self.cmb_view.addItems([
                "PCA 2D Score Plot (95% Ellipses)",
                "PCA 3D Interactive Score Plot",
                "PCA Biplot (Scores + Vectors)",
                "PCA Multi-Component Loadings (PC1-PC4)",
                "Explained Variance (Scree Plot)",
                "Hotelling's T² vs Q-Residuals (Outlier Plot)",
                "Mean Spectra"
            ])
            self.cmb_view.setCurrentIndex(0)
        elif method == "LDA":
            self.cmb_view.addItems(["LDA Score Plot (Class Separation)", "Mean Spectra"])
            self.cmb_view.setCurrentIndex(0)
        elif method == "Correlation Matrix":
            self.cmb_view.addItems(["Wavelength Cross-Correlation Heatmap", "Mean Spectra"])
            self.cmb_view.setCurrentIndex(0)
        else:
            self.cmb_view.addItems(["Mean Spectra", f"{method} Score Plot"])
            self.cmb_view.setCurrentIndex(1)
            
        self.cmb_view.blockSignals(False)
        
        self.btn_export.setEnabled(True)
        self.btn_dr.setEnabled(True)
        self.log(f"{method} complete.")
        self.draw_current_plot()

    def on_fs_complete(self, res):
        _, method, importances, vips = res
        self.cached_fs_data = {
            'method': method,
            'importances': importances,
            'vips': vips
        }
        
        self.cmb_view.blockSignals(True)
        self.cmb_view.clear()
        self.cmb_view.addItems(["Mean Spectra", f"{method} Feature Importance", "PLS-DA VIP Scores", "Annotated Mean Spectra"])
        self.cmb_view.setCurrentIndex(1) 
        self.cmb_view.blockSignals(False)
        
        self.btn_export.setEnabled(True)
        self.btn_fs.setEnabled(True)
        self.log(f"{method} Feature Selection complete.")
        self.draw_current_plot()
        
    def on_reg_complete(self, res):
        _, method, y_pred, coef, r2, rmse = res
        self.cached_reg_data = {
            'method': method,
            'y_pred': y_pred,
            'coef': coef,
            'r2': r2,
            'rmse': rmse
        }
        
        self.cmb_view.blockSignals(True)
        self.cmb_view.clear()
        self.cmb_view.addItems(["Mean Spectra", f"{method} True vs Predicted", f"{method} Coefficients"])
        self.cmb_view.setCurrentIndex(1)
        self.cmb_view.blockSignals(False)
        
        self.btn_export.setEnabled(True)
        self.btn_reg.setEnabled(True)
        self.log(f"{method} Regression complete (R2: {r2:.3f}, RMSE: {rmse:.3f}).")
        self.draw_current_plot()
        
    def on_idx_complete(self, res):
        _, idx_name, vals = res
        self.cached_idx_data = {
            'name': idx_name,
            'vals': vals
        }
        
        self.cmb_view.blockSignals(True)
        self.cmb_view.clear()
        self.cmb_view.addItems(["Mean Spectra", f"{idx_name} Boxplot"])
        self.cmb_view.setCurrentIndex(1)
        self.cmb_view.blockSignals(False)
        
        self.btn_export.setEnabled(True)
        self.btn_idx.setEnabled(True)
        self.log(f"{idx_name} calculation complete.")
        self.draw_current_plot()

    def _style_ax(self, ax, title, xlabel, ylabel):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    def draw_single_plot(self, ax, plot_name):
        X = self.cached_raw_data.get('X')
        y = self.cached_raw_data.get('y')
        wv = self.cached_raw_data.get('wv')
        if X is None: return
        
        unique_classes = np.unique(y)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        if plot_name == "Mean Spectra":
            for i, c in enumerate(unique_classes):
                idx = (y == c)
                mean_spec = np.mean(X[idx], axis=0)
                std_spec = np.std(X[idx], axis=0)
                style = self.class_styles.get(c, {})
                c_name = style.get('name', str(c))
                c_color = style.get('color', colors[i % len(colors)])
                
                ax.plot(wv, mean_spec, label=c_name, color=c_color, linewidth=2.5)
                ax.fill_between(wv, mean_spec - std_spec, mean_spec + std_spec, alpha=0.2, color=c_color)
            self._style_ax(ax, "Mean Spectra", "Wavelength (nm)", "Intensity")
            ax.legend(loc='best', frameon=False)
            
        elif "2D Score Plot" in plot_name or plot_name == "t-SNE Score Plot":
            method = self.cached_dr_data['method']
            scores = self.cached_dr_data['scores']
            var_ratio = self.cached_dr_data.get('var_ratio')
            
            for i, c in enumerate(unique_classes):
                idx = (y == c)
                style = self.class_styles.get(c, {})
                c_name = style.get('name', str(c))
                c_color = style.get('color', colors[i % len(colors)])
                ax.scatter(scores[idx, 0], scores[idx, 1], label=c_name, alpha=0.85, c=c_color, s=50, edgecolors='k')
                
                # 95% Confidence Ellipse
                pts = scores[idx, :2]
                if len(pts) >= 4 and method == "PCA":
                    cov = np.cov(pts, rowvar=False)
                    mean_pt = np.mean(pts, axis=0)
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    vals, vecs = vals[order], vecs[:, order]
                    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                    w, h = 2 * 2.447 * np.sqrt(np.maximum(vals, 1e-8))
                    from matplotlib.patches import Ellipse
                    ell = Ellipse(xy=mean_pt, width=w, height=h, angle=theta,
                                  edgecolor=c_color, facecolor=c_color, alpha=0.15, linestyle='--', linewidth=2.0)
                    ax.add_patch(ell)
                
            xl = f"PC1 ({var_ratio[0]*100:.1f}%)" if (var_ratio is not None and len(var_ratio)>0) else "Dim 1"
            yl = f"PC2 ({var_ratio[1]*100:.1f}%)" if (var_ratio is not None and len(var_ratio)>1) else "Dim 2"
            self._style_ax(ax, f"{method} 2D Score Plot (with 95% Hotelling Ellipses)", xl, yl)
            ax.legend(loc='best', frameon=False)

        elif "Biplot" in plot_name:
            scores = self.cached_dr_data['scores']
            components = self.cached_dr_data['components']
            var_ratio = self.cached_dr_data.get('var_ratio')
            
            for i, c in enumerate(unique_classes):
                idx = (y == c)
                style = self.class_styles.get(c, {})
                c_name = style.get('name', str(c))
                c_color = style.get('color', colors[i % len(colors)])
                ax.scatter(scores[idx, 0], scores[idx, 1], label=c_name, alpha=0.4, c=c_color, s=35)
                
            # Top 6 contributing wavelengths
            magnitudes = np.sqrt(components[0]**2 + components[1]**2)
            top_i = np.argsort(magnitudes)[-6:]
            scale = min(np.max(np.abs(scores[:, 0])), np.max(np.abs(scores[:, 1]))) * 0.75 / (np.max(magnitudes) + 1e-8)
            for ti in top_i:
                vx = components[0, ti] * scale
                vy = components[1, ti] * scale
                ax.arrow(0, 0, vx, vy, color='#dc2626', width=scale*0.008, head_width=scale*0.035, length_includes_head=True, alpha=0.85)
                ax.text(vx*1.12, vy*1.12, f"{wv[ti]:.0f}nm", color='#b91c1c', fontweight='bold', fontsize=9)
                
            xl = f"PC1 ({var_ratio[0]*100:.1f}%)"
            yl = f"PC2 ({var_ratio[1]*100:.1f}%)"
            self._style_ax(ax, "PCA Biplot (Scores + Leading Spectral Vectors)", xl, yl)
            ax.legend(loc='best', frameon=False)

        elif "Multi-Component Loadings" in plot_name or plot_name == "PCA Loading Plot":
            components = self.cached_dr_data['components']
            var_ratio = self.cached_dr_data.get('var_ratio', [0, 0, 0, 0])
            for k in range(min(4, len(components))):
                lbl = f"PC{k+1} ({var_ratio[k]*100:.1f}%)"
                ax.plot(wv, components[k], label=lbl, linewidth=2.0)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.7)
            self._style_ax(ax, "PCA Multi-Component Loadings (PC1 - PC4)", "Wavelength (nm)", "Loading Coefficient")
            ax.legend(loc='best', frameon=False)

        elif "Explained Variance" in plot_name or "Scree" in plot_name:
            var_ratio = self.cached_dr_data['var_ratio']
            comps = range(1, len(var_ratio)+1)
            ax.bar(comps, var_ratio*100, color='#2ca02c', alpha=0.8, edgecolor='k', label='Individual Variance (%)')
            ax.plot(comps, np.cumsum(var_ratio*100), color='#d62728', marker='o', linewidth=2.2, label='Cumulative Variance (%)')
            
            # Kaiser / Average threshold line
            avg_var = 100.0 / len(var_ratio)
            ax.axhline(avg_var, color='#0284c7', linestyle='--', label=f'Avg Variance Threshold ({avg_var:.1f}%)')
            ax.axhline(90.0, color='#64748b', linestyle=':', label='90% Target')
            
            self._style_ax(ax, "Explained Variance (Scree & Cumulative Plot)", "Principal Component", "Variance Explained (%)")
            ax.set_xticks(comps)
            ax.legend(loc='best', frameon=False)

        elif "Outlier Plot" in plot_name or "Hotelling" in plot_name:
            t2 = self.cached_dr_data['t2']
            q = self.cached_dr_data['q_residuals']
            t2_95 = self.cached_dr_data['t2_thresh_95']
            q_95 = self.cached_dr_data['q_thresh_95']
            
            for i, c in enumerate(unique_classes):
                idx = (y == c)
                style = self.class_styles.get(c, {})
                c_name = style.get('name', str(c))
                c_color = style.get('color', colors[i % len(colors)])
                ax.scatter(t2[idx], q[idx], label=c_name, c=c_color, s=50, edgecolors='k', alpha=0.85)
                
            ax.axvline(t2_95, color='#dc2626', linestyle='--', linewidth=1.8, label=f'95% T² Limit ({t2_95:.1f})')
            ax.axhline(q_95, color='#7c3aed', linestyle='--', linewidth=1.8, label=f'95% Q Limit ({q_95:.2f})')
            ax.set_xscale('log')
            ax.set_yscale('log')
            self._style_ax(ax, "Hotelling's T² vs Q-Residuals Outlier Diagnostic", "Hotelling's T² (Within-Model Distance)", "Q-Residuals / SPE (Residual Distance)")
            ax.legend(loc='best', frameon=False)

        elif "LDA" in plot_name:
            scores = self.cached_dr_data['scores']
            for i, c in enumerate(unique_classes):
                idx = (y == c)
                style = self.class_styles.get(c, {})
                c_name = style.get('name', str(c))
                c_color = style.get('color', colors[i % len(colors)])
                ax.scatter(scores[idx, 0], scores[idx, 1], label=c_name, c=c_color, s=50, edgecolors='k', alpha=0.85)
                centroid = np.mean(scores[idx, :2], axis=0)
                ax.scatter(centroid[0], centroid[1], c=c_color, s=160, marker='X', edgecolors='k', linewidths=1.5)
            self._style_ax(ax, "Supervised Linear Discriminant Analysis (LDA)", "Linear Discriminant 1 (LD1)", "Linear Discriminant 2 (LD2)")
            ax.legend(loc='best', frameon=False)

        elif "Correlation Heatmap" in plot_name:
            corr = self.cached_dr_data['scores']
            im = ax.imshow(corr, cmap='coolwarm', vmin=-1.0, vmax=1.0, extent=[wv[0], wv[-1], wv[-1], wv[0]])
            self.figure.colorbar(im, ax=ax, label="Pearson Correlation (r)")
            self._style_ax(ax, "Wavelength-to-Wavelength Correlation Matrix", "Wavelength (nm)", "Wavelength (nm)")

        elif plot_name.endswith("Feature Importance"):
            method = self.cached_fs_data['method']
            importances = self.cached_fs_data['importances']
            ax.plot(wv, importances, color='#1f77b4', linewidth=2.0)
            self._style_ax(ax, f"{method} Feature Importance", "Wavelength (nm)", "Importance Score")

        elif plot_name == "PLS-DA VIP Scores":
            vips = self.cached_fs_data['vips']
            if vips is not None:
                ax.plot(wv, vips, color='#ff7f0e', linewidth=2.0)
                ax.axhline(y=1.0, color='r', linestyle='--', linewidth=1.5)
                self._style_ax(ax, "PLS-DA VIP Scores", "Wavelength (nm)", "VIP Score")
            else:
                ax.text(0.5, 0.5, "VIP Scores unavailable\n(Requires >= 2 classes)", ha='center', va='center')

        elif plot_name == "Annotated Mean Spectra":
            importances = self.cached_fs_data['importances']
            top_idx = np.argsort(importances)[-10:]
            for i, c in enumerate(unique_classes):
                idx = (y == c)
                mean_spec = np.mean(X[idx], axis=0)
                style = self.class_styles.get(c, {})
                c_name = style.get('name', str(c))
                c_color = style.get('color', colors[i % len(colors)])
                ax.plot(wv, mean_spec, label=c_name, color=c_color, linewidth=2.5)
                
            for ti in top_idx:
                ax.axvline(x=wv[ti], color='gray', linestyle='--', alpha=0.8)
                ax.text(wv[ti], ax.get_ylim()[1]*0.95, f"{wv[ti]:.1f}", rotation=90, fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0))
                
            self._style_ax(ax, "Annotated Mean Spectra (Top 10)", "Wavelength (nm)", "Intensity")
            ax.legend(loc='best', frameon=False)
            
        elif plot_name.endswith("True vs Predicted"):
            method = self.cached_reg_data['method']
            y_pred = self.cached_reg_data['y_pred']
            r2 = self.cached_reg_data['r2']
            rmse = self.cached_reg_data['rmse']
            
            ax.scatter(y, y_pred, alpha=0.7, color='#1f77b4', edgecolors='k')
            min_val = min(np.min(y), np.min(y_pred))
            max_val = max(np.max(y), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            ax.text(0.05, 0.95, f"R² = {r2:.3f}\nRMSE = {rmse:.3f}", transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            self._style_ax(ax, f"{method}: True vs Predicted", "True Value", "Predicted Value")

        elif plot_name.endswith("Coefficients"):
            method = self.cached_reg_data['method']
            coef = self.cached_reg_data['coef']
            ax.plot(wv, coef, color='#9467bd', linewidth=2.0)
            self._style_ax(ax, f"{method} Regression Coefficients", "Wavelength (nm)", "Coefficient Weight")

        elif plot_name.endswith("Boxplot"):
            idx_name = self.cached_idx_data['name']
            vals = self.cached_idx_data['vals']
            
            data_to_plot = [vals[y == c] for c in unique_classes]
            try:
                bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=[str(c) for c in unique_classes])
            except TypeError:
                bp = ax.boxplot(data_to_plot, patch_artist=True, labels=[str(c) for c in unique_classes])
            for patch, c in zip(bp['boxes'], unique_classes):
                style = self.class_styles.get(c, {})
                c_color = style.get('color', '#1f77b4')
                patch.set_facecolor(c_color)
                patch.set_alpha(0.7)
                
            self._style_ax(ax, f"{idx_name} Distribution", "Class", f"{idx_name} Value")

    def draw_current_plot(self):
        if self.cmb_view.count() == 0: return
        plot_name = self.cmb_view.currentText()
        self.figure.clear()
        
        if "3D Interactive" in plot_name:
            ax = self.figure.add_subplot(111, projection='3d')
            scores = self.cached_dr_data.get('scores')
            var_ratio = self.cached_dr_data.get('var_ratio', [0, 0, 0])
            y = self.cached_raw_data.get('y')
            if scores is not None and y is not None and scores.shape[1] >= 3:
                unique_classes = np.unique(y)
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
                for i, c in enumerate(unique_classes):
                    idx = (y == c)
                    style = self.class_styles.get(c, {})
                    c_name = style.get('name', str(c))
                    c_color = style.get('color', colors[i % len(colors)])
                    ax.scatter(scores[idx, 0], scores[idx, 1], scores[idx, 2], label=c_name, c=c_color, s=45, edgecolors='k', alpha=0.85)
                    centroid = np.mean(scores[idx, :3], axis=0)
                    ax.scatter(centroid[0], centroid[1], centroid[2], c=c_color, s=150, marker='X', edgecolors='white', linewidths=1.5)
                ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}%)")
                ax.set_zlabel(f"PC3 ({var_ratio[2]*100:.1f}%)")
                ax.set_title("3D PCA Score Scatter & Class Centroids (Interactive)")
                ax.legend(loc='best')
        else:
            ax = self.figure.add_subplot(111)
            self.draw_single_plot(ax, plot_name)
            
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw()

    def export_results(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        out_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory", options=options)
        if not out_dir: return
        
        out_path = Path(out_dir)
        import pandas as pd
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        
        temp_fig = Figure(figsize=(10, 6))
        temp_canvas = FigureCanvasAgg(temp_fig)
        
        plots_to_save = {
            "Mean Spectra": "Mean_Spectra.png",
        }
        
        if self.cached_dr_data:
            if self.cached_dr_data['method'] == "PCA":
                plots_to_save["PCA Score Plot"] = "PCA_Score_Plot.png"
                plots_to_save["PCA Loading Plot"] = "PCA_Loading_Plot.png"
                plots_to_save["Explained Variance (Scree Plot)"] = "PCA_Explained_Variance.png"
            else:
                plots_to_save["t-SNE Score Plot"] = "t-SNE_Score_Plot.png"
                
        if self.cached_fs_data:
            method = self.cached_fs_data['method']
            safe_method = method.replace(" ", "_").replace("(", "").replace(")", "")
            fs_filename = "RF_Feature_Importance.png" if method == "Random Forest" else f"{safe_method}_Feature_Importance.png"
            plots_to_save[f"{method} Feature Importance"] = fs_filename
            plots_to_save["PLS-DA VIP Scores"] = "PLSDA_VIP_Scores.png"
            plots_to_save["Annotated Mean Spectra"] = "Annotated_Mean_Spectra.png"
            
        if self.cached_reg_data:
            method = self.cached_reg_data['method']
            safe_method = method.replace(" ", "_").replace("(", "").replace(")", "")
            plots_to_save[f"{method} True vs Predicted"] = f"{safe_method}_True_vs_Predicted.png"
            plots_to_save[f"{method} Coefficients"] = f"{safe_method}_Coefficients.png"
            
        if self.cached_idx_data:
            idx_name = self.cached_idx_data['name']
            plots_to_save[f"{idx_name} Boxplot"] = f"{idx_name}_Boxplot.png"
            
        for plot_name, filename in plots_to_save.items():
            temp_fig.clear()
            ax = temp_fig.add_subplot(111)
            self.draw_single_plot(ax, plot_name)
            temp_fig.tight_layout()
            temp_fig.savefig(str(out_path / filename), dpi=900, bbox_inches='tight')
            
        # Export CSVs
        if self.cached_dr_data:
            d = self.cached_dr_data
            y = self.cached_raw_data['y']
            wv = self.cached_raw_data['wv']
            df = pd.DataFrame(d['scores'], columns=[f"Dim_{i+1}" for i in range(d['scores'].shape[1])])
            df['Class'] = y
            df.to_csv(out_path / f"{d['method']}_Scores.csv", index=False)
            if d['components'] is not None:
                df_loadings = pd.DataFrame(d['components'].T, columns=[f"PC_{i+1}" for i in range(d['components'].shape[0])])
                df_loadings.insert(0, 'Wavelength', wv)
                df_loadings.to_csv(out_path / "PCA_Loadings.csv", index=False)
                
        if self.cached_fs_data:
            d = self.cached_fs_data
            wv = self.cached_raw_data['wv']
            df_imp = pd.DataFrame({'Wavelength': wv, 'Importance': d['importances']})
            df_imp = df_imp.sort_values(by='Importance', ascending=False)
            safe_method = d['method'].replace(" ", "_").replace("(", "").replace(")", "")
            df_imp.to_csv(out_path / f"{safe_method}_Feature_Importances.csv", index=False)
            
            top10 = df_imp.head(10)
            with open(out_path / "Top_10_Features.txt", "w") as f:
                f.write(f"Top 10 Selected Wavelengths ({d['method']}):\n\n")
                for i, row in enumerate(top10.itertuples()):
                    f.write(f"{i+1}. Wavelength: {row.Wavelength:.2f} nm (Score: {row.Importance:.4f})\n")
                    
        # PDF Generation
        try:
            from reportlab.lib.pagesizes import letter, landscape
            from reportlab.pdfgen import canvas as pdf_canvas
            
            pdf_path = out_path / "Hypervision_Analytical_Report.pdf"
            c = pdf_canvas.Canvas(str(pdf_path), pagesize=landscape(letter))
            width, height = landscape(letter)
            
            c.setFont("Helvetica-Bold", 24)
            c.drawCentredString(width / 2.0, height - 150, "Hypervision Analytical Report")
            c.setFont("Helvetica", 14)
            c.drawCentredString(width / 2.0, height - 190, "Advanced Hyperspectral Data Analysis")
            c.setFont("Helvetica-Oblique", 12)
            c.drawCentredString(width / 2.0, height - 230, "Created by Lalit Pandurang Patil, PhD Scholar")
            c.drawCentredString(width / 2.0, height - 250, "Division of Plant Pathology, Indian Agricultural Research Institute, New Delhi")
            
            for plot_name, filename in plots_to_save.items():
                img_path = str(out_path / filename)
                if os.path.exists(img_path):
                    c.showPage()
                    c.setFont("Helvetica-Bold", 16)
                    c.drawString(50, height - 50, plot_name)
                    c.drawImage(img_path, 50, 50, width=width-100, height=height-150, preserveAspectRatio=True)
            
            c.save()
            self.log(f"PDF Report successfully generated: {pdf_path}")
        except Exception as e:
            self.log(f"Warning: Could not generate PDF report. ({e})")
                    
        self.log(f"Exported all requested plots and data to {out_dir}")
        QMessageBox.information(self, "Success", f"Analysis Exported Successfully!\nPDF Report and CSV Data saved to:\n{out_dir}")
