from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QGroupBox, QSpinBox, QProgressBar, QTabWidget, QWidget, QFormLayout, QDoubleSpinBox, QCheckBox
from PyQt5.QtCore import Qt

class OpenCubeDialog(QDialog):
    def __init__(self, parent, document, title="Open Cube", action_text="Open"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(600, 500)
        self.document = document
        
        layout = QVBoxLayout(self)
        
        # Header and Data files
        form = QFormLayout()
        self.header_edit = QLineEdit(str(document.data_header))
        self.header_edit.setReadOnly(True)
        self.data_edit = QLineEdit(str(document.data_file))
        self.data_edit.setReadOnly(True)
        form.addRow("Header file:", self.header_edit)
        form.addRow("Data file:", self.data_edit)
        layout.addLayout(form)
        
        # Wavelengths Group
        wave_group = QGroupBox("Wavelengths:")
        wave_layout = QHBoxLayout(wave_group)
        
        # Left controls
        left_layout = QVBoxLayout()
        sel_layout = QHBoxLayout()
        sel_layout.addWidget(QLabel("Selected:"))
        self.sel_count = QSpinBox()
        self.sel_count.setRange(0, 10000)
        self.sel_count.setReadOnly(True)
        sel_layout.addWidget(self.sel_count)
        left_layout.addLayout(sel_layout)
        
        from_to_layout = QFormLayout()
        self.from_spin = QDoubleSpinBox()
        self.from_spin.setRange(0, 10000)
        self.to_spin = QDoubleSpinBox()
        self.to_spin.setRange(0, 10000)
        if document.wavelengths is not None and len(document.wavelengths) > 0:
            self.from_spin.setValue(float(document.wavelengths[0]))
            self.to_spin.setValue(float(document.wavelengths[-1]))
        from_to_layout.addRow("From:", self.from_spin)
        from_to_layout.addRow("To:", self.to_spin)
        left_layout.addLayout(from_to_layout)
        
        btn_select = QPushButton("Select")
        btn_select.clicked.connect(self.select_all)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self.clear_all)
        btn_select_rgb = QPushButton("Select RGB")
        btn_select_rgb.clicked.connect(self.select_rgb)
        left_layout.addWidget(btn_select)
        left_layout.addWidget(btn_clear)
        left_layout.addStretch()
        left_layout.addWidget(btn_select_rgb)
        
        wave_layout.addLayout(left_layout)
        
        # Right list
        self.wave_list = QListWidget()
        if document.wavelengths is not None:
            for w in document.wavelengths:
                item = QListWidgetItem(f"{w:.3f} nm")
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.wave_list.addItem(item)
        self.wave_list.itemChanged.connect(self.update_count)
        wave_layout.addWidget(self.wave_list)
        
        layout.addWidget(wave_group)
        
        # Progress and Action
        bottom_layout = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)
        bottom_layout.addWidget(self.progress)
        
        self.btn_action = QPushButton(action_text)
        self.btn_action.clicked.connect(self.do_action)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        bottom_layout.addWidget(self.btn_action)
        bottom_layout.addWidget(btn_cancel)
        layout.addLayout(bottom_layout)
        
        self.update_count()
        
    def update_count(self):
        count = sum(1 for i in range(self.wave_list.count()) if self.wave_list.item(i).checkState() == Qt.Checked)
        self.sel_count.setValue(count)
        
    def select_all(self):
        for i in range(self.wave_list.count()):
            self.wave_list.item(i).setCheckState(Qt.Checked)
            
    def clear_all(self):
        for i in range(self.wave_list.count()):
            self.wave_list.item(i).setCheckState(Qt.Unchecked)
            
    def select_rgb(self):
        self.clear_all()
        if self.document.wavelengths is None: return
        import numpy as np
        waves = np.array(self.document.wavelengths)
        for target in [670.0, 550.0, 480.0]: # approx RGB
            idx = (np.abs(waves - target)).argmin()
            self.wave_list.item(idx).setCheckState(Qt.Checked)
            
    def do_action(self):
        self.accept()

class OrthoRectificationDialog(QDialog):
    def __init__(self, parent, document):
        super().__init__(parent)
        self.setWindowTitle("Ortho Rectification")
        self.resize(800, 600)
        self.document = document
        
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # Settings Tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        
        form = QFormLayout()
        form.addRow("Frame Index File:", QLineEdit("frameIndex.txt"))
        form.addRow("GPS/IMU File:", QLineEdit("imu_gps.txt"))
        form.addRow("DEM:", QLineEdit("float.hdr"))
        form.addRow("Resolution (m):", QLineEdit("0.120"))
        settings_layout.addLayout(form)
        
        group_layout = QHBoxLayout()
        lens_group = QGroupBox("Lens")
        lens_form = QFormLayout(lens_group)
        lens_form.addRow("EFL (mm):", QDoubleSpinBox())
        lens_form.addRow("Pixel pitch (um):", QDoubleSpinBox())
        group_layout.addWidget(lens_group)
        
        sensor_group = QGroupBox("Sensor position")
        sensor_form = QFormLayout(sensor_group)
        sensor_form.addRow("Alpha (deg):", QDoubleSpinBox())
        sensor_form.addRow("Beta (deg):", QDoubleSpinBox())
        sensor_form.addRow("Gamma (deg):", QDoubleSpinBox())
        group_layout.addWidget(sensor_group)
        settings_layout.addLayout(group_layout)
        
        offset_layout = QHBoxLayout()
        gps_group = QGroupBox("GPS/IMU Offsets")
        gps_form = QFormLayout(gps_group)
        gps_form.addRow("Roll (deg):", QDoubleSpinBox())
        gps_form.addRow("Pitch (deg):", QDoubleSpinBox())
        gps_form.addRow("Yaw (deg):", QDoubleSpinBox())
        offset_layout.addWidget(gps_group)
        settings_layout.addLayout(offset_layout)
        
        settings_layout.addStretch()
        tabs.addTab(settings_tab, "Settings")
        tabs.addTab(QWidget(), "Preview")
        tabs.addTab(QWidget(), "Create")
        
        layout.addWidget(tabs)
        
        bottom = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setValue(0)
        bottom.addWidget(self.progress)
        btn_start = QPushButton("Start O.R.")
        btn_start.clicked.connect(self.accept)
        bottom.addWidget(btn_start)
        layout.addLayout(bottom)
