# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['spectral_gui_2026_07_16.py'],
    pathex=['.'],
    binaries=[],
    datas=[('icons', 'icons')],
    hiddenimports=['datacube_segmentation_studio', 'scientific_dossier_generator', 'chemometrics_advanced_studio', 'spatial_classification_studio', 'sklearn.utils._typedefs', 'reportlab', 'docx', 'scipy.special.cython_special', 'openpyxl', 'spectral_library', 'spectral_library_dialog', 'gmail_auth_dialog', 'preprocessing_studio', 'model_visualization_studio', 'sklearn.calibration', 'sklearn.discriminant_analysis', 'sklearn.gaussian_process', 'mpl_toolkits.mplot3d'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide6', 'PyQt6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Hypervision_Win',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons/Hypervision_Win.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Hypervision_Win',
)
