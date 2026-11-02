import re

with open("spectral_core.py", "r") as f:
    content = f.read()

# 1. Add _resolve_image_path
resolve_func = """def _resolve_image_path(header_path: str) -> str | None:
    from pathlib import Path
    p = Path(header_path)
    if p.suffix.lower() == '.hdr':
        base = p.with_suffix('')
        if base.exists():
            return str(base)
        for ext in ('.raw', '.dat', '.img', '.bin'):
            if base.with_suffix(ext).exists():
                return str(base.with_suffix(ext))
    return None

class EnviDocument:"""
content = content.replace("class EnviDocument:", resolve_func)

# 2. Fix envi.open calls
content = content.replace("self._data_spy = envi.open(self.data_header)", "self._data_spy = envi.open(self.data_header, image=_resolve_image_path(self.data_header))")
content = content.replace("spy = envi_module.open(header)", "spy = envi_module.open(header, image=_resolve_image_path(header))")

# 3. Fix Wavelength count check
wavelength_check = """    if wavelengths.size != n_bands:
        if wavelengths.size > n_bands:
            warnings.append(f"WAVELENGTH_COUNT_TRUNCATED_FROM_{wavelengths.size}_TO_{n_bands}")
            wavelengths = wavelengths[:n_bands]
        else:
            raise InputContractError(
                f"Wavelength count {wavelengths.size} does not match cube band count {n_bands}."
            )"""
content = re.sub(
    r"    if wavelengths.size != n_bands:\n        raise InputContractError\(\n            f\"Wavelength count \{wavelengths.size\} does not match cube band count \{n_bands\}\.\"\n        \)",
    wavelength_check,
    content
)

# 4. Fix PCA pixels
content = content.replace("if valid.sum() < max(10, bands):", "if valid.sum() < 10:")

# 5. Add fixed_rois to run_batch signature
content = content.replace(
    '    extraction_mode: str = "auto_foreground",\n    progress: Optional[Callable[[int, str], None]] = None,',
    '    extraction_mode: str = "auto_foreground",\n    fixed_rois: Optional[list] = None,\n    progress: Optional[Callable[[int, str], None]] = None,'
)

# 6. Add fixed_rois extraction logic in run_batch
fixed_roi_logic = """            foreground_fraction: Optional[float] = None
            if extraction_mode == "fixed_roi":
                if not fixed_rois:
                    raise SpectralViewError("Fixed ROI extraction requested but no ROI coordinates provided.")
                for roi_idx, (x1, y1, x2, y2, label) in enumerate(fixed_rois):
                    mask = np.zeros((document.height, document.width), dtype=bool)
                    y1_c = max(0, min(y1, document.height))
                    y2_c = max(0, min(y2, document.height))
                    x1_c = max(0, min(x1, document.width))
                    x2_c = max(0, min(x2, document.width))
                    mask[y1_c:y2_c, x1_c:x2_c] = True
                    if not np.any(mask):
                        raise SpectralViewError(f"Fixed ROI '{label}' is outside the image bounds for this sample.")
                    spectrum, qc = document.mean_spectrum(mask, processing_config.chunk_rows)
                    if processing_config.preprocessing == "absorbance" and qc.units != "reflectance":
                        raise SpectralViewError(f"Absorbance preprocessing requires reflectance.")
                    spectrum = preprocess_spectra(
                        spectrum, processing_config.preprocessing, document.wavelengths, processing_config.sg_window, processing_config.sg_polyorder
                    )
                    sample_results.append(BatchSampleResult(
                        sample_id=f"{sample_id}_ROI_{label}", folder=folder, status="SUCCESS", message=";",
                        calibration_mode=qc.mode, units=qc.units, n_bands=document.n_bands, foreground_fraction=1.0,
                        wavelengths=document.wavelengths.copy(), spectrum=np.asarray(spectrum, dtype=np.float32).copy()
                    ))
                provenance.append(document.provenance())
                continue
            
            if extraction_mode == "auto_foreground":"""

content = content.replace(
    '            foreground_fraction: Optional[float] = None\n            if extraction_mode == "auto_foreground":',
    fixed_roi_logic
)

with open("spectral_core.py", "w") as f:
    f.write(content)
