"""Scientific core for SpectralView 2.1.12.

The module deliberately contains no Qt dependencies. GUI, batch and tests use the
same calibration, wavelength, segmentation and classification functions.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.decomposition import PCA

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - dependency guard
    savgol_filter = None

VERSION = "2.1.12"
RELEASE_STATUS = "STABLE_CANDIDATE_NOT_EXTERNALLY_VALIDATED"


class SpectralViewError(RuntimeError):
    """Base exception for governed scientific failures."""


class InputContractError(SpectralViewError):
    """Raised when input files or metadata violate the input contract."""


class CalibrationError(SpectralViewError):
    """Raised when calibration cannot be performed safely."""


class WavelengthCompatibilityError(SpectralViewError):
    """Raised when spectra cannot be compared on a common wavelength grid."""


@dataclass(frozen=True)
class CalibrationConfig:
    input_is_reflectance: bool = False
    epsilon: float = 1e-6
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    reject_nonfinite_fraction: float = 0.20


@dataclass(frozen=True)
class ProcessingConfig:
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    preprocessing: str = "none"
    sg_window: int = 11
    sg_polyorder: int = 2
    sam_threshold_deg: float = 12.0
    chunk_rows: int = 32
    segmentation_min_area_px: int = 25
    segmentation_polarity: str = "auto"
    segmentation_morphology_size: int = 3
    preview_max_dimension: int = 1000


@dataclass
class DiscoveryResult:
    folder: str
    data_header: Optional[str]
    dark_header: Optional[str]
    white_header: Optional[str]
    warnings: List[str] = field(default_factory=list)


@dataclass
class CalibrationQC:
    mode: str
    units: str
    finite_fraction: float
    low_denominator_fraction: float
    below_clip_fraction: float
    above_clip_fraction: float
    warnings: List[str] = field(default_factory=list)


@dataclass
class SegmentationObject:
    object_id: int
    area_px: int
    centroid_x: float
    centroid_y: float
    x: int
    y: int
    width: int
    height: int
    mean_spectrum: np.ndarray


@dataclass
class SegmentationResult:
    mask: np.ndarray
    labels: np.ndarray
    objects: List[SegmentationObject]
    foreground_fraction: float
    border_fraction: float
    polarity: str
    warnings: List[str] = field(default_factory=list)


@dataclass
class SAMResult:
    class_map: np.ndarray
    angle_map_deg: np.ndarray
    valid_fraction: float
    unknown_fraction: float


@dataclass
class BatchSampleResult:
    sample_id: str
    folder: str
    status: str
    message: str
    calibration_mode: str = ""
    units: str = ""
    n_bands: int = 0
    foreground_fraction: Optional[float] = None
    wavelengths: Optional[np.ndarray] = None
    spectrum: Optional[np.ndarray] = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: os.PathLike[str] | str, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding=encoding, delete=False, dir=path.parent, prefix=f".{path.name}.") as fh:
        fh.write(text)
        tmp = Path(fh.name)
    os.replace(tmp, path)


def atomic_write_csv(path: os.PathLike[str] | str, rows: Iterable[Sequence[Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", delete=False, dir=path.parent, prefix=f".{path.name}.") as fh:
        writer = csv.writer(fh)
        for row in rows:
            writer.writerow(row)
        tmp = Path(fh.name)
    os.replace(tmp, path)


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_header_name(path: Path) -> str:
    return path.stem.lower().replace("-", "_").replace(" ", "_")


def discover_envi_folder(folder: os.PathLike[str] | str) -> DiscoveryResult:
    """Deterministically identify data, dark and white ENVI headers.

    Filename discovery is explicit and conservative. Ambiguous data headers are
    reported rather than silently interpreted as the same acquisition.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise InputContractError(f"Not a directory: {folder_path}")

    headers = sorted(folder_path.glob("*.hdr"), key=lambda p: p.name.lower())
    warnings: List[str] = []
    if not headers:
        return DiscoveryResult(str(folder_path), None, None, None, ["NO_ENVI_HEADERS"])

    dark_tokens = ("dark", "black", "shutter")
    white_tokens = ("white", "reference", "ref_white", "panel")
    data_tokens = ("data", "raw", "sample", "capture", "image", "cube")

    darks: List[Path] = []
    whites: List[Path] = []
    candidates: List[Tuple[int, Path]] = []

    for header in headers:
        name = _normalise_header_name(header)
        if any(token in name for token in dark_tokens):
            darks.append(header)
            continue
        if any(token in name for token in white_tokens):
            whites.append(header)
            continue
        score = sum(2 for token in data_tokens if token in name)
        if name in {"data", "raw", "sample"}:
            score += 5
        candidates.append((score, header))

    data_header: Optional[Path] = None
    if candidates:
        candidates.sort(key=lambda item: (-item[0], item[1].name.lower()))
        best_score = candidates[0][0]
        best = [p for score, p in candidates if score == best_score]
        data_header = best[0]
        if len(best) > 1:
            warnings.append("AMBIGUOUS_DATA_HEADERS:" + ",".join(p.name for p in best))

    if len(darks) > 1:
        warnings.append("MULTIPLE_DARK_HEADERS:" + ",".join(p.name for p in darks))
    if len(whites) > 1:
        warnings.append("MULTIPLE_WHITE_HEADERS:" + ",".join(p.name for p in whites))

    return DiscoveryResult(
        folder=str(folder_path),
        data_header=str(data_header) if data_header else None,
        dark_header=str(darks[0]) if darks else None,
        white_header=str(whites[0]) if whites else None,
        warnings=warnings,
    )


def find_dataset_folders(parent: os.PathLike[str] | str) -> List[str]:
    parent_path = Path(parent)
    found: List[str] = []
    for root, _dirs, files in os.walk(parent_path):
        if any(name.lower().endswith(".hdr") for name in files):
            result = discover_envi_folder(root)
            if result.data_header:
                found.append(str(Path(root)))
    return sorted(set(found), key=lambda p: p.lower())


def parse_wavelengths(metadata: Mapping[str, Any], n_bands: int) -> Tuple[np.ndarray, bool, List[str]]:
    warnings: List[str] = []
    raw = metadata.get("wavelength")
    if raw is None or raw == []:
        warnings.append("WAVELENGTH_METADATA_MISSING_USING_BAND_INDEX")
        return np.arange(n_bands, dtype=np.float64), False, warnings

    if isinstance(raw, str):
        raw = raw.strip().strip("{}").split(",")
    try:
        wavelengths = np.asarray([float(str(value).strip()) for value in raw], dtype=np.float64)
    except Exception as exc:
        raise InputContractError(f"Invalid wavelength metadata: {exc}") from exc

    units_raw = metadata.get("wavelength units", metadata.get("wavelength_units", ""))
    units = str(units_raw).strip().lower().replace("μ", "u").replace("µ", "u")
    if units in {"um", "micrometer", "micrometers", "micrometre", "micrometres", "micron", "microns"}:
        wavelengths = wavelengths * 1000.0
        warnings.append("WAVELENGTHS_CONVERTED_FROM_MICROMETERS_TO_NM")
    elif units in {"nm", "nanometer", "nanometers", "nanometre", "nanometres"}:
        pass
    elif units:
        raise InputContractError(f"Unsupported wavelength units: {units_raw}")
    else:
        warnings.append("WAVELENGTH_UNITS_MISSING_VALUES_TREATED_AS_NM")

    if wavelengths.size != n_bands:
        if wavelengths.size > n_bands:
            warnings.append(f"WAVELENGTH_COUNT_TRUNCATED_FROM_{wavelengths.size}_TO_{n_bands}")
            wavelengths = wavelengths[:n_bands]
        else:
            raise InputContractError(
                f"Wavelength count {wavelengths.size} does not match cube band count {n_bands}."
            )
    if not np.all(np.isfinite(wavelengths)):
        raise InputContractError("Wavelength metadata contains NaN or infinite values.")
    if len(np.unique(wavelengths)) != len(wavelengths):
        raise InputContractError("Wavelength metadata contains duplicate values.")
    if np.all(np.diff(wavelengths) < 0):
        warnings.append("WAVELENGTH_ORDER_DESCENDING")
    elif not np.all(np.diff(wavelengths) > 0):
        raise InputContractError("Wavelengths must be strictly monotonic.")
    return wavelengths, True, warnings


def nearest_band(wavelengths: np.ndarray, target_nm: float, tolerance_nm: Optional[float] = None) -> int:
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    idx = int(np.argmin(np.abs(wavelengths - target_nm)))
    if tolerance_nm is not None and abs(float(wavelengths[idx]) - target_nm) > tolerance_nm:
        raise WavelengthCompatibilityError(
            f"No band within {tolerance_nm:g} nm of {target_nm:g} nm; nearest is {wavelengths[idx]:g} nm."
        )
    return idx


def _finite_metrics(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.isfinite(values).sum() / values.size)


def calibrate_array(
    raw: np.ndarray,
    dark: Optional[np.ndarray],
    white: Optional[np.ndarray],
    config: CalibrationConfig,
) -> Tuple[np.ndarray, CalibrationQC]:
    """Calibrate one array or region using one governed formula."""
    raw_f = np.asarray(raw, dtype=np.float32)
    if raw_f.ndim < 1:
        raise CalibrationError("Raw data must contain at least one dimension.")

    warnings: List[str] = []
    low_denominator_fraction = 0.0

    if config.input_is_reflectance:
        calibrated = raw_f.copy()
        mode = "DECLARED_REFLECTANCE_INPUT"
        units = "reflectance"
        if dark is not None or white is not None:
            warnings.append("REFERENCES_IGNORED_INPUT_DECLARED_REFLECTANCE")
    elif white is not None:
        dark_f = np.zeros_like(raw_f, dtype=np.float32) if dark is None else np.asarray(dark, dtype=np.float32)
        white_f = np.asarray(white, dtype=np.float32)
        try:
            denominator = white_f - dark_f
            numerator = raw_f - dark_f
        except ValueError as exc:
            raise CalibrationError(f"Reference arrays are not broadcast-compatible: {exc}") from exc
        low = np.abs(denominator) <= config.epsilon
        low_denominator_fraction = float(np.mean(low))
        safe_denominator = np.where(low, np.nan, denominator)
        calibrated = numerator / safe_denominator
        mode = "RAW_DARK_WHITE_REFLECTANCE" if dark is not None else "RAW_WHITE_REFLECTANCE_NO_DARK"
        units = "reflectance"
        if dark is None:
            warnings.append("WHITE_CALIBRATION_WITHOUT_DARK_REFERENCE")
    elif dark is not None:
        try:
            calibrated = raw_f - np.asarray(dark, dtype=np.float32)
        except ValueError as exc:
            raise CalibrationError(f"Dark reference is not broadcast-compatible: {exc}") from exc
        mode = "DARK_CORRECTED_COUNTS"
        units = "dark-corrected counts"
        warnings.append("WHITE_REFERENCE_MISSING_OUTPUT_IS_NOT_REFLECTANCE")
    else:
        calibrated = raw_f.copy()
        mode = "RAW_COUNTS_UNCALIBRATED"
        units = "raw counts"
        warnings.append("DARK_AND_WHITE_REFERENCES_MISSING")

    finite_fraction = _finite_metrics(calibrated)
    below_clip_fraction = 0.0
    above_clip_fraction = 0.0
    finite_mask = np.isfinite(calibrated)

    if config.clip_min is not None and np.any(finite_mask):
        below_clip_fraction = float(np.mean(calibrated[finite_mask] < config.clip_min))
    if config.clip_max is not None and np.any(finite_mask):
        above_clip_fraction = float(np.mean(calibrated[finite_mask] > config.clip_max))

    if units == "reflectance":
        if config.clip_min is not None:
            calibrated = np.where(np.isfinite(calibrated), np.maximum(calibrated, config.clip_min), calibrated)
        if config.clip_max is not None:
            calibrated = np.where(np.isfinite(calibrated), np.minimum(calibrated, config.clip_max), calibrated)

    nonfinite_fraction = 1.0 - finite_fraction
    if nonfinite_fraction > config.reject_nonfinite_fraction:
        raise CalibrationError(
            f"Non-finite calibrated fraction {nonfinite_fraction:.3f} exceeds allowed "
            f"{config.reject_nonfinite_fraction:.3f}."
        )
    if low_denominator_fraction > 0:
        warnings.append(f"LOW_WHITE_MINUS_DARK_DENOMINATOR_FRACTION={low_denominator_fraction:.6f}")

    qc = CalibrationQC(
        mode=mode,
        units=units,
        finite_fraction=finite_fraction,
        low_denominator_fraction=low_denominator_fraction,
        below_clip_fraction=below_clip_fraction,
        above_clip_fraction=above_clip_fraction,
        warnings=warnings,
    )
    return calibrated.astype(np.float32, copy=False), qc


def apply_snv(values: np.ndarray, axis: int = -1) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mean = np.nanmean(arr, axis=axis, keepdims=True)
    std = np.nanstd(arr, axis=axis, keepdims=True)
    std = np.where(std <= np.finfo(np.float32).eps, 1.0, std)
    return (arr - mean) / std


def preprocess_spectra(
    values: np.ndarray,
    method: str,
    wavelengths: Optional[np.ndarray] = None,
    sg_window: int = 11,
    sg_polyorder: int = 2,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    key = method.strip().lower().replace(" ", "_")
    if key in {"", "none", "raw"}:
        return arr.copy()
    if key == "snv":
        return apply_snv(arr, axis=-1)
    if key in {"absorbance", "log_1_r"}:
        safe = np.clip(arr, 1e-8, None)
        return -np.log10(safe)
    if key in {"savitzky_golay", "sg_smooth", "savgol"}:
        if savgol_filter is None:
            raise SpectralViewError("SciPy is required for Savitzky-Golay preprocessing.")
        window = _validated_sg_window(arr.shape[-1], sg_window, sg_polyorder)
        return savgol_filter(arr, window_length=window, polyorder=sg_polyorder, axis=-1, mode="interp").astype(np.float32)
    if key in {"first_derivative", "derivative_1"}:
        if wavelengths is None:
            return np.gradient(arr, axis=-1).astype(np.float32)
        return np.gradient(arr, np.asarray(wavelengths, dtype=np.float64), axis=-1).astype(np.float32)
    if key in {"second_derivative", "derivative_2"}:
        if wavelengths is None:
            first = np.gradient(arr, axis=-1)
            return np.gradient(first, axis=-1).astype(np.float32)
        wl = np.asarray(wavelengths, dtype=np.float64)
        first = np.gradient(arr, wl, axis=-1)
        return np.gradient(first, wl, axis=-1).astype(np.float32)
    raise SpectralViewError(f"Unsupported preprocessing method: {method}")


def _validated_sg_window(n_bands: int, requested: int, polyorder: int) -> int:
    if n_bands <= polyorder + 1:
        raise SpectralViewError("Too few bands for requested Savitzky-Golay polynomial order.")
    window = min(int(requested), n_bands if n_bands % 2 == 1 else n_bands - 1)
    if window % 2 == 0:
        window -= 1
    window = max(window, polyorder + 2 + ((polyorder + 2) % 2 == 0))
    if window > n_bands:
        window = n_bands if n_bands % 2 == 1 else n_bands - 1
    if window <= polyorder:
        raise SpectralViewError("Invalid Savitzky-Golay window/polyorder combination.")
    return window


def robust_uint8(channel: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0, gamma: float = 1.0) -> np.ndarray:
    arr = np.asarray(channel, dtype=np.float32)
    finite = np.isfinite(arr)
    output = np.zeros(arr.shape, dtype=np.uint8)
    if not np.any(finite):
        return output
    low, high = np.nanpercentile(arr[finite], [low_percentile, high_percentile])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return output
    norm = np.clip((arr - low) / (high - low), 0.0, 1.0)
    if gamma <= 0:
        raise ValueError("Gamma must be greater than zero.")
    norm = np.power(norm, 1.0 / gamma)
    output[finite] = np.round(norm[finite] * 255.0).astype(np.uint8)
    return output


def compose_rgb(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    requested_nm: Tuple[float, float, float],
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
    gamma: float = 1.0,
    tolerance_nm: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    arr = np.asarray(cube)
    indices = tuple(nearest_band(wavelengths, value, tolerance_nm) for value in requested_nm)
    channels = [robust_uint8(arr[:, :, idx], low_percentile, high_percentile, gamma) for idx in indices]
    return np.stack(channels, axis=2), indices


def pca_rgb(
    cube: np.ndarray,
    max_fit_samples: int = 100_000,
    random_state: int = 0,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise SpectralViewError("PCA RGB requires a 3-D cube with at least three bands.")
    flat = arr.reshape(-1, arr.shape[2])
    valid = np.all(np.isfinite(flat), axis=1)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < 3:
        raise SpectralViewError("Too few finite pixels for PCA rendering.")
    rng = np.random.default_rng(random_state)
    if valid_indices.size > max_fit_samples:
        fit_indices = rng.choice(valid_indices, size=max_fit_samples, replace=False)
    else:
        fit_indices = valid_indices
    fit_values = flat[fit_indices]
    centered = fit_values - np.mean(fit_values, axis=0, keepdims=True)
    numerical_rank = int(np.linalg.matrix_rank(centered))
    if numerical_rank == 0:
        grey = robust_uint8(np.nanmean(arr, axis=2), low_percentile, high_percentile)
        return np.stack([grey, grey, grey], axis=2)
    n_components = min(3, numerical_rank, arr.shape[2])
    model = PCA(n_components=n_components, random_state=random_state)
    model.fit(fit_values)
    transformed = np.full((flat.shape[0], 3), np.nan, dtype=np.float32)
    transformed[valid, :n_components] = model.transform(flat[valid]).astype(np.float32)
    image = transformed.reshape(arr.shape[0], arr.shape[1], 3)
    return np.stack(
        [robust_uint8(image[:, :, idx], low_percentile, high_percentile) for idx in range(3)],
        axis=2,
    )


def resample_spectrum(
    source_wavelengths: np.ndarray,
    spectrum: np.ndarray,
    target_wavelengths: np.ndarray,
    require_full_overlap: bool = True,
) -> np.ndarray:
    src_x = np.asarray(source_wavelengths, dtype=np.float64)
    src_y = np.asarray(spectrum, dtype=np.float64)
    dst_x = np.asarray(target_wavelengths, dtype=np.float64)
    if src_x.ndim != 1 or dst_x.ndim != 1 or src_y.shape[-1] != src_x.size:
        raise WavelengthCompatibilityError("Invalid source/target spectral dimensions.")
    if src_x[0] > src_x[-1]:
        src_x = src_x[::-1]
        src_y = src_y[..., ::-1]
    if dst_x[0] > dst_x[-1]:
        reverse_output = True
        interp_x = dst_x[::-1]
    else:
        reverse_output = False
        interp_x = dst_x
    if require_full_overlap and (interp_x[0] < src_x[0] or interp_x[-1] > src_x[-1]):
        raise WavelengthCompatibilityError(
            f"Target range {interp_x[0]:g}-{interp_x[-1]:g} nm is outside source range "
            f"{src_x[0]:g}-{src_x[-1]:g} nm."
        )
    result = np.interp(interp_x, src_x, src_y, left=np.nan, right=np.nan)
    return result[::-1].astype(np.float32) if reverse_output else result.astype(np.float32)


def sam_classify_array(
    spectra: np.ndarray,
    references: np.ndarray,
    threshold_deg: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify spectra using genuine Spectral Angle Mapper.

    Returns zero-based class indexes and minimum angle. Unknown pixels are -1.
    """
    values = np.asarray(spectra, dtype=np.float32)
    refs = np.asarray(references, dtype=np.float32)
    if values.ndim != 2 or refs.ndim != 2 or values.shape[1] != refs.shape[1]:
        raise SpectralViewError("SAM requires two matrices with the same band count.")
    if refs.shape[0] == 0:
        raise SpectralViewError("SAM requires at least one reference spectrum.")

    ref_norm = np.linalg.norm(refs, axis=1)
    if np.any(~np.isfinite(ref_norm)) or np.any(ref_norm <= np.finfo(np.float32).eps):
        raise SpectralViewError("Reference library contains non-finite or zero-norm spectra.")
    refs_unit = refs / ref_norm[:, None]

    value_norm = np.linalg.norm(values, axis=1)
    valid = np.all(np.isfinite(values), axis=1) & (value_norm > np.finfo(np.float32).eps)
    classes = np.full(values.shape[0], -1, dtype=np.int32)
    angles = np.full(values.shape[0], np.nan, dtype=np.float32)
    if np.any(valid):
        unit = values[valid] / value_norm[valid, None]
        cosines = np.clip(unit @ refs_unit.T, -1.0, 1.0)
        all_angles = np.degrees(np.arccos(cosines))
        best = np.argmin(all_angles, axis=1)
        best_angle = all_angles[np.arange(all_angles.shape[0]), best]
        accepted = best_angle <= float(threshold_deg)
        valid_indices = np.flatnonzero(valid)
        classes[valid_indices[accepted]] = best[accepted].astype(np.int32)
        angles[valid_indices] = best_angle.astype(np.float32)
    return classes, angles


def _border_fraction(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 1.0
    border = np.concatenate([mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]])
    return float(np.mean(border > 0))


def segment_objects(
    cube: np.ndarray,
    min_area_px: int = 25,
    max_area_fraction: float = 0.95,
    polarity: str = "auto",
    morphology_size: int = 3,
) -> SegmentationResult:
    """Segment foreground objects from a calibrated ROI using PCA + governed Otsu.

    Auto polarity selects the candidate with lower border occupancy while rejecting
    implausibly empty or nearly full masks. The algorithm remains a heuristic and
    therefore returns explicit QC metrics rather than a validation claim.
    """
    arr = np.asarray(cube, dtype=np.float32)
    if arr.ndim != 3 or min(arr.shape[:2]) < 2:
        raise SpectralViewError("ROI segmentation requires a non-empty 3-D cube.")
    h, w, bands = arr.shape
    flat = arr.reshape(-1, bands)
    valid = np.all(np.isfinite(flat), axis=1)
    if valid.sum() < 10:
        raise SpectralViewError("Too few finite ROI pixels for PCA segmentation.")

    pca = PCA(n_components=1, random_state=0)
    scores = np.full(flat.shape[0], np.nan, dtype=np.float32)
    scores[valid] = pca.fit_transform(flat[valid]).ravel().astype(np.float32)
    pc1 = scores.reshape(h, w)
    pc1_u8 = robust_uint8(pc1, 1.0, 99.0)
    blurred = cv2.GaussianBlur(pc1_u8, (3, 3), 0)
    _threshold, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates = {"normal": binary, "inverted": cv2.bitwise_not(binary)}
    requested = polarity.strip().lower()
    if requested in {"normal", "inverted"}:
        selected_name = requested
    elif requested == "auto":
        scored: List[Tuple[float, str]] = []
        for name, candidate in candidates.items():
            fg = float(np.mean(candidate > 0))
            border = _border_fraction(candidate)
            implausible = 10.0 if fg <= 0.001 or fg >= max_area_fraction else 0.0
            scored.append((border + implausible + abs(fg - 0.25) * 0.05, name))
        selected_name = min(scored)[1]
    else:
        raise ValueError("Polarity must be auto, normal or inverted.")

    mask = candidates[selected_name].copy()
    kernel_size = max(1, int(morphology_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    objects: List[SegmentationObject] = []
    next_id = 1
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < int(min_area_px):
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        object_mask = labels == label
        clean[object_mask] = 255
        mean_spectrum = np.nanmean(arr[object_mask], axis=0).astype(np.float32)
        objects.append(
            SegmentationObject(
                object_id=next_id,
                area_px=area,
                centroid_x=float(centroids[label][0]),
                centroid_y=float(centroids[label][1]),
                x=x,
                y=y,
                width=width,
                height=height,
                mean_spectrum=mean_spectrum,
            )
        )
        next_id += 1

    _n, relabelled, _stats, _centroids = cv2.connectedComponentsWithStats(clean, connectivity=8)
    fg_fraction = float(np.mean(clean > 0))
    warnings: List[str] = []
    if not objects:
        warnings.append("NO_OBJECTS_AFTER_AREA_FILTER")
    if fg_fraction > max_area_fraction:
        warnings.append("FOREGROUND_FRACTION_IMPLAUSIBLY_HIGH")
    return SegmentationResult(
        mask=clean,
        labels=relabelled,
        objects=objects,
        foreground_fraction=fg_fraction,
        border_fraction=_border_fraction(clean),
        polarity=selected_name,
        warnings=warnings,
    )


INDEX_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "NDVI": {"wavelengths": (800.0, 670.0), "formula": "(a-b)/(a+b)"},
    "GNDVI": {"wavelengths": (800.0, 550.0), "formula": "(a-b)/(a+b)"},
    "NDRE": {"wavelengths": (790.0, 720.0), "formula": "(a-b)/(a+b)"},
    "PRI": {"wavelengths": (531.0, 570.0), "formula": "(a-b)/(a+b)"},
    "PSRI": {"wavelengths": (680.0, 500.0, 750.0), "formula": "(a-b)/c"},
    "WBI": {"wavelengths": (900.0, 970.0), "formula": "a/b"},
    "MSI": {"wavelengths": (1600.0, 820.0), "formula": "a/b"},
    "CIRE": {"wavelengths": (800.0, 710.0), "formula": "a/b-1"},
    "MTCI": {"wavelengths": (754.0, 709.0, 681.0), "formula": "(a-b)/(b-c)"},
    "MCARI": {"wavelengths": (700.0, 670.0, 550.0), "formula": "((a-b)-0.2*(a-c))*(a/b)"},
    "MND705": {"wavelengths": (750.0, 705.0, 445.0), "formula": "(a-b)/(a+b-2*c)"},
    "RED EDGE": {"wavelengths": (750.0, 705.0), "formula": "(a-b)/(a+b)"},
    "SAVI": {"wavelengths": (800.0, 670.0), "formula": "1.5*(a-b)/(a+b+0.5)"},
    "EVI": {"wavelengths": (800.0, 670.0, 450.0), "formula": "2.5*(a-b)/(a+6*b-7.5*c+1)"},
    "NDWI": {"wavelengths": (860.0, 1240.0), "formula": "(a-b)/(a+b)"},
    "CRI1": {"wavelengths": (510.0, 550.0), "formula": "(1/a)-(1/b)"},
    "ARI": {"wavelengths": (550.0, 700.0), "formula": "(1/a)-(1/b)"},
}


def compute_spectral_index(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    index_name: str,
    tolerance_nm: float = 15.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    key = index_name.upper()
    if key not in INDEX_DEFINITIONS:
        raise SpectralViewError(f"Unknown spectral index: {index_name}")
    definition = INDEX_DEFINITIONS[key]
    requested = definition["wavelengths"]
    indices = [nearest_band(wavelengths, value, tolerance_nm) for value in requested]
    bands = [np.asarray(cube[:, :, idx], dtype=np.float32) for idx in indices]
    eps = np.finfo(np.float32).eps
    with np.errstate(divide="ignore", invalid="ignore"):
        if key in {"NDVI", "GNDVI", "NDRE", "PRI"}:
            result = (bands[0] - bands[1]) / np.where(np.abs(bands[0] + bands[1]) <= eps, np.nan, bands[0] + bands[1])
        elif key == "PSRI":
            result = (bands[0] - bands[1]) / np.where(np.abs(bands[2]) <= eps, np.nan, bands[2])
        elif key in {"WBI", "MSI"}:
            result = bands[0] / np.where(np.abs(bands[1]) <= eps, np.nan, bands[1])
        elif key == "CIRE":
            result = bands[0] / np.where(np.abs(bands[1]) <= eps, np.nan, bands[1]) - 1.0
        elif key == "MTCI":
            result = (bands[0] - bands[1]) / np.where(np.abs(bands[1] - bands[2]) <= eps, np.nan, bands[1] - bands[2])
        elif key == "MCARI":
            result = ((bands[0] - bands[1]) - 0.2 * (bands[0] - bands[2])) * (
                bands[0] / np.where(np.abs(bands[1]) <= eps, np.nan, bands[1])
            )
        elif key in {"RED EDGE", "NDWI"}:
            result = (bands[0] - bands[1]) / np.where(np.abs(bands[0] + bands[1]) <= eps, np.nan, bands[0] + bands[1])
        elif key == "MND705":
            denom = bands[0] + bands[1] - 2.0 * bands[2]
            result = (bands[0] - bands[1]) / np.where(np.abs(denom) <= eps, np.nan, denom)
        elif key == "SAVI":
            result = 1.5 * (bands[0] - bands[1]) / np.where(np.abs(bands[0] + bands[1] + 0.5) <= eps, np.nan, bands[0] + bands[1] + 0.5)
        elif key == "EVI":
            denom = bands[0] + 6.0 * bands[1] - 7.5 * bands[2] + 1.0
            result = 2.5 * (bands[0] - bands[1]) / np.where(np.abs(denom) <= eps, np.nan, denom)
        elif key in {"CRI1", "ARI"}:
            inv0 = 1.0 / np.where(np.abs(bands[0]) <= eps, np.nan, bands[0])
            inv1 = 1.0 / np.where(np.abs(bands[1]) <= eps, np.nan, bands[1])
            result = inv0 - inv1
        else:  # pragma: no cover
            raise SpectralViewError(f"Index formula not implemented: {key}")
    metadata = {
        "index": key,
        "requested_wavelengths_nm": list(requested),
        "actual_wavelengths_nm": [float(wavelengths[idx]) for idx in indices],
        "band_indices": indices,
    }
    return result.astype(np.float32), metadata


def extract_exposure_value(metadata: Mapping[str, Any]) -> Optional[float]:
    """Extract a comparable numeric exposure/integration-time value when present."""
    import re

    normalised = {str(key).strip().lower().replace("_", " "): value for key, value in metadata.items()}
    for key in ("integration time", "exposure time", "exposure", "integrationtime"):
        if key not in normalised:
            continue
        value = normalised[key]
        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _resolve_image_path(header_path: str) -> str | None:
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

class EnviDocument:
    """Lazy ENVI document with shared region calibration and chunked analysis."""

    def __init__(
        self,
        data_header: str,
        dark_header: Optional[str] = None,
        white_header: Optional[str] = None,
        calibration_config: Optional[CalibrationConfig] = None,
    ) -> None:
        try:
            from spectral.io import envi
        except Exception as exc:  # pragma: no cover - environment dependency
            raise SpectralViewError("The 'spectral' package is required to open ENVI files.") from exc

        self.data_header = str(Path(data_header).resolve())
        self.dark_header = str(Path(dark_header).resolve()) if dark_header else None
        self.white_header = str(Path(white_header).resolve()) if white_header else None
        self.calibration_config = calibration_config or CalibrationConfig()

        self._data_spy = envi.open(self.data_header, image=_resolve_image_path(self.data_header))
        self.raw = self._open_memmap(self._data_spy)
        if self.raw.ndim != 3:
            raise InputContractError(f"Expected a 3-D ENVI cube, got shape {self.raw.shape}.")
        self.height, self.width, self.n_bands = map(int, self.raw.shape)
        self.metadata = dict(self._data_spy.metadata)
        self.wavelengths, self.has_physical_wavelengths, self.metadata_warnings = parse_wavelengths(
            self.metadata, self.n_bands
        )
        self.exposure_value = extract_exposure_value(self.metadata)

        self.dark = self._load_reference(envi, self.dark_header, "dark")
        self.white = self._load_reference(envi, self.white_header, "white")
        self.last_qc: Optional[CalibrationQC] = None

    @staticmethod
    def _open_memmap(spy: Any) -> np.ndarray:
        try:
            return spy.open_memmap(writable=False)
        except Exception:
            return np.asarray(spy.load())

    def _load_reference(self, envi_module: Any, header: Optional[str], kind: str) -> Optional[Dict[str, Any]]:
        if not header:
            return None
        spy = envi_module.open(header, image=_resolve_image_path(header))
        arr = self._open_memmap(spy)
        if arr.ndim != 3 or int(arr.shape[2]) != self.n_bands:
            raise InputContractError(
                f"{kind.title()} reference shape {arr.shape} is incompatible with data shape {self.raw.shape}."
            )
        ref_metadata = dict(spy.metadata)
        ref_wavelengths, has_wavelengths, ref_warnings = parse_wavelengths(ref_metadata, int(arr.shape[2]))
        self.metadata_warnings.extend(f"{kind.upper()}_{warning}" for warning in ref_warnings)
        reference_exposure = extract_exposure_value(ref_metadata)
        if self.exposure_value is not None and reference_exposure is not None:
            scale = max(abs(self.exposure_value), abs(reference_exposure), 1e-12)
            if abs(self.exposure_value - reference_exposure) / scale > 0.01:
                raise InputContractError(
                    f"{kind.title()} reference exposure/integration time {reference_exposure:g} does not match "
                    f"data value {self.exposure_value:g}."
                )
        elif self.exposure_value is None or reference_exposure is None:
            self.metadata_warnings.append(f"{kind.upper()}_EXPOSURE_COMPATIBILITY_NOT_VERIFIABLE")
        if has_wavelengths and self.has_physical_wavelengths and not np.allclose(
            ref_wavelengths, self.wavelengths, rtol=0, atol=0.05
        ):
            raise InputContractError(f"{kind.title()} reference wavelengths do not match data wavelengths.")
        if arr.shape[:2] == self.raw.shape[:2]:
            return {"mode": "pixel", "array": arr, "profile": None}
        profile = np.nanmean(np.asarray(arr, dtype=np.float32), axis=(0, 1)).astype(np.float32)
        return {"mode": "profile", "array": None, "profile": profile.reshape(1, 1, -1)}

    def _reference_region(self, ref: Optional[Dict[str, Any]], y1: int, y2: int, x1: int, x2: int) -> Optional[np.ndarray]:
        if ref is None:
            return None
        if ref["mode"] == "pixel":
            return np.asarray(ref["array"][y1:y2, x1:x2, :], dtype=np.float32)
        return ref["profile"]

    def calibrated_region(self, y1: int, y2: int, x1: int, x2: int) -> Tuple[np.ndarray, CalibrationQC]:
        y1 = max(0, min(self.height, int(y1)))
        y2 = max(y1, min(self.height, int(y2)))
        x1 = max(0, min(self.width, int(x1)))
        x2 = max(x1, min(self.width, int(x2)))
        if y2 <= y1 or x2 <= x1:
            raise InputContractError("Requested region is empty.")
        raw = np.asarray(self.raw[y1:y2, x1:x2, :], dtype=np.float32)
        dark = self._reference_region(self.dark, y1, y2, x1, x2)
        white = self._reference_region(self.white, y1, y2, x1, x2)
        calibrated, qc = calibrate_array(raw, dark, white, self.calibration_config)
        self.last_qc = qc
        return calibrated, qc

    def calibrated_spectrum(self, y: int, x: int) -> Tuple[np.ndarray, CalibrationQC]:
        region, qc = self.calibrated_region(y, y + 1, x, x + 1)
        return region[0, 0, :].copy(), qc

    def calibrated_preview(self, max_dimension: int = 1200) -> Tuple[np.ndarray, CalibrationQC, float, float]:
        step_y = max(1, math.ceil(self.height / max_dimension))
        step_x = max(1, math.ceil(self.width / max_dimension))
        raw = np.asarray(self.raw[::step_y, ::step_x, :], dtype=np.float32)
        if self.dark is None:
            dark = None
        elif self.dark["mode"] == "pixel":
            dark = np.asarray(self.dark["array"][::step_y, ::step_x, :], dtype=np.float32)
        else:
            dark = self.dark["profile"]
        if self.white is None:
            white = None
        elif self.white["mode"] == "pixel":
            white = np.asarray(self.white["array"][::step_y, ::step_x, :], dtype=np.float32)
        else:
            white = self.white["profile"]
        calibrated, qc = calibrate_array(raw, dark, white, self.calibration_config)
        self.last_qc = qc
        scale_x = self.width / calibrated.shape[1]
        scale_y = self.height / calibrated.shape[0]
        return calibrated, qc, scale_x, scale_y

    def iter_calibrated_chunks(self, chunk_rows: int = 32) -> Iterator[Tuple[int, int, np.ndarray, CalibrationQC]]:
        for y1 in range(0, self.height, max(1, int(chunk_rows))):
            y2 = min(self.height, y1 + max(1, int(chunk_rows)))
            chunk, qc = self.calibrated_region(y1, y2, 0, self.width)
            yield y1, y2, chunk, qc

    def mean_spectrum(self, mask: Optional[np.ndarray] = None, chunk_rows: int = 32) -> Tuple[np.ndarray, CalibrationQC]:
        if mask is not None and mask.shape != (self.height, self.width):
            raise InputContractError(
                f"Mask shape {mask.shape} does not match cube spatial shape {(self.height, self.width)}."
            )
        sum_values = np.zeros(self.n_bands, dtype=np.float64)
        counts = np.zeros(self.n_bands, dtype=np.int64)
        last_qc: Optional[CalibrationQC] = None
        for y1, y2, chunk, qc in self.iter_calibrated_chunks(chunk_rows):
            last_qc = qc
            if mask is not None:
                selected = chunk[mask[y1:y2] > 0]
            else:
                selected = chunk.reshape(-1, self.n_bands)
            if selected.size == 0:
                continue
            finite = np.isfinite(selected)
            sum_values += np.nansum(selected, axis=0)
            counts += finite.sum(axis=0)
        if np.any(counts == 0):
            raise CalibrationError("No finite observations available for one or more bands.")
        return (sum_values / counts).astype(np.float32), last_qc or CalibrationQC(
            "UNKNOWN", "unknown", 0.0, 0.0, 0.0, 0.0, ["NO_CALIBRATION_QC"]
        )

    def sam_classify(
        self,
        references: np.ndarray,
        preprocessing: str = "none",
        threshold_deg: float = 12.0,
        sg_window: int = 11,
        sg_polyorder: int = 2,
        chunk_rows: int = 32,
        progress: Optional[Callable[[int], None]] = None,
    ) -> SAMResult:
        refs_proc = preprocess_spectra(references, preprocessing, self.wavelengths, sg_window, sg_polyorder)
        class_map = np.full((self.height, self.width), -1, dtype=np.int32)
        angle_map = np.full((self.height, self.width), np.nan, dtype=np.float32)
        for y1, y2, chunk, _qc in self.iter_calibrated_chunks(chunk_rows):
            flat = chunk.reshape(-1, self.n_bands)
            flat_proc = preprocess_spectra(flat, preprocessing, self.wavelengths, sg_window, sg_polyorder)
            classes, angles = sam_classify_array(flat_proc, refs_proc, threshold_deg)
            class_map[y1:y2] = classes.reshape(y2 - y1, self.width)
            angle_map[y1:y2] = angles.reshape(y2 - y1, self.width)
            if progress:
                progress(int(round(y2 / self.height * 100)))
        valid = np.isfinite(angle_map)
        accepted = class_map >= 0
        valid_fraction = float(np.mean(valid))
        unknown_fraction = float(np.mean(valid & ~accepted)) if np.any(valid) else 1.0
        return SAMResult(class_map, angle_map, valid_fraction, unknown_fraction)

    def provenance(self) -> Dict[str, Any]:
        data_path = Path(self.data_header)
        result: Dict[str, Any] = {
            "data_header": self.data_header,
            "data_header_sha256": sha256_file(data_path),
            "shape": [self.height, self.width, self.n_bands],
            "has_physical_wavelengths": self.has_physical_wavelengths,
            "wavelength_min": float(np.min(self.wavelengths)),
            "wavelength_max": float(np.max(self.wavelengths)),
            "dark_header": self.dark_header,
            "white_header": self.white_header,
            "calibration_config": asdict(self.calibration_config),
        }
        for key, path in (("dark", self.dark_header), ("white", self.white_header)):
            if path:
                result[f"{key}_header_sha256"] = sha256_file(path)
        return result


def build_foreground_mask_from_preview(
    document: EnviDocument,
    min_area_px_preview: int = 25,
    polarity: str = "auto",
    morphology_size: int = 3,
    max_dimension: int = 1000,
) -> Tuple[np.ndarray, SegmentationResult, CalibrationQC]:
    preview, qc, _scale_x, _scale_y = document.calibrated_preview(max_dimension=max_dimension)
    segmentation = segment_objects(
        preview,
        min_area_px=min_area_px_preview,
        polarity=polarity,
        morphology_size=morphology_size,
    )
    if not segmentation.objects:
        raise SpectralViewError("Automatic foreground segmentation found no accepted objects.")
    full_mask = cv2.resize(
        segmentation.mask,
        (document.width, document.height),
        interpolation=cv2.INTER_NEAREST,
    )
    return full_mask, segmentation, qc


def run_batch(
    parent_dir: str,
    output_dir: str,
    processing_config: ProcessingConfig,
    extraction_mode: str = "auto_foreground",
    fixed_rois: Optional[list] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> Dict[str, Any]:
    """Process all governed ENVI datasets and publish atomic evidence files."""
    started = utc_now_iso()
    folders = find_dataset_folders(parent_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sample_results: List[BatchSampleResult] = []
    provenance: List[Dict[str, Any]] = []

    for index, folder in enumerate(folders):
        sample_id = Path(folder).name
        if progress:
            progress(int(index / max(1, len(folders)) * 100), sample_id)
        discovery = discover_envi_folder(folder)
        if not discovery.data_header:
            sample_results.append(BatchSampleResult(sample_id, folder, "FAILED_INPUT", "No data ENVI header."))
            continue
        try:
            document = EnviDocument(
                discovery.data_header,
                discovery.dark_header,
                discovery.white_header,
                processing_config.calibration,
            )
            foreground_fraction: Optional[float] = None
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
            
            if extraction_mode == "auto_foreground":
                mask, segmentation, _preview_qc = build_foreground_mask_from_preview(
                    document,
                    min_area_px_preview=processing_config.segmentation_min_area_px,
                    polarity=processing_config.segmentation_polarity,
                    morphology_size=processing_config.segmentation_morphology_size,
                    max_dimension=processing_config.preview_max_dimension,
                )
                spectrum, qc = document.mean_spectrum(mask, processing_config.chunk_rows)
                foreground_fraction = segmentation.foreground_fraction
            elif extraction_mode == "whole_frame":
                spectrum, qc = document.mean_spectrum(None, processing_config.chunk_rows)
            else:
                raise SpectralViewError(f"Unknown extraction mode: {extraction_mode}")

            if processing_config.preprocessing == "absorbance" and qc.units != "reflectance":
                raise SpectralViewError(
                    f"Absorbance preprocessing requires reflectance, but sample units are {qc.units}."
                )
            spectrum = preprocess_spectra(
                spectrum,
                processing_config.preprocessing,
                document.wavelengths,
                processing_config.sg_window,
                processing_config.sg_polyorder,
            )
            sample_results.append(
                BatchSampleResult(
                    sample_id=sample_id,
                    folder=folder,
                    status="SUCCESS_WITH_WARNING" if qc.warnings or discovery.warnings else "SUCCESS",
                    message=";".join(discovery.warnings + qc.warnings),
                    calibration_mode=qc.mode,
                    units=qc.units,
                    n_bands=document.n_bands,
                    foreground_fraction=foreground_fraction,
                    wavelengths=document.wavelengths.copy(),
                    spectrum=np.asarray(spectrum, dtype=np.float32).copy(),
                )
            )
            provenance.append(document.provenance())
        except InputContractError as exc:
            sample_results.append(BatchSampleResult(sample_id, folder, "FAILED_INPUT", str(exc)))
        except CalibrationError as exc:
            sample_results.append(BatchSampleResult(sample_id, folder, "FAILED_CALIBRATION", str(exc)))
        except SpectralViewError as exc:
            status = "FAILED_SEGMENTATION" if "segment" in str(exc).lower() else "FAILED_ANALYSIS"
            sample_results.append(BatchSampleResult(sample_id, folder, status, str(exc)))
        except Exception as exc:  # guarded terminal status
            sample_results.append(BatchSampleResult(sample_id, folder, "FAILED_ANALYSIS", f"{type(exc).__name__}: {exc}"))

    status_rows: List[List[Any]] = [[
        "sample_id", "folder", "status", "message", "calibration_mode", "units", "n_bands", "foreground_fraction"
    ]]
    for result in sample_results:
        status_rows.append([
            result.sample_id,
            result.folder,
            result.status,
            result.message,
            result.calibration_mode,
            result.units,
            result.n_bands,
            "" if result.foreground_fraction is None else f"{result.foreground_fraction:.8f}",
        ])
    atomic_write_csv(out / "sample_status.csv", status_rows)

    spectra_rows: List[List[Any]] = [["sample_id", "wavelength", "value", "units", "preprocessing"]]
    for result in sample_results:
        if result.spectrum is None or result.wavelengths is None:
            continue
        for wavelength, value in zip(result.wavelengths, result.spectrum):
            spectra_rows.append([
                result.sample_id,
                f"{float(wavelength):.8g}",
                "" if not np.isfinite(value) else f"{float(value):.10g}",
                result.units,
                processing_config.preprocessing,
            ])
    atomic_write_csv(out / "spectra_long.csv", spectra_rows)

    counts: Dict[str, int] = {}
    for result in sample_results:
        counts[result.status] = counts.get(result.status, 0) + 1
    manifest = {
        "software": "SpectralView",
        "version": VERSION,
        "release_status": RELEASE_STATUS,
        "started_utc": started,
        "completed_utc": utc_now_iso(),
        "parent_dir": str(Path(parent_dir).resolve()),
        "output_dir": str(out.resolve()),
        "processing_config": asdict(processing_config),
        "extraction_mode": extraction_mode,
        "dataset_count": len(folders),
        "terminal_status_counts": counts,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
        "input_provenance": provenance,
    }
    atomic_write_text(out / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    if progress:
        progress(100, "Complete")
    return manifest
