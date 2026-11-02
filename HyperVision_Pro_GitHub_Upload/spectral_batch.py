"""Command-line governed batch runner for SpectralView 2.1.12."""
from __future__ import annotations

import argparse
import json

from spectral_core import CalibrationConfig, ProcessingConfig, run_batch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpectralView governed ENVI batch runner")
    parser.add_argument("--parent", required=True, help="Parent directory containing ENVI dataset folders.")
    parser.add_argument("--output", required=True, help="Evidence output directory.")
    parser.add_argument(
        "--extraction",
        choices=("auto_foreground", "whole_frame"),
        default="auto_foreground",
    )
    parser.add_argument("--input-reflectance", action="store_true")
    parser.add_argument("--clip-reflectance", action="store_true")
    parser.add_argument("--clip-max", type=float, default=1.5)
    parser.add_argument(
        "--preprocessing",
        choices=("none", "snv", "savitzky_golay", "first_derivative", "second_derivative", "absorbance"),
        default="none",
    )
    parser.add_argument("--chunk-rows", type=int, default=32)
    parser.add_argument("--min-area", type=int, default=25)
    parser.add_argument("--polarity", choices=("auto", "normal", "inverted"), default="auto")
    parser.add_argument("--morphology", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    calibration = CalibrationConfig(
        input_is_reflectance=args.input_reflectance,
        clip_min=0.0 if args.clip_reflectance else None,
        clip_max=args.clip_max if args.clip_reflectance else None,
    )
    config = ProcessingConfig(
        calibration=calibration,
        preprocessing=args.preprocessing,
        chunk_rows=max(1, args.chunk_rows),
        segmentation_min_area_px=max(1, args.min_area),
        segmentation_polarity=args.polarity,
        segmentation_morphology_size=max(1, args.morphology),
    )
    manifest = run_batch(
        args.parent,
        args.output,
        config,
        extraction_mode=args.extraction,
        progress=lambda value, detail: print(f"[{value:3d}%] {detail}", flush=True),
    )
    print(json.dumps(manifest["terminal_status_counts"], sort_keys=True))
    failures = sum(
        count for status, count in manifest["terminal_status_counts"].items() if status.startswith("FAILED")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
