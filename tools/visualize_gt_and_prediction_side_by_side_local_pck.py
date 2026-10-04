#!/usr/bin/env python3
"""Render ground truth and inference side by side with LocalPCK."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from local_pck_visualization import (
    add_local_pck_args, load_comparison, local_pck_radii, metric_label,
)
from visualize_gt_and_prediction_side_by_side import add_header, draw_panel
from visualize_gt_vs_prediction import PRED_COLOR, TOLERANCE_COLOR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="MMPose config")
    parser.add_argument("checkpoint", type=Path, help="trained checkpoint")
    parser.add_argument("image", type=Path, nargs="?", help="COCO image (default: first image)")
    parser.add_argument(
        "--annotations", type=Path,
        default=Path("data/LumbarCoco/annotations/lumbar_keypoints_val.json"),
        help="COCO annotation file (default: validation annotations)",
    )
    parser.add_argument("--output", type=Path, help="output image path")
    parser.add_argument("--device", default="auto", help="inference device (default: auto)")
    parser.add_argument("--score-thr", type=float, default=0.2,
                        help="hide predictions below this confidence (default: 0.2)")
    parser.add_argument("--show-labels", action="store_true",
                        help="draw landmark names next to predictions")
    parser.add_argument("--show-bbox", action="store_true",
                        help="draw the model input bounding box")
    parser.add_argument("--hide-bbox", action="store_false", dest="show_bbox",
                        help=argparse.SUPPRESS)
    add_local_pck_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (image_path, image, annotation, names, skeleton, gt, gt_visible,
     pred, pred_visible, _, correct, total) = load_comparison(args)
    bbox = annotation["bbox"]
    radii = local_pck_radii(
        gt, gt_visible, args.local_pck_mode, args.local_pck_thr
    )
    tolerance_panel = draw_panel(
        image, gt, np.zeros_like(gt_visible), names, skeleton, bbox, TOLERANCE_COLOR,
        prediction=False, show_labels=False, show_bbox=args.show_bbox,
        tolerance_gt=gt, tolerance_radii=radii,
    )
    pred_panel = draw_panel(
        image, pred, pred_visible, names, skeleton, bbox, PRED_COLOR,
        prediction=True, show_labels=args.show_labels, show_bbox=args.show_bbox,
        tolerance_gt=gt, tolerance_radii=radii,
    )
    tolerance_panel = add_header(
        tolerance_panel, "LOCALPCK TOLERANCE",
        f"Valid GT points: {total}/{len(gt)}   Radius: orange",
        TOLERANCE_COLOR,
    )
    pred_panel = add_header(
        pred_panel, "MODEL INFERENCE",
        metric_label(correct, total, args.local_pck_mode, args.local_pck_thr),
        PRED_COLOR,
    )
    divider = np.full((tolerance_panel.shape[0], 10, 3), 235, dtype=np.uint8)
    comparison = np.hstack((tolerance_panel, divider, pred_panel))
    output = args.output or Path("output/comparisons") / (
        f"{image_path.stem}_gt_and_prediction_local_pck.jpg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), comparison):
        raise OSError(f"could not write visualization: {output}")
    print(f"Side-by-side visualization written to {output}")


if __name__ == "__main__":
    main()
