#!/usr/bin/env python3
"""Overlay LumbarCoco ground truth and predictions with LocalPCK."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from local_pck_visualization import (
    add_local_pck_args, load_comparison, local_pck_radii, metric_label,
)
from visualize_gt_vs_prediction import draw_overlay


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
                        help="draw landmark names next to ground-truth points")
    add_local_pck_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (image_path, image, annotation, names, skeleton, gt, gt_visible,
     pred, pred_visible, scores, correct, total) = load_comparison(args)
    radii = local_pck_radii(
        gt, gt_visible, args.local_pck_mode, args.local_pck_thr
    )
    visualization = draw_overlay(
        image, gt, gt_visible, pred, pred_visible, scores,
        annotation["bbox"], names, skeleton, args.show_labels,
        metrics=metric_label(correct, total, args.local_pck_mode, args.local_pck_thr),
        tolerance_radii=radii, local_pck_only=True,
    )
    output = args.output or Path("output/comparisons") / (
        f"{image_path.stem}_gt_vs_pred_local_pck.jpg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), visualization):
        raise OSError(f"could not write visualization: {output}")
    print(f"Visualization written to {output}")


if __name__ == "__main__":
    main()
