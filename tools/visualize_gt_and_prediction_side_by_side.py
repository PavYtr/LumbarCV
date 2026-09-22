#!/usr/bin/env python3
"""Render ground truth and model inference as two side-by-side panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from visualize_gt_vs_prediction import (
    BBOX_COLOR,
    GT_COLOR,
    PRED_COLOR,
    TEXT_COLOR,
    draw_skeleton,
    load_coco_sample,
    point,
    predict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="MMPose config")
    parser.add_argument("checkpoint", type=Path, help="trained checkpoint")
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help=(
            "image present in the COCO file; if omitted, use its first image"
        ),
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("data/LumbarCoco/annotations/lumbar_keypoints_val.json"),
        help="COCO annotation file (default: validation annotations)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "output image (default: "
            "output/comparisons/<image>_gt_and_prediction.jpg)"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="inference device, for example cuda:0 or cpu (default: auto)",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.2,
        help="hide predictions below this confidence (default: 0.2)",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="draw landmark names in both panels",
    )
    parser.add_argument(
        "--hide-bbox",
        action="store_true",
        help="do not draw the model input bounding box",
    )
    return parser.parse_args()


def draw_panel(
    image: np.ndarray,
    keypoints: np.ndarray,
    visible: np.ndarray,
    names: list[str],
    skeleton: list[tuple[int, int]],
    bbox_xywh: list[float],
    color: tuple[int, int, int],
    prediction: bool,
    show_labels: bool,
    show_bbox: bool,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.8, min(width, height) / 900.0)
    line_width = max(1, round(2 * scale)) / 3
    radius = max(3, round(5 * scale))

    if show_bbox:
        x, y, bbox_width, bbox_height = bbox_xywh
        cv2.rectangle(
            canvas,
            (round(x), round(y)),
            (round(x + bbox_width), round(y + bbox_height)),
            BBOX_COLOR,
            line_width,
            cv2.LINE_AA,
        )

    skeleton_layer = canvas.copy()
    draw_skeleton(
        skeleton_layer,
        keypoints,
        skeleton,
        visible,
        color,
        line_width + 1,
    )
    cv2.addWeighted(skeleton_layer, 0.85, canvas, 0.15, 0, canvas)

    for index, landmark in enumerate(keypoints):
        if not visible[index]:
            continue
        center = point(landmark)
        if prediction:
            cv2.drawMarker(
                canvas,
                center,
                color,
                cv2.MARKER_TILTED_CROSS,
                radius * 2 + 4,
                line_width + 1,
                cv2.LINE_AA,
            )
        else:
            cv2.circle(
                canvas,
                center,
                radius + 1,
                (0, 0, 0),
                line_width + 2,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                center,
                radius,
                color,
                line_width,
                cv2.LINE_AA,
            )

        if show_labels and index < len(names):
            cv2.putText(
                canvas,
                names[index],
                (center[0] + radius + 3, center[1] - radius - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42 * scale,
                color,
                max(1, line_width),
                cv2.LINE_AA,
            )
    return canvas


def add_header(
    panel: np.ndarray,
    title: str,
    subtitle: str,
    accent: tuple[int, int, int],
) -> np.ndarray:
    height, width = panel.shape[:2]
    scale = max(0.8, min(width, height) / 900.0)
    header_height = max(76, round(100 * scale))
    header = np.full((header_height, width, 3), 18, dtype=np.uint8)
    accent_width = max(7, round(10 * scale))
    header[:, :accent_width] = accent

    cv2.putText(
        header,
        title,
        (accent_width + 18, round(header_height * 0.43)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.7, 0.9 * scale),
        TEXT_COLOR,
        max(2, round(2 * scale)),
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        subtitle,
        (accent_width + 18, round(header_height * 0.78)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, 0.55 * scale),
        (205, 205, 205),
        max(1, round(1.5 * scale)),
        cv2.LINE_AA,
    )
    return np.vstack((header, panel))


def comparison_metrics(
    gt: np.ndarray,
    gt_visible: np.ndarray,
    pred: np.ndarray,
    pred_visible: np.ndarray,
    bbox_xywh: list[float],
) -> tuple[float | None, float | None]:
    comparable = gt_visible & pred_visible
    if not np.any(comparable):
        return None, None
    errors = np.linalg.norm(pred[comparable] - gt[comparable], axis=1)
    norm = max(float(bbox_xywh[2]), float(bbox_xywh[3]), 1.0)
    return float(errors.mean()), float(np.mean(errors <= 0.05 * norm))


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.score_thr <= 1.0:
        raise ValueError("--score-thr must be between 0 and 1")

    image_path, _, annotation, names, skeleton = load_coco_sample(
        args.annotations, args.image
    )
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")

    raw_gt = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
    gt = raw_gt[:, :2]
    gt_visible = raw_gt[:, 2] > 0
    pred, scores = predict(
        args.config,
        args.checkpoint,
        image,
        annotation["bbox"],
        args.device,
    )
    if len(pred) != len(gt):
        raise ValueError(
            f"model returned {len(pred)} keypoints, but annotation has {len(gt)}"
        )
    pred_visible = scores >= args.score_thr

    ground_truth_panel = draw_panel(
        image,
        gt,
        gt_visible,
        names,
        skeleton,
        annotation["bbox"],
        GT_COLOR,
        prediction=False,
        show_labels=args.show_labels,
        show_bbox=not args.hide_bbox,
    )
    prediction_panel = draw_panel(
        image,
        pred,
        pred_visible,
        names,
        skeleton,
        annotation["bbox"],
        PRED_COLOR,
        prediction=True,
        show_labels=args.show_labels,
        show_bbox=not args.hide_bbox,
    )

    mean_error, pck = comparison_metrics(
        gt, gt_visible, pred, pred_visible, annotation["bbox"]
    )
    gt_subtitle = f"Annotated landmarks: {int(gt_visible.sum())}/{len(gt)}"
    if mean_error is None or pck is None:
        pred_subtitle = "No comparable landmarks"
    else:
        pred_subtitle = (
            f"Visible: {int(pred_visible.sum())}/{len(pred)}   "
            f"Mean error: {mean_error:.1f}px   PCK@0.05: {pck:.3f}"
        )

    ground_truth_panel = add_header(
        ground_truth_panel, "GROUND TRUTH", gt_subtitle, GT_COLOR
    )
    prediction_panel = add_header(
        prediction_panel, "MODEL INFERENCE", pred_subtitle, PRED_COLOR
    )
    divider = np.full(
        (ground_truth_panel.shape[0], 10, 3), 235, dtype=np.uint8
    )
    comparison = np.hstack((ground_truth_panel, divider, prediction_panel))

    output = args.output or Path("output/comparisons") / (
        f"{image_path.stem}_gt_and_prediction.jpg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), comparison):
        raise OSError(f"could not write visualization: {output}")
    print(f"Side-by-side visualization written to {output}")


if __name__ == "__main__":
    main()
