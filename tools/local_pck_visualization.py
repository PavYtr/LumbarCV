"""Shared LocalPCK calculation and input loading for visualization scripts."""

from __future__ import annotations

import argparse
import cv2
import numpy as np

from visualize_gt_vs_prediction import load_coco_sample, predict


KEYPOINT_NAMES = [
    *(f"L{level}{plate}{side}" for level in range(1, 6)
      for plate in ("T", "B") for side in ("F", "B")),
    "S1TF", "S1TB",
]
SEGMENTS = np.arange(22, dtype=np.int64).reshape(-1, 2)


def add_local_pck_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--local-pck-mode", choices=("segment", "image_mean"),
        default="segment", help="GT segment length normalization (default: segment)",
    )
    parser.add_argument(
        "--local-pck-thr", type=float, default=0.05,
        help="fraction of segment length allowed as error (default: 0.05)",
    )


def local_pck(
    gt: np.ndarray,
    visible: np.ndarray,
    pred: np.ndarray,
    mode: str = "segment",
    threshold: float = 0.05,
) -> tuple[int, int]:
    """Return correct and eligible point counts, matching metric.LocalPCK."""
    if pred.shape != gt.shape:
        raise ValueError("prediction shape must match ground truth")
    radii = local_pck_radii(gt, visible, mode, threshold)
    eligible = np.isfinite(radii)
    if not eligible.any():
        return 0, 0
    errors = np.linalg.norm(pred[eligible] - gt[eligible], axis=1)
    correct = np.count_nonzero(np.isfinite(errors) & (errors <= radii[eligible]))
    return int(correct), int(eligible.sum())


def local_pck_radii(
    gt: np.ndarray,
    visible: np.ndarray,
    mode: str = "segment",
    threshold: float = 0.05,
) -> np.ndarray:
    """Pixel tolerance for each point; NaN marks an ineligible GT segment."""
    if gt.shape != (22, 2) or visible.shape != (22,):
        raise ValueError("LocalPCK requires the 22-point LumbarCoco layout")
    if mode not in ("segment", "image_mean"):
        raise ValueError("mode must be 'segment' or 'image_mean'")
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")

    endpoints = gt[SEGMENTS]
    lengths = np.linalg.norm(endpoints[:, 0] - endpoints[:, 1], axis=1)
    valid = (
        visible[SEGMENTS].all(axis=1)
        & np.isfinite(endpoints).all(axis=(1, 2))
        & np.isfinite(lengths)
        & (lengths > 0)
    )
    radii = np.full(22, np.nan, dtype=np.float64)
    indices = SEGMENTS[valid].reshape(-1)
    if len(indices):
        norms = (
            np.full(len(indices), lengths[valid].mean())
            if mode == "image_mean" else np.repeat(lengths[valid], 2)
        )
        radii[indices] = threshold * norms
    return radii


def metric_label(correct: int, total: int, mode: str, threshold: float) -> str:
    value = correct / total if total else 0.0
    return f"LocalPCK/{mode}@{threshold:g}: {value:.3f} ({correct}/{total})"


def load_comparison(args: argparse.Namespace):
    if not 0 <= args.score_thr <= 1:
        raise ValueError("--score-thr must be between 0 and 1")
    if not np.isfinite(args.local_pck_thr) or args.local_pck_thr < 0:
        raise ValueError("--local-pck-thr must be finite and nonnegative")

    image_path, _, annotation, names, skeleton = load_coco_sample(
        args.annotations, args.image
    )
    if names != KEYPOINT_NAMES:
        raise ValueError("LocalPCK requires the ordered 22-point LumbarCoco layout")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    raw_gt = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
    gt = raw_gt[:, :2]
    gt_visible = raw_gt[:, 2] > 0
    pred, scores = predict(
        args.config, args.checkpoint, image, annotation["bbox"], args.device
    )
    if pred.shape != gt.shape:
        raise ValueError(
            f"model returned {len(pred)} keypoints, but annotation has {len(gt)}"
        )
    pred_visible = scores >= args.score_thr
    correct, total = local_pck(
        gt, gt_visible, pred, args.local_pck_mode, args.local_pck_thr
    )
    return (image_path, image, annotation, names, skeleton, gt, gt_visible,
            pred, pred_visible, scores, correct, total)
