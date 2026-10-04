#!/usr/bin/env python3
"""Save a side-by-side comparison of a training sample before/after augmentation."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from mmcv.transforms import Compose
from mmengine.config import Config
from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config", nargs="?", type=Path,
        default=Path("configs/rtmpose-m_lumbar-coco2000_bbox-localpck.py"),
        help="training config (default: bbox augmentation config)",
    )
    parser.add_argument("--index", type=int, default=0, help="train sample index")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--output", type=Path,
        default=Path("output/visualizations/augmentation_comparison.jpg"),
        help="path to the comparison image",
    )
    parser.add_argument(
        "--hide-keypoints", action="store_true",
        help="show only the images without landmark markers",
    )
    return parser.parse_args()


def pipelines(config: Config) -> tuple[Compose, Compose]:
    pipeline = config.train_dataloader.dataset.pipeline
    types = [step.get("type") for step in pipeline]
    if types.count("GetBBoxCenterScale") != 1 or types.count("TopdownAffine") != 1:
        raise ValueError("train pipeline must contain one GetBBoxCenterScale and one TopdownAffine")
    start = types.index("GetBBoxCenterScale")
    end = types.index("TopdownAffine")
    if start >= end:
        raise ValueError("GetBBoxCenterScale must precede TopdownAffine")
    # The left panel uses the same image loading, bbox padding and affine crop.
    # The right panel also includes every training augmentation between them.
    baseline = [*pipeline[:start + 1], pipeline[end]]
    augmented = pipeline[:end + 1]
    return Compose(baseline), Compose(augmented)


def render_panel(sample: dict, title: str, hide_keypoints: bool) -> np.ndarray:
    image = sample["img"].copy()
    width = image.shape[1]
    if not hide_keypoints:
        points = np.asarray(sample["transformed_keypoints"])[0]
        visible = np.asarray(sample["keypoints_visible"])[0]
        if visible.ndim > 1:
            visible = visible[:, 0]
        valid = (visible > 0) & np.isfinite(points).all(axis=1)
        for start, end in sample.get("skeleton_links", []):
            if valid[start] and valid[end]:
                cv2.line(image, tuple(np.rint(points[start]).astype(int)),
                         tuple(np.rint(points[end]).astype(int)),
                         (0, 215, 255), 1, cv2.LINE_AA)
        for index, point in enumerate(points):
            if valid[index]:
                color = (40, 220, 40) if index % 2 else (35, 145, 255)
                cv2.circle(image, tuple(np.rint(point).astype(int)), 3,
                           color, -1, cv2.LINE_AA)

    header = np.full((46, width, 3), (35, 35, 35), dtype=np.uint8)
    cv2.putText(header, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack((header, image))


def make_comparison(data: dict, baseline_pipeline: Compose,
                    augmented_pipeline: Compose, seed: int,
                    hide_keypoints: bool = False) -> np.ndarray:
    before = baseline_pipeline(copy.deepcopy(data))
    random.seed(seed)
    np.random.seed(seed)
    after = augmented_pipeline(copy.deepcopy(data))
    if before is None or after is None:
        raise RuntimeError("training pipeline rejected the selected sample")
    if before["img"].shape != after["img"].shape:
        raise ValueError("baseline and augmented crops have different image sizes")
    return np.hstack((
        render_panel(before, "Before", hide_keypoints),
        render_panel(after, "After", hide_keypoints),
    ))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    register_all_modules()
    config = Config.fromfile(str(args.config))

    dataset_config = config.train_dataloader.dataset.copy()
    dataset_config.pipeline = []
    dataset = DATASETS.build(dataset_config)
    if not 0 <= args.index < len(dataset):
        raise ValueError(f"--index must be between 0 and {len(dataset) - 1}")

    baseline_pipeline, augmented_pipeline = pipelines(config)
    data = dataset.get_data_info(args.index)
    comparison = make_comparison(
        data, baseline_pipeline, augmented_pipeline, args.seed,
        args.hide_keypoints)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), comparison):
        raise OSError(f"could not write image: {args.output}")
    print(f"Sample: {data['img_path']} (index {args.index}, seed {args.seed})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
