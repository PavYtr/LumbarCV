#!/usr/bin/env python3
"""Save LocalPCK overlays and a contact sheet of selected test images."""

from __future__ import annotations

import argparse
import json
import os
import random
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Config custom_imports live at the project root.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model

from local_pck_visualization import local_pck, local_pck_radii, metric_label
from visualize_gt_vs_prediction import as_numpy, draw_overlay, load_coco_sample


DEFAULT_RUN = ROOT / "work_dirs/rtmpose-m-lumbar-coco2000-randombbox-localpck"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="number of test images")
    parser.add_argument("--seed", type=int, help="selection seed (default: new each run)")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint (default: best in run directory)")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "configs/rtmpose-m_lumbar-coco2000_bbox-localpck.py",
    )
    parser.add_argument(
        "--annotations", type=Path,
        default=ROOT / "data/LumbarCoco2000/annotations/lumbar_keypoints_test.json",
    )
    parser.add_argument("--output-dir", type=Path,
                        help="directory for individual images")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.2)
    parser.add_argument("--local-pck-thr", type=float, default=0.1)
    parser.add_argument("--columns", type=int, default=2,
                        help="number of columns in grid.jpg (default: 2)")
    return parser.parse_args()


def make_grid_tile(
    overlay: np.ndarray, bbox: list[float], title: str, subtitle: str
) -> np.ndarray:
    """Crop around the lumbar region so landmarks remain legible in the grid."""
    image_height, image_width = overlay.shape[:2]
    x, y, width, height = bbox
    x0 = max(0, int(x - width * 0.2))
    y0 = max(0, int(y - height * 0.15))
    x1 = min(image_width, int(x + width * 1.2))
    y1 = min(image_height, int(y + height * 1.15))
    crop = overlay[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"invalid bounding box for grid tile: {bbox}")
    tile_width, tile_height, header_height = 640, 780, 92
    tile = np.full((tile_height, tile_width, 3), 20, dtype=np.uint8)
    available_height = tile_height - header_height
    scale = min(tile_width / crop.shape[1], available_height / crop.shape[0])
    resized = cv2.resize(
        crop, (max(1, round(crop.shape[1] * scale)),
               max(1, round(crop.shape[0] * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    left = (tile_width - resized.shape[1]) // 2
    top = header_height + (available_height - resized.shape[0]) // 2
    tile[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    cv2.putText(tile, title, (12, 35), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(tile, subtitle, (12, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (245, 245, 245), 1, cv2.LINE_AA)
    return tile


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.columns < 1:
        raise ValueError("--columns must be positive")
    if not 0 <= args.score_thr <= 1 or not np.isfinite(args.local_pck_thr) or args.local_pck_thr < 0:
        raise ValueError("invalid score or LocalPCK threshold")

    checkpoint = args.checkpoint
    if checkpoint is None:
        candidates = list(args.run_dir.glob("best_*.pth"))
        if not candidates:
            raise FileNotFoundError(f"no best checkpoint in {args.run_dir}")
        checkpoint = max(candidates, key=lambda path: path.stat().st_mtime)
    with args.annotations.open(encoding="utf-8") as stream:
        images = json.load(stream)["images"]
    if args.count > len(images):
        raise ValueError(f"requested {args.count} images, test set has {len(images)}")
    seed = args.seed if args.seed is not None else secrets.randbelow(1_000_000_000)
    selected = random.Random(seed).sample(images, args.count)
    output_dir = (args.output_dir or ROOT / "output/test_visualizations" /
                  checkpoint.stem / f"seed_{seed}_count_{args.count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selection seed: {seed}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)

    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = init_model(str(args.config), str(checkpoint), device=device)
    tiles = []
    for index, info in enumerate(selected, 1):
        image_path = args.annotations.parent.parent / "images" / info["file_name"]
        _, _, annotation, names, skeleton = load_coco_sample(args.annotations, image_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read {image_path}")
        x, y, width, height = annotation["bbox"]
        bbox = np.array([[x, y, x + width, y + height]], dtype=np.float32)
        samples = inference_topdown(model, image, bboxes=bbox)
        if len(samples) != 1:
            raise RuntimeError(f"expected one prediction for {image_path}")
        instances = samples[0].pred_instances
        pred = as_numpy(instances.keypoints)[0, :, :2]
        scores = (as_numpy(instances.keypoint_scores)[0]
                  if "keypoint_scores" in instances
                  else np.ones(len(pred), dtype=np.float32))
        gt_raw = np.asarray(annotation["keypoints"], dtype=np.float32).reshape(-1, 3)
        gt, visible = gt_raw[:, :2], gt_raw[:, 2] > 0
        if pred.shape != gt.shape:
            raise ValueError(f"keypoint count differs for {image_path}")
        correct, total = local_pck(gt, visible, pred, threshold=args.local_pck_thr)
        radii = local_pck_radii(gt, visible, threshold=args.local_pck_thr)
        label = metric_label(correct, total, "segment", args.local_pck_thr)
        overlay = draw_overlay(
            image, gt, visible, pred, scores >= args.score_thr, scores,
            annotation["bbox"], names, skeleton, False,
            metrics=label, tolerance_radii=radii, local_pck_only=True,
        )
        path = output_dir / f"{index:02d}_{image_path.stem}_overlay.jpg"
        if not cv2.imwrite(str(path), overlay):
            raise OSError(f"could not write {path}")
        grid_overlay = draw_overlay(
            image, gt, visible, pred, scores >= args.score_thr, scores,
            annotation["bbox"], names, skeleton, False,
            metrics=label, tolerance_radii=radii, show_banner=False,
            local_pck_only=True,
        )
        tiles.append(make_grid_tile(
            grid_overlay, annotation["bbox"], f"{index:02d}  {image_path.stem}", label
        ))
        print(f"Saved {index}/{args.count}: {path}", flush=True)
    blank = np.full_like(tiles[0], 20)
    rows = [
        np.hstack(tiles[start:start + args.columns] +
                  [blank] * max(0, args.columns - len(tiles[start:start + args.columns])))
        for start in range(0, len(tiles), args.columns)
    ]
    grid_path = output_dir / "grid.jpg"
    if not cv2.imwrite(str(grid_path), np.vstack(rows)):
        raise OSError(f"could not write {grid_path}")
    print(f"Grid saved: {grid_path}", flush=True)


if __name__ == "__main__":
    main()
