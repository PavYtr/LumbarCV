#!/usr/bin/env python3
"""Overlay LumbarCoco ground truth and model predictions on one image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model


# OpenCV uses BGR colors.
GT_COLOR = (60, 220, 60)
PRED_COLOR = (255, 80, 230)
ERROR_COLOR = (210, 210, 210)
BBOX_COLOR = (60, 180, 255)
TEXT_COLOR = (245, 245, 245)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="MMPose config")
    parser.add_argument("checkpoint", type=Path, help="trained checkpoint")
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        help=(
            "image present in the COCO file; if omitted, use the first image "
            "from the selected annotation file"
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
        help="output image (default: output/comparisons/<image>_gt_vs_pred.jpg)",
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
        help="draw landmark names next to ground-truth points",
    )
    return parser.parse_args()


def load_coco_sample(
    annotation_path: Path, image_path: Path | None
) -> tuple[Path, dict, dict, list[str], list[tuple[int, int]]]:
    with annotation_path.open(encoding="utf-8") as stream:
        coco = json.load(stream)

    images = coco.get("images", [])
    if not images:
        raise ValueError(f"no images found in {annotation_path}")

    if image_path is None:
        image_info = images[0]
        dataset_root = annotation_path.parent.parent
        image_path = dataset_root / "images" / image_info["file_name"]
        matching_images = [image_info]
    else:
        matching_images = [
            item
            for item in images
            if Path(item["file_name"]).name == image_path.name
        ]
    if not matching_images:
        examples = ", ".join(item["file_name"] for item in images[:3])
        raise ValueError(
            f"{image_path.name!r} is absent from {annotation_path}. "
            f"Use a real image name, for example: {examples}"
        )
    if len(matching_images) > 1:
        raise ValueError(
            f"{image_path.name!r} occurs more than once in {annotation_path}"
        )

    image_info = matching_images[0]
    matching_annotations = [
        item
        for item in coco.get("annotations", [])
        if item["image_id"] == image_info["id"]
    ]
    if len(matching_annotations) != 1:
        raise ValueError(
            f"expected one annotation for image id {image_info['id']}, "
            f"found {len(matching_annotations)}"
        )

    categories = {
        category["id"]: category for category in coco.get("categories", [])
    }
    annotation = matching_annotations[0]
    category = categories.get(annotation["category_id"])
    if category is None:
        raise ValueError(
            f"category {annotation['category_id']} is absent from {annotation_path}"
        )

    names = category.get("keypoints", [])
    # COCO skeleton indices are one-based; NumPy arrays are zero-based.
    skeleton = [(start - 1, end - 1) for start, end in category.get("skeleton", [])]
    return image_path, image_info, annotation, names, skeleton


def as_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def predict(
    config: Path,
    checkpoint: Path,
    image: np.ndarray,
    bbox_xywh: list[float],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # MMPose 1.3/MMEngine call torch.load without weights_only. Since PyTorch
    # 2.6 the default is True, which rejects older MMEngine checkpoints that
    # also store NumPy-backed optimizer/runtime metadata. The checkpoint is an
    # explicit CLI input, so load it in legacy mode only during init_model and
    # restore torch.load immediately afterwards.
    original_torch_load = torch.load

    def load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = load_trusted_checkpoint
    try:
        model = init_model(str(config), str(checkpoint), device=device)
    finally:
        torch.load = original_torch_load
    x, y, width, height = bbox_xywh
    bbox_xyxy = np.array([[x, y, x + width, y + height]], dtype=np.float32)
    samples = inference_topdown(model, image, bboxes=bbox_xyxy)
    if len(samples) != 1:
        raise RuntimeError(f"expected one prediction, received {len(samples)}")

    instances = samples[0].pred_instances
    keypoints = as_numpy(instances.keypoints)[0, :, :2]
    if "keypoint_scores" in instances:
        scores = as_numpy(instances.keypoint_scores)[0]
    else:
        scores = np.ones(len(keypoints), dtype=np.float32)
    return keypoints, scores


def point(point: np.ndarray) -> tuple[int, int]:
    return tuple(np.rint(point).astype(int))


def draw_skeleton(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list[tuple[int, int]],
    visible: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for start, end in skeleton:
        if start >= len(keypoints) or end >= len(keypoints):
            continue
        if visible[start] and visible[end]:
            cv2.line(
                canvas,
                point(keypoints[start]),
                point(keypoints[end]),
                color,
                thickness,
                cv2.LINE_AA,
            )


def draw_overlay(
    image: np.ndarray,
    gt: np.ndarray,
    gt_visible: np.ndarray,
    pred: np.ndarray,
    pred_visible: np.ndarray,
    scores: np.ndarray,
    bbox_xywh: list[float],
    names: list[str],
    skeleton: list[tuple[int, int]],
    show_labels: bool,
) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = max(0.8, min(width, height) / 900.0)
    line_width = max(1, round(2 * scale))
    radius = max(3, round(5 * scale))

    x, y, bbox_width, bbox_height = bbox_xywh
    cv2.rectangle(
        canvas,
        (round(x), round(y)),
        (round(x + bbox_width), round(y + bbox_height)),
        BBOX_COLOR,
        line_width,
        cv2.LINE_AA,
    )

    # Pale connector lines make both the direction and magnitude of each
    # landmark error visible without overpowering the radiograph.
    for gt_point, pred_point, is_gt_visible, is_pred_visible in zip(
        gt, pred, gt_visible, pred_visible
    ):
        if is_gt_visible and is_pred_visible:
            cv2.line(
                canvas,
                point(gt_point),
                point(pred_point),
                ERROR_COLOR,
                max(1, line_width // 2),
                cv2.LINE_AA,
            )

    skeleton_layer = canvas.copy()
    draw_skeleton(
        skeleton_layer, gt, skeleton, gt_visible, GT_COLOR, line_width + 1
    )
    draw_skeleton(
        skeleton_layer, pred, skeleton, pred_visible, PRED_COLOR, line_width
    )
    cv2.addWeighted(skeleton_layer, 0.8, canvas, 0.2, 0, canvas)

    for index, (gt_point, pred_point) in enumerate(zip(gt, pred)):
        if gt_visible[index]:
            center = point(gt_point)
            cv2.circle(canvas, center, radius + 1, (0, 0, 0), line_width + 2, cv2.LINE_AA)
            cv2.circle(canvas, center, radius, GT_COLOR, line_width, cv2.LINE_AA)
            if show_labels and index < len(names):
                cv2.putText(
                    canvas,
                    names[index],
                    (center[0] + radius + 3, center[1] - radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42 * scale,
                    GT_COLOR,
                    max(1, line_width),
                    cv2.LINE_AA,
                )
        if pred_visible[index]:
            cv2.drawMarker(
                canvas,
                point(pred_point),
                PRED_COLOR,
                cv2.MARKER_TILTED_CROSS,
                radius * 2 + 4,
                line_width + 1,
                cv2.LINE_AA,
            )

    comparable = gt_visible & pred_visible
    if np.any(comparable):
        errors = np.linalg.norm(pred[comparable] - gt[comparable], axis=1)
        mean_error = float(errors.mean())
        norm = max(float(bbox_width), float(bbox_height), 1.0)
        pck = float(np.mean(errors <= 0.05 * norm))
        metrics = f"Mean error: {mean_error:.1f}px   PCK@0.05: {pck:.3f}"
    else:
        metrics = "No comparable keypoints"

    legend = (
        f"GT: green circles   Prediction: magenta crosses   "
        f"visible predictions: {int(pred_visible.sum())}/{len(scores)}"
    )
    font_scale = max(0.45, 0.65 * scale)
    row_height = max(25, round(31 * scale))
    banner_height = row_height * 2 + 8
    banner = canvas.copy()
    cv2.rectangle(banner, (0, 0), (width, banner_height), (15, 15, 15), -1)
    cv2.addWeighted(banner, 0.78, canvas, 0.22, 0, canvas)
    cv2.putText(
        canvas,
        legend,
        (12, row_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_COLOR,
        max(1, line_width),
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        metrics,
        (12, row_height * 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        TEXT_COLOR,
        max(1, line_width),
        cv2.LINE_AA,
    )
    return canvas


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

    visualization = draw_overlay(
        image,
        gt,
        gt_visible,
        pred,
        pred_visible,
        scores,
        annotation["bbox"],
        names,
        skeleton,
        args.show_labels,
    )
    output = args.output or Path("output/comparisons") / (
        f"{image_path.stem}_gt_vs_pred.jpg"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), visualization):
        raise OSError(f"could not write visualization: {output}")
    print(f"Visualization written to {output}")


if __name__ == "__main__":
    main()
