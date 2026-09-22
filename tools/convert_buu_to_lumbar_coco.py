#!/usr/bin/env python3
"""Convert the lateral BUU-LSPINE_400 annotations to COCO keypoints.

BUU stores one endplate per CSV row as ``x1,y1,x2,y2,class``.  All lateral
images face image-left, so the left endpoint is anterior/front and the right
endpoint is posterior/back.  (A few source CSVs reverse their endpoint order.)
Rows are ordered L1 top, L1 bottom, ..., L5 bottom, S1 top.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from datetime import date
from pathlib import Path

from PIL import Image


KEYPOINT_NAMES = [
    *(f"L{level}{plate}{side}" for level in range(1, 6)
      for plate in ("T", "B") for side in ("F", "B")),
    "S1TF",
    "S1TB",
]

# COCO category skeleton indices are one-based.
SKELETON = []
for _level in range(5):
    _base = _level * 4 + 1
    SKELETON.extend([
        [_base, _base + 1],       # top endplate
        [_base + 2, _base + 3],   # bottom endplate
        [_base, _base + 2],       # anterior vertebral edge
        [_base + 1, _base + 3],   # posterior vertebral edge
    ])
SKELETON.append([21, 22])          # S1 top endplate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path("BUU-LSPINE_400"),
        help="BUU-LSPINE_400 root containing the LA directory")
    parser.add_argument(
        "--output", type=Path, default=Path("data/LumbarCoco"),
        help="output LumbarCoco root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split", type=float, nargs=3, default=(0.8, 0.1, 0.1),
        metavar=("TRAIN", "VAL", "TEST"),
        help="train/val/test fractions (default: 0.8 0.1 0.1)")
    parser.add_argument(
        "--image-mode", choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="how images are exposed under OUTPUT/images")
    parser.add_argument(
        "--bbox-padding", type=float, default=0.10,
        help="fractional padding around the landmark bounding box")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing annotation JSON files/image link")
    return parser.parse_args()


def read_keypoints(csv_path: Path, width: int, height: int) -> tuple[list[float], list[float]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 11:
        raise ValueError(f"{csv_path}: expected 11 rows, found {len(rows)}")

    keypoints: list[float] = []
    xy: list[tuple[float, float]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 5:
            raise ValueError(
                f"{csv_path}:{row_number}: expected 5 values, found {len(row)}")
        x1, y1, x2, y2 = map(float, row[:4])
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError(f"{csv_path}:{row_number}: non-finite coordinate")
        if not (0 <= x1 < width and 0 <= x2 < width and
                0 <= y1 < height and 0 <= y2 < height):
            raise ValueError(f"{csv_path}:{row_number}: coordinate outside image")
        # BUU's endpoint ordering is inconsistent in three LA files. Since all
        # views are normalized to face image-left, sort each pair by x rather
        # than assuming the first CSV endpoint is always anterior.
        if x1 > x2:
            x1, y1, x2, y2 = x2, y2, x1, y1

        # Visibility 2 means labelled and visible in the COCO convention.
        keypoints.extend((x1, y1, 2, x2, y2, 2))
        xy.extend(((x1, y1), (x2, y2)))

    return keypoints, [coord for point in xy for coord in point]


def padded_bbox(flat_xy: list[float], width: int, height: int,
                padding: float) -> list[float]:
    xs, ys = flat_xy[0::2], flat_xy[1::2]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    pad_x = max(x1 - x0, 1.0) * padding
    pad_y = max(y1 - y0, 1.0) * padding
    x0, y0 = max(0.0, x0 - pad_x), max(0.0, y0 - pad_y)
    x1, y1 = min(float(width), x1 + pad_x), min(float(height), y1 + pad_y)
    return [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]


def coco_document(records: list[dict]) -> dict:
    images = []
    annotations = []
    for record in records:
        image_id = record["id"]
        images.append({
            "id": image_id,
            "file_name": record["file_name"],
            "width": record["width"],
            "height": record["height"],
        })
        bbox = record["bbox"]
        x, y, w, h = bbox
        annotations.append({
            "id": image_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": bbox,
            "area": round(w * h, 4),
            "iscrowd": 0,
            "num_keypoints": len(KEYPOINT_NAMES),
            "keypoints": record["keypoints"],
            "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]],
        })

    return {
        "info": {
            "description": "LumbarCoco: BUU-LSPINE_400 lateral lumbar landmarks",
            "version": "1.0",
            "year": date.today().year,
            "date_created": date.today().isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": "lumbar_spine",
            "supercategory": "spine",
            "keypoints": KEYPOINT_NAMES,
            "skeleton": SKELETON,
        }],
    }


def prepare_images(source_la: Path, images_dir: Path, mode: str,
                   overwrite: bool) -> None:
    if images_dir.is_symlink() or images_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{images_dir} already exists; use --overwrite")
        if images_dir.is_symlink() or images_dir.is_file():
            images_dir.unlink()
        else:
            shutil.rmtree(images_dir)

    if mode == "symlink":
        images_dir.symlink_to(source_la.resolve(), target_is_directory=True)
        return

    images_dir.mkdir(parents=True)
    for image_path in sorted(source_la.glob("*.jpg")):
        destination = images_dir / image_path.name
        if mode == "copy":
            shutil.copy2(image_path, destination)
        else:
            destination.hardlink_to(image_path.resolve())


def main() -> None:
    args = parse_args()
    if any(value < 0 for value in args.split) or not math.isclose(sum(args.split), 1.0):
        raise ValueError("--split values must be non-negative and sum to 1")
    if args.bbox_padding < 0:
        raise ValueError("--bbox-padding must be non-negative")

    source_la = args.source / "LA"
    csv_paths = sorted(source_la.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no CSV files found in {source_la}")

    records = []
    for csv_path in csv_paths:
        image_path = csv_path.with_suffix(".jpg")
        if not image_path.is_file():
            raise FileNotFoundError(f"missing image for {csv_path}: {image_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        keypoints, flat_xy = read_keypoints(csv_path, width, height)
        patient_id = csv_path.stem.split("-", 1)[0]
        records.append({
            "id": int(patient_id),
            "file_name": image_path.name,
            "width": width,
            "height": height,
            "keypoints": keypoints,
            "bbox": padded_bbox(flat_xy, width, height, args.bbox_padding),
        })

    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("patient/image IDs are not unique")

    shuffled = records.copy()
    random.Random(args.seed).shuffle(shuffled)
    n_train = int(len(shuffled) * args.split[0])
    n_val = int(len(shuffled) * args.split[1])
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }

    annotations_dir = args.output / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_records in splits.items():
        output_path = annotations_dir / f"lumbar_keypoints_{split_name}.json"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} already exists; use --overwrite")
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(coco_document(split_records), stream, indent=2)
            stream.write("\n")

    prepare_images(source_la, args.output / "images", args.image_mode, args.overwrite)
    print(
        f"Converted {len(records)} LA images: "
        + ", ".join(f"{name}={len(items)}" for name, items in splits.items()))
    print(f"LumbarCoco written to {args.output}")


if __name__ == "__main__":
    main()
