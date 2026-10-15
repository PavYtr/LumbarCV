#!/usr/bin/env python3
"""Collect real-image margins and L1/S1 lean from a COCO training split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from augmentations.vertebrae_crop import AXIS_INDICES, KEYPOINT_NAMES, SIDES, measure


def summarize(samples):
    """Arithmetic means over valid training records; average ratios directly."""
    if not samples:
        raise ValueError('Cannot summarize empty statistics')
    angles = np.asarray([sample['angle_deg'] for sample in samples], dtype=float)
    return dict(
        count=len(samples),
        angle_deg=float(np.mean(angles)),
        absolute_angle_deg=float(np.mean(np.abs(angles))),
        margins_px={side: float(np.mean([s['margins_px'][side] for s in samples]))
                    for side in SIDES},
        margins_relative={side: float(np.mean([s['margins_relative'][side] for s in samples]))
                          for side in SIDES},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=Path(
        'data/LumbarCoco2000/annotations/lumbar_keypoints_train.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/vertebrae_statistics_train.json'))
    args = parser.parse_args()
    raw = args.annotations.read_bytes()
    coco = json.loads(raw)
    if any(c['keypoints'] != KEYPOINT_NAMES for c in coco['categories']):
        raise ValueError('Dataset keypoint names/order differ from expected lumbar layout')
    images = {image['id']: image for image in coco['images']}
    samples, skipped = [], []
    for annotation in coco['annotations']:
        image = images[annotation['image_id']]
        try:
            if annotation.get('iscrowd', 0):
                raise ValueError('Crowd annotation')
            kp = np.asarray(annotation['keypoints'], dtype=float).reshape(22, 3)
            labelled = kp[:, 2] > 0
            if np.any(kp[labelled, :2] < 0) or np.any(
                    kp[labelled, :2] >= [image['width'], image['height']]):
                raise ValueError('Labelled keypoint outside source image')
            sample = measure(kp[:, :2], kp[:, 2], image['width'], image['height'])
            sample.update(image_id=image['id'], annotation_id=annotation['id'],
                          file_name=image['file_name'], width=image['width'], height=image['height'])
            samples.append(sample)
        except ValueError as error:
            skipped.append(dict(annotation_id=annotation['id'], reason=str(error)))
    if not samples:
        raise ValueError('No valid training samples')
    document = dict(
        schema_version=1, annotations=str(args.annotations),
        annotations_sha256=hashlib.sha256(raw).hexdigest(),
        keypoint_names=KEYPOINT_NAMES, axis_indices=list(AXIS_INDICES),
        angle_convention='degrees from downward vertical; positive S1 right of L1',
        margin_definition='minimum clearance over each KP-box side to rotated original image polygon; negative clearance clamped to zero',
        relative_normalization='left/right: verticalized KP width; top/bottom: verticalized KP height',
        sampling='independent uniform margins [0, mean relative margin]; angle +/- mean absolute angle',
        mean=summarize(samples), samples=samples, skipped=skipped)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    angles = [s['angle_deg'] for s in samples]
    print(f'Saved {len(samples)} samples; skipped {len(skipped)}: {args.output}')
    print('Mean parameters: ' + json.dumps(document['mean']))
    print(f'Lean: min={min(angles):.2f}, median={np.median(angles):.2f}, max={max(angles):.2f} deg')


if __name__ == '__main__':
    main()
