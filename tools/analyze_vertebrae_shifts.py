#!/usr/bin/env python3
"""Analyze original BUU COCO landmarks, independently of the training pipeline.

--annotations accepts JSON files and/or directories (all *.json recursively).
Repeated images are merged by file_name, with IDs and dimensions checked.
Identical landmark records are deduplicated; conflicting instances produce NaN
for all five pairs rather than choosing an arbitrary spine annotation.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs._base_.datasets.lumbar_coco import KEYPOINT_NAMES

LEVELS = ('L1_L2', 'L2_L3', 'L3_L4', 'L4_L5', 'L5_S1')
PAIR_INDICES = tuple(
    tuple(KEYPOINT_NAMES.index(name) for name in (
        f'L{level}BF', f'L{level}BB',
        f'L{level + 1}TF' if level < 5 else 'S1TF',
        f'L{level + 1}TB' if level < 5 else 'S1TB'))
    for level in range(1, 6)
)
PERCENTILES = (50, 75, 80, 90, 95, 99)


def relative_shift(upper, lower):
    """C1/C2 for two (2, 2) endplates; raise ValueError if undefined.

    Sorting projected endpoints makes the result invariant to endpoint order,
    rotation and reflection. Subtract the lower origin for numerical stability.
    No center-distance approximation and no clipping of ratios above one.
    """
    upper, lower = np.asarray(upper, dtype=float), np.asarray(lower, dtype=float)
    if upper.shape != (2, 2) or lower.shape != (2, 2):
        raise ValueError('invalid_plate_shape')
    if not np.isfinite(upper).all() or not np.isfinite(lower).all():
        raise ValueError('nonfinite_coordinate')
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        # Choose the lower endpoint at image-left (y breaks vertical ties).
        lower = lower[np.lexsort((lower[:, 1], lower[:, 0]))]
        delta = lower[1] - lower[0]
        length = np.hypot(*delta)
        if not np.isfinite(length):
            raise ValueError('nonfinite_plate_length')
        if length <= 0:
            raise ValueError('zero_lower_plate_length')
        upper_length = np.hypot(*(upper[1] - upper[0]))
        if not np.isfinite(upper_length):
            raise ValueError('nonfinite_plate_length')
        if upper_length <= 0:
            raise ValueError('zero_upper_plate_length')
        u = delta / length
        upper_left, upper_right = np.sort((upper - lower[0]) @ u)
        lower_left, lower_right = np.sort((lower - lower[0]) @ u)
        if not np.isfinite([upper_left, upper_right, lower_left, lower_right]).all():
            raise ValueError('nonfinite_projection')
        result = max(0., lower_right - upper_right, upper_left - lower_left) / length
    if not np.isfinite(result):
        raise ValueError('nonfinite_shift')
    return float(result)


def annotation_key(annotation):
    # IDs, bbox and segmentation do not affect this landmark measurement.
    return json.dumps({k: annotation.get(k) for k in
                       ('category_id', 'keypoints', 'iscrowd')}, sort_keys=True)


def load_images(paths):
    """Validate layout and group instances across splits without image repeats."""
    images, ids, sources = {}, {}, []
    duplicate_images = duplicate_annotations = 0
    for path in paths:
        raw = path.read_bytes()
        coco = json.loads(raw)
        categories = {c['id']: c for c in coco['categories']}
        if not categories or any(c.get('keypoints') != KEYPOINT_NAMES
                                 for c in categories.values()):
            raise ValueError(f'{path}: unexpected category/keypoint layout')
        local = {}
        for image in coco['images']:
            image_id, name = image['id'], image['file_name']
            if image_id in local and local[image_id] != name:
                raise ValueError(f'{path}: image ID {image_id} has multiple names')
            if image_id in ids and ids[image_id] != name:
                raise ValueError(f'{path}: image ID collision for {image_id}')
            ids[image_id] = local[image_id] = name
            if name in images:
                old = images[name]['image']
                if any(old.get(k) != image.get(k) for k in ('id', 'width', 'height')):
                    raise ValueError(f'{path}: conflicting metadata for {name}')
                duplicate_images += 1
            else:
                images[name] = dict(image=image, annotations={})
        for annotation in coco['annotations']:
            if annotation['image_id'] not in local:
                raise ValueError(f'{path}: annotation references unknown image')
            if annotation.get('category_id') not in categories:
                raise ValueError(f'{path}: annotation references unknown category')
            group = images[local[annotation['image_id']]]['annotations']
            key = annotation_key(annotation)
            if key in group:
                duplicate_annotations += 1
            group[key] = annotation
        sources.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                            images=len(coco['images']), annotations=len(coco['annotations'])))
    audit = dict(sources=sources, duplicate_image_records=duplicate_images,
                 duplicate_annotation_records=duplicate_annotations,
                 images_with_multiple_distinct_annotations=sum(
                     len(g['annotations']) > 1 for g in images.values()))
    return images, audit


def measure_pair(annotation, image, indices):
    if annotation.get('iscrowd', 0):
        raise ValueError('crowd_annotation')
    flat = annotation.get('keypoints')
    if not isinstance(flat, list) or len(flat) > 66 or len(flat) % 3:
        raise ValueError('invalid_keypoint_array')
    points = []
    for index in indices:
        if 3 * index + 3 > len(flat):
            raise ValueError('missing_keypoint')
        try:
            point = np.asarray(flat[3 * index:3 * index + 3], dtype=float)
        except (TypeError, ValueError):
            raise ValueError('nonnumeric_keypoint') from None
        if not np.isfinite(point).all():
            raise ValueError('nonfinite_keypoint')
        if point[2] == 0:
            raise ValueError('unlabelled_keypoint')
        if point[2] == 1:
            raise ValueError('invisible_keypoint')
        if point[2] != 2:
            raise ValueError('invalid_visibility')
        try:
            bounds = np.asarray([image['width'], image['height']], dtype=float)
        except (KeyError, ValueError, TypeError):
            raise ValueError('invalid_image_dimensions') from None
        if not np.isfinite(bounds).all() or np.any(bounds <= 0):
            raise ValueError('invalid_image_dimensions')
        if np.any(point[:2] < 0) or np.any(point[:2] >= bounds):
            raise ValueError('coordinate_outside_image')
        points.append(point[:2])
    return relative_shift(points[:2], points[2:])


def analyze(images):
    rows, issues = [], []
    for name, group in sorted(images.items()):
        image = group['image']
        annotations = list(group['annotations'].values())
        row = dict(image_id=image['id'], image_name=name)
        for level, indices in zip(LEVELS, PAIR_INDICES):
            try:
                if not annotations:
                    raise ValueError('missing_annotation')
                if len(annotations) > 1:
                    raise ValueError('ambiguous_multiple_annotations')
                row[level] = measure_pair(annotations[0], image, indices)
            except ValueError as error:
                row[level] = float('nan')
                issues.append(dict(image_id=image['id'], image_name=name,
                                   level=level, reason=str(error)))
        valid = [row[level] for level in LEVELS if np.isfinite(row[level])]
        row['max_shift'] = max(valid) if valid else float('nan')
        rows.append(row)
    return rows, issues


def summarize(values):
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    return dict(total_count=len(values), valid_count=len(valid),
                invalid_count=len(values) - len(valid),
                minimum=float(valid.min()) if len(valid) else None,
                maximum=float(valid.max()) if len(valid) else None,
                mean=float(valid.mean()) if len(valid) else None,
                median=float(np.median(valid)) if len(valid) else None,
                std=float(valid.std(ddof=0)) if len(valid) else None)


def statistics(rows, issues, audit):
    maxima = np.array([row['max_shift'] for row in rows])
    valid = maxima[np.isfinite(maxima)]
    thresholds = {}
    for p in PERCENTILES:
        threshold = float(np.percentile(valid, p)) if len(valid) else None
        thresholds[str(p)] = dict(value=threshold, images_above=int(
            np.count_nonzero(valid > threshold)) if threshold is not None else 0)
    return dict(
        processed_images=len(rows),
        all_shifts=summarize([row[level] for row in rows for level in LEVELS]),
        max_shift=dict(**summarize(maxima), percentiles=thresholds),
        by_level={level: summarize([row[level] for row in rows]) for level in LEVELS},
        invalid_reasons=dict(Counter(issue['reason'] for issue in issues)),
        invalid_measurements=issues, input_audit=audit,
        definition='C1=max(0, lower_right-upper_right, upper_left-lower_left); '
                   'C2=lower_right-lower_left; S=C1/C2 in lower top-endplate axis',
        visibility_policy='Only COCO v=2 is accepted; v=0 and v=1 are invalid',
        std_ddof=0, percentile_method='linear', threshold_comparison='strictly greater',
        keypoint_names=KEYPOINT_NAMES,
        pair_indices={level: list(indices) for level, indices in zip(LEVELS, PAIR_INDICES)},
        json_missing_values='null (CSV uses NaN)')


def plot_histograms(rows, stats, output, bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_values = np.array([row[level] for row in rows for level in LEVELS])
    all_values = all_values[np.isfinite(all_values)]
    edges = np.linspace(0, max(float(all_values.max()) if len(all_values) else 0, .01), bins + 1)

    def draw(ax, values, title):
        values = np.asarray(values)
        values = values[np.isfinite(values)]
        ax.hist(values, bins=edges, color='steelblue', edgecolor='white')
        ax.set(title=f'{title} (n={len(values)})', xlabel='Relative protrusion S = C1 / C2',
               ylabel='Count')
        ax.grid(axis='y', alpha=.3)
        ax.set_axisbelow(True)

    for field, title, filename in (
        (None, 'All adjacent vertebral pairs', 'all_shifts_hist.png'),
        ('max_shift', 'Maximum shift per image', 'max_shifts_hist.png')):
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        draw(ax, all_values if field is None else [row[field] for row in rows], title)
        if field:
            for p, color in zip((80, 90, 95), ('darkorange', 'crimson', 'purple')):
                value = stats['max_shift']['percentiles'][str(p)]['value']
                if value is not None:
                    ax.axvline(value, color=color, linestyle='--', label=f'P{p}: {value:.4f}')
            if stats['max_shift']['valid_count']:
                ax.legend()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True, layout='constrained')
    for ax, level in zip(axes.flat, LEVELS):
        draw(ax, [row[level] for row in rows], level.replace('_', '–'))
    axes.flat[-1].set_visible(False)
    fig.suptitle('Relative shift by vertebral level (common bins)')
    fig.savefig(output / 'shifts_by_level.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, nargs='+', required=True,
                        help='COCO JSON files or directories; include every split')
    parser.add_argument('--output-dir', type=Path, default=Path('reports/buu_vertebrae_shifts'))
    parser.add_argument('--bins', type=int, default=40)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error('--bins must be positive')
    paths = set()
    for path in args.annotations:
        if not path.exists():
            parser.error(f'annotation path does not exist: {path}')
        if path.is_dir():
            paths.update(p.resolve() for p in path.rglob('*.json'))
        else:
            paths.add(path.resolve())
    if not paths:
        parser.error('no annotation JSON files found')
    images, audit = load_images(sorted(paths))
    if not images:
        parser.error('no images in annotations')
    rows, issues = analyze(images)
    stats = statistics(rows, issues, audit)
    # Guard against overwriting an input file through an output-path collision.
    output = args.output_dir.resolve()
    if output / 'shift_statistics.json' in paths:
        parser.error('output would overwrite input annotations')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'vertebrae_shifts.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['image_id', 'image_name', *LEVELS, 'max_shift'])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: 'NaN' if isinstance(v, float) and np.isnan(v) else v
                             for k, v in row.items()})
    (output / 'shift_statistics.json').write_text(
        json.dumps(stats, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    plot_histograms(rows, stats, output, args.bins)
    print(json.dumps({k: v for k, v in stats.items() if k != 'invalid_measurements'},
                     indent=2, allow_nan=False))
    print(f'Saved analysis to {output}')


if __name__ == '__main__':
    main()
