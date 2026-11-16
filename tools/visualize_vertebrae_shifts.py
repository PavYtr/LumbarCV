#!/usr/bin/env python3
"""Draw all five signed posterior-corner measurements on original BUU images.

Each PNG shows one radiograph with five constructions and offset percentage labels.
Uses the same keypoint mapping, validity checks and signed ratio as the
whole-dataset analyzer; no model or training pipeline is loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_vertebrae_shifts import LEVELS, PAIR_INDICES, load_images, measure_pair

PLATE_COLOR = '#36e6ed'
PERP_COLOR = '#ff67e6'
SHIFT_COLOR = '#ffcc33'
POINT_COLOR = '#ff726a'


def build_measurements(group):
    """Construct P,A,B,Q in source pixels, preserving the analyzer's validity policy."""
    measurements = []
    annotations = list(group['annotations'].values())
    for level, indices in zip(LEVELS, PAIR_INDICES):
        entry = dict(level=level, indices=list(indices), valid=False)
        try:
            if not annotations:
                raise ValueError('missing_annotation')
            if len(annotations) != 1:
                raise ValueError('ambiguous_multiple_annotations')
            annotation = annotations[0]
            shift = measure_pair(annotation, group['image'], indices)
            flat = annotation['keypoints']
            p, a, b = np.array([flat[3 * i:3 * i + 2] for i in indices], dtype=float)
            length = float(np.linalg.norm(b - a))
            u = (b - a) / length
            q = a + np.dot(p - a, u) * u
            entry.update(valid=True, P=p.tolist(), A=a.tolist(), B=b.tolist(), Q=q.tolist(),
                         lower_plate_length_px=length,
                         signed_displacement_px=shift * length,
                         shift=shift, percent=100 * shift)
        except ValueError as error:
            entry.update(shift=None, percent=None, reason=str(error))
        measurements.append(entry)
    return measurements


def draw_construction(ax, entry):
    """Plate AB, perpendicular PQ and signed displacement QB in image coordinates."""
    p, a, b, q = (np.array(entry[key]) for key in ('P', 'A', 'B', 'Q'))
    length = entry['lower_plate_length_px']
    u = (b - a) / length
    t = float(np.dot(q - a, u))
    extension = .12 * length
    ends = [a + (min(0, t) - extension) * u,
            a + (max(length, t) + extension) * u]
    ax.plot(*np.array(ends).T, color=PLATE_COLOR, linestyle=':', linewidth=1, alpha=.7)
    ax.plot(*np.array([a, b]).T, color=PLATE_COLOR, linewidth=2.5, zorder=3)
    ax.plot(*np.array([p, q]).T, color=PERP_COLOR, linestyle='--', linewidth=2, zorder=4)
    ax.plot(*np.array([q, b]).T, color=SHIFT_COLOR, linewidth=1.8, zorder=5)
    # Arrow follows Q -> B; sign is measured along anatomical A -> B.
    if np.linalg.norm(b - q) > .015 * length:
        ax.annotate('', xy=b, xytext=q,
                    arrowprops=dict(arrowstyle='->', color=SHIFT_COLOR, lw=1,
                                    mutation_scale=6, shrinkA=0, shrinkB=0), zorder=6)
    ax.scatter(*np.array([a, b]).T, c=PLATE_COLOR, s=7, zorder=7, edgecolors='black', linewidths=.3)
    ax.scatter(*p, c=POINT_COLOR, s=10, zorder=8, edgecolors='black', linewidths=.3)
    ax.scatter(*q, c=SHIFT_COLOR, marker='D', s=7, zorder=9, edgecolors='black', linewidths=.3)
    # Right-angle marker at Q, shortened if the actual perpendicular is small.
    perpendicular = p - q
    distance = float(np.linalg.norm(perpendicular))
    if distance > 1e-8:
        normal = perpendicular / distance
        tangent = u if t <= length else -u
        size = min(.045 * length, .25 * distance)
        square = np.array([q + size * tangent, q + size * (tangent + normal), q + size * normal])
        ax.plot(*square.T, color='white', linewidth=1, zorder=6)


def render_image(image, image_info, measurements, output, dpi=140):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    height, width = image.shape[:2]
    fig = plt.figure(figsize=(10, 10 * height / width), facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_xlim(-.5, width - .5)
    ax.set_ylim(height - .5, -.5)
    invalid = []
    for entry in measurements:
        if not entry['valid']:
            invalid.append(f"{entry['level'].replace('_', '-')}: NaN ({entry['reason']})")
            continue
        draw_construction(ax, entry)
        points = np.array([entry[key] for key in ('P', 'A', 'B', 'Q')])
        length = entry['lower_plate_length_px']
        # Keep text outside the construction, with an additional 44-point gap.
        # Switch sides near the image boundary to keep the annotation visible.
        right = float(points[:, 0].max()) + .12 * length
        left = float(points[:, 0].min()) - .12 * length
        on_right = right < .73 * width
        anchor = (right if on_right else left,
                  (entry['P'][1] + entry['Q'][1]) / 2)
        ax.annotate(f"{entry['level'].replace('_', '-')}\n{entry['percent']:+.2f}%",
                    xy=anchor, xytext=(44 if on_right else -44, 0),
                    textcoords='offset points', ha='left' if on_right else 'right',
                    va='center', color='white', fontsize=13,
                    bbox=dict(facecolor='black', alpha=.7, edgecolor='none', pad=4),
                    arrowprops=dict(arrowstyle='-', color='white', alpha=.45,
                                    linewidth=.8, shrinkA=5, shrinkB=4),
                    annotation_clip=False, zorder=12)
    if invalid:
        ax.text(.02, .02, '\n'.join(invalid), transform=ax.transAxes,
                color=POINT_COLOR, fontsize=11, va='bottom',
                bbox=dict(facecolor='black', alpha=.75, edgecolor='none', pad=4))
    fig.savefig(output, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, nargs='+',
                        default=[Path('data/LumbarCoco2000/annotations')])
    parser.add_argument('--images-dir', type=Path, default=Path('data/LumbarCoco2000/images'))
    parser.add_argument('--output-dir', type=Path, default=Path('reports/buu_vertebrae_shifts/visualizations'))
    parser.add_argument('--image-ids', nargs='+', help='Explicit COCO image IDs (overrides --count)')
    parser.add_argument('--count', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dpi', type=int, default=140)
    args = parser.parse_args()
    if args.count < 1 or args.dpi < 1:
        parser.error('--count and --dpi must be positive')
    paths = set()
    for path in args.annotations:
        if not path.exists():
            parser.error(f'annotation path does not exist: {path}')
        if path.is_dir():
            paths.update(path.rglob('*.json'))
        else:
            paths.add(path)
    if not paths:
        parser.error('no COCO annotation files found')
    images, audit = load_images(sorted(paths))
    candidates = sorted(images.values(), key=lambda group: group['image']['file_name'])
    if args.image_ids:
        by_id = {str(group['image']['id']): group for group in candidates}
        missing = set(args.image_ids) - by_id.keys()
        if missing:
            parser.error(f'unknown image IDs: {sorted(missing)}')
        selected = [by_id[key] for key in dict.fromkeys(args.image_ids)]
    else:
        if args.count > len(candidates):
            parser.error(f'--count exceeds available images ({len(candidates)})')
        indices = np.random.default_rng(args.seed).choice(len(candidates), args.count, replace=False)
        selected = [candidates[int(i)] for i in indices]
    records = []
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for group in selected:
        info = group['image']
        image_path = (args.images_dir / info['file_name']).resolve()
        measurements = build_measurements(group)
        destination = output / f"{info['id']}_{Path(info['file_name']).stem}_shifts.png"
        if destination == image_path:
            raise ValueError('output would overwrite the source image')
        with Image.open(image_path) as original:
            if original.size != (info['width'], info['height']):
                raise ValueError(f'{image_path}: image size differs from COCO metadata')
            image = np.array(original.convert('RGB'))
        render_image(image, info, measurements, destination, args.dpi)
        records.append(dict(image_id=info['id'], image_name=info['file_name'],
                            source_image=str(image_path), output=destination.name,
                            measurements=measurements))
        print(f"{destination.name}: " + ', '.join(
            f"{entry['level']}={entry['percent']:+.2f}%" if entry['valid'] else
            f"{entry['level']}=NaN ({entry['reason']})" for entry in measurements))
    (output / 'visualization_measurements.json').write_text(
        json.dumps(dict(seed=args.seed, sign_convention='positive anterior; negative posterior',
                        coordinates='original image pixels', input_audit=audit, images=records),
                   indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'Saved {len(records)} visualizations to {output}')


if __name__ == '__main__':
    main()
