#!/usr/bin/env python3
"""Create a separate BUU COCO split with the lowest/highest 5% max_shift in test.

One score per image: algebraic maximum of the five signed posterior-corner
shifts. The middle 90% is shuffled with seed=42, then split 8:1 train/val.
Logarithmic Y is used only for plots; ranks use original signed scores.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_vertebrae_shifts import analyze, histogram_edges, load_images, summarize


def split_rows(rows, seed=42):
    """Exact 5% tails by rank, with unique images and deterministic tie handling."""
    n = len(rows)
    if n < 20 or n % 20:
        raise ValueError('image count must be a positive multiple of 20 for exact 5% tails')
    if len({r['image_id'] for r in rows}) != n or len({r['image_name'] for r in rows}) != n:
        raise ValueError('duplicate images in split input')
    if any(not np.isfinite(r['max_shift']) for r in rows):
        raise ValueError('every image must have a finite max_shift')
    ranked = sorted(rows, key=lambda r: (r['max_shift'], r['image_id'], r['image_name']))
    count = n // 20
    lower, upper = ranked[:count], ranked[-count:]
    middle = ranked[count:-count].copy()
    random.Random(seed).shuffle(middle)
    n_train = 8 * n // 10
    splits = dict(train=middle[:n_train], val=middle[n_train:], test=lower + upper)
    return splits, ranked, lower, upper


def coco_split(template, groups, rows, split):
    """Preserve all original image/annotation fields, IDs, categories and skeleton."""
    document = copy.deepcopy(template)
    document['info'] = dict(document.get('info', {}),
                            split_scheme='max_shift_rank_tails_5_percent_each', split=split)
    document['images'] = [copy.deepcopy(groups[row['image_name']]['image']) for row in rows]
    document['annotations'] = [copy.deepcopy(annotation) for row in rows
                               for annotation in groups[row['image_name']]['annotations'].values()]
    return document


def plot_splits(splits, lower, upper, output, bins=40):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_values = np.array([r['max_shift'] for rows in splits.values() for r in rows]) * 100
    edges = histogram_edges(all_values, bins)
    low, high = lower[-1]['max_shift'] * 100, upper[0]['max_shift'] * 100
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    ax.hist(all_values, bins=edges, color='steelblue', edgecolor='white', log=True,
            label=f'All images (n={len(all_values)})')
    ax.axvspan(edges[0], low, color='purple', alpha=.15)
    ax.axvspan(high, edges[-1], color='darkorange', alpha=.15)
    for value, color, label in ((low, 'purple', f'Left test: {len(lower)} images, S <= {low:.3f}%'),
                                (high, 'darkorange', f'Right test: {len(upper)} images, S >= {high:.3f}%')):
        ax.axvline(value, color=color, linestyle='--', linewidth=1.7, label=label)
    ax.axvline(0, color='black', linewidth=.9, alpha=.5)
    ax.set(title='Maximum signed shift per image: test contains both 5% tails',
           xlabel='max_shift (%)  [+ anterior, - posterior]', ylabel='Image count (log scale)')
    ax.set_ylim(bottom=.8)
    ax.grid(axis='y', which='both', alpha=.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.savefig(output / 'max_shifts_log_hist.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    for split, color in (('train', 'steelblue'), ('val', 'seagreen'), ('test', 'crimson')):
        values = [r['max_shift'] * 100 for r in splits[split]]
        ax.hist(values, bins=edges, histtype='step', linewidth=1.8, color=color,
                label=f'{split}: n={len(values)}')
    ax.set_yscale('log')
    ax.set_ylim(bottom=.8)
    ax.axvline(low, color='purple', linestyle='--', alpha=.6)
    ax.axvline(high, color='darkorange', linestyle='--', alpha=.6)
    ax.set(title='Maximum signed shift by split (common bins)',
           xlabel='max_shift (%)  [+ anterior, - posterior]', ylabel='Image count (log scale)')
    ax.grid(axis='y', which='both', alpha=.25)
    ax.set_axisbelow(True)
    ax.legend()
    fig.savefig(output / 'split_distributions_log.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, nargs='+',
                        default=[Path('data/LumbarCoco2000/annotations')])
    parser.add_argument('--images-dir', type=Path, default=Path('data/LumbarCoco2000/images'))
    parser.add_argument('--output', type=Path, default=Path('data/LumbarCoco2000MaxShiftTails'))
    parser.add_argument('--report-dir', type=Path,
                        default=Path('reports/buu_vertebrae_shifts/max_shift_tail_split'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bins', type=int, default=40)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error('--bins must be positive')
    paths = set()
    for path in args.annotations:
        if not path.exists():
            parser.error(f'missing annotations: {path}')
        if path.is_dir():
            paths.update(p.resolve() for p in path.rglob('*.json'))
        else:
            paths.add(path.resolve())
    if not paths:
        parser.error('no COCO annotation JSON files found')
    paths = sorted(paths)
    groups, audit = load_images(paths)
    rows, issues = analyze(groups)
    if issues:
        parser.error(f'input has {len(issues)} invalid measurements; fix them before tail selection')
    try:
        splits, ranked, lower, upper = split_rows(rows, args.seed)
    except ValueError as error:
        parser.error(str(error))
    template = json.loads(paths[0].read_text())
    for path in paths[1:]:
        if json.loads(path.read_text())['categories'] != template['categories']:
            parser.error('input COCO categories differ')
    source_images = args.images_dir.resolve()
    if not source_images.is_dir():
        parser.error(f'missing image directory: {source_images}')
    for group in groups.values():
        if not (source_images / group['image']['file_name']).is_file():
            parser.error(f"missing image: {group['image']['file_name']}")
    output, report = args.output.resolve(), args.report_dir.resolve()
    destinations = {split: output / 'annotations' / f'lumbar_keypoints_{split}.json' for split in splits}
    if any(path in paths for path in destinations.values()):
        parser.error('output must not replace source annotations')
    if output.is_relative_to(source_images) or report.is_relative_to(source_images):
        parser.error('output must be outside the original image directory')
    artifacts = [*destinations.values(), report / 'split_statistics.json', report / 'split_manifest.csv',
                 report / 'max_shifts_log_hist.png', report / 'split_distributions_log.png']
    existing = [str(path) for path in artifacts if path.exists()]
    if existing:
        parser.error(f'refusing to overwrite an existing split; choose new output/report directories: {existing}')
    image_link = output / 'images'
    if image_link.exists() or image_link.is_symlink():
        if not image_link.is_symlink() or image_link.resolve() != source_images:
            parser.error(f'{image_link} already exists and is not the expected image symlink')
    else:
        output.mkdir(parents=True, exist_ok=True)
        image_link.symlink_to(source_images, target_is_directory=True)
    report.mkdir(parents=True, exist_ok=True)
    (output / 'annotations').mkdir(parents=True, exist_ok=True)
    members = {name: {r['image_id'] for r in entries} for name, entries in splits.items()}
    assert set.union(*members.values()) == {r['image_id'] for r in ranked}
    assert all(not members[a] & members[b] for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')))
    output_hashes = {}
    for name, entries in splits.items():
        document = coco_split(template, groups, entries, name)
        raw = (json.dumps(document, indent=2, allow_nan=False) + '\n').encode('utf-8')
        destinations[name].write_bytes(raw)
        output_hashes[name] = hashlib.sha256(raw).hexdigest()
    low_ids = {r['image_id'] for r in lower}
    high_ids = {r['image_id'] for r in upper}
    assignment = {r['image_id']: split for split, entries in splits.items() for r in entries}
    with (report / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['rank', 'image_id', 'image_name', 'max_shift',
                                                    'max_shift_percent', 'split', 'selection'],
                                lineterminator='\n')
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            ident = row['image_id']
            selection = 'lower_tail' if ident in low_ids else 'upper_tail' if ident in high_ids else 'middle'
            writer.writerow(dict(rank=rank, image_id=ident, image_name=row['image_name'],
                                 max_shift=row['max_shift'], max_shift_percent=100 * row['max_shift'],
                                 split=assignment[ident], selection=selection))
    stats = dict(
        seed=args.seed, dataset=str(output), images_dir=str(source_images), processed_images=len(ranked),
        selection='Sort by (signed max_shift, image_id, image_name); first 5% and last 5% to test; '
                  'shuffle middle 90% with random.Random(seed), then divide train:val=8:1',
        metric='Algebraic maximum of five signed posterior-corner shifts per image',
        log_scale='Y image counts only; X signed max_shift; selection uses raw ranks',
        images_per_tail=len(lower),
        boundaries=dict(lower_test_max=lower[-1]['max_shift'], middle_min=ranked[len(lower)]['max_shift'],
                        middle_max=ranked[-len(upper)-1]['max_shift'], upper_test_min=upper[0]['max_shift']),
        interpolated_percentiles=dict(zip(('5', '95'), map(float, np.percentile(
            [r['max_shift'] for r in ranked], (5, 95))))),
        counts={name: len(entries) for name, entries in splits.items()},
        max_shift_statistics={name: summarize([r['max_shift'] for r in entries]) for name, entries in splits.items()},
        tail_statistics=dict(lower=summarize([r['max_shift'] for r in lower]),
                             upper=summarize([r['max_shift'] for r in upper])),
        input_audit=audit, output_annotations_sha256=output_hashes,
        validation=dict(unique_images=True, complete_coverage=True, disjoint_splits=True,
                        original_image_and_annotation_records_preserved=True))
    for source in audit['sources']:
        if hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() != source['sha256']:
            raise RuntimeError('source annotations changed during split creation')
    (report / 'split_statistics.json').write_text(json.dumps(stats, indent=2, allow_nan=False) + '\n')
    plot_splits(splits, lower, upper, report, args.bins)
    print(json.dumps({key: stats[key] for key in ('dataset', 'counts', 'images_per_tail', 'boundaries')}, indent=2))
    print(f'Saved split report and log-Y plots to {report}')


if __name__ == '__main__':
    main()
