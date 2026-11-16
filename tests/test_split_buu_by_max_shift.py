"""Exact image-level tails, reproducibility and preservation of COCO records."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.analyze_vertebrae_shifts import KEYPOINT_NAMES
from tools.split_buu_by_max_shift import coco_split, split_rows

ROOT = Path(__file__).resolve().parents[1]


def rows():
    return [dict(image_id=i, image_name=f'{i}.jpg', max_shift=(i - 20) / 100)
            for i in range(1, 41)]


class SplitTests(unittest.TestCase):
    def test_exact_tails_and_complete_disjoint_coverage(self):
        splits, ranked, lower, upper = split_rows(rows())
        self.assertEqual({k: len(v) for k, v in splits.items()}, dict(train=32, val=4, test=4))
        self.assertEqual([r['image_id'] for r in lower], [1, 2])
        self.assertEqual([r['image_id'] for r in upper], [39, 40])
        membership = [r['image_id'] for entries in splits.values() for r in entries]
        self.assertEqual(sorted(membership), list(range(1, 41)))
        self.assertEqual(len(membership), len(set(membership)))
        self.assertEqual([r['image_id'] for r in ranked], list(range(1, 41)))

    def test_seed_only_changes_middle_and_input_is_not_mutated(self):
        data = rows()
        snapshot = copy.deepcopy(data)
        a, *_ = split_rows(data, seed=42)
        b, *_ = split_rows(data[::-1], seed=42)
        c, *_ = split_rows(data, seed=43)
        self.assertEqual(a, b)
        self.assertEqual(a['test'], c['test'])
        self.assertNotEqual(a['train'], c['train'])
        self.assertEqual(data, snapshot)

    def test_ties_are_resolved_by_id_without_overlap(self):
        data = rows()[::-1]
        for row in data:
            row['max_shift'] = 0.
        splits, _, lower, upper = split_rows(data)
        self.assertEqual([r['image_id'] for r in lower], [1, 2])
        self.assertEqual([r['image_id'] for r in upper], [39, 40])
        self.assertEqual(len({r['image_id'] for r in splits['test']}), 4)

    def test_invalid_input_is_not_silently_dropped(self):
        bad = rows()
        bad[0]['max_shift'] = float('nan')
        for data in (bad, rows()[:-1], rows()[1:] + [rows()[1]]):
            with self.assertRaises(ValueError):
                split_rows(data)

    def test_coco_records_preserved(self):
        image = dict(id=1, file_name='1.jpg', width=200, height=200)
        annotation = dict(id=44, image_id=1, category_id=1, keypoints=[1, 2, 2],
                          bbox=[1, 2, 3, 4], segmentation=[[1, 2, 3, 4]], custom='keep')
        template = dict(info=dict(description='source'), licenses=[], categories=[dict(id=1)],
                        images=[image], annotations=[annotation])
        groups = {'1.jpg': dict(image=image, annotations={'a': annotation})}
        output = coco_split(template, groups, [dict(image_name='1.jpg')], 'test')
        self.assertEqual(output['images'], [image])
        self.assertEqual(output['annotations'], [annotation])
        self.assertEqual(output['categories'], template['categories'])
        output['annotations'][0]['bbox'][0] = 99
        self.assertEqual(annotation['bbox'][0], 1)
        self.assertNotIn('split', template['info'])

    def test_cli_creates_separate_coco_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.json'
            image_dir = root / 'source_images'
            image_dir.mkdir()
            images, annotations = [], []
            for i in range(1, 21):
                name = f'{i}.jpg'
                Image.new('RGB', (200, 200)).save(image_dir / name)
                images.append(dict(id=i, file_name=name, width=200, height=200))
                kp = np.array([[x, y, 2] for y in range(10, 120, 10) for x in (20, 40)])
                kp[[3, 7, 11, 15, 19], 0] = 30 + i
                annotations.append(dict(id=i, image_id=i, category_id=1,
                                        keypoints=kp.flatten().tolist(), bbox=[10, 10, 80, 120]))
            source.write_text(json.dumps(dict(info={}, licenses=[], images=images,
                                             annotations=annotations,
                                             categories=[dict(id=1, keypoints=KEYPOINT_NAMES)])))
            old_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            output, report = root / 'new_dataset', root / 'report'
            command = [sys.executable, str(ROOT / 'tools/split_buu_by_max_shift.py'),
                       '--annotations', str(source), '--images-dir', str(image_dir),
                       '--output', str(output), '--report-dir', str(report)]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            original = {a['id']: a for a in annotations}
            ids = []
            for split, count in (('train', 16), ('val', 2), ('test', 2)):
                doc = json.loads((output / 'annotations' / f'lumbar_keypoints_{split}.json').read_text())
                self.assertEqual(len(doc['images']), count)
                ids.extend(im['id'] for im in doc['images'])
                for a in doc['annotations']:
                    self.assertEqual(a, original[a['id']])
            self.assertEqual(sorted(ids), list(range(1, 21)))
            self.assertEqual((output / 'images').resolve(), image_dir.resolve())
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), old_hash)
            for name in ('max_shifts_log_hist.png', 'split_distributions_log.png'):
                with Image.open(report / name) as image:
                    image.verify()
            snapshot = (report / 'split_manifest.csv').read_bytes()
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn('refusing to overwrite', second.stderr)
            self.assertEqual((report / 'split_manifest.csv').read_bytes(), snapshot)


if __name__ == '__main__':
    unittest.main()
