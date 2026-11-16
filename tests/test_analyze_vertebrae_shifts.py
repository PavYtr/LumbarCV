"""Standalone geometry and COCO aggregation checks; no MMPose dependency."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.analyze_vertebrae_shifts import (
    KEYPOINT_NAMES, LEVELS, PAIR_INDICES, analyze, load_images,
    relative_shift, statistics, histogram_edges,
)


def document():
    points = [[x, y, 2] for y in range(10, 120, 10) for x in (20, 40)]
    return dict(images=[dict(id=1, file_name='spine.jpg', width=200, height=200)],
                annotations=[dict(id=1, image_id=1, category_id=1,
                                  keypoints=np.array(points).flatten().tolist())],
                categories=[dict(id=1, keypoints=KEYPOINT_NAMES)])


class GeometryTests(unittest.TestCase):
    def test_aligned_posterior_corners(self):
        self.assertEqual(relative_shift([10, 0], [[0, 5], [10, 5]]), 0)

    def test_shift_right_is_negative(self):
        self.assertAlmostEqual(relative_shift([12, 0], [[0, 5], [10, 5]]), -.2)

    def test_shift_left_is_positive(self):
        self.assertAlmostEqual(relative_shift([7, 0], [[0, 5], [10, 5]]), .3)

    def test_rotation_translation_reflection_and_scale_preserve_sign(self):
        lower = np.array([[0., 5], [10, 5]])
        for x, expected in ((7, .3), (12, -.2), (10, 0)):
            point = np.array([float(x), 0])
            for angle in (0, 30, 90, 135, 180, 270):
                theta = np.deg2rad(angle)
                rotation = np.array([[np.cos(theta), -np.sin(theta)],
                                     [np.sin(theta), np.cos(theta)]])
                for reflection in (1, -1):
                    matrix = rotation @ np.diag([reflection, 1]) * 3.5
                    with self.subTest(x=x, angle=angle, reflection=reflection):
                        self.assertAlmostEqual(relative_shift(
                            point @ matrix.T + 100, lower @ matrix.T + 100), expected)

    def test_oblique_plate_perpendicular_projection(self):
        # A=(2,3), B=(8,11), |AB|=10; Q=A+.7*(B-A).
        # P is offset perpendicularly from Q, which must not affect S=.3.
        a, b = np.array([2., 3]), np.array([8., 11])
        q = a + .7 * (b - a)
        for distance in (-100, 0, 50):
            self.assertAlmostEqual(relative_shift(q + distance * np.array([-.8, .6]), [a, b]), .3)

    def test_zero_lower_plate_length(self):
        with self.assertRaisesRegex(ValueError, 'zero_lower_plate_length'):
            relative_shift([10, 0], [[5, 5], [5, 5]])

    def test_projection_outside_segment_is_not_clipped(self):
        self.assertEqual(relative_shift([30, 0], [[0, 5], [10, 5]]), -2)
        self.assertEqual(relative_shift([-10, 0], [[0, 5], [10, 5]]), 2)

    def test_nonfinite(self):
        for point, lower in (([np.nan, 0], [[0, 5], [10, 5]]),
                             ([10, 0], [[0, 5], [np.inf, 5]])):
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                relative_shift(point, lower)

    def test_invalid_geometry_shape(self):
        with self.assertRaisesRegex(ValueError, 'invalid_geometry_shape'):
            relative_shift([[0, 0], [10, 0]], [[0, 5], [10, 5]])

    def test_histogram_bins_include_negative_values(self):
        for values in ((-.5, -.1, .3, np.nan), (-.5, -.1), (0, 0), (np.nan,)):
            with self.subTest(values=values):
                edges = histogram_edges(values, 40)
                self.assertEqual(len(edges), 41)
                self.assertTrue(np.all(np.diff(edges) > 0))
                valid = np.array(values)[np.isfinite(values)]
                counts, _ = np.histogram(valid, bins=edges)
                self.assertEqual(counts.sum(), len(valid))
                self.assertLessEqual(edges[0], 0)
                self.assertGreaterEqual(edges[-1], 0)


class CocoTests(unittest.TestCase):
    def run_document(self, d):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'coco.json'
            path.write_text(json.dumps(d))
            images, audit = load_images([path])
            rows, issues = analyze(images)
            return rows, issues, statistics(rows, issues, audit)

    def test_verified_pair_mapping(self):
        self.assertEqual(PAIR_INDICES, ((3, 4, 5), (7, 8, 9),
                                      (11, 12, 13), (15, 16, 17),
                                      (19, 20, 21)))

    def test_duplicates_within_and_across_splits(self):
        d = document()
        duplicate = copy.deepcopy(d['annotations'][0])
        duplicate['id'] = 99
        d['annotations'].append(duplicate)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / f'{i}.json' for i in range(2)]
            for p in paths:
                p.write_text(json.dumps(d))
            images, audit = load_images(paths)
            rows, issues = analyze(images)
        self.assertEqual(len(rows), 1)
        self.assertFalse(issues)
        self.assertEqual(audit['duplicate_image_records'], 1)
        self.assertEqual(audit['duplicate_annotation_records'], 3)

    def test_conflicting_instances_are_reported(self):
        d = document()
        other = copy.deepcopy(d['annotations'][0])
        other['keypoints'][9] += 3
        d['annotations'].append(other)
        rows, issues, stats = self.run_document(d)
        self.assertTrue(np.isnan(rows[0]['max_shift']))
        self.assertEqual(len(issues), 5)
        self.assertEqual(stats['invalid_reasons'], {'ambiguous_multiple_annotations': 5})

    def test_bad_point_only_invalidates_affected_pair(self):
        for visibility, xy in ((0, 20), (1, 20), (3, 20), (2, -1), (2, np.nan), (2, 'bad')):
            with self.subTest(visibility=visibility, xy=xy):
                d = document()
                d['annotations'][0]['keypoints'][9:12] = [xy, 20, visibility]
                rows, issues, stats = self.run_document(d)
                self.assertTrue(np.isnan(rows[0]['L1_L2']))
                self.assertEqual(rows[0]['max_shift'], 0)
                self.assertEqual(len(issues), 1)
                self.assertEqual(stats['all_shifts']['valid_count'], 4)

    def test_upper_anterior_point_is_not_used(self):
        for point in ([float('nan'), -100, 0], [40, 20, 2]):
            d = document()
            # Index 2 (L1BF) may be invalid or coincide with L1BB.
            # Neither affects a measurement requiring only L1BB, L2TF, L2TB.
            d['annotations'][0]['keypoints'][6:9] = point
            rows, issues, _ = self.run_document(d)
            self.assertEqual(rows[0]['L1_L2'], 0)
            self.assertFalse(issues)

    def test_negative_shifts_and_algebraic_maximum(self):
        d = document()
        for gap, index in enumerate((3, 7, 11, 15, 19)):
            d['annotations'][0]['keypoints'][3 * index] = 42 + gap * 2
        rows, issues, stats = self.run_document(d)
        self.assertFalse(issues)
        np.testing.assert_allclose([rows[0][level] for level in LEVELS], [-.1, -.2, -.3, -.4, -.5])
        self.assertAlmostEqual(rows[0]['max_shift'], -.1)
        self.assertEqual(stats['all_shifts']['negative_count'], 5)
        self.assertEqual(stats['max_shift']['negative_count'], 1)

    def test_missing_point(self):
        d = document()
        d['annotations'][0]['keypoints'] = d['annotations'][0]['keypoints'][:-3]
        rows, issues, _ = self.run_document(d)
        self.assertTrue(np.isnan(rows[0]['L5_S1']))
        self.assertEqual(issues[0]['reason'], 'missing_keypoint')

    def test_zero_length_saved_as_nan(self):
        d = document()
        d['annotations'][0]['keypoints'][15:18] = d['annotations'][0]['keypoints'][12:15]
        rows, issues, _ = self.run_document(d)
        self.assertTrue(np.isnan(rows[0]['L1_L2']))
        self.assertEqual(issues[0]['reason'], 'zero_lower_plate_length')

    def test_missing_annotation_and_strict_json(self):
        d = document()
        d['annotations'] = []
        rows, issues, stats = self.run_document(d)
        self.assertEqual(stats['max_shift']['valid_count'], 0)
        self.assertEqual(stats['all_shifts']['invalid_count'], 5)
        self.assertEqual(stats['max_shift']['percentiles']['95']['value'], None)
        json.dumps(stats, allow_nan=False)

    def test_wrong_layout_is_rejected(self):
        d = document()
        d['categories'][0]['keypoints'] = KEYPOINT_NAMES[::-1]
        with self.assertRaisesRegex(ValueError, 'layout'):
            self.run_document(d)

    def test_percentiles_strict_exceedance(self):
        rows = [dict.fromkeys((*LEVELS, 'max_shift'), float(i)) for i in range(5)]
        stats = statistics(rows, [], {})
        self.assertEqual(stats['max_shift']['percentiles']['50'], dict(value=2., images_above=2))
        self.assertEqual(stats['all_shifts']['valid_count'], 25)


if __name__ == '__main__':
    unittest.main()
