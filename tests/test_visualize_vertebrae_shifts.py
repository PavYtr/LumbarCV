"""Check that rendered constructions agree with the dataset measurement."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.visualize_vertebrae_shifts import build_measurements, render_image


def group():
    points = np.array([[x, y, 2] for y in range(10, 120, 10) for x in (20, 40)])
    return dict(image=dict(id=1, file_name='spine.jpg', width=200, height=200),
                annotations={'one': dict(id=1, keypoints=points.flatten().tolist())})


class VisualizationTests(unittest.TestCase):
    def test_projection_matches_positive_and_negative_ratios(self):
        for x, ratio in ((30, .5), (40, 0), (50, -.5), (70, -1.5)):
            with self.subTest(x=x):
                data = group()
                data['annotations']['one']['keypoints'][9] = x  # L1BB
                entry = build_measurements(data)[0]
                self.assertTrue(entry['valid'])
                np.testing.assert_allclose(entry['Q'], [x, 30])
                self.assertAlmostEqual(entry['shift'], ratio)
                self.assertAlmostEqual(entry['percent'], 100 * ratio)
                self.assertAlmostEqual(entry['signed_displacement_px'], 20 * ratio)
                p, a, b, q = (np.array(entry[key]) for key in ('P', 'A', 'B', 'Q'))
                self.assertAlmostEqual(float((p - q) @ (b - a)), 0)

    def test_invalid_pair_is_reported_without_dropping_other_levels(self):
        data = group()
        data['annotations']['one']['keypoints'][11] = 0
        entries = build_measurements(data)
        self.assertEqual(len(entries), 5)
        self.assertEqual(entries[0]['reason'], 'unlabelled_keypoint')
        self.assertIsNone(entries[0]['percent'])
        self.assertTrue(all(entry['valid'] for entry in entries[1:]))
        # Missing or multiple distinct annotations remain explicit for all levels.
        for annotations, reason in (({}, 'missing_annotation'),
                                    ({'a': {}, 'b': {}}, 'ambiguous_multiple_annotations')):
            data['annotations'] = annotations
            self.assertTrue(all(e.get('reason') == reason for e in build_measurements(data)))

    def test_zero_plate_cannot_be_rendered_as_valid(self):
        data = group()
        flat = data['annotations']['one']['keypoints']
        flat[15:18] = flat[12:15]
        entry = build_measurements(data)[0]
        self.assertFalse(entry['valid'])
        self.assertEqual(entry['reason'], 'zero_lower_plate_length')

    def test_render_with_invalid_pair_saves_png(self):
        data = group()
        data['annotations']['one']['keypoints'][11] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'measurement.png'
            render_image(np.zeros((200, 200, 3), dtype=np.uint8), data['image'],
                         build_measurements(data), path, dpi=60)
            with Image.open(path) as image:
                self.assertEqual(image.format, 'PNG')
                self.assertEqual(image.size, (600, 600))
                image.verify()


if __name__ == '__main__':
    unittest.main()
