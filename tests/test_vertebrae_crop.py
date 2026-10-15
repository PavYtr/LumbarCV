"""Geometry, containment, and validation leakage checks; no model/training."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from mmcv.transforms import Compose
from mmengine.config import Config
from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules

from augmentations.vertebrae_crop import FullImageBBox, RandomVertebraeCrop
from augmentations.vertebrae_crop import KEYPOINT_NAMES, measure, rotation, spine_angle

from tools.collect_vertebrae_statistics import summarize

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / 'configs/rtmpose-m_lumbar-coco2000_fullimage.py'
AUGMENTED = ROOT / 'configs/rtmpose-m_lumbar-coco2000_vertebrae-crop.py'


def synthetic_points():
    points = []
    for y in np.linspace(20, 70, 11):
        points.extend([[40, y], [60, y]])
    return np.asarray(points, dtype=np.float32)


class VertebraeCropTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules()
        cls.baseline = Config.fromfile(str(BASELINE))
        cls.augmented = Config.fromfile(str(AUGMENTED))

    def test_statistics_arithmetic_means(self):
        samples = [
            dict(angle_deg=-10, margins_px=dict(left=10, right=20, top=30, bottom=40),
                 margins_relative=dict(left=.1, right=.2, top=.3, bottom=.4)),
            dict(angle_deg=4, margins_px=dict(left=30, right=40, top=50, bottom=60),
                 margins_relative=dict(left=.3, right=.4, top=.5, bottom=.6)),
        ]
        means = summarize(samples)
        self.assertEqual(means['count'], 2)
        self.assertEqual(means['angle_deg'], -3)
        self.assertEqual(means['absolute_angle_deg'], 7)
        self.assertEqual(means['margins_px'], dict(left=20, right=30, top=40, bottom=50))
        for side, expected in dict(left=.2, right=.3, top=.4, bottom=.5).items():
            self.assertAlmostEqual(means['margins_relative'][side], expected)
        with self.assertRaises(ValueError):
            summarize([])

    def test_mean_ranges_and_random_draws(self):
        path = ROOT / 'reports/vertebrae_statistics_train.json'
        means = json.loads(path.read_text())['mean']
        transform = RandomVertebraeCrop(path)
        self.assertEqual(transform.rotation_range,
                         (-means['absolute_angle_deg'], means['absolute_angle_deg']))
        for side, mean in means['margins_relative'].items():
            self.assertEqual(transform.expansion_ranges[side], (0, mean))
        data = dict(keypoints=synthetic_points()[None], keypoints_visible=np.ones((1, 22)),
                    bbox_center=np.array([[50, 50]], np.float32),
                    bbox_scale=np.array([[100, 100]], np.float32))
        np.random.seed(5)
        draws = [transform(copy.deepcopy(data))['vertebrae_crop'] for _ in range(100)]
        angles = [d['sampled_angle_deg'] for d in draws]
        self.assertLess(min(angles), 0)
        self.assertGreater(max(angles), 0)
        for draw in draws:
            self.assertLessEqual(abs(draw['sampled_angle_deg']), means['absolute_angle_deg'])
            for side, value in draw['sampled_margins_relative'].items():
                self.assertTrue(0 <= value <= means['margins_relative'][side])
        for side in means['margins_relative']:
            self.assertGreater(np.ptp([d['sampled_margins_relative'][side] for d in draws]), 0)

    def test_explicit_ranges_set_crop_geometry(self):
        transform = RandomVertebraeCrop(
            expansion_ranges=dict(left=(.2, .2), right=(.6, .6),
                                  top=(.1, .1), bottom=(.3, .3)),
            rotation_range=(0, 0))
        data = dict(keypoints=synthetic_points()[None], keypoints_visible=np.ones((1, 22)),
                    bbox_center=np.array([[50, 50]], np.float32),
                    bbox_scale=np.array([[100, 100]], np.float32))
        out = transform(data)
        np.testing.assert_allclose(out['bbox_center'], [[54, 50]])
        np.testing.assert_allclose(out['bbox_scale'], [[40, 74]])
        self.assertEqual(out['vertebrae_crop']['sampled_angle_deg'], 0)
        with self.assertRaises(ValueError):
            RandomVertebraeCrop(expansion_ranges=dict(left=(-1, 0)), rotation_range=(0, 0))

    def test_axis_and_unrotated_margins(self):
        p = synthetic_points()
        s = measure(p, np.ones(22), 100, 100)
        self.assertEqual(s['angle_deg'], 0)
        self.assertEqual(s['margins_px'], dict(left=40, right=40, top=20, bottom=30))
        self.assertEqual(s['margins_relative'], dict(left=2, right=2, top=.4, bottom=.6))
        tilted = p @ rotation(-20).T
        self.assertAlmostEqual(spine_angle(tilted, np.ones(22)), 20)
        vertical = tilted @ rotation(spine_angle(tilted, np.ones(22))).T
        np.testing.assert_allclose(vertical, p, atol=1e-5)

    def test_black_canvas_not_counted_as_margin(self):
        p = synthetic_points()
        s = measure(p, np.ones(22), 100, 100)
        self.assertEqual(s['margins_px']['left'], 40)
        # A 30-degree source lean gives a rotated real-image quadrilateral.
        tilted = (p - 50) @ rotation(-30).T + 50
        result = measure(tilted, np.ones(22), 100, 100)
        r = rotation(result['angle_deg'])
        corners = np.array([[0, 0], [100, 0], [100, 100], [0, 100]]) @ r.T
        q = tilted @ r.T
        canvas_margin = q[:, 0].min() - corners[:, 0].min()
        self.assertLess(result['margins_px']['left'], canvas_margin - 1)

    def test_invalid_axis_and_statistics_fail(self):
        p = synthetic_points()
        visibility = np.ones(22)
        visibility[20] = 0
        with self.assertRaises(ValueError):
            spine_angle(p, visibility)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bad.json'
            path.write_text(json.dumps(dict(schema_version=1, keypoint_names=KEYPOINT_NAMES, samples=[])))
            with self.assertRaises(ValueError):
                RandomVertebraeCrop(path)

    def test_configs_and_no_gt_validation(self):
        a, b = self.baseline.to_dict(), self.augmented.to_dict()
        for key in a:
            if key not in ('train_pipeline', 'train_dataloader'):
                self.assertEqual(a[key], b[key], key)
        for key in a['train_dataloader']:
            if key != 'dataset':
                self.assertEqual(a['train_dataloader'][key], b['train_dataloader'][key])
        for key in a['train_dataloader']['dataset']:
            if key != 'pipeline':
                self.assertEqual(a['train_dataloader']['dataset'][key], b['train_dataloader']['dataset'][key])
        # Evaluation works without landmarks and ignores the annotation bbox.
        pipeline = Compose(self.baseline.val_pipeline[1:-1])
        data = dict(img=np.full((200, 100, 3), 70, np.uint8),
                    bbox=np.array([[40, 60, 50, 80]], np.float32))
        out = pipeline(copy.deepcopy(data))
        np.testing.assert_equal(out['bbox'], [[0, 0, 100, 200]])
        self.assertEqual(out['img'].shape, (512, 256, 3))
        data.update(keypoints=synthetic_points()[None], keypoints_visible=np.ones((1, 22)))
        out_gt = pipeline(data)
        np.testing.assert_equal(out['img'], out_gt['img'])
        np.testing.assert_equal(out['bbox_center'], out_gt['bbox_center'])
        np.testing.assert_equal(out['bbox_scale'], out_gt['bbox_scale'])

    def test_all_training_landmarks_contained_and_rigid(self):
        coco = json.loads((ROOT / 'data/LumbarCoco2000/annotations/lumbar_keypoints_train.json').read_text())
        images = {im['id']: im for im in coco['images']}
        transform = RandomVertebraeCrop(ROOT / 'reports/vertebrae_statistics_train.json')
        from mmpose.datasets.transforms import TopdownAffine
        from mmpose.structures.bbox import get_warp_matrix
        np.random.seed(21)
        for annotation in coco['annotations']:
            p = np.asarray(annotation['keypoints'], np.float32).reshape(22, 3)
            image = images[annotation['image_id']]
            data = dict(keypoints=p[None, :, :2].copy(), keypoints_visible=p[None, :, 2],
                        bbox_center=np.array([[image['width']/2, image['height']/2]], np.float32),
                        bbox_scale=np.array([[image['width'], image['height']]], np.float32))
            original = data['keypoints'].copy()
            out = transform(data)
            np.testing.assert_equal(out['keypoints'], original)
            scale = TopdownAffine._fix_aspect_ratio(out['bbox_scale'], .5)[0]
            matrix = get_warp_matrix(out['bbox_center'][0], scale, out['bbox_rotation'][0], (256, 512))
            q = cv2.transform(original, matrix)[0]
            self.assertTrue(np.all(q >= 0) and np.all(q < [256, 512]), image['file_name'])
            self.assertAlmostEqual(spine_angle(q, p[:, 2]), out['vertebrae_crop']['sampled_angle_deg'], places=3)
            # Similarity transform: no bending, shear or anisotropic scaling.
            gram = matrix[:, :2].T @ matrix[:, :2]
            np.testing.assert_allclose(gram, np.eye(2) * gram[0, 0], atol=1e-6)

    def test_real_pipeline_targets_and_skip(self):
        cfg = copy.deepcopy(self.augmented.train_dataloader.dataset)
        cfg.pipeline = []
        dataset = DATASETS.build(cfg)
        steps = self.augmented.train_dataloader.dataset.pipeline
        pipeline = Compose(steps)
        np.random.seed(42)
        for index in (0, 7, 100, 800, 1599):
            out = pipeline(dataset.get_data_info(index))
            self.assertEqual(tuple(out['inputs'].shape), (3, 512, 256))
            labels = out['data_samples'].gt_instance_labels
            self.assertEqual(labels.keypoint_x_labels.shape[-2:], (22, 512))
            self.assertEqual(labels.keypoint_y_labels.shape[-2:], (22, 1024))
            self.assertTrue((labels.keypoint_weights > 0).all())
        crop = RandomVertebraeCrop(ROOT / 'reports/vertebrae_statistics_train.json', apply_prob=0)
        data = FullImageBBox()(dict(img=np.zeros((100, 100, 3), np.uint8)))
        data['bbox_center'] = np.array([[50, 50]], np.float32)
        data['bbox_scale'] = np.array([[100, 100]], np.float32)
        before = copy.deepcopy(data)
        out = crop(data)
        np.testing.assert_equal(out['bbox_center'], before['bbox_center'])
        np.testing.assert_equal(out['bbox_scale'], before['bbox_scale'])


if __name__ == '__main__':
    unittest.main()
