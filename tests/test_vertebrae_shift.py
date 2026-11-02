"""Regression checks for preserving landmarks through shift, crop and targets."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from mmcv.transforms import Compose
from mmengine.config import Config
from mmpose.registry import DATASETS
from mmpose.utils import register_all_modules

from augmentations.vertebrae_crop import RandomVertebraeCrop, rotation
from augmentations.vertebrae_shift import RandomVertebraeShift


ROOT = Path(__file__).resolve().parents[1]


def reference_slide_image(image, upper, delta):
    """Frozen pre-optimization implementation: pixel/rounding regression oracle."""
    height, width = image.shape[:2]
    matrix = np.array([[1, 0, delta[0]], [0, 1, delta[1]]], np.float64)
    mask = upper.astype(np.float32)
    weight = cv2.warpAffine(mask, matrix, (width, height),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=1)
    source = image.astype(np.result_type(image.dtype, np.float32))
    source[~upper] = 0
    shifted = cv2.warpAffine(source, matrix, (width, height),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if image.ndim == 3:
        shifted = shifted.reshape(image.shape)
        weight = weight[..., None]
    np.divide(shifted, weight, out=shifted, where=weight > 0)
    if np.issubdtype(image.dtype, np.integer):
        limits = np.iinfo(image.dtype)
        shifted = np.clip(np.rint(shifted), limits.min, limits.max)
    output = image.copy()
    output[upper] = shifted[upper].astype(image.dtype)
    return output


def sample():
    points = np.array([[x, y] for y in np.linspace(20, 70, 11)
                       for x in (40, 60)], dtype=np.float32)
    return dict(img=np.full((100, 100, 3), 100, np.uint8),
                keypoints=points[None], keypoints_visible=np.ones((1, 22), np.float32),
                num_keypoints=22, bbox_center=np.array([[50, 50]], np.float32),
                bbox_scale=np.array([[100, 100]], np.float32))


class VertebraeShiftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_all_modules()

    def shift(self, data, gap=0, ratio=10, sign=1):
        with patch('numpy.random.randint', side_effect=[gap, sign]), \
                patch('numpy.random.uniform', return_value=ratio):
            return RandomVertebraeShift(prob=1, max_shift_ratio=ratio)(data)

    def assert_preserved(self, before, after):
        np.testing.assert_array_equal(after['keypoints_visible'], before['keypoints_visible'])
        self.assertEqual(after['num_keypoints'], before['num_keypoints'])
        visible = before['keypoints_visible'].reshape(22, -1)[:, 0] > 0
        xy = after['keypoints'][0, visible, :2]
        self.assertTrue(np.isfinite(xy).all())
        self.assertTrue((xy >= 0).all())
        self.assertTrue((xy < after['img'].shape[1::-1]).all())

    def test_optimized_pixels_match_reference_exactly(self):
        rng = np.random.default_rng(21)
        height, width = 47, 61
        y, x = np.ogrid[:height, :width]
        masks = [np.zeros((height, width), bool),
                 np.ones((height, width), bool),
                 y + .31 * x < 28, x - .27 * y < 24,
                 (x + y) % 2 == 0]
        deltas = [(0, 0), (3, -2), (5.015625, -.484375),
                  (-7.234, 4.891), (width + .25, -height - .75)]
        dtypes = (np.uint8, np.int8, np.uint16, np.int16, np.int32,
                  np.float16, np.float32, np.float64, np.uint32, np.int64)
        for dtype in dtypes:
            for channels in (None, 1, 3):
                shape = (height, width) if channels is None else (height, width, channels)
                image = rng.uniform(0, 120, shape).astype(dtype)
                before = image.copy()
                for mask in masks:
                    for delta in deltas:
                        with self.subTest(dtype=dtype, channels=channels, delta=delta):
                            expected = reference_slide_image(image, mask, delta)
                            actual = RandomVertebraeShift._slide_image(image, mask, delta)
                            self.assertEqual(actual.dtype, expected.dtype)
                            self.assertEqual(actual.shape, expected.shape)
                            self.assertEqual(actual.tobytes(), expected.tobytes())
                            np.testing.assert_array_equal(image, before)

    def test_real_images_match_reference_and_preserve_random_stream(self):
        class ReferenceShift(RandomVertebraeShift):
            _slide_image = staticmethod(reference_slide_image)

        config = Config.fromfile(str(ROOT / 'configs/rtmpose-m_lumbar-coco2000_vertebrae-shift-crop.py'))
        dataset_config = copy.deepcopy(config.train_dataloader.dataset)
        dataset_config.pipeline = []
        dataset = DATASETS.build(dataset_config)
        load = Compose(config.train_pipeline[:1])
        for index in (0, 7, 100, 800, 1599):
            data = load(dataset.get_data_info(index))
            for probability, ratio in ((0, .25), (1, .25), (1, 1.25)):
                with self.subTest(index=index, probability=probability, ratio=ratio):
                    np.random.seed(42 + index)
                    expected = ReferenceShift(probability, ratio)(copy.deepcopy(data))
                    expected_state = np.random.get_state()
                    np.random.seed(42 + index)
                    actual = RandomVertebraeShift(probability, ratio)(copy.deepcopy(data))
                    actual_state = np.random.get_state()
                    self.assertEqual(actual['img'].tobytes(), expected['img'].tobytes())
                    np.testing.assert_array_equal(actual['keypoints'], expected['keypoints'])
                    np.testing.assert_array_equal(actual['keypoints_visible'], expected['keypoints_visible'])
                    self.assertEqual(actual_state[0], expected_state[0])
                    np.testing.assert_array_equal(actual_state[1], expected_state[1])
                    self.assertEqual(actual_state[2:], expected_state[2:])

    def test_large_shift_preserves_all_visibility_layouts_and_directions(self):
        for shape in ((1, 22), (1, 22, 1), (1, 22, 2)):
            for angle in (-35, 0, 35):
                for sign in (0, 1):
                    with self.subTest(shape=shape, angle=angle, sign=sign):
                        data = sample()
                        data['keypoints'][0] = (data['keypoints'][0] - 50) @ rotation(angle).T + 50
                        data['keypoints_visible'] = np.ones(shape, np.float32)
                        if shape[-1] == 2:
                            data['keypoints_visible'][..., 1] = .75
                        before = copy.deepcopy(data)
                        out = self.shift(data, sign=sign)
                        self.assert_preserved(before, out)
                        self.assertGreater(np.abs(out['keypoints'] - before['keypoints']).max(), 0)

    def test_edge_point_prevents_outward_shift(self):
        for x, sign in ((0, 0), (99, 1)):
            with self.subTest(x=x):
                data = sample()
                data['keypoints'][0, 0, 0] = x
                before = copy.deepcopy(data)
                out = self.shift(data, sign=sign)
                self.assert_preserved(before, out)
                np.testing.assert_array_equal(out['keypoints'], before['keypoints'])
                np.testing.assert_array_equal(out['img'], before['img'])

    def test_image_moves_with_points_and_stationary_region_is_unchanged(self):
        data = sample()
        data['img'][:] = 0
        for x, y in data['keypoints'][0].astype(int):
            cv2.circle(data['img'], (x, y), 1, (255, 255, 255), -1)
        before = copy.deepcopy(data)
        out = self.shift(data, ratio=.25)
        np.testing.assert_allclose(out['keypoints'][0, :4], before['keypoints'][0, :4] + [5, 0])
        np.testing.assert_array_equal(out['keypoints'][0, 4:], before['keypoints'][0, 4:])
        np.testing.assert_array_equal(out['img'][28:], before['img'][28:])
        for x, y in out['keypoints'][0].astype(int):
            self.assertEqual(out['img'][y, x, 0], 255)

    def test_unlabelled_points_stay_unlabelled(self):
        data = sample()
        data['keypoints_visible'][0, 0] = 0
        data['keypoints'][0, 0] = [-1000, -1000]
        data['num_keypoints'] = 21
        before = copy.deepcopy(data)
        self.assert_preserved(before, self.shift(data))

    def test_full_shift_crop_pipeline_keeps_targets_including_skipped_crop(self):
        config = Config.fromfile(str(ROOT / 'configs/rtmpose-m_lumbar-coco2000_vertebrae-shift-crop.py'))
        for crop_prob in (0, 1):
            steps = copy.deepcopy(config.train_pipeline[1:])
            steps[0].update(prob=1, max_shift_ratio=10)
            steps[3]['apply_prob'] = crop_prob
            pipeline = Compose(steps)
            for seed in range(20):
                with self.subTest(crop_prob=crop_prob, seed=seed):
                    np.random.seed(seed)
                    out = pipeline(sample())
                    labels = out['data_samples'].gt_instance_labels
                    self.assertTrue((labels.keypoint_weights > 0).all())
                    self.assertTrue((labels.keypoint_x_labels.sum(axis=-1) > 0).all())
                    self.assertTrue((labels.keypoint_y_labels.sum(axis=-1) > 0).all())

    def test_all_training_landmarks_survive_shift_and_crop(self):
        from mmpose.datasets.transforms import TopdownAffine
        coco = json.loads((ROOT / 'data/LumbarCoco2000/annotations/lumbar_keypoints_train.json').read_text())
        images = {image['id']: image for image in coco['images']}
        shift = RandomVertebraeShift(prob=1, max_shift_ratio=1.25)
        crop = RandomVertebraeCrop(ROOT / 'reports/vertebrae_statistics_train.json')
        affine = TopdownAffine(input_size=(256, 512))
        np.random.seed(21)
        for annotation in coco['annotations']:
            image = images[annotation['image_id']]
            # Uniformly reduce source geometry to keep this exhaustive check
            # cheap; gap directions, angles and relative shifts are unchanged.
            factor = 128 / max(image['width'], image['height'])
            width, height = np.ceil(np.array([image['width'], image['height']]) * factor).astype(int)
            points = np.asarray(annotation['keypoints'], np.float32).reshape(22, 3)
            data = dict(img=np.zeros((height, width, 3), np.uint8),
                        keypoints=points[None, :, :2] * factor,
                        keypoints_visible=points[None, :, 2].copy(),
                        num_keypoints=int(np.count_nonzero(points[:, 2])),
                        bbox_center=np.array([[width / 2, height / 2]], np.float32),
                        bbox_scale=np.array([[width, height]], np.float32))
            before = copy.deepcopy(data)
            out = shift(data)
            self.assert_preserved(before, out)
            out = affine(crop(out))
            xy = out['transformed_keypoints'][0]
            self.assertTrue(((xy >= 0) & (xy < [256, 512])).all(), image['file_name'])

    def test_real_images_keep_all_target_weights(self):
        config = Config.fromfile(str(ROOT / 'configs/rtmpose-m_lumbar-coco2000_vertebrae-shift-crop.py'))
        dataset_config = copy.deepcopy(config.train_dataloader.dataset)
        dataset_config.pipeline = []
        dataset = DATASETS.build(dataset_config)
        for crop_prob in (0, 1):
            steps = copy.deepcopy(config.train_pipeline)
            steps[1].update(prob=1, max_shift_ratio=1.25)
            steps[4]['apply_prob'] = crop_prob
            pipeline = Compose(steps)
            for index in (0, 7, 100, 800, 1599):
                with self.subTest(crop_prob=crop_prob, index=index):
                    np.random.seed(42 + index)
                    out = pipeline(dataset.get_data_info(index))
                    labels = out['data_samples'].gt_instance_labels
                    self.assertTrue((labels.keypoint_weights > 0).all())
                    self.assertTrue((labels.keypoint_x_labels.sum(axis=-1) > 0).all())
                    self.assertTrue((labels.keypoint_y_labels.sum(axis=-1) > 0).all())


class CropBoundaryTests(unittest.TestCase):
    def crop(self, data):
        return RandomVertebraeCrop(
            expansion_ranges={side: (0, 0) for side in ('left', 'right', 'top', 'bottom')},
            rotation_range=(0, 0), safety_margin_px=0)(data)

    def test_zero_margins_keep_points_strictly_inside_affine_output(self):
        from mmpose.datasets.transforms import TopdownAffine
        out = TopdownAffine(input_size=(256, 512))(self.crop(sample()))
        xy = out['transformed_keypoints'][0]
        self.assertTrue((xy >= 0).all())
        self.assertTrue((xy < [256, 512]).all())

    def test_crop_uses_visibility_channel_and_preserves_weights(self):
        for channels in (1, 2):
            with self.subTest(channels=channels):
                data = sample()
                data['keypoints_visible'] = np.ones((1, 22, channels), np.float32)
                if channels == 2:
                    data['keypoints_visible'][..., 1] = .75
                before = copy.deepcopy(data)
                out = self.crop(data)
                np.testing.assert_array_equal(out['keypoints_visible'], before['keypoints_visible'])
                np.testing.assert_array_equal(out['keypoints'], before['keypoints'])


if __name__ == '__main__':
    unittest.main()
