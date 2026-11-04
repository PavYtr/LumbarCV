"""Mask remapping must preserve all lumbar landmarks, including on failure."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from augmentations.grid_distortion import LumbarGridDistortion


class MaskRemappingTest(unittest.TestCase):
    def setUp(self):
        self.sample = dict(
            img=np.zeros((64, 64, 3), dtype=np.uint8),
            keypoints=np.full((1, 22, 2), 30, dtype=np.float32),
            keypoints_visible=np.ones((1, 22), dtype=np.float32),
            bbox=np.array([[0, 0, 64, 64]], dtype=np.float32))
        self.transform = LumbarGridDistortion(p=1, keypoint_remapping_method='mask')
        self.good = dict(
            image=np.ones_like(self.sample['img']),
            keypoints=np.full((22, 2), 31, dtype=np.float32),
            point_ids=np.arange(22))
        self.bad = dict(image=self.good['image'],
                        keypoints=self.good['keypoints'][:-1],
                        point_ids=np.arange(21))
        self.augmentation = Mock()
        self.augmentation.transforms = [SimpleNamespace(
            num_steps=5, keypoint_remapping_method='mask')]
        self.transform.augmentation = self.augmentation

    def test_retry_preserves_all_landmarks(self):
        self.augmentation.side_effect = [self.bad, self.good]
        output = self.transform(self.sample)
        np.testing.assert_array_equal(output['keypoints'][0], self.good['keypoints'])
        np.testing.assert_array_equal(output['img'], self.good['image'])
        self.assertEqual(self.augmentation.call_count, 2)

    def test_exhausted_retries_leave_original_sample_intact(self):
        original = copy.deepcopy(self.sample)
        self.augmentation.return_value = self.bad
        with self.assertWarnsRegex(RuntimeWarning, 'keeping the original'):
            output = self.transform(self.sample)
        for key, value in original.items():
            np.testing.assert_array_equal(output[key], value)
        self.assertEqual(self.augmentation.call_count, 3)

    def test_direct_still_rejects_invalid_output(self):
        self.augmentation.transforms[0].keypoint_remapping_method = 'direct'
        self.augmentation.return_value = self.bad
        with self.assertRaisesRegex(RuntimeError, 'lost landmarks'):
            self.transform(self.sample)
        self.assertEqual(self.augmentation.call_count, 1)


if __name__ == '__main__':
    unittest.main()
