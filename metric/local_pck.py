"""Lumbar segment normalized PCK metric for MMPose 1.3.x."""

import numpy as np
from mmengine.evaluator import BaseMetric
from mmpose.registry import METRICS


KEYPOINT_NAMES = [
    *(f'L{level}{plate}{side}' for level in range(1, 6)
      for plate in ('T', 'B') for side in ('F', 'B')),
    'S1TF', 'S1TB',
]
SEGMENT_NAMES = [
    *(pair for level in range(1, 6) for pair in
      ((f'L{level}TF', f'L{level}TB'), (f'L{level}BF', f'L{level}BB'))),
    ('S1TF', 'S1TB'),
]
SEGMENTS = np.array([(KEYPOINT_NAMES.index(a), KEYPOINT_NAMES.index(b))
                     for a, b in SEGMENT_NAMES], dtype=np.int64)


@METRICS.register_module()
class LocalPCK(BaseMetric):
    """Fraction of eligible visible points with error <= threshold * length.

    mode='segment' uses each point's own GT segment length. mode='image_mean'
    uses the mean length of all valid GT segments in the image. A segment is
    valid only when both endpoints are visible, finite and have nonzero length.
    Each valid segment contributes its two endpoints to the denominator.
    """

    def __init__(self, threshold=0.05, mode='segment',
                 collect_device='cpu', prefix=None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        if threshold < 0 or not np.isfinite(threshold):
            raise ValueError('threshold must be finite and nonnegative')
        if mode not in ('segment', 'image_mean'):
            raise ValueError("mode must be 'segment' or 'image_mean'")
        self.threshold = float(threshold)
        self.mode = mode

    def process(self, data_batch, data_samples):
        for sample in data_samples:
            gt = np.asarray(sample['gt_instances']['keypoints'])[0, :, :2]
            pred = np.asarray(sample['pred_instances']['keypoints'])[0, :, :2]
            visible = np.asarray(sample['gt_instances']['keypoints_visible'])[0]
            if visible.ndim > 1:
                visible = visible[:, 0]
            if len(gt) != len(KEYPOINT_NAMES) or pred.shape != gt.shape:
                raise ValueError('LocalPCK requires the 22-point LumbarCoco layout')

            endpoints = gt[SEGMENTS]
            lengths = np.linalg.norm(endpoints[:, 0] - endpoints[:, 1], axis=1)
            valid = ((visible[SEGMENTS] > 0).all(axis=1)
                     & np.isfinite(endpoints).all(axis=(1, 2))
                     & np.isfinite(lengths) & (lengths > 0))
            if not valid.any():
                self.results.append((0, 0))
                continue

            indices = SEGMENTS[valid].reshape(-1)
            norms = (np.full(len(indices), lengths[valid].mean())
                     if self.mode == 'image_mean'
                     else np.repeat(lengths[valid], 2))
            errors = np.linalg.norm(pred[indices] - gt[indices], axis=1)
            correct = np.count_nonzero(np.isfinite(errors)
                                       & (errors <= self.threshold * norms))
            self.results.append((int(correct), len(indices)))

    def compute_metrics(self, results):
        correct = sum(item[0] for item in results)
        total = sum(item[1] for item in results)
        return {'PCK': correct / total if total else 0.0}
