"""Albumentations GridDistortion for 22-point lumbar MMPose samples."""

import inspect

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LumbarGridDistortion(BaseTransform):
    """Warp pixels and 22 ordered lumbar landmarks using Albumentations.

    Run on image pixels before TopdownAffine/GenerateTarget. For distortion
    after spine alignment, that alignment must already be applied to pixels
    and landmarks: RandomVertebraeCrop alone only sets pending affine geometry.
    Coordinates are pixel xy, shape (1, 22, D), D >= 2. Visibility may be
    (1, 22), (1, 22, 1) or (1, 22, 2); its first channel selects labels.
    Extra coordinate/visibility channels and original annotation records are
    preserved. Points are never removed or reordered. Unlabelled out-of-frame
    placeholders are retained; invalid labelled inputs fail explicitly.
    A full-canvas xyxy bbox stays unchanged because the canvas size is fixed.

    ``distort_range`` maps to ``distort_limit`` in Albumentations 2.0.8.
    Older APIs without the requested keypoint remapping support are rejected
    instead of silently ignoring options or substituting another algorithm.
    """

    def __init__(self, num_steps=5, distort_range=(-0.05, 0.05),
                 normalized=True, p=0.5, interpolation=cv2.INTER_LINEAR,
                 border_mode=cv2.BORDER_CONSTANT, fill=0,
                 keypoint_remapping_method='direct'):
        super().__init__()
        import albumentations as A

        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('p must be finite and between 0 and 1')
        self.p = float(p)
        parameters = inspect.signature(A.GridDistortion).parameters
        range_name = 'distort_range' if 'distort_range' in parameters else 'distort_limit'
        options = dict(
            num_steps=num_steps, normalized=normalized,
            interpolation=interpolation, border_mode=border_mode, fill=fill,
            keypoint_remapping_method=keypoint_remapping_method)
        options[range_name] = distort_range
        missing = options.keys() - parameters.keys()
        if missing:
            raise RuntimeError(
                f'Albumentations {A.__version__} lacks GridDistortion options '
                f'{sorted(missing)}; use albumentations==2.0.8')
        # Probability and seeds come from MMPose's worker-seeded NumPy RNG.
        # Albumentations 2 uses an independent RNG; reseed it per application
        # to avoid identical streams in forked dataloader workers.
        self.augmentation = A.Compose(
            [A.GridDistortion(**options, p=1.0, strict=True)],
            keypoint_params=A.KeypointParams(
                format='xy', remove_invisible=False,
                label_fields=['point_ids'], check_each_transform=False))

    def transform(self, results):
        if self.p == 0 or np.random.random() >= self.p:
            return results
        if results.get('transformed_keypoints') is not None:
            raise ValueError('LumbarGridDistortion must precede final TopdownAffine')
        points = np.asarray(results['keypoints'])
        visibility = np.asarray(results['keypoints_visible'])
        if (points.ndim != 3 or points.shape[:2] != (1, 22) or
                points.shape[2] < 2 or not np.issubdtype(points.dtype, np.floating)):
            raise ValueError('Expected floating-point keypoints with shape (1, 22, D), D >= 2')
        if visibility.shape not in ((1, 22), (1, 22, 1), (1, 22, 2)):
            raise ValueError('Unexpected keypoints_visible shape')
        if not np.isfinite(points).all() or not np.isfinite(visibility).all():
            raise ValueError('Non-finite input keypoints or visibility')
        height, width = results['img'].shape[:2]
        if min(height, width) < self.augmentation.transforms[0].num_steps:
            raise ValueError('GridDistortion needs image dimensions >= num_steps')
        if 'bbox' in results and not np.allclose(results['bbox'], [[0, 0, width, height]]):
            raise ValueError('LumbarGridDistortion expects a full-image xyxy bbox')
        xy = points[0, :, :2].copy()
        visible = visibility[0] if visibility.ndim == 2 else visibility[0, :, 0]
        inside = ((xy >= 0).all(axis=1) & (xy < [width, height]).all(axis=1))
        if not np.all(inside[visible > 0]):
            raise ValueError('Labelled input keypoint outside the image')
        # Do not send out-of-frame unlabelled placeholders to inverse lookup.
        # Keep all 22 entries and restore those placeholders afterwards.
        xy[~inside] = 0
        self.augmentation.set_random_seed(int(np.random.randint(0, 2**32)))
        output = self.augmentation(
            image=results['img'], keypoints=xy, point_ids=np.arange(22))
        ids = np.asarray(output['point_ids'], dtype=int)
        warped = np.asarray(output['keypoints'], dtype=points.dtype)
        if (warped.shape != (22, 2) or not np.isfinite(warped).all() or
                not np.array_equal(np.sort(ids), np.arange(22))):
            raise RuntimeError('GridDistortion lost landmarks or returned invalid coordinates')
        updated = points.copy()
        updated[0, ids, :2] = warped
        updated[0, ~inside, :2] = points[0, ~inside, :2]
        xy = updated[0, :, :2]
        in_frame = ((xy >= 0).all(axis=1) & (xy < [width, height]).all(axis=1))
        weights = visibility.copy()
        weights[0, ~in_frame] = 0
        results['img'] = output['image']
        results['keypoints'] = updated
        results['keypoints_visible'] = weights
        return results
