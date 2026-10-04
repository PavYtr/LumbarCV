"""Asymmetric lumbar crop expansion for MMPose 1.3.x."""

import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomLumbarBBoxExpansion(BaseTransform):
    """Expand independently chosen sides, then rotate the crop.

    Run after GetBBoxCenterScale and before TopdownAffine. Expansion fractions
    use the *original* bbox width/height, before GetBBoxCenterScale padding.
    The image, original bbox, and keypoints are left untouched. Expanded
    axis-aligned sides are clipped to the image bounds. If a sampled rotation
    would exclude visible keypoints, scale is enlarged just enough to contain
    them in the rotated rectangle. TopdownAffine may enlarge it further to
    match the input aspect ratio, so the final rotated crop can cross an image
    edge and receive padding from cv2.warpAffine.
    """

    SIDES = ('left', 'right', 'top', 'bottom')

    def __init__(self, expansion_ranges, side_probs, apply_prob=1.0,
                 rotation_range=(-15.0, 15.0), clip_to_image=True):
        super().__init__()
        if set(expansion_ranges) != set(self.SIDES):
            raise ValueError('expansion_ranges must specify all four sides')
        if set(side_probs) != set(self.SIDES):
            raise ValueError('side_probs must specify all four sides')
        self.expansion_ranges = {}
        for side in self.SIDES:
            low, high = expansion_ranges[side]
            if not 0 <= low <= high:
                raise ValueError(f'Invalid expansion range for {side}')
            self.expansion_ranges[side] = (float(low), float(high))
        self.side_probs = {}
        for side in self.SIDES:
            prob = side_probs[side]
            if not 0 <= prob <= 1:
                raise ValueError(f'Invalid selection probability for {side}')
            self.side_probs[side] = float(prob)
        if not any(self.side_probs.values()):
            raise ValueError('At least one side must have nonzero probability')
        if not 0 <= apply_prob <= 1:
            raise ValueError('apply_prob must be between 0 and 1')
        low, high = rotation_range
        if low > high:
            raise ValueError('rotation_range must be increasing')
        self.apply_prob = float(apply_prob)
        self.rotation_range = (float(low), float(high))
        self.clip_to_image = bool(clip_to_image)

    def transform(self, results):
        centers = np.asarray(results['bbox_center'])
        scales = np.asarray(results['bbox_scale'])
        bboxes = np.asarray(results['bbox'])
        if centers.shape != (1, 2) or scales.shape != (1, 2) or bboxes.shape != (1, 4):
            raise ValueError('RandomLumbarBBoxExpansion expects one top-down bbox')

        rotation = float(np.asarray(results.get('bbox_rotation', [0.0]))[0])
        if np.random.random() >= self.apply_prob:
            results['bbox_rotation'] = np.array([rotation], dtype=np.float32)
            return results

        chosen = {side: np.random.random() < self.side_probs[side]
                  for side in self.SIDES}
        if not any(chosen.values()):
            # Conditional on augmentation, at least one side must expand.
            weights = np.array([self.side_probs[s] for s in self.SIDES])
            chosen[self.SIDES[np.random.choice(4, p=weights / weights.sum())]] = True

        width, height = bboxes[0, 2:] - bboxes[0, :2]
        if width <= 0 or height <= 0:
            raise ValueError('Original bbox must have positive width and height')
        amounts = {}
        for side in self.SIDES:
            low, high = self.expansion_ranges[side]
            size = width if side in ('left', 'right') else height
            amounts[side] = (np.random.uniform(low, high) * size
                             if chosen[side] else 0.0)

        center = centers[0].astype(np.float64, copy=True)
        scale = scales[0].astype(np.float64, copy=True)
        low = center - scale / 2 - [amounts['left'], amounts['top']]
        high = center + scale / 2 + [amounts['right'], amounts['bottom']]
        if self.clip_to_image:
            if 'img_shape' in results:
                image_height, image_width = results['img_shape'][:2]
            else:
                image_height, image_width = results['img'].shape[:2]
            low = np.maximum(low, 0)
            high = np.minimum(high, [image_width, image_height])
            if np.any(high <= low):
                raise ValueError('Expanded bbox has no overlap with the image')
        center = (low + high) / 2
        scale = high - low
        rotation += np.random.uniform(*self.rotation_range)

        if 'keypoints' in results and 'keypoints_visible' in results:
            points = np.asarray(results['keypoints'])[0, :, :2]
            visible = np.asarray(results['keypoints_visible'])[0]
            if visible.ndim > 1:
                visible = visible[:, 0]
            points = points[(visible > 0) & np.isfinite(points).all(axis=1)]
            if len(points):
                theta = np.deg2rad(rotation)
                cosine, sine = np.cos(theta), np.sin(theta)
                offsets = points - center
                # Crop axes are rotated +theta in source coordinates, so
                # project source points onto them with the inverse rotation.
                rotated_x = offsets[:, 0] * cosine + offsets[:, 1] * sine
                rotated_y = -offsets[:, 0] * sine + offsets[:, 1] * cosine
                # Relative margin survives float32 conversion for large crops.
                required = 2 * np.array([np.max(np.abs(rotated_x)),
                                         np.max(np.abs(rotated_y))])
                scale = np.maximum(scale, required * 1.0001 + 1e-3)

        results['bbox_center'] = center[None].astype(centers.dtype)
        results['bbox_scale'] = scale[None].astype(scales.dtype)
        results['bbox_rotation'] = np.array([rotation], dtype=np.float32)
        return results
