"""Rigid sliding of two lumbar image half-planes along an intervertebral gap."""

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

from .vertebrae_crop import KEYPOINT_NAMES


# F/B identify matching anatomical ends, including in mirrored images. Do not
# sort each plate by x: that would break correspondence on steeply tilted plates.
INTERVERTEBRAL_INDICES = tuple(
    tuple(KEYPOINT_NAMES.index(name) for name in (
        f'L{level}BF', f'L{level}BB',
        f'L{level + 1}TF' if level < 5 else 'S1TF',
        f'L{level + 1}TB' if level < 5 else 'S1TB'))
    for level in range(1, 6)
)


@TRANSFORMS.register_module()
class RandomVertebraeShift(BaseTransform):
    """Translate the cranial half-plane parallel to a randomly selected gap.

    Run on one top-down instance after LoadImage and before crop calculation,
    TopdownAffine and GenerateTarget. ``keypoints`` must have shape (1, 22, D),
    D >= 2; ``keypoints_visible`` may have shape (1, 22), (1, 22, 1) or
    (1, 22, 2). Only image pixels and xy coordinates are updated; visibility,
    weights and landmark counts are preserved. The sampled displacement is
    reduced when necessary to keep every labelled point inside the image.
    Raw annotations, bbox and image metadata are preserved. Use FullImageBBox
    for a full-image crop, or calculate a landmark-based crop afterwards.

    Each gap is sampled with probability 1/5. If its four defining landmarks
    are unavailable or degenerate, skip without resampling another gap.
    Points exactly on the cut belong to the stationary half-plane.

    Args:
        prob (float): Probability of applying the transform. Defaults to 0.5.
        max_shift_ratio (float): Maximum displacement divided by cut segment
            length. Defaults to 0.25 (up to 25% in either direction).
    """

    def __init__(self, prob=0.5, max_shift_ratio=0.25):
        super().__init__()
        if not np.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError('prob must be finite and between 0 and 1')
        if not np.isfinite(max_shift_ratio) or max_shift_ratio < 0:
            raise ValueError('max_shift_ratio must be finite and nonnegative')
        self.prob = float(prob)
        self.max_shift_ratio = float(max_shift_ratio)

    @staticmethod
    def _cut(points, visible, gap):
        indices = list(INTERVERTEBRAL_INDICES[gap])
        plate = points[indices, :2].astype(np.float64)
        if not np.all(visible[indices] > 0) or not np.isfinite(plate).all():
            return None
        a, b, c, d = plate
        m1, m2 = (a + c) / 2, (b + d) / 2
        tangent = m2 - m1
        length = np.linalg.norm(tangent)
        if length < 1e-6:
            return None
        normal = np.array([-tangent[1], tangent[0]]) / length
        upper_side = ((a + b) / 2 - m1) @ normal
        if abs(upper_side) < 1e-6:
            return None
        # Negative signed distances always mean the anatomical upper part,
        # irrespective of plate direction, reflection or image rotation.
        if upper_side > 0:
            normal = -normal
        return m1, tangent, normal

    @staticmethod
    def _slide_image(image, upper, delta):
        height, width = image.shape[:2]
        matrix = np.array([[1, 0, delta[0]], [0, 1, delta[1]]], np.float64)
        upper_u8 = upper.astype(np.uint8)
        copy_supported = image.dtype in (
            np.uint8, np.int8, np.uint16, np.int16, np.int32,
            np.float32, np.float64)
        mask = upper.astype(np.float32)
        # Normalize interpolation at the cut using only upper source pixels.
        # Otherwise lower pixels bleed across the cut, or a masked source
        # produces a dark seam. Border weight 1 retains black outer padding.
        weight = cv2.warpAffine(mask, matrix, (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=1)
        source_dtype = np.result_type(image.dtype, np.float32)
        if copy_supported:
            # Mask before expanding uint8 images to float32. OpenCV zeroes
            # unselected pixels when no destination is supplied and avoids
            # NumPy's boolean assignment on the larger float array.
            source = cv2.copyTo(image, upper_u8).reshape(image.shape)
            source = source.astype(source_dtype, copy=False)
        else:
            source = image.astype(source_dtype)
            source[~upper] = 0
        shifted = cv2.warpAffine(source, matrix, (width, height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if image.ndim == 3:
            # OpenCV squeezes singleton channels.
            shifted = shifted.reshape(image.shape)
            weight = weight[..., None]
        np.divide(shifted, weight, out=shifted, where=weight > 0)
        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            # Keep NumPy's original rounding, but reuse the warp buffer.
            np.rint(shifted, out=shifted)
            np.clip(shifted, limits.min, limits.max, out=shifted)
        output = image.copy()
        # One destination half-plane mask controls compositing: lower pixels
        # are never overwritten, and no separately rounded masks can overlap.
        if copy_supported:
            # copyTo writes directly through the mask instead of gathering
            # and scattering two large arrays of selected RGB pixels.
            cv2.copyTo(shifted.astype(image.dtype, copy=False), upper_u8, output)
        else:
            output[upper] = shifted[upper].astype(image.dtype)
        return output

    def transform(self, results):
        if np.random.random() >= self.prob or self.max_shift_ratio == 0:
            return results
        if results.get('transformed_keypoints') is not None:
            raise ValueError('RandomVertebraeShift must precede TopdownAffine')
        if any(results.get(key) is None for key in (
                'img', 'keypoints', 'keypoints_visible')):
            return results
        points = np.asarray(results['keypoints'])
        visibility = np.asarray(results['keypoints_visible'])
        if points.ndim != 3 or points.shape[1] != 22 or points.shape[2] < 2:
            return results
        if points.shape[0] != 1:
            raise ValueError('RandomVertebraeShift expects one top-down instance')
        if (visibility.shape != (1, 22) and
                visibility.shape not in ((1, 22, 1), (1, 22, 2))):
            return results
        visible = visibility[0] if visibility.ndim == 2 else visibility[0, :, 0]
        gap = np.random.randint(5)
        cut = self._cut(points[0], visible, gap)
        if cut is None:
            return results
        origin, tangent, normal = cut
        ratio = np.random.uniform(0, self.max_shift_ratio)
        sign = 1 if np.random.randint(2) else -1
        delta = sign * ratio * tangent
        if not np.any(delta):
            return results

        image = results['img']
        height, width = image.shape[:2]
        xy = points[0, :, :2].astype(np.float64)
        labelled = visible > 0
        inside = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) &
                  (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height))
        # Invalid source labels cannot be repaired by a rigid slide. Preserve
        # the sample instead of deleting their visibility or clipping points.
        if not np.all(inside[labelled]):
            return results
        moving = ((xy - origin) @ normal < 0) & np.isfinite(xy).all(axis=1)
        moving_labelled = xy[moving & labelled]
        fraction = 1.0
        if len(moving_labelled):
            # Bound the common translation, not individual coordinates: the
            # image and all moving landmarks must undergo the same rigid slide.
            # Keep moving labels at most on the last pixel center, leaving room
            # for float32 rounding at the exclusive width/height boundaries.
            for axis, last_pixel in enumerate((width - 1, height - 1)):
                if delta[axis] > 0:
                    fraction = min(fraction, (last_pixel - moving_labelled[:, axis].max()) / delta[axis])
                elif delta[axis] < 0:
                    fraction = min(fraction, moving_labelled[:, axis].min() / -delta[axis])
        if fraction <= 0:
            return results
        delta *= fraction

        shifted_points = points.astype(np.result_type(points.dtype, np.float32), copy=True)
        shifted_points[0, moving, :2] += delta
        xy = shifted_points[0, :, :2]
        inside = (np.isfinite(xy).all(axis=1) & (xy[:, 0] >= 0) &
                  (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height))
        # Check the stored dtype too, before committing either pixels or points.
        if not np.all(inside[labelled]):
            return results
        y, x = np.ogrid[:height, :width]
        upper = ((x - origin[0]) * normal[0] +
                 (y - origin[1]) * normal[1]) < 0

        results['img'] = self._slide_image(image, upper, delta)
        results['keypoints'] = shifted_points
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(prob={self.prob}, '
                f'max_shift_ratio={self.max_shift_ratio})')
