"""RandomVertebraeCrop and its shared geometry for empirical spine crops.

All rotation, landmark-axis and real-image margin calculations live here.
The statistics collector uses these same helpers as the augmentation.
"""
import json
from pathlib import Path

import numpy as np
from mmcv.transforms import BaseTransform
from mmpose.registry import TRANSFORMS

KEYPOINT_NAMES = [
    *(f'L{level}{plate}{side}' for level in range(1, 6)
      for plate in ('T', 'B') for side in ('F', 'B')),
    'S1TF', 'S1TB',
]
AXIS_INDICES = (0, 1, 20, 21)
SIDES = ('left', 'right', 'top', 'bottom')


def rotation(degrees):
    theta = np.deg2rad(degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def spine_angle(points, visible):
    """Signed lean from vertical: positive when S1 is right of L1."""
    points = np.asarray(points, dtype=np.float64)
    visible = np.asarray(visible).reshape(-1)
    if points.shape != (22, 2) or visible.shape != (22,):
        raise ValueError('Expected 22 lumbar landmarks')
    if not np.all(visible[list(AXIS_INDICES)] > 0):
        raise ValueError('L1 top and S1 endpoints must be labelled')
    if not np.isfinite(points[visible > 0]).all():
        raise ValueError('Non-finite labelled landmark')
    delta = points[20:22].mean(axis=0) - points[:2].mean(axis=0)
    if delta[1] <= 0:
        raise ValueError('S1 midpoint must lie below L1 midpoint')
    return float(np.rad2deg(np.arctan2(delta[0], delta[1])))


def line_interval(polygon, fixed, axis):
    """Intersect a horizontal (axis=0) or vertical line with the real image."""
    values = []
    other = 1 - axis
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if abs(b[other] - a[other]) < 1e-10:
            if abs(fixed - a[other]) < 1e-8:
                values.extend([a[axis], b[axis]])
        else:
            t = (fixed - a[other]) / (b[other] - a[other])
            if -1e-9 <= t <= 1 + 1e-9:
                values.append(a[axis] + t * (b[axis] - a[axis]))
    if not values:
        raise ValueError('Landmark region does not intersect original image')
    return min(values), max(values)


def measure(points, visible, width, height):
    """Minimum real-image clearance over each side of the verticalized KP box.

    Use the rotated original rectangle, never its axis-aligned canvas. Boundary
    coordinates are continuous [0,width] x [0,height], as for COCO bboxes.
    A KP box corner may lie outside the polygon even though every KP is inside;
    such a side has zero available margin, with signed clearance retained.
    """
    angle = spine_angle(points, visible)
    r = rotation(angle)
    valid = np.asarray(visible).reshape(-1) > 0
    q = np.asarray(points, dtype=np.float64)[valid] @ r.T
    low, high = q.min(axis=0), q.max(axis=0)
    size = high - low
    if np.any(size <= 0):
        raise ValueError('Degenerate landmark region')
    polygon = np.array([[0, 0], [width, 0], [width, height], [0, height]]) @ r.T
    # For a convex polygon the most restrictive side boundary occurs at an
    # endpoint of the box side (piecewise convex/concave boundary functions).
    horizontal = [line_interval(polygon, y, 0) for y in (low[1], high[1])]
    vertical = [line_interval(polygon, x, 1) for x in (low[0], high[0])]
    signed = np.array([
        low[0] - max(v[0] for v in horizontal),
        min(v[1] for v in horizontal) - high[0],
        low[1] - max(v[0] for v in vertical),
        min(v[1] for v in vertical) - high[1],
    ])
    margins = np.maximum(signed, 0)
    return dict(
        angle_deg=angle,
        keypoint_extent_px=size.tolist(),
        margins_px=dict(zip(SIDES, margins.tolist())),
        margins_relative=dict(zip(SIDES, (margins / size[[0, 0, 1, 1]]).tolist())),
        signed_clearance_px=dict(zip(SIDES, signed.tolist())),
    )


@TRANSFORMS.register_module()
class FullImageBBox(BaseTransform):
    """Override annotation/detector bbox without consulting ground-truth points."""

    def transform(self, results):
        height, width = results['img'].shape[:2]
        results['bbox'] = np.array([[0, 0, width, height]], dtype=np.float32)
        results['bbox_score'] = np.ones(1, dtype=np.float32)
        results.pop('bbox_center', None)
        results.pop('bbox_scale', None)
        results['bbox_rotation'] = np.zeros(1, dtype=np.float32)
        return results


@TRANSFORMS.register_module()
class RandomVertebraeCrop(BaseTransform):
    """Run after FullImageBBox/GetBBoxCenterScale, before TopdownAffine.

    Virtually verticalize the L1-top/S1 axis, then apply a sampled lean and
    crop around all labelled points using independently sampled margins.
    Default ranges come from training means: each margin in [0, mean] and
    target lean in [-mean(abs(angle)), +mean(abs(angle))]. Explicit ranges
    may also be passed without a statistics file.
    Store center/scale/rotation in source coordinates; do not warp the image
    or mutate source keypoints. TopdownAffine performs the sole image warp,
    with black padding and the same transform for keypoints. Its aspect-ratio
    correction only expands this crop, preserving all labelled points.
    """

    def __init__(self, statistics_file=None, expansion_ranges=None,
                 rotation_range=None, apply_prob=1.0, safety_margin_px=2.0):
        super().__init__()
        if not 0 <= apply_prob <= 1 or not np.isfinite(safety_margin_px) or safety_margin_px < 0:
            raise ValueError('Invalid probability or safety margin')
        if statistics_file is not None:
            document = json.loads(Path(statistics_file).read_text())
            if document.get('schema_version') != 1 or document.get('keypoint_names') != KEYPOINT_NAMES:
                raise ValueError('Statistics schema/keypoint order mismatch')
            means = document.get('mean')
            if not isinstance(means, dict) or means.get('count', 0) <= 0:
                raise ValueError('Statistics lack training means; rerun collect_vertebrae_statistics.py')
            if expansion_ranges is None:
                expansion_ranges = {side: (0.0, means['margins_relative'][side]) for side in SIDES}
            if rotation_range is None:
                limit = float(means['absolute_angle_deg'])
                if not np.isfinite(limit) or limit < 0:
                    raise ValueError('Invalid mean absolute angle')
                rotation_range = (-limit, limit)
        if expansion_ranges is None or set(expansion_ranges) != set(SIDES):
            raise ValueError('Provide expansion_ranges for all four sides or a statistics file')
        self.expansion_ranges = {}
        for side in SIDES:
            low, high = expansion_ranges[side]
            if not np.isfinite([low, high]).all() or not 0 <= low <= high:
                raise ValueError(f'Invalid expansion range for {side}')
            self.expansion_ranges[side] = (float(low), float(high))
        if rotation_range is None:
            raise ValueError('Provide rotation_range or a statistics file')
        low, high = rotation_range
        if not np.isfinite([low, high]).all() or not -90 < low <= high < 90:
            raise ValueError('rotation_range must be ordered and inside (-90, 90)')
        self.rotation_range = (float(low), float(high))
        self.apply_prob = float(apply_prob)
        self.safety_margin_px = float(safety_margin_px)

    def transform(self, results):
        if np.asarray(results['bbox_center']).shape != (1, 2):
            raise ValueError('RandomVertebraeCrop expects one top-down instance')
        if float(np.asarray(results.get('bbox_rotation', [0]))[0]) != 0:
            raise ValueError('Do not combine RandomVertebraeCrop with another rotation')
        if np.random.random() >= self.apply_prob:
            return results
        points = np.asarray(results['keypoints'])[0, :, :2]
        visible = np.asarray(results['keypoints_visible'])[0].reshape(-1)
        source_angle = spine_angle(points, visible)
        sampled_angle = float(np.random.uniform(*self.rotation_range))
        sampled_margins = {side: float(np.random.uniform(*self.expansion_ranges[side]))
                           for side in SIDES}
        # R(source_angle) verticalizes, R(-sample_angle) restores sampled lean.
        net_angle = source_angle - sampled_angle
        r = rotation(net_angle)
        q = points[visible > 0].astype(np.float64) @ r.T
        low, high = q.min(axis=0), q.max(axis=0)
        vertical_points = points[visible > 0].astype(np.float64) @ rotation(source_angle).T
        extent = np.ptp(vertical_points, axis=0)
        margins = np.array([sampled_margins[s] for s in SIDES])
        margins = margins * extent[[0, 0, 1, 1]] + self.safety_margin_px
        low -= margins[[0, 2]]
        high += margins[[1, 3]]
        results['bbox_center'] = (((low + high) / 2) @ r)[None].astype(np.float32)
        results['bbox_scale'] = (high - low)[None].astype(np.float32)
        # MMPose rotates source crop axes +rot, applying R(-rot) to pixels.
        results['bbox_rotation'] = np.array([-net_angle], dtype=np.float32)
        results['vertebrae_crop'] = dict(
            source_angle_deg=source_angle, sampled_angle_deg=sampled_angle,
            sampled_margins_relative=sampled_margins)
        return results
