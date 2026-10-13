"""Shared geometry for preview, analysis and sample capture.

Four-value ROIs are legacy axis-aligned rectangles. An optional fifth value
rotates the rectangle clockwise around its centre, independently of the legacy
text orientation correction (0/90/180/270).
"""
import math

import cv2
import numpy as np


def roi_corners(roi):
    if len(roi) not in (4, 5):
        raise ValueError('选区数据无效，请重新画框')
    x, y, w, h = map(float, roi[:4])
    angle = float(roi[4]) if len(roi) == 5 else 0.0
    if not all(math.isfinite(v) for v in (x, y, w, h, angle)) or w < 1 or h < 1:
        raise ValueError('选区尺寸或角度无效，请重新画框')
    local = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
    radians = math.radians(angle % 360)
    rotation = np.array([[math.cos(radians), -math.sin(radians)],
                         [math.sin(radians), math.cos(radians)]])
    return (local @ rotation.T + [x+w/2, y+h/2]).astype(np.float32)


def roi_in_bounds(roi, width, height):
    points = roi_corners(roi)
    return bool(np.all(points >= -1e-4) and np.all(points[:, 0] <= width+1e-4)
                and np.all(points[:, 1] <= height+1e-4))


def crop_roi(frame, roi, orientation=0, strict=False):
    if frame is None or roi is None:
        return None
    fh, fw = frame.shape[:2]
    points = roi_corners(roi)
    angle = float(roi[4]) % 360 if len(roi) == 5 else 0
    if (strict or angle) and not roi_in_bounds(roi, fw, fh):
        raise ValueError('旋转选区超出视频边界，请重新画框或调整角度')
    x, y, w, h = map(int, roi[:4])
    if not angle:
        # Preserve legacy crops pixel for pixel, including their clamp behavior.
        x1, y1 = max(0, min(fw-1, x)), max(0, min(fh-1, y))
        x2, y2 = max(x1+1, min(fw, x+w)), max(y1+1, min(fh, y+h))
        image = frame[y1:y2, x1:x2].copy()
    else:
        # Corners describe pixel edges; sample at output pixel centres to avoid
        # trimming strokes or introducing a half-pixel shift at arbitrary angles.
        destination = np.array([[-.5, -.5], [w-.5, -.5],
                                [w-.5, h-.5], [-.5, h-.5]], np.float32)
        transform = cv2.getPerspectiveTransform(points-.5, destination)
        image = cv2.warpPerspective(frame, transform, (w, h), flags=cv2.INTER_CUBIC)
    corrections = {90: cv2.ROTATE_90_COUNTERCLOCKWISE,
                   180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}
    correction = float(orientation) % 360
    if correction in corrections:
        return cv2.rotate(image, corrections[correction])
    if correction:
        raise ValueError('文字朝向必须为 0°、90°、180°或270°；倾斜请拖动框上旋转手柄')
    return image
