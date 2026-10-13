"""Exact recognition evidence, independent of approximate container frame seeks."""
import os
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from core.segment_dataset import read_frame_at, read_image, write_image


LOCATOR_FIELDS = ('locator_version', 'source_video', 'frame_index', 'snapshot_path')


def save_evidence(frame, source, seconds, frame_index, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (uuid4().hex + '.png')
    write_image(path, frame)
    return dict(locator_version=1, source_video=os.path.abspath(source),
                video_sec=float(seconds), frame_index=frame_index,
                snapshot_path=str(path.resolve()))


def load_snapshot(record):
    path = record.get('snapshot_path')
    if not path or not Path(path).is_file():
        return None
    try:
        return read_image(path)
    except (OSError, cv2.error, ValueError):
        return None


def locate_record(cap, record):
    """Verify new records by pixels; never present an unverified seek as exact.

    Legacy records have no image identity and remain explicitly approximate.
    A bounded forward scan tolerates inaccurate MP4 seeks around keyframes.
    """
    seconds = float(record['video_sec'])
    expected = load_snapshot(record)
    ok, frame = read_frame_at(cap, seconds)
    if expected is None:
        return frame if ok else None, False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    tolerance = 1. / max(1., fps) + 1e-4

    def same_time():
        return abs(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 - seconds) <= tolerance

    if ok and same_time() and np.array_equal(frame, expected):
        return frame, True
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0., seconds - 60.) * 1000)
    thumb = cv2.resize(expected, (80, 45))
    for _ in range(6000):
        ok, frame = cap.read()
        if not ok:
            break
        if (same_time() and frame.shape == expected.shape and
                np.array_equal(cv2.resize(frame, (80, 45)), thumb) and
                np.array_equal(frame, expected)):
            return frame, True
        if cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 > seconds + 60:
            break
    return None, False
