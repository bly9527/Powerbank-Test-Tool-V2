"""Recover earlier visual evidence after a delayed numeric recognition.

Uses an independent decoder. Every returned timestamp belongs to a decoded
image; no time offset is subtracted from the instrument reading.
"""
import cv2
import os
from collections import OrderedDict, Counter
from core.roi_geometry import crop_roi
from core.segment_dataset import read_frame_at
from core.visual_tracking import VisualTracker


class TimingCropCache:
    """Bounded, lossless SOC crops shared by both units in one analysis.

    Only the coarse search uses this cache. Fine decoding and the full evidence
    frame always come from the original video. No image similarity shortcuts.
    """
    def __init__(self, regions=(), max_entries=512, max_bytes=16*1024*1024):
        self.regions = tuple((tuple(roi), rotation) for roi, rotation in regions if roi)
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self.bytes_used = 0
        self.frames = OrderedDict()
        self.stats = Counter()
        self.identity = None

    def begin(self, source):
        path = os.path.normcase(os.path.abspath(source))
        try:
            stat = os.stat(path)
            identity = (path, stat.st_size, stat.st_mtime_ns)
        except OSError:
            identity = (path, None, None)
        if identity != self.identity:
            self.clear()
            self.identity = identity

    def clear(self):
        self.frames.clear()
        self.bytes_used = 0

    def read_crop(self, cap, seconds, roi, rotation):
        key, region = float(seconds), (tuple(roi), rotation)
        previous = self.frames.get(key)
        if previous is not None and region in previous[1]:
            self.frames.move_to_end(key)
            self.stats['coarse_hits'] += 1
            return True, previous[1][region], previous[0]
        self.stats['coarse_decodes'] += 1
        ok, image = read_frame_at(cap, seconds)
        if not ok:
            return False, None, 0.
        actual = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
        crop = crop_roi(image, roi, rotation, strict=True)
        if self.max_entries:
            crops = {region: crop}
            for other_roi, other_rotation in self.regions:
                other = (other_roi, other_rotation)
                if other not in crops:
                    try:
                        crops[other] = crop_roi(image, other_roi, other_rotation, strict=True)
                    except ValueError:
                        # A disabled/invalid second unit must not block this one.
                        pass
            if key in self.frames:
                self.bytes_used -= sum(c.nbytes for c in self.frames[key][1].values())
            self.frames[key] = (actual, crops)
            self.bytes_used += sum(c.nbytes for c in crops.values())
            while len(self.frames) > self.max_entries or self.bytes_used > self.max_bytes:
                _, (_, removed) = self.frames.popitem(last=False)
                self.bytes_used -= sum(c.nbytes for c in removed.values())
        return True, crop, actual


def refine_keyframe(source, roi, rotation, value, anchor_frame, anchor_sec,
                    anchor_index, *, lower_bound=0., competing=(),
                    max_lookback=120., cancelled=lambda: False, crop_cache=None):
    """Return an earlier frame supported by >=3 distinct matching frames.

    A half-second coarse scan locates candidate episodes; a frame-by-frame
    scan around each episode keeps its first clear image, not its third vote.
    Ambiguous shapes, missing digits and isolated matches cannot move a record.
    This is a bounded evidence search, not a claim of finding the physical
    transition when the display is unreadable or lies outside the window.
    """
    start = max(0., float(lower_bound), float(anchor_sec) - max_lookback)
    if start >= anchor_sec or cancelled():
        return None
    matcher = VisualTracker()
    matcher.add_manual_reference(crop_roi(anchor_frame, roi, rotation, strict=True), value)
    for other_value, other_frame in competing:
        if other_value != value:
            matcher.add_manual_reference(crop_roi(other_frame, roi, rotation, strict=True), other_value)
    if not matcher.manual_references:
        return None
    if crop_cache is not None:
        crop_cache.begin(source)
    coarse = cv2.VideoCapture(str(source))
    fine = cv2.VideoCapture(str(source))
    try:
        if not coarse.isOpened() or not fine.isOpened():
            return None
        fps = coarse.get(cv2.CAP_PROP_FPS) or 30.
        last_fine_end = -1.
        sample = start
        while sample < anchor_sec and not cancelled():
            if crop_cache is not None:
                ok, crop, seconds = crop_cache.read_crop(coarse, sample, roi, rotation)
            else:
                ok, image = read_frame_at(coarse, sample)
                seconds = coarse.get(cv2.CAP_PROP_POS_MSEC) / 1000.
                crop = crop_roi(image, roi, rotation, strict=True) if ok else None
            sample += .5
            if not ok:
                break
            if seconds >= anchor_sec or seconds <= last_fine_end:
                continue
            if matcher.match_manual(crop) != value:
                continue
            fine_start = max(start, seconds - 2.)
            fine_end = min(anchor_sec, seconds + 1.)
            ok, image = read_frame_at(fine, fine_start)
            matches = []
            seen = set()
            while ok and not cancelled():
                position = fine.get(cv2.CAP_PROP_POS_MSEC) / 1000.
                index = int(fine.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                if position > fine_end:
                    break
                if index not in seen and position < anchor_sec - .5 / fps:
                    seen.add(index)
                    matched = matcher.match_manual(crop_roi(image, roi, rotation, strict=True))
                    if matched == value:
                        matches = [item for item in matches if position - item[1] <= .5]
                        matches.append((image.copy(), position, index))
                        if len(matches) >= 3:
                            first, sec, frame_index = matches[0]
                            return dict(frame=first, video_sec=sec, frame_index=frame_index,
                                        timing_original_sec=float(anchor_sec),
                                        timing_search_start_sec=start,
                                        timing_method='visual_backtrack')
                    elif matched is not None:
                        matches.clear()
                ok, image = fine.read()
            last_fine_end = fine_end
        return None
    finally:
        coarse.release()
        fine.release()
