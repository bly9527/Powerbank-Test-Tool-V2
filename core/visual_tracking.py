"""Per-unit temporary image references. Changes never imply a numeric decrement."""
from collections import deque

import cv2
import numpy as np


class VisualTracker:
    def __init__(self, interval=.5):
        self.references = deque(maxlen=8)
        self.value = None
        self.pending = []
        self.last_index = None
        self.last_seconds = None
        self.interval = interval
        self.status = '等待稳定数字'
        self.last_numeric_seconds = None
        self.manual_references = []

    def add_manual_reference(self, image, value):
        sig = self.signature(image)
        if sig is not None and type(value) is int and 0 <= value <= 100:
            self.manual_references.append((value, sig))

    def match_manual(self, image):
        sig = self.signature(image)
        if sig is None:
            return None
        scores = {}
        for value, ref in self.manual_references:
            # A one-pixel tolerance handles anti-aliasing without accepting
            # missing strokes. Both foregrounds must be covered almost fully.
            kernel = np.ones((3, 3), np.uint8)
            precision = np.count_nonzero(sig & cv2.dilate(ref, kernel)) / max(1, np.count_nonzero(sig))
            recall = np.count_nonzero(ref & cv2.dilate(sig, kernel)) / max(1, np.count_nonzero(ref))
            score = min(precision, recall) if self.similarity(sig, ref) >= .75 else 0.
            scores[value] = max(score, scores.get(value, 0.))
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if ordered and ordered[0][1] >= .975 and (len(ordered) == 1 or ordered[0][1]-ordered[1][1] >= .08):
            return ordered[0][0]
        return None

    @staticmethod
    def signature(image):
        if image is None or image.size == 0:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        lo, hi = np.percentile(gray, [10, 99])
        if hi - lo < 20:
            return None
        normalized = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0, 1)
        mask = (normalized > .55).astype(np.uint8)
        count = float(mask.mean())
        if not .015 < count < .65:
            return None
        # Keep stroke proportions and small alignment shifts without using brightness.
        return cv2.resize(mask, (96, 64), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def similarity(a, b):
        best = 0.
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                shifted = np.roll(b, (dy, dx), axis=(0, 1))
                union = np.count_nonzero(a | shifted)
                score = np.count_nonzero(a & shifted) / max(1, union)
                best = max(best, score)
        return best

    def _brief_missing_digit(self, sig, decoded, seconds):
        """Ignore brief missing digits, without emitting the previous value.

        A multiplexed 99 display can contain only the rightmost 9 in a frame.
        Require both a shorter numeric suffix and a strict foreground subset.
        The protection expires after two seconds of no complete observation,
        so a sustained new reading is still examined by the SOC state machine.
        """
        if decoded is None:
            return False
        if self.pending:
            anchor, _, _, last_sec, _ = self.pending[-1]
            references = [item[1] for item in self.pending]
        else:
            anchor, last_sec, references = self.value, self.last_numeric_seconds, self.references
        if (anchor is None or last_sec is None or seconds - last_sec > 2.
                or len(str(decoded)) >= len(str(anchor))
                or not str(anchor).endswith(str(decoded))):
            return False
        pixels = np.count_nonzero(sig)
        kernel = np.ones((3, 3), np.uint8)
        return any(pixels < .8 * np.count_nonzero(ref) and
                   np.count_nonzero(sig & cv2.dilate(ref, kernel)) / max(1, pixels) >= .97
                   for ref in references)

    def update(self, image, decoded, frame, seconds, index):
        """Return (value, FIRST full frame, seconds, index) after stable evidence."""
        if index == self.last_index or (self.last_seconds is not None and seconds <= self.last_seconds):
            return None
        gap = self.last_seconds is not None and seconds - self.last_seconds > max(2., self.interval * 3)
        self.last_index, self.last_seconds = index, seconds
        if gap:
            self.pending.clear()
        if self.pending and seconds - self.pending[-1][3] > max(2., self.interval * 3):
            self.pending.clear()
        sig = self.signature(image)
        if sig is None:
            self.status = '画面不完整，保持原值'
            return None
        if self._brief_missing_digit(sig, decoded, seconds):
            self.status = '短暂缺位画面，等待完整数字'
            return None
        score = max((self.similarity(sig, ref) for ref in self.references), default=0.)
        same = score >= .94
        if same and decoded is not None and decoded != self.value:
            self.pending.clear()
            self.status = '画面对比与数字结果冲突，待复核'
            return None
        if (same and decoded is None and self.last_numeric_seconds is not None
                and seconds - self.last_numeric_seconds <= 2.):
            self.pending.clear()
            self.status = '与已确认参考一致，短暂 OCR 漏读'
            return self.value, frame, seconds, index
        if score >= .7 and decoded is not None and decoded == self.value:
            self.last_numeric_seconds = seconds
            self.pending.clear()
            self.references.append(sig)
            self.status = '参考画面与数字识别一致'
            return decoded, frame, seconds, index
        # OCR failures and disagreements cannot be turned into invented values.
        if decoded is None:
            self.status = '画面待识别，不能按顺序推算'
            return None
        if self.pending and (self.pending[-1][0] != decoded or
                             max(self.similarity(sig, item[1]) for item in self.pending) < .7):
            self.pending.clear()
        self.pending.append((decoded, sig, frame.copy(), seconds, index))
        required = 5 if self.value is None else 3
        self.status = f'稳定画面确认 {len(self.pending)}/{required}，候选 {decoded}%'
        if len(self.pending) < required:
            return None
        first = self.pending[0]
        self.value = decoded
        self.last_numeric_seconds = seconds
        self.references.clear()
        self.references.extend(item[1] for item in self.pending)
        self.pending.clear()
        self.status = f'已建立 {decoded}% 临时参考'
        return decoded, first[2], first[3], first[4]


def suggested_range(records, target, mode, source, duration):
    """Bracket a missing value using only records from this video and unit."""
    import os
    direction = 1 if mode == 'charge' else -1
    before, after = [], []
    for soc, record in records.items():
        if os.path.normcase(os.path.abspath(record.get('source_video') or '')) != os.path.normcase(os.path.abspath(source)):
            continue
        seconds = record.get('video_sec')
        if not isinstance(seconds, (int, float)) or not 0 <= seconds <= duration:
            continue
        delta = (int(soc) - target) * direction
        if delta < 0:
            before.append(seconds)
        elif delta > 0:
            after.append(seconds)
    start, end = max(before, default=0.), min(after, default=duration)
    return (start, end) if start < end else (0., duration)
