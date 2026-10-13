"""Temporal checks for already-matched segment-display readings."""
from collections import deque
from dataclasses import dataclass
from statistics import median


@dataclass
class SOCCandidate:
    value: int
    first_sec: float
    gap_sec: float
    required_gap_sec: float
    payload: object
    count: int = 1


class SegmentSOCTracker:
    """Keep suspicious matches out of the trusted SOC and recorded results.

    Normal forward changes up to five percent use the configured debounce.
    Larger changes need an unreadable gap and stronger, time-separated evidence.
    Gap eligibility is frozen at first sight; repeating a wrong match must never
    manufacture that gap. A reverse change is retained for review, not promoted.
    """

    def __init__(self, mode, debounce_count=2, sample_interval=1.0):
        self.mode = mode
        self.debounce_count = max(1, int(debounce_count))
        self.sample_interval = max(.2, float(sample_interval))
        self._loss_timeout_sec = max(2.0, self.sample_interval * 2)
        self.confirmed = None
        self.last_seen_sec = None
        self.last_change_sec = None
        self.candidate = None
        self.misses = 0
        self.pending = {}
        self.new_pending = None
        self.last_rejection = None
        self._seconds_per_percent = deque(maxlen=8)
        self._last_sample_sec = None
        self._last_readable_sec = None
        self._missing_since_sec = None
        self._anchors = []

    @property
    def needs_probe(self):
        return self.candidate is not None or self.misses > 0

    def reset_candidate(self, seconds=None):
        self.candidate = None
        if seconds is not None:
            self._note_missing(float(seconds))
        else:
            self.misses = 0

    def _note_missing(self, seconds):
        self.misses += 1
        if self._missing_since_sec is None:
            self._missing_since_sec = (self._last_readable_sec
                                       if self._last_readable_sec is not None else seconds)
        # Use time rather than number of frames: fast sampling must not discard
        # the first good frame during a brief blink, while a long loss can restart.
        if seconds - self._missing_since_sec > self._loss_timeout_sec:
            self.candidate = None

    def _difference(self, value):
        return (value - self.confirmed if self.mode == 'charge'
                else self.confirmed - value)

    def _required_gap(self, diff):
        # Permit up to four times the recently observed rate. No battery-size
        # assumption is needed; without observations use a modest sampling floor.
        unit_seconds = max(self.sample_interval,
                           median(self._seconds_per_percent) * .25
                           if self._seconds_per_percent else 1.0)
        return max(10.0, self.sample_interval * 3, diff * unit_seconds)

    def update(self, value, seconds, capture_payload):
        """Return a newly confirmed candidate, retaining its FIRST frame payload."""
        self.new_pending = None
        self.last_rejection = None
        seconds = float(seconds)
        if self._last_sample_sec is not None and seconds <= self._last_sample_sec:
            return None
        self._last_sample_sec = seconds
        if value is None or not 0 <= value <= 100:
            self._note_missing(seconds)
            return None

        gap = (seconds - self._missing_since_sec
               if self._missing_since_sec is not None else 0)
        if gap > self._loss_timeout_sec:
            self.candidate = None
        self._last_readable_sec = seconds
        self._missing_since_sec = None
        self.misses = 0
        if value == self.confirmed:
            self.last_seen_sec = seconds
            self.candidate = None
            return None

        if self.candidate is None or self.candidate.value != value:
            diff = self._difference(value) if self.confirmed is not None else 0
            self.candidate = SOCCandidate(value, seconds, gap,
                                         self._required_gap(diff), capture_payload())
        else:
            self.candidate.count += 1
        candidate = self.candidate
        diff = self._difference(value) if self.confirmed is not None else 0
        reason = None
        required_count = self.debounce_count
        if self.confirmed is not None and diff < 0:
            reason = '电量变化与充放电方向相反'
        elif self.confirmed is not None and diff > 5:
            required_count = max(4, self.debounce_count + 2)
            if candidate.gap_sec < candidate.required_gap_sec:
                reason = '短时间内电量大幅跳变'
            elif (candidate.count < required_count or
                  seconds - candidate.first_sec < max(1.0, self.sample_interval * 2)):
                reason = '漏读后恢复尚待更多画面确认'

        self.last_rejection = reason
        if reason or candidate.count < required_count:
            if (reason and candidate.count >= self.debounce_count) or value in self.pending:
                entry = {'soc': value, 'video_sec': candidate.first_sec,
                         'reason': reason or '电量变化尚待更多画面确认'}
                if self.pending.get(value) != entry:
                    self.pending[value] = entry
                    self.new_pending = entry
            return None

        return self._accept(candidate, seconds)

    def _accept(self, candidate, seconds):
        value = candidate.value
        diff = self._difference(value) if self.confirmed is not None else 0
        if (self.confirmed is not None and 0 < diff <= 5 and
                self.last_change_sec is not None):
            duration = candidate.first_sec - self.last_change_sec
            if duration > 0:
                self._seconds_per_percent.append(duration / diff)
        if value != self.confirmed:
            self._anchors.append((value, candidate.first_sec))
            self.last_change_sec = candidate.first_sec
        self.confirmed = value
        self.last_seen_sec = seconds
        self.pending.pop(value, None)
        self.candidate = None
        return candidate

    def correction_candidate(self):
        """Expose a sustained reverse episode for independent image verification."""
        candidate = self.candidate
        if (candidate is None or self.confirmed is None or
                self._difference(candidate.value) >= 0 or
                candidate.count < max(4, self.debounce_count + 2)):
            return None
        if len(self._anchors) > 1:
            previous = self._anchors[-2][0]
            diff = (candidate.value - previous if self.mode == 'charge'
                    else previous - candidate.value)
            if not 0 <= diff <= 5:
                return None
        return candidate

    def correct_last_anchor(self, expected_value, verified_sec=None, verified_payload=None):
        """Caller must first prove a contradictory reading in the OLD video frames.

        Never relax normal direction checks. Only the most recent anchor can be
        replaced; the current candidate keeps its original timestamp and payload.
        """
        candidate = self.correction_candidate()
        if candidate is None or self.confirmed != expected_value or not self._anchors:
            return None
        if verified_sec is not None and verified_payload is not None:
            # Only the independently re-decoded original frame may move time
            # back. Neighbouring votes alone never invent an earlier transition.
            if verified_sec < self._anchors[-1][1] - .05 or verified_sec > candidate.first_sec:
                return None
            candidate = SOCCandidate(candidate.value, verified_sec, candidate.gap_sec,
                                     candidate.required_gap_sec, verified_payload, candidate.count)
        self._anchors.pop()
        if self._anchors:
            self.confirmed, self.last_change_sec = self._anchors[-1]
        else:
            self.confirmed = self.last_change_sec = None
        self._seconds_per_percent.clear()
        # Restore rates from surviving anchors, excluding the disproved event.
        for (old_value, old_sec), (new_value, new_sec) in zip(self._anchors, self._anchors[1:]):
            diff = new_value - old_value if self.mode == 'charge' else old_value - new_value
            if 0 < diff <= 5 and new_sec > old_sec:
                self._seconds_per_percent.append((new_sec - old_sec) / diff)
        self.pending.pop(expected_value, None)
        self.last_rejection = None
        return self._accept(candidate, self._last_sample_sec)
