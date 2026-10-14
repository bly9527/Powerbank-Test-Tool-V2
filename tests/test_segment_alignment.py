"""Real second-unit crops must match without relaxing missing-stroke gates."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from core.segment_dataset import SegmentMatcher, read_image

FIXTURES = Path(__file__).parent/'fixtures'/'dual_segment'


CASES = [(15, 100), (70, 99), (130, 98), (185, 97), (500, 92),
         (1000, 83), (2000, 72), (4000, 50), (8000, 6)]


@pytest.mark.parametrize('seconds,expected', CASES)
def test_real_second_unit_uses_geometry_not_lower_threshold(seconds, expected):
    matcher = SegmentMatcher(FIXTURES/'library')
    crop = read_image(FIXTURES/f'unit2-{seconds}.png')
    with patch.object(matcher, '_score_aligned', return_value={}):
        assert matcher.extract_soc(crop)[0] is None
    assert matcher.extract_soc(crop)[0] == expected
    assert '角度校正后' in matcher.last_reason
    assert matcher.threshold == .78
    assert matcher.MAX_STROKE_MISMATCH == .45


def test_real_second_unit_serial_and_high_speed_agree():
    serial = SegmentMatcher(FIXTURES/'library')
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = SegmentMatcher(FIXTURES/'library', executor=pool, workers=2)
        for sec, _ in CASES:
            crop = read_image(FIXTURES/f'unit2-{sec}.png')
            assert parallel.extract_soc(crop) == serial.extract_soc(crop)


def test_direct_conflict_and_invalid_are_not_overridden():
    matcher = SegmentMatcher(FIXTURES/'library')
    crop = read_image(FIXTURES/'unit2-70.png')
    for scores in ({99: (.95, 1., .1), 98: (.95, 1., .1)}, {'invalid': (.95, 1., .1)}):
        with patch.object(matcher, '_score_templates', return_value=scores), \
                patch.object(matcher, '_score_aligned') as aligned:
            assert matcher.extract_soc(crop)[0] is None
            aligned.assert_not_called()


def test_affine_bounds_reject_flips_large_moves_and_nonfinite():
    assert SegmentMatcher._safe_alignment(np.eye(2, 3, dtype=np.float32))
    for matrix in ([[-1, 0, 104], [0, 1, 0]], [[1, 0, 6], [0, 1, 0]],
                   [[1.16, 0, -8.32], [0, 1, 0]], [[1, .19, -6.84], [0, 1, 0]],
                   [[1, 0, float('nan')], [0, 1, 0]]):
        assert not SegmentMatcher._safe_alignment(np.array(matrix, dtype=np.float32))


def test_alignment_failure_remains_unrecognized():
    import cv2
    matcher = SegmentMatcher(FIXTURES/'library')
    with patch('core.segment_dataset.cv2.findTransformECC', side_effect=cv2.error('no convergence')):
        assert matcher.extract_soc(read_image(FIXTURES/'unit2-70.png'))[0] is None


def test_dim_strokes_are_recovered_from_the_original_pixels():
    matcher = SegmentMatcher(FIXTURES/'library')
    assert matcher.extract_soc(read_image(FIXTURES/'unit2-2000.png'))[0] == 72
    assert '恢复暗笔画' in matcher.last_reason
