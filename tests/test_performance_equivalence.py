"""Performance work must preserve exact evidence, ties and cache isolation."""
import onnxruntime  # Windows DLL order.
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import get_ident, Lock
from unittest.mock import patch

import cv2
import numpy as np

from core.segment_dataset import SegmentMatcher, read_frame_at, read_image
from core.keyframe_timing import TimingCropCache, refine_keyframe

FIXTURE = Path(__file__).parent/'fixtures'/'dual_segment'


def test_alignment_runs_in_pool_and_preserves_cold_and_cached_results():
    serial = SegmentMatcher(FIXTURE/'library', cache_size=0)
    threads, lock = set(), Lock()
    main_thread = get_ident()
    with ThreadPoolExecutor(max_workers=2) as pool:
        parallel = SegmentMatcher(FIXTURE/'library', executor=pool, workers=2)
        original = parallel._align_templates

        def observed(*args):
            with lock:
                threads.add(get_ident())
            return original(*args)

        with patch.object(parallel, '_align_templates', side_effect=observed):
            for seconds in (70, 130, 500, 2000):
                image = read_image(FIXTURE/f'unit2-{seconds}.png')
                expected = serial.extract_soc(image), serial.last_reason
                for _ in range(2):
                    assert (parallel.extract_soc(image), parallel.last_reason) == expected
    assert len(threads) == 2 and main_thread not in threads
    assert parallel.cache_stats['direct_hits'] >= 4
    assert parallel.cache_stats['alignment_hits'] >= 4


def test_cache_is_exact_bounded_and_does_not_return_mutable_score_dict():
    matcher = SegmentMatcher([], cache_size=2)
    calls = []

    def calculate(candidates):
        calls.append(candidates)
        return {99: (.8, .9, .3)}

    a = np.zeros((72,104), np.uint8)
    first = matcher._cached_scores(matcher._direct_cache, {():a}, calculate, 'test')
    first.clear()
    assert matcher._cached_scores(matcher._direct_cache, {():a.copy()}, calculate, 'test')
    assert len(calls) == 1
    a[35,50] = 1  # Even one differing pixel must recompute.
    matcher._cached_scores(matcher._direct_cache, {():a}, calculate, 'test')
    matcher.threshold += .01
    matcher._cached_scores(matcher._direct_cache, {():a}, calculate, 'test')
    assert len(calls) == 3 and len(matcher._direct_cache) == 2
    other = SegmentMatcher([])
    assert not other._direct_cache and not other._alignment_cache


def video(path, count=60, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), fps, (160,80))
    assert writer.isOpened()
    frames = []
    for index in range(count):
        frame = np.zeros((80,160,3),np.uint8)
        cv2.putText(frame, '100' if index < 15 else '99', (5,60), 0, 1.6, (255,255,255), 3)
        frames.append(frame)
        writer.write(frame)
    writer.release()
    return frames


def legacy_read(cap, seconds):
    position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
    if not (position > 0 and timestamp < seconds and seconds-timestamp <= 5):
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.,seconds-1.)*1000)
    while True:
        ok, frame = cap.read()
        if not ok or cap.get(cv2.CAP_PROP_POS_MSEC)/1000 >= seconds-1e-6:
            return ok, frame


def test_grab_retrieve_retains_exact_frames_timestamps_seeks_and_eof(tmp_path):
    path = tmp_path/'decode.avi'
    video(path)
    a,b = cv2.VideoCapture(str(path)),cv2.VideoCapture(str(path))
    try:
        for sec in (0.,.1,.11,1.5,1.5,.2,4.75,5.9,9.):
            old_ok, old = legacy_read(a,sec)
            new_ok, new = read_frame_at(b,sec)
            assert old_ok == new_ok
            if old_ok:
                assert np.array_equal(old,new)
                assert a.get(cv2.CAP_PROP_POS_FRAMES) == b.get(cv2.CAP_PROP_POS_FRAMES)
                assert a.get(cv2.CAP_PROP_POS_MSEC) == b.get(cv2.CAP_PROP_POS_MSEC)
    finally:
        a.release(); b.release()


def test_coarse_cache_shares_only_identical_requests_and_keeps_unit_crops(tmp_path):
    path = tmp_path/'cache.avi'
    video(path)
    regions = [((0,0,80,80),0),((80,0,80,80),0)]
    cache = TimingCropCache(regions,max_entries=2,max_bytes=80000)
    cache.begin(str(path))
    cap = cv2.VideoCapture(str(path))
    try:
        for i, (roi, angle) in enumerate(regions):
            ok, crop, seconds = cache.read_crop(cap,1.,roi,angle)
            assert ok and seconds == 1. and crop.shape == (80,80,3)
        assert cache.stats == {'coarse_decodes':1, 'coarse_hits':1}
        cache.read_crop(cap,1.0001,*regions[0])
        assert cache.stats['coarse_decodes'] == 2  # No rounded timestamp keys.
        for sec in (2.,3.):
            cache.read_crop(cap,sec,*regions[0])
        assert len(cache.frames) <= 2 and cache.bytes_used <= 80000
        cache.begin(str(tmp_path/'different.avi'))
        assert not cache.frames and cache.bytes_used == 0
    finally:
        cap.release()


def test_cached_refinement_keeps_first_frame_and_full_evidence(tmp_path):
    path = tmp_path/'timing.avi'
    frames = video(path)
    roi = (0,0,160,80)
    cache = TimingCropCache([(roi,0)])
    arguments = (str(path),roi,0,99,frames[50],5.,50)
    expected = refine_keyframe(*arguments,competing=[(100,frames[0])])
    for _ in range(2):
        actual = refine_keyframe(*arguments,competing=[(100,frames[0])],crop_cache=cache)
        assert actual['frame_index'] == expected['frame_index'] == 15
        assert np.array_equal(actual.pop('frame'),expected['frame'])
        assert actual == {k:v for k,v in expected.items() if k != 'frame'}
    assert cache.stats['coarse_hits'] > 0


def test_coarse_cache_invalidates_replaced_source_and_obeys_byte_limit(tmp_path):
    path = tmp_path/'bytes.avi'
    video(path)
    cache = TimingCropCache(max_bytes=1)
    cache.begin(str(path))
    cap = cv2.VideoCapture(str(path))
    try:
        assert cache.read_crop(cap,1.,(0,0,160,80),0)[0]
        assert not cache.frames and cache.bytes_used == 0
    finally:
        cap.release()
    # Changing content identity at the same path also invalidates old crops.
    path.write_bytes(b'replaced')
    cache.frames[0.] = (0.,{})
    cache.begin(str(path))
    assert not cache.frames
