"""Compare exact real-video crops with a saved pre-optimization matcher.

Alternates old/new order; measures cold matching separately from exact cache
hits. Also compares decoded pixels and timestamps on the original video.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--baseline-dir', type=Path, required=True)
parser.add_argument('--library', required=True)
parser.add_argument('--video', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--threads', type=int, default=2)
args = parser.parse_args()
if os.name == 'nt':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
import cv2
import numpy as np
from core.segment_dataset import SegmentMatcher, read_image, read_frame_at
cv2.setNumThreads(1)
spec = importlib.util.spec_from_file_location('core._baseline_segment', args.baseline_dir/'segment_dataset.py')
old_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old_module)
seconds = (15,70,130,185,500,1000,2000,4000,8000)
images = [read_image(BASE/'tests/fixtures/dual_segment'/f'unit2-{s}.png') for s in seconds]
times = {'before':[], 'after':[], 'exact_cache':[]}
values = {}
with ThreadPoolExecutor(max_workers=args.threads) as pool:
    old = old_module.SegmentMatcher(args.library,executor=pool,workers=args.threads)
    new = SegmentMatcher(args.library,executor=pool,workers=args.threads)
    for repeat in range(3):
        order = [('before',old),('after',new)]
        if repeat % 2:
            order.reverse()
        for name,matcher in order:
            if name == 'after':
                new._direct_cache.clear();new._alignment_cache.clear()
            start = time.perf_counter()
            result = []
            for image in images:
                result.append((matcher.extract_soc(image),matcher.last_reason))
            times[name].append(time.perf_counter()-start)
            values[name] = result
        assert values['before'] == values['after'], values
        start = time.perf_counter()
        cached = [(new.extract_soc(im),new.last_reason) for im in images]
        times['exact_cache'].append(time.perf_counter()-start)
        assert cached == values['before']
        print('MATCH_ROUND',repeat,times,flush=True)

decode_times = {'before':[], 'after':[]}
requests = (0.,.1,.2,4.,5.5,63.9,63.9333,64.,65.5,70.,124.5,125.,182.4333,184.5,200.,70.,70.)
for repeat in range(2):
    results = {}
    order = [('before',old_module.read_frame_at),('after',read_frame_at)]
    if repeat:
        order.reverse()
    for name,reader in order:
        cap = cv2.VideoCapture(args.video)
        outputs = []
        start = time.perf_counter()
        try:
            for requested in requests:
                ok,frame = reader(cap,requested)
                assert ok
                outputs.append((cap.get(cv2.CAP_PROP_POS_FRAMES),cap.get(cv2.CAP_PROP_POS_MSEC),frame))
        finally:
            cap.release()
        decode_times[name].append(time.perf_counter()-start)
        results[name] = outputs
    for a,b in zip(results['before'],results['after']):
        assert a[:2] == b[:2] and np.array_equal(a[2],b[2]), (a[:2],b[:2])

report = dict(threads=args.threads,template_count=len(new.templates),
              matching_seconds=times,matching_medians={k:statistics.median(v) for k,v in times.items()},
              decoder_seconds=decode_times,decoded_frames_equal=True,
              matching_results_equal=True,cache_stats=dict(new.cache_stats),values=values['after'])
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('MATCH_AND_DECODE_IDENTICAL',flush=True)
