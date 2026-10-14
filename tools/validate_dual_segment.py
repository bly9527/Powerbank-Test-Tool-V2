"""Bounded real-video acceptance for PB313 12V discharge on 2026-10-13.

ROIs are specific to that recording. Outputs use isolated app data; no user
archive or sample annotation is changed. Run in both CPU modes for comparison.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import importlib.util
import hashlib

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--video', required=True)
parser.add_argument('--library', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--end', type=float, default=200.)
parser.add_argument('--high-speed', action='store_true')
parser.add_argument('--gpu', action='store_true', help='Use DirectML for OCR')
parser.add_argument('--gpu-device', type=int, default=0, help='DXGI adapter index')
parser.add_argument('--threads', type=int, default=2)
parser.add_argument('--baseline-dir', type=Path)
parser.add_argument('--compare', type=Path, help='Earlier results.json; require identical records and evidence')
args = parser.parse_args()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
os.environ['LOCALAPPDATA'] = str(out/'appdata')
from core.runtime_bootstrap import prepare_runtime
import onnxruntime  # Must precede Qt imports on Windows.
if os.name == 'nt':
    import ctypes
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
if args.baseline_dir:
    for name in ('segment_dataset', 'keyframe_timing', 'video_worker'):
        spec = importlib.util.spec_from_file_location('core.'+name, args.baseline_dir/(name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
from PyQt5.QtCore import QCoreApplication
from core.video_worker import VideoAnalysisWorker
from core.segment_dataset import SEGMENT_MODE
import core.video_worker as worker_module
if args.threads < 1:
    parser.error('--threads must be positive')
worker_module.high_speed_threads = lambda: args.threads

app = QCoreApplication([])
worker = VideoAnalysisWorker(
    args.video,
    roi_soc_1=(970, 222, 66, 37), roi_time_1=(166, 320, 61, 70), roi_cap_1=(228, 320, 50, 70),
    roi_soc_2=(1290, 253, 75, 48), roi_time_2=(162, 768, 55, 51), roi_cap_2=(218, 768, 49, 51),
    active_units=[1, 2], modes={1: 'discharge', 2: 'discharge'}, digit_type=SEGMENT_MODE,
    segment_paths={1: args.library, 2: args.library}, sample_interval_sec=1.,
    cpu_threads=args.threads, high_speed=args.high_speed, gpu=args.gpu, gpu_device=args.gpu_device,
    timing_refine={1: True, 2: True}, end_sec=args.end)
records = {1: {}, 2: {}}
errors = []


def detected(unit, record):
    records[unit][record['soc']] = {k: v for k, v in record.items() if k != 'time_obj'}
    print(unit, record['soc'], record['video_sec'], record['time_str'], record['capacity'], flush=True)


worker.point_detected_signal.connect(detected)
worker.error_signal.connect(errors.append)
backend_messages = []
if hasattr(worker, 'backend_signal'):
    worker.backend_signal.connect(backend_messages.append)
progress_bucket = [-1]


def progress(seconds, *_):
    bucket = int(seconds//20)
    if bucket != progress_bucket[0]:
        progress_bucket[0] = bucket
        print('PROGRESS', round(seconds, 3), flush=True)


worker.progress_signal.connect(progress)
started, cpu_started = time.perf_counter(), time.process_time()
worker.run()
result = dict(video=args.video, library=args.library, end_sec=args.end,
              gpu_requested=args.gpu, backend_messages=backend_messages,
              high_speed=args.high_speed, threads=args.threads, records=records, errors=errors,
              elapsed_seconds=time.perf_counter()-started, cpu_seconds=time.process_time()-cpu_started,
              performance=getattr(worker, 'performance_stats', {}))
(out/'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
assert not errors, errors
if args.gpu:
    assert worker.ocr_engine.ocr.active, backend_messages
for unit in (1, 2):
    for soc in (99, 98, 97):
        record = records[unit][soc]
        assert record['time_str'] and record['capacity'] is not None, (unit, record)
        assert Path(record['snapshot_path']).is_file()
    assert records[unit][99]['video_sec'] < records[unit][98]['video_sec'] < records[unit][97]['video_sec']
if args.compare:
    baseline = json.loads(args.compare.read_text(encoding='utf-8'))
    for unit in (1,2):
        old_records = baseline['records'][str(unit)]
        assert set(map(str,records[unit])) == set(old_records), (unit,'SOC mismatch')
        for soc, record in records[unit].items():
            old_record = old_records[str(soc)]
            assert {k:v for k,v in record.items() if k != 'snapshot_path'} == {
                k:v for k,v in old_record.items() if k != 'snapshot_path'}, (unit,soc,'record mismatch')
            if 'snapshot_path' in record:
                assert hashlib.sha256(Path(record['snapshot_path']).read_bytes()).digest() == hashlib.sha256(
                    Path(old_record['snapshot_path']).read_bytes()).digest(), (unit,soc,'image mismatch')
    comparison = dict(records_equal=True,evidence_png_equal=True,
                      before_seconds=baseline['elapsed_seconds'],after_seconds=result['elapsed_seconds'],
                      speedup=baseline['elapsed_seconds']/result['elapsed_seconds'],
                      saved_fraction=1-result['elapsed_seconds']/baseline['elapsed_seconds'])
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    print('RECORDS_AND_EVIDENCE_IDENTICAL',comparison,flush=True)
print('DUAL_SEGMENT_OK', flush=True)
