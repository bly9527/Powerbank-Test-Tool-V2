"""Compare GPU/CPU meter OCR on exact source frames without modifying user data."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.runtime_bootstrap import prepare_runtime
import onnxruntime as ort
import cv2
from core.gpu_backend import directml_devices, preferred_device
from core.ocr_engine import OCREngine
from core.meter_reader import read_meter_fields
from core.segment_dataset import read_frame_at
from core.roi_geometry import crop_roi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    device = preferred_device(directml_devices())
    if not device:
        raise RuntimeError('No GPU found')
    print('RUNTIME', ort.__version__, 'DEVICE', device, flush=True)
    cpu = OCREngine(cpu_threads=2)
    gpu = OCREngine(cpu_threads=2, gpu=True, gpu_device=device['id'], backend_callback=lambda x: print(x, flush=True))
    if not gpu.ocr.active:
        raise RuntimeError(gpu.backend_status)
    cap = cv2.VideoCapture(args.video)
    samples = []
    try:
        # The six validated 99/98/97% records, each from its own unit's frame.
        for unit, seconds in ((1, 64.), (1, 124.5), (1, 182.4333),
                              (2, 65.5), (2, 125.), (2, 184.5)):
            ok, frame = read_frame_at(cap, seconds)
            actual = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
            if not ok or frame is None:
                raise RuntimeError(f'Missing frame {seconds}')
            rois = ((166, 320, 61, 70), (228, 320, 50, 70)) if unit == 1 else (
                    (162, 768, 55, 51), (218, 768, 49, 51))
            samples.append((unit, seconds, actual, *(crop_roi(frame, r, 0) for r in rois)))
    finally:
        cap.release()
    timings = {'cpu': [], 'gpu': []}
    outputs = {}
    for round_index in range(args.rounds):
        for name, engine in ([('cpu', cpu), ('gpu', gpu)] if round_index % 2 == 0 else
                             [('gpu', gpu), ('cpu', cpu)]):
            started = time.perf_counter()
            rows = []
            for unit, seconds, actual, time_crop, cap_crop in samples:
                fields = read_meter_fields(time_crop, cap_crop, engine)
                rows.append(dict(unit=unit, requested=seconds, actual=actual,
                                 time=fields['time_str'], capacity=fields['capacity']))
            elapsed = time.perf_counter() - started
            timings[name].append(elapsed)
            outputs[name] = rows
            print(name, round_index, round(elapsed, 3), rows, flush=True)
    result = {'runtime': ort.__version__, 'device': device, 'timings': timings,
              'median_seconds': {k: statistics.median(v) for k, v in timings.items()},
              'outputs': outputs, 'equal': outputs['cpu'] == outputs['gpu'],
              'gpu_active': gpu.ocr.active, 'backend': gpu.backend_status}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    assert result['equal'] and result['gpu_active'], result


if __name__ == '__main__':
    main()
