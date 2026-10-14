"""Bounded candidate extraction and OCR of the exact selected snapshot."""
from pathlib import Path
import math

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from core.segment_dataset import read_frame_at, write_image, read_image
from core.roi_geometry import crop_roi
from core.ocr_engine import OCREngine
from core.meter_reader import read_meter_fields


def selected_fields(frame, rois, rotations, engine, cap_unit='Ah', single_line=False):
    crops = {key: crop_roi(frame, roi, rotations.get(key, 0), strict=True)
             for key, roi in rois.items()}
    return read_meter_fields(crops['time'], crops['cap'], engine, cap_unit, single_line)


class CaptureTask(QThread):
    candidate = pyqtSignal(dict)
    result = pyqtSignal(dict)
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, source, directory, rois, rotations, *, start=0., end=0., step=.5,
                 selected=None, cap_unit='Ah', single_line=False, target_index=None, parent=None):
        super().__init__(parent)
        self.source, self.directory = source, Path(directory)
        self.rois, self.rotations = dict(rois), dict(rotations)
        self.start_sec, self.end_sec, self.step = start, end, step
        self.selected, self.cap_unit = selected, cap_unit
        self.single_line = single_line
        self.target_index = target_index

    def run(self):
        cap = None
        try:
            if self.selected is not None:
                frame = read_image(self.selected['path'])
                fields = selected_fields(frame, self.rois, self.rotations,
                                         OCREngine(cpu_threads=2), self.cap_unit, self.single_line)
                if not self.isInterruptionRequested():
                    self.result.emit(dict(self.selected, **fields))
                return
            if not all(math.isfinite(x) for x in (self.start_sec, self.end_sec, self.step)) or self.step <= 0 or self.start_sec < 0 or self.end_sec < self.start_sec:
                raise ValueError('请设置有效的起止时间和采样间隔')
            count = int((self.end_sec - self.start_sec) / self.step + 1e-7) + 1
            if count > 240:
                raise ValueError('每批最多 240 张，请缩小时间范围或增大间隔，然后对目标附近加密采集。')
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                raise ValueError('无法打开视频')
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.target_index is not None:
                target = int(self.target_index)
                if not 0 <= target < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                    raise ValueError('已到视频边界')
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, target - 60))
                for _ in range(180):
                    if self.isInterruptionRequested():
                        return
                    ok, frame = cap.read()
                    index = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    if not ok or index > target:
                        break
                    if index == target:
                        path = self.directory / f'{index}.png'
                        write_image(path, frame)
                        self.candidate.emit({'path': str(path), 'frame_index': index,
                            'video_sec': cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.,
                            'thumbnail': crop_roi(frame, self.rois['soc'], self.rotations.get('soc',0), strict=True)})
                        return
                raise ValueError('无法精确定位相邻帧，请在该时刻附近重新采集')
            seen = set()
            for i in range(count):
                if self.isInterruptionRequested():
                    break
                ok, frame = read_frame_at(cap, self.start_sec + i * self.step)
                if not ok:
                    break
                index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.
                if seconds > self.end_sec + 1 / max(1, cap.get(cv2.CAP_PROP_FPS)):
                    break
                if index in seen:
                    continue
                seen.add(index)
                crop = crop_roi(frame, self.rois['soc'], self.rotations.get('soc', 0), strict=True)
                path = self.directory / f'{index}.png'
                write_image(path, frame)
                self.candidate.emit({'path': str(path), 'frame_index': index,
                                     'video_sec': seconds, 'thumbnail': crop.copy()})
                self.progress.emit(round((i + 1) * 100 / count))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
