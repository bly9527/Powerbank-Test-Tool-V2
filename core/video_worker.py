# 必须在最顶部导入 onnxruntime，避免 Windows 下 C++ 运行时 DLL 冲突
import onnxruntime
import cv2
import time as pytime
import traceback
import os
from pathlib import Path
from uuid import uuid4
from core.record_evidence import save_evidence
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from core.performance import high_speed_threads
from datetime import time
from PyQt5.QtCore import QThread, pyqtSignal
from core.ocr_engine import OCREngine
from core.excel_filler import get_target_soc_list
from core.segment_dataset import (SEGMENT_MODE, SegmentMatcher, read_frame_at,
                                 mask_ocr_regions, validate_ignore_regions)
from core.roi_geometry import crop_roi
from core.soc_tracker import SegmentSOCTracker


class VideoAnalysisWorker(QThread):
    """
    视频后台分析工作线程 (支持 1# 和 2# 双样机独立/同步识别提取，支持各选区方向旋转)
    """
    # 信号定义:
    # progress_signal: cur_sec, total_sec, pct, soc1_str, cap1_str, soc2_str, cap2_str
    progress_signal = pyqtSignal(float, float, int, str, str, str, str)
    # point_detected_signal: unit_id (1 或 2), record (dict)
    point_detected_signal = pyqtSignal(int, dict)
    review_pending_signal = pyqtSignal(int, dict)
    # finished_signal: all_data_1, all_data_2
    finished_signal = pyqtSignal(dict, dict)
    # error_signal: 错误提示信息
    error_signal = pyqtSignal(str)

    def __init__(self, video_path,
                 roi_soc_1=None, roi_time_1=None, roi_cap_1=None,
                 roi_soc_2=None, roi_time_2=None, roi_cap_2=None,
                 rot_soc_1=0, rot_time_1=0, rot_cap_1=0,
                 rot_soc_2=0, rot_time_2=0, rot_cap_2=0,
                 rotations=None,
                 mode='discharge', modes=None, active_units=None,
                 sample_interval_sec=1.0, debounce_count=2,
                 cap_unit='Ah', digit_type=0,
                 segment_paths=None,
                 cpu_threads=None,
                 high_speed=False,
                 roi_soc=None, roi_time=None, roi_cap=None,
                 lcd_ignore_regions=None,
                 **kwargs):
        super().__init__()
        self.video_path = video_path
        self.digit_type = digit_type
        self.segment_paths = segment_paths or {}
        self.lcd_ignore_regions = {unit: validate_ignore_regions((lcd_ignore_regions or {}).get(unit, ()))
                                   for unit in (1, 2)}
        self.segment_diagnostics = {}

        # 兼容单样机或旧参数调用
        self.roi_soc_1 = roi_soc_1 if roi_soc_1 is not None else roi_soc
        self.roi_time_1 = roi_time_1 if roi_time_1 is not None else roi_time
        self.roi_cap_1 = roi_cap_1 if roi_cap_1 is not None else roi_cap

        self.roi_soc_2 = roi_soc_2
        self.roi_time_2 = roi_time_2
        self.roi_cap_2 = roi_cap_2

        # 选区旋转角度设置 (0, 90, 180, 270)
        if rotations:
            self.rot_soc_1 = rotations.get('soc_1', rot_soc_1)
            self.rot_time_1 = rotations.get('time_1', rot_time_1)
            self.rot_cap_1 = rotations.get('cap_1', rot_cap_1)
            self.rot_soc_2 = rotations.get('soc_2', rot_soc_2)
            self.rot_time_2 = rotations.get('time_2', rot_time_2)
            self.rot_cap_2 = rotations.get('cap_2', rot_cap_2)
        else:
            self.rot_soc_1 = rot_soc_1
            self.rot_time_1 = rot_time_1
            self.rot_cap_1 = rot_cap_1
            self.rot_soc_2 = rot_soc_2
            self.rot_time_2 = rot_time_2
            self.rot_cap_2 = rot_cap_2

        self.mode = mode  # legacy single-unit callers
        self.modes = modes or {1: mode, 2: mode}
        self.active_units = set(active_units or (1, 2))
        self.cpu_threads = cpu_threads
        self.high_speed = bool(high_speed)
        self.sample_interval = max(0.2, float(sample_interval_sec))
        self.debounce_count = max(1, int(debounce_count))
        self.cap_unit = cap_unit  # 'Ah' 或 'mAh'
        self.is_running = True
        self.target_socs = {unit: set(get_target_soc_list(self.modes[unit])) for unit in (1, 2)}

        # OCR 引擎在 run() 线程内实例化，避免跨线程调用 ONNX 运行时
        self.ocr_engine = None
        self.evidence_directory = (Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) /
                                   "PowerbankTestTool" / "record_evidence" / uuid4().hex)

    def stop(self):
        self.is_running = False

    @staticmethod
    def crop(frame, roi, rotation_angle=0):
        """
        截取选区并根据旋转角度将图像回正为正向文字
        """
        return crop_roi(frame, roi, rotation_angle)

    def _is_valid_transition(self, current_val, confirmed_val, unit=1):
        """
        LCD 单帧变化校验。数码管模式由独立的时间状态机处理。
        """
        if current_val is None or not (0 <= current_val <= 100):
            return False
        mode = self.modes[unit]
        if mode == 'discharge' and confirmed_val == 0:
            return True
        if confirmed_val is None:
            if mode == 'discharge':
                return current_val >= 80 or (0 <= current_val <= 100)
            else:
                return current_val <= 20 or (0 <= current_val <= 100)
        else:
            if mode == 'discharge':
                # 放电模式：电量单调递减，单次允许跳降 0 ~ 5%
                return 0 <= (confirmed_val - current_val) <= 5
            else:
                # 充电模式：电量单调递增，单次允许跳增 0 ~ 5%
                return 0 <= (current_val - confirmed_val) <= 5

    def _process_segment_soc(self, unit, frame, seconds, matcher, tracker, records, last_capacity):
        value, _ = matcher.extract_soc(self.crop(frame, getattr(self, f'roi_soc_{unit}'),
                                               getattr(self, f'rot_soc_{unit}')))

        def capture():
            return (self.crop(frame, getattr(self, f'roi_time_{unit}'),
                              getattr(self, f'rot_time_{unit}')).copy(),
                    self.crop(frame, getattr(self, f'roi_cap_{unit}'),
                              getattr(self, f'rot_cap_{unit}')).copy(),
                    frame.copy(), getattr(self, "_decoded_frame_index", None))

        accepted = tracker.update(value, seconds, capture)
        diagnostics = self.segment_diagnostics[unit]
        if value is None:
            reason = getattr(matcher, 'last_reason', '未识别').split('：')[0]
            diagnostics['rejected'][reason] += 1
        elif tracker.last_rejection:
            diagnostics['rejected'][tracker.last_rejection] += 1
        if tracker.new_pending:
            self.review_pending_signal.emit(unit, dict(tracker.new_pending))
        if accepted is None or accepted.value in records:
            return value, None

        time_image, capacity_image, evidence_frame, evidence_index = accepted.payload
        time_str, time_obj, _ = self.ocr_engine.extract_time(time_image)
        if not time_str or time_obj is None:
            time_str, time_obj = '--', None
        capacity, _, _ = self.ocr_engine.extract_capacity(capacity_image, target_unit=self.cap_unit)
        if capacity is None:
            capacity = last_capacity
        record = {'soc': accepted.value, 'time_str': time_str, 'time_obj': time_obj,
                  'capacity': capacity, 'video_sec': accepted.first_sec,
                  'is_target': accepted.value in self.target_socs[unit]}
        if record["is_target"]:
            record.update(save_evidence(evidence_frame, self.video_path, accepted.first_sec,
                                        evidence_index, self.evidence_directory))
        records[accepted.value] = record
        self.point_detected_signal.emit(unit, record)
        return value, record

    def run(self):
        pool = None
        previous_cv_threads = cv2.getNumThreads()
        try:
            threads = high_speed_threads() if self.high_speed else self.cpu_threads
            if self.high_speed and self.digit_type == SEGMENT_MODE:
                # Parallelize independent templates, not frames or tracker updates.
                cv2.setNumThreads(1)
                pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix='segment-match')
            elif threads:
                cv2.setNumThreads(threads)
            # 在工作线程内部初始化 OCR 引擎
            self.ocr_engine = OCREngine(cpu_threads=threads) if threads else OCREngine()

            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_signal.emit(f"无法打开视频文件: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            total_duration = total_frames / fps

            # 判断哪几台样机激活了选区
            has_unit_1 = 1 in self.active_units and bool(self.roi_soc_1 and self.roi_time_1 and self.roi_cap_1)
            has_unit_2 = 2 in self.active_units and bool(self.roi_soc_2 and self.roi_time_2 and self.roi_cap_2)

            if not has_unit_1 and not has_unit_2:
                self.error_signal.emit("未检测到有效的样机选区框，请至少为 1# 或 2# 样机绘制选区！")
                cap.release()
                return

            segment_matchers = {}
            segment_trackers = {}
            if self.digit_type == SEGMENT_MODE:
                for unit, active in ((1, has_unit_1), (2, has_unit_2)):
                    if active:
                        path = self.segment_paths.get(unit)
                        if not path:
                            raise ValueError(f'{unit}# 尚未选择数码管样本库')
                        matcher = (SegmentMatcher(path, executor=pool, workers=threads)
                                   if pool is not None else SegmentMatcher(path))
                        if not matcher.labels:
                            raise ValueError(f'{unit}# 样本库没有已确认的数字，请先采集并标注')
                        segment_matchers[unit] = matcher
                        tracker = SegmentSOCTracker(self.modes[unit], self.debounce_count, self.sample_interval)
                        segment_trackers[unit] = tracker
                        self.segment_diagnostics[unit] = {'labels': sorted(matcher.labels),
                                                          'rejected': Counter(), 'pending': tracker.pending}

            # 1# 样机状态追踪
            confirmed_soc_1 = None
            candidate_soc_1 = None
            candidate_count_1 = 0
            last_valid_time_str_1 = "00:00:00"
            last_valid_cap_1 = 0.0
            extracted_data_1 = {}

            # 2# 样机状态追踪
            confirmed_soc_2 = None
            candidate_soc_2 = None
            candidate_count_2 = 0
            last_valid_time_str_2 = "00:00:00"
            last_valid_cap_2 = 0.0
            extracted_data_2 = {}

            cur_sec = 0.0
            last_decoded_sec = -1.0
            lcd_candidates = {}
            lcd_last_seen = {1: -float("inf"), 2: -float("inf")}

            while self.is_running and cur_sec <= total_duration:
                frame_idx = int(cur_sec * fps)
                if frame_idx >= total_frames:
                    break

                ret, frame = read_frame_at(cap, cur_sec)

                if not ret or frame is None:
                    if self.digit_type == SEGMENT_MODE:
                        for tracker in segment_trackers.values():
                            tracker.reset_candidate(cur_sec)
                    cur_sec += self.sample_interval
                    continue

                sample_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                self._decoded_frame_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                if sample_sec <= last_decoded_sec:
                    cur_sec += self.sample_interval
                    continue
                last_decoded_sec = sample_sec

                # ==========================================
                # 处理 1# 样机 (包含方向回正)
                # ==========================================
                if has_unit_1:
                    if self.digit_type == SEGMENT_MODE:
                        soc_val_1, record = self._process_segment_soc(
                            1, frame, sample_sec, segment_matchers[1], segment_trackers[1],
                            extracted_data_1, last_valid_cap_1)
                        confirmed_soc_1 = segment_trackers[1].confirmed
                        if record:
                            last_valid_cap_1 = record['capacity']
                            last_valid_time_str_1 = record['time_str']
                    else:
                        crop_soc_1 = self.crop(frame, self.roi_soc_1, self.rot_soc_1)
                        crop_soc_1 = mask_ocr_regions(crop_soc_1, self.lcd_ignore_regions[1])
                        soc_val_1, _ = self.ocr_engine.extract_soc(crop_soc_1, digit_type=self.digit_type)

                    if self.digit_type != SEGMENT_MODE and sample_sec - lcd_last_seen[1] > max(2., self.sample_interval * 2):
                        candidate_soc_1, candidate_count_1 = None, 0
                    if self.digit_type != SEGMENT_MODE and self._is_valid_transition(soc_val_1, confirmed_soc_1, 1):
                        lcd_last_seen[1] = sample_sec
                        if soc_val_1 == candidate_soc_1:
                            candidate_count_1 += 1
                        else:
                            candidate_soc_1 = soc_val_1
                            candidate_count_1 = 1
                            lcd_candidates[1] = (frame.copy(), sample_sec, self._decoded_frame_index)

                        if candidate_count_1 >= self.debounce_count and confirmed_soc_1 != candidate_soc_1:
                            confirmed_soc_1 = candidate_soc_1

                        # 恢复识别后只补充缺失节点，不覆盖已捕获的时间和容量。
                        if candidate_count_1 >= self.debounce_count and confirmed_soc_1 not in extracted_data_1:
                            # 识别 1# 时间 (带旋转回正)
                            point_frame, point_sec_1, point_index = lcd_candidates[1]
                            crop_t_1 = self.crop(point_frame, self.roi_time_1, self.rot_time_1)
                            t1_str, t1_obj, _ = self.ocr_engine.extract_time(crop_t_1)
                            if not t1_str or not t1_obj:
                                m, s = divmod(int(point_sec_1), 60)
                                h, m = divmod(m, 60)
                                t1_str = f"{h:02d}:{m:02d}:{s:02d}"
                                t1_obj = time(h % 24, m, s)
                            last_valid_time_str_1 = t1_str

                            # 识别 1# 容量 (带旋转回正)
                            crop_c_1 = self.crop(point_frame, self.roi_cap_1, self.rot_cap_1)
                            c1_val, _, _ = self.ocr_engine.extract_capacity(crop_c_1, target_unit=self.cap_unit)
                            if c1_val is not None:
                                last_valid_cap_1 = c1_val
                            else:
                                c1_val = last_valid_cap_1

                            rec_1 = {
                                'soc': confirmed_soc_1,
                                'time_str': t1_str,
                                'time_obj': t1_obj,
                                'capacity': c1_val,
                                'video_sec': point_sec_1,
                                'is_target': (confirmed_soc_1 in self.target_socs[1])
                            }
                            if rec_1["is_target"]:
                                rec_1.update(save_evidence(point_frame, self.video_path, point_sec_1,
                                                            point_index, self.evidence_directory))
                            extracted_data_1[confirmed_soc_1] = rec_1
                            self.point_detected_signal.emit(1, rec_1)

                # ==========================================
                # 处理 2# 样机 (包含方向回正)
                # ==========================================
                if has_unit_2:
                    if self.digit_type == SEGMENT_MODE:
                        soc_val_2, record = self._process_segment_soc(
                            2, frame, sample_sec, segment_matchers[2], segment_trackers[2],
                            extracted_data_2, last_valid_cap_2)
                        confirmed_soc_2 = segment_trackers[2].confirmed
                        if record:
                            last_valid_cap_2 = record['capacity']
                            last_valid_time_str_2 = record['time_str']
                    else:
                        crop_soc_2 = self.crop(frame, self.roi_soc_2, self.rot_soc_2)
                        crop_soc_2 = mask_ocr_regions(crop_soc_2, self.lcd_ignore_regions[2])
                        soc_val_2, _ = self.ocr_engine.extract_soc(crop_soc_2, digit_type=self.digit_type)

                    if self.digit_type != SEGMENT_MODE and sample_sec - lcd_last_seen[2] > max(2., self.sample_interval * 2):
                        candidate_soc_2, candidate_count_2 = None, 0
                    if self.digit_type != SEGMENT_MODE and self._is_valid_transition(soc_val_2, confirmed_soc_2, 2):
                        lcd_last_seen[2] = sample_sec
                        if soc_val_2 == candidate_soc_2:
                            candidate_count_2 += 1
                        else:
                            candidate_soc_2 = soc_val_2
                            candidate_count_2 = 1
                            lcd_candidates[2] = (frame.copy(), sample_sec, self._decoded_frame_index)

                        if candidate_count_2 >= self.debounce_count and confirmed_soc_2 != candidate_soc_2:
                            confirmed_soc_2 = candidate_soc_2

                        # 恢复识别后只补充缺失节点，不覆盖已捕获的时间和容量。
                        if candidate_count_2 >= self.debounce_count and confirmed_soc_2 not in extracted_data_2:
                            # 识别 2# 时间 (带旋转回正)
                            point_frame, point_sec_2, point_index = lcd_candidates[2]
                            crop_t_2 = self.crop(point_frame, self.roi_time_2, self.rot_time_2)
                            t2_str, t2_obj, _ = self.ocr_engine.extract_time(crop_t_2)
                            if not t2_str or not t2_obj:
                                m, s = divmod(int(point_sec_2), 60)
                                h, m = divmod(m, 60)
                                t2_str = f"{h:02d}:{m:02d}:{s:02d}"
                                t2_obj = time(h % 24, m, s)
                            last_valid_time_str_2 = t2_str

                            # 识别 2# 容量 (带旋转回正)
                            crop_c_2 = self.crop(point_frame, self.roi_cap_2, self.rot_cap_2)
                            c2_val, _, _ = self.ocr_engine.extract_capacity(crop_c_2, target_unit=self.cap_unit)
                            if c2_val is not None:
                                last_valid_cap_2 = c2_val
                            else:
                                c2_val = last_valid_cap_2

                            rec_2 = {
                                'soc': confirmed_soc_2,
                                'time_str': t2_str,
                                'time_obj': t2_obj,
                                'capacity': c2_val,
                                'video_sec': point_sec_2,
                                'is_target': (confirmed_soc_2 in self.target_socs[2])
                            }
                            if rec_2["is_target"]:
                                rec_2.update(save_evidence(point_frame, self.video_path, point_sec_2,
                                                            point_index, self.evidence_directory))
                            extracted_data_2[confirmed_soc_2] = rec_2
                            self.point_detected_signal.emit(2, rec_2)

                # ==========================================
                # 发送综合进度
                # ==========================================
                progress_pct = int((cur_sec / max(total_duration, 1.0)) * 100)
                soc1_disp = f"{confirmed_soc_1}%" if confirmed_soc_1 is not None else ("--" if has_unit_1 else "未启用")
                cap1_disp = f"{last_valid_cap_1:.4f} Ah" if has_unit_1 else "未启用"
                soc2_disp = f"{confirmed_soc_2}%" if confirmed_soc_2 is not None else ("--" if has_unit_2 else "未启用")
                cap2_disp = f"{last_valid_cap_2:.4f} Ah" if has_unit_2 else "未启用"

                self.progress_signal.emit(
                    cur_sec, total_duration, progress_pct,
                    soc1_disp, cap1_disp, soc2_disp, cap2_disp
                )

                # A blinking digit can be dark at every whole-second sample.
                # Probe the other phase while a display is missing or a new
                # value is still awaiting confirmation.
                if self.digit_type == SEGMENT_MODE and (
                    (has_unit_1 and (soc_val_1 is None or segment_trackers[1].needs_probe))
                    or (has_unit_2 and (soc_val_2 is None or segment_trackers[2].needs_probe))
                ):
                    cur_sec += min(self.sample_interval / 2, 0.5)
                else:
                    cur_sec += self.sample_interval

            cap.release()
            self.finished_signal.emit(extracted_data_1, extracted_data_2)

        except Exception as e:
            err_details = f"视频分析过程中发生异常: {str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(err_details)
        finally:
            if 'cap' in locals():
                cap.release()
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
            cv2.setNumThreads(previous_cv_threads)
