# 必须在最顶部导入 onnxruntime，避免 Windows 下 C++ 运行时 DLL 冲突
import onnxruntime
import cv2
import time as pytime
import traceback
from datetime import time
from PyQt5.QtCore import QThread, pyqtSignal
from core.ocr_engine import OCREngine
from core.excel_filler import get_target_soc_list


class VideoAnalysisWorker(QThread):
    """
    视频后台分析工作线程 (支持 1# 和 2# 双样机独立/同步识别提取，支持各选区方向旋转)
    """
    # 信号定义:
    # progress_signal: cur_sec, total_sec, pct, soc1_str, cap1_str, soc2_str, cap2_str
    progress_signal = pyqtSignal(float, float, int, str, str, str, str)
    # point_detected_signal: unit_id (1 或 2), record (dict)
    point_detected_signal = pyqtSignal(int, dict)
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
                 mode='discharge', sample_interval_sec=1.0, debounce_count=2,
                 cap_unit='Ah', digit_type=0,
                 roi_soc=None, roi_time=None, roi_cap=None,
                 **kwargs):
        super().__init__()
        self.video_path = video_path
        self.digit_type = digit_type

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

        self.mode = mode  # 'discharge' 或 'charge'
        self.sample_interval = max(0.2, float(sample_interval_sec))
        self.debounce_count = max(1, int(debounce_count))
        self.cap_unit = cap_unit  # 'Ah' 或 'mAh'
        self.is_running = True
        self.target_socs = set(get_target_soc_list(mode))

        # OCR 引擎在 run() 线程内实例化，避免跨线程调用 ONNX 运行时
        self.ocr_engine = None

    def stop(self):
        self.is_running = False

    @staticmethod
    def crop(frame, roi, rotation_angle=0):
        """
        截取选区并根据旋转角度将图像回正为正向文字
        """
        if frame is None or roi is None:
            return None
        x, y, w, h = roi
        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 1, x))
        y1 = max(0, min(fh - 1, y))
        x2 = max(x1 + 1, min(fw, x + w))
        y2 = max(y1 + 1, min(fh, y + h))
        cropped = frame[y1:y2, x1:x2]

        if rotation_angle and rotation_angle % 360 != 0:
            ang = rotation_angle % 360
            if ang == 90:
                # 顺时针侧放 -> 逆时针旋转 90 度回正
                return cv2.rotate(cropped, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif ang == 180:
                # 倒立摆放 -> 旋转 180 度回正
                return cv2.rotate(cropped, cv2.ROTATE_180)
            elif ang == 270:
                # 逆时针侧放 -> 顺时针旋转 90 度回正
                return cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

        return cropped

    def _is_valid_transition(self, current_val, confirmed_val):
        """
        单调性校验：判断当前识别电量相对于已确认电量是否合理
        """
        if current_val is None or not (0 <= current_val <= 100):
            return False
        if confirmed_val is None:
            if self.mode == 'discharge':
                return current_val >= 80 or (0 <= current_val <= 100)
            else:
                return current_val <= 20 or (0 <= current_val <= 100)
        else:
            if self.mode == 'discharge':
                # 放电模式：电量单调递减，单次允许跳降 0 ~ 5%
                return 0 <= (confirmed_val - current_val) <= 5
            else:
                # 充电模式：电量单调递增，单次允许跳增 0 ~ 5%
                return 0 <= (current_val - confirmed_val) <= 5

    def run(self):
        try:
            # 在工作线程内部初始化 OCR 引擎
            self.ocr_engine = OCREngine()

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
            has_unit_1 = bool(self.roi_soc_1 and self.roi_time_1 and self.roi_cap_1)
            has_unit_2 = bool(self.roi_soc_2 and self.roi_time_2 and self.roi_cap_2)

            if not has_unit_1 and not has_unit_2:
                self.error_signal.emit("未检测到有效的样机选区框，请至少为 1# 或 2# 样机绘制选区！")
                cap.release()
                return

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

            while self.is_running and cur_sec <= total_duration:
                frame_idx = int(cur_sec * fps)
                if frame_idx >= total_frames:
                    break

                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frames_to_skip = frame_idx - current_frame

                # 如果需要跳过的帧数在合理范围内（如小于 300 帧即 10 秒），使用 grab() 逐帧快进以保证精确性
                # 否则使用 set() 进行大跨度跳转（虽然可能会跳到最近的关键帧，但速度更快）
                if 0 < frames_to_skip < 300:
                    for _ in range(frames_to_skip):
                        cap.grab()
                    ret, frame = cap.retrieve()
                else:
                    if frames_to_skip != 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()

                if not ret or frame is None:
                    cur_sec += self.sample_interval
                    continue

                # ==========================================
                # 处理 1# 样机 (包含方向回正)
                # ==========================================
                if has_unit_1:
                    crop_soc_1 = self.crop(frame, self.roi_soc_1, self.rot_soc_1)
                    soc_val_1, _ = self.ocr_engine.extract_soc(crop_soc_1, digit_type=self.digit_type)

                    if self._is_valid_transition(soc_val_1, confirmed_soc_1):
                        if soc_val_1 == candidate_soc_1:
                            candidate_count_1 += 1
                        else:
                            candidate_soc_1 = soc_val_1
                            candidate_count_1 = 1

                        if candidate_count_1 >= self.debounce_count and confirmed_soc_1 != candidate_soc_1:
                            confirmed_soc_1 = candidate_soc_1

                            # 识别 1# 时间 (带旋转回正)
                            crop_t_1 = self.crop(frame, self.roi_time_1, self.rot_time_1)
                            t1_str, t1_obj, _ = self.ocr_engine.extract_time(crop_t_1)
                            if not t1_str or not t1_obj:
                                m, s = divmod(int(cur_sec), 60)
                                h, m = divmod(m, 60)
                                t1_str = f"{h:02d}:{m:02d}:{s:02d}"
                                t1_obj = time(h % 24, m, s)
                            last_valid_time_str_1 = t1_str

                            # 识别 1# 容量 (带旋转回正)
                            crop_c_1 = self.crop(frame, self.roi_cap_1, self.rot_cap_1)
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
                                'video_sec': cur_sec,
                                'is_target': (confirmed_soc_1 in self.target_socs)
                            }
                            extracted_data_1[confirmed_soc_1] = rec_1
                            self.point_detected_signal.emit(1, rec_1)

                # ==========================================
                # 处理 2# 样机 (包含方向回正)
                # ==========================================
                if has_unit_2:
                    crop_soc_2 = self.crop(frame, self.roi_soc_2, self.rot_soc_2)
                    soc_val_2, _ = self.ocr_engine.extract_soc(crop_soc_2, digit_type=self.digit_type)

                    if self._is_valid_transition(soc_val_2, confirmed_soc_2):
                        if soc_val_2 == candidate_soc_2:
                            candidate_count_2 += 1
                        else:
                            candidate_soc_2 = soc_val_2
                            candidate_count_2 = 1

                        if candidate_count_2 >= self.debounce_count and confirmed_soc_2 != candidate_soc_2:
                            confirmed_soc_2 = candidate_soc_2

                            # 识别 2# 时间 (带旋转回正)
                            crop_t_2 = self.crop(frame, self.roi_time_2, self.rot_time_2)
                            t2_str, t2_obj, _ = self.ocr_engine.extract_time(crop_t_2)
                            if not t2_str or not t2_obj:
                                m, s = divmod(int(cur_sec), 60)
                                h, m = divmod(m, 60)
                                t2_str = f"{h:02d}:{m:02d}:{s:02d}"
                                t2_obj = time(h % 24, m, s)
                            last_valid_time_str_2 = t2_str

                            # 识别 2# 容量 (带旋转回正)
                            crop_c_2 = self.crop(frame, self.roi_cap_2, self.rot_cap_2)
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
                                'video_sec': cur_sec,
                                'is_target': (confirmed_soc_2 in self.target_socs)
                            }
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

                cur_sec += self.sample_interval

            cap.release()
            self.finished_signal.emit(extracted_data_1, extracted_data_2)

        except Exception as e:
            err_details = f"视频分析过程中发生异常: {str(e)}\n\n{traceback.format_exc()}"
            self.error_signal.emit(err_details)
