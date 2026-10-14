from core.runtime_bootstrap import prepare_runtime
import onnxruntime  # Before OpenCV/Qt DLLs on Windows.
import re
import cv2
import numpy as np
from datetime import time
from rapidocr_onnxruntime import RapidOCR

class OCREngine:
    def __init__(self, cpu_threads=None, *, gpu=False, gpu_device=0, backend_callback=None):
        # 初始化 RapidOCR 引擎
        self.ocr = (RapidOCR(intra_op_num_threads=int(cpu_threads), inter_op_num_threads=1)
                    if cpu_threads else RapidOCR())
        self.backend_status = 'OCR：CPU'
        if gpu:
            from core.gpu_backend import AcceleratedOCR
            def changed(status):
                self.backend_status = status
                if backend_callback:
                    backend_callback(status)
            self.ocr = AcceleratedOCR(self.ocr, int(gpu_device), changed)
        elif backend_callback:
            backend_callback(self.backend_status)

    def preprocess_soc_crop(self, crop_img):
        """
        对数码管或屏幕截取的电量区域进行预处理
        """
        if crop_img is None or crop_img.size == 0:
            return crop_img
        h, w = crop_img.shape[:2]
        if h < 40 or w < 60:
            scale = max(2.0, 60.0 / max(h, 1))
            crop_img = cv2.resize(crop_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return crop_img

    def extract_soc(self, crop_img, digit_type=0):
        """
        从截取图像中提取电量百分比数字 (0 ~ 100)
        digit_type: 0(智能混合), 1(普通LCD数字), 2(断带数码管), 3(高亮发粘数码管)
        """
        if crop_img is None or crop_img.size == 0:
            return None, 0.0

        processed = self.preprocess_soc_crop(crop_img)
        res, _ = self.ocr(processed)

        if not res:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            inverted = cv2.bitwise_not(enhanced)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            filters = []
            if digit_type == 1:
                # 普通 LCD 数字
                filters = [enhanced, inverted]
            elif digit_type == 2:
                # 断带数码管：需要让白底黑字的黑色变粗（腐蚀）
                thicker_text = cv2.erode(inverted, kernel, iterations=1)
                filters = [thicker_text, inverted, enhanced]
            elif digit_type == 3:
                # 高亮粘连数码管：需要让白底黑字的黑色变细（膨胀）
                thinner_text = cv2.dilate(inverted, kernel, iterations=1)
                filters = [thinner_text, inverted, enhanced]
            else:
                # 智能混合：按序尝试所有方案
                thinner_text = cv2.dilate(inverted, kernel, iterations=1)
                thicker_text = cv2.erode(inverted, kernel, iterations=1)
                filters = [enhanced, inverted, thinner_text, thicker_text]

            for img_filter in filters:
                res, _ = self.ocr(img_filter)
                if res:
                    break

        if not res:
            return None, 0.0

        # 按文本框左侧 X 坐标从左到右排序
        res.sort(key=lambda x: min(pt[0] for pt in x[0]))

        # 优先策略 1: 将所有识别框从左到右拼接为完整数字（解决 99 被拆为 9 和 9 的问题）
        combined_digits = ""
        scores = []
        for item in res:
            text = item[1].strip()
            score = float(item[2])
            digits = re.sub(r'[^\d]', '', text)
            if digits:
                combined_digits += digits
                scores.append(score)

        if combined_digits:
            try:
                combined_val = int(combined_digits)
                if 0 <= combined_val <= 100:
                    avg_score = sum(scores) / max(len(scores), 1)
                    return combined_val, avg_score
            except ValueError:
                pass

        # 备选策略 2: 单个候选检测
        candidates = []
        for item in res:
            text = item[1].strip()
            score = float(item[2])
            clean_text = re.sub(r'[^\d]', '', text)
            if clean_text:
                try:
                    val = int(clean_text)
                    if 0 <= val <= 100:
                        candidates.append((val, score))
                except ValueError:
                    pass

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0], candidates[0][1]

        return None, 0.0

    def extract_time(self, crop_img):
        """
        从 Power-Z 截图中提取时间字符串与 datetime.time
        支持时、分、秒各一至两位，例如 2:17:3、00:02:36 或 2:3。
        返回: (time_str, datetime.time对象, confidence)
        """
        if crop_img is None or crop_img.size == 0:
            return None, None, 0.0

        processed = self.preprocess_soc_crop(crop_img)
        res, _ = self.ocr(processed)
        if not res:
            return None, None, 0.0

        def parse_items(items):
            # A time split into several OCR boxes must be reassembled within
            # its row. Never accept a valid-looking prefix of malformed text.
            ordered = sorted(items, key=lambda item: min(pt[0] for pt in item[0]))
            text = ''.join(item[1].strip() for item in ordered)
            normalized = re.sub(r'\s+', '', text.replace('：', ':').replace(';', ':').replace('；', ':'))
            m3 = re.fullmatch(r'([0-9]{1,2}):([0-9]{1,2}):([0-9]{1,2})', normalized)
            m2 = re.fullmatch(r'([0-9]{1,2}):([0-9]{1,2})', normalized)
            if m3:
                h, m, s = map(int, m3.groups())
            elif m2:
                h = 0
                m, s = map(int, m2.groups())
            else:
                return None, None, 0.0
            if not (0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60):
                return None, None, 0.0
            score = sum(float(item[2]) for item in items) / len(items)
            return f'{h:02d}:{m:02d}:{s:02d}', time(h, m, s), score

        def parse_line(items):
            whole = parse_items(items)
            if whole[0] is not None:
                return whole
            ordered = sorted(items, key=lambda item: min(pt[0] for pt in item[0]))
            candidates = []
            for start in range(len(ordered)):
                for end in range(start + 1, len(ordered) + 1):
                    part = ordered[start:end]
                    parsed = parse_items(part)
                    if parsed[0] is None:
                        continue
                    left = min(pt[0] for item in part for pt in item[0])
                    right = max(pt[0] for item in part for pt in item[0])
                    height = max(max(pt[1] for pt in item[0]) - min(pt[1] for pt in item[0]) for item in part)
                    others = ordered[:start] + ordered[end:]
                    if any(any(char in entry[1] for char in ':：;；') for entry in others):
                        continue
                    # A neighboring digit can be a missing piece of a split
                    # time. Capacity in the next table column is farther away.
                    if any(any(ch.isdigit() for ch in entry[1]) and
                           max(0, left - max(pt[0] for pt in entry[0]),
                               min(pt[0] for pt in entry[0]) - right) <= max(8, height * .6)
                           for entry in others):
                        continue
                    candidates.append((len(part), parsed))
            if candidates and len({result[0] for _, result in candidates}) == 1:
                return max(candidates, key=lambda item: item[0])[1]
            return None, None, 0.0

        whole = parse_line(res)
        if whole[0] is not None:
            return whole

        # Power-Z shows "全部" and "窗口" on separate rows. OCR may return
        # both times; concatenating them invalidates two correct readings.
        lines = []
        for item in sorted(res, key=lambda item: sum(pt[1] for pt in item[0]) / len(item[0])):
            center_y = sum(pt[1] for pt in item[0]) / len(item[0])
            height = max(pt[1] for pt in item[0]) - min(pt[1] for pt in item[0])
            if lines and abs(center_y - lines[-1][0]) <= max(5, min(height, lines[-1][2]) * .55):
                line_items = lines[-1][1] + [item]
                center = sum(sum(pt[1] for pt in entry[0]) / len(entry[0]) for entry in line_items) / len(line_items)
                lines[-1] = (center, line_items, max(lines[-1][2], height))
            else:
                lines.append((center_y, [item], height))
        for _, items, _ in lines:
            result = parse_line(items)
            if result[0] is not None:
                return result
        return None, None, 0.0

    def extract_capacity(self, crop_img, target_unit='Ah'):
        """
        从 Power-Z 截图中提取容量数值 (默认转换为 Ah)
        可能格式: '0.1303 Ah', '130.3 mAh', '1.3043', '1304.3'
        target_unit: 'Ah' (若检测到是 mAh 则自动除以 1000) 或 'raw'
        返回: (val_float, raw_str, confidence)
        """
        if crop_img is None or crop_img.size == 0:
            return None, "", 0.0

        processed = self.preprocess_soc_crop(crop_img)
        res, _ = self.ocr(processed)
        if not res:
            return None, "", 0.0

        # A Power-Z column can include its header plus total/window rows.
        # Keep the first numeric row, matching the top displayed test-time row.
        # Never concatenate duplicate rows or take the first numeric fragment
        # of a malformed single row.
        lines = []
        for item in sorted(res, key=lambda x: sum(pt[1] for pt in x[0]) / len(x[0])):
            center = sum(pt[1] for pt in item[0]) / len(item[0])
            height = max(pt[1] for pt in item[0]) - min(pt[1] for pt in item[0])
            if lines and abs(center - lines[-1][0]) <= max(5., min(height, lines[-1][2]) * .55):
                lines[-1][1].append(item)
            else:
                lines.append([center, [item], height])
        numeric_lines = [items for _, items, _ in lines if any(re.search(r'\d', item[1]) for item in items)]
        if len(lines) > 1 and numeric_lines:
            header_units = [re.search(r'(mAh|Ah)', item[1], re.IGNORECASE).group(1)
                            for _, items, _ in lines for item in items
                            if not re.search(r'\d', item[1]) and re.search(r'(mAh|Ah)', item[1], re.IGNORECASE)]
            res = list(numeric_lines[0])
            if len(set(unit.lower() for unit in header_units)) == 1 and not any('ah' in item[1].lower() for item in res):
                # Unit-only box stays on this row and cannot introduce a digit.
                rightmost = max(res, key=lambda item: max(pt[0] for pt in item[0]))
                box = [[pt[0] + 10000, pt[1]] for pt in rightmost[0]]
                res.append([box, header_units[0], 1.])
        res.sort(key=lambda x: min(pt[0] for pt in x[0]))
        full_text = " ".join([item[1] for item in res])
        avg_score = sum([float(item[2]) for item in res]) / max(len(res), 1)

        # A fragmented reading such as "0 5 109" is not a measured zero.
        # Require the complete field so a lost decimal or a neighbouring
        # table column cannot silently become the first numeric fragment.
        m = re.fullmatch(
            r'\s*(?:容量\s*[:：]?\s*)?(\d+(?:\.\d+)?)\s*(mAh|Ah)?\s*',
            full_text, re.IGNORECASE,
        )
        if not m:
            return None, full_text, 0.0

        raw_val = float(m.group(1))
        lower_text = full_text.lower()

        is_mah = 'mah' in lower_text
        is_ah = 'ah' in lower_text and not is_mah

        val = raw_val
        if target_unit == 'Ah':
            if is_mah or (raw_val > 100.0 and not is_ah):
                val = round(raw_val / 1000.0, 4)
            else:
                val = round(raw_val, 4)

        return val, full_text, avg_score
