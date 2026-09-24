import re
import cv2
import numpy as np
from datetime import time
from rapidocr_onnxruntime import RapidOCR

class OCREngine:
    def __init__(self):
        # 初始化 RapidOCR 引擎
        self.ocr = RapidOCR()

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
        常见格式: 00:02:36 或 0:02:36 或 02:36
        返回: (time_str, datetime.time对象, confidence)
        """
        if crop_img is None or crop_img.size == 0:
            return None, None, 0.0

        processed = self.preprocess_soc_crop(crop_img)
        res, _ = self.ocr(processed)
        if not res:
            return None, None, 0.0

        # 按左到右排序
        res.sort(key=lambda x: min(pt[0] for pt in x[0]))

        for item in res:
            text = item[1].strip()
            score = float(item[2])
            normalized = text.replace('：', ':').replace(';', ':')
            # 匹配 hh:mm:ss
            m3 = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', normalized)
            if m3:
                h, m, s = int(m3.group(1)), int(m3.group(2)), int(m3.group(3))
                if 0 <= h < 24 and 0 <= m < 60 and 0 <= s < 60:
                    time_str = f"{h:02d}:{m:02d}:{s:02d}"
                    return time_str, time(h, m, s), score

            # 匹配 mm:ss
            m2 = re.search(r'(\d{1,2}):(\d{2})', normalized)
            if m2:
                m, s = int(m2.group(1)), int(m2.group(2))
                if 0 <= m < 60 and 0 <= s < 60:
                    time_str = f"00:{m:02d}:{s:02d}"
                    return time_str, time(0, m, s), score

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

        res.sort(key=lambda x: min(pt[0] for pt in x[0]))
        full_text = " ".join([item[1] for item in res])
        avg_score = sum([float(item[2]) for item in res]) / max(len(res), 1)

        m = re.search(r'(\d+(?:\.\d+)?)', full_text)
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
