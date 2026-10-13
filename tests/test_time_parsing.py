import onnxruntime  # Windows DLL ordering
import unittest
from datetime import time
from types import SimpleNamespace

from core.ocr_engine import OCREngine


class TimeParsingTests(unittest.TestCase):
    def extract_boxes(self, boxes):
        engine = OCREngine.__new__(OCREngine)
        engine.preprocess_soc_crop = lambda image: image
        results = [
            [[[x, y], [x + width, y], [x + width, y + height], [x, y + height]], text, .95]
            for x, y, width, height, text in boxes
        ]
        engine.ocr = lambda image: (list(reversed(results)), None)
        return engine.extract_time(SimpleNamespace(size=1))

    def extract(self, parts):
        engine = OCREngine.__new__(OCREngine)
        engine.preprocess_soc_crop = lambda image: image
        results = [
            [[[i*50, 0], [i*50+40, 0], [i*50+40, 20], [i*50, 20]], text, .95]
            for i, text in enumerate(parts)
        ]
        # Reverse detection order to exercise left-to-right reconstruction.
        engine.ocr = lambda image: (list(reversed(results)), None)
        return engine.extract_time(SimpleNamespace(size=1))

    def test_single_digit_seconds_and_minutes(self):
        for raw, expected in [
            ('2:17:3', '02:17:03'), ('02:17:9', '02:17:09'),
            ('2:7:3', '02:07:03'), ('0:0:0', '00:00:00'),
            ('2:17:10', '02:17:10'), ('23:59:59', '23:59:59'),
        ]:
            with self.subTest(raw=raw):
                result = self.extract([raw])
                self.assertEqual(result[0], expected)
                self.assertEqual(result[1], time.fromisoformat(expected))

    def test_original_supported_formats_and_punctuation(self):
        for raw, expected in [
            ('00:02:36', '00:02:36'), ('0:02:36', '00:02:36'),
            ('02:36', '00:02:36'), ('2:3', '00:02:03'),
            (' 2 ： 17 ： 3 ', '02:17:03'), ('2;17；3', '02:17:03'),
        ]:
            with self.subTest(raw=raw):
                self.assertEqual(self.extract([raw])[0], expected)

    def test_split_time_is_not_truncated_to_minutes_seconds(self):
        self.assertEqual(self.extract(['2:17', ':3'])[0], '02:17:03')
        self.assertEqual(self.extract(['2', ':', '17', ':', '03'])[0], '02:17:03')
        self.assertEqual(self.extract(['02:', '36'])[0], '00:02:36')

    def test_invalid_time_never_falls_back_to_substring(self):
        for raw in ['2:17:', ':17:3', '2:17:345', '123:17:03', '24:17:03',
                    '2:60:03', '2:17:60', '2:17:x', '2:17:3:4', 'label2:17', '2:17extra']:
            with self.subTest(raw=raw):
                self.assertEqual(self.extract([raw]), (None, None, 0.0))
        self.assertEqual(self.extract([]), (None, None, 0.0))

    def test_power_z_two_rows_choose_upper_meter_time(self):
        result = self.extract_boxes([
            (0, 0, 40, 12, '累计时间'),
            (10, 20, 40, 12, '0:6:25'),
            (10, 38, 40, 12, '0:6:24'),
        ])
        self.assertEqual(result[0], '00:06:25')

    def test_time_column_can_include_row_label_and_distant_capacity(self):
        result = self.extract_boxes([
            (0, 20, 25, 12, '全部'),
            (38, 20, 40, 12, '0:6:25'),
            (95, 20, 40, 12, '0.3211'),
            (0, 38, 25, 12, '窗口'),
            (38, 38, 40, 12, '0:6:24'),
            (95, 38, 40, 12, '0.3202'),
        ])
        self.assertEqual(result[0], '00:06:25')

    def test_adjacent_or_malformed_time_fragments_are_rejected(self):
        self.assertEqual(self.extract_boxes([
            (0, 0, 40, 20, '2:17'), (45, 0, 10, 20, '3')
        ])[0], None)
        self.assertEqual(self.extract_boxes([
            (0, 0, 40, 20, '2:17:'), (45, 0, 20, 20, '3:4')
        ])[0], None)


if __name__ == '__main__':
    unittest.main()
