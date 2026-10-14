"""Deterministic parser checks with synthetic OCR boxes, not live recognition."""
import unittest
from types import SimpleNamespace

from core.ocr_engine import OCREngine


class CapacityParsingTests(unittest.TestCase):
    def extract(self, text, target_unit='Ah'):
        engine = OCREngine.__new__(OCREngine)
        engine.preprocess_soc_crop = lambda image: image
        box = [[[0, 0], [100, 0], [100, 20], [0, 20]], text, .99]
        engine.ocr = lambda image: ([box], None)
        return engine.extract_capacity(SimpleNamespace(size=1), target_unit=target_unit)

    def test_supported_capacity_formats_normalize_to_amp_hours(self):
        fixtures = (
            ('0.1303 Ah', .1303),
            ('130.3 mAh', .1303),
            ('1.3043', 1.3043),
            ('1304.3', 1.3043),
        )
        for text, expected in fixtures:
            with self.subTest(text=text):
                value, raw, confidence = self.extract(text)
                self.assertEqual(value, expected)
                self.assertEqual(raw, text)
                self.assertEqual(confidence, .99)

    def test_raw_capacity_mode_preserves_milliamp_hour_value(self):
        value, raw, confidence = self.extract('130.3 mAh', target_unit='raw')
        self.assertEqual(value, 130.3)
        self.assertEqual(raw, '130.3 mAh')
        self.assertEqual(confidence, .99)


if __name__ == '__main__':
    unittest.main()
