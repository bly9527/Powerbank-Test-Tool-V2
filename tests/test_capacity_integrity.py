"""Reject partial capacity readings observed in PB313 video crops."""
import onnxruntime  # Initialize the Windows runtime before OpenCV/Qt.
import unittest
from types import SimpleNamespace

from core.ocr_engine import OCREngine


class CapacityIntegrityTests(unittest.TestCase):
    def extract(self, parts):
        engine = OCREngine.__new__(OCREngine)
        engine.preprocess_soc_crop = lambda image: image
        boxes = [
            [[[i * 40, 0], [i * 40 + 35, 0], [i * 40 + 35, 20], [i * 40, 20]], text, .99]
            for i, text in enumerate(parts)
        ]
        engine.ocr = lambda image: (list(reversed(boxes)), None)
        return engine.extract_capacity(SimpleNamespace(size=1))

    def test_real_fragmented_readings_do_not_become_first_number(self):
        for parts in [['0', '5', '109'], ['3', '.0521', '1'], ['8', '677'],
                      ['6', '205', '59'], ['5.', '1.9', '8', '9'], ['101', '38', '8']]:
            with self.subTest(parts=parts):
                value, raw, confidence = self.extract(parts)
                self.assertIsNone(value)
                self.assertEqual(raw, ' '.join(parts))
                self.assertEqual(confidence, 0.0)

    def test_number_and_separate_unit_remain_supported(self):
        for parts, expected in [(['0.5038', 'Ah'], .5038),
                                (['510.9', 'mAh'], .5109),
                                (['容量', '0.5038', 'Ah'], .5038),
                                (['0'], 0.0)]:
            with self.subTest(parts=parts):
                self.assertEqual(self.extract(parts)[0], expected)

    def test_invalid_fields_cannot_yield_numeric_substrings(self):
        for text in ['-0.5 Ah', '1e3', '0.5.038', '0:09:53', '0.5038 2.5562', '8.']:
            with self.subTest(text=text):
                self.assertIsNone(self.extract([text])[0])
