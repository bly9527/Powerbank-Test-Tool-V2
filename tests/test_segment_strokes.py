"""Regression cases for thin digits and high-correlation wrong strokes."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.segment_dataset import SegmentMatcher, feature, write_image, write_json


def display(text, thickness=2):
    segments = {'9': 'abfgcd', '8': 'abcdefg', '3': 'abgcd', '6': 'afgecd'}
    lines = {'a': ((3, 0), (15, 0)), 'b': ((18, 3), (18, 14)),
             'c': ((18, 18), (18, 29)), 'd': ((3, 32), (15, 32)),
             'e': ((0, 18), (0, 29)), 'f': ((0, 3), (0, 14)),
             'g': ((3, 16), (15, 16))}
    image = np.zeros((55, 70, 3), np.uint8)
    for i, digit in enumerate(text):
        for segment in segments[digit]:
            p, q = lines[segment]
            cv2.line(image, (p[0]+10+i*28, p[1]+10),
                     (q[0]+10+i*28, q[1]+10), (255, 255, 255), thickness)
    return image


class SegmentStrokeTests(unittest.TestCase):
    def test_thin_one_survives_larger_roi_and_brightness_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = np.full((50, 60, 3), 20, np.uint8)
            sample[15:25, 31:33] = 210
            sample[29:39, 31:33] = 210
            write_image(root/'one.png', sample)
            write_json(root/'labels.json', {'one.png': 1})
            matcher = SegmentMatcher(root)
            for size in (100, 150, 240):
                for background, light in ((20, 210), (45, 130), (0, 80)):
                    wide = np.full((size, size, 3), background, np.uint8)
                    wide[30:40, 51:53] = light
                    wide[44:54, 51:53] = light
                    # Bright area is allowed to be much less than 0.2%.
                    self.assertEqual(matcher.extract_soc(wide)[0], 1)

    def test_blank_dim_noise_and_isolated_glints_are_not_digits(self):
        rng = np.random.default_rng(42)
        for image in (np.full((60, 80), 120, np.uint8),
                      rng.integers(15, 30, (60, 80), dtype=np.uint8)):
            self.assertIsNone(feature(image))
        glints = np.full((60, 80), 20, np.uint8)
        glints[::10, ::10] = 255
        self.assertIsNone(feature(glints))

    def test_missing_discriminating_stroke_cannot_pass_on_global_similarity(self):
        for actual, wrong, thickness in ((a, b, t) for a, b in (('98', '99'), ('96', '36'))
                                         for t in (1, 2, 3)):
            with self.subTest(actual=actual, thickness=thickness), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                candidate = display(actual, thickness)
                template = display(wrong, thickness)
                normalized = feature(candidate)
                padded = cv2.copyMakeBorder(normalized, 3, 3, 3, 3, cv2.BORDER_CONSTANT)
                correlation = cv2.minMaxLoc(cv2.matchTemplate(
                    padded, feature(template), cv2.TM_CCOEFF_NORMED))[1]
                self.assertGreater(correlation, .94)  # Previous .88 gate would accept.
                write_image(root/'wrong.png', template)
                write_json(root/'labels.json', {'wrong.png': int(wrong)})
                matcher = SegmentMatcher(root)
                self.assertIsNone(matcher.extract_soc(candidate)[0])
                self.assertIn('笔画证据不足', matcher.last_reason)
                write_image(root/'correct.png', candidate)
                write_json(root/'labels.json', {'wrong.png': int(wrong), 'correct.png': int(actual)})
                matcher = SegmentMatcher(root)
                self.assertEqual(matcher.extract_soc(candidate)[0], int(actual))
                self.assertEqual(matcher.extract_soc(template)[0], int(wrong))

    def test_incomplete_multi_digit_annotation_is_not_a_template_for_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            one = np.zeros((50, 60, 3), np.uint8)
            one[15:25, 31:33] = 210
            one[29:39, 31:33] = 210
            write_image(root/'incomplete.png', one)
            write_image(root/'one.png', one)
            write_json(root/'labels.json', {'incomplete.png': 19, 'one.png': 1})
            before = (root/'labels.json').read_bytes()
            matcher = SegmentMatcher(root)
            self.assertEqual(matcher.extract_soc(one)[0], 1)
            self.assertEqual([s['label'] for s in matcher.skipped_templates], [19])
            self.assertEqual((root/'labels.json').read_bytes(), before)

    def test_ineligible_sample_cannot_displace_valid_sample_of_same_label(self):
        with tempfile.TemporaryDirectory() as directory:
            matcher = SegmentMatcher(directory)
            image = np.zeros((72, 104), np.uint8)
            matcher.templates = [(98, image), (98, image)]
            matcher.template_ink = [1., 1.]
            matcher.template_regions = [(), ()]
            matcher.template_grids = [np.full((9, 13), .46), np.full((9, 13), .44)]
            prepared = {(): (image, 1., [[np.zeros((9, 13))]])}
            with patch('core.segment_dataset.cv2.matchTemplate', side_effect=[
                    np.array([[.99]], np.float32), np.array([[.96]], np.float32)]):
                evidence = matcher._score_templates(prepared, 0, 2)[98]
            self.assertAlmostEqual(evidence[1], .96, places=5)
            self.assertTrue(matcher._admissible(evidence))

    def test_two_narrow_ones_are_valid_even_below_width_ratio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.zeros((70, 60, 3), np.uint8)
            for column in (20, 38):
                image[10:30, column:column+2] = 255
                image[34:54, column:column+2] = 255
            write_image(root/'eleven.png', image)
            write_json(root/'labels.json', {'eleven.png': 11})
            matcher = SegmentMatcher(root)
            self.assertEqual(matcher.extract_soc(image)[0], 11)
            self.assertEqual(matcher.skipped_templates, [])

    def test_invalid_and_conflicting_annotations_still_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = display('98')
            write_image(root/'one.png', image)
            write_image(root/'two.png', image)
            for other in ('invalid', 99):
                write_json(root/'labels.json', {'one.png': 98, 'two.png': other})
                self.assertIsNone(SegmentMatcher(root).extract_soc(image)[0])


if __name__ == '__main__':
    unittest.main()
