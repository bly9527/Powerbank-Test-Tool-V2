import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import tempfile
from pathlib import Path
from collections import Counter
from datetime import time
from types import SimpleNamespace
from unittest.mock import patch, Mock

import onnxruntime
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem
from core.soc_tracker import SegmentSOCTracker
from core.video_worker import VideoAnalysisWorker


class CorrectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def worker(self):
        roi = (0, 0, 1, 1)
        worker = VideoAnalysisWorker('fake', roi_soc_1=roi, roi_time_1=roi, roi_cap_1=roi)
        worker.ocr_engine = SimpleNamespace(
            extract_time=lambda image: ('00:00:10', time(0, 0, 10), 1),
            extract_capacity=lambda image, **kw: (.5, '.5', 1))
        worker.segment_diagnostics = {1: {'rejected': Counter()}}
        return worker

    def test_worker_revokes_only_proven_error_then_reuses_percentage(self):
        for proof in (False, {'first_frame': None}, {'first_frame': (np.zeros((1, 1, 3), np.uint8), 2, 2.)}):
            with self.subTest(proof=bool(proof)):
                worker = self.worker()
                tracker = SegmentSOCTracker('charge')
                matcher = SimpleNamespace(extract_soc=lambda image: (98, .90), last_reason='ok')
                records = {97: {'soc': 97, 'video_sec': 0., 'capacity': .1},
                           99: {'soc': 99, 'video_sec': 2., 'capacity': .2}}
                for sec, value in enumerate([97, 97, 99, 99]):
                    tracker.update(value, sec, lambda: None)
                revoked, detected = [], []
                worker.point_revoked_signal.connect(lambda u, r: revoked.append(r))
                worker.point_detected_signal.connect(lambda u, r: detected.append(r))
                with patch.object(worker, '_verify_old_segment_record', return_value=proof) as verify, patch('core.video_worker.save_evidence', return_value={}):
                    for sec in range(10, 14):
                        worker._process_segment_soc(1, np.zeros((1, 1, 3), np.uint8), sec,
                                                    matcher, tracker, records, .2)
                    self.assertEqual(verify.call_count, 1)
                    if not proof:
                        self.assertEqual(set(records), {97, 99})
                        self.assertEqual(revoked, [])
                        continue
                    self.assertEqual(set(records), {97, 98})
                    expected_sec = 2. if proof['first_frame'] is not None else 10.
                    self.assertEqual(records[98]['video_sec'], expected_sec)
                    self.assertEqual(revoked[0]['soc'], 99)
                    matcher.extract_soc = lambda image: (99, .99)
                    for sec in (20, 21):
                        worker._process_segment_soc(1, np.zeros((1, 1, 3), np.uint8), sec,
                                                    matcher, tracker, records, .5)
                    self.assertEqual(records[99]['video_sec'], 20.)
                    self.assertEqual([r['soc'] for r in detected], [98, 99])

    def test_old_video_verification_requires_distinct_explicit_frames(self):
        worker = self.worker()
        class Capture:
            def __init__(self, *args): self.pos = 0; self.released = False
            def isOpened(self): return True
            def get(self, prop):
                return self.pos if prop == cv2.CAP_PROP_POS_FRAMES else (self.pos - 1) * 500
            def release(self): self.released = True
        def read(cap, seconds):
            cap.pos = int(seconds * 2) + 1
            return True, np.full((1, 1, 3), cap.pos, np.uint8)
        for values, expected in (([(98, .9)] * 7, True),
                                 ([(None, 0)] * 7, False),
                                 ([(99, .99)] + [(98, .9)] * 6, False),
                                 ([(98, .9)] * 3 + [(99, .93)] + [(98, .9)] * 3, False),
                                 ([(99, .9)] * 2 + [(98, .9)] * 5, False)):
            capture = Capture()
            iterator = iter(values)
            matcher = SimpleNamespace(extract_soc=lambda im: next(iterator), last_reason='original')
            with patch('core.video_worker.cv2.VideoCapture', return_value=capture), patch('core.video_worker.read_frame_at', read):
                proof = worker._verify_old_segment_record(1, {'soc': 99, 'video_sec': 10.}, 98, matcher)
            self.assertEqual(bool(proof), expected)
            self.assertTrue(capture.released)
            self.assertEqual(matcher.last_reason, 'original')
        capture = Capture()
        matcher = SimpleNamespace(extract_soc=lambda im: (98, .99), last_reason='original')
        with patch('core.video_worker.cv2.VideoCapture', return_value=capture), patch('core.video_worker.read_frame_at', lambda cap, sec: (True, np.zeros((1, 1, 3), np.uint8))):
            self.assertFalse(worker._verify_old_segment_record(1, {'soc': 99, 'video_sec': 10.}, 98, matcher))

    def test_low_fps_original_frame_is_detected_before_duplicate_center_request(self):
        worker = self.worker()
        class Capture:
            def __init__(self): self.second = 0
            def isOpened(self): return True
            def get(self, prop):
                return self.second + 1 if prop == cv2.CAP_PROP_POS_FRAMES else self.second * 1000
            def release(self): pass
        def read(cap, seconds):
            import math
            cap.second = math.ceil(seconds)
            return True, np.full((1, 1, 3), cap.second, np.uint8)
        for original_value in (99, 98):
            for record in ({'soc': 99, 'video_sec': 10.},
                           {'soc': 99, 'video_sec': 10., 'frame_index': 10}):
                matcher = SimpleNamespace(last_reason='ok', extract_soc=lambda im:
                    (original_value, .93) if int(im[0, 0, 0]) == 10 else (98, .90))
                with patch('core.video_worker.cv2.VideoCapture', return_value=Capture()), patch('core.video_worker.read_frame_at', read):
                    proof = worker._verify_old_segment_record(1, record, 98, matcher)
                if original_value == 99:
                    self.assertFalse(proof)
                else:
                    self.assertTrue(proof)
                    self.assertEqual(proof['first_frame'][1:], (10, 10.))
                    self.assertEqual(int(proof['first_frame'][0][0, 0, 0]), 10)

    def test_encoded_video_real_matcher_corrects_old_error_but_keeps_true_center(self):
        from core.segment_dataset import SegmentMatcher, write_image, write_json
        from core.record_evidence import load_snapshot
        patterns = {'7': 'abc', '8': 'abcdefg', '9': 'abcdfg'}
        segments = {'a': (6, 4, 28, 10), 'b': (28, 10, 34, 31),
                    'c': (28, 35, 34, 56), 'd': (6, 56, 28, 62),
                    'e': (0, 35, 6, 56), 'f': (0, 10, 6, 31),
                    'g': (6, 30, 28, 36)}
        def picture(value):
            image = np.zeros((80, 120, 3), np.uint8)
            for pos, digit in enumerate(str(value)):
                for segment in patterns[digit]:
                    x1, y1, x2, y2 = segments[segment]
                    cv2.rectangle(image, (12 + pos * 45 + x1, y1 + 6),
                                  (12 + pos * 45 + x2, y2 + 6), (255, 255, 255), -1)
            return image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for value in (97, 98, 99): write_image(root / f'{value}.png', picture(value))
            write_json(root / 'labels.json', {f'{value}.png': value for value in (97, 98, 99)})
            matcher = SegmentMatcher(root)
            for value in (97, 98, 99): self.assertEqual(matcher.extract_soc(picture(value))[0], value)
            for true_center in (False, True):
                path = root / f'center-{true_center}.avi'
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 10, (160, 80))
                self.assertTrue(writer.isOpened())
                frames = []
                for index in range(110):
                    frame = np.zeros((80, 160, 3), np.uint8)
                    frame[:, :120] = picture(99 if true_center and index == 20 else 98)
                    frame[:, 120:] = index
                    writer.write(frame)
                    frames.append(frame)
                writer.release()
                worker = self.worker()
                worker.video_path = str(path)
                worker.roi_soc_1, worker.roi_time_1, worker.roi_cap_1 = (0, 0, 120, 80), (120, 0, 20, 80), (140, 0, 20, 80)
                worker.evidence_directory = root / 'evidence'
                tracker = SegmentSOCTracker('charge')
                for sec, value in ((0, 97), (1, 97), (2, 99), (3, 99)):
                    tracker.update(value, sec, lambda: None)
                records = {97: {'soc': 97, 'video_sec': 0., 'capacity': .1},
                           99: {'soc': 99, 'video_sec': 2., 'capacity': .2}}
                revoked = []
                worker.point_revoked_signal.connect(lambda u, record: revoked.append(record))
                master = cv2.VideoCapture(str(path))
                try:
                    master.read()
                    before = master.get(cv2.CAP_PROP_POS_FRAMES)
                    for sec in (6, 7, 8, 9):
                        worker._process_segment_soc(1, frames[sec * 10], sec, matcher, tracker, records, .2)
                    self.assertEqual(master.get(cv2.CAP_PROP_POS_FRAMES), before)
                finally:
                    master.release()
                if true_center:
                    self.assertEqual(set(records), {97, 99})
                    self.assertEqual(revoked, [])
                    self.assertEqual(tracker.confirmed, 99)
                else:
                    self.assertEqual(set(records), {97, 98})
                    self.assertEqual(records[98]['video_sec'], 2.)
                    self.assertEqual(records[98]['frame_index'], 20)
                    np.testing.assert_array_equal(load_snapshot(records[98]), frames[20])
                    self.assertEqual([record['soc'] for record in revoked], [99])

    def test_ui_revocation_clears_export_source_without_erasing_newer_record(self):
        from ui.main_window import MainWindow
        from core.excel_filler import get_target_soc_list
        targets = get_target_soc_list('charge')
        table = QTableWidget(len(targets), 5)
        for row in range(len(targets)):
            for col in range(5): table.setItem(row, col, QTableWidgetItem('old'))
        record = {'soc': 99, 'video_sec': 2., 'reason': 'verified contradiction'}
        window = SimpleNamespace(_closing=False, extracted_records_1={99: record},
            extracted_records_2={99: {'video_sec': 8.}}, dirty_units=set(),
            unit_modes={1: SimpleNamespace(currentData=lambda: 'charge')}, table_1=table)
        window._selected_record_preview = dict(record)
        window._selected_record_unit = 1
        window.btn_locate_record = Mock()
        window.lbl_frame_info = Mock()
        window.lbl_status = Mock()
        MainWindow.on_worker_point_revoked(window, 1, record)
        self.assertIsNone(window._selected_record_preview)
        window.btn_locate_record.setEnabled.assert_called_once_with(False)
        self.assertEqual(window.extracted_records_1, {})
        self.assertIn(99, window.extracted_records_2)
        self.assertEqual(table.item(targets.index(99), 1).text(), '--')
        window.extracted_records_1[99] = {'video_sec': 20.}
        MainWindow.on_worker_point_revoked(window, 1, record)
        self.assertEqual(window.extracted_records_1[99]['video_sec'], 20.)

if __name__ == '__main__': unittest.main()
