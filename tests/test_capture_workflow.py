"""Regression tests for exact-frame recapture, meter fields and UI routing."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime
import tempfile
import unittest
from pathlib import Path
from datetime import time
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
from core.manual_capture import CaptureTask, selected_fields
from core.meter_reader import read_meter_fields
from core.segment_dataset import read_image, write_image
from core.segment_dataset import SEGMENT_MODE
from core.video_worker import VideoAnalysisWorker
from core.record_evidence import load_snapshot
from ui.manual_capture_dialog import ManualCaptureDialog


class CaptureWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_single_line_uses_strict_parsers_and_explicit_units(self):
        class Engine:
            def __init__(self, capacity, confidence=.99):
                self.answers = iter([('03:10:57', .99), (capacity, confidence)])
            def ocr(self, image, **kwargs):
                assert kwargs == dict(use_det=False, use_cls=False)
                return [next(self.answers)], None
        image = np.ones((20, 100, 3), np.uint8)
        for raw, unit, expected in [('8.5047', 'Ah', 8.5047), ('83', 'mAh', .083),
                                     ('8504.7 mAh', 'Ah', 8.5047), ('8.5047 Ah', 'mAh', 8.5047)]:
            with self.subTest(raw=raw, unit=unit):
                result = read_meter_fields(image, image, Engine(raw), unit, True)
                self.assertEqual(result['time_obj'], time(3, 10, 57))
                self.assertEqual(result['capacity'], expected)
        for raw, confidence in [('8 5047', .99), ('-8.5', .99), ('nan', .99), ('8.5', .89)]:
            with self.subTest(raw=raw, confidence=confidence):
                self.assertIsNone(read_meter_fields(image, image, Engine(raw, confidence), single_line=True)['capacity'])

    def test_ordinary_ocr_uses_explicit_mah_even_for_small_values(self):
        class Engine:
            def extract_time(self, image): return None, None, 0
            def extract_capacity(self, image, **kwargs): return 83, '83', .99
        image = np.ones((20, 100, 3), np.uint8)
        result = read_meter_fields(image, image, Engine(), 'mAh')
        self.assertEqual(result['capacity'], .083)
        self.assertEqual(result['time_str'], '--')
        self.assertIsNone(result['time_obj'])

    def test_candidate_ocr_uses_saved_image_without_reopening_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'selected.png'
            frame = np.full((30, 90, 3), 47, np.uint8)
            write_image(path, frame)
            selected = {'path': str(path), 'video_sec': 123.45, 'frame_index': 3703}
            task = CaptureTask('nonexistent.mp4', tmp, {}, {}, selected=selected)
            result, errors = [], []
            task.result.connect(result.append); task.error.connect(errors.append)
            def fields(actual, *args):
                self.assertTrue(np.array_equal(actual, frame))
                return dict(time_str='00:00:47', time_obj=time(0,0,47), capacity=.83)
            with patch('core.manual_capture.OCREngine'), patch('core.manual_capture.selected_fields', side_effect=fields), \
                    patch('core.manual_capture.cv2.VideoCapture', side_effect=AssertionError('Must not reseek')):
                task.run()
            self.assertEqual(errors, [])
            self.assertEqual(result[0]['frame_index'], 3703)
            self.assertEqual(result[0]['video_sec'], 123.45)

    def test_exact_previous_next_frames_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'tiny.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 10, (80, 40))
            self.assertTrue(writer.isOpened())
            for i in range(12): writer.write(np.full((40,80,3), i, np.uint8))
            writer.release()
            for index in (0, 4, 5, 6, 11):
                task = CaptureTask(str(path), tmp, {'soc': [0,0,80,40]}, {}, target_index=index)
                candidates, errors = [], []
                task.candidate.connect(candidates.append); task.error.connect(errors.append); task.run()
                self.assertEqual(errors, [])
                self.assertEqual(candidates[0]['frame_index'], index)
                self.assertAlmostEqual(candidates[0]['video_sec'], index/10)
                self.assertTrue(np.all(read_image(candidates[0]['path']) == index))
            for index in (-1, 12):
                task = CaptureTask(str(path), tmp, {'soc': [0,0,80,40]}, {}, target_index=index)
                errors = []; task.error.connect(errors.append); task.run()
                self.assertEqual(errors, ['已到视频边界'])

    def test_invalid_scan_ranges_do_not_emit_candidates(self):
        for start, end, step in [(0,1,0), (2,1,.5), (-1,1,.5), (0,float('inf'),.5)]:
            task = CaptureTask('missing', '.', {}, {}, start=start, end=end, step=step)
            errors = []; task.error.connect(errors.append); task.run()
            self.assertTrue(errors)

    def test_manual_edit_rejects_invalid_capacity_and_keeps_missing_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'LOCALAPPDATA': tmp}):
            path = Path(tmp)/'frame.png'; frame = np.ones((30,90,3),np.uint8)
            write_image(path, frame)
            rois = {k: [0,0,90,30] for k in ('soc','time','cap')}
            dialog = ManualCaptureDialog('video.mp4', 2, 99, rois, {}, 0, 1, 10)
            dialog.add_candidate(dict(path=str(path), frame_index=9, video_sec=.9, thumbnail=frame))
            dialog.candidates.setCurrentRow(0)
            dialog.read_complete(dict(time_str='--', time_obj=None, capacity=None, capacity_raw='unreadable'))
            for value in ('nan', 'inf', '-1'):
                dialog.capacity_edit.setText(value)
                with patch.object(QMessageBox, 'warning') as warning:
                    dialog.confirm()
                self.assertTrue(warning.called)
                self.assertIsNone(dialog.record)
            dialog.capacity_edit.clear()
            with patch.object(QMessageBox, 'question', return_value=QMessageBox.Yes): dialog.confirm()
            self.assertIsNone(dialog.record['capacity'])
            self.assertIsNone(dialog.record['time_obj'])
            self.assertEqual(dialog.record['capture_unit'], 2)
            self.assertEqual(dialog.record['video_sec'], .9)

    def test_cancel_removes_temporary_candidates_without_creating_record(self):
        rois = {k: [0,0,90,30] for k in ('soc','time','cap')}
        dialog = ManualCaptureDialog('video.mp4', 1, 98, rois, {}, 0, 1, 10)
        directory = Path(dialog.temp.name)
        dialog.reject()
        self.assertFalse(directory.exists())
        self.assertIsNone(dialog.record)

    def test_worker_visual_tracking_keeps_both_units_first_frame_fields(self):
        class Engine:
            def __init__(self, **kwargs): pass
            def extract_time(self, image):
                value = int(image[0,0,0])
                return f'00:00:{value:02}', time(0,0,value), 1
            def extract_capacity(self, image, **kwargs):
                value = int(image[0,0,0]) / 100
                return value, str(value), 1
        class Matcher:
            labels = set(range(101))
            last_reason = ''
            first_seen = {}
            def __init__(self, path, **kwargs): self.unit = 1 if path == 'a' else 2
            def extract_soc(self, image):
                value = int(image[0,0,0])
                self.first_seen.setdefault((self.unit,value), int(image[0,1,0]))
                return value, 1
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'LOCALAPPDATA':tmp}):
            path = Path(tmp)/'dual.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'FFV1'), 10, (400,80))
            self.assertTrue(writer.isOpened())
            for i in range(60):
                frame = np.zeros((80,400,3), np.uint8)
                for unit, value in ((1, 100 if i<20 else 99 if i<36 else 98), (2, 20 if i<16 else 21)):
                    x = (unit-1)*200
                    cv2.putText(frame,str(value),(x+5,55),cv2.FONT_HERSHEY_SIMPLEX,1.4,(255,255,255),3)
                    frame[0,x] = value
                    frame[0,x+1] = i
                    frame[:,x+150:x+200] = i
                writer.write(frame)
            writer.release()
            summaries = []
            for high_speed in (False, True):
                Matcher.first_seen = {}
                rois = {f'roi_{key}_{unit}':(x if key=='soc' else x+150,0,150 if key=='soc' else 50,80)
                        for unit,x in ((1,0),(2,200)) for key in ('soc','time','cap')}
                worker = VideoAnalysisWorker(str(path), **rois, modes={1:'discharge',2:'charge'},
                    digit_type=SEGMENT_MODE, segment_paths={1:'a',2:'b'}, sample_interval_sec=.2,
                    visual_tracking={1:True,2:True}, high_speed=high_speed, cpu_threads=2)
                results, errors = [], []
                worker.finished_signal.connect(lambda a,b:results.append((a,b)))
                worker.error_signal.connect(errors.append)
                with patch('core.video_worker.OCREngine', Engine), patch('core.video_worker.SegmentMatcher', Matcher):
                    worker.run()
                self.assertEqual(errors, [])
                first, second = results[0]
                self.assertEqual(list(first), [100,99,98])
                self.assertEqual(list(second), [20,21])
                expected = {1:{100:0,99:20,98:36}, 2:{20:0,21:16}}
                summary = []
                for unit, records in enumerate((first, second), 1):
                    for soc, rec in records.items():
                        # Sampling need not land exactly on the physical change;
                        # retain the first decoded frame, never the later confirmation.
                        index = Matcher.first_seen[(unit,soc)]
                        self.assertLessEqual(abs(index-expected[unit][soc]), 2)
                        self.assertAlmostEqual(rec['video_sec'], index/10)
                        self.assertEqual(rec['time_obj'], time(0,0,index))
                        self.assertEqual(rec['capacity'], index/100)
                        if rec['is_target']:
                            self.assertEqual(rec['frame_index'], index)
                            self.assertEqual(int(load_snapshot(rec)[0,(unit-1)*200+150,0]),index)
                        summary.append((unit,soc,rec['video_sec'],rec['capacity']))
                summaries.append(summary)
            self.assertEqual(summaries[0], summaries[1])
