import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime
import cv2
import numpy as np
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import QApplication
from core.performance import high_speed_threads
from core.segment_dataset import SegmentMatcher, SEGMENT_MODE, write_image, write_json
from core.video_worker import VideoAnalysisWorker
from ui.main_window import MainWindow


def digit(value):
    image = np.zeros((70, 110, 3), np.uint8)
    cv2.putText(image, str(value), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return image


def library(root):
    labels = {}
    for value in range(80):
        name = f'{value}.png'
        write_image(root / name, digit(value))
        labels[name] = value
    write_json(root / 'labels.json', labels)
    return labels


class CPUHighSpeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_budget_reserves_capacity_and_handles_single_core(self):
        self.assertEqual([high_speed_threads(n) for n in (1, 2, 6, 12, 16)], [1, 1, 5, 10, 13])
        with patch('core.performance.os.cpu_count', return_value=None):
            self.assertEqual(high_speed_threads(), 1)

    def test_parallel_decisions_scores_and_tie_reasons_equal_serial(self):
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=3) as pool:
            root = Path(tmp)
            labels = library(root)
            for scenario in ('valid', 'ambiguous', 'invalid'):
                if scenario == 'ambiguous':
                    write_image(root / 'conflict.png', digit(25))
                    labels['conflict.png'] = 26
                elif scenario == 'invalid':
                    labels.pop('conflict.png')
                    labels['25.png'] = 'invalid'
                write_json(root / 'labels.json', labels)
                serial = SegmentMatcher(root)
                parallel = SegmentMatcher(root, executor=pool, workers=3)
                self.assertGreater(len(parallel.templates), 64)
                for image in (digit(25), digit(60), np.zeros((70, 110, 3), np.uint8)):
                    expected = serial.extract_soc(image)
                    self.assertEqual(parallel.extract_soc(image), expected)
                    self.assertEqual(parallel.last_reason, serial.last_reason)
                value, _ = parallel.extract_soc(digit(25))
                self.assertEqual(value, 25 if scenario == 'valid' else None)

    def test_worker_pool_cleanup_and_results_on_success_stop_and_error(self):
        class Capture:
            def __init__(self, *_): self.pos = 0
            def isOpened(self): return True
            def get(self, prop):
                return {cv2.CAP_PROP_FRAME_COUNT: 3, cv2.CAP_PROP_FPS: 1,
                        cv2.CAP_PROP_POS_FRAMES: self.pos,
                        cv2.CAP_PROP_POS_MSEC: max(0, self.pos-1)*1000}.get(prop, 0)
            def set(self, prop, value):
                self.pos = int(value/1000) if prop == cv2.CAP_PROP_POS_MSEC else int(value)
            def read(self):
                if self.pos >= 3: return False, None
                self.pos += 1
                return True, digit(25)
            def release(self): pass

        class OCR:
            def __init__(self, **_): pass
            def extract_time(self, _): return '00:00:01', time(0, 0, 1), 1
            def extract_capacity(self, _, **kw): return 1.0, '1.0', 1

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library(root)
            baseline = None
            for scenario in ('serial', 'fast', 'stop', 'error'):
                roi = (0, 0, 110, 70)
                worker = VideoAnalysisWorker('fake', roi_soc_1=roi, roi_time_1=roi, roi_cap_1=roi,
                    roi_soc_2=roi, roi_time_2=roi, roi_cap_2=roi, digit_type=SEGMENT_MODE,
                    modes={1: 'charge', 2: 'charge'}, segment_paths={1: str(root), 2: str(root)},
                    high_speed=scenario != 'serial')
                pools, results, errors = [], [], []
                def make_pool(**kw):
                    pool = ThreadPoolExecutor(**kw)
                    pools.append(pool)
                    return pool
                worker.finished_signal.connect(lambda a, b: results.append((a, b)))
                worker.error_signal.connect(errors.append)
                if scenario == 'stop': worker.progress_signal.connect(lambda *_: worker.stop())
                previous = cv2.getNumThreads()
                with patch('core.video_worker.ThreadPoolExecutor', side_effect=make_pool), \
                     patch('core.video_worker.high_speed_threads', return_value=3), \
                     patch('core.video_worker.cv2.VideoCapture', Capture), \
                     patch('core.video_worker.OCREngine', side_effect=RuntimeError('test error') if scenario == 'error' else OCR):
                    worker.run()
                self.assertEqual(cv2.getNumThreads(), previous)
                self.assertEqual(len(pools), int(scenario != 'serial'))
                for pool in pools:
                    with self.assertRaises(RuntimeError): pool.submit(lambda: None)
                if scenario == 'serial':
                    baseline = results
                    self.assertIn(25, baseline[0][0])
                    self.assertIn(25, baseline[0][1])
                elif scenario == 'fast': self.assertEqual(results, baseline)
                elif scenario == 'stop': self.assertEqual(results, [({}, {})])
                self.assertEqual(bool(errors), scenario == 'error')

    def test_ui_mode_persists_and_locks_manual_threads_during_high_speed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = lambda *_: QSettings(str(root / 'settings.ini'), QSettings.IniFormat)
            with patch('ui.main_window.QSettings', side_effect=settings), \
                 patch('core.project_archive.projects_home', return_value=root / 'projects'), \
                 patch('ui.main_window.ResourceMonitor.start'):
                window = MainWindow()
                window.cb_cpu_threads.setCurrentIndex(window.cb_cpu_threads.findData(1))
                window.cb_cpu_high_speed.setChecked(True)
                self.assertFalse(window.cb_cpu_threads.isEnabled())
                window.close()
                reopened = MainWindow()
                self.assertTrue(reopened.cb_cpu_high_speed.isChecked())
                self.assertEqual(reopened.cb_cpu_threads.currentData(), 1)
                self.assertFalse(reopened.cb_cpu_threads.isEnabled())
                reopened.cb_cpu_high_speed.setChecked(False)
                self.assertTrue(reopened.cb_cpu_threads.isEnabled())
                reopened.worker = SimpleNamespace(isRunning=lambda: True)
                reopened.update_segment_controls()
                self.assertFalse(reopened.cb_cpu_high_speed.isEnabled())
                self.assertFalse(reopened.cb_cpu_threads.isEnabled())
                reopened.worker = None
                reopened.update_segment_controls()
                self.assertTrue(reopened.cb_cpu_high_speed.isEnabled())
                self.assertTrue(reopened.cb_cpu_threads.isEnabled())
                reopened.close()
