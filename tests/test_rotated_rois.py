import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime
import math
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtCore import QEvent, QPointF, QSettings, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QMessageBox

from core.roi_geometry import crop_roi, roi_corners, roi_in_bounds
from core.segment_dataset import (SampleCaptureWorker, crop_digits, samples,
    read_image, read_json, set_label, SegmentMatcher, write_json)
from core.video_worker import VideoAnalysisWorker
from core.ocr_engine import OCREngine
from ui.main_window import MainWindow
from ui.video_canvas import VideoCanvas
from ui.segment_dialog import SegmentDatasetDialog


def tilted_frame(angle):
    image = np.zeros((64, 120, 3), np.uint8)
    cv2.putText(image, '98', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 3)
    # Corner colours catch reversed top/bottom, mirroring and swapped axes.
    for point, colour in [((8, 8), (0, 0, 255)), ((111, 8), (0, 255, 0)),
                          ((111, 55), (255, 0, 0)), ((8, 55), (255, 255, 0))]:
        cv2.circle(image, point, 4, colour, -1)
    roi = (140, 128, 120, 64, angle)
    src = np.array([[-.5, -.5], [119.5, -.5], [119.5, 63.5], [-.5, 63.5]], np.float32)
    transform = cv2.getPerspectiveTransform(src, roi_corners(roi)-.5)
    frame = cv2.warpPerspective(image, transform, (400, 320))
    return image, frame, roi


class RotatedRoiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_full_circle_crop_keeps_digits_and_corner_order(self):
        for angle in list(range(0, 360, 7)) + [0.1, 89.9, 90, 180, 270, 359.9]:
            with self.subTest(angle=angle):
                image, frame, roi = tilted_frame(angle)
                result = crop_roi(frame, roi)
                self.assertEqual(result.shape, image.shape)
                self.assertLess(np.abs(result.astype(float)-image).mean(), 9)
                for x, y in [(8, 8), (111, 8), (111, 55), (8, 55)]:
                    np.testing.assert_allclose(result[y, x], image[y, x], atol=2)

    def test_legacy_orientation_is_pixel_exact_in_all_three_consumers(self):
        frame = np.arange(200*300*3, dtype=np.uint32).reshape(200, 300, 3).astype(np.uint8)
        roi = (20, 30, 120, 64)
        original = frame[30:94, 20:140]
        for angle, code in [(0, None), (90, cv2.ROTATE_90_COUNTERCLOCKWISE),
                            (180, cv2.ROTATE_180), (270, cv2.ROTATE_90_CLOCKWISE)]:
            expected = original if code is None else cv2.rotate(original, code)
            for consumer in (MainWindow.crop_helper, VideoAnalysisWorker.crop, crop_digits):
                np.testing.assert_array_equal(consumer(frame, roi, angle), expected)

    def test_arbitrary_tilt_and_legacy_orientation_share_one_crop(self):
        _, frame, roi = tilted_frame(17.3)
        for orientation in (0, 90, 180, 270):
            expected = crop_digits(frame, roi, orientation)
            np.testing.assert_array_equal(MainWindow.crop_helper(frame, roi, orientation), expected)
            np.testing.assert_array_equal(VideoAnalysisWorker.crop(frame, roi, orientation), expected)

    def test_out_of_bounds_and_invalid_geometry_rejected(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        roi = (0, 0, 80, 30, 45)
        self.assertFalse(roi_in_bounds(roi, 100, 100))
        for consumer in (MainWindow.crop_helper, VideoAnalysisWorker.crop, crop_digits):
            with self.assertRaisesRegex(ValueError, '边界'):
                consumer(frame, roi)
        for roi in [(0, 0, 0, 10), (0, 0, 10, 10, float('nan'))]:
            with self.assertRaises(ValueError):
                crop_roi(frame, roi)

    def test_drag_handle_all_six_rois_after_preview_resize_and_reset(self):
        canvas = VideoCanvas()
        canvas.set_frame(np.zeros((320, 400, 3), np.uint8))
        canvas.resize(800, 500)
        canvas.show()
        self.app.processEvents()
        for key in canvas.rois:
            canvas.rois = dict.fromkeys(canvas.rois)
            canvas.rois[key] = (140, 128, 120, 64)
            canvas.set_active_roi_type(key)
            start = canvas.rotation_handle(key)[1].toPoint()
            QTest.mousePress(canvas, Qt.LeftButton, pos=start)
            self.assertEqual(canvas.rotating_roi, key)
            centre = canvas.rotation_handle(key)[2]
            radius = math.hypot(start.x()-centre.x(), start.y()-centre.y())
            radians = math.atan2(start.y()-centre.y(), start.x()-centre.x())+math.radians(37)
            end = QPointF(centre.x()+radius*math.cos(radians), centre.y()+radius*math.sin(radians))
            event = QMouseEvent(QEvent.MouseMove,
                                end, Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
            QApplication.sendEvent(canvas, event)
            QTest.mouseRelease(canvas, Qt.LeftButton, pos=end.toPoint())
            self.assertAlmostEqual(canvas.rois[key][4], 37, delta=1)
            self.assertFalse(canvas.is_drawing)
            QTest.mouseDClick(canvas, Qt.LeftButton, pos=canvas.rotation_handle(key)[1].toPoint())
            self.assertEqual(canvas.rois[key][4], 0)
        canvas.close()

    def test_json_presets_keep_angle_and_real_video_capture_matches_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            image, frame, roi = tilted_frame(27)
            source = root/'tilted.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (400, 320))
            self.assertTrue(writer.isOpened())
            for _ in range(12):
                writer.write(frame)
            writer.release()
            decoded = cv2.VideoCapture(str(source))
            ok, frame = decoded.read()
            decoded.release()
            self.assertTrue(ok)
            write_json(root/'preset.json', dict(roi=roi))
            restored = read_json(root/'preset.json', {})['roi']
            worker = SampleCaptureWorker(str(source), root/'samples', restored, interval=.2, end=1)
            errors = []
            worker.failed.connect(errors.append)
            worker.run()
            self.assertEqual(errors, [])
            rows = samples(root/'samples')
            self.assertEqual(len(rows), 5)
            captured = read_image(root/'samples'/rows[0]['key'])
            np.testing.assert_array_equal(captured, VideoAnalysisWorker.crop(frame, restored))
            self.assertLess(np.abs(captured.astype(float)-image).mean(), 12)
            set_label(root/'samples', rows[0]['key'], 98)
            matcher = SegmentMatcher(root/'samples')
            self.assertEqual(matcher.extract_soc(captured)[0], 98)

    @unittest.skipUnless(Path('C:/Windows/Fonts/arial.ttf').exists(), 'Windows display font required')
    def test_real_ocr_on_clean_digits_across_arbitrary_angles(self):
        engine = OCREngine(cpu_threads=2)
        display = Image.new('RGB', (180, 110), 'white')
        ImageDraw.Draw(display).text((18, 12), '98',
            font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 72), fill='black')
        original = np.array(display)
        self.assertEqual(engine.extract_soc(original, 1)[0], 98)
        src = np.array([[-.5, -.5], [179.5, -.5], [179.5, 109.5], [-.5, 109.5]], np.float32)
        for angle in (0, 17.3, 90, 137, 180, 270, 359.9):
            with self.subTest(angle=angle):
                roi = (110, 100, 180, 110, angle)
                transform = cv2.getPerspectiveTransform(src, roi_corners(roi)-.5)
                frame = cv2.warpPerspective(original, transform, (400, 320), borderValue=(255, 255, 255))
                self.assertEqual(engine.extract_soc(crop_roi(frame, roi), 1)[0], 98)

    def test_invalid_frame_does_not_start_analysis_or_clear_existing_records(self):
        with patch('ui.main_window.ResourceMonitor.start'):
            window = MainWindow()
        window.video_path = 'unused.avi'
        window.current_frame_cv = np.zeros((100, 100, 3), np.uint8)
        for key in ('soc_1', 'time_1', 'cap_1'):
            window.canvas.rois[key] = (0, 0, 80, 30, 45)
        window.extracted_records_1[99] = {'capacity': 1}
        with patch('ui.main_window.QMessageBox.warning') as warning, \
                patch('ui.main_window.VideoAnalysisWorker') as worker:
            window.start_analysis()
            warning.assert_called_once()
            worker.assert_not_called()
        self.assertEqual(window.extracted_records_1[99]['capacity'], 1)
        window.close()


class SampleRetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def build_library(self, root):
        _, frame, roi = tilted_frame(0)
        source = root/'samples.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (400, 320))
        for _ in range(20):
            writer.write(frame)
        writer.release()
        worker = SampleCaptureWorker(str(source), root/'library', roi, interval=.2, end=1)
        worker.run()
        return root/'library'

    def open_dialog(self, root, library):
        settings = QSettings(str(root/'settings.ini'), QSettings.IniFormat)
        with patch('ui.segment_dialog.QSettings', return_value=settings), \
                patch('ui.segment_dialog.data_home', return_value=root/'state'), \
                patch('core.segment_dataset.data_home', return_value=root/'state'):
            return SegmentDatasetDialog({1: [str(library)], 2: []})

    def test_accept_reject_and_close_preserve_samples_and_reopen(self):
        for action in ('accept', 'reject', 'close'):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                library = self.build_library(root)
                before = samples(library)
                set_label(library, before[0]['key'], 98)
                original_files = {p: p.read_bytes() for p in library.rglob('*') if p.is_file()}
                dialog = self.open_dialog(root, library)
                dialog.show()
                self.app.processEvents()
                getattr(dialog, action)()
                for path, content in original_files.items():
                    self.assertTrue(path.exists())
                    self.assertEqual(path.read_bytes(), content)
                reopened = self.open_dialog(root, library)
                self.assertEqual(len(reopened._all_rows), len(before))
                self.assertEqual(sum(r['key'] not in reopened._labels for r in reopened._all_rows), 4)
                reopened.close()

    def test_clear_requires_confirmation_and_keeps_labelled_and_invalid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = self.build_library(root)
            rows = samples(library)
            set_label(library, rows[0]['key'], 98)
            set_label(library, rows[1]['key'], 'invalid')
            dialog = self.open_dialog(root, library)
            with patch('ui.segment_dialog.QMessageBox.question', return_value=QMessageBox.No):
                dialog.clear_all_unlabelled()
            self.assertEqual(len(samples(library)), 5)
            with patch('ui.segment_dialog.QMessageBox.question', return_value=QMessageBox.Yes):
                dialog.clear_all_unlabelled()
            self.assertEqual(len(samples(library)), 2)
            self.assertTrue((library/rows[0]['key']).exists())
            self.assertTrue((library/rows[1]['key']).exists())
            dialog.close()

    def test_close_while_capture_running_keeps_partial_images(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            library = self.build_library(root)
            dialog = self.open_dialog(root, library)
            worker = SampleCaptureWorker(str(root/'samples.avi'), library, (140, 128, 120, 64),
                                         interval=.2, end=2)
            # Pause after the first actual saved image so shutdown is deterministic.
            worker.progress.connect(lambda *_: worker.msleep(30), Qt.DirectConnection)
            dialog.worker = worker
            worker.completed.connect(dialog.capture_done)
            worker.finished.connect(dialog._finish_pending_close)
            dialog.show()
            worker.start()
            deadline = time.monotonic()+3
            while not worker.batch_id or not list((library/'captures'/worker.batch_id).glob('*.png')):
                self.app.processEvents()
                if time.monotonic() >= deadline:
                    self.fail('capture did not save its first frame')
                time.sleep(.005)
            dialog.close()
            while dialog.isVisible() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(.01)
            self.assertFalse(dialog.isVisible())
            self.assertFalse(worker.isRunning())
            new_rows = samples(library, worker.batch_id)
            self.assertGreater(len(new_rows), 0)
            self.assertTrue(all((library/r['key']).exists() for r in new_rows))
            reopened = self.open_dialog(root, library)
            self.assertEqual(len(reopened._all_rows), 5+len(new_rows))
            reopened.close()


if __name__ == '__main__':
    unittest.main()
