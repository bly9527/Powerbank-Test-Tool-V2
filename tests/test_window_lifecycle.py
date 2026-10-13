import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime  # Keep DLL initialization ahead of the Qt/OpenCV imports on Windows.

import tempfile
import time
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

from PyQt5.QtCore import QPoint, QPointF, QSettings, QThread, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication, QMessageBox, QScrollArea

from core.project_archive import create_project
from ui.main_window import MainWindow
from ui.segment_dialog import SegmentDatasetDialog


class StoppableWorker(QThread):
    def __init__(self):
        super().__init__()
        self.stopping = False

    def stop(self):
        self.stopping = True

    def run(self):
        while not self.stopping:
            self.msleep(10)
        self.msleep(300)


class WindowLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_until_hidden(self, window):
        deadline = time.monotonic() + 3
        while window.isVisible() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertFalse(window.isVisible())

    def test_switch_rejects_running_analysis_and_preserves_unsaved_results(self):
        template = next(Path(__file__).resolve().parents[1].glob('PB519*.xlsx'))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = lambda *_: QSettings(str(root / 'settings.ini'), QSettings.IniFormat)
            with patch('core.project_archive.projects_home', return_value=root / 'projects'), \
                    patch('ui.main_window.QSettings', side_effect=settings), \
                    patch('ui.main_window.ResourceMonitor.start'):
                first = create_project('first', template)
                second = create_project('second', template)
                window = MainWindow()
                window.reload_projects(str(first.root))
                target = window.cb_project.findData(str(second.root))
                self.assertGreater(target, 0)

                window.worker = StoppableWorker()
                window.worker.start()
                with patch('ui.main_window.QMessageBox.information'):
                    window.cb_project.setCurrentIndex(target)
                self.assertEqual(window.cb_project.currentData(), str(first.root))
                self.assertEqual(window.project.root, first.root)
                window.worker.stop()
                window.worker.wait(2000)
                window.worker = None

                window.on_worker_point_detected(1, {'soc': 99, 'time_str': '00:00:05',
                    'capacity': 0.1, 'video_sec': 5, 'time_obj': None})
                with patch('ui.main_window.QMessageBox.question', return_value=QMessageBox.No):
                    window.cb_project.setCurrentIndex(target)
                self.assertEqual(window.project.root, first.root)
                self.assertEqual(window.table_1.item(0, 1).text(), '00:00:05')
                with patch('ui.main_window.QMessageBox.question', return_value=QMessageBox.Yes):
                    window.cb_project.setCurrentIndex(target)
                self.assertEqual(window.project.root, second.root)
                self.assertEqual(window.table_1.item(0, 1).text(), '--')
                self.assertFalse(window.dirty_units)
                window.close()

    def test_main_window_close_waits_without_blocking_interface(self):
        with patch('ui.main_window.ResourceMonitor.start'):
            window = MainWindow()
        window.show()
        window.worker = StoppableWorker()
        window.worker.start()
        started = time.monotonic()
        window.close()
        self.assertLess(time.monotonic() - started, 0.2)
        self.assertTrue(window._closing)
        self.wait_until_hidden(window)
        self.assertFalse(window.worker.isRunning())

    def test_resource_monitor_close_remains_responsive(self):
        with patch('core.resource_monitor.physical_cores', side_effect=lambda: (time.sleep(0.4), 8)[1]):
            window = MainWindow()
            window.show()
            started = time.monotonic()
            window.close()
            self.assertLess(time.monotonic() - started, 0.2)
            self.wait_until_hidden(window)
            self.assertFalse(window.resource_monitor.isRunning())

    def test_csv_export_keeps_columns_and_chinese_text(self):
        with tempfile.TemporaryDirectory() as temp, patch('ui.main_window.ResourceMonitor.start'):
            window = MainWindow()
            window.on_worker_point_detected(1, {'soc': 99, 'time_str': '00:00:05',
                'capacity': 0.1, 'video_sec': 5, 'time_obj': None})
            output = Path(temp) / 'results.csv'
            with patch('ui.main_window.QFileDialog.getSaveFileName', return_value=(str(output), 'CSV')), \
                    patch('ui.main_window.QMessageBox.information'):
                window.export_to_csv()
            with output.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['样机'], '1#')
            self.assertEqual(rows[0]['百分比(SOC)'], '99%')
            self.assertEqual(rows[0]['容量(Ah)'], '0.1')
            window.close()

    def test_wheel_does_not_change_left_panel_inputs_or_project(self):
        with patch('ui.main_window.ResourceMonitor.start'):
            window = MainWindow()
        window.sp_interval.setValue(1.0)
        event = QWheelEvent(QPointF(2, 2), QPointF(2, 2), QPoint(), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        QApplication.sendEvent(window.sp_interval, event)
        self.assertEqual(window.sp_interval.value(), 1.0)
        window.cb_cpu_threads.setCurrentIndex(0)
        event = QWheelEvent(QPointF(2, 2), QPointF(2, 2), QPoint(), QPoint(0, 120),
            Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
        QApplication.sendEvent(window.cb_cpu_threads, event)
        self.assertEqual(window.cb_cpu_threads.currentIndex(), 0)
        window.close()

    def test_delete_current_project_requires_confirmation_and_refreshes_list(self):
        template = next(Path(__file__).resolve().parents[1].glob('PB519*.xlsx'))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = lambda *_: QSettings(str(root / 'settings.ini'), QSettings.IniFormat)
            with patch('core.project_archive.projects_home', return_value=root / 'projects'), \
                    patch('ui.main_window.QSettings', side_effect=settings), \
                    patch('ui.main_window.ResourceMonitor.start'):
                project = create_project('temporary', template)
                window = MainWindow()
                window.reload_projects(str(project.root))
                with patch('ui.main_window.QMessageBox.question', return_value=QMessageBox.No):
                    window.delete_current_project()
                self.assertTrue(project.root.exists())
                with patch('ui.main_window.QMessageBox.question', return_value=QMessageBox.Yes), \
                        patch('ui.main_window.QMessageBox.information'):
                    window.delete_current_project()
                self.assertFalse(project.root.exists())
                self.assertIsNone(window.project)
                self.assertEqual(window.cb_project.count(), 1)
                self.assertFalse(window.btn_delete_project.isEnabled())
                window.close()

    def test_capture_dialog_closes_after_worker_finishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = lambda *_: QSettings(str(root / 'settings.ini'), QSettings.IniFormat)
            with patch('ui.segment_dialog.QSettings', side_effect=settings):
                dialog = SegmentDatasetDialog({1: [str(root / 'samples')], 2: []})
            dialog.show()
            dialog.worker = StoppableWorker()
            dialog.worker.finished.connect(dialog._finish_pending_close)
            dialog.worker.start()
            dialog.close()
            self.assertIsNotNone(dialog._pending_close_result)
            self.wait_until_hidden(dialog)
            self.assertFalse(dialog.worker.isRunning())

    def test_sample_dialog_wheel_does_not_change_capture_interval(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = lambda *_: QSettings(str(root / 'settings.ini'), QSettings.IniFormat)
            with patch('ui.segment_dialog.QSettings', side_effect=settings):
                dialog = SegmentDatasetDialog({1: [str(root / 'samples')], 2: []})
            dialog.interval.setValue(0.2)
            event = QWheelEvent(QPointF(2, 2), QPointF(2, 2), QPoint(), QPoint(0, 120),
                Qt.NoButton, Qt.NoModifier, Qt.ScrollUpdate, False)
            QApplication.sendEvent(dialog.interval, event)
            self.assertEqual(dialog.interval.value(), 0.2)
            dialog.close()

    def test_sample_dialog_remains_usable_at_low_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = lambda *_: QSettings(str(root / 'settings.ini'), QSettings.IniFormat)
            with patch('ui.segment_dialog.QSettings', side_effect=settings):
                dialog = SegmentDatasetDialog({1: [str(root / 'samples')], 2: []})
            dialog.resize(640, 480)
            dialog.show()
            self.app.processEvents()
            scroll = dialog.findChild(QScrollArea)
            self.assertIsNotNone(scroll)
            self.assertGreater(scroll.verticalScrollBar().maximum(), 0)
            self.assertTrue(dialog.save_selection.isVisible())
            self.assertLessEqual(dialog.minimumSizeHint().height(), 480)
            self.assertEqual((dialog.canvas.minimumWidth(), dialog.canvas.minimumHeight()), (320, 180))
            dialog.close()


if __name__ == '__main__':
    unittest.main()
