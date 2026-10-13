import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QSettings
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox
from core.segment_dataset import (feature, SegmentMatcher, load_ignore_regions,
    save_ignore_regions, write_image, write_json, export_template_package,
    import_template_package, import_dataset)
from ui.ignore_regions_dialog import IgnoreRegionsCanvas, IgnoreRegionsDialog


def digits(text, brightness=180):
    image = np.zeros((80,200,3), np.uint8)
    cv2.putText(image, str(text), (55,59), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                (brightness,)*3, 3, cv2.LINE_8)
    return image


def library(root, number, image, regions=()):
    batch = root/'captures'/'batch'
    batch.mkdir(parents=True)
    write_image(batch/'0.png', image)
    write_json(batch/'manifest.json', {'source':'test.mp4', 'unit':1, 'interval':1, 'samples':[
        {'image':'0.png','video_sec':0,'frame_index':0}]})
    write_json(root/'labels.json', {'captures/batch/0.png':number})
    save_ignore_regions(root, regions)


class IgnoreRegionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_icon_changes_do_not_change_digit_features_or_source(self):
        region = ((0,0,.22,1),)
        for number in (1,7,8,99,100):
            base = digits(number,60)
            bright = base.copy()
            bright[5:75,5:40] = 255
            original = bright.copy()
            a,b = feature(base,region),feature(bright,region)
            self.assertIsNotNone(a)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(bright,original)
        self.assertIsNone(feature(base,((0,0,1,1),)))

    def test_separate_template_rules_and_legacy_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            first,second = Path(tmp)/'first',Path(tmp)/'second'
            image = digits(8)
            library(first,8,image,((0,0,.22,1),))
            library(second,8,image,((.9,0,1,1),))
            matcher = SegmentMatcher([first,second])
            # Identical processed samples must retain their distinct masks.
            self.assertEqual(len(matcher.templates),2)
            candidate = image.copy();candidate[:,:40]=255
            self.assertEqual(matcher.extract_soc(candidate)[0],8)
            candidate = image.copy();candidate[:,185:]=255
            self.assertEqual(matcher.extract_soc(candidate)[0],8)
            (second/'recognition.json').unlink()
            self.assertEqual(load_ignore_regions(second),())
            self.assertEqual(SegmentMatcher(second).extract_soc(image)[0],8)

    def test_package_roundtrip_and_import_conflict_no_partial_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            regions = ((0,0,.22,1),)
            library(root/'source',7,digits(7),regions)
            export_template_package(root/'source',root/'template.zip','icons')
            with patch('core.segment_dataset.data_home',return_value=root/'home'):
                imported,count = import_template_package(root/'template.zip')
            self.assertEqual(count,1)
            self.assertEqual(load_ignore_regions(imported),regions)
            self.assertEqual(SegmentMatcher(imported).extract_soc(digits(7))[0],7)
            library(root/'other',1,digits(1))
            before = list((root/'other'/'captures').iterdir())
            with self.assertRaises(ValueError):import_dataset(root/'source',root/'other')
            self.assertEqual(list((root/'other'/'captures').iterdir()),before)
            self.assertEqual(load_ignore_regions(root/'other'),())
            save_ignore_regions(root/'empty',regions)
            with self.assertRaises(ValueError):import_dataset(root/'other',root/'empty')
            self.assertEqual(load_ignore_regions(root/'empty'),regions)
            self.assertFalse((root/'empty'/'captures').exists())

    def test_invalid_configuration_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as tmp:
            for regions in ([[0,0,2,1]],[[0,0,float('nan'),1]],[[True,0,1,1]],[[0,0,0,1]]):
                with self.assertRaises(ValueError):save_ignore_regions(tmp,regions)
            Path(tmp,'recognition.json').write_text('{bad')
            with self.assertRaises(ValueError):load_ignore_regions(tmp)

    def test_sample_editor_save_cancel_and_rotated_capture_entry(self):
        from ui.segment_dialog import SegmentDatasetDialog
        from core.segment_dataset import crop_digits
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'template'
            original=digits(8);library(source,8,original)
            settings=QSettings(str(root/'ui.ini'),QSettings.IniFormat)
            with patch('core.segment_dataset.data_home',return_value=root), \
                 patch('ui.segment_dialog.data_home',return_value=root), \
                 patch('ui.segment_dialog.QSettings',return_value=settings):
                window=SegmentDatasetDialog({1:[str(source)]})
                try:
                    with patch('ui.segment_dialog.IgnoreRegionsDialog') as editor:
                        editor.return_value.exec_.return_value=QDialog.Accepted
                        editor.return_value.canvas.regions=[(0,0,.22,1)]
                        window.edit_ignore_regions(True)
                        np.testing.assert_array_equal(editor.call_args.args[0],original)
                        self.assertEqual(load_ignore_regions(source),((0,0,.22,1),))
                        editor.return_value.exec_.return_value=QDialog.Rejected
                        editor.return_value.canvas.regions=[]
                        window.edit_ignore_regions(True)
                        self.assertEqual(load_ignore_regions(source),((0,0,.22,1),))
                        window.cap=MagicMock()
                        window.cap.read.return_value=(True,original)
                        window.cap.get.return_value=0
                        window.preview()
                        window.canvas.rois['soc_1']=(0,0,200,80)
                        window.canvas.rotations['soc_1']=90
                        window.edit_ignore_regions(False)
                        np.testing.assert_array_equal(editor.call_args.args[0],
                            crop_digits(original,(0,0,200,80),90))
                        calls=editor.call_count
                        window.cap.read.return_value=(False,None)
                        window.preview()
                        with patch('ui.segment_dialog.QMessageBox.warning') as warning:
                            window.edit_ignore_regions(False)
                        warning.assert_called_once()
                        self.assertEqual(editor.call_count,calls)
                        self.assertIsNone(window.canvas._frame_buffer)
                    self.assertIn('忽略区域：1 个',window.library_info.text())
                finally:window.close()

    def test_mouse_coordinates_resizing_undo_clear_and_preview(self):
        image = digits(8)
        canvas = IgnoreRegionsCanvas(image)
        canvas.resize(600,400);canvas.show();self.app.processEvents()
        try:
            rect=canvas.image_rect()
            start=QPoint(round(rect.x()),round(rect.y()))
            end=QPoint(round(rect.x()+rect.width()*.2),round(rect.y()+rect.height()))
            QTest.mousePress(canvas,Qt.LeftButton,pos=start)
            QTest.mouseRelease(canvas,Qt.LeftButton,pos=end)
            np.testing.assert_allclose(canvas.regions,[(0,0,.2,1)],atol=.002)
            canvas.resize(400,500)
            self.assertEqual(len(canvas.regions),1)
            canvas.undo();self.assertEqual(canvas.regions,[])
            canvas.regions=[(0,0,.2,1)];canvas.clear_regions()
            self.assertEqual(canvas.regions,[])
        finally:canvas.close()
        dialog=IgnoreRegionsDialog(image,((0,0,1,1),))
        try:
            self.assertFalse(dialog.buttons.button(QDialogButtonBox.Save).isEnabled())
            dialog.canvas.clear_regions()
            self.assertTrue(dialog.buttons.button(QDialogButtonBox.Save).isEnabled())
        finally:dialog.close()


if __name__ == '__main__':unittest.main()
