import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime
import tempfile
import unittest
from pathlib import Path
from datetime import time
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog
from PyQt5.QtCore import QSettings
from core.segment_dataset import mask_ocr_regions, SEGMENT_MODE
from core.video_worker import VideoAnalysisWorker
from ui.main_window import MainWindow
from ui.ignore_regions_dialog import IgnoreRegionsDialog


class LCDIgnoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_light_dark_backgrounds_and_non_destructive_mask(self):
        for background,foreground in [(15,240),(240,15)]:
            clear=np.full((80,120,3),background,np.uint8)
            cv2.putText(clear,'8',(45,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(foreground,)*3,3)
            icon=clear.copy();icon[5:65,2:20]=foreground
            before=icon.copy()
            np.testing.assert_array_equal(mask_ocr_regions(icon,((0,0,.2,1),)),clear)
            np.testing.assert_array_equal(icon,before)
            self.assertIs(mask_ocr_regions(icon,()),icon)
            self.assertIsNone(mask_ocr_regions(icon,((0,0,1,1),)))
            dialog=IgnoreRegionsDialog(icon,((0,0,.2,1),),lcd_unit=1)
            try:
                self.assertIn('LCD',dialog.windowTitle())
                self.assertIsNotNone(dialog.preview.pixmap())
            finally:dialog.close()

    def test_unit_settings_save_cancel_clear_restart_and_mode_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=lambda org,app:QSettings(str(Path(tmp)/(app+'.ini')),QSettings.IniFormat)
            with patch('ui.main_window.QSettings',side_effect=settings),patch('ui.main_window.ResourceMonitor.start'):
                w=MainWindow()
                try:
                    w.current_frame_cv=np.full((80,200,3),240,np.uint8)
                    w.set_preview_position('test.mp4',0)
                    w.canvas.rois['soc_1']=(0,0,100,80)
                    w.canvas.rois['soc_2']=(100,0,100,80)
                    with patch('ui.main_window.IgnoreRegionsDialog') as editor:
                        editor.return_value.exec_.return_value=QDialog.Accepted
                        editor.return_value.canvas.regions=[(0,0,.2,1)]
                        w.edit_lcd_ignore_regions(1)
                        self.assertEqual(editor.call_args.kwargs['lcd_unit'],1)
                        self.assertEqual(w.lcd_ignore_regions[1],((0,0,.2,1),))
                        self.assertEqual(w.lcd_ignore_regions[2],())
                        editor.return_value.exec_.return_value=QDialog.Rejected
                        editor.return_value.canvas.regions=[]
                        w.edit_lcd_ignore_regions(1)
                        self.assertEqual(w.lcd_ignore_regions[1],((0,0,.2,1),))
                        w.cb_digit_type.setCurrentIndex(1)
                        calls=editor.call_count;w.edit_lcd_ignore_regions(1)
                        self.assertEqual(editor.call_count,calls)
                        self.assertTrue(w.lcd_ignore_buttons[1].isHidden())
                finally:w.close()
                w=MainWindow()
                try:
                    self.assertEqual(w.lcd_ignore_regions[1],((0,0,.2,1),))
                    self.assertEqual(w.lcd_ignore_regions[2],())
                    w.current_frame_cv=np.full((80,200,3),240,np.uint8)
                    w.set_preview_position('test.mp4',0)
                    w.canvas.rois['soc_1']=(0,0,100,80)
                    with patch('ui.main_window.IgnoreRegionsDialog') as editor:
                        editor.return_value.exec_.return_value=QDialog.Accepted
                        editor.return_value.canvas.regions=[]
                        w.edit_lcd_ignore_regions(1)
                    self.assertEqual(w.lcd_ignore_regions[1],())
                    self.assertEqual(w.lcd_settings.value('unit_1'),'[]')
                    w.video_cap=MagicMock()
                    w.video_cap.read.return_value=(False,None)
                    w.load_frame(100)
                    with patch('ui.main_window.IgnoreRegionsDialog') as editor, \
                         patch('ui.main_window.QMessageBox.warning') as warning:
                        w.edit_lcd_ignore_regions(1)
                        editor.assert_not_called();warning.assert_called_once()
                finally:w.close()

    def test_preview_and_worker_receive_identical_rotated_masked_crops(self):
        captured=[]
        class OCR:
            def __init__(self,**kwargs):pass
            def extract_soc(self,image,**kwargs):
                captured.append(image.copy());return 50,1.
            def extract_time(self,image):return '00:00:05',time(0,0,5),1.
            def extract_capacity(self,image,**kwargs):return .5,'0.5',1.
        image=np.full((80,200,3),230,np.uint8)
        cv2.putText(image,'8',(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.4,(10,10,10),3)
        image[10:40,5:15]=0
        regions={1:((0,0,.2,1),),2:()}
        rois={f'{kind}_{unit}':(0 if unit==1 else 100,0,100,80)
              for unit in (1,2) for kind in ('soc','time','cap')}
        with tempfile.TemporaryDirectory() as tmp,patch('ui.main_window.ResourceMonitor.start'), \
             patch('ui.main_window.QSettings',side_effect=lambda org,app:QSettings(str(Path(tmp)/(app+'.ini')),QSettings.IniFormat)):
            w=MainWindow()
            try:
                w.current_frame_cv=image.copy();w.canvas.rois.update(rois)
                w.canvas.rotations['soc_1']=90
                w.lcd_ignore_regions=regions
                w.ocr_tester=OCR()
                with patch('ui.main_window.QMessageBox.information'):w.test_current_frame_ocr()
                self.assertEqual(len(captured),2)
                expected=[c.copy() for c in captured];captured.clear()
            finally:w.close()
            path=Path(tmp)/'lcd.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),10,(200,80))
            for _ in range(30):writer.write(image)
            writer.release()
            worker=VideoAnalysisWorker(str(path),roi_soc_1=rois['soc_1'],roi_soc_2=rois['soc_2'],
                roi_time_1=rois['time_1'],roi_time_2=rois['time_2'],roi_cap_1=rois['cap_1'],roi_cap_2=rois['cap_2'],
                digit_type=1,rot_soc_1=90,lcd_ignore_regions=regions,sample_interval_sec=.5)
            worker.evidence_directory=Path(tmp)/'evidence'
            errors=[];worker.error_signal.connect(errors.append)
            with patch('core.video_worker.OCREngine',OCR):worker.run()
            self.assertEqual(errors,[])
            self.assertGreaterEqual(len(captured),2)
            for i,crop in enumerate(captured):np.testing.assert_array_equal(crop,expected[i%2])


if __name__=='__main__':unittest.main()
