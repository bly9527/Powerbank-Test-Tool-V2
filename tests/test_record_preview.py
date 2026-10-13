import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import onnxruntime
import cv2
import numpy as np
import tempfile
import unittest
from pathlib import Path
from datetime import time
from unittest.mock import patch
from PyQt5.QtWidgets import QApplication
from core.record_evidence import save_evidence, load_snapshot, locate_record
from core.video_worker import VideoAnalysisWorker

class RecordPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.app=QApplication.instance() or QApplication([])

    def test_lcd_keeps_candidate_pixels_and_actual_timestamp(self):
        # Mock OCR only: the decoder, crop and evidence pipeline are real.
        class OCR:
            def __init__(self,**kw):pass
            def extract_soc(self,im,**kw):return (1 if int(im[0,0,0])<20 else 2),1
            def extract_time(self,im):
                n=int(im[0,0,0]);return f'00:00:{n:02d}',time(0,0,n),1
            def extract_capacity(self,im,**kw):return int(im[0,0,0])/100,'',1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frames.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),10,(80,40))
            frames=[]
            for i in range(40):
                f=np.full((40,80,3),i,np.uint8);frames.append(f);writer.write(f)
            writer.release()
            worker=VideoAnalysisWorker(str(path),roi_soc_1=(0,0,20,40),roi_time_1=(20,0,20,40),roi_cap_1=(40,0,20,40),active_units=[1],mode='charge',digit_type=1,sample_interval_sec=.5)
            worker.evidence_directory=Path(tmp)/'evidence';results=[];errors=[]
            worker.finished_signal.connect(lambda a,b:results.append(a));worker.error_signal.connect(errors.append)
            with patch('core.video_worker.OCREngine',OCR):worker.run()
            self.assertEqual(errors,[])
            for soc,index in [(1,0),(2,20)]:
                rec=results[0][soc]
                self.assertEqual(rec['frame_index'],index)
                self.assertAlmostEqual(rec['video_sec'],index/10)
                self.assertEqual(rec['capacity'],index/100)
                np.testing.assert_array_equal(load_snapshot(rec),frames[index])
                cap=cv2.VideoCapture(str(path))
                try:
                    image,verified=locate_record(cap,rec)
                    self.assertTrue(verified);np.testing.assert_array_equal(image,frames[index])
                finally:cap.release()

    def test_snapshot_click_does_not_seek_and_export_keeps_precision(self):
        from ui.main_window import MainWindow
        with tempfile.TemporaryDirectory() as tmp, patch('ui.main_window.ResourceMonitor.start'):
            window=MainWindow()
            try:
                im=np.full((40,80,3),72,np.uint8)
                rec=dict(soc=99,time_str='01:45:03',time_obj=time(1,45,3),capacity=3.4021,is_target=True)
                rec.update(save_evidence(im,'missing.mp4',6354.033333,143411,tmp))
                window.on_worker_point_detected(1,rec)
                with patch('ui.main_window.cv2.VideoCapture',side_effect=AssertionError('snapshot must not seek')):
                    window.on_table_cell_clicked(0,3,1)
                np.testing.assert_array_equal(window.current_frame_cv,im)
                exported=window.get_table_data_for_export(1)[99]
                self.assertEqual(exported['video_sec'],6354.033333)
                self.assertEqual(exported['snapshot_path'],rec['snapshot_path'])
                window.locate_selected_record()
                self.assertIn('不存在',window.lbl_status.text())
                np.testing.assert_array_equal(window.current_frame_cv,im)
            finally:window.dirty_units.clear();window.close()

    def test_wrong_seek_is_verified_by_pixels_and_legacy_is_not_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected=np.full((12,12,3),7,np.uint8)
            rec=save_evidence(expected,'source.mp4',10,100,tmp)
            class Capture:
                def set(self,*args):self.i=0
                def read(self):
                    self.i+=1
                    return (True,expected.copy()) if self.i==3 else (True,np.zeros_like(expected))
                def get(self,prop):return 10000
            cap=Capture()
            with patch('core.record_evidence.read_frame_at',return_value=(True,np.zeros_like(expected))):
                frame,verified=locate_record(cap,rec)
                self.assertTrue(verified);np.testing.assert_array_equal(frame,expected)
                frame,verified=locate_record(cap,{'video_sec':10})
                self.assertFalse(verified)

    def test_identical_picture_at_wrong_timestamp_is_not_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            expected=np.full((12,12,3),7,np.uint8)
            rec=save_evidence(expected,'source.mp4',10,100,tmp)
            class Capture:
                def set(self,*args):pass
                def read(self):return False,None
                def get(self,prop):return 25 if prop==cv2.CAP_PROP_FPS else 20000
            with patch('core.record_evidence.read_frame_at',return_value=(True,expected.copy())):
                self.assertEqual(locate_record(Capture(),rec),(None,False))

    def test_archived_snapshot_reloads_without_mixing_new_analysis_rows(self):
        from ui.main_window import MainWindow
        from core.project_archive import create_project, ProjectArchive
        template=next(Path(__file__).resolve().parents[1].glob('PB519*.xlsx'))
        with tempfile.TemporaryDirectory() as tmp, patch('core.project_archive.projects_home',return_value=Path(tmp)/'projects'), patch('ui.main_window.ResourceMonitor.start'):
            project=create_project('preview',template)
            group=next(g for g in project.groups['放电时间及曲线'] if g['col_offset']==0)
            frame=np.full((40,80,3),93,np.uint8)
            rec=dict(soc=99,time_str='00:01:23',capacity=.1)
            rec.update(save_evidence(frame,'not-present.mp4',90.033333,2701,Path(tmp)/'live'))
            project.save_results([dict(unit=1,mode='discharge',group=group,records={99:rec})],'not-present.mp4')
            Path(rec['snapshot_path']).unlink()
            window=MainWindow()
            try:
                window.project=ProjectArchive(project.root)
                window.load_excel_groups()
                window.unit_groups[1].setCurrentIndex(0)
                window.refresh_target_tables(restore=True)
                self.assertIn(99,window.extracted_records_1)
                self.assertFalse(window.dirty_units)
                window.on_table_cell_clicked(0,3,1)
                np.testing.assert_array_equal(window.current_frame_cv,frame)
                self.assertEqual(window.get_table_data_for_export(1)[99]['video_sec'],90.033333)
                window.refresh_target_tables()
                self.assertEqual(window.extracted_records_1,{})
                self.assertEqual(window.table_1.item(0,1).text(),'--')
                self.assertFalse(window.btn_locate_record.isEnabled())
            finally:window.dirty_units.clear();window.close()

    def test_step_buttons_use_displayed_video_and_clamp_boundaries(self):
        from ui.main_window import MainWindow
        with tempfile.TemporaryDirectory() as tmp, patch('ui.main_window.ResourceMonitor.start'):
            path=Path(tmp)/'steps.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),10,(80,40))
            for i in range(40):writer.write(np.full((40,80,3),i,np.uint8))
            writer.release()
            window=MainWindow()
            try:
                self.assertFalse(window.btn_seek_forward.isEnabled())
                window.video_path=str(path)
                window.slider_frame.setRange(0,39)
                window.set_preview_position(str(path),0)
                window.sp_seek_seconds.setValue(1.2)
                window.btn_seek_forward.click()
                self.assertAlmostEqual(window._preview_seconds,1.2)
                self.assertEqual(int(window.current_frame_cv[0,0,0]),12)
                self.assertEqual(window.slider_frame.value(),12)
                window.btn_seek_back.click()
                window.btn_seek_back.click()
                self.assertAlmostEqual(window._preview_seconds,0)
                window.sp_seek_seconds.setValue(100)
                window.btn_seek_forward.click()
                self.assertAlmostEqual(window._preview_seconds,3.9)
                self.assertEqual(int(window.current_frame_cv[0,0,0]),39)
                # A displayed archived snapshot may belong to a different video.
                window.video_path=str(Path(tmp)/'other.mp4')
                window.set_preview_position(str(path),1)
                window.sp_seek_seconds.setValue(.5)
                window.btn_seek_forward.click()
                self.assertAlmostEqual(window._preview_seconds,1.5)
                self.assertEqual(int(window.current_frame_cv[0,0,0]),15)
                self.assertFalse(window.canvas.isEnabled())
                self.assertEqual(window.extracted_records_1,{})
            finally:window.dirty_units.clear();window.close()

    def test_wrong_source_does_not_report_exact_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec=save_evidence(np.ones((12,12,3),np.uint8),'source.mp4',10,100,tmp)
            class Capture:
                def set(self,*args):pass
                def read(self):return False,None
                def get(self,prop):return 25 if prop==cv2.CAP_PROP_FPS else 10000
            with patch('core.record_evidence.read_frame_at',return_value=(True,np.zeros((12,12,3),np.uint8))):
                self.assertEqual(locate_record(Capture(),rec),(None,False))

if __name__=='__main__':unittest.main()
