import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime  # DLL ordering on Windows
import tempfile
import unittest
from pathlib import Path
from datetime import time
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from core.segment_dataset import (SEGMENT_MODE, SegmentMatcher, SampleCaptureWorker,
    write_image, write_json, read_json, samples, set_label)
from core.video_worker import VideoAnalysisWorker


class SegmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_labelled_only_and_ambiguous_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.zeros((70, 100, 3), np.uint8)
            cv2.putText(image, '25', (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255,255,255), 3)
            write_image(root/'sample.png', image)
            self.assertEqual(SegmentMatcher(root).extract_soc(image), (None, 0.0))
            set_label(root, 'sample.png', 25)
            self.assertEqual(SegmentMatcher(root).extract_soc(image)[0], 25)
            self.assertIsNone(SegmentMatcher(root).extract_soc(np.zeros_like(image))[0])
            write_image(root/'ambiguous.png', image)
            set_label(root, 'ambiguous.png', 26)
            self.assertIsNone(SegmentMatcher(root).extract_soc(image)[0])
            set_label(root, 'ambiguous.png', None)
            set_label(root, 'sample.png', 'invalid')
            self.assertIsNone(SegmentMatcher(root).extract_soc(image)[0])

    def test_real_video_capture_metadata_cancel_and_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            source = str(Path(folder)/'test.avi')
            writer = cv2.VideoWriter(source, cv2.VideoWriter_fourcc(*'MJPG'), 10, (80,60))
            self.assertTrue(writer.isOpened())
            for i in range(30):
                image=np.zeros((60,80,3),np.uint8)
                cv2.putText(image,str(i),(10,40),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                writer.write(image)
            writer.release()
            root=Path(folder)/'中文样本库'
            worker=SampleCaptureWorker(source,root,(10,10,40,30),270,2,.5,0,2)
            errors=[]
            worker.failed.connect(errors.append)
            worker.run()
            self.assertEqual(errors,[])
            rows=samples(root)
            self.assertEqual(len(rows),4)
            self.assertEqual([r['frame_index'] for r in rows],[0,5,10,15])
            self.assertTrue(all(r['unit']==2 for r in rows))
            from core.segment_dataset import read_image
            self.assertEqual(read_image(root/rows[0]['key']).shape[:2],(40,30))
            self.assertFalse((root/'labels.json').exists())
            cancelled=SampleCaptureWorker(source,root,(10,10,40,30))
            cancelled.stop()
            cancelled.run()
            manifests=list(root.glob('captures/*/manifest.json'))
            self.assertEqual({read_json(p,{})['status'] for p in manifests},{'complete','cancelled'})

    def test_both_workers_use_templates_and_timestamp_first_confirmed_candidate(self):
        sequence=[10,10,0,0,9,None,9,9] + [v for v in range(8,-1,-1) for _ in range(2)]
        class Capture:
            def __init__(self,*a):self.pos=0
            def isOpened(self):return True
            def get(self, prop):
                return {cv2.CAP_PROP_FRAME_COUNT:len(sequence),cv2.CAP_PROP_FPS:1,
                    cv2.CAP_PROP_POS_FRAMES:self.pos,cv2.CAP_PROP_POS_MSEC:max(0,self.pos-1)*1000}.get(prop,0)
            def set(self, prop, value):
                self.pos=int(value/1000) if prop==cv2.CAP_PROP_POS_MSEC else int(value)
            def read(self):
                if self.pos>=len(sequence):return False,None
                image=np.full((1,1,3),self.pos,np.uint8);self.pos+=1
                return True,image
            def release(self):pass
        class OCR:
            def extract_soc(self,*a,**kw):raise AssertionError('Segment mode invoked ordinary SOC OCR')
            def extract_time(self,image):
                second=int(image[0,0,0])
                return f'00:00:{second:02d}',time(0,0,second),1
            def extract_capacity(self,image,**kw):
                value=float(image[0,0,0])
                return value,str(value),1
        class Matcher:
            labels=set(range(101))
            def __init__(self,path):self.unit=1 if path=='a' else 2
            def extract_soc(self,image):
                value=sequence[int(image[0,0,0])]
                return (100-value if self.unit==2 and value is not None else value),1
        roi=(0,0,1,1)
        worker=VideoAnalysisWorker('fake',roi_soc_1=roi,roi_time_1=roi,roi_cap_1=roi,
            roi_soc_2=roi,roi_time_2=roi,roi_cap_2=roi,digit_type=SEGMENT_MODE,
            modes={1:'discharge',2:'charge'},segment_paths={1:'a',2:'b'})
        results=[];errors=[]
        worker.finished_signal.connect(lambda a,b:results.append((a,b)))
        worker.error_signal.connect(errors.append)
        with patch('core.video_worker.cv2.VideoCapture',Capture),patch('core.video_worker.OCREngine',OCR),patch('core.video_worker.SegmentMatcher',Matcher):
            worker.run()
        self.assertEqual(errors,[])
        for unit,records in enumerate(results[0],1):
            near=9 if unit==1 else 91
            final=0 if unit==1 else 100
            self.assertEqual(set(records),set(range(11)) if unit==1 else set(range(90,101)))
            self.assertEqual(records[near]['video_sec'],4)
            self.assertEqual(records[near]['time_str'],'00:00:04')
            self.assertEqual(records[near]['capacity'],4)
            self.assertEqual(records[final]['video_sec'],24)
            self.assertEqual(records[final]['time_str'],'00:00:24')

    def test_blinking_digit_seen_only_between_whole_seconds(self):
        sequence=[None,5,None,5,None,5]
        class Capture:
            def __init__(self,*args):self.pos=0
            def isOpened(self):return True
            def get(self,prop):
                return {cv2.CAP_PROP_FRAME_COUNT:len(sequence),cv2.CAP_PROP_FPS:2,
                    cv2.CAP_PROP_POS_FRAMES:self.pos,
                    cv2.CAP_PROP_POS_MSEC:max(0,self.pos-1)*500}.get(prop,0)
            def set(self,prop,value):
                self.pos=int(value/500) if prop==cv2.CAP_PROP_POS_MSEC else int(value)
            def read(self):
                if self.pos>=len(sequence):return False,None
                frame=np.full((1,1,3),self.pos,np.uint8)
                self.pos+=1
                return True,frame
            def release(self):pass
        class OCR:
            def extract_soc(self,*args,**kwargs):raise AssertionError('Ordinary SOC OCR called')
            def extract_time(self,image):return None,None,0
            def extract_capacity(self,image,**kwargs):return float(image[0,0,0]),'',1
        class Matcher:
            labels={5}
            def __init__(self,*args):pass
            def extract_soc(self,image):return sequence[int(image[0,0,0])],1
        roi=(0,0,1,1)
        worker=VideoAnalysisWorker('fake',roi_soc_1=roi,roi_time_1=roi,roi_cap_1=roi,
            digit_type=SEGMENT_MODE,segment_paths={1:'a'},sample_interval_sec=1,debounce_count=2)
        results=[];errors=[]
        worker.finished_signal.connect(lambda first,second:results.append(first))
        worker.error_signal.connect(errors.append)
        with patch('core.video_worker.cv2.VideoCapture',Capture),patch('core.video_worker.OCREngine',OCR),patch('core.video_worker.SegmentMatcher',Matcher):
            worker.run()
        self.assertEqual(errors,[])
        self.assertEqual(results[0][5]['video_sec'],.5)
        self.assertEqual(results[0][5]['capacity'],1.0)
        self.assertEqual(results[0][5]['time_str'],'--')
        self.assertIsNone(results[0][5]['time_obj'])

    def test_unreadable_time_is_visible_and_capacity_can_still_export(self):
        from ui.main_window import MainWindow
        with patch('ui.main_window.ResourceMonitor.start'):
            window=MainWindow()
        record={'soc':99,'time_str':'--','time_obj':None,
                'capacity':0.0806,'video_sec':120.5,'is_target':True}
        window.on_worker_point_detected(1,record)
        self.assertIn('待核对',window.table_1.item(0,4).text())
        exported=window.get_table_data_for_export(1)
        self.assertEqual(exported[99]['capacity'],0.0806)
        self.assertIsNone(exported[99]['time_obj'])
        self.assertEqual(exported[99]['video_sec'],120.5)
        window.close()

    def test_only_lcd_and_segment_modes_are_visible(self):
        from ui.main_window import MainWindow
        with patch('ui.main_window.ResourceMonitor.start'):
            window=MainWindow()
        self.assertEqual(window.cb_digit_type.count(),2)
        self.assertEqual(window.cb_digit_type.itemData(0),1)
        self.assertEqual(window.cb_digit_type.itemData(1),SEGMENT_MODE)
        self.assertFalse(window.btn_segment_dataset.isEnabled())
        with patch('ui.main_window.SegmentDatasetDialog',side_effect=AssertionError('library opened')):
            window.open_segment_dataset()
        window.cb_digit_type.setCurrentIndex(1)
        self.assertTrue(window.btn_segment_dataset.isEnabled())
        window.cb_digit_type.setCurrentIndex(0)
        self.assertFalse(window.btn_segment_dataset.isEnabled())
        window.close()
        worker=VideoAnalysisWorker('unused',digit_type=1,modes={1:'charge',2:'discharge'})
        self.assertTrue(worker._is_valid_transition(15,13,1))
        self.assertFalse(worker._is_valid_transition(20,13,1))
        self.assertTrue(worker._is_valid_transition(13,15,2))
        self.assertFalse(worker._is_valid_transition(20,15,2))

    def test_pending_soc_can_be_previewed_but_not_exported_or_occupy_record(self):
        from ui.main_window import MainWindow
        with patch('ui.main_window.ResourceMonitor.start'):
            window=MainWindow()
        try:
            window.on_worker_review_pending(1, {'soc':99, 'video_sec':4,
                'reason':'短时间内电量大幅跳变'})
            self.assertNotIn(99,window.extracted_records_1)
            self.assertNotIn(99,window.get_table_data_for_export(1))
            self.assertIn('电量跳变待核对',window.table_1.item(0,4).text())
            self.assertEqual(window.table_1.item(0,3).text(),'4.0s')
            record={'soc':99,'time_str':'00:03:20','time_obj':time(0,3,20),
                    'capacity':1.2,'video_sec':200,'is_target':True}
            window.on_worker_point_detected(1,record)
            self.assertEqual(window.get_table_data_for_export(1)[99]['video_sec'],200)
            self.assertNotIn('待核对',window.table_1.item(0,4).text())
            window.on_worker_review_pending(1, {'soc':99,'video_sec':201,'reason':'反向'})
            self.assertEqual(window.table_1.item(0,3).text(),'200.0s')
        finally:
            window.close()

    def test_completion_reports_pending_percentages_outside_export_targets(self):
        from types import SimpleNamespace
        from ui.main_window import MainWindow
        from core.excel_filler import get_target_soc_list
        with patch('ui.main_window.ResourceMonitor.start'):
            window=MainWindow()
        try:
            window.worker=SimpleNamespace(
                target_socs={u:set(get_target_soc_list(window.unit_modes[u].currentData())) for u in (1,2)},
                segment_diagnostics={1:{'labels':list(range(101)), 'rejected':{},
                    'pending':{14:{'soc':14,'video_sec':123,'reason':'反向'}}}})
            with patch('ui.main_window.QMessageBox.information') as message:
                window.on_worker_finished({}, {})
            self.assertIn('14%（123.0s）',message.call_args.args[2])
            self.assertNotIn(14,window.get_table_data_for_export(1))
        finally:
            window.worker=None
            window.close()


if __name__ == '__main__':
    unittest.main()
