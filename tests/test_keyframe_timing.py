import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import cv2
import numpy as np
from core.keyframe_timing import refine_keyframe
from core.ocr_engine import OCREngine
from core.video_worker import VideoAnalysisWorker
from core.record_evidence import load_snapshot
from core.segment_dataset import SEGMENT_MODE


def picture(value):
    image = np.zeros((80,160,3),np.uint8)
    cv2.putText(image,str(value),(5,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(255,255,255),3)
    return image


class KeyframeTimingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/'timing.avi'

    def video(self, values):
        writer=cv2.VideoWriter(str(self.path),cv2.VideoWriter_fourcc(*'FFV1'),10,(160,80))
        self.assertTrue(writer.isOpened())
        for value in values: writer.write(picture(value))
        writer.release()

    def refine(self, sec=5., **kwargs):
        return refine_keyframe(str(self.path),(0,0,160,80),0,99,picture(99),sec,int(sec*10),**kwargs)

    def test_returns_first_frame_not_confirmation_frame(self):
        self.video([100]*15+[99]*45)
        result=self.refine(competing=[(100,picture(100))])
        self.assertEqual(result['frame_index'],15)
        self.assertAlmostEqual(result['video_sec'],1.5)
        self.assertEqual(result['timing_original_sec'],5.)
        self.assertTrue(np.array_equal(result['frame'],picture(99)))

    def test_one_matching_frame_does_not_move_record(self):
        self.video([100]*15+[99]+[100]*34+[99]*10)
        self.assertIsNone(self.refine(competing=[(100,picture(100))]))

    def test_bounds_prevent_crossing_previous_event_or_search_window(self):
        self.video([99]*20+[100]*20+[99]*20)
        self.assertAlmostEqual(self.refine(lower_bound=3.)['video_sec'],4.)
        self.assertAlmostEqual(self.refine(max_lookback=.5)['video_sec'],4.5)

    def test_ambiguous_references_do_not_change_timestamp(self):
        self.video([99]*60)
        self.assertIsNone(self.refine(competing=[(98,picture(99))]))

    def test_cancel_does_not_open_video_and_missing_source_is_safe(self):
        with patch('core.keyframe_timing.cv2.VideoCapture') as capture:
            self.assertIsNone(self.refine(cancelled=lambda:True))
            capture.assert_not_called()
        self.assertIsNone(self.refine())

    def test_worker_reads_meter_from_recovered_frame(self):
        from collections import Counter
        from datetime import time
        from core.soc_tracker import SegmentSOCTracker
        class Engine:
            def extract_time(self, image):
                number=int(image[0,0,0]); return f'00:00:{number:02}',time(0,0,number),1.
            def extract_capacity(self, image, **kwargs):
                number=int(image[0,0,0])/100; return number,str(number),1.
        class Matcher:
            last_reason='matched'
            def extract_soc(self,image):return 99,1.
        worker=VideoAnalysisWorker('fake',roi_soc_1=(0,0,10,10),roi_time_1=(10,0,10,10),
            roi_cap_1=(20,0,10,10),timing_refine={1:True},digit_type=SEGMENT_MODE)
        worker.ocr_engine=Engine();worker.segment_diagnostics[1]={'rejected':Counter()}
        worker.evidence_directory=Path(self.tmp.name)/'evidence'
        tracker=SegmentSOCTracker('discharge',2,1.)
        current=np.full((10,30,3),50,np.uint8)
        earlier=np.full((10,30,3),15,np.uint8)
        record={}
        recovered=dict(frame=earlier,video_sec=1.5,frame_index=15,timing_original_sec=5.,
                       timing_search_start_sec=0.,timing_method='visual_backtrack')
        with patch('core.video_worker.refine_keyframe',return_value=recovered):
            worker._decoded_frame_index=50
            worker._process_segment_soc(1,current,5.,Matcher(),tracker,record,None)
            worker._decoded_frame_index=60
            worker._process_segment_soc(1,current,6.,Matcher(),tracker,record,None)
        self.assertEqual(record[99]['time_str'],'00:00:15')
        self.assertEqual(record[99]['capacity'],.15)
        self.assertEqual(record[99]['video_sec'],1.5)
        self.assertEqual(record[99]['frame_index'],15)
        self.assertTrue(np.array_equal(load_snapshot(record[99]),earlier))


class MeterColumnTests(unittest.TestCase):
    def extract(self, rows):
        engine=OCREngine.__new__(OCREngine)
        engine.preprocess_soc_crop=lambda image:image
        boxes=[]
        for row, texts in enumerate(rows):
            for col,text in enumerate(texts):
                x,y=col*50,row*30
                boxes.append([[[x,y],[x+45,y],[x+45,y+20],[x,y+20]],text,.99])
        engine.ocr=lambda image:(boxes,None)
        return engine.extract_capacity(SimpleNamespace(size=1))

    def test_header_total_and_window_are_not_concatenated(self):
        self.assertEqual(self.extract([['容量Ah'],['0.0270'],['0.0270']])[0],.027)
        self.assertEqual(self.extract([['容量mAh'],['27.0'],['1.0']])[0],.027)

    def test_malformed_total_does_not_silently_choose_window(self):
        self.assertIsNone(self.extract([['容量Ah'],['0','0270'],['0.0270']])[0])
