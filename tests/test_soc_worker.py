import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
import tempfile
from datetime import time
from pathlib import Path
from unittest.mock import patch

import onnxruntime
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication

from core.video_worker import VideoAnalysisWorker
from core.segment_dataset import SEGMENT_MODE, write_image, write_json


def frame_number(image):
    return int(image[0,0,0]) + int(image[0,0,1])*256


class SOCWorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def test_two_units_reject_outliers_recover_and_keep_actual_first_crops(self):
        values=([1]*4+[2]*4+[99]*12+[3]*10+[None]*150+[30]*10+
                [None]*220+[99]*10)
        second_values=[80]*8+[81]*12+[79]*8+[78]*252+[None]*120+[60]*20

        class Capture:
            def __init__(self,*args):self.pos=0
            def isOpened(self):return True
            def get(self,prop):
                return {cv2.CAP_PROP_FRAME_COUNT:len(values),cv2.CAP_PROP_FPS:2,
                        cv2.CAP_PROP_POS_FRAMES:self.pos,
                        cv2.CAP_PROP_POS_MSEC:max(0,self.pos-1)*500}.get(prop,0)
            def set(self,prop,value):
                self.pos=int(value/500) if prop==cv2.CAP_PROP_POS_MSEC else int(value)
            def read(self):
                if self.pos>=len(values):return False,None
                image=np.array([[[self.pos%256,self.pos//256,0]]],dtype=np.uint8)
                self.pos+=1
                return True,image
            def release(self):pass

        class OCR:
            def extract_soc(self,*args,**kwargs):raise AssertionError('Used ordinary SOC OCR')
            def extract_time(self,image):
                sec=frame_number(image)//2
                minutes,seconds=divmod(sec,60)
                return f'00:{minutes:02d}:{seconds:02d}',time(0,minutes,seconds),1
            def extract_capacity(self,image,**kwargs):
                cap=frame_number(image)/1000
                return cap,str(cap),1

        class Matcher:
            labels=set(range(101))
            last_reason='未识别'
            def __init__(self,path):self.unit=1 if path=='a' else 2
            def extract_soc(self,image):
                val=(values if self.unit==1 else second_values)[frame_number(image)]
                return val,1

        roi=(0,0,1,1)
        worker=VideoAnalysisWorker('fake',roi_soc_1=roi,roi_time_1=roi,roi_cap_1=roi,
            roi_soc_2=roi,roi_time_2=roi,roi_cap_2=roi,digit_type=SEGMENT_MODE,
            modes={1:'charge',2:'discharge'},segment_paths={1:'a',2:'b'})
        results=[];errors=[];pending=[];emitted=[]
        worker.finished_signal.connect(lambda a,b:results.append((a,b)))
        worker.error_signal.connect(errors.append)
        worker.review_pending_signal.connect(lambda u,r:pending.append((u,r)))
        worker.point_detected_signal.connect(lambda u,r:emitted.append((u,r)))
        with patch('core.video_worker.cv2.VideoCapture',Capture),\
             patch('core.video_worker.OCREngine',OCR),\
             patch('core.video_worker.SegmentMatcher',Matcher):
            worker.run()
        self.assertEqual(errors,[])
        first,second=results[0]
        self.assertEqual(set(first),{1,2,3,30,99})
        self.assertEqual(set(second),{80,79,78,60})
        self.assertTrue(any(u==1 and r['soc']==99 and r['video_sec']==4 for u,r in pending))
        self.assertEqual(first[30]['video_sec'],90)
        self.assertEqual(first[30]['time_str'],'00:01:30')
        self.assertEqual(first[30]['capacity'],.180)
        self.assertEqual(first[99]['video_sec'],205)
        self.assertEqual(first[99]['capacity'],.410)
        self.assertEqual(second[60]['video_sec'],200)
        self.assertEqual(second[60]['capacity'],.400)
        self.assertEqual(second[78]['video_sec'],14.5)
        self.assertEqual(second[78]['capacity'],.029)
        self.assertTrue(any(u==2 and r['soc']==81 and r['video_sec']==4 for u,r in pending))
        self.assertNotIn(99,worker.segment_diagnostics[1]['pending'])
        self.assertEqual([r['video_sec'] for u,r in emitted if u==1 and r['soc']==99],[205])

    def test_real_video_and_real_template_matcher_quarantine_a_large_jump(self):
        class OCR:
            def extract_soc(self,*args,**kwargs):raise AssertionError('Used ordinary SOC OCR')
            def extract_time(self,image):
                sec=int(image[0,0,0])//10
                return f'00:00:{sec:02d}',time(0,0,sec),1
            def extract_capacity(self,image,**kwargs):
                cap=int(image[0,0,0])/1000
                return cap,str(cap),1

        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            pictures={}
            for value in (1,2,3,99):
                image=np.zeros((80,120,3),np.uint8)
                cv2.putText(image,str(value),(5,60),cv2.FONT_HERSHEY_SIMPLEX,1.7,(255,255,255),3)
                pictures[value]=image
                write_image(folder/f'{value}.png',image)
            write_json(folder/'labels.json',{f'{v}.png':v for v in pictures})
            path=folder/'outlier.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),10,(300,80))
            self.assertTrue(writer.isOpened())
            for i,value in enumerate([1]*10+[2]*10+[99]*10+[3]*10):
                frame=np.zeros((80,300,3),np.uint8)
                frame[:,:120]=pictures[value]
                frame[:,120:,0]=i
                writer.write(frame)
            writer.release()
            worker=VideoAnalysisWorker(str(path),roi_soc_1=(0,0,120,80),
                roi_time_1=(120,0,80,80),roi_cap_1=(200,0,80,80),
                digit_type=SEGMENT_MODE,mode='charge',sample_interval_sec=.2,
                segment_paths={1:str(folder)})
            results=[];errors=[];pending=[]
            worker.finished_signal.connect(lambda a,b:results.append(a))
            worker.error_signal.connect(errors.append)
            worker.review_pending_signal.connect(lambda u,r:pending.append(r))
            with patch('core.video_worker.OCREngine',OCR):
                worker.run()
            self.assertEqual(errors,[])
            self.assertEqual(set(results[0]),{1,2,3})
            self.assertTrue(any(r['soc']==99 for r in pending))
            self.assertAlmostEqual(results[0][3]['video_sec'],3,places=3)
            self.assertEqual(results[0][3]['capacity'],.030)

            # At 1 FPS, dense sampling can decode the SAME frame several times.
            # A single frame showing 2 must not satisfy the two-frame debounce.
            path=folder/'low-fps.avi'
            writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'FFV1'),1,(300,80))
            self.assertTrue(writer.isOpened())
            for i,value in enumerate([1,1,2,1,1]):
                frame=np.zeros((80,300,3),np.uint8)
                frame[:,:120]=pictures[value]
                frame[:,120:,0]=i
                writer.write(frame)
            writer.release()
            worker=VideoAnalysisWorker(str(path),roi_soc_1=(0,0,120,80),
                roi_time_1=(120,0,80,80),roi_cap_1=(200,0,80,80),
                digit_type=SEGMENT_MODE,mode='charge',sample_interval_sec=.2,
                segment_paths={1:str(folder)})
            results=[];errors=[]
            worker.finished_signal.connect(lambda a,b:results.append(a))
            worker.error_signal.connect(errors.append)
            with patch('core.video_worker.OCREngine',OCR):
                worker.run()
            self.assertEqual(errors,[])
            self.assertEqual(set(results[0]),{1})


if __name__=='__main__':
    unittest.main()
