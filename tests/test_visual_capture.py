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
from PyQt5.QtWidgets import QApplication
from core.visual_tracking import VisualTracker, suggested_range
from core.manual_capture import CaptureTask, selected_fields
from core.segment_dataset import write_image
from ui.manual_capture_dialog import ManualCaptureDialog


def digit(text):
    image = np.zeros((70, 150, 3), np.uint8)
    cv2.putText(image, text, (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)
    return image


class VisualTrackingTests(unittest.TestCase):
    def test_changes_need_numeric_evidence_and_keep_first_frame(self):
        tracker = VisualTracker()
        for i in range(4):
            self.assertIsNone(tracker.update(digit('100'),100,digit('100'),i*.5,i))
        result = tracker.update(digit('100'),100,digit('100'),2,4)
        self.assertEqual((result[0],result[2],result[3]),(100,0,0))
        for i in range(5,9):
            self.assertIsNone(tracker.update(digit('99'),None,digit('99'),i*.5,i))
        self.assertEqual(tracker.value,100)
        for i in (9,10):
            self.assertIsNone(tracker.update(digit('98'),98,digit('98'),i*.5,i))
        result = tracker.update(digit('98'),98,digit('98'),5.5,11)
        self.assertEqual((result[0],result[2]),(98,4.5))
        self.assertNotEqual(result[0],99)

    def test_duplicate_dark_and_unstable_frames_cannot_confirm(self):
        tracker = VisualTracker()
        for _ in range(8):
            tracker.update(digit('100'),100,digit('100'),0,0)
        self.assertIsNone(tracker.value)
        blank = np.zeros((70,150,3),np.uint8)
        self.assertIsNone(tracker.update(blank,99,blank,1,1))
        for i in range(2,10):
            self.assertIsNone(tracker.update(digit(str(i)),i,digit(str(i)),i,i))
        self.assertIsNone(tracker.value)

    def test_independent_units_and_gap_reset(self):
        a,b = VisualTracker(),VisualTracker()
        for i in range(5):
            a.update(digit('100'),100,digit('100'),i,i)
            b.update(digit('20'),20,digit('20'),i,i)
        self.assertEqual((a.value,b.value),(100,20))
        a.update(digit('99'),99,digit('99'),5,5)
        self.assertIsNone(a.update(digit('99'),99,digit('99'),20,20))
        self.assertEqual(a.value,100)

    def test_range_uses_correct_video_and_direction(self):
        records={100:{'source_video':'video.mp4','video_sec':10},
                 98:{'source_video':'video.mp4','video_sec':30},
                 97:{'source_video':'other.mp4','video_sec':5}}
        self.assertEqual(suggested_range(records,99,'discharge','video.mp4',100),(10,30))
        self.assertEqual(suggested_range({},99,'discharge','video.mp4',100),(0,100))

    def test_brief_missing_digit_does_not_reset_complete_evidence(self):
        tracker = VisualTracker()
        full = digit('99')
        partial = full.copy()
        partial[:, :35] = 0
        for i in range(9):
            image, value = (full, 99) if i % 2 == 0 else (partial, 9)
            result = tracker.update(image, value, image, i * .2, i)
            if i < 8:
                self.assertIsNone(result)
        self.assertEqual((result[0], result[2], result[3]), (99, 0., 0))
        self.assertIsNone(tracker.update(partial, 9, partial, 1.8, 9))
        # Persistent readings remain eligible after the short dropout window.
        for i in range(3):
            result = tracker.update(partial, 9, partial, 4. + i * .2, 20 + i)
        self.assertEqual(result[0], 9)

    def test_same_image_conflicting_number_and_expired_bridge_are_rejected(self):
        tracker = VisualTracker()
        full = digit('99')
        for i in range(5):
            tracker.update(full, 99, full, i * .2, i)
        self.assertIsNone(tracker.update(full, 98, full, 1., 5))
        self.assertEqual(tracker.update(full, None, full, 1.2, 6)[0], 99)
        self.assertIsNone(tracker.update(full, None, full, 3., 7))
        self.assertEqual(tracker.value, 99)

    def test_manual_reference_needs_complete_and_unambiguous_shape(self):
        tracker = VisualTracker()
        tracker.add_manual_reference(digit('99'), 99)
        tracker.add_manual_reference(digit('100'), 100)
        self.assertEqual(tracker.match_manual(digit('99')), 99)
        partial = digit('99'); partial[:, :35] = 0
        self.assertIsNone(tracker.match_manual(partial))
        tracker.add_manual_reference(digit('99'), 98)
        self.assertIsNone(tracker.match_manual(digit('99')))


class ManualCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_selected_fields_use_only_supplied_frame_and_unit_rois(self):
        frame = np.zeros((30,90,3),np.uint8)
        frame[:,30:60]=47;frame[:,60:90]=83
        class OCR:
            def extract_time(self,image):
                self_value=int(image[0,0,0])
                return f'00:00:{self_value:02}',time(0,0,self_value),1
            def extract_capacity(self,image,**kwargs):
                return int(image[0,0,0])/100,'0.83',1
        result=selected_fields(frame,{'soc':[0,0,30,30],'time':[30,0,30,30],'cap':[60,0,30,30]}, {},OCR())
        self.assertEqual(result['time_str'],'00:00:47')
        self.assertEqual(result['capacity'],.83)

    def test_candidate_limit_rejects_instead_of_truncating(self):
        task=CaptureTask('missing.mp4','unused',{}, {},start=0,end=1000,step=.5)
        errors=[];task.error.connect(errors.append);task.run()
        self.assertIn('240',errors[0])

    def test_exact_snapshot_survives_dialog_and_switch_invalidates_ocr(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{'LOCALAPPDATA':tmp}):
            frame=digit('99');path=Path(tmp)/'frame.png';write_image(path,frame)
            rois={key:[0,0,150,70] for key in ('soc','time','cap')}
            dialog=ManualCaptureDialog('video.mp4',2,99,rois,{},0,1,10)
            for index in (10,11):
                dialog.add_candidate({'path':str(path),'frame_index':index,'video_sec':index/10,'thumbnail':frame})
            dialog.candidates.setCurrentRow(0)
            fields=dict(time_str='00:00:47',time_obj=time(0,0,47),capacity=.83,capacity_raw='.83')
            dialog.read_complete(fields)
            dialog.candidates.setCurrentRow(1)
            self.assertIsNone(dialog.fields)
            self.assertFalse(dialog.confirm_button.isEnabled())
            dialog.read_complete(fields)
            dialog.confirm()
            self.assertEqual(dialog.record['capture_unit'],2)
            self.assertEqual(dialog.record['frame_index'],11)
            self.assertEqual(dialog.record['video_sec'],1.1)
            self.assertTrue(Path(dialog.record['snapshot_path']).exists())
            self.assertFalse(Path(dialog.temp.name).exists())
