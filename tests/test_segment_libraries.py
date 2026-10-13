import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import onnxruntime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import QApplication

from core.segment_dataset import (write_json, read_json, write_image, read_image,
    set_label, samples, capture_sessions, SegmentMatcher, SampleCaptureWorker,
    register_template, template_catalog, import_dataset, read_frame_at,
    export_template_package, import_template_package, clean_unlabelled_capture_batches)
from core.segment_dataset import clean_all_unlabelled_samples
from ui.segment_dialog import SegmentDatasetDialog


def make_batch(root, batch_id, value, created, interval=5):
    folder = Path(root)/'captures'/batch_id
    folder.mkdir(parents=True)
    image = np.zeros((70, 100, 3), np.uint8)
    cv2.putText(image, str(value), (4, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255,255,255), 3)
    write_image(folder/'a.png', image)
    write_json(folder/'manifest.json', dict(source='video.mp4',unit=1,roi=[0,0,100,70],
        rotation=0,interval=interval,created_at=created,status='complete',samples=[
            dict(image='a.png',video_sec=created,requested_sec=created,frame_index=0)]))
    set_label(root, f'captures/{batch_id}/a.png', value)
    return image


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_annotation_advance_preserves_input_and_shows_existing_labels(self):
        cases = ((None, 3, 1, 25), (80, 3, 1, 25), (None, 501, 499, 25),
                 (None, 3, 1, 'invalid'), (None, 3, 1, None))
        for destination_label, count, current, saved_label in cases:
            with self.subTest(destination_label=destination_label, count=count), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                library = root / 'library'
                folder = library / 'captures' / 'batch'
                folder.mkdir(parents=True)
                image = np.full((30, 50, 3), 120, np.uint8)
                for index in {0, current, current + 1}:
                    write_image(folder / f'{index}.png', image)
                write_json(folder / 'manifest.json', dict(source='video.mp4', unit=1, interval=1,
                    created_at=1, samples=[dict(image=f'{i}.png', video_sec=i, frame_index=i)
                                           for i in range(count)]))
                set_label(library, 'captures/batch/0.png', 99)
                if saved_label is None:
                    set_label(library, f'captures/batch/{current}.png', 17)
                if destination_label is not None:
                    set_label(library, f'captures/batch/{current + 1}.png', destination_label)
                settings = QSettings(str(root / 'ui.ini'), QSettings.IniFormat)
                with patch('core.segment_dataset.data_home', return_value=root), \
                     patch('ui.segment_dialog.data_home', return_value=root), \
                     patch('ui.segment_dialog.QSettings', return_value=settings):
                    dialog = SegmentDatasetDialog({1: [str(library)], 2: [str(library)]})
                    try:
                        dialog.list.setCurrentRow(current)
                        dialog.label_value.setValue(25)
                        dialog.label_sample(saved_label)
                        row = dialog.rows[dialog.list.currentRow()]
                        self.assertEqual(row['key'], f'captures/batch/{current + 1}.png')
                        self.assertEqual(dialog.label_value.value(), destination_label or 25)
                        labels = read_json(library / 'labels.json', {})
                        self.assertEqual(labels.get(f'captures/batch/{current}.png'), saved_label)
                        self.assertEqual(labels['captures/batch/0.png'], 99)
                        self.assertEqual(labels.get(row['key']), destination_label)
                    finally:
                        dialog.close()

    def test_annotation_filtered_zero_index_restores_destination_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / 'library'
            make_batch(library, 'first', 99, 2)
            make_batch(library, 'second', 80, 1)
            settings = QSettings(str(root / 'ui.ini'), QSettings.IniFormat)
            with patch('core.segment_dataset.data_home', return_value=root), \
                 patch('ui.segment_dialog.data_home', return_value=root), \
                 patch('ui.segment_dialog.QSettings', return_value=settings):
                dialog = SegmentDatasetDialog({1: [str(library)]})
                try:
                    dialog.batch.setCurrentIndex(dialog.batch.findData('*'))
                    dialog.filter.setCurrentIndex(2)
                    self.assertEqual(dialog.label_value.value(), 99)
                    dialog.label_sample(None)
                    self.assertEqual(dialog.list.currentRow(), 0)
                    self.assertEqual(dialog.label_value.value(), 80)
                    self.assertNotIn('captures/first/a.png', read_json(library / 'labels.json', {}))
                finally:
                    dialog.close()

    def test_annotation_removing_last_filtered_page_preserves_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = root / 'library'
            folder = library / 'captures' / 'batch'
            folder.mkdir(parents=True)
            for index in (0, 500):
                write_image(folder / f'{index}.png', np.full((30, 50, 3), 120, np.uint8))
            write_json(folder / 'manifest.json', dict(source='video.mp4', unit=1, interval=1,
                created_at=1, samples=[dict(image=f'{i}.png', video_sec=i, frame_index=i)
                                       for i in range(501)]))
            settings = QSettings(str(root / 'ui.ini'), QSettings.IniFormat)
            with patch('core.segment_dataset.data_home', return_value=root), \
                 patch('ui.segment_dialog.data_home', return_value=root), \
                 patch('ui.segment_dialog.QSettings', return_value=settings):
                dialog = SegmentDatasetDialog({1: [str(library)]})
                try:
                    dialog.filter.setCurrentIndex(1)
                    dialog.page.setValue(2)
                    dialog.label_value.setValue(25)
                    dialog.label_sample(25)
                    self.assertEqual(dialog.page.value(), 1)
                    self.assertEqual(dialog.page.maximum(), 1)
                    self.assertEqual(dialog.label_value.value(), 25)
                    self.assertEqual(dialog.rows[dialog.list.currentRow()]['key'], 'captures/batch/0.png')
                    self.assertEqual(read_json(library / 'labels.json', {}), {'captures/batch/500.png': 25})
                finally:
                    dialog.close()

    def test_union_of_templates_and_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            a,b = Path(tmp)/'a',Path(tmp)/'b'
            first=make_batch(a,'old',25,1)
            second=make_batch(b,'new',80,2)
            matcher=SegmentMatcher([str(a),str(b)])
            self.assertEqual(matcher.labels,{25,80})
            self.assertEqual(matcher.extract_soc(first)[0],25)
            self.assertEqual(matcher.extract_soc(second)[0],80)
            write_image(b/'captures/new/a.png',first)
            matcher=SegmentMatcher([str(a),str(b)])
            self.assertIsNone(matcher.extract_soc(first)[0])
            self.assertIn('候选难以区分',matcher.last_reason)

    def test_named_templates_import_preserves_existing_labels_and_source(self):
        with tempfile.TemporaryDirectory() as tmp, patch('core.segment_dataset.data_home',return_value=Path(tmp)):
            a=Path(register_template('已有模板'))
            b=Path(register_template('新模板'))
            make_batch(a,'batch',99,1)
            make_batch(b,'batch',20,2)
            original=(a/'labels.json').read_bytes()
            before=(b/'labels.json').read_bytes()
            self.assertEqual(import_dataset(a,b),1)
            self.assertEqual((a/'labels.json').read_bytes(),original)
            labels=read_json(b/'labels.json',{})
            self.assertEqual(set(labels.values()),{20,99})
            self.assertEqual(labels['captures/batch/a.png'],20)
            self.assertNotEqual(before,(b/'labels.json').read_bytes())
            self.assertEqual(len(capture_sessions(b)),2)
            register_template('改名后',b)
            self.assertEqual(next(e['name'] for e in template_catalog() if e['path']==str(b)),'改名后')
            with self.assertRaises(ValueError):import_dataset(a,a)

    def test_shared_template_only_packages_confirmed_images(self):
        with tempfile.TemporaryDirectory() as tmp, patch('core.segment_dataset.data_home',return_value=Path(tmp)/'libraries'):
            source=Path(register_template('白色数码管'))
            make_batch(source,'one',99,1)
            extra=source/'captures/one/unlabelled.png'
            write_image(extra,np.zeros((40,40,3),np.uint8))
            manifest=read_json(source/'captures/one/manifest.json',{})
            manifest['samples'].append(dict(image='unlabelled.png',video_sec=3,frame_index=5))
            write_json(source/'captures/one/manifest.json',manifest)
            package=Path(tmp)/'share.zip'
            self.assertEqual(export_template_package(source,package,'白色数码管'),1)
            imported,count=import_template_package(package)
            self.assertEqual(count,1)
            self.assertNotEqual(Path(imported),source)
            self.assertEqual(len(samples(imported)),1)
            self.assertEqual(SegmentMatcher(imported).labels,{99})
            self.assertTrue(extra.exists())

    def test_close_cleanup_only_touches_new_unlabelled_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            make_batch(root,'old',80,1)
            make_batch(root,'new',99,2)
            for batch in ('old','new'):
                write_image(root/'captures'/batch/'unlabelled.png',np.zeros((40,40,3),np.uint8))
                manifest=read_json(root/'captures'/batch/'manifest.json',{})
                manifest['samples'].append(dict(image='unlabelled.png',video_sec=3,frame_index=5))
                write_json(root/'captures'/batch/'manifest.json',manifest)
            self.assertEqual(clean_unlabelled_capture_batches(root,['new']),1)
            self.assertTrue((root/'captures/old/unlabelled.png').exists())
            self.assertTrue((root/'captures/new/a.png').exists())
            self.assertFalse((root/'captures/new/unlabelled.png').exists())
            self.assertEqual(len(samples(root,'new')),1)

    def test_clear_all_unlabelled_keeps_numbers_and_invalid_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_batch(root, 'keep_number', 80, 1)
            make_batch(root, 'mixed', 99, 2)
            folder = root / 'captures' / 'mixed'
            for name in ('invalid.png', 'remove.png'):
                write_image(folder / name, np.zeros((40, 40, 3), np.uint8))
            manifest = read_json(folder / 'manifest.json', {})
            manifest['samples'] += [
                dict(image='invalid.png', video_sec=3, frame_index=3),
                dict(image='remove.png', video_sec=4, frame_index=4)]
            write_json(folder / 'manifest.json', manifest)
            set_label(root, 'captures/mixed/invalid.png', 'invalid')
            empty = root / 'captures' / 'empty'
            empty.mkdir(parents=True)
            write_image(empty / 'only.png', np.zeros((20, 20, 3), np.uint8))
            write_json(empty / 'manifest.json', dict(source='x.mp4', unit=1, created_at=3,
                samples=[dict(image='only.png', video_sec=0, frame_index=0)]))
            write_json(root / 'hidden_batches.json', ['mixed', 'empty'])

            self.assertEqual(clean_all_unlabelled_samples(root), 2)
            self.assertTrue((folder / 'a.png').exists())
            self.assertTrue((folder / 'invalid.png').exists())
            self.assertFalse((folder / 'remove.png').exists())
            self.assertFalse(empty.exists())
            self.assertEqual(read_json(root / 'hidden_batches.json', []), ['mixed'])
            self.assertEqual(set(read_json(root / 'labels.json', {}).values()), {99, 80, 'invalid'})

    def test_dialog_template_selection_independent_from_capture_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a,b=root/'a',root/'b'
            make_batch(a,'zzz_old',99,1)
            make_batch(a,'aaa_new',80,2,.2)
            make_batch(b,'batch',20,3)
            settings=QSettings(str(root/'ui.ini'),QSettings.IniFormat)
            with patch('core.segment_dataset.data_home',return_value=root),patch('ui.segment_dialog.data_home',return_value=root),patch('ui.segment_dialog.QSettings',return_value=settings):
                dialog=SegmentDatasetDialog({1:[str(a)],2:[str(b)]})
                self.assertEqual(dialog.batch.currentData(),'aaa_new')
                self.assertIn('0.2s',dialog.batch.currentText())
                dialog.unit.setCurrentIndex(1)
                self.assertEqual(dialog.root(),a)
                self.assertEqual(dialog.paths,{1:[str(a)],2:[str(b)]})
                dialog.select_all_templates(1,True)
                self.assertEqual(set(dialog.paths[1]),{str(a),str(b)})
                dialog.select_all_templates(2,False)
                self.assertEqual(dialog.paths[2],[])
                dialog.interval.setValue(.2)
                dialog.close()
                reopened=SegmentDatasetDialog({1:[str(a)],2:[str(b)]})
                self.assertEqual(reopened.interval.value(),.2)
                write_json(a/'hidden_batches.json',['aaa_new'])
                reopened.refresh_samples()
                self.assertEqual(reopened.batch.currentData(),'zzz_old')
                self.assertTrue((a/'captures/aaa_new/a.png').exists())
                self.assertEqual(len(read_json(a/'labels.json',{})),2)
                reopened.show_history.setChecked(True)
                reopened.batch.setCurrentIndex(reopened.batch.findData('aaa_new'))
                reopened.restore_current_batch()
                self.assertEqual(read_json(a/'hidden_batches.json',[]),[])
                reopened.close()

    def test_dense_capture_point_two_seconds_has_distinct_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);video=str(root/'tiny.avi')
            writer=cv2.VideoWriter(video,cv2.VideoWriter_fourcc(*'MJPG'),10,(80,60))
            for i in range(20):writer.write(np.full((60,80,3),i*10,np.uint8))
            writer.release()
            worker=SampleCaptureWorker(video,root/'dataset',(0,0,40,30),interval=.2,start=0,end=1)
            errors=[];worker.failed.connect(errors.append);worker.run()
            self.assertFalse(errors)
            rows=samples(root/'dataset')
            self.assertEqual(len(rows),5)
            self.assertEqual([r['frame_index'] for r in rows],[0,2,4,6,8])
            self.assertEqual(capture_sessions(root/'dataset')[0]['interval'],.2)

    def test_seek_decodes_forward_and_eof_is_not_interior_failure(self):
        class Capture:
            def __init__(self,*a):self.pos=0;self.last=0
            def isOpened(self):return True
            def get(self,p):return {cv2.CAP_PROP_FPS:10,cv2.CAP_PROP_FRAME_COUNT:100,
                cv2.CAP_PROP_POS_FRAMES:self.pos,cv2.CAP_PROP_POS_MSEC:self.last*1000}.get(p,0)
            def set(self,p,v):self.pos=max(0,int(v/1000*10)-2)
            def read(self):
                if self.pos>=99:return False,None
                self.last=self.pos/10;self.pos+=1
                return True,np.zeros((30,40,3),np.uint8)
            def release(self):pass
        cap=Capture()
        self.assertTrue(read_frame_at(cap,5)[0])
        self.assertAlmostEqual(cap.last,5)
        with tempfile.TemporaryDirectory() as tmp,patch('core.segment_dataset.cv2.VideoCapture',Capture):
            worker=SampleCaptureWorker('fake',tmp,(0,0,20,20),interval=.2,start=9.5,end=10)
            errors=[];worker.failed.connect(errors.append);worker.run()
            self.assertFalse(errors)
            self.assertEqual(capture_sessions(tmp)[0]['end_reason'],'video_eof')
        class Broken(Capture):
            def read(self):
                if self.pos>=50:return False,None
                return super().read()
        with tempfile.TemporaryDirectory() as tmp,patch('core.segment_dataset.cv2.VideoCapture',Broken):
            worker=SampleCaptureWorker('fake',tmp,(0,0,20,20),interval=.2,start=4.5,end=8)
            errors=[];worker.failed.connect(errors.append);worker.run()
            self.assertTrue(errors)
            self.assertEqual(capture_sessions(tmp)[0]['status'],'error')
            self.assertGreater(len(samples(tmp)),0)


if __name__=='__main__':unittest.main()
