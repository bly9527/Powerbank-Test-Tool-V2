import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.project_archive import ProjectArchive, create_project, group_key


class ArchiveEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        template = next(Path(__file__).resolve().parents[1].glob('PB519*.xlsx'))
        with patch('core.project_archive.projects_home', return_value=self.directory / 'projects'):
            self.project = create_project('evidence test', template)
        self.group = next(g for g in self.project.groups['充电时间及曲线'] if g['col_offset'] == 0)

    def save(self, record):
        self.project.save_results([{'unit': 1, 'mode': 'charge', 'group': self.group,
                                   'records': {1: record}}], 'original.mp4')
        reopened = ProjectArchive(self.project.root)
        return reopened, reopened.metadata['entries'][group_key(self.group)]['records']['1']

    def test_snapshot_and_precise_locator_survive_source_deletion_and_reopen(self):
        snapshot = self.directory / 'live.png'
        snapshot.write_bytes(b'original frame bytes')
        record = {'time_str': '00:01:23', 'capacity': 0.1234,
                  'video_sec': 87.123456789, 'frame_index': 1314,
                  'source_video': 'capture.mp4', 'snapshot_path': str(snapshot), 'locator_version': 1}
        project, stored = self.save(record)
        self.assertFalse(Path(stored['snapshot_path']).is_absolute())
        snapshot.unlink()
        resolved = project.resolve_record(stored)
        self.assertEqual(Path(resolved['snapshot_path']).read_bytes(), b'original frame bytes')
        self.assertEqual(resolved['video_sec'], 87.123456789)
        self.assertEqual(resolved['frame_index'], 1314)
        self.assertEqual(resolved['source_video'], 'capture.mp4')
        self.assertEqual(resolved['capacity'], 0.1234)
        # Re-saving a resolved archived row reuses its evidence, without making
        # another copy or mutating the caller's record.
        _, again = self.save(resolved)
        self.assertEqual(again['snapshot_path'], stored['snapshot_path'])
        self.assertEqual(len(list((project.root / 'evidence').glob('*.png'))), 1)
        self.assertEqual(record['snapshot_path'], str(snapshot))

    def test_legacy_rows_remain_loadable_without_fabricated_precise_locator(self):
        project, stored = self.save({'time_str': '00:01:23', 'capacity': 0.1234, 'video_sec': 87.1})
        self.assertNotIn('locator_version', stored)
        resolved = project.resolve_record(stored, source_video='original.mp4')
        self.assertEqual(resolved['snapshot_path'], '')
        self.assertEqual(resolved['source_video'], 'original.mp4')
        self.assertNotIn('frame_index', resolved)

    def test_resolver_rejects_traversal_external_and_malformed_paths(self):
        outside = self.directory / 'outside.png'
        outside.write_bytes(b'not evidence')
        for value in ('../outside.png', 'evidence/../../outside.png', str(outside),
                      '\x00.png', None, 123, {'path': 'evidence/a.png'}, 'report.xlsx'):
            with self.subTest(value=value):
                self.assertEqual(self.project.resolve_record({'snapshot_path': value})['snapshot_path'], '')

    def test_legacy_record_preserves_its_source_when_resaved_with_another_video(self):
        restored = self.project.resolve_record(
            {'time_str': '00:01:23', 'capacity': 0.1234, 'video_sec': 87.1},
            source_video='video_A.mp4')
        self.project.save_results([{'unit': 1, 'mode': 'charge', 'group': self.group,
                                   'records': {1: restored}}], 'video_B.mp4')
        entry = ProjectArchive(self.project.root).metadata['entries'][group_key(self.group)]
        self.assertEqual(entry['video_path'], 'video_B.mp4')
        self.assertEqual(entry['records']['1']['source_video'], 'video_A.mp4')
        self.assertNotIn('locator_version', entry['records']['1'])

    def test_missing_snapshot_and_invalid_locator_never_create_evidence_reference(self):
        base = {'time_str': '00:00:01', 'video_sec': 1.234, 'frame_index': 3,
                'snapshot_path': str(self.directory / 'missing.png'), 'locator_version': 1}
        _, stored = self.save(base)
        self.assertEqual(stored['locator_version'], 1)
        self.assertEqual(stored['source_video'], 'original.mp4')
        self.assertNotIn('snapshot_path', stored)
        for change in ({'frame_index': -1}, {'frame_index': True}, {'video_sec': '1'},
                       {'video_sec': float('inf')}, {'locator_version': 12}):
            with self.subTest(change=change):
                _, stored = self.save(dict(base, **change))
                self.assertNotIn('locator_version', stored)
                self.assertNotIn('snapshot_path', stored)

    def test_failed_snapshot_copy_does_not_publish_metadata(self):
        snapshot = self.directory / 'live.png'
        snapshot.write_bytes(b'frame')
        previous = self.project.metadata_path.read_bytes()
        with patch('core.project_archive.shutil.copyfile', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.save({'time_str': '00:00:01', 'video_sec': 1.234, 'frame_index': 3,
                           'snapshot_path': str(snapshot), 'locator_version': 1})
        self.assertEqual(self.project.metadata_path.read_bytes(), previous)
        self.assertEqual(json.loads(previous)['entries'], {})


if __name__ == '__main__':
    unittest.main()
