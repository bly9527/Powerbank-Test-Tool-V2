import unittest

from core.soc_tracker import SegmentSOCTracker


class SOCTrackerTests(unittest.TestCase):
    def feed(self, tracker, readings):
        accepted = []
        for second, value in readings:
            result = tracker.update(value, second, lambda s=second: ('frame', s))
            if result:
                accepted.append(result)
        return accepted

    def test_charge_large_misreads_never_advance_or_occupy_slots(self):
        for outlier in (60, 99):
            with self.subTest(outlier=outlier):
                tracker = SegmentSOCTracker('charge')
                sequence = [(i, v) for i, v in enumerate([1, 1, 2, 2, outlier,
                                                         outlier, outlier, 3, 3])]
                self.assertEqual([r.value for r in self.feed(tracker, sequence)], [1, 2, 3])
                self.assertEqual(tracker.confirmed, 3)
                self.assertIn(outlier, tracker.pending)

    def test_waiting_on_wrong_value_does_not_create_a_recovery_gap(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 2), (1, 2)])
        self.assertEqual(self.feed(tracker, [(2, 99), (3, 99), (100, 99), (1000, 99)]), [])
        self.assertEqual(tracker.confirmed, 2)

    def test_recovery_after_long_gap_keeps_first_frame(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20)] + [(s, None) for s in range(5, 60, 5)])
        records = self.feed(tracker, [(61, 30), (62, 30), (63, 30), (64, 30)])
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0].value, records[0].first_sec, records[0].payload),
                         (30, 61, ('frame', 61)))
        self.assertNotIn(30, tracker.pending)

    def test_gap_uses_latest_observed_anchor_not_first_record(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20), (600, 20)])
        self.assertEqual(self.feed(tracker, [(601, 30), (602, 30), (603, 30), (1000, 30)]), [])
        self.assertEqual(tracker.confirmed, 20)

    def test_repeated_reverse_reading_does_not_become_trusted(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20)])
        self.assertEqual(self.feed(tracker, [(s, 19) for s in range(2, 20)]), [])
        self.assertEqual([r.value for r in self.feed(tracker, [(20, 21), (21, 21)])], [21])

    def test_earlier_false_soc_does_not_block_later_real_soc(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 2), (1, 2), (2, 99), (3, 99), (4, 2)])
        self.assertIn(99, tracker.pending)
        self.feed(tracker, [(s, None) for s in (10, 20, 30, 40)])
        records = self.feed(tracker, [(s, 99) for s in (110, 111, 112, 113)])
        self.assertEqual([r.value for r in records], [99])
        self.assertEqual(records[0].first_sec, 110)
        self.assertNotIn(99, tracker.pending)

    def test_alternating_rejected_values_cannot_manufacture_an_unreadable_gap(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20)])
        self.assertEqual(self.feed(tracker, [(2, 99), (3, 98), (100, 99),
                                           (101, 99), (102, 99), (103, 99)]), [])
        self.assertEqual(tracker.confirmed, 20)

    def test_one_long_missing_interval_restarts_the_same_recovery_candidate(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20), (2, 30), (3, 30), (100, None)])
        records = self.feed(tracker, [(101, 30), (102, 30), (103, 30), (104, 30)])
        self.assertEqual([r.value for r in records], [30])
        self.assertEqual(records[0].first_sec, 101)

    def test_loss_duration_is_completed_when_the_next_readable_frame_arrives(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20), (2, 30), (3, 30), (4, None)])
        records = self.feed(tracker, [(100, 30), (101, 30), (102, 30), (103, 30)])
        self.assertEqual([r.value for r in records], [30])
        self.assertEqual(records[0].first_sec, 100)

    def test_multiple_fast_dark_frames_do_not_discard_first_good_payload(self):
        tracker = SegmentSOCTracker('charge', sample_interval=.2)
        records = self.feed(tracker, [(0, 5), (.1, None), (.2, None), (.3, None),
                                     (.4, None), (.5, 5)])
        self.assertEqual(records[0].first_sec, 0)

    def test_one_missing_sample_at_normal_interval_keeps_first_frame(self):
        tracker = SegmentSOCTracker('discharge')
        self.feed(tracker, [(0, 10), (1, 10)])
        records = self.feed(tracker, [(4, 9), (5, None), (6, 9)])
        self.assertEqual(records[0].first_sec, 4)

    def test_duplicate_timestamps_are_not_independent_confirmations(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 1), (1, 1)])
        self.assertEqual(self.feed(tracker, [(2, 2), (2, 2), (1.9, 2)]), [])
        self.assertEqual(tracker.confirmed, 1)
        records = self.feed(tracker, [(3, 2)])
        self.assertEqual([r.value for r in records], [2])

    def test_pending_metadata_tracks_latest_episode_not_old_outlier(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20), (2, 30), (3, 30)])
        self.assertEqual(tracker.pending[30]['video_sec'], 2)
        self.feed(tracker, [(100, None), (101, 30)])
        self.assertEqual(tracker.pending[30]['video_sec'], 101)
        self.assertIn('更多画面', tracker.pending[30]['reason'])

    def test_verified_last_anchor_can_be_replaced_without_losing_first_frame(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 97), (1, 97), (2, 99), (3, 99)])
        self.feed(tracker, [(10, 98), (11, 98), (12, 98), (13, 98)])
        self.assertIsNotNone(tracker.correction_candidate())
        self.assertIsNone(tracker.correct_last_anchor(97))
        corrected = tracker.correct_last_anchor(99)
        self.assertEqual((corrected.value, corrected.first_sec, corrected.payload),
                         (98, 10, ('frame', 10)))
        self.assertNotIn(98, tracker.pending)
        later = self.feed(tracker, [(20, 99), (21, 99)])
        self.assertEqual(later[0].first_sec, 20)

    def test_return_to_previous_anchor_preserves_its_time_and_rate(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 96), (1, 96), (60, 97), (61, 97), (70, 99), (71, 99)])
        self.feed(tracker, [(sec, 97) for sec in range(100, 104)])
        corrected = tracker.correct_last_anchor(99)
        self.assertEqual(corrected.value, 97)
        self.assertEqual(tracker.last_change_sec, 60)
        self.assertEqual(tracker._anchors, [(96, 0), (97, 60)])
        self.assertEqual(list(tracker._seconds_per_percent), [60])
        self.feed(tracker, [(120, 98), (121, 98)])
        self.assertEqual(list(tracker._seconds_per_percent), [60, 60])

    def test_reverse_beyond_previous_anchor_cannot_be_corrected(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 97), (1, 97), (2, 99), (3, 99)])
        self.feed(tracker, [(i, 96) for i in range(10, 20)])
        self.assertIsNone(tracker.correction_candidate())
        self.assertIsNone(tracker.correct_last_anchor(99))
        self.assertEqual(tracker.confirmed, 99)

    def test_discharge_correction_preserves_direction_checks(self):
        tracker = SegmentSOCTracker('discharge')
        self.feed(tracker, [(0, 10), (1, 10), (2, 8), (3, 8)])
        self.feed(tracker, [(i, 9) for i in range(10, 14)])
        corrected = tracker.correct_last_anchor(8)
        self.assertEqual((corrected.value, corrected.first_sec), (9, 10))
        self.assertEqual([x.value for x in self.feed(tracker, [(20, 8), (21, 8)])], [8])

    def test_discharge_tail_all_single_digits_and_zero(self):
        tracker = SegmentSOCTracker('discharge')
        readings = [(i, v) for i, v in enumerate([v for v in range(10, -1, -1) for _ in range(2)])]
        self.assertEqual([r.value for r in self.feed(tracker, readings)], list(range(10, -1, -1)))

    def test_mid_video_initial_percentage_is_allowed(self):
        for mode in ('charge', 'discharge'):
            tracker = SegmentSOCTracker(mode)
            records = self.feed(tracker, [(0, 60), (1, 60)])
            self.assertEqual([r.value for r in records], [60])

    def test_blink_misses_keep_first_frame_but_long_loss_resets_it(self):
        tracker = SegmentSOCTracker('charge')
        records = self.feed(tracker, [(0, None), (.5, 5), (1, None), (1.5, 5)])
        self.assertEqual(records[0].first_sec, .5)
        tracker = SegmentSOCTracker('charge')
        records = self.feed(tracker, [(0, 5)] + [(s, None) for s in range(1, 5)] + [(5, 5), (6, 5)])
        self.assertEqual(records[0].first_sec, 5)

    def test_recent_rate_requires_a_plausible_gap_for_large_change(self):
        tracker = SegmentSOCTracker('charge')
        self.feed(tracker, [(0, 20), (1, 20), (120, 21), (121, 21)])
        self.assertEqual(self.feed(tracker, [(150 + s, 30) for s in range(5)]), [])
        self.feed(tracker, [(200 + s, None) for s in range(4)])
        records = self.feed(tracker, [(500 + s, 30) for s in range(4)])
        self.assertEqual([r.value for r in records], [30])


if __name__ == '__main__':
    unittest.main()
