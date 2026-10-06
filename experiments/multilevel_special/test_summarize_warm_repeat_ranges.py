import copy
import math
import unittest

import summarize_warm_repeat_ranges as ranges


class WarmRepeatRangesTest(unittest.TestCase):
    def fixture(self):
        cells, points = [], []
        for code in ranges.CODES:
            for workload in ranges.WORKLOAD_ORDER:
                row = {'topology_code': code, 'workload': workload,
                       'method': code, 'lsearch': '100', 'status': 'complete',
                       'warm_median_qps': 1_000_000/3, 'recall_min': '.95',
                       'mean_selectivity': '.1'}
                cells.append(row)
                points.append({**row, 'warm_1_ms': 2., 'warm_2_ms': 4.,
                               'warm_1_recall': .95, 'warm_2_recall': .96,
                               'qps_at_median_time': 1_000_000/3,
                               'warm_cv': math.sqrt(2)/3})
        return cells, points

    def test_cross_repeat_range_does_not_pair_repeat_indices(self):
        self.assertEqual((2., 6.), ranges.observed_ratio(
            {'times': [2., 4.]}, {'times': [8., 12.]}))
        self.assertEqual((2., 6.), ranges.observed_ratio(
            {'times': [2., 4.]}, {'times': [12., 8.]}))

    def test_reconstructs_median_time_qps_and_keeps_nc_absent(self):
        cells, points = self.fixture()
        cells[0]['status'] = 'unavailable'
        result = ranges.validate_points(points[1:], cells)
        self.assertEqual(125, len(result))
        self.assertAlmostEqual(1_000_000/3, next(iter(result.values()))['qps'])

    def test_rejects_coordinated_deletion_from_both_inputs(self):
        cells, points = self.fixture()
        with self.assertRaisesRegex(ValueError, 'complete 14 by 9'):
            ranges.validate_points(points[1:], cells[1:])

    def test_rejects_wrong_crossing_and_non_crossing_recall(self):
        cells, points = self.fixture()
        for field, value in [('lsearch', '200'), ('warm_1_recall', .89),
                             ('warm_1_ms', float('nan')),
                             ('qps_at_median_time', 1.)]:
            changed = copy.deepcopy(points)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                ranges.validate_points(changed, cells)

    def test_rejects_duplicate_or_missing_warm_records(self):
        cells, points = self.fixture()
        for changed in (points + [points[0]], points[1:]):
            with self.assertRaises(ValueError):
                ranges.validate_points(changed, cells)


if __name__ == '__main__':
    unittest.main()
