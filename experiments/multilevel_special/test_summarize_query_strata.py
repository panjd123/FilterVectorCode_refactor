"""Check the scientific joins, histograms, and absence semantics of C3."""
import copy
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import extract_query_recall
import summarize_query_strata as subject


class QueryStrataTest(unittest.TestCase):
    def test_raw_export_refuses_existing_output_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            marker = output/'keep.txt'
            marker.write_text('existing audit')
            with self.assertRaisesRegex(ValueError, 'already exists'):
                extract_query_recall.extract(output/'missing_repo', output)
            self.assertEqual(marker.read_text(), 'existing audit')

    def setUp(self):
        self.coverage_rows = [
            dict(workload='w', QueryID='0', QuerySize='1', labels='1',
                 eligible_count='95', true_selectivity='.95'),
            dict(workload='w', QueryID='1', QuerySize='2', labels='1,2',
                 eligible_count='10', true_selectivity='.1')]
        self.coverage = subject.validate_coverage(self.coverage_rows, n=100,
                                                  workloads=['w'], batch_size=2)
        self.points = [dict(topology_code='T', workload='w', method='trie', lsearch='8',
                            warm_1_recall='.9', warm_2_recall='.95')]
        self.records = [dict(workload='w', method='trie', topology_code='T', lsearch='8',
                             repeat=str(repeat), QueryID=str(q), QuerySize=str(q+1),
                             Recall=str(recall))
                        for repeat, recalls in [(1, [.8, 1.]), (2, [.9, 1.])]
                        for q, recall in enumerate(recalls)]

    def aggregate(self, records=None):
        return subject.aggregate_records(self.records if records is None else records,
                                         self.coverage, self.points, batch_size=2)

    def test_bin_boundaries_are_right_closed(self):
        for i, upper in enumerate(subject.SELECTIVITY_UPPER):
            self.assertEqual(subject.selectivity_bin(upper), i)
            if upper != 1:
                self.assertEqual(subject.selectivity_bin(math.nextafter(upper, 1.)), i+1)
        for value in [0, -1, 1.01, math.nan, math.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.selectivity_bin(value)

    def test_coverage_requires_exact_identity_and_denominator(self):
        variants = [self.coverage_rows[:-1], self.coverage_rows+self.coverage_rows[:1]]
        for field, value in [('QuerySize', '3'), ('labels', '1,1'),
                             ('eligible_count', '9'), ('true_selectivity', '.94')]:
            rows = copy.deepcopy(self.coverage_rows)
            rows[0][field] = value
            variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                subject.validate_coverage(rows, n=100, workloads=['w'], batch_size=2)

    def test_empty_predicate_means_full_database(self):
        row = dict(workload='w', QueryID='0', QuerySize='0', labels='',
                   eligible_count='100', true_selectivity='1')
        result = subject.validate_coverage([row], n=100, workloads=['w'], batch_size=1)
        self.assertEqual(result['w', 0]['size_stratum'], '0')
        self.assertEqual(result['w', 0]['selectivity_bin'], '(95,100]')
        row.update(eligible_count='90', true_selectivity='.9')
        with self.assertRaises(ValueError):
            subject.validate_coverage([row], n=100, workloads=['w'], batch_size=1)

    def test_batch_crossing_does_not_make_each_stratum_cross(self):
        hist, checks = self.aggregate()
        self.assertEqual([r['reconstructed_recall'] for r in checks], [.9, .95])
        single = subject.summarize_hist(hist['T', 'w', 1, 'size', '1'])
        self.assertEqual(single['mean_recall'], .8)
        self.assertEqual(single['recall_below_0p9'], 1)
        self.assertEqual(hist['T', 'w', 1, 'selectivity', '(80,95]'], Counter({8: 1}))

    def test_record_join_rejects_missing_duplicate_and_wrong_crossing(self):
        variants = [self.records[:-1], self.records+self.records[:1]]
        for field, value in [('repeat', '0'), ('lsearch', '9'), ('method', 'lng'),
                             ('QuerySize', '2'), ('QueryID', '2'), ('Recall', '.7')]:
            records = copy.deepcopy(self.records)
            records[0][field] = value
            variants.append(records)
        for records in variants:
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.aggregate(records)

    def test_recall_grid_rejects_invalid_and_nonfinite_values(self):
        for value in ['nan', 'inf', '-.1', '1.1', '.85']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                subject.recall_hits(value)
        self.assertEqual(subject.recall_hits('.9'), 9)

    def test_no_crossing_and_empty_stratum_are_not_zero_recall(self):
        hist, _ = self.aggregate()
        cells = [dict(topology_code='T', workload='w', method='trie', lsearch='8', status='complete'),
                 dict(topology_code='L', workload='w', method='lng', lsearch='', status='unavailable')]
        rows = subject.summarize_strata(hist, self.coverage, cells, 'size')
        empty = next(r for r in rows if r['topology_code']=='T' and r['stratum']=='0')
        missing = next(r for r in rows if r['topology_code']=='L' and r['stratum']=='1')
        self.assertEqual((empty['status'], empty['count'], empty['mean_recall']), ('empty_stratum', 0, ''))
        self.assertEqual((missing['status'], missing['count'], missing['recall_zero']), ('no_crossing', '', ''))
        self.assertEqual(missing['workload_stratum_count'], 1)

    def test_prior_histogram_check_includes_empty_strata(self):
        hist, _ = self.aggregate()
        reference = []
        for repeat in (1, 2):
            for size in subject.SIZE_STRATA:
                reference.append(dict(topology_code='T', workload='w', repeat=str(repeat),
                                      query_size_stratum=size,
                                      **subject.summarize_hist(hist.get(('T', 'w', repeat, 'size', size), Counter()))))
        subject.check_reference(hist, reference, self.points)
        with self.assertRaises(ValueError):
            subject.check_reference(hist, reference[1:], self.points)
        reference[1]['recall_hist_0.8'] = 0
        with self.assertRaises(ValueError):
            subject.check_reference(hist, reference, self.points)


if __name__ == '__main__':
    unittest.main()
