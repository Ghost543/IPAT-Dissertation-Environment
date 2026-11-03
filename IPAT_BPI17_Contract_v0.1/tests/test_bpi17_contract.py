import copy
import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('preflight',ROOT/'scripts/preflight_bpi17_contract.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
CONTRACT=json.loads((ROOT/'dataset_contract.json').read_text())


def offer(t, oid='o', label=1, aid='a', length=6):
    return dict(application_id=aid,offer_id=oid,creation_time=t,label=label,prefix_length=length)


def event(t,name='O_Create Offer',oid='o',label='true'):
    return f'<event><date key="time:timestamp" value="{t}"/><string key="concept:name" value="{name}"/><string key="EventID" value="{oid}"/><boolean key="Accepted" value="{label}"/></event>'


class ContractTests(unittest.TestCase):
    def parse(self,events,aid='a'):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.xes.gz'
            with gzip.open(p,'wt') as f:f.write(f'<log xmlns="http://www.xes-standard.org/"><trace><string key="concept:name" value="{aid}"/>{events}</trace></log>')
            return list(m.candidates(p))

    def test_exact_boundaries_enter_later_partition(self):
        for t,s in zip(CONTRACT['split']['boundaries'],['validation','calibration','test']):
            self.assertEqual(m.assign_application([offer(t)],CONTRACT)[0]['split'],s)

    def test_timezone_comparison_uses_instant(self):
        r=m.assign_application([offer('2016-07-01T01:00:00+02:00')],CONTRACT)
        self.assertEqual(r[0]['split'],'train')

    def test_all_offers_of_crossing_application_excluded(self):
        rows=[offer('2016-06-30T23:00:00Z','a'),offer('2016-07-01T00:00:00Z','b'),offer('2016-09-05T00:00:00Z','c')]
        actual=m.assign_application(rows,CONTRACT)
        self.assertEqual([r['split'] for r in actual],['excluded']*3)
        self.assertEqual([r['provisional_split'] for r in actual],['train','validation','calibration'])
        self.assertTrue(all('split' not in r for r in rows))

    def test_long_prefix_retained_and_future_label_not_substituted(self):
        es=''.join(event(f'2016-01-01T00:{i//60:02d}:{i%60:02d}Z','W_Work',str(i),'false') for i in range(138))
        es+=event('2016-01-01T00:02:18Z',oid='creation',label='false')
        es+=event('2016-01-01T00:02:19Z','O_Accepted','later','true')
        app,n,rows=self.parse(es)[0]
        self.assertEqual(n,140)
        self.assertEqual(rows[0]['prefix_length'],139)
        self.assertEqual(rows[0]['label'],0)
        self.assertEqual(rows[0]['offer_id'],'creation')
        self.assertEqual(len(rows),1)

    def test_invalid_label_rejected(self):
        with self.assertRaises(ValueError):self.parse(event('2016-01-01T00:00:00Z',label='unknown'))

    def test_duplicate_offer_rejected(self):
        with self.assertRaises(ValueError):self.parse(event('2016-01-01T00:00:00Z')+event('2016-01-01T00:00:01Z'))

    def test_missing_application_id_rejected(self):
        with self.assertRaises(ValueError):self.parse(event('2016-01-01T00:00:00Z'),aid='')

    def test_timestamp_order_and_timezone_rejected(self):
        for second in ['2016-01-01T00:00:00Z','2015-01-01T00:00:00Z','2016-01-01T00:00:01']:
            with self.subTest(second=second),self.assertRaises(ValueError):
                self.parse(event('2016-01-01T00:00:00Z')+event(second,oid='p'))

    def test_joint_length_label_counts(self):
        rows=[offer('2016-01-01T00:00:00Z',str(i),y,length=n) for i,(n,y) in enumerate([(6,0),(10,1),(11,0),(41,1)])]
        s=m.summarise(rows)
        self.assertEqual(s['labels'],{'0':2,'1':2})
        self.assertEqual(s['length_strata']['1–10']['offers'],2)
        self.assertEqual(s['length_strata']['41+']['labels'],{'0':0,'1':1})

    def test_changed_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.gz';p.write_bytes(b'wrong archive')
            with self.assertRaisesRegex(ValueError,'SHA-256'):m.preflight(p,CONTRACT)


if __name__=='__main__':unittest.main()
