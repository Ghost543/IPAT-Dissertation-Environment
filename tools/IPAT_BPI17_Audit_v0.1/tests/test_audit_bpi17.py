import gzip
import importlib.util
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

spec=importlib.util.spec_from_file_location('audit_bpi17',Path(__file__).parents[1]/'scripts'/'audit_bpi17.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def event(name,time,**extra):
    e=ET.Element('event')
    for k,v in {'concept:name':name,'time:timestamp':time,**extra}.items():
        ET.SubElement(e,'string',key=k,value=v)
    return e


def fixture():
    t=ET.Element('trace')
    ET.SubElement(t,'string',key='concept:name',value='app1')
    ET.SubElement(t,'float',key='RequestedAmount',value='100')
    t.append(event('A_Create Application','2016-10-19T00:00:00Z'))
    t.append(event('O_Create Offer','2016-10-19T01:00:00Z',EventID='o1',Accepted='true',CreditScore='900'))
    t.append(event('O_Created','2016-10-19T01:00:01Z',OfferID='o1'))
    t.append(event('O_Create Offer','2016-10-20T01:00:00Z',EventID='o2',Accepted='false',CreditScore='800'))
    t.append(event('O_Created','2016-10-20T01:00:01Z',OfferID='o2'))
    t.append(event('O_Accepted','2016-10-21T00:00:00Z',OfferID='o1'))
    return t


class AuditTests(unittest.TestCase):
    def test_prefix_excludes_future_events(self):
        *_,offers,missing,total=m.analyse_trace(fixture())
        self.assertEqual([o['prefix_length'] for o in offers],[2,4])
        self.assertEqual(total,6)
        self.assertEqual(missing['CreditScore'],3)

    def test_outcome_lookup_matches_offer_id(self):
        offers=m.analyse_trace(fixture())[6]
        self.assertEqual([o['later_accepted_event'] for o in offers],[True,False])

    def test_attribute_and_event_disagreement_is_visible(self):
        t=fixture(); t[-1].find("string[@key='OfferID']").set('value','o2')
        o=m.analyse_trace(t)[6]
        self.assertEqual((o[0]['label'],o[0]['later_accepted_event']),('true',False))
        self.assertEqual((o[1]['label'],o[1]['later_accepted_event']),('false',True))

    def test_out_of_order_timestamp_detected(self):
        t=fixture(); t[-1].find("string[@key='time:timestamp']").set('value','2016-01-01T00:00:00Z')
        self.assertEqual(m.analyse_trace(t)[4],1)

    def test_missing_timezone_rejected(self):
        with self.assertRaises(ValueError):m.timestamp('2016-01-01T00:00:00')

    def test_group_boundary_and_count_reconciliation(self):
        log=ET.Element('log')
        ET.SubElement(log,'int',key='meta_concept:named_events_total',value='6')
        log.append(fixture())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.gz';p.write_bytes(gzip.compress(ET.tostring(log)))
            r=m.audit(p)
            self.assertEqual(r['integrity_status'],'PASS')
            self.assertEqual(r['reference_boundary']['crossing_applications'],1)
            self.assertEqual(r['reference_boundary']['offer_counts'],{'before':1,'on_or_after':1})
            self.assertEqual(r['readiness_status'],'REVIEW_REQUIRED')

    def test_bad_embedded_count_is_not_pass(self):
        log=ET.Element('log');ET.SubElement(log,'int',key='meta_concept:named_events_total',value='7');log.append(fixture())
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.gz';p.write_bytes(gzip.compress(ET.tostring(log)))
            self.assertEqual(m.audit(p)['integrity_status'],'REVIEW_REQUIRED')

    def test_truncated_archive_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.gz';p.write_bytes(gzip.compress(b'<log><trace/>')[:-5])
            with self.assertRaises((EOFError,ET.ParseError)):m.audit(p)


if __name__=='__main__':unittest.main()
