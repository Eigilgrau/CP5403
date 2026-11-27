import json
from pathlib import Path
import unittest
from unittest.mock import patch
import phm_evidence_menu_smoke as smoke
from phm_output_schema import json_parser


class SmokeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent.parent
        read_text,read_bytes=Path.read_text,Path.read_bytes
        def guard(method):
            def read(path,*args,**kwargs):
                if 'private_' in path.name or path.suffix=='.duckdb' or 'heldout_test' in str(path):
                    raise AssertionError('Private/test data accessed')
                return method(path,*args,**kwargs)
            return read
        with patch.object(Path,'read_text',guard(read_text)),patch.object(Path,'read_bytes',guard(read_bytes)):
            cls.q,cls.r=smoke.prepare(root)

    def sample(self):
        return {'predicted_class':'NoRecordedTarget','selected_comparison_id':'K001','literature_citations':[]}

    def test_four_calls_and_no_free_text(self):
        self.assertEqual(len(self.q),4)
        for q in self.q:self.assertEqual(set(q['output_schema']['properties']),set(self.sample()))

    def test_menu_matches_all_visible_values(self):
        for q,r in zip(self.q,self.r):
            d=json.loads(q['messages'][1]['content']);rows,resolved=smoke.menu(d)
            self.assertEqual(len(rows),93)
            for key,value in resolved.items():
                raw=json.dumps({**self.sample(),'selected_comparison_id':key})
                result=smoke.validate(raw,r)
                self.assertTrue(result['valid'],result)
                self.assertEqual(result['verified_selected_comparison']['query_value'],value['query_value'])
                x,y=value['query_value'],value['historical_value']
                self.assertEqual(value['relation'],'lower' if x<y else 'higher' if x>y else 'equal')

    def test_real_generation_parser_accepts_output(self):
        p=json_parser(self.q[0]['output_schema'])
        for ch in json.dumps(self.sample()):
            self.assertIn(ch,p.get_allowed_characters());p=p.add_character(ch)
        self.assertTrue(p.can_end())

    def test_unknown_id_null_and_extra_prose_rejected(self):
        for value in [{**self.sample(),'selected_comparison_id':'K999'},
                      {**self.sample(),'selected_comparison_id':None},
                      {**self.sample(),'explanation':'invented'},
                      {**self.sample(),'predicted_class':'unknown'}]:
            self.assertFalse(smoke.validate(json.dumps(value),self.r[0])['valid'])

    def test_duplicate_fields_rejected(self):
        raw=json.dumps(self.sample())[:-1]+',"predicted_class":"RuntimeExcessive"}'
        self.assertFalse(smoke.validate(raw,self.r[0])['valid'])

    def test_citations_membership_and_duplicates(self):
        for ids in [['invented'],['invented','invented']]:
            self.assertFalse(smoke.validate(json.dumps({**self.sample(),'literature_citations':ids}),self.r[0])['valid'])
        for q,r in zip(self.q,self.r):
            if q['condition']=='D':
                key=q['literature'][0]['chunk_id']
                self.assertTrue(smoke.validate(json.dumps({**self.sample(),'literature_citations':[key]}),r)['valid'])
                self.assertFalse(smoke.validate(json.dumps({**self.sample(),'literature_citations':[key,key]}),r)['valid'])

    def test_abstention_and_override_not_forced(self):
        v={**self.sample(),'predicted_class':smoke.base.ABSTAIN,'selected_comparison_id':None}
        self.assertTrue(smoke.validate(json.dumps(v),self.r[0])['valid'])
        v={**self.sample(),'predicted_class':'RuntimeExcessive'}
        result=smoke.validate(json.dumps(v),self.r[0])
        self.assertTrue(result['valid']);self.assertEqual(result['decision_relation_to_score'],'overrides')

    def test_tampered_menu_rejected(self):
        from copy import deepcopy
        r=deepcopy(self.r[0]);d=json.loads(r['_request']['messages'][1]['content'])
        d['comparison_menu'][0][3]='invalid';r['_request']['messages'][1]['content']=json.dumps(d)
        self.assertFalse(smoke.validate(json.dumps(self.sample()),r)['valid'])

    def test_scoring_refused(self):
        with self.assertRaises(ValueError):smoke.capture(None,None,None,[],[],score=True)

    def test_missing_values_not_offered_as_comparisons(self):
        d=json.loads(self.q[0]['messages'][1]['content'])
        d['query']['feature_values'][d['feature_order'][0]]=None
        rows,resolved=smoke.menu(d)
        self.assertEqual(len(rows),90)


if __name__=='__main__':unittest.main()
