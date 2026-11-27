import json
from pathlib import Path
import unittest
from unittest.mock import patch
import phm_design_smoke as smoke
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
        r=self.r[0]
        return {'predicted_class':'NoRecordedTarget','selected_features':[next(iter(r['feature_values']))],
                'selected_case_id':r['historical_cases'][0]['case_id'],'supporting_case_ids':[],
                'conflicting_case_ids':[],'literature_citations':[], 'decision_justification':'Evidence remains uncertain.'}

    def test_four_calls_and_two_conditions(self):
        self.assertEqual(len(self.q),4)
        self.assertEqual({q['condition'] for q in self.q},{'C','D'})
        self.assertEqual(len({q['case_id'] for q in self.q}),2)

    def test_parser_prevents_character_161(self):
        # Exercise the actual generation parser, not merely a post-generation validator.
        p=json_parser({'type':'string','minLength':1,'maxLength':160})
        for ch in '"'+'x'*160:
            self.assertIn(ch,p.get_allowed_characters());p=p.add_character(ch)
        self.assertNotIn('x',p.get_allowed_characters())
        self.assertIn('"',p.get_allowed_characters())

    def test_full_schema_accepts_boundary_and_validator_rejects_excess(self):
        value=self.sample();value['decision_justification']='x'*160
        raw=json.dumps(value);p=json_parser(self.q[0]['output_schema'])
        for ch in raw:
            self.assertIn(ch,p.get_allowed_characters());p=p.add_character(ch)
        self.assertTrue(p.can_end())
        self.assertTrue(smoke.validate(raw,self.r[0])['valid'])
        value['decision_justification']+='x'
        self.assertFalse(smoke.validate(json.dumps(value),self.r[0])['valid'])

    def test_unknown_evidence_and_incomplete_json_rejected(self):
        v=self.sample();v['supporting_case_ids']=['invented']
        self.assertFalse(smoke.validate(json.dumps(v),self.r[0])['valid'])
        self.assertFalse(smoke.validate('{"predicted_class":',self.r[0])['valid'])

    def test_scoring_refused(self):
        with self.assertRaises(ValueError):smoke.capture(None,None,None,[],[],score=True)


if __name__=='__main__':unittest.main()
