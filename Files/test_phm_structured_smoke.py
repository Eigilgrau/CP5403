import json
from pathlib import Path
import unittest
from unittest.mock import patch
import phm_structured_smoke as smoke
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
        r=self.r[0];visible=json.loads(self.q[0]['messages'][1]['content'])
        h=visible['historical_cases'][0];feature='ch2_median_of_component_changes'
        x=visible['query']['feature_values'][feature];y=h['feature_values'][feature]
        return {'predicted_class':'NoRecordedTarget','selected_features':[feature],
                'selected_case_id':h['case_id'],'supporting_case_ids':[],
                'conflicting_case_ids':[],'literature_citations':[],
                'comparison_feature':feature,'comparison_relation':'lower' if x<y else 'higher' if x>y else 'equal'}

    def test_exactly_four_requests_without_free_text(self):
        self.assertEqual(len(self.q),4)
        self.assertEqual(len({q['case_id'] for q in self.q}),2)
        for q in self.q:self.assertNotIn('decision_justification',q['output_schema']['properties'])

    def test_real_schema_and_numeric_verification(self):
        raw=json.dumps(self.sample());p=json_parser(self.q[0]['output_schema'])
        for ch in raw:
            self.assertIn(ch,p.get_allowed_characters());p=p.add_character(ch)
        self.assertTrue(p.can_end())
        result=smoke.validate(raw,self.r[0])
        self.assertTrue(result['valid'],result)
        self.assertIsNotNone(result['verified_selected_comparison'])

    def test_wrong_numeric_relation_rejected(self):
        value=self.sample();value['comparison_relation']='equal' if value['comparison_relation']!='equal' else 'higher'
        result=smoke.validate(json.dumps(value),self.r[0])
        self.assertFalse(result['valid'])
        self.assertIn('comparison_relation_disagrees',str(result['errors']))

    def test_abstention_is_permitted(self):
        value=self.sample();value.update(predicted_class=smoke.base.ABSTAIN,selected_features=[],selected_case_id=None,comparison_feature=None,comparison_relation=None)
        result=smoke.validate(json.dumps(value),self.r[0]);self.assertTrue(result['valid'],result)

    def test_unknown_evidence_and_incomplete_json_rejected(self):
        v=self.sample();v['supporting_case_ids']=['invented']
        self.assertFalse(smoke.validate(json.dumps(v),self.r[0])['valid'])
        self.assertFalse(smoke.validate('{"predicted_class":',self.r[0])['valid'])

    def test_scoring_refused(self):
        with self.assertRaises(ValueError):smoke.capture(None,None,None,[],[],score=True)


if __name__=='__main__':unittest.main()
