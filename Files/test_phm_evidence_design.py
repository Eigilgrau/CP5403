import json
from pathlib import Path
import unittest
from unittest.mock import patch
import phm_evidence_design as d
from phm_output_schema import json_parser

class DesignChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent.parent
        original=Path.read_text
        def guarded(path,*args,**kwargs):
            if 'private_' in path.name or 'heldout_test' in str(path):raise AssertionError('Forbidden outcomes accessed')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guarded):
            cls.requests=d.base.read_rows(root/d.OUTPUT/'reviewed_candidate_requests.jsonl')
        cls.q=next(q for q in cls.requests if q['condition']=='C')

    def sample(self):
        return {'predicted_class':'NoRecordedTarget','supporting_evidence_ids':['T01','K001'],
                'challenging_evidence_ids':['T01'],'limitation_codes':['overlapping_class_distributions'],
                'literature_context_ids':[]}

    def test_all_candidate_schemas_and_fact_resolution(self):
        for q in self.requests:
            raw=json.dumps(self.sample());parser=json_parser(q['output_schema'])
            for ch in raw:
                self.assertIn(ch,parser.get_allowed_characters());parser=parser.add_character(ch)
            self.assertTrue(parser.can_end())
            result=d.validate(raw,q)
            self.assertTrue(result['valid'],result)
            self.assertFalse(result['semantic_grounding_established'])

    def test_unknown_and_duplicate_references(self):
        for refs in [['T99'],['T01','T01']]:
            value=self.sample();value['supporting_evidence_ids']=refs
            self.assertFalse(d.validate(json.dumps(value),self.q)['valid'])

    def test_bare_decision_not_accepted(self):
        value=self.sample();value['supporting_evidence_ids']=[]
        self.assertFalse(d.validate(json.dumps(value),self.q)['valid'])

    def test_override_remains_llm_choice_and_review_required(self):
        value=self.sample();payload=json.loads(self.q['messages'][1]['content'])
        top=max(payload['query']['class_scores'],key=payload['query']['class_scores'].get)
        value['predicted_class']=next(c for c in d.base.LABELS if c!=top)
        result=d.validate(json.dumps(value),self.q)
        self.assertTrue(result['valid']);self.assertTrue(result['overrides_numerical_ranking'])
        self.assertTrue(result['review_warnings']);self.assertFalse(result['diagnostic_sufficiency_established'])

    def test_abstention_with_limitation_allowed(self):
        value=self.sample();value.update(predicted_class=d.base.ABSTAIN,supporting_evidence_ids=[],challenging_evidence_ids=[])
        self.assertTrue(d.validate(json.dumps(value),self.q)['valid'])

    def test_literature_context_not_diagnostic_proof(self):
        q=next(q for q in self.requests if q['condition']=='D');value=self.sample()
        value['literature_context_ids']=[q['literature'][0]['chunk_id']]
        result=d.validate(json.dumps(value),q);self.assertTrue(result['valid'])
        self.assertTrue(any('entailment' in x for x in result['review_warnings']))
        self.assertFalse(d.validate(json.dumps(value),self.q)['valid'])

    def test_duplicate_json_keys_rejected(self):
        raw=json.dumps(self.sample())[:-1]+',"predicted_class":"RuntimeExcessive"}'
        self.assertFalse(d.validate(raw,self.q)['valid'])

if __name__=='__main__':unittest.main()
