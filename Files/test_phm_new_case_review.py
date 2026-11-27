import json
from pathlib import Path
import unittest
from unittest.mock import patch
import phm_new_case_review as runner

class NewCaseReviewChecks(unittest.TestCase):
    def test_prepare_avoids_private_data_and_preserves_design(self):
        root=Path(__file__).resolve().parent.parent
        rt,rb=Path.read_text,Path.read_bytes
        def guard(method):
            def call(path,*a,**kw):
                if 'private_' in path.name or path.suffix=='.duckdb' or 'heldout_test' in str(path):
                    raise AssertionError('Private outcomes accessed')
                return method(path,*a,**kw)
            return call
        with patch.object(Path,'read_text',guard(rt)),patch.object(Path,'read_bytes',guard(rb)):
            requests,records=runner.prepare(root)
        self.assertEqual(len(requests),12)
        self.assertEqual(len({q['case_id'] for q in requests}),6)
        original=runner.base.read_rows(root/runner.design.OUTPUT/'reviewed_candidate_requests.jsonl')
        byid={q['request_id']:q for q in original}
        for q,r in zip(requests,records):
            self.assertEqual(q,byid[q['request_id']])
            self.assertNotIn('reference_label',r)
            v={'predicted_class':'NoRecordedTarget','supporting_evidence_ids':['T01','K001'],
                'challenging_evidence_ids':['T01'],'limitation_codes':['overlapping_class_distributions'],
                'literature_context_ids':[]}
            self.assertTrue(runner.validate(json.dumps(v),r)['valid'])

    def test_scoring_is_disabled(self):
        with self.assertRaises(ValueError):runner.capture(None,None,None,[],[],score=True)

if __name__=='__main__':unittest.main()
