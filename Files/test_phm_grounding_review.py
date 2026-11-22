import json
from pathlib import Path
import unittest
from unittest.mock import patch

import phm_grounding_review as review


class GroundingChecks(unittest.TestCase):
    def test_real_pack_keeps_format_failures_but_skips_truncation(self):
        root=Path(__file__).resolve().parent.parent
        run=root/'Outputs/runs/20261122T012217_paired_CD_two_stage_validation_dc07454d'
        original=Path.read_text
        def guard(path,*args,**kwargs):
            if 'private' in path.name or path.name=='paired_results.json':
                raise AssertionError('Scoring labels accessed')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):
            requests,manifest,manual=review.build(run)
        self.assertEqual(len(requests),5)
        self.assertEqual(sum(x.get('statements',0) for x in manifest),15)
        self.assertEqual(sum(x['review_status']=='skipped_unparseable' for x in manifest),1)
        for q in requests:
            payload=json.loads(q['messages'][1]['content'])
            self.assertNotIn('reference_label',q['messages'][1]['content'])
            self.assertNotIn('saved_qwen_decision',payload['evidence'])
            self.assertNotIn('condition',payload)
            self.assertNotIn('expected_verdict',q['messages'][1]['content'])
        self.assertTrue(all(a['verdict'] is None for row in manual for a in row['assessments']))

    def test_calibration_keys_are_separate_and_citations_are_independent(self):
        requests,answers=review.calibration()
        self.assertEqual(len(requests),8)
        self.assertEqual(len({q['request_id'] for q in requests}),8)
        for q in requests:
            self.assertNotIn('expected_verdict',review.canonical(q))
        self.assertEqual(answers[6]['expected_verdict'],'supported')
        self.assertEqual(answers[6]['expected_citation_verdicts'],['does_not_support'])

    def test_judge_output_requires_full_coverage_and_real_sources(self):
        q=review.calibration()[0][0]
        item={'statement_id':'s1','verdict':'supported','rationale':'Query slope is lower.',
              'evidence_refs':['QUERY','case_A'],
              'citation_checks':[{'source_id':'case_A','verdict':'supports'}]}
        self.assertTrue(review.validate_judgment(q,{'assessments':[item]}))
        for bad in [{'assessments':[]},{'assessments':[dict(item,evidence_refs=['invented'])]},
                    {'assessments':[dict(item,citation_checks=[])]},
                    {'assessments':[dict(item,statement_id='s2')]}]:
            with self.assertRaises(ValueError):review.validate_judgment(q,bad)


if __name__=='__main__':
    unittest.main()
