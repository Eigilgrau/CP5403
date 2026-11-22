import json
from pathlib import Path
import unittest
from unittest.mock import patch

import phm_explanation_judge as review


class ExplanationJudgeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent.parent
        config=json.loads((cls.root/'Files/experiment_config.json').read_text(encoding='utf-8'))
        model,revision=review.judge.PROFILES['qwen14b']
        config['generator'].update(model=model,revision=revision,max_input_tokens=6144,max_new_tokens=256)
        read=Path.read_text
        def guard(path,*args,**kwargs):
            if 'answers' in path.name or 'private' in path.name or path.name=='paired_results.json':
                raise AssertionError('Private scoring data accessed')
            return read(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):
            cls.reqs,cls.recs,cls.mapping,cls.manifest=review.prepare(cls.root,config)

    def test_coverage_and_blinding(self):
        self.assertEqual(sum(q['task']=='grounding' for q in self.reqs),15)
        self.assertEqual(sum(m['review_status']=='skipped_unparseable' for m in self.manifest),1)
        for q in self.reqs:
            data=json.loads(q['messages'][1]['content'])
            self.assertNotIn('condition',data)
            self.assertNotIn('reference_label',q['messages'][1]['content'])
            for r in data['python_verified_relations']:
                self.assertIn(r['feature'].replace('_',' ').lower(),data['statement'].replace('_',' ').lower())

    def test_duplicate_refs_preserve_verdict_without_repair(self):
        rec=self.recs[0];req=rec['_request'];ref=req['output_schema']['properties']['evidence_refs']['items']['enum'][0]
        answer={'verdict':'supported','basis':'numeric_relation','evidence_refs':[ref,ref]}
        result=review.validate(json.dumps(answer),rec)
        self.assertTrue(result['verdict_valid']);self.assertFalse(result['strict_valid'])
        self.assertEqual(result['parsed'],answer)
        self.assertEqual(result['reference_defects'],['duplicate_reference_ids'])
        answer['evidence_refs']=['invented']
        self.assertEqual(review.validate(json.dumps(answer),rec)['reference_defects'],['unknown_reference_ids'])
        self.assertFalse(review.validate('{',rec)['verdict_valid'])

    def test_failure_denominators_and_completion_are_visible(self):
        details=[]
        for i,q in enumerate(self.reqs):
            verdict='supported' if q['task']=='grounding' else 'supports'
            answer={'verdict':verdict,'basis':'direct_text','evidence_refs':[]}
            checked=review.validate(json.dumps(answer),self.recs[i]);checked['status']='valid'
            if i==0:checked={'status':'invalid_response','verdict_valid':False,'strict_valid':False,'parsed':None}
            details.append({'request_id':q['request_id'],'checks':checked,'response':{}})
        metrics,rows=review.summarise(details,self.mapping,self.manifest)
        self.assertEqual(metrics['conditions']['C']['source_explanations'],3)
        self.assertEqual(metrics['conditions']['C']['parseable_explanations'],2)
        self.assertEqual(metrics['conditions']['C']['grounding']['attempts'],6)
        self.assertEqual(metrics['conditions']['C']['grounding']['verdict_counts']['judge_failure'],1)
        self.assertEqual(metrics['conditions']['C']['grounding']['supported_fraction_all_attempts'],5/6)


if __name__=='__main__':unittest.main()
