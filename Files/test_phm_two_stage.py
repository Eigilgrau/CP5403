import json
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import phm_two_stage as pipeline
from phm_output_schema import json_parser


class TwoStageChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.config = json.loads((cls.root/'Files/experiment_config.json').read_text(encoding='utf-8'))
        cls.config['generator'].update(max_input_tokens=6144, max_new_tokens=384)
        original = Path.read_text
        def guard(path, *args, **kwargs):
            if 'private' in path.name:
                raise AssertionError('Private labels read before scoring')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guard):
            cls.prepared = pipeline.prepare(cls.root, cls.config, 3)
        cls.contract, cls.records, cls.requests, _ = cls.prepared

    def decision(self, index=0):
        r = self.records[index]
        return {'predicted_class':'NoRecordedTarget',
                'selected_features':[next(k for k,v in r['feature_values'].items() if v is not None)],
                'selected_case_id':r['historical_cases'][0]['case_id'],
                'supporting_case_ids':[r['historical_cases'][0]['case_id']],
                'conflicting_case_ids':[r['historical_cases'][1]['case_id']], 'literature_citations':[]}

    def explanation(self):
        return {'claims':[{'text':'Historical analogies provide limited support for the saved decision.',
                           'case_ids':[self.records[0]['historical_cases'][0]['case_id']], 'literature_ids':[]}],
                'limitation':{'text':'Anonymous channels prevent physical interpretation.', 'case_ids':[], 'literature_ids':[]}}

    def test_stage_one_has_no_prose_and_preserves_pairing(self):
        for c,d in zip(self.requests[:3],self.requests[3:]):
            self.assertEqual(c['messages'][0],d['messages'][0])
            self.assertEqual(c['generator'],d['generator'])
            a=json.loads(c['messages'][1]['content']);b=json.loads(d['messages'][1]['content']);b.pop('literature')
            self.assertEqual(a,b)
            self.assertNotIn('explanation',c['output_schema']['properties'])

    def test_valid_decision_assembles_numbers_without_prose(self):
        checked=pipeline.validate(json.dumps(self.decision()),self.records[0])
        self.assertTrue(checked['valid'],checked['errors'])
        self.assertEqual(checked['numerical_evidence_origin'],'deterministic_source_lookup')
        self.assertTrue(checked['numerical_evidence']['comparison_claims'])
        for key,value in [('predicted_class','invented'),('selected_features',['invented']),
                          ('literature_citations',['invented']),('supporting_case_ids',['invented'])]:
            d=self.decision();d[key]=value
            self.assertFalse(pipeline.validate(json.dumps(d),self.records[0])['valid'])
        d=self.decision();d['conflicting_case_ids']=d['supporting_case_ids']
        self.assertFalse(pipeline.validate(json.dumps(d),self.records[0])['valid'])

    def test_second_stage_keeps_all_evidence_and_cannot_change_class(self):
        d=self.decision();q=pipeline.explanation_request(self.requests[0],d)
        payload=json.loads(q['messages'][1]['content']);old=json.loads(self.requests[0]['messages'][1]['content'])
        self.assertEqual(payload.pop('saved_qwen_decision'),d);self.assertEqual(payload,old)
        rec={**self.records[0],'_request':q}
        e=self.explanation();self.assertTrue(pipeline.validate(json.dumps(e),rec)['valid'])
        e['predicted_class']='RuntimeExcessive'
        self.assertFalse(pipeline.validate(json.dumps(e),rec)['valid'])
        e=self.explanation();e['claims'][0]['literature_ids']=['invented']
        self.assertFalse(pipeline.validate(json.dumps(e),rec)['valid'])
        e=self.explanation();e['claims'][0]['text']='The value is 900.'
        self.assertFalse(pipeline.validate(json.dumps(e),rec)['valid'])

    def test_cache_identity_depends_on_saved_decision(self):
        d=self.decision();a=pipeline.explanation_request(self.requests[0],d)
        d['predicted_class']='RuntimeExcessive';b=pipeline.explanation_request(self.requests[0],d)
        self.assertNotEqual(a['request_id'],b['request_id'])

    def test_known_channel_numbers_are_not_measurements(self):
        q=pipeline.explanation_request(self.requests[0],self.decision())
        rec={**self.records[0],'_request':q}
        e=self.explanation();e['claims'][0]['text']='Channel 2 varies; channel 5 has an unknown physical meaning.'
        self.assertTrue(pipeline.validate(json.dumps(e),rec)['valid'])
        for text in ['Channel 2 has a value of 5.', 'Channel 99 varies.']:
            e['claims'][0]['text']=text
            self.assertFalse(pipeline.validate(json.dumps(e),rec)['valid'])

    def test_schema_accepts_both_stages(self):
        for request,data in [(self.requests[0],self.decision()),
                             (pipeline.explanation_request(self.requests[0],self.decision()),self.explanation())]:
            parser=json_parser(request['output_schema'])
            for char in json.dumps(data):
                self.assertIn(char,parser.get_allowed_characters());parser=parser.add_character(char)
            self.assertTrue(parser.can_end())

    def test_end_to_end_failure_preserves_classification_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'Files').mkdir()
            (root/'Files/experiment_config.json').write_text(json.dumps(self.config),encoding='utf-8')
            bundle=root/'Datasets/PHM15-diagnosis/validation_pilot_v1';bundle.mkdir(parents=True)
            shutil.copy2(self.root/'Datasets/PHM15-diagnosis/validation_pilot_v1/private_validation_labels.jsonl',bundle)
            def fake_infer(reqs,recs,cache,config,report,retry_errors,**kwargs):
                cache.mkdir(parents=True,exist_ok=True)
                for i,(q,r) in enumerate(zip(reqs,recs)):
                    if q['task_stage']=='decision':
                        raw='{' if i==1 else json.dumps(self.decision(i))
                    else:
                        e=self.explanation();e['claims'][0]['case_ids']=[r['historical_cases'][0]['case_id']]
                        raw='{"claims":[' if i==0 else json.dumps(e)
                    response={'origin':'model_inference','raw_output':raw,'raw_output_sha256':pipeline.base.fingerprint(raw),
                              'error':None,**{k:q[k] for k in ('request_id','case_id','condition','generator','evidence_sha256')}}
                    (cache/(q['request_id']+'.json')).write_text(json.dumps(response),encoding='utf-8')
            with patch.object(pipeline,'prepare',return_value=self.prepared), patch.object(pipeline,'_infer',side_effect=fake_infer):
                first=pipeline.run(root,3)
            self.assertEqual(first['C']['classification_valid'],2)
            self.assertEqual(first['C']['explanation_valid'],1)
            self.assertEqual(first['C']['full_output_valid'],1)
            # Actual inference helper must return from cache without importing a GPU model.
            with patch.object(pipeline,'prepare',return_value=self.prepared),patch.dict('sys.modules',{'torch':None,'transformers':None}):
                resumed=pipeline.run(root,3)
            self.assertEqual(first,resumed)
            latest=json.loads((root/'Outputs/LATEST_RUN.json').read_text(encoding='utf-8'))
            self.assertEqual(latest['status'],'completed_with_issues')


if __name__=='__main__':
    unittest.main()
