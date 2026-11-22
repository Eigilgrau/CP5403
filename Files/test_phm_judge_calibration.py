import json
from copy import deepcopy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import phm_judge_calibration as judge
from phm_output_schema import json_parser


class JudgeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent.parent
        cls.pack=cls.root/'Outputs/reviews/grounding_v1'
        cls.config=json.loads((cls.root/'Files/experiment_config.json').read_text(encoding='utf-8'))
        cls.config['generator'].update(model=judge.MODEL,revision=judge.REVISION,max_input_tokens=6144,max_new_tokens=512)
        original=Path.read_text
        def guard(path,*args,**kwargs):
            if 'answers' in path.name or 'private' in path.name:
                raise AssertionError('Answer leakage during preparation')
            return original(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):
            cls.requests,cls.records=judge.prepare(cls.pack,cls.config)
        cls.answers=[json.loads(line) for line in (cls.pack/'calibration_answers_DO_NOT_SEND_TO_JUDGE.jsonl').read_text(encoding='utf-8').splitlines()]

    def fixture(self,index):
        expected=self.answers[index]
        statement=json.loads(self.requests[index]['messages'][1]['content'])['statements'][0]
        return {'assessments':[{'statement_id':'s1','verdict':expected['expected_verdict'],
                              'rationale':'Calibration fixture stored only in a disposable test directory.',
                              'evidence_refs':['QUERY','case_A'],
                              'citation_checks':[{'source_id':c,'verdict':v} for c,v in zip(statement['cited_sources'],expected['expected_citation_verdicts'])]}]}

    def test_schema_and_python_validation_agree(self):
        for i,q in enumerate(self.requests):
            raw=json.dumps(self.fixture(i));parser=json_parser(q['output_schema'])
            for char in raw:
                self.assertIn(char,parser.get_allowed_characters());parser=parser.add_character(char)
            self.assertTrue(parser.can_end())
            self.assertTrue(judge.validate(raw,self.records[i])['valid'])

    def test_invalid_output_counts_as_failure(self):
        details=[{'source_request_id':q['source_request_id'],'checks':{'status':'valid','parsed':self.fixture(i)}} for i,q in enumerate(self.requests)]
        details[0]['checks']={'status':'invalid_response','parsed':None}
        scored=judge.score(details,self.answers)
        self.assertEqual(scored['attempts'],8)
        self.assertEqual(scored['fully_matching_examples'],7)

    def test_qwen_uses_same_prompts_but_distinct_identity_and_cache(self):
        cfg=deepcopy(self.config);profile=judge.PROFILES['qwen']
        cfg['generator'].update(model=profile['model'],revision=profile['revision'])
        cfg['judge_version']=profile['version']
        requests,_=judge.prepare(self.pack,cfg)
        for llama,qwen in zip(self.requests,requests):
            self.assertEqual(llama['messages'],qwen['messages'])
            self.assertEqual(llama['output_schema'],qwen['output_schema'])
            self.assertNotEqual(llama['request_id'],qwen['request_id'])
        self.assertNotEqual(profile['cache'],judge.PROFILES['llama']['cache'])

    def test_cached_run_needs_no_gpu_and_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'Files').mkdir()
            (root/'Files/experiment_config.json').write_text(json.dumps(self.config),encoding='utf-8')
            pack=root/'Outputs/reviews/grounding_v1';pack.mkdir(parents=True)
            for name in ['calibration_requests.jsonl','calibration_answers_DO_NOT_SEND_TO_JUDGE.jsonl']:
                shutil.copy2(self.pack/name,pack/name)
            cache=root/'Outputs/judges/llama_grounding_calibration_v1/responses';cache.mkdir(parents=True)
            for i,q in enumerate(self.requests):
                raw=json.dumps(self.fixture(i))
                envelope={'origin':'model_inference','raw_output':raw,'raw_output_sha256':judge.base.fingerprint(raw),'error':None,
                          **{k:q[k] for k in ['request_id','case_id','condition','generator','evidence_sha256']}}
                (cache/(q['request_id']+'.json')).write_text(json.dumps(envelope),encoding='utf-8')
            with patch.dict('sys.modules',{'torch':None,'transformers':None}):
                result=judge.run(root)
            self.assertEqual(result['fully_matching_examples'],8)
            q=self.requests[0];path=cache/(q['request_id']+'.json')
            envelope=json.loads(path.read_text());envelope['raw_output']='{}';path.write_text(json.dumps(envelope))
            self.assertEqual(judge.inspect(path,q,self.records[0])[0]['status'],'invalid_cache')


if __name__=='__main__':
    unittest.main()
