import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import phm_judge_v2 as judge
from phm_output_schema import json_parser


class JudgeV2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent.parent
        cls.pack=cls.root/'Outputs/reviews/grounding_v2'
        cls.config=json.loads((cls.root/'Files/experiment_config.json').read_text(encoding='utf-8'))
        model,revision=judge.PROFILES['qwen4b']
        cls.config['generator'].update(model=model,revision=revision,max_input_tokens=6144,max_new_tokens=160)
        read=Path.read_text
        def guard(path,*args,**kwargs):
            if 'answers' in path.name or 'private' in path.name:raise AssertionError('Answer leak')
            return read(path,*args,**kwargs)
        with patch.object(Path,'read_text',guard):
            cls.reqs,cls.recs=judge.prepare(cls.pack,cls.config)

    def test_computed_relations_and_missing_values(self):
        ev={'query':{'feature_values':{'a':-2,'b':None,'c':True}},'historical_cases':[{'case_id':'h','feature_values':{'a':-6,'b':4,'c':1}}]}
        out=judge.numeric_relations(ev)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['signed_relation'],'higher')
        self.assertEqual(out[0]['absolute_magnitude_relation'],'lower')

    def test_citation_prompt_is_limited_to_designated_source(self):
        self.assertEqual(len(self.reqs),36)
        for q in self.reqs:
            data=json.loads(q['messages'][1]['content'])
            self.assertNotIn('expected',data)
            if q['task']=='citation':
                ev=data['evidence'];ids=[h['case_id'] for h in ev['historical_cases']]+[p['chunk_id'] for p in ev['literature']]
                self.assertEqual(ids,[q['source_id']])

    def test_schema_and_reference_validation(self):
        raw=json.dumps({'verdict':'supported','basis':'numeric_relation','evidence_refs':['QUERY','case_A']})
        self.assertTrue(judge.validate(raw,self.recs[0])['valid'])
        parser=json_parser(self.reqs[0]['output_schema'])
        for c in raw:
            self.assertIn(c,parser.get_allowed_characters());parser=parser.add_character(c)
        self.assertTrue(parser.can_end())
        bad=json.dumps({'verdict':'supported','basis':'numeric_relation','evidence_refs':['invented']})
        self.assertFalse(judge.validate(bad,self.recs[0])['valid'])

    def test_cached_consolidated_run_and_failure_denominators(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'Files').mkdir()
            (root/'Files/experiment_config.json').write_text(json.dumps(self.config),encoding='utf-8')
            pack=root/'Outputs/reviews/grounding_v2';pack.mkdir(parents=True)
            for name in ['inputs.jsonl','answers_DO_NOT_SEND_TO_JUDGE.jsonl']:shutil.copy2(self.pack/name,pack/name)
            answers={a['item_id']:a for a in judge.load_rows(pack/'answers_DO_NOT_SEND_TO_JUDGE.jsonl')}
            cache=root/'Outputs/judges/grounding_v2/qwen4b/responses';cache.mkdir(parents=True)
            for i,q in enumerate(self.reqs):
                a=answers[q['case_id']];verdict=a['grounding'] if q['task']=='grounding' else a['citations'][q['source_id']]
                raw='{' if i==0 else json.dumps({'verdict':verdict,'basis':'direct_text','evidence_refs':[]})
                envelope={'origin':'model_inference','error':None,'raw_output':raw,'raw_output_sha256':judge.base.fingerprint(raw),
                          **{k:q[k] for k in ['request_id','case_id','condition','generator','evidence_sha256']}}
                (cache/(q['request_id']+'.json')).write_text(json.dumps(envelope),encoding='utf-8')
            with patch.dict('sys.modules',{'torch':None,'transformers':None}):
                result=judge.run(root,('qwen4b',))
            self.assertEqual(result['qwen4b']['original_grounding'],{'attempts':8,'valid':7,'matches':7})
            self.assertEqual(result['qwen4b']['new_grounding']['attempts'],12)


if __name__=='__main__':unittest.main()
