import json
import unittest
import tempfile
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import phm_selection_pilot as pilot
from phm_evidence_selection import validate_selection
from phm_output_schema import json_parser


class SelectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.config = json.loads((cls.root / 'Files/experiment_config.json').read_text(encoding='utf-8'))
        cls.config['generator']['max_input_tokens'] = 6144
        original = Path.read_text
        def guard(path, *a, **kw):
            if path.name == 'private_validation_labels.jsonl':
                raise AssertionError('Labels read during preparation')
            return original(path, *a, **kw)
        with patch.object(Path, 'read_text', guard):
            cls.contract, cls.records, cls.cs, cls.ds, cls.retrieval = pilot.prepare(cls.root, cls.config, 3)

    def fixture(self, index=0):
        r = self.records[index]
        return {'predicted_class': 'NoRecordedTarget', 'selected_features': [next(n for n,v in r['feature_values'].items() if v is not None)],
                'selected_case_id': r['historical_cases'][0]['case_id'], 'literature_citations': [],
                'explanation': 'The model score and historical analogy provide limited evidence.',
                'limitation': 'Anonymous channel meanings prevent a physical interpretation.'}

    def test_selected_case_alone_supplies_all_comparisons(self):
        for case in self.records[0]['historical_cases']:
            f = self.fixture(); f['selected_case_id'] = case['case_id']
            checked = validate_selection(json.dumps(f), self.records[0])
            self.assertTrue(checked['valid'], checked['errors'])
            self.assertEqual(checked['parsed']['comparison_claims'], pilot.base.comparison_claims(self.records[0], case))
            self.assertEqual(checked['model_selection'], f)

    def test_rejects_unknown_duplicate_and_injected_values(self):
        for key, value in [('selected_case_id','invented'), ('selected_features',['invented']),
                           ('literature_citations',['invented']), ('comparison_claims',[])]:
            f=self.fixture();f[key]=value
            self.assertFalse(validate_selection(json.dumps(f),self.records[0])['valid'])
        f=self.fixture();f['selected_features']*=2
        self.assertFalse(validate_selection(json.dumps(f),self.records[0])['valid'])

    def test_channel_shorthand_is_not_a_numerical_claim(self):
        f=self.fixture();f['explanation']='The ch2_iqr shorthand describes channel variability; physical meaning remains unknown.'
        self.assertTrue(validate_selection(json.dumps(f),self.records[0])['valid'])
        f['explanation']='The score is 0.9.'
        self.assertIn('numerical_prose_not_allowed:explanation',validate_selection(json.dumps(f),self.records[0])['errors'])

    def test_schema_accepts_selection_and_has_no_prose_character_cap(self):
        parser=json_parser(self.cs[0]['output_schema']); raw=json.dumps(self.fixture())
        for char in raw:
            self.assertIn(char,parser.get_allowed_characters());parser=parser.add_character(char)
        self.assertTrue(parser.can_end())
        self.assertNotIn('maxLength',self.cs[0]['output_schema']['properties']['explanation'])
        for c,d in zip(self.cs,self.ds):
            self.assertEqual(c['evidence_sha256'],d['evidence_sha256'])
            self.assertEqual(c['generator'],d['generator'])
            a=json.loads(c['messages'][1]['content']);b=json.loads(d['messages'][1]['content']);b.pop('literature')
            self.assertEqual(a,b)

    def test_cached_end_to_end_scoring_without_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'Files').mkdir()
            (root/'Files/experiment_config.json').write_text(json.dumps(self.config),encoding='utf-8')
            bundle=root/'Datasets/PHM15-diagnosis/validation_pilot_v1';bundle.mkdir(parents=True)
            shutil.copy2(self.root/'Datasets/PHM15-diagnosis/validation_pilot_v1/private_validation_labels.jsonl',bundle)
            for request,record in zip(self.cs+self.ds,self.records+self.records):
                i=self.records.index(record);raw=json.dumps(self.fixture(i))
                envelope={'origin':'model_inference','raw_output':raw,'raw_output_sha256':pilot.base.fingerprint(raw),'error':None,
                          **{k:request[k] for k in ['request_id','case_id','condition','generator','evidence_sha256']}}
                dest=root/'Outputs/diagnosis'/request['condition']/'responses';dest.mkdir(parents=True,exist_ok=True)
                (dest/(request['request_id']+'.json')).write_text(json.dumps(envelope),encoding='utf-8')
            # Synthetic cache is disposable and never enters the project corpus.
            with patch.object(pilot,'prepare',return_value=(self.contract,self.records,self.cs,self.ds,self.retrieval)),patch.dict('sys.modules',{'torch':None,'transformers':None}):
                result=pilot.run(root,3)
            self.assertEqual(result['C']['valid_outputs'],3)
            self.assertEqual(result['D']['valid_outputs'],3)


if __name__=='__main__':unittest.main()
