"""Checks against the prepared validation bundle and reviewed literature spans."""
from pathlib import Path
import json
import unittest
from copy import deepcopy
from unittest.mock import patch
import numpy as np
import phm_condition_d as d
from phm_literature_index import rows, sha


class PairedChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parent.parent
        cls.config = json.loads((cls.root / 'Files/experiment_config.json').read_text(encoding='utf-8'))
        cls.config['generator']['max_input_tokens'] = 6144
        original = Path.read_text
        def guarded(path, *args, **kwargs):
            if path.name in {'private_labels.jsonl', 'private_validation_labels.jsonl'}:
                raise AssertionError('Preparation must not read scoring labels')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded):
            cls.contract, cls.records, cls.cs, cls.ds, cls.retrieval = d.prepare(cls.root, cls.config, 3)

    def test_paired_evidence_and_generator_preserved(self):
        for c, other in zip(self.cs, self.ds):
            self.assertEqual(c['case_id'], other['case_id'])
            self.assertEqual(c['generator'], other['generator'])
            self.assertEqual(c['evidence_sha256'], other['evidence_sha256'])
            self.assertTrue(other['messages'][1]['content'].startswith(c['messages'][1]['content']))
            self.assertNotEqual(c['request_id'], other['request_id'])
            self.assertEqual(len(other['literature']), 2)
            self.assertEqual(len({p['document_id'] for p in other['literature']}), 2)

    def test_citation_membership_and_json_failures(self):
        record = self.records[0]
        fixture = {'predicted_class': 'InsufficientEvidence', 'feature_evidence': [],
                   'case_comparison': None, 'comparison_claims': [],
                   'literature_citations': [self.ds[0]['literature'][0]['chunk_id']],
                   'explanation': 'The supplied evidence is insufficient.', 'limitation': 'Physical channel meanings are unavailable.'}
        self.assertTrue(d.validate_d(json.dumps(fixture), record, self.ds[0]['literature'])['valid'])
        fixture['literature_citations'] = ['invented_source']
        self.assertFalse(d.validate_d(json.dumps(fixture), record, self.ds[0]['literature'])['valid'])
        self.assertFalse(d.validate_d('not json', record, self.ds[0]['literature'])['valid'])

    def test_ranked_scores_are_derived_from_inputs_in_both_conditions(self):
        for record, c, other in zip(self.records, self.cs, self.ds):
            ranking = sorted(record['class_scores'], key=record['class_scores'].get, reverse=True)
            marker = d.base.canonical({'score_order_descending': ranking})
            self.assertIn(marker, c['messages'][1]['content'])
            self.assertIn(marker, other['messages'][1]['content'])
            self.assertIn(d.REASONING_RULES, c['messages'][0]['content'])
            self.assertIn(d.REASONING_RULES, other['messages'][0]['content'])

    def test_numeric_prose_rejected_but_channel_feature_names_allowed(self):
        fixture = {'predicted_class': 'InsufficientEvidence', 'feature_evidence': [],
                   'case_comparison': None, 'comparison_claims': [], 'literature_citations': [],
                   'explanation': 'The score of 0.064 is higher than 0.874.',
                   'limitation': 'Physical channel meanings are unavailable.'}
        check = d.validate_paired(json.dumps(fixture), self.records[0])
        self.assertIn('numerical_prose_not_allowed:explanation', check['errors'])
        fixture['explanation'] = 'The meaning of ch2 is unknown; ' + next(iter(self.records[0]['feature_values'])) + ' is an observed feature.'
        self.assertTrue(d.validate_paired(json.dumps(fixture), self.records[0])['valid'])

    def test_generation_schema_blocks_missing_fields_and_unsupplied_ids(self):
        from phm_output_schema import json_parser
        def accepts(schema, obj):
            parser = json_parser(schema)
            for character in json.dumps(obj):
                if character not in parser.get_allowed_characters():
                    return False
                parser = parser.add_character(character)
            return parser.can_end()
        fixture = {'predicted_class': 'InsufficientEvidence', 'feature_evidence': [],
                   'case_comparison': None, 'comparison_claims': [], 'literature_citations': [],
                   'explanation': 'Insufficient evidence.', 'limitation': 'Unknown physical channel meanings.'}
        self.assertTrue(accepts(self.cs[0]['output_schema'], fixture))
        bad = deepcopy(fixture)
        del bad['literature_citations']
        self.assertFalse(accepts(self.cs[0]['output_schema'], bad))
        fixture['literature_citations'] = [self.ds[0]['literature'][0]['chunk_id']]
        self.assertTrue(accepts(self.ds[0]['output_schema'], fixture))
        self.assertFalse(accepts(self.cs[0]['output_schema'], fixture))
        fixture['literature_citations'] = ['invented_source']
        self.assertFalse(accepts(self.ds[0]['output_schema'], fixture))
        fixture['case_comparison'] = None
        fixture['literature_citations'] = []
        source_claim = d.base.comparison_claims(self.records[0], self.records[0]['historical_cases'][0])[0]
        fixture['comparison_claims'] = [source_claim]
        fixture['explanation'] = 'Evidence is limited. ' * 25
        self.assertGreater(len(fixture['explanation']), 330)
        self.assertTrue(accepts(self.cs[0]['output_schema'], fixture))
        fixture['comparison_claims'][0]['query_value'] = 123456789.0
        self.assertFalse(accepts(self.cs[0]['output_schema'], fixture))
        fixture['comparison_claims'] = []
        fixture['literature_citations'] = []
        fixture['case_comparison'] = {'case_id': 'invented_case', 'similarity_feature': next(iter(self.records[0]['feature_values'])),
                                      'difference_feature': next(iter(self.records[0]['feature_values']))}
        self.assertFalse(accepts(self.ds[0]['output_schema'], fixture))

    def test_reviewed_chunks_exactly_match_source_spans(self):
        index = self.root / 'Datasets/PHM15-literature-D-approved-v1'
        prepared = self.root / 'Datasets/PHM15-literature-D-v1'
        paragraphs = {}
        for doc in rows(prepared / 'inventory.jsonl'):
            paragraphs.update({p['paragraph_id']: p for p in rows(prepared / doc['document_id'] / 'paragraphs.jsonl')})
        chunks = rows(index / 'approved_chunks.jsonl')
        for chunk in chunks:
            passage = ' '.join(paragraphs[s['paragraph_id']]['text'][s['start']:s['end']]
                               for s in chunk['passage_source_spans'])
            self.assertEqual(chunk['text'], passage[chunk['passage_start']:chunk['passage_end']])
            self.assertLessEqual(chunk['token_count'], 256)
        vectors = np.load(index / 'embeddings.npy', allow_pickle=False)
        self.assertEqual(vectors.shape, (len(chunks), 384))
        self.assertTrue(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5))

    def test_d_cache_retains_provenance_checks(self):
        import tempfile
        request = self.ds[0]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.json'
            path.write_text(json.dumps({'origin': 'synthetic'}), encoding='utf-8')
            self.assertEqual(d.inspect_d(path, request, self.records[0])[0]['status'], 'invalid_cache')

    def test_cached_paired_run_scores_real_label_filename_without_gpu(self):
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'Files').mkdir()
            (root / 'Files/experiment_config.json').write_text(json.dumps(self.config), encoding='utf-8')
            bundle = root / 'Datasets/PHM15-diagnosis/validation_pilot_v1'
            bundle.mkdir(parents=True)
            shutil.copy2(self.root / 'Datasets/PHM15-diagnosis/validation_pilot_v1/private_validation_labels.jsonl', bundle)
            for request in self.cs + self.ds:
                dest = root / 'Outputs/diagnosis' / request['condition'] / 'responses'
                dest.mkdir(parents=True, exist_ok=True)
                fixture = {'predicted_class': 'InsufficientEvidence', 'feature_evidence': [],
                           'case_comparison': None, 'comparison_claims': [], 'literature_citations': [],
                           'explanation': 'Insufficient evidence.', 'limitation': 'Physical channel meanings are unavailable.'}
                raw = json.dumps(fixture)
                envelope = {'origin': 'model_inference', 'raw_output': raw,
                            'raw_output_sha256': d.base.fingerprint(raw), 'error': None,
                            **{k: request[k] for k in ['request_id', 'case_id', 'condition', 'generator', 'evidence_sha256']}}
                (dest / (request['request_id'] + '.json')).write_text(json.dumps(envelope), encoding='utf-8')
            # Synthetic responses remain inside this disposable test workspace.
            with patch.object(d, 'prepare', return_value=(self.contract, self.records, self.cs, self.ds, self.retrieval)), \
                 patch.dict('sys.modules', {'torch': None, 'transformers': None}):
                result = d.run(root, maximum=3)
            self.assertEqual(result['C']['queries'], 3)
            self.assertEqual(result['D']['valid_outputs'], 3)
            latest = json.loads((root / 'Outputs/LATEST_RUN.json').read_text(encoding='utf-8'))
            self.assertTrue((root / latest['run_directory'] / 'paired_metrics.json').exists())


if __name__ == '__main__':
    unittest.main()
