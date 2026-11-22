import json
import unittest
from pathlib import Path
from unittest.mock import patch
import phm_selection_pilot as old
import phm_filtered_selection_pilot as new

class FilterChecks(unittest.TestCase):
    def test_filter_preserves_pairing_and_reuses_identical_inputs(self):
        root = Path(__file__).resolve().parent.parent
        config = json.loads((root/'Files/experiment_config.json').read_text(encoding='utf-8'))
        config['generator']['max_input_tokens'] = 6144
        original = Path.read_text
        def guard(path, *args, **kwargs):
            if 'private' in path.name:
                raise AssertionError('Private labels read during preparation')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guard):
            a = old.prepare(root, config, 10)
            b = new.prepare(root, config, 10)
        self.assertEqual(a[2], b[2])
        changed = 0
        for prev, cur, retrieval in zip(a[3], b[3], b[4]):
            same = prev['literature'] == cur['literature']
            self.assertEqual(prev['request_id'] == cur['request_id'], same)
            changed += not same
            self.assertEqual(prev['generator'], cur['generator'])
            self.assertEqual(prev['messages'][0], cur['messages'][0])
            x = json.loads(prev['messages'][1]['content'])
            y = json.loads(cur['messages'][1]['content'])
            x.pop('literature'); y.pop('literature')
            self.assertEqual(x, y)
            self.assertTrue(set(retrieval['chunks']) <= set(retrieval['eligible_chunks']))
            self.assertEqual(len({p['document_id'] for p in cur['literature']}), 2)
        self.assertEqual(changed, 5)
