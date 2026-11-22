from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phm_diagnosis as base
import phm_heldout_export as export
import phm_final_metrics as metrics
import phm_two_stage as pipeline


class HeldoutChecks(unittest.TestCase):
    def test_public_loader_never_reads_test_labels_and_rejects_changes(self):
        root=Path(__file__).resolve().parent.parent
        _,dictionary,records=base.load_bundle(root/'Datasets/PHM15-diagnosis/validation_pilot_v1')
        with tempfile.TemporaryDirectory() as folder:
            test_root=Path(folder);bundle=test_root/export.FOLDER;bundle.mkdir(parents=True)
            (test_root/'Files').mkdir();(test_root/'Files/sentinel.py').write_text('unchanged')
            index=test_root/'Datasets/PHM15-literature-D-approved-v1/index_manifest.json'
            index.parent.mkdir(parents=True);index.write_text('{}')
            synthetic=[]
            for i in range(100):
                row=deepcopy(records[0]);row['case_id']=f'fixture_{i}';synthetic.append(row)
            (bundle/'test_evidence.jsonl').write_text(''.join(base.canonical(r)+'\n' for r in synthetic),encoding='utf-8')
            (bundle/'feature_dictionary.json').write_text(json.dumps(dictionary),encoding='utf-8')
            contract={'scope':'heldout_test','historical_k':3,'case_ids':[r['case_id'] for r in synthetic],
                'frozen_code_sha256':{'sentinel.py':export.sha(test_root/'Files/sentinel.py')},
                'literature_index_sha256':export.sha(index),
                'files':{n:export.sha(bundle/n) for n in ['test_evidence.jsonl','feature_dictionary.json']}}
            (bundle/'export_contract.json').write_text(json.dumps(contract),encoding='utf-8')
            read=Path.read_text
            def guard(path,*args,**kwargs):
                if 'private' in path.name:raise AssertionError('Private file accessed')
                return read(path,*args,**kwargs)
            with patch.object(Path,'read_text',guard):
                self.assertEqual(len(export.load_test_bundle(bundle)[2]),100)
            (test_root/'Files/sentinel.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Frozen evaluation code changed'):
                export.load_test_bundle(bundle)

    def test_heldout_rejects_partial_sample(self):
        with self.assertRaisesRegex(ValueError,'complete frozen 100-case'):
            pipeline.prepare(Path('.'),{},3,heldout=True)

    def test_pending_test_inference_cannot_open_labels(self):
        original_root=Path(__file__).resolve().parent.parent
        config=json.loads((original_root/'Files/experiment_config.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'Files').mkdir()
            (root/'Files/experiment_config.json').write_text(json.dumps(config),encoding='utf-8')
            q={'case_id':'fixture','condition':'C','task_stage':'decision','request_id':'pending'}
            prepared=({},[{'case_id':'fixture','_request':q}],[q],[])
            # No labels exist in this workspace. The deliberate pending guard must
            # fire rather than attempting scoring or a label-file read.
            with patch.object(pipeline,'prepare',return_value=prepared),patch.object(pipeline,'_infer',return_value={'new_responses':0}):
                with self.assertRaisesRegex(RuntimeError,'test labels have not been opened'):
                    pipeline.run(root,100,heldout=True)

    def test_failed_predictions_stay_in_metrics(self):
        rows=[{'reference_label':label,'predicted_class':label} for label in base.LABELS]
        rows.append({'reference_label':'NoRecordedTarget','predicted_class':None})
        result=metrics.classification(rows)
        self.assertEqual(result['accuracy'],.75)
        self.assertEqual(result['per_class']['NoRecordedTarget']['recall'],.5)
        self.assertAlmostEqual(result['macro_f1'],(2/3+1+1)/3)

    def test_paired_bootstrap_is_reproducible_and_keeps_clusters_paired(self):
        rows=[];sites={}
        for i in range(6):
            sites[str(i)]=i//2
            for condition in ('C','D'):
                label=base.LABELS[i%3]
                rows.append({'case_id':str(i),'condition':condition,'reference_label':label,'predicted_class':label,
                             'decision_status':'valid','explanation_status':'valid','full_output_valid':True})
        a=metrics.analyse(rows,sites,draws=20);b=metrics.analyse(rows,sites,draws=20)
        self.assertEqual(a,b)
        self.assertEqual(a['site_cluster_bootstrap']['percentile_95_intervals']['delta_macro_f1'],[0,0])
        self.assertEqual(a['paired_prediction_disagreements'],0)
        rows.pop()
        with self.assertRaises(ValueError):metrics.analyse(rows,sites,draws=20)


if __name__=='__main__':unittest.main()
