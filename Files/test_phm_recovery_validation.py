import json
import unittest
from pathlib import Path
import tempfile
import shutil
from copy import deepcopy
from unittest.mock import patch
import phm_recovery_validation as recovery
import phm_two_stage as old


class RecoveryChecks(unittest.TestCase):
    def test_revision_preserves_evidence_and_generator_and_changes_identity(self):
        q={'condition':'C','messages':[{'role':'system','content':old.DECISION_PROMPT},{'role':'user','content':'unchanged'}],
           'output_schema':{'properties':{},'required':[]},'generator':{'max_new_tokens':384},'request_id':'old'}
        before=deepcopy(q);r=recovery.revised(q)
        self.assertEqual(q,before)
        self.assertEqual(q['messages'][1],r['messages'][1])
        self.assertEqual(q['generator'],r['generator'])
        self.assertNotEqual(q['request_id'],r['request_id'])

    def test_override_is_not_forced_to_score_leader(self):
        parsed={k:[] for k in old.DECISION_KEYS}
        parsed.update(predicted_class='RuntimeExcessive',selected_case_id='case',decision_justification='Observed temporal differences contradict the leading class.')
        record={'_request':{'assembly_version':recovery.VERSION,'task_stage':'decision'}}
        with patch.object(old,'validate',return_value={'valid':True,'parsed':{'predicted_class':'RuntimeExcessive'}}):
            result=recovery.validate(json.dumps(parsed),record)
        self.assertTrue(result['valid'])
        self.assertEqual(result['parsed']['predicted_class'],'RuntimeExcessive')
        parsed['decision_justification']=' '
        self.assertFalse(recovery.validate(json.dumps(parsed),record)['valid'])

    def test_failed_classification_stays_in_denominator(self):
        rows=[]
        for variant in ('original','revised'):
            for c in ('C','D'):
                for cohort,label in [('natural','NoRecordedTarget'),('fault_challenge','RuntimeExcessive')]:
                    rows.append(dict(variant=variant,condition=c,cohort=cohort,status='invalid_response',reference_label=label,predicted_class=None,score_top=label))
        m=recovery.metrics(rows)
        self.assertEqual(m['original']['C']['all_selected']['queries'],2)
        self.assertEqual(m['revised']['D']['all_selected']['accuracy'],0)
        self.assertFalse(recovery.improvement_gate(m)['passed'])

    def test_gate_requires_both_conditions_and_no_fault_recall_loss(self):
        item={'all_selected':{'queries':52,'macro_f1':.3,'per_class':{x:{'recall':.5} for x in recovery.base.LABELS}},'natural':{'accuracy':.5},'valid':52,'harmful_overrides':10}
        m={v:{c:deepcopy(item) for c in ('C','D')} for v in ('original','revised')}
        for c in ('C','D'):
            m['revised'][c]['all_selected']['macro_f1']=.34
            m['revised'][c]['harmful_overrides']=5
        self.assertTrue(recovery.improvement_gate(m)['passed'])
        m['revised']['D']['all_selected']['per_class']['RuntimeExcessive']['recall']=.4
        self.assertFalse(recovery.improvement_gate(m)['passed'])

    def test_runner_stops_or_continues_and_scores_runtime_failures(self):
        root=Path(__file__).resolve().parent.parent
        folder=root/recovery.BUNDLE
        if not folder.exists():self.skipTest('Local recovery export not present')
        contract=json.loads((folder/'export_contract.json').read_text())
        records=recovery.base.read_rows(folder/'validation_evidence.jsonl')
        requests=recovery.base.read_rows(folder/'requests.jsonl')
        for passes in (False,True):
            with self.subTest(gate_passes=passes), tempfile.TemporaryDirectory() as tmp:
                target=Path(tmp);(target/'Files').mkdir()
                shutil.copy2(root/'Files/experiment_config.json',target/'Files/experiment_config.json')
                (target/recovery.BUNDLE).mkdir(parents=True)
                for name in ('private_validation_labels.jsonl','private_export_audit.json'):
                    shutil.copy2(folder/name,target/recovery.BUNDLE/name)
                def fake_capture(reporter,bundle,cache,reqs,recs,score=False):
                    out=[]
                    for i,(q,r) in enumerate(zip(reqs,recs)):
                        parsed={'predicted_class':'NoRecordedTarget','selected_features':[next(iter(r['feature_values']))],
                                'selected_case_id':r['historical_cases'][0]['case_id'],'supporting_case_ids':[],
                                'conflicting_case_ids':[],'literature_citations':[]}
                        # A cached model error lacks 'valid'; runner must still score it as failure.
                        check={'status':'runtime_error'} if i==0 else {'status':'valid','valid':True,'parsed':parsed}
                        out.append({'case_id':q['case_id'],'condition':q['condition'],'checks':check,'response':{}})
                    return out
                with patch.object(recovery,'load_bundle',return_value=(contract,records,requests)), \
                     patch.object(recovery,'_infer',return_value={}) as infer, \
                     patch.object(recovery,'capture',side_effect=fake_capture), \
                     patch.object(recovery,'improvement_gate',return_value={'passed':passes}):
                    recovery.run(target)
                self.assertEqual(infer.call_count,2 if passes else 1)
                latest=json.loads((target/'Outputs/LATEST_RUN.json').read_text())
                report=target/latest['run_directory']
                result=json.loads((report/'classification_results.json').read_text())
                self.assertEqual(len(result),208)
                self.assertIsNone(result[0]['predicted_class'])
                self.assertEqual((report/'explanation_completion.json').exists(),passes)


if __name__=='__main__':unittest.main()
