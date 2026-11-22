"""Freeze and export a site-balanced, label-blind held-out evaluation sample."""
import argparse
import hashlib
import json
from pathlib import Path

import phm_diagnosis as base
from phm_run_reporting import atomic_json, atomic_text

FOLDER='Datasets/PHM15-diagnosis/heldout_test_v1'
FILES=['phm_diagnosis.py','phm_diagnosis_qwen.py','phm_output_schema.py','phm_condition_d.py',
       'phm_evidence_selection.py','phm_filtered_selection_pilot.py','phm_two_stage.py',
       'phm_literature_index.py','phm_run_reporting.py','phm_heldout_export.py',
       'experiment_config.json','literature_retrieval_policy_v1.json','phm_final_metrics.py']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def protocol():
    return {'version':'heldout-test-v1','seed':5403,'sites':10,'per_site':10,'queries':100,
        'selection':'Within each test site rank eligible decision times by SHA256(seed, site, timestamp); accept feature-eligible windows spaced at least three hours apart until ten are retained.',
        'selection_inputs':'Existing eligibility status, site, timestamp and measurement coverage only. No reference labels, predictions or retrieval results determine selection.',
        'estimand':'Equal weight per held-out site, sampled from the existing eligible decision-time cohort. Not class-balanced, population-prevalence estimation, or independent fault-event sampling.',
        'coverage':'Frozen query_from_measurements rule: usable and recent component fractions at least 0.75, with three-hour history.',
        'pipeline':'Frozen two-stage Qwen C/D; 384 output tokens per stage, 6144 input tokens; same model, retrieval and prompts as development.',
        'scoring':'Private labels exported after sample selection; opened by inference runner only after both stages and conditions. No tuning on test outcomes.',
        'primary':'Macro-F1 over three declared classes; accuracy, per-class precision/recall/F1 and confusion matrix; failures and abstentions remain errors.',
        'reliability':'Separate classification validity, explanation validity, full-output validity and correct-plus-full-valid rates.',
        'uncertainty':'Site-cluster paired bootstrap intervals (2000 draws, seed 5403); only ten site clusters, so intervals have limited precision.',
        'grounding':'LLM judge scores are exploratory, not validated primary grounding metrics. No judge run is required for classification evaluation.',
        'failure_policy':'Cache every attempt, including failures; no automatic retries. Infrastructure interruptions may resume pending requests only.',
        'missing_classes':'Report zero class support explicitly; do not resample after seeing labels. Three-class macro-F1 uses zero for undefined class F1 and reports supports.'}


def load_test_bundle(folder):
    folder=Path(folder);root=folder.parents[2]
    contract=json.loads((folder/'export_contract.json').read_text(encoding='utf-8'))
    if contract.get('scope')!='heldout_test' or contract.get('historical_k')!=3:
        raise ValueError('Not a frozen held-out bundle')
    for name,expected in contract['frozen_code_sha256'].items():
        if sha(root/'Files'/name)!=expected:raise ValueError('Frozen evaluation code changed: '+name)
    index=root/'Datasets/PHM15-literature-D-approved-v1/index_manifest.json'
    if sha(index)!=contract['literature_index_sha256']:raise ValueError('Frozen literature index changed')
    for name in ['test_evidence.jsonl','feature_dictionary.json']:
        if sha(folder/name)!=contract['files'][name]:raise ValueError('Test evidence integrity failed: '+name)
    dictionary=json.loads((folder/'feature_dictionary.json').read_text(encoding='utf-8'))
    records=base.read_rows(folder/'test_evidence.jsonl')
    if [r['case_id'] for r in records]!=contract['case_ids'] or len(records)!=100 or len({r['case_id'] for r in records})!=100:
        raise ValueError('Frozen sample identity mismatch')
    for record in records:base.validate_evidence(record,dictionary['feature_order'],3)
    return contract,dictionary,records


def export(root):
    import duckdb
    import pandas as pd
    from phm_case_system import load_system, query_from_measurements
    root=Path(root).resolve();output=root/FOLDER
    if (output/'export_contract.json').exists():
        contract,_,records=load_test_bundle(output)
        for name,expected in contract['files'].items():
            if sha(output/name)!=expected:raise ValueError('Existing bundle altered: '+name)
        print('Existing frozen bundle verified:',len(records));return output
    output.mkdir(parents=True,exist_ok=True)
    plan=protocol();plan_path=output/'evaluation_protocol.json'
    if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
        raise ValueError('Existing protocol differs; do not overwrite')
    atomic_json(plan_path,plan)
    frozen={name:sha(root/'Files'/name) for name in FILES}
    source=root/'Datasets/PHM15-prepared/case_inputs.duckdb'
    system=root/'Datasets/PHM15-prepared/case_system.duckdb'
    with duckdb.connect(str(system),read_only=True) as db:
        fitted_contract=json.loads(db.sql('select contract_json from case_system_contract').fetchall()[0][0])
    for name,expected in fitted_contract['code_sha256'].items():
        if sha(root/'Files'/name)!=expected:raise ValueError('Numerical source changed: '+name)
    if sha(source)!=fitted_contract['input_sha256']['case_inputs.duckdb']:
        raise ValueError('Numerical input database changed')
    bank=load_system(system)
    dictionary=json.loads((root/'Datasets/PHM15-diagnosis/validation_pilot_v1/feature_dictionary.json').read_text(encoding='utf-8'))
    train_lookup=bank.training.set_index('case_id')
    evidence=[];audit=[];rejections=[];selected=[]
    with duckdb.connect(str(source),read_only=True) as db:
        splits=dict(db.sql('select site_id,split from site_split').fetchall())
        test_sites=sorted(s for s,v in splits.items() if v=='test')
        if len(test_sites)!=10 or any(splits[int(s)]!='train' for s in bank.training.site_id):
            raise ValueError('Unexpected train/test site allocation')
        validation_audit=json.loads((root/'Datasets/PHM15-diagnosis/validation_pilot_v1/private_export_audit.json').read_text(encoding='utf-8'))
        if set(test_sites)&{int(r['site_id']) for r in validation_audit}:raise ValueError('Validation/test site overlap')
        for site in test_sites:
            times=[r[0] for r in db.execute("select decision_time from private_decisions where site_id=? and status='eligible' order by decision_time",[site]).fetchall()]
            times.sort(key=lambda t:base.fingerprint([plan['seed'],site,t.isoformat()]))
            accepted=[];coverage_skips=spacing_skips=0
            for time in times:
                if any(abs(pd.Timestamp(time)-pd.Timestamp(t))<pd.Timedelta(hours=3) for t in accepted):
                    spacing_skips+=1;continue
                query,reason=query_from_measurements(db,site,time)
                if query is None:coverage_skips+=1;continue
                record=bank.evidence(query,site_id=site,k=3)
                base.validate_evidence(record,dictionary['feature_order'])
                ids=[h['case_id'] for h in record['historical_cases']]
                sites=[int(train_lookup.loc[i,'site_id']) for i in ids]
                if len(set(sites))!=3 or any(splits[s]!='train' for s in sites):raise ValueError('Training-only retrieval failed')
                evidence.append(record);accepted.append(time)
                selected.append({'case_id':record['case_id'],'site_id':site,'decision_time':time})
                audit.append({'case_id':record['case_id'],'site_id':site,'decision_time':time.isoformat(),
                              'historical_case_ids':ids,'historical_site_ids':sites})
                if len(accepted)==plan['per_site']:break
            if len(accepted)!=plan['per_site']:raise ValueError('Insufficient eligible cases for a frozen site quota')
            rejections.append({'site_id':site,'eligible_times':len(times),'coverage_rejections':coverage_skips,'spacing_rejections':spacing_skips})
            print('Selected held-out site',site,':',len(accepted),'cases',flush=True)
        ids={r['case_id'] for r in evidence}
        validation_ids={r['case_id'] for r in validation_audit}
        if ids & (set(bank.training.case_id)|validation_ids):raise ValueError('Test/development case overlap')
        # Lock case identities and evidence before accessing any outcome values.
        atomic_text(output/'test_evidence.jsonl',''.join(base.canonical(r)+'\n' for r in evidence))
        atomic_json(output/'private_export_audit.json',audit)
        atomic_json(output/'feature_dictionary.json',dictionary)
        db.register('frozen_selection',pd.DataFrame(selected))
        labels=db.sql('select s.case_id,d.reference_label from frozen_selection s join private_decisions d using(site_id,decision_time)').fetchall()
        if len(labels)!=100 or {r[0] for r in labels}!=ids or any(r[1] not in base.LABELS for r in labels):
            raise ValueError('Frozen sample scoring labels incomplete')
        label_lookup=dict(labels)
        atomic_text(output/'private_test_labels.jsonl',''.join(base.canonical({'case_id':r['case_id'],'reference_label':label_lookup[r['case_id']]})+'\n' for r in evidence))
    atomic_json(output/'selection_audit.json',rejections)
    names=['test_evidence.jsonl','feature_dictionary.json','private_test_labels.jsonl','private_export_audit.json','evaluation_protocol.json','selection_audit.json']
    contract={'scope':'heldout_test','schema_version':'1.0','historical_k':3,'case_ids':[r['case_id'] for r in evidence],
        'native_features':len(bank.features),'frozen_code_sha256':frozen,'source_database_sha256':fitted_contract['input_sha256']['case_inputs.duckdb'],
        'fitted_system_sha256':sha(system),'literature_index_sha256':sha(root/'Datasets/PHM15-literature-D-approved-v1/index_manifest.json'),
        'files':{name:sha(output/name) for name in names},'training_cases':len(bank.training),'heldout_sites':10,
        'test_labels_used_for_selection':False,'test_labels_exported_privately_after_selection':True,'test_inference_executed':False}
    atomic_json(output/'export_contract.json',contract)
    load_test_bundle(output)
    print('Frozen held-out evidence exported:',len(evidence),'cases. No class distribution or test scores inspected.')
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=Path('.'))
    args=parser.parse_args();export(args.root)
