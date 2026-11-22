"""Local export of a fixed validation cohort; no held-out rows or model inference."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import pandas as pd
import duckdb

import phm_diagnosis as base
import phm_filtered_selection_pilot as filtered
import phm_two_stage as old
from phm_case_system import load_system, query_from_measurements
from phm_run_reporting import atomic_json, atomic_text, load_experiment_config
from phm_recovery_validation import BUNDLE, VERSION, revised, sha, load_bundle


def export(root):
    root=Path(root).resolve();folder=root/BUNDLE
    if (folder/'export_contract.json').exists():
        load_bundle(root);print('Existing recovery bundle verified.');return
    source=root/'Datasets/PHM15-prepared/case_inputs.duckdb'
    system=root/'Datasets/PHM15-prepared/case_system.duckdb'
    with duckdb.connect(str(system),read_only=True) as db:
        fitted=json.loads(db.sql('select contract_json from case_system_contract').fetchall()[0][0])
    for name,expected in fitted['code_sha256'].items():
        if sha(root/'Files'/name)!=expected:raise ValueError('Numerical code changed: '+name)
    if sha(source)!=fitted['input_sha256']['case_inputs.duckdb']:raise ValueError('Source database changed')
    bank=load_system(system)
    dictionary=json.loads((root/'Datasets/PHM15-diagnosis/validation_pilot_v1/feature_dictionary.json').read_text())
    evidence=[];audit=[];labels=[];accepted={};train_lookup=bank.training.set_index('case_id')
    with duckdb.connect(str(source),read_only=True) as db:
        splits=dict(db.sql('select site_id,split from site_split').fetchall())
        sites=sorted(s for s,v in splits.items() if v=='validation')
        if len(sites)!=10 or any(splits[int(s)]!='train' for s in bank.training.site_id):raise ValueError('Split integrity failed')
        def add(site,time,cohort):
            if any(abs(pd.Timestamp(time)-pd.Timestamp(t))<pd.Timedelta(hours=3) for t in accepted.get(site,[])):return False
            q,_=query_from_measurements(db,site,time)
            if q is None:return False
            record=bank.evidence(q,site_id=site,k=3)
            base.validate_evidence(record,dictionary['feature_order'],3)
            historical_sites=[int(train_lookup.loc[h['case_id'],'site_id']) for h in record['historical_cases']]
            if len(set(historical_sites))!=3 or any(splits[s]!='train' for s in historical_sites):raise ValueError('Retrieval split integrity failed')
            evidence.append(record);accepted.setdefault(site,[]).append(time)
            audit.append({'case_id':record['case_id'],'site_id':site,'decision_time':time.isoformat(),
                          'cohort':cohort,'split':'validation','historical_site_ids':historical_sites})
            return True
        # Primary cohort: fixed per-site quotas with no outcome-dependent selection.
        for site in sites:
            times=[r[0] for r in db.execute("select decision_time from private_decisions where site_id=? and status='eligible'",[site]).fetchall()]
            times.sort(key=lambda t:base.fingerprint([5403,VERSION,'natural',site,t.isoformat()]))
            count=0
            for t in times:
                if add(site,t,'natural'):count+=1
                if count==4:break
            if count!=4:raise ValueError('Insufficient natural validation windows')
            print('Selected validation site quota',len(evidence),flush=True)
        # Supplemental fault challenge: validation labels allowed for sampling, not prompts.
        # Prefer distinct episodes and sites; never substitute challenge prevalence for natural prevalence.
        for label in base.LABELS[1:]:
            candidates=db.execute("""select d.site_id,d.decision_time,d.episode_id from private_decisions d
                join site_split s using(site_id) where s.split='validation' and d.status='eligible'
                and d.reference_label=?""",[label]).fetchall()
            candidates.sort(key=lambda x:base.fingerprint([5403,VERSION,'challenge',label,x[0],x[1].isoformat()]))
            count=0;used_episodes=set();used_sites=set()
            for prefer_sites in (True,False):
                for site,t,episode in candidates:
                    if episode in used_episodes or (prefer_sites and site in used_sites):continue
                    if add(site,t,'fault_challenge'):
                        used_episodes.add(episode);used_sites.add(site);count+=1
                    if count==6:break
                if count==6:break
            if count!=6:raise ValueError('Insufficient distinct fault episodes for challenge')
        # Outcomes are collected only after sample selection; never passed into bank.evidence.
        for a in audit:
            label,episode=db.execute('select reference_label,episode_id from private_decisions where site_id=? and decision_time=?',[a['site_id'],a['decision_time']]).fetchone()
            labels.append({'case_id':a['case_id'],'reference_label':label})
            a['episode_id']=episode
    if len(evidence)!=52 or len({r['case_id'] for r in evidence})!=52:raise ValueError('Wrong validation sample')
    # Construct C/D once locally. Colab needs no numerical pickle or embedding installation.
    temporary_contract={'scope':'post-test development validation','case_ids':[r['case_id'] for r in evidence]}
    config=deepcopy(load_experiment_config(root/'Files/experiment_config.json'))
    config['generator'].update(max_input_tokens=6144,max_new_tokens=384)
    def loader(_):return temporary_contract,dictionary,evidence
    _,_,cs,ds,retrieval=filtered.prepare(root,config,52,folder,loader)
    requests=[]
    # Interleave matched original/revised requests to reduce order confounding.
    for i in range(52):
        pair=[old.decision_request(cs[i]),old.decision_request(ds[i])]
        if i%2:pair.reverse()
        for original in pair:
            versions=[original,revised(original)]
            if i%2:versions.reverse()
            requests.extend(versions)
    files={
        'validation_evidence.jsonl':''.join(base.canonical(r)+'\n' for r in evidence),
        'private_validation_labels.jsonl':''.join(base.canonical(r)+'\n' for r in labels),
        'private_export_audit.json':json.dumps(audit,indent=2),
        'feature_dictionary.json':json.dumps(dictionary,indent=2),
        'requests.jsonl':''.join(base.canonical(r)+'\n' for r in requests),
        'literature_retrieval.json':json.dumps(retrieval,indent=2)}
    for name,content in files.items():atomic_text(folder/name,content)
    names=['phm_recovery_validation.py','phm_recovery_export.py','test_phm_recovery_validation.py',
           'phm_diagnosis.py','phm_two_stage.py','phm_diagnosis_qwen.py','phm_run_reporting.py',
           'phm_evidence_selection.py','phm_output_schema.py','phm_final_metrics.py','phm_heldout_export.py',
           'phm_filtered_selection_pilot.py','phm_condition_d.py','phm_literature_index.py',
           'phm_literature.py','experiment_config.json','requirements-qwen-colab.txt',
           'requirements-literature-colab.txt','requirements-literature.txt']
    contract={'scope':'post-test development validation','version':VERSION,
        'case_ids':[r['case_id'] for r in evidence], 'historical_k':3,
        'selection':'40 label-blind windows: four per validation site; plus six distinct episodes per fault class. Seed 5403, SHA256 ranking, minimum three-hour spacing per site.',
        'counts':{cohort:dict(Counter(labels[i]['reference_label'] for i,a in enumerate(audit) if a['cohort']==cohort)) for cohort in ('natural','fault_challenge')},
        'interpretation':'Post-hoc development after held-out audit, not independent confirmation. Natural cohort is site-balanced, not deployment-prevalence weighted. Supplemental challenge reported separately; combined metrics reflect designed mixture.',
        'stop_rule':'In BOTH C/D: combined macro-F1 gain >=0.02, natural accuracy not lower, harmful overrides reduced, >=95% valid decisions, neither fault recall lower. If false stop before explanations. This is a pragmatic screen, not significance.',
        'model_and_budget':'Same Qwen checkpoint, greedy NF4, 6144 input and 384 output tokens for both variants and stages.',
        'maximum_calls':{'decisions':208,'explanations_if_gate_passes':208},
        'final_test_accessed':False,'retries':False,
        'files':{name:sha(folder/name) for name in files},
        'code_sha256':{name:sha(root/'Files'/name) for name in names}}
    atomic_json(folder/'export_contract.json',contract)
    print(json.dumps(contract['counts'],indent=2));print('Exported',folder)


if __name__=='__main__':export(Path(__file__).resolve().parent.parent)
