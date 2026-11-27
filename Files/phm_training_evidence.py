"""Training-only descriptive evidence export. No generation, scoring or model fitting."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS=('NoRecordedTarget','RuntimeExcessive','SupplyTempTooCold')
VERSION='training-reference-v2'
OUTPUT='Datasets/PHM15-diagnosis/training_reference_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(values):
    a=np.asarray(values,dtype=float)
    return a[np.isfinite(a)]


def band(values):
    a=finite(values)
    if not len(a):return {'n':0,'p10':None,'median':None,'p90':None}
    lo,mid,hi=np.quantile(a,[.1,.5,.9])
    return {'n':len(a),'p10':float(lo),'median':float(mid),'p90':float(hi)}


def percentile_interval(values,value):
    a=finite(values)
    if not len(a) or value is None or not np.isfinite(value):return None
    # Ties represented as an interval, not falsely called extreme percentiles.
    return [round(float(np.mean(a<value)),4),round(float(np.mean(a<=value)),4)]


def assert_train_only(training,splits,query_ids):
    if not training.case_id.is_unique:raise ValueError('Duplicate training cases')
    if any(splits.get(int(s))!='train' for s in training.site_id):raise ValueError('Non-training site in reference bank')
    if set(training.case_id)&set(query_ids):raise ValueError('Query/reference case overlap')
    if not set(training.reference_label)<=set(LABELS):raise ValueError('Unexpected training label')


def build_catalog(training,encoded,features):
    if len(training)!=len(encoded) or not training.index.equals(encoded.index):raise ValueError('Training feature alignment mismatch')
    catalog={}
    for f in features:
        by_class={label:band(encoded.loc[training.reference_label.eq(label),f]) for label in LABELS}
        pooled=band(encoded[f]);span=(pooled['p90']-pooled['p10']) if pooled['n'] else 0
        medians=[b['median'] for b in by_class.values() if b['n']]
        separation=(max(medians)-min(medians))/span if span>0 and len(medians)==3 else 0.
        overlaps=[]
        for a,b in itertools.combinations(LABELS,2):
            x,y=by_class[a],by_class[b]
            overlaps.append({'classes':[a,b],'central_80_bands_overlap':None if not x['n'] or not y['n'] else max(x['p10'],y['p10'])<=min(x['p90'],y['p90'])})
        catalog[f]={'pooled':pooled,'by_class':by_class,'pairwise_overlap':overlaps,
                    'median_separation_over_pooled_central_span':float(separation)}
    # Fixed descriptive prioritisation, never tuned on query labels or outputs.
    order=sorted((f for f in features if f.startswith('ch')),key=lambda f:(-catalog[f]['median_separation_over_pooled_central_span'],f))
    selected=[];groups=Counter()
    for f in order:
        group=f.split('_')[0]
        if groups[group]>=2:continue
        selected.append(f);groups[group]+=1
        if len(selected)==8:break
    return catalog,selected


def retrieval_summary(record,train_lookup):
    hist=record['historical_cases'];seen=set();counts=Counter()
    for h in hist:
        if h['case_id'] not in train_lookup.index:raise ValueError('Retrieved case not in training bank')
        source=train_lookup.loc[h['case_id']]
        if source.reference_label!=h['recorded_outcome']:raise ValueError('Retrieved outcome mismatch')
        site=int(source.site_id)
        if site in seen:raise ValueError('Repeated training site in retrieval')
        seen.add(site);counts[h['recorded_outcome']]+=1
    maximum=max(counts.values());leaders=[k for k,v in counts.items() if v==maximum]
    return {'k':len(hist),'class_counts':[counts[c] for c in LABELS],
            'largest_count':maximum,'tie_for_largest':len(leaders)>1,
            'mixed_recorded_outcomes':len(counts)>1,
            'interpretation':'Selected-neighbour label agreement, not a probability or a decision rule.'}


def export(root):
    import duckdb
    from phm_case_system import load_system
    from phm_run_reporting import atomic_json,atomic_text
    import phm_evidence_menu_smoke as menu
    root=Path(root).resolve();out=root/OUTPUT
    source=root/'Datasets/PHM15-prepared/case_system.duckdb'
    inputs=root/'Datasets/PHM15-prepared/case_inputs.duckdb'
    public=root/menu.BUNDLE
    if (out/'manifest.json').exists():
        m=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
        if m['exporter_sha256']!=sha(__file__):raise ValueError('Existing exporter changed; use a new version')
        if m['case_system_sha256']!=sha(source):raise ValueError('Source changed')
        for name,digest in m['files'].items():
            if sha(out/name)!=digest:raise ValueError('Existing export changed: '+name)
        print('Existing training-reference export verified.');return
    contract=json.loads((public/'export_contract.json').read_text(encoding='utf-8'))
    for name in ('validation_evidence.jsonl','requests.jsonl'):
        if sha(public/name)!=contract['files'][name]:raise ValueError('Public evidence hash mismatch')
    with duckdb.connect(str(source),read_only=True) as db:
        fitted=json.loads(db.sql('select contract_json from case_system_contract').fetchall()[0][0])
    for name,digest in fitted['code_sha256'].items():
        if sha(root/'Files'/name)!=digest:raise ValueError('Fitted numerical source changed')
    if sha(inputs)!=fitted['input_sha256']['case_inputs.duckdb']:raise ValueError('Input database changed')
    # Only split membership is queried here. No decision outcomes or held-out records.
    with duckdb.connect(str(inputs),read_only=True) as db:
        splits=dict(db.sql('select site_id,split from site_split').fetchall())
    bank=load_system(source)
    records=menu.base.read_rows(public/'validation_evidence.jsonl')
    assert_train_only(bank.training,splits,[r['case_id'] for r in records])
    if set(bank.training.case_id)!=set(bank.model.training_case_ids_):raise ValueError('Bank/model case mismatch')
    catalog,selected=build_catalog(bank.training,bank.encoded_training,bank.features)
    train_lookup=bank.training.set_index('case_id')
    class_counts={c:int(bank.training.reference_label.eq(c).sum()) for c in LABELS}
    provenance={'version':VERSION,'training_cases':len(bank.training),'training_sites':int(bank.training.site_id.nunique()),
        'class_case_counts':class_counts,'model_id':bank.model_id,'feature_count':len(bank.features),
        'query_labels_used':False,'refitting':False,'scoring':False,'independent_observations_assumed':False,
        'source':'Frozen historical training bank; native encoded values before clipping/imputation.',
        'limitations':['Case-bank sampling is not plant fault prevalence. Repeated windows/sites are correlated.',
                      'Univariate distributions ignore interactions and operating-regime confounding.',
                      'Central 80% bands are descriptive quantiles, not normal ranges, fault thresholds or confidence intervals.',
                      'Training class labels are recorded outcomes, not verified physical health.',
                      'No evidence-sufficiency rule or improved diagnostic performance established.'],
        'selection':'Eight channel features ranked by between-class median range / pooled p90-p10; maximum two per channel; alphabetical ties; fixed before queries; descriptive prioritisation, not validated feature importance.'}
    shared={'class_order':list(LABELS),'class_case_counts':[class_counts[c] for c in LABELS],
            'feature_order':selected,'band_columns':['p10','median','p90','observed_n'],
            'bands':[[[None if catalog[f]['by_class'][c][k] is None else float(format(catalog[f]['by_class'][c][k],'.6g')) for k in ('p10','median','p90','n')] for c in LABELS] for f in selected],
            'display_precision':'6 significant digits; percentile intervals computed from unrounded training values.',
            'cautions':provenance['limitations']}
    evidence=[]
    for r in records:
        intervals=[[percentile_interval(bank.encoded_training.loc[bank.training.reference_label.eq(c),f],r['feature_values'].get(f)) for c in LABELS] for f in selected]
        evidence.append({'case_id':r['case_id'],'query_percentile_intervals':intervals,
                         'percentile_definition':'[fraction strictly below query, fraction at or below query]; ties are intervals, not uncertainty estimates.',
                         'retrieved_label_agreement':retrieval_summary(r,train_lookup)})
    by_id={r['case_id']:r for r in evidence}
    previews=[]
    for previous in menu.base.read_rows(public/'requests.jsonl'):
        if previous['assembly_version']!=menu.old.VERSION:continue
        q=menu.revised(previous);q.pop('request_id')
        q.update(prompt_version=VERSION+'-preview-'+q['condition'],assembly_version=VERSION)
        payload=json.loads(q['messages'][1]['content'])
        payload['training_reference']={**shared,**by_id[q['case_id']]}
        q['messages'][1]['content']=menu.base.canonical(payload)
        q['messages'][0]['content']+='\nThe training_reference is descriptive training-only context. Compare overlapping class bands and conflicting neighbours. Do not interpret quantiles as fault thresholds or combine marginal percentiles as class probabilities. Feature selection is not validated importance. Shared evidence does not force a class.'
        previews.append(menu.old.identify(q))
    atomic_json(out/'training_catalog.json',{'provenance':provenance,'features':catalog,'prompt_features':selected})
    atomic_json(out/'shared_prompt_reference.json',shared)
    atomic_text(out/'query_evidence.jsonl',''.join(menu.base.canonical(r)+'\n' for r in evidence))
    atomic_text(out/'candidate_requests_NOT_FOR_INFERENCE.jsonl',''.join(menu.base.canonical(q)+'\n' for q in previews))
    names=['training_catalog.json','shared_prompt_reference.json','query_evidence.jsonl','candidate_requests_NOT_FOR_INFERENCE.jsonl']
    atomic_json(out/'manifest.json',{'version':VERSION,'exporter_sha256':sha(__file__),'case_system_sha256':sha(source),
        'public_evidence_sha256':sha(public/'validation_evidence.jsonl'),'training_case_ids_sha256':hashlib.sha256('\n'.join(sorted(bank.training.case_id)).encode()).hexdigest(),
        'files':{n:sha(out/n) for n in names},'provenance':provenance,'queries':len(records),'candidate_requests':len(previews),
        'status':'local preparation only; no inference runner or evaluation enabled'})
    print(json.dumps(provenance,indent=2));print('Exported:',out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    export(p.parse_args().root)
