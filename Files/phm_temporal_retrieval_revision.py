"""One fixed training-only retrieval revision; no LLM or held-out evaluation."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from phm_case_system import load_system, read_features, NamedPreparation, feature_group
from phm_run_reporting import atomic_json
import hashlib

OUT='Datasets/PHM15-diagnosis/temporal_retrieval_revision_v1'
LABELS=['NoRecordedTarget','RuntimeExcessive','SupplyTempTooCold']
PLAN={'version':'temporal-component-retrieval-v1','scope':'training/validation development only',
      'representation':'Retain original native features; add three-hour component extrema, one-hour component change extremes and dispersion, and last-quarter-hour component levels/coverage.',
      'minimum_60m_bins':3,'minimum_15m_bins':1,'query_windows':'fixed trailing 15 and 60 minutes, never event-aligned',
      'fit':'Training-only median imputation, 0.5/99.5 percentile clipping, 0.95 Spearman redundancy screen, standard scaling, equal weight per feature group.',
      'retrieval':'Three nearest training cases from distinct sites, same rule as baseline.',
      'stop_rule':'Continue only if nearest-label diagnostic macro-F1 improves by >=0.02, hit@3 does not fall, and no class loses >0.05 nearest-label recall. Pragmatic development screen, not significance.',
      'not_performed':['numerical classifier refitting','LLM inference','test outcome access','LLM accuracy evaluation','automatic pipeline promotion']}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recent(window,time,component_count):
    if component_count<1:raise ValueError('No reference components')
    if len(window) and ((window.bin_end>time).any() or (window.bin_end<=time-pd.Timedelta(minutes=60)).any()):
        raise ValueError('Measurements outside trailing window')
    result={}
    for ch in ('ch1','ch2','ch5'):
        last=[];changes=[];medians=[];counts=[]
        for _,g in window.groupby('component'):
            g=g.sort_values('bin_end');g=g[g[ch].notna()]
            if g.bin_end.duplicated().any():raise ValueError('Duplicate component bins')
            late=g[g.bin_end>time-pd.Timedelta(minutes=15)]
            if len(late):last.append(float(late[ch].median()))
            if len(g)>=3:
                changes.append(float(g[ch].iloc[-1]-g[ch].iloc[0]));medians.append(float(g[ch].median()));counts.append(len(g))
        for name,values in [('last15_level',last),('last60_change',changes)]:
            for stat,fn in [('min',np.min),('median',np.median),('max',np.max)]:
                result[f'{ch}_{name}_{stat}']=float(fn(values)) if values else np.nan
        result[f'{ch}_last60_component_median_spread']=float(np.quantile(medians,.9)-np.quantile(medians,.1)) if medians else np.nan
        result[f'{ch}_last15_coverage_fraction']=len(last)/component_count
        result[f'{ch}_last60_coverage_fraction']=len(changes)/component_count
    return result


def extract(db,frame,split):
    outputs=[]
    for site,points in frame.groupby('site_id',sort=True):
        db.register('audit_points',points[['case_id','site_id','decision_time']])
        w=db.execute("""select p.case_id,b.component,b.bin_end,b.ch1,b.ch2,b.ch5 from component_bins b
            join audit_points p on b.site_id=p.site_id and b.bin_end>p.decision_time-interval '60 minutes'
            and b.bin_end<=p.decision_time order by p.case_id,b.component,b.bin_end""").df()
        groups={i:g for i,g in w.groupby('case_id')}
        for r in points.itertuples():
            window=groups.get(r.case_id,w.iloc[:0])
            outputs.append({'case_id':r.case_id,**recent(window,r.decision_time,int(r.observed_components))})
        db.unregister('audit_points')
        print(split,'site',int(site),'cases',len(points),flush=True)
    return pd.DataFrame(outputs).set_index('case_id').loc[frame.case_id].reset_index()


def score(frame,links,training):
    lookup=training.set_index('case_id');results=[]
    for r in frame.itertuples():
        ids=links[r.case_id];labels=[lookup.loc[i,'reference_label'] for i in ids]
        results.append({'case_id':r.case_id,'label':r.reference_label,'nearest_label':labels[0],
                        'hit_at_3':r.reference_label in labels,'mixed_outcomes':len(set(labels))>1})
    return {'n':len(results),'nearest_label_macro_f1':float(f1_score([r['label'] for r in results],[r['nearest_label'] for r in results],labels=LABELS,average='macro',zero_division=0)),
        'nearest_label_agreement':sum(r['label']==r['nearest_label'] for r in results)/len(results),
        'hit_at_3':sum(r['hit_at_3'] for r in results)/len(results),
        'mixed_outcome_rate':sum(r['mixed_outcomes'] for r in results)/len(results),
        'per_class_recall':{c:sum(r['label']==c and r['nearest_label']==c for r in results)/sum(r['label']==c for r in results) for c in LABELS}}


def main(root):
    root=Path(root);out=root/OUT;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        print((out/'result.json').read_text());return
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=PLAN:raise ValueError('Protocol changed')
    atomic_json(out/'protocol.json',PLAN)
    source=root/'Datasets/PHM15-prepared/case_system.duckdb';input_db=root/'Datasets/PHM15-prepared/case_inputs.duckdb'
    with duckdb.connect(str(source),read_only=True) as db:
        fitted=json.loads(db.sql('select contract_json from case_system_contract').fetchall()[0][0])
        validation=read_features(db,'private_validation_cases')
        baseline_links=db.sql('select * from retrieval_links order by query_case_id,rank').df()
    for name,digest in fitted['code_sha256'].items():
        if sha(root/'Files'/name)!=digest:raise ValueError('Numerical source changed')
    if sha(input_db)!=fitted['input_sha256']['case_inputs.duckdb']:raise ValueError('Input source changed')
    bank=load_system(source);training=bank.training.copy()
    with duckdb.connect(str(input_db),read_only=True) as db:
        splits=dict(db.sql('select site_id,split from site_split').fetchall())
        assert all(splits[int(s)]=='train' for s in training.site_id)
        assert all(splits[int(s)]=='validation' for s in validation.site_id)
        assert not set(training.case_id)&set(validation.case_id)
        added={}
        for name,frame in [('train',training),('validation',validation)]:
            path=out/(name+'_recent_features.jsonl')
            if path.exists():
                extra=pd.read_json(path,lines=True)
                assert list(extra.case_id)==list(frame.case_id)
            else:
                extra=extract(db,frame,name);extra.to_json(path,orient='records',lines=True,double_precision=15)
            added[name]=extra
    def assemble(frame,extra):
        encoded=bank.model.named_steps['encode'].transform(frame)
        x=encoded[bank.features].reset_index(drop=True).copy()
        for ch in ('ch1','ch2','ch5'):
            for kind in ('lowest_component_change','highest_component_change','lowest_component_iqr','highest_component_iqr'):
                f=ch+'_'+kind
                if f not in x:x[f]=frame[f].to_numpy()
        for f in extra.columns:
            if f!='case_id':x[f]=extra[f].to_numpy()
        return x
    x=assemble(training,added['train']);y=assemble(validation,added['validation'])
    assert list(x.columns)==list(y.columns)
    prepare=NamedPreparation(clip=True).fit(x)
    train_ready=prepare.transform(x);valid_ready=prepare.transform(y)
    scaler=StandardScaler().fit(train_ready)
    a=scaler.transform(train_ready);b=scaler.transform(valid_ready)
    names=list(train_ready.columns);groups=[feature_group(f) for f in names];counts=Counter(groups)
    weights=np.array([1/np.sqrt(counts[g]*len(counts)) for g in groups]);a*=weights;b*=weights
    links={};all_links=[]
    for i,r in enumerate(validation.itertuples()):
        dist=np.linalg.norm(a-b[i],axis=1);chosen=[];seen=set()
        for j in np.argsort(dist,kind='stable'):
            site=int(training.iloc[j].site_id)
            if site==int(r.site_id) or site in seen:continue
            chosen.append(str(training.iloc[j].case_id));seen.add(site)
            all_links.append({'query_case_id':r.case_id,'rank':len(chosen),'historical_case_id':chosen[-1],'distance':float(dist[j])})
            if len(chosen)==3:break
        assert len(chosen)==3;links[r.case_id]=chosen
    old={i:list(g.historical_case_id) for i,g in baseline_links.groupby('query_case_id',sort=False)}
    baseline=score(validation,old,training);candidate=score(validation,links,training)
    checks={'macro_f1_gain_at_least_002':candidate['nearest_label_macro_f1']-baseline['nearest_label_macro_f1']>=.02,
            'hit_at_3_not_lower':candidate['hit_at_3']>=baseline['hit_at_3'],
            'no_class_recall_loss_above_005':all(candidate['per_class_recall'][c]>=baseline['per_class_recall'][c]-.05 for c in LABELS)}
    pd.DataFrame(all_links).to_json(out/'candidate_retrieval_links.jsonl',orient='records',lines=True,double_precision=15)
    atomic_json(out/'feature_transform.json',{'input_features':list(x.columns),'retained_features':names,
        'fit_split':'train','training_rows':len(x),'validation_rows':len(y),
        'clipping_lower':prepare.lower_.tolist(),'clipping_upper':prepare.upper_.tolist(),
        'medians':prepare.imputer_.statistics_.tolist(),'scaler_mean':scaler.mean_.tolist(),
        'scaler_scale':scaler.scale_.tolist(),'weights':weights.tolist()})
    result={'scope':'Descriptive validation retrieval diagnostics, not LLM performance or independent test evidence.',
        'baseline':baseline,'candidate':candidate,'checks':checks,'passed':all(checks.values()),
        'features_before_screen':len(x.columns),'features_retained':len(names),'no_LLM_run':True,'heldout_outcomes_accessed':False,
        'next_action':'Review locally; no automatic promotion or GPU run.' if all(checks.values()) else 'STOP redevelopment of this candidate; do not run Qwen on it.',
        'provenance':{'exporter_sha256':sha(__file__),'case_system_sha256':sha(source),'case_inputs_sha256':sha(input_db)}}
    atomic_json(out/'result.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main(Path(__file__).resolve().parent.parent)
