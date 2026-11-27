"""Read-only train/validation foundation audit. No refit, test outcomes or LLM calls."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
from phm_run_reporting import atomic_json


def main(root):
    root=Path(root);out=root/'Outputs/reviews/foundation_audit_v1'
    with duckdb.connect(str(root/'Datasets/PHM15-prepared/case_system.duckdb'),read_only=True) as db:
        train=db.sql('select * from private_training_cases').df()
        validation=db.sql('select * from private_validation_cases').df()
        links=db.sql('select * from retrieval_links order by query_case_id,rank').df()
        metrics=db.sql('select * from validation_metrics order by macro_f1 desc').df().to_dict('records')
    with duckdb.connect(str(root/'Datasets/PHM15-prepared/case_inputs.duckdb'),read_only=True) as db:
        splits=dict(db.sql('select site_id,split from site_split').fetchall())
        episodes=db.sql("select e.* from private_episodes e join site_split s using(site_id) where s.split in ('train','validation')").df()
    assert all(splits[int(s)]=='train' for s in train.site_id)
    assert all(splits[int(s)]=='validation' for s in validation.site_id)
    assert not set(train.case_id)&set(validation.case_id)
    timeline={};agg={}
    for name,frame in [('train',train),('validation',validation)]:
        cases=[]
        for r in frame.itertuples():
            e=episodes.loc[episodes.site_id.eq(r.site_id)]
            overlaps=e[(e.start_at<r.decision_time)&(e.end_at>r.decision_time-pd.Timedelta(hours=3))]
            active=e[(e.start_at<=r.decision_time)&(e.end_at>r.decision_time)]
            if r.reference_label=='NoRecordedTarget':assert active.empty
            else:assert len(active)==1 and active.iloc[0].event_name==r.reference_label
            cases.append({'case_id':r.case_id,'label':r.reference_label,
                'fault_started_within_3h':bool(len(active) and (r.decision_time-active.iloc[0].start_at).total_seconds()<10800),
                'lookback_contains_recorded_target':not overlaps.empty,
                'multiple_target_episodes_in_lookback':len(overlaps)>1})
        timeline[name]={'cases':len(frame),'by_class':{label:{'n':sum(x['label']==label for x in cases),
            **{k:sum(x['label']==label and x[k] for x in cases) for k in ['fault_started_within_3h','lookback_contains_recorded_target','multiple_target_episodes_in_lookback']}}
            for label in sorted(frame.reference_label.unique())}}
        agg[name]={'components_median':float(frame.observed_components.median()),'components_max':int(frame.observed_components.max()),'channels':{}}
        for ch in ['ch1','ch2','ch5']:
            lo=frame[ch+'_lowest_component_change'];hi=frame[ch+'_highest_component_change'];med=frame[ch+'_median_of_component_changes']
            valid=lo.notna()&hi.notna()&med.notna();opposed=valid&(lo<0)&(hi>0)
            extremes=pd.concat([lo.abs(),hi.abs()],axis=1).max(axis=1)
            ratio=(med.abs()/extremes).where(valid&(extremes>0))
            agg[name]['channels'][ch]={'usable_windows':int(valid.sum()),'opposite_component_change_signs':int(opposed.sum()),'median_abs_median_change_over_max_abs_component_change':float(ratio.median())}
    t=train.set_index('case_id');v=validation.set_index('case_id');rows=[];gaps=[]
    for case_id,group in links.groupby('query_case_id',sort=False):
        assert case_id in v.index
        q=v.loc[case_id];hs=t.loc[group.historical_case_id]
        assert len(hs)==3 and hs.site_id.nunique()==3
        labels=list(hs.reference_label);counts=Counter(labels);winner=counts.most_common();tie=winner[0][1]==1
        rows.append({'label':q.reference_label,'nearest_correct':labels[0]==q.reference_label,
                     'true_label_present':q.reference_label in labels,'mixed_outcomes':len(counts)>1,
                     'majority_correct_ties_abstain':not tie and winner[0][0]==q.reference_label,'three_way_tie':tie,
                     'nearest_distance':float(group.iloc[0].distance)})
        gaps.extend(abs(hs.occupied_fraction-float(q.occupied_fraction)).tolist())
    assert len(rows)==len(validation)
    retrieval={'queries':len(rows),'by_class':{label:{'n':sum(x['label']==label for x in rows),
        **{k:sum(x['label']==label and x[k] for x in rows) for k in ['nearest_correct','true_label_present','mixed_outcomes','majority_correct_ties_abstain','three_way_tie']}}
        for label in sorted(validation.reference_label.unique())},
        'overall':{k:sum(x[k] for x in rows) for k in ['nearest_correct','true_label_present','mixed_outcomes','majority_correct_ties_abstain','three_way_tie']},
        'nearest_distance_median_when_label_matches':float(np.median([r['nearest_distance'] for r in rows if r['nearest_correct']])),
        'nearest_distance_median_when_label_differs':float(np.median([r['nearest_distance'] for r in rows if not r['nearest_correct']])),
        'occupancy_fraction_absolute_gap_p50_p90':np.quantile(gaps,[.5,.9]).tolist(),
        'neighbour_pairs_with_occupancy_fraction_gap_above_half':sum(x>.5 for x in gaps),'neighbour_pairs':len(gaps)}
    result={'scope':'Existing training and validation cohorts only. Descriptive foundation audit; no refit, new LLM run or held-out outcome access.',
        'window_alignment':timeline,'aggregation':agg,'retrieval':retrieval,'previously_saved_numerical_validation_metrics':metrics,
        'interpretation_limits':['Nearest-label agreement is a retrieval diagnostic, not a proposed replacement classifier.',
            'Class-balanced/capped validation cohort is not deployment prevalence.',
            'Opposing component changes and median compression demonstrate lost detail, not proof that lost detail predicts faults.',
            'Mixed pre/post-onset histories may be legitimate online diagnosis inputs; this is a task/representation mismatch risk, not automatically label leakage or a bug.']}
    atomic_json(out/'foundation_audit.json',result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main(Path(__file__).resolve().parent.parent)
