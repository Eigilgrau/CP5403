"""Fixed held-out classification metrics and paired site-cluster intervals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
from statistics import mean

from phm_diagnosis import LABELS, ABSTAIN
from phm_heldout_export import FOLDER, sha
from phm_run_reporting import atomic_json


def classification(rows):
    per={}
    for label in LABELS:
        tp=sum(r['reference_label']==label and r['predicted_class']==label for r in rows)
        fp=sum(r['reference_label']!=label and r['predicted_class']==label for r in rows)
        fn=sum(r['reference_label']==label and r['predicted_class']!=label for r in rows)
        per[label]={'support':tp+fn,'precision':tp/(tp+fp) if tp+fp else 0,
                    'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    return {'queries':len(rows),'accuracy':sum(r['reference_label']==r['predicted_class'] for r in rows)/len(rows),
            'macro_f1':mean(p['f1'] for p in per.values()),'per_class':per}


def percentile(values,p):
    ordered=sorted(values);at=(len(ordered)-1)*p;lo=int(at);hi=min(lo+1,len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)


def analyse(rows,site_by_case,draws=2000,seed=5403):
    groups={c:[r for r in rows if r['condition']==c] for c in ('C','D')}
    ids={c:[r['case_id'] for r in group] for c,group in groups.items()}
    if len(ids['C'])!=len(set(ids['C'])) or set(ids['C'])!=set(ids['D']) or len(ids['D'])!=len(set(ids['D'])):
        raise ValueError('Paired query identities differ')
    lookups={c:{r['case_id']:r for r in group} for c,group in groups.items()}
    if any(lookups['C'][i]['reference_label']!=lookups['D'][i]['reference_label'] for i in ids['C']):
        raise ValueError('Paired reference labels differ')
    metrics={c:classification(group) for c,group in groups.items()}
    for c,group in groups.items():
        metrics[c].update(classification_valid=sum(r['decision_status']=='valid' for r in group),
            explanation_valid=sum(r['explanation_status']=='valid' for r in group),
            full_output_valid=sum(r['full_output_valid'] for r in group),
            correct_and_full_valid=sum(r['full_output_valid'] and r['predicted_class']==r['reference_label'] for r in group),
            abstentions=sum(r['predicted_class']==ABSTAIN for r in group),
            confusion_matrix={label:dict(Counter(r['predicted_class'] or 'InvalidOutput' for r in group if r['reference_label']==label)) for label in LABELS},
            mean_generation_seconds=mean(r.get('generation_seconds',0) for r in group),
            mean_input_tokens=mean(r.get('input_tokens',0) for r in group),
            mean_output_tokens=mean(r.get('output_tokens',0) for r in group))
    sites=sorted({site_by_case[i] for i in ids['C']})
    by_site={c:{s:[r for r in group if site_by_case[r['case_id']]==s] for s in sites} for c,group in groups.items()}
    rng=random.Random(seed);samples={k:[] for k in ('C_accuracy','D_accuracy','delta_accuracy','C_macro_f1','D_macro_f1','delta_macro_f1')}
    for _ in range(draws):
        selected=rng.choices(sites,k=len(sites))
        scored={c:classification([r for s in selected for r in by_site[c][s]]) for c in ('C','D')}
        for metric in ('accuracy','macro_f1'):
            for c in ('C','D'):samples[c+'_'+metric].append(scored[c][metric])
            samples['delta_'+metric].append(scored['D'][metric]-scored['C'][metric])
    return {'scope':'Frozen held-out C/D classification evaluation; semantic explanation support is not established by structural validity.',
        'conditions':metrics,'paired_prediction_disagreements':sum(lookups['C'][i]['predicted_class']!=lookups['D'][i]['predicted_class'] for i in ids['C']),
        'paired_D_minus_C':{k:metrics['D'][k]-metrics['C'][k] for k in ('accuracy','macro_f1')},
        'site_cluster_bootstrap':{'sites':len(sites),'draws':draws,'seed':seed,
            'percentile_95_intervals':{k:[percentile(v,.025),percentile(v,.975)] for k,v in samples.items()},
            'limitation':'Only ten held-out site clusters. Intervals do not capture model-selection uncertainty or establish population representativeness.'},
        'metric_convention':'Three-class macro-F1 with zero for undefined F1; per-class support is reported. Abstentions and failed classification outputs remain errors.',
        'cost_scope':'Original generation totals across both stages, including reused attempts; excludes loading/retrieval. Not notebook wall-clock time.'}


def run(root,run_folder):
    root=Path(root).resolve();folder=Path(run_folder).resolve();bundle=root/FOLDER
    contract=json.loads((bundle/'export_contract.json').read_text(encoding='utf-8'))
    if sha(root/'Files/phm_final_metrics.py')!=contract['frozen_code_sha256']['phm_final_metrics.py']:
        raise ValueError('Frozen scorer changed')
    if sha(bundle/'private_export_audit.json')!=contract['files']['private_export_audit.json']:
        raise ValueError('Site-audit integrity failed')
    plan=json.loads((folder/'evaluation_contract.json').read_text(encoding='utf-8'))
    if plan['scope']!='frozen held-out test':raise ValueError('Only the frozen held-out run may be scored here')
    rows=json.loads((folder/'paired_results.json').read_text(encoding='utf-8'))
    if len(rows)!=200 or any({r['case_id'] for r in rows if r['condition']==c}!=set(contract['case_ids']) for c in ('C','D')):
        raise ValueError('Incomplete frozen held-out sample')
    audit=json.loads((bundle/'private_export_audit.json').read_text(encoding='utf-8'))
    result=analyse(rows,{r['case_id']:r['site_id'] for r in audit})
    result['paired_results_sha256']=sha(folder/'paired_results.json')
    atomic_json(folder/'final_evaluation_metrics.json',result)
    print('Saved final_evaluation_metrics.json; classification scores computed without an LLM judge.')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();run(args.root,args.run)
