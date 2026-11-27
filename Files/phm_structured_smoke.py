"""Four generation checks only. No outcome labels, scoring, or automatic expansion."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import phm_diagnosis as base
import phm_two_stage as old
from phm_diagnosis_qwen import _infer
from phm_run_reporting import RunReport, atomic_json, atomic_text

VERSION='structured-comparison-smoke-v1'
BUNDLE='Datasets/PHM15-diagnosis/recovery_validation_v1'
PROMPT=old.DECISION_PROMPT+"""
Also return comparison_feature and comparison_relation. No prose or justification text.
For a class decision, choose comparison_feature from selected_features and compare its query value with that feature in selected_case_id. comparison_relation is lower, equal or higher (query relative to historical). Use the displayed values. These numerical relations do not prove a shared fault or a physical mechanism. Select a discriminative feature where possible, considering conflicting cases rather than only a matching baseline.
When abstaining, use null for both comparison fields and selected_case_id, and an empty selected_features list.
"""


def revised(q):
    q=deepcopy(q)
    q.update(prompt_version=VERSION+'-'+q['condition'],assembly_version=VERSION)
    q['messages'][0]['content']=PROMPT
    names=q['output_schema']['properties']['selected_features']['items']['enum']
    q['output_schema']['properties'].update(
        comparison_feature={'anyOf':[{'type':'string','enum':names},{'type':'null'}]},
        comparison_relation={'anyOf':[{'type':'string','enum':['lower','equal','higher']},{'type':'null'}]})
    q['output_schema']['required'].extend(['comparison_feature','comparison_relation'])
    return old.identify(q)


def prepare(root):
    folder=Path(root)/BUNDLE
    contract=json.loads((folder/'export_contract.json').read_text(encoding='utf-8'))
    # Only these three PUBLIC artifacts are opened. No database, private audit or labels.
    for name in ('validation_evidence.jsonl','requests.jsonl','feature_dictionary.json'):
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=contract['files'][name]:
            raise ValueError('Public file not synced: '+name)
    records=base.read_rows(folder/'validation_evidence.jsonl')
    dictionary=json.loads((folder/'feature_dictionary.json').read_text(encoding='utf-8'))
    # Select an agreement and a conflict between numerical ranking and nearest historical
    # label when available. This is PUBLIC evidence, never the query's reference label.
    def agrees(r):return max(r['class_scores'],key=r['class_scores'].get)==r['historical_cases'][0]['recorded_outcome']
    selected=[]
    for flag in (True,False):
        candidate=next((r for r in records if agrees(r)==flag),None)
        if candidate:selected.append(candidate)
    for r in records:
        if len(selected)==2:break
        if r not in selected:selected.append(r)
    ids={r['case_id'] for r in selected}
    requests=[revised(q) for q in base.read_rows(folder/'requests.jsonl')
              if q['assembly_version']==old.VERSION and q['case_id'] in ids]
    if len(requests)!=4 or {(q['case_id'],q['condition']) for q in requests}!={(i,c) for i in ids for c in ('C','D')}:
        raise ValueError('Smoke check must contain exactly two paired cases')
    for r in selected:base.validate_evidence(r,dictionary['feature_order'],3)
    lookup={r['case_id']:r for r in selected}
    return requests,[{**lookup[q['case_id']],'_request':q} for q in requests]


def validate(raw,record):
    try:
        parsed=json.loads(raw,object_pairs_hook=base._unique_object)
        if not isinstance(parsed,dict) or set(parsed)!=old.DECISION_KEYS|{'comparison_feature','comparison_relation'}:
            raise ValueError('unexpected_fields')
        feature=parsed.pop('comparison_feature');relation=parsed.pop('comparison_relation')
        result=old.validate(base.canonical(parsed),record)
        if not result['valid']:return result
        verified=None
        if parsed['predicted_class']==base.ABSTAIN:
            if feature is not None or relation is not None or parsed['selected_case_id'] is not None or parsed['selected_features']:
                raise ValueError('abstention_requires_null_comparison')
        else:
            if feature not in parsed['selected_features']:raise ValueError('comparison_feature_not_selected')
            visible=json.loads(record['_request']['messages'][1]['content'])
            other=next((h for h in visible['historical_cases'] if h['case_id']==parsed['selected_case_id']),None)
            if other is None:raise ValueError('comparison_case_missing')
            x=visible['query']['feature_values'].get(feature);y=other['feature_values'].get(feature)
            if x is None or y is None:raise ValueError('comparison_value_missing')
            expected='lower' if x<y else 'higher' if x>y else 'equal'
            if relation!=expected:raise ValueError('comparison_relation_disagrees_with_visible_values')
            verified={'feature':feature,'query_value':x,'historical_value':y,'relation':relation,
                      'case_id':other['case_id'],'historical_recorded_outcome':other['recorded_outcome'],
                      'origin':'Python verifies the LLM-selected relation using displayed evidence; not LLM-generated prose.'}
        result['parsed'].update(comparison_feature=feature,comparison_relation=relation)
        result['verified_selected_comparison']=verified
        result['semantic_support']='Numerical relation checked; diagnostic sufficiency not established.'
        return result
    except (ValueError,TypeError,KeyError) as exc:
        return {'valid':False,'errors':[str(exc)],'parsed':None}


def inspect(path,q,record):
    check,response=base.inspect_cache(path,q,record)
    if response is None or check['status'] in ('invalid_cache','runtime_error'):return check,response
    result=validate(response.get('raw_output'),record)
    return {'status':'valid' if result['valid'] else 'invalid_response',**result},response


def capture(reporter,bundle,cache,requests,records,score=False):
    if score:raise ValueError('Scoring is disabled in engineering smoke checks')
    result=[]
    for q,r in zip(requests,records,strict=True):
        check,response=inspect(Path(cache)/(q['request_id']+'.json'),q,r)
        result.append({'case_id':q['case_id'],'condition':q['condition'],'checks':check,'response':response})
    atomic_json(reporter.folder/'engineering_responses.json',result)
    return result


def run(root,prepare_only=False):
    root=Path(root).resolve();requests,records=prepare(root)
    if prepare_only:
        print('Prepared exactly FOUR responses; scoring and automatic expansion are absent.');return
    report=RunReport(root,'four_response_structured_smoke')
    try:
        config=deepcopy(report.configure(root/'Files/experiment_config.json'))
        config['generator'].update(max_input_tokens=6144,max_new_tokens=384)
        if any(q['generator']!=config['generator'] for q in requests):raise ValueError('Generator settings differ')
        atomic_text(report.folder/'engineering_requests.jsonl',''.join(base.canonical(q)+'\n' for q in requests))
        atomic_json(report.folder/'engineering_contract.json',{
            'version':VERSION,'maximum_calls':4,'free_text_fields':0,
            'reference_labels_loaded':False,'performance_scoring':False,'automatic_expansion':False,
            'scope':'Technical completion and evidence-reference checks only. No accuracy or grounding claim.'})
        cache=root/'Outputs/diagnosis/structured_smoke_v1'
        _infer(requests,records,cache,config,report,False,inspect_fn=inspect,validate_fn=validate,capture_fn=capture)
        outputs=capture(report,None,cache,requests,records)
        summary={'expected_responses':4,'valid_outputs':sum(o['checks']['status']=='valid' for o in outputs),
                 'token_limit_hits':sum(bool((o['response'] or {}).get('generation_hit_limit')) for o in outputs),
                 'statuses':[o['checks']['status'] for o in outputs],
                 'next_action':'STOP for raw-output review. No larger run is authorised by this result.',
                 'accuracy_evaluated':False,'semantic_grounding_established':False}
        atomic_json(report.folder/'engineering_summary.json',summary)
        atomic_text(report.folder/'REVIEW_OUTPUTS.txt','\n\n'.join(
            f"{o['condition']} / {o['case_id']} / {o['checks']['status']}\n"+str((o['response'] or {}).get('raw_output'))+'\nVerified comparison: '+json.dumps(o['checks'].get('verified_selected_comparison')) for o in outputs))
        report.data['metrics']['engineering']=summary
        if summary['valid_outputs']!=4:report.data['status']='completed_with_issues'
        print(json.dumps(summary,indent=2));report.save()
    except BaseException as exc:
        report.fail(exc);raise
    finally:
        report.finish();print('Saved engineering check:',report.folder)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args();run(args.root,args.prepare_only)
