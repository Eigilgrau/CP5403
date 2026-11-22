"""Exploratory Qwen14B review of saved PHM explanations, not diagnosis generation."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import phm_diagnosis as base
import phm_grounding_review as grounding
import phm_judge_v2 as judge
from phm_diagnosis_qwen import _infer
from phm_run_reporting import RunReport, atomic_json, atomic_text

VERSION='phm-grounding-exploratory-v1'


def prepare(root,config):
    pack=root/'Outputs/reviews/grounding_v1'
    summary=json.loads((pack/'pack_summary.json').read_text(encoding='utf-8'))
    source=root/'Outputs/runs'/summary['source_run']
    for name,expected in summary['source_sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Source report hash mismatch: '+name)
    manifests=json.loads((pack/'review_manifest.json').read_text(encoding='utf-8'))
    ready={m['judge_request_id']:m for m in manifests if m['review_status']=='ready'}
    requests=[];mapping={}
    for original in judge.load_rows(pack/'judge_requests.jsonl'):
        if original['request_id']!=grounding.fingerprint({k:v for k,v in original.items() if k!='request_id'}):
            raise ValueError('Grounding pack request fingerprint mismatch')
        meta=ready[original['request_id']]
        payload=json.loads(original['messages'][1]['content'])
        for statement in payload['statements']:
            item={'item_id':original['request_id']+'_'+statement['statement_id'], 'partition':'PHM_exploratory',
                  'statement':statement['text'],'evidence':payload['evidence'],'cited_sources':statement['cited_sources']}
            tasks=[('grounding',None)]+[('citation',c) for c in item['cited_sources']]
            for task,citation in tasks:
                q=judge.build_request(item,task,citation,config)
                visible=json.loads(q['messages'][1]['content'])
                # Compute aids only for explicit feature names (including space-form
                # spellings). All original observations remain available unchanged.
                text=statement['text'].lower().replace('_',' ')
                visible['python_verified_relations']=[r for r in visible['python_verified_relations']
                                                     if r['feature'].lower().replace('_',' ') in text]
                visible['numeric_aid_scope']='Only explicitly named features have computed comparison aids. Original observations are retained.'
                q['messages'][1]['content']=base.canonical(visible)
                q['evidence_sha256']=base.fingerprint(visible)
                q['prompt_version']=VERSION
                q.pop('request_id');q['request_id']=base.fingerprint(q)
                mapping[q['request_id']]={'condition':meta['condition'],'case_id':meta['case_id'],
                    'statement_id':statement['statement_id'],'kind':statement['kind'],'text':statement['text'],
                    'source_request_id':meta['source_request_id'],'task':task,'source_id':citation,
                    'citation_kind':None if citation is None else 'literature' if citation in
                        {p['chunk_id'] for p in payload['evidence']['literature']} else 'historical_case'}
                requests.append(q)
    if len(ready)!=len(judge.load_rows(pack/'judge_requests.jsonl')):raise ValueError('Review mapping coverage mismatch')
    return requests,[{'case_id':q['case_id'],'_request':q} for q in requests],mapping,manifests


def validate(raw,record):
    """Retain explicit verdicts separately from reference-array defects."""
    try:
        a=json.loads(raw,object_pairs_hook=base._unique_object)
        if not isinstance(a,dict) or set(a)!={'verdict','basis','evidence_refs'}:
            raise ValueError('Invalid judge fields')
        props=record['_request']['output_schema']['properties']
        if a['verdict'] not in props['verdict']['enum'] or a['basis'] not in props['basis']['enum']:
            raise ValueError('Invalid verdict or basis')
        refs=a['evidence_refs'];defects=[]
        if not isinstance(refs,list) or any(not isinstance(r,str) for r in refs):
            defects.append('malformed_reference_array')
        else:
            if len(set(refs))!=len(refs):defects.append('duplicate_reference_ids')
            if any(r not in props['evidence_refs']['items']['enum'] for r in refs):defects.append('unknown_reference_ids')
        return {'valid':True,'verdict_valid':True,'strict_valid':not defects,'reference_defects':defects,
                'errors':[],'parsed':a,'reference_relevance':'Not established by identifier validation'}
    except (ValueError,TypeError,KeyError) as exc:
        return {'valid':False,'verdict_valid':False,'strict_valid':False,'errors':[str(exc)],'parsed':None}


def inspect(path,request,record):
    checked,response=base.inspect_cache(path,request,record)
    if response is None or checked['status'] in {'runtime_error','invalid_cache'}:return checked,response
    result=validate(response.get('raw_output'),record)
    status='invalid_response' if not result['valid'] else 'valid' if result['strict_valid'] else 'valid_with_reference_issues'
    return {'status':status,**result},response


def capture(report,bundle,cache,requests,records,score=False):
    details=[]
    for q,r in zip(requests,records,strict=True):
        checks,response=inspect(Path(cache)/(q['request_id']+'.json'),q,r)
        details.append({'request_id':q['request_id'],'checks':checks,'response':response})
    atomic_json(report.folder/'judge_responses.json',details)
    return details


def summarise(details,mapping,manifest):
    results=[]
    for d in details:
        checks=d['checks'];a=checks.get('parsed') if checks.get('verdict_valid') else None
        results.append({**mapping[d['request_id']],'request_id':d['request_id'],'status':checks['status'],
                        'verdict':a['verdict'] if a else None,'basis':a['basis'] if a else None,
                        'evidence_refs':a['evidence_refs'] if a else None,'strict_valid':checks.get('strict_valid',False),
                        'reference_defects':checks.get('reference_defects',[]),
                        **{k:(d.get('response') or {}).get(k) for k in ('input_tokens','output_tokens','generation_seconds')}})
    metrics={'scope':'Exploratory LLM-judge estimates on five complete explanations, not human-validated grounding accuracy.',
             'comparison_warning':'C and D have different completion coverage. Do not interpret their support rates as an unbiased paired comparison.',
             'reference_policy':'Explicit verdicts are retained with reference defects separately flagged; no raw output is repaired.',
             'conditions':{}}
    for condition in ('C','D'):
        m=[r for r in manifest if r['condition']==condition]
        group={'source_explanations':len(m),'parseable_explanations':sum(r['review_status']=='ready' for r in m),
               'unparseable_explanations':sum(r['review_status']!='ready' for r in m)}
        for task in ('grounding','citation'):
            subset=[r for r in results if r['condition']==condition and r['task']==task]
            verdicts=Counter(r['verdict'] or 'judge_failure' for r in subset)
            group[task]={'attempts':len(subset),'verdict_counts':dict(verdicts),
                'strict_valid':sum(r['strict_valid'] for r in subset),
                'reference_defect_outputs':sum(bool(r['reference_defects']) for r in subset),
                'supported_fraction_all_attempts':(verdicts['supported' if task=='grounding' else 'supports']/len(subset)) if subset else None}
            if task=='citation':
                group[task]['by_source_type']={}
                for kind in ('historical_case','literature'):
                    by_kind=[r for r in subset if r['citation_kind']==kind]
                    group[task]['by_source_type'][kind]={'attempts':len(by_kind),
                        'verdict_counts':dict(Counter(r['verdict'] or 'judge_failure' for r in by_kind)),
                        'supported_fraction_all_attempts':sum(r['verdict']=='supports' for r in by_kind)/len(by_kind) if by_kind else None}
        metrics['conditions'][condition]=group
    return metrics,results


def run(root,prepare_only=False):
    root=Path(root).resolve();report=RunReport(root,'Qwen14B_PHM_grounding_exploratory')
    try:
        config=deepcopy(report.configure(root/'Files/experiment_config.json'))
        model,revision=judge.PROFILES['qwen14b']
        config['generator'].update(model=model,revision=revision,max_input_tokens=6144,max_new_tokens=256)
        atomic_json(report.folder/'judge_config.json',config)
        reqs,recs,mapping,manifest=prepare(root,config)
        atomic_text(report.folder/'judge_requests.jsonl',''.join(base.canonical(r)+'\n' for r in reqs))
        atomic_json(report.folder/'statement_mapping.json',mapping)
        atomic_json(report.folder/'source_coverage.json',manifest)
        report.note('Qwen14B assesses saved text only. No labels, reference answers or C/D names enter judge prompts. Output budget 256 accommodates longer real case IDs; calibration used 160. Evidence reference defects remain separate from explicit verdict validity.')
        if prepare_only:return {'inference_executed':False,'requests':len(reqs),'grounding_requests':sum(q['task']=='grounding' for q in reqs),'citation_requests':sum(q['task']=='citation' for q in reqs)}
        cache=root/'Outputs/judges/PHM_grounding_v1/qwen14b/responses'
        with report.step('judge_saved_explanations'):
            result=_infer(reqs,recs,cache,config,report,False,inspect_fn=inspect,validate_fn=validate,capture_fn=capture)
        atomic_json(report.folder/'judge_inference_summary.json',result)
        runtime=report.data['environment'].pop('qwen_runtime',None)
        if runtime:report.data['environment']['judge_runtime']=runtime
        details=capture(report,None,cache,reqs,recs)
        metrics,results=summarise(details,mapping,manifest)
        atomic_json(report.folder/'grounding_metrics.json',metrics)
        atomic_json(report.folder/'statement_judgments.json',results)
        report.data['metrics']['grounding']=metrics
        if any(not r['strict_valid'] for r in results):report.data['status']='completed_with_issues'
        report.save();return metrics
    except BaseException as exc:
        report.fail(exc);raise
    finally:
        report.finish();print('Saved report:',report.folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.prepare_only),indent=2))
