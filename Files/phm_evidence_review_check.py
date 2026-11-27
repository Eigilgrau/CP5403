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

import phm_evidence_design as design
import phm_evidence_menu_smoke as selection
VERSION='evidence-review-check-v1'


def prepare(root):
    root=Path(root)
    folder=root/design.OUTPUT
    contract=json.loads((folder/'design_contract.json').read_text(encoding='utf-8'))
    if design.sha(root/'Files/phm_evidence_design.py')!=contract['code_sha256']:
        raise ValueError('Design code not synced')
    if design.sha(folder/'reviewed_candidate_requests.jsonl')!=contract['requests_sha256']:
        raise ValueError('Design requests not synced')
    if design.sha(root/design.SOURCE/'manifest.json')!=contract['source_manifest_sha256']:
        raise ValueError('Training-reference manifest changed')
    original,records=selection.prepare(root)
    keys={(q['case_id'],q['condition']) for q in original}
    lookup={r['case_id']:r for r in records}
    requests=[q for q in base.read_rows(folder/'reviewed_candidate_requests.jsonl') if (q['case_id'],q['condition']) in keys]
    if len(requests)!=4:raise ValueError('Expected exactly four paired checks')
    return requests,[{**lookup[q['case_id']],'_request':q} for q in requests]


def validate(raw,record):
    return design.validate(raw,record['_request'])


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
    report=RunReport(root,'four_response_evidence_review')
    try:
        config=deepcopy(report.configure(root/'Files/experiment_config.json'))
        config['generator'].update(max_input_tokens=6144,max_new_tokens=384)
        if any(q['generator']!=config['generator'] for q in requests):raise ValueError('Generator settings differ')
        atomic_text(report.folder/'engineering_requests.jsonl',''.join(base.canonical(q)+'\n' for q in requests))
        atomic_json(report.folder/'engineering_contract.json',{
            'version':VERSION,'maximum_calls':4,'free_text_fields':0,
            'reference_labels_loaded':False,'performance_scoring':False,'automatic_expansion':False,
            'scope':'Technical completion and evidence-reference checks only. No accuracy or grounding claim.'})
        cache=root/'Outputs/diagnosis/evidence_review_check_v1'
        _infer(requests,records,cache,config,report,False,inspect_fn=inspect,validate_fn=validate,capture_fn=capture)
        outputs=capture(report,None,cache,requests,records)
        summary={'expected_responses':4,'valid_outputs':sum(o['checks']['status']=='valid' for o in outputs),
                 'token_limit_hits':sum(bool((o['response'] or {}).get('generation_hit_limit')) for o in outputs),
                 'statuses':[o['checks']['status'] for o in outputs],
                 'next_action':'STOP for raw-output review. No larger run is authorised by this result.',
                 'accuracy_evaluated':False,'semantic_grounding_established':False}
        atomic_json(report.folder/'engineering_summary.json',summary)
        atomic_text(report.folder/'REVIEW_OUTPUTS.txt','\n\n'.join(
            f"{o['condition']} / {o['case_id']} / {o['checks']['status']}\n"+str((o['response'] or {}).get('raw_output'))+'\nResolved evidence and review warnings: '+json.dumps(o['checks'],indent=2) for o in outputs))
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
