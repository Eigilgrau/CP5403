"""Run eight synthetic grounding checks with a separately pinned Llama judge.

No diagnosis generation or PHM explanation judging is performed here.
Expected answers are opened only after every calibration attempt is saved.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import phm_diagnosis as base
import phm_grounding_review as grounding
from phm_diagnosis_qwen import _infer
from phm_output_schema import object_schema, FORMAT_ENFORCER_VERSION
from phm_run_reporting import RunReport, atomic_json, atomic_text

MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
VERSION = 'llama-grounding-calibration-v1'
PROFILES = {
    'llama': {'model': MODEL, 'revision': REVISION, 'name': 'Llama',
              'version': VERSION, 'cache': 'llama_grounding_calibration_v1',
              'scope': 'Separate model family; not independent human validation.'},
    'qwen': {'model': 'Qwen/Qwen3-4B-Instruct-2507',
             'revision': 'cdbee75f17c01a7cc42f958dc650907174af0554', 'name': 'Qwen',
             'version': 'qwen-grounding-calibration-v1', 'cache': 'qwen_grounding_calibration_v1',
             'scope': 'Same checkpoint as diagnosis; model self-evaluation, not independent validation.'}}


def schema(request):
    payload=json.loads(request['messages'][1]['content'])
    statements=payload['statements']; evidence=payload['evidence']
    sources=['QUERY','NOTES']+[h['case_id'] for h in evidence['historical_cases']]+[p['chunk_id'] for p in evidence['literature']]
    citations=sorted({c for s in statements for c in s['cited_sources']})
    citation=object_schema({'source_id':{'type':'string', **({'enum':citations} if citations else {})},
                            'verdict':{'type':'string','enum':list(grounding.RUBRIC['citation_verdicts'])}})
    item=object_schema({'statement_id':{'type':'string','enum':[s['statement_id'] for s in statements]},
                        'verdict':{'type':'string','enum':list(grounding.RUBRIC['verdicts'])},
                        'rationale':{'type':'string','minLength':1},
                        'evidence_refs':{'type':'array','maxItems':len(sources),'items':{'type':'string','enum':sources}},
                        'citation_checks':{'type':'array','maxItems':len(citations),'items':citation}})
    return object_schema({'assessments':{'type':'array','minItems':len(statements),'maxItems':len(statements),'items':item}})


def prepare(pack, config):
    prepared, records=[],[]
    for line in (pack/'calibration_requests.jsonl').read_text(encoding='utf-8').splitlines():
        original=json.loads(line)
        if original['request_id']!=grounding.fingerprint({k:v for k,v in original.items() if k!='request_id'}):
            raise ValueError('Calibration request fingerprint mismatch')
        request={**deepcopy(original),'source_request_id':original['request_id'],
                 'case_id':original['request_id'],'condition':'judge_calibration',
                 'generator':deepcopy(config['generator']),'evidence_sha256':grounding.fingerprint(original['messages']),
                 'prompt_version':config.get('judge_version',VERSION),'format_enforcer_version':FORMAT_ENFORCER_VERSION,'output_schema':schema(original)}
        request.pop('request_id');request['request_id']=base.fingerprint(request)
        prepared.append(request);records.append({'case_id':request['case_id'],'_judge_request':request})
    if len(prepared)!=8 or len({q['request_id'] for q in prepared})!=8:
        raise ValueError('Expected eight distinct calibration requests')
    return prepared,records


def validate(raw, record):
    try:
        answer=json.loads(raw,object_pairs_hook=base._unique_object)
        grounding.validate_judgment(record['_judge_request'],answer)
        return {'valid':True,'errors':[],'parsed':answer}
    except (ValueError,TypeError,KeyError) as exc:
        return {'valid':False,'errors':[str(exc)],'parsed':None}


def inspect(path, request, record):
    check,response=base.inspect_cache(path,request,record)
    if response is None or check['status'] in {'runtime_error','invalid_cache'}:
        return check,response
    result=validate(response.get('raw_output'),record)
    return {'status':'valid' if result['valid'] else 'invalid_response',**result},response


def capture(report,bundle,cache,requests,records,score=False):
    details=[]
    for q,r in zip(requests,records,strict=True):
        check,response=inspect(Path(cache)/(q['request_id']+'.json'),q,r)
        details.append({'request_id':q['request_id'],'source_request_id':q['source_request_id'],
                        'checks':check,'response':response})
    atomic_json(report.folder/'judge_responses.json',details)
    return details


def score(details, answers):
    lookup={a['request_id']:a for a in answers}
    if len(lookup)!=len(details) or set(lookup)!={d['source_request_id'] for d in details}:
        raise ValueError('Calibration answer coverage mismatch')
    rows=[]
    for d in details:
        expected=lookup[d['source_request_id']]
        parsed=d['checks'].get('parsed') if d['checks']['status']=='valid' else None
        assessment=parsed['assessments'][0] if parsed else None
        verdict=assessment['verdict'] if assessment else None
        citations=[c['verdict'] for c in assessment['citation_checks']] if assessment else None
        rows.append({'source_request_id':d['source_request_id'],'status':d['checks']['status'],
                     'verdict':verdict,'expected_verdict':expected['expected_verdict'],
                     'statement_matches':verdict==expected['expected_verdict'],
                     'citation_verdicts':citations,'expected_citation_verdicts':expected['expected_citation_verdicts'],
                     'citation_matches':citations==expected['expected_citation_verdicts']})
    return {'scope':'Eight synthetic assistant-authored calibration examples; not human agreement or research performance.',
            'attempts':len(rows),'valid_outputs':sum(r['status']=='valid' for r in rows),
            'statement_matches':sum(r['statement_matches'] for r in rows),
            'fully_matching_examples':sum(r['statement_matches'] and r['citation_matches'] for r in rows),
            'cited_examples':sum(bool(r['expected_citation_verdicts']) for r in rows),
            'cited_examples_matching':sum(bool(r['expected_citation_verdicts']) and r['citation_matches'] for r in rows),
            'next_action':'Review this calibration before real-explanation judging. Passing is a basic check, not validation of judge reliability.',
            'rows':rows}


def run(root,prepare_only=False,judge='llama'):
    profile=PROFILES[judge]
    root=Path(root).resolve();pack=root/'Outputs/reviews/grounding_v1'
    report=RunReport(root,profile['name']+'_grounding_calibration')
    try:
        config=deepcopy(report.configure(root/'Files/experiment_config.json'))
        config['generator'].update(model=profile['model'],revision=profile['revision'],max_input_tokens=6144,max_new_tokens=512)
        config['judge_version']=profile['version']
        config['judge_scope']=profile['scope']
        atomic_json(report.folder/'judge_config.json',config)
        requests,records=prepare(pack,config)
        atomic_text(report.folder/'judge_requests.jsonl',''.join(base.canonical(r)+'\n' for r in requests))
        report.note(profile['scope']+' Calibration only. Expected answers are excluded from prompts. No automatic retry or model fallback.')
        if prepare_only:
            return {'requests':len(requests),'inference_executed':False,'model':profile['model'],'revision':profile['revision'],'judge_scope':profile['scope']}
        cache=root/'Outputs/judges'/profile['cache']/'responses'
        with report.step(profile['name']+'_calibration_inference'):
            result=_infer(requests,records,cache,config,report,False,
                          inspect_fn=inspect,validate_fn=validate,capture_fn=capture)
        atomic_json(report.folder/'judge_inference_summary.json',result)
        # The shared inference helper uses a legacy runtime key; identify the actual judge explicitly.
        runtime=report.data['environment'].pop('qwen_runtime',None)
        if runtime:
            report.data['environment']['judge_runtime']=runtime
            atomic_json(report.folder/'judge_runtime.json',runtime)
        details=capture(report,None,cache,requests,records)
        answers=[json.loads(line) for line in (pack/'calibration_answers_DO_NOT_SEND_TO_JUDGE.jsonl').read_text(encoding='utf-8').splitlines()]
        metrics=score(details,answers)
        metrics['judge_model']=profile['model']
        metrics['judge_scope']=profile['scope']
        atomic_json(report.folder/'calibration_metrics.json',metrics)
        report.data['metrics']['judge_calibration']=metrics
        if metrics['fully_matching_examples']!=metrics['attempts']:
            report.data['status']='completed_with_issues'
        report.save()
        return metrics
    except BaseException as exc:
        report.fail(exc);raise
    finally:
        report.finish();print('Saved report:',report.folder)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--judge',choices=sorted(PROFILES),default='llama')
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.prepare_only,args.judge),indent=2))
