"""One bounded, post-test development comparison. Never loads held-out data."""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import phm_diagnosis as base
import phm_two_stage as old
from phm_diagnosis_qwen import _infer
from phm_final_metrics import classification
from phm_run_reporting import RunReport, atomic_json, atomic_text

BUNDLE = 'Datasets/PHM15-diagnosis/recovery_validation_v1'
VERSION = 'override-justification-v1'
EXTRA = '''
Also return decision_justification: one short sentence, at most twenty-eight words.
If your class differs from score_order_descending[0], state concrete observed counterevidence against that leading class and why it supports your choice. Merely finding a historical case with the chosen label, a matching baseline value, or a general literature passage is insufficient counterevidence.
If you agree with the leading class, briefly identify the supporting evidence. If abstaining, state the unresolved conflict. Do not invent physical channel meanings. You retain the final classification decision; no rule forces you to follow the numerical ranking.
'''


def revised(request):
    q = deepcopy(request)
    q['prompt_version'] = VERSION + '-' + q['condition']
    q['assembly_version'] = VERSION
    q['output_schema']['properties']['decision_justification'] = {'type': 'string', 'minLength': 1}
    q['output_schema']['required'].append('decision_justification')
    q['messages'][0]['content'] = q['messages'][0]['content'].replace(
        'Return only the required JSON fields, without prose:', 'Return only JSON with these fields:') + EXTRA
    return old.identify(q)


def validate(raw, record):
    q = record['_request']
    if q.get('assembly_version') != VERSION or q['task_stage'] != 'decision':
        return old.validate(raw, record)
    try:
        parsed = json.loads(raw, object_pairs_hook=base._unique_object)
        if not isinstance(parsed, dict) or set(parsed) != old.DECISION_KEYS | {'decision_justification'}:
            raise ValueError('unexpected_revised_fields')
        reason = parsed.pop('decision_justification')
        if not isinstance(reason, str) or not reason.strip() or len(reason.split()) > 28:
            raise ValueError('justification_requires_one_to_28_words')
        check = old.validate(base.canonical(parsed), record)
        if check['valid']:
            check['parsed']['decision_justification'] = reason
            check['justification_support'] = 'Not automatically established; text presence is not entailment.'
        return check
    except (ValueError, TypeError, KeyError) as exc:
        return {'valid': False, 'errors': [str(exc)], 'parsed': None}


def inspect(path, request, record):
    check, response = base.inspect_cache(path, request, record)
    if response is None or check['status'] in {'runtime_error', 'invalid_cache'}:
        return check, response
    checked = validate(response.get('raw_output'), {**record, '_request': request})
    return {'status': 'valid' if checked['valid'] else 'invalid_response', **checked}, response


def capture(reporter, bundle, cache, requests, records, score=False):
    details = []
    for q, r in zip(requests, records, strict=True):
        check, response = inspect(Path(cache)/(q['request_id']+'.json'), q, r)
        details.append({'case_id':q['case_id'], 'condition':q['condition'], 'request_id':q['request_id'],
                        'checks':check, 'response':response})
    if requests:
        atomic_json(reporter.folder/(requests[0]['task_stage']+'_responses.json'), details)
    return details


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(root):
    folder = Path(root)/BUNDLE
    contract = json.loads((folder/'export_contract.json').read_text(encoding='utf-8'))
    if contract['scope'] != 'post-test development validation' or contract['version'] != VERSION:
        raise ValueError('Wrong recovery bundle scope')
    for name, expected in contract['code_sha256'].items():
        if sha(Path(root)/'Files'/name) != expected:
            raise ValueError('Code not synced or changed: '+name)
    # Private outcomes are deliberately not read during setup or generation.
    for name, expected in contract['files'].items():
        if name.startswith('private_'):
            continue
        if sha(folder/name) != expected:
            raise ValueError('Bundle not synced or changed: '+name)
    records = base.read_rows(folder/'validation_evidence.jsonl')
    if [r['case_id'] for r in records] != contract['case_ids']:
        raise ValueError('Sample identity mismatch')
    dictionary = json.loads((folder/'feature_dictionary.json').read_text(encoding='utf-8'))
    for r in records:
        base.validate_evidence(r, dictionary['feature_order'], 3)
    requests=base.read_rows(folder/'requests.jsonl')
    expected={(r['case_id'],c,v) for r in records for c in ('C','D') for v in ('original','revised')}
    observed={(q['case_id'],q['condition'],'revised' if q['assembly_version']==VERSION else 'original') for q in requests}
    if len(requests)!=len(expected) or observed!=expected or len({q['request_id'] for q in requests})!=len(requests):
        raise ValueError('Incomplete or duplicate paired request set')
    generators={base.canonical(q['generator']) for q in requests}
    if len(generators)!=1:raise ValueError('Unequal generation settings')
    return contract, records, requests


def metrics(rows):
    out = {}
    for variant in ('original','revised'):
        out[variant] = {}
        for c in ('C','D'):
            group = [r for r in rows if r['variant']==variant and r['condition']==c]
            m = {'all_selected':classification(group)}
            for cohort in ('natural','fault_challenge'):
                sub = [r for r in group if r['cohort']==cohort]
                m[cohort] = classification(sub)
            m.update(valid=sum(r['status']=='valid' for r in group),
                     harmful_overrides=sum(r['score_top']==r['reference_label'] and r['predicted_class'] not in (None,base.ABSTAIN,r['score_top']) for r in group),
                     helpful_overrides=sum(r['score_top']!=r['reference_label'] and r['predicted_class']==r['reference_label'] for r in group),
                     abstentions=sum(r['predicted_class']==base.ABSTAIN for r in group))
            out[variant][c] = m
    return out


def improvement_gate(scored):
    checks = {}
    for c in ('C','D'):
        a,b = scored['original'][c],scored['revised'][c]
        checks[c] = {
            'macro_f1_gain_at_least_002': b['all_selected']['macro_f1']-a['all_selected']['macro_f1'] >= .02-1e-12,
            'natural_accuracy_not_lower': b['natural']['accuracy'] >= a['natural']['accuracy'],
            'harmful_overrides_reduced': b['harmful_overrides'] < a['harmful_overrides'],
            'at_least_95_percent_valid': b['valid']/b['all_selected']['queries'] >= .95,
            'neither_fault_recall_lower': all(b['all_selected']['per_class'][label]['recall'] >= a['all_selected']['per_class'][label]['recall'] for label in base.LABELS[1:])}
    return {'passed':all(all(x.values()) for x in checks.values()), 'checks':checks,
            'meaning':'Pragmatic development stop rule, not statistical significance or independent test confirmation.'}


def run(root, prepare_only=False):
    root = Path(root).resolve()
    contract, records, requests = load_bundle(root)
    if prepare_only:
        print(f"Ready: {len(records)} validation cases; {len(requests)} decision calls; explanation calls only if the fixed gate passes.")
        return
    report = RunReport(root, 'bounded_CD_override_validation')
    try:
        config = deepcopy(report.configure(root/'Files/experiment_config.json'))
        config['generator'].update(max_input_tokens=6144,max_new_tokens=384)
        if any(q['generator']!=config['generator'] for q in requests):
            raise ValueError('Runtime settings differ from prepared requests')
        lookup = {r['case_id']:r for r in records}
        attached = [{**lookup[q['case_id']], '_request':q} for q in requests]
        cache = root/'Outputs/diagnosis/recovery_validation_v1'
        atomic_json(report.folder/'evaluation_contract.json',contract)
        atomic_text(report.folder/'decision_requests.jsonl',''.join(base.canonical(q)+'\n' for q in requests))
        _infer(requests, attached, cache/'decision', config, report, False,
               inspect_fn=inspect, validate_fn=validate, capture_fn=capture)
        outputs = capture(report,None,cache/'decision',requests,attached)
        if any(x['checks']['status'] in ('not_run','invalid_cache') for x in outputs):
            raise RuntimeError('Incomplete or invalid-cache decisions. Resume pending calls only; no comparison scored.')
        folder=root/BUNDLE
        for name in ('private_validation_labels.jsonl','private_export_audit.json'):
            if sha(folder/name)!=contract['files'][name]:
                raise ValueError('Private scoring file changed: '+name)
        labels={x['case_id']:x['reference_label'] for x in base.read_rows(folder/'private_validation_labels.jsonl')}
        cohorts={x['case_id']:x['cohort'] for x in json.loads((folder/'private_export_audit.json').read_text())}
        rows=[]
        for q,o in zip(requests,outputs,strict=True):
            record=lookup[q['case_id']]; parsed=o['checks'].get('parsed') or {}
            rows.append({'case_id':q['case_id'],'condition':q['condition'],
                'variant':'revised' if q['assembly_version']==VERSION else 'original',
                'cohort':cohorts[q['case_id']], 'status':o['checks']['status'],
                'reference_label':labels[q['case_id']],
                'predicted_class':parsed.get('predicted_class') if o['checks']['status']=='valid' else None,
                'score_top':max(record['class_scores'],key=record['class_scores'].get),
                'decision_justification':parsed.get('decision_justification')})
        scored=metrics(rows); gate=improvement_gate(scored)
        atomic_json(report.folder/'classification_results.json',rows)
        atomic_json(report.folder/'comparison_metrics.json',scored)
        atomic_json(report.folder/'improvement_gate.json',gate)
        print('Classification gate:',json.dumps(gate),flush=True)
        if gate['passed']:
            explanations=[]; exp_records=[]; parents=[]
            for q,o in zip(requests,outputs,strict=True):
                if o['checks']['status']!='valid':continue
                ex=old.explanation_request(q,o['checks']['parsed'])
                explanations.append(ex); exp_records.append({**lookup[q['case_id']], '_request':ex})
                parents.append((q['case_id'],q['condition'],'revised' if q['assembly_version']==VERSION else 'original'))
            atomic_text(report.folder/'explanation_requests.jsonl',''.join(base.canonical(q)+'\n' for q in explanations))
            _infer(explanations,exp_records,cache/'explanation',config,report,False,
                   inspect_fn=inspect,validate_fn=validate,capture_fn=capture)
            exp_outputs=capture(report,None,cache/'explanation',explanations,exp_records)
            if any(x['checks']['status'] in ('not_run','invalid_cache') for x in exp_outputs):
                raise RuntimeError('Explanation phase incomplete. Resume pending calls only.')
            exp_lookup={key:o for key,o in zip(parents,exp_outputs,strict=True)}
            completion={}
            for v in ('original','revised'):
                completion[v]={c:{'queries':len(records),'valid_explanations':sum(exp_lookup.get((r['case_id'],c,v),{}).get('checks',{}).get('status')=='valid' for r in records)} for c in ('C','D')}
            atomic_json(report.folder/'explanation_completion.json',completion)
            report.note('Classification gate passed; explanation completion recorded. Semantic grounding not established. No automatic promotion or test rerun.')
        else:
            report.note('Classification gate failed: stopped automatically. No explanation calls, model escalation or retries. Retain negative development result.')
        report.data['metrics']['comparison']=scored
        report.data['metrics']['improvement_gate']=gate
        if any(x['checks']['status']!='valid' for x in outputs):report.data['status']='completed_with_issues'
        if gate['passed'] and any(x['checks']['status']!='valid' for x in exp_outputs):report.data['status']='completed_with_issues'
        report.save()
    except BaseException as exc:
        report.fail(exc)
        raise
    finally:
        report.finish()
        print('Saved comparison:',report.folder)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args();run(a.root,a.prepare_only)
