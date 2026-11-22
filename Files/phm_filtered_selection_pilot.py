"""Paired C/D validation pilot with a separately reviewed literature index."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import argparse
import json
import re
from time import perf_counter
import numpy as np

import phm_diagnosis as base
from phm_diagnosis_qwen import _infer
from phm_literature_index import Encoder, rows, sha
from phm_run_reporting import RunReport, atomic_json, atomic_text
from phm_output_schema import output_schema, FORMAT_ENFORCER_VERSION

QUERY_CONTEXT = {
    'NoRecordedTarget': 'Missing or incomplete fault logs do not prove healthy operation. False alarms, operating transitions, sensor uncertainty and diagnostic limitations.',
    'RuntimeExcessive': 'HVAC excessive operating runtime, occupancy schedules, scheduling conflicts between source and load equipment, control sequencing and persistent operation.',
    'SupplyTempTooCold': 'HVAC supply air temperature too low, temperature setpoint control, sensor errors, cooling heating and airflow interactions; alternative fault explanations.'}

PAIRED_PROMPT_VERSION = 'paired-cd-v4-evidence-values'
REASONING_RULES = '''
Decision discipline for both conditions:
The supplied score_order_descending is computed from the query's regression scores. Read its first label as the largest score; never invert this ordering. These are uncalibrated estimates, not ground truth.
Evaluate NoRecordedTarget alongside both fault categories. A retrieved fault case does not establish that the query has a fault. A matched baseline value alone does not establish a discriminative fault pattern. If you reject the leading model label, identify concrete counterevidence and acknowledge the disagreement; do not claim a lower score is higher.
Anonymous channel changes describe measurements only. Lower values do not mean colder supply air; variability does not establish degradation or excessive runtime. Unknown physical meaning is not a missing measurement. If the evidence conflicts and cannot support a choice, InsufficientEvidence is permitted.
Keep numerical values exclusively in feature_evidence and comparison_claims. In explanation and limitation, use qualitative comparisons and exact feature names, without numeric quantities, probabilities, dates or thresholds. Keep these two prose fields together under sixty words. Do not repeat the structured facts in prose.
'''


def validate_paired(raw, record):
    result = base.validate_response(raw, record)
    parsed = result.get('parsed')
    if isinstance(parsed, dict):
        for field in ['explanation', 'limitation']:
            prose = parsed.get(field)
            if not isinstance(prose, str):
                continue
            for name in sorted(record['feature_values'], key=len, reverse=True):
                prose = prose.replace(name, '')
            for historical in record['historical_cases']:
                prose = prose.replace(historical['case_id'], '')
            prose = re.sub(r'\bch\d+\b', '', prose)
            if re.search(r'\d', prose):
                result['errors'].append('numerical_prose_not_allowed:' + field)
    result['valid'] = not result['errors']
    return result


def inspect_c_paired(path, request, record):
    checked, response = base.inspect_cache(path, request, record)
    if response is None or checked['status'] in {'runtime_error', 'invalid_cache'}:
        return checked, response
    validation = validate_paired(response.get('raw_output'), record)
    return {'status': 'valid' if validation['valid'] else 'invalid_response', **validation}, response


def validate_d(raw, record, supplied):
    result = validate_paired(raw, record)
    errors = [e for e in result['errors'] if e != 'literature_citations_forbidden_in_C']
    parsed = result.get('parsed')
    if isinstance(parsed, dict) and 'literature_citations' in parsed:
        citations = parsed['literature_citations']
        allowed = {p['chunk_id'] for p in supplied}
        if (not isinstance(citations, list) or len(citations) > len(allowed)
                or any(not isinstance(c, str) or c not in allowed for c in citations)
                or (isinstance(citations, list) and len({str(c) for c in citations}) != len(citations))):
            errors.append('invalid_or_unsupplied_literature_citation')
    return {**result, 'errors': errors, 'valid': not errors,
            'citation_scope': 'Citation membership only; claim support requires review.'}


def inspect_d(path, request, record):
    checked, response = base.inspect_cache(path, request, record)
    if response is None or checked['status'] in {'runtime_error', 'invalid_cache'}:
        return checked, response
    validation = validate_d(response.get('raw_output'), record, request['literature'])
    return {'status': 'valid' if validation['valid'] else 'invalid_response', **validation}, response


def _prepare_previous(root, config, maximum, bundle=None, loader=None):
    bundle = bundle or root / 'Datasets/PHM15-diagnosis/validation_pilot_v1'
    contract, dictionary, records = (loader or base.load_bundle)(bundle)
    if not 1 <= maximum <= len(records):
        raise ValueError('Invalid validation query count')
    records = records[:maximum]
    index = root / 'Datasets/PHM15-literature-D-approved-v1'
    manifest = json.loads((index / 'index_manifest.json').read_text(encoding='utf-8'))
    for name, key in [('approved_chunks.jsonl', 'chunks_sha256'), ('embeddings.npy', 'embeddings_sha256')]:
        if sha(index / name) != manifest[key]:
            raise ValueError('Literature index integrity failed')
    chunks = rows(index / 'approved_chunks.jsonl')
    if not chunks or any(not c.get('diagnostic_approved') for c in chunks):
        raise ValueError('Unapproved literature is prohibited')
    policy_path = root / 'Files/literature_retrieval_policy_v1.json'
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    known_ids = {c['chunk_id'] for c in chunks}
    if set(policy['eligible']) != set(QUERY_CONTEXT) or any(not ids or not set(ids) <= known_ids for ids in policy['eligible'].values()):
        raise ValueError('Retrieval policy contains missing or unknown approved passages')
    encoder = Encoder(index / 'model_assets')
    if encoder.model_hash != manifest['onnx_sha256']:
        raise ValueError('Embedding checkpoint changed')
    vectors = np.load(index / 'embeddings.npy', allow_pickle=False)
    c_requests, d_requests, retrieval = [], [], []
    for record in records:
        # Uses only a model estimate already available to C, never scoring labels.
        hypothesis = max(record['class_scores'], key=record['class_scores'].get)
        query = QUERY_CONTEXT[hypothesis]
        started = perf_counter()
        scores = vectors @ encoder.encode([query])[0]
        picked, documents = [], set()
        for i in np.argsort(-scores, kind='stable'):
            chunk = chunks[int(i)]
            if chunk['chunk_id'] not in policy['eligible'][hypothesis]:
                continue
            if chunk['document_id'] in documents:
                continue
            picked.append(chunk)
            documents.add(chunk['document_id'])
            if len(picked) == config['retrieval']['literature_k']:
                break
        if len(picked) != config['retrieval']['literature_k']:
            raise ValueError('Insufficient eligible documents; refusing silent retrieval fallback')
        baseline_ids, baseline_documents = [], set()
        for i in np.argsort(-scores, kind='stable'):
            candidate = chunks[int(i)]
            if candidate['document_id'] not in baseline_documents:
                baseline_ids.append(candidate['chunk_id'])
                baseline_documents.add(candidate['document_id'])
            if len(baseline_ids) == config['retrieval']['literature_k']:
                break
        unchanged_evidence = baseline_ids == [p['chunk_id'] for p in picked]
        retrieval_seconds = perf_counter() - started
        c = base.make_request(record, dictionary, config)
        c.pop('request_id')
        c['prompt_version'] = PAIRED_PROMPT_VERSION + '-C'
        c['output_schema'] = output_schema(record)
        c['format_enforcer_version'] = FORMAT_ENFORCER_VERSION
        c['messages'][0]['content'] += REASONING_RULES
        ranking = sorted(record['class_scores'], key=record['class_scores'].get, reverse=True)
        c['messages'][1]['content'] += '\nComputed query score order (largest first):\n' + base.canonical({'score_order_descending': ranking})
        c['request_id'] = base.fingerprint(c)
        d = deepcopy(c)
        d.pop('request_id')
        d.update(condition='D', prompt_version=PAIRED_PROMPT_VERSION + '-D',
                 literature=picked, literature_index_sha256=sha(index / 'index_manifest.json'),
                 retrieval_policy_version=policy['version'], retrieval_policy_sha256=sha(policy_path))
        # Identical model inputs reuse their original cache, including failures.
        if unchanged_evidence:
            d.pop('retrieval_policy_version')
            d.pop('retrieval_policy_sha256')
        d['output_schema'] = output_schema(record, picked)
        d['messages'][0]['content'] = d['messages'][0]['content'].replace(
            'using only the supplied observations, regression scores and historical cases.',
            'using the supplied observations, regression scores, historical cases and literature.')
        d['messages'][0]['content'] = d['messages'][0]['content'].replace(
            'Literature citations must be empty in C.',
            'In D, literature_citations is a list of supplied chunk_id strings supporting your explanation; leave it empty if none apply.')
        d['messages'][0]['content'] += ('\nLiterature is contextual evidence, not the query answer. '
            'External HVAC descriptions cannot establish anonymous channel meanings or a physical root cause. '
            'Do not copy an external equipment example as this query diagnosis. '
            'State when the physical measurements needed to apply a literature rule are unavailable. '
            'Explicitly assess the retrieved text: use a supported principle in your explanation or a supported applicability limit in your limitation, and cite the relevant chunk_id in literature_citations. '
            'If neither passage supports any statement about this case or its limitations, leave citations empty and state that the literature is inapplicable. Never cite a passage merely to fill the field.')
        visible = [{'chunk_id': x['chunk_id'], 'title': x['title'], 'pdf_pages': [x['page_start'], x['page_end']],
                    'scope': x['scope'], 'text': x['text']} for x in picked]
        d['messages'][1]['content'] += '\nRetrieved literature:\n' + base.canonical(visible)
        d['request_id'] = base.fingerprint(d)
        c_requests.append(c)
        d_requests.append(d)
        retrieval.append({'case_id': record['case_id'], 'query': query, 'query_basis': 'highest regression score; no reference label',
                          'chunks': [x['chunk_id'] for x in picked], 'retrieval_policy_version': policy['version'],
                          'eligible_chunks': policy['eligible'][hypothesis], 'retrieval_seconds': retrieval_seconds})
    return contract, records, c_requests, d_requests, retrieval


from phm_evidence_selection import make_selection_request, validate_selection, inspect_selection

# Explicit versioned runner: V1-V4 implementations and outputs remain available.
validate_paired = validate_selection
validate_d = validate_selection
inspect_c_paired = inspect_selection
inspect_d = inspect_selection


def prepare(root, config, maximum, bundle=None, loader=None):
    contract, records, cs, ds, retrieval = _prepare_previous(root, config, maximum, bundle, loader)
    cs = [make_selection_request(r, case) for r, case in zip(cs, records, strict=True)]
    ds = [make_selection_request(r, case) for r, case in zip(ds, records, strict=True)]
    return contract, records, cs, ds, retrieval


def run(root, maximum=10, prepare_only=False):
    root = Path(root).resolve()
    report = RunReport(root, 'paired_CD_filtered_retrieval_pilot')
    try:
        config = deepcopy(report.configure(root / 'Files/experiment_config.json'))
        # Same allowance for both conditions; retain every original feature/history entry.
        config['generator']['max_input_tokens'] = 6144
        atomic_json(report.folder / 'paired_config.json', config)
        report.note('Both C and D use a 6144-token input allowance, otherwise the original generator settings. C requests are unchanged and reuse matching caches; D uses a versioned passage eligibility filter. Generation settings and prompts remain unchanged.')
        with report.step('prepare_paired_evidence'):
            contract, records, cs, ds, retrieval = prepare(root, config, maximum)
            requests = cs + ds
            atomic_text(report.folder / 'diagnostic_requests.jsonl', ''.join(json.dumps(r) + '\n' for r in requests))
            atomic_json(report.folder / 'literature_retrieval.json', retrieval)
        lookup = {d['case_id']: d for d in ds}

        def capture(reporter, bundle, cache, reqs, recs, score=False):
            results = []
            details = []
            for request, record in zip(reqs, recs, strict=True):
                path = Path(cache) / (request['request_id'] + '.json')
                checked, response = (inspect_d if request['condition'] == 'D' else inspect_c_paired)(path, request, record)
                parsed = checked.get('parsed')
                row = {'case_id': record['case_id'], 'condition': request['condition'], 'status': checked['status'],
                       'predicted_class': parsed.get('predicted_class') if isinstance(parsed, dict) else None,
                       **{k: (response or {}).get(k) for k in ['input_tokens', 'output_tokens', 'generation_seconds']}}
                results.append(row)
                details.append({'request_id': request['request_id'], 'checks': checked, 'response': response})
            condition = reqs[0]['condition']
            atomic_json(reporter.folder / f'{condition}_results.json', results)
            atomic_json(reporter.folder / f'{condition}_responses.json', details)
            reporter.data['metrics'][condition] = dict(Counter(r['status'] for r in results))
            reporter.save()
            return results

        if prepare_only:
            return {'report': str(report.folder), 'requests': len(requests), 'inference_executed': False}
        uncached = any((inspect_d if r['condition'] == 'D' else inspect_c_paired)(
            root / 'Outputs/diagnosis' / r['condition'] / 'responses' / (r['request_id'] + '.json'),
            r, record)[0]['status'] == 'not_run'
            for r, record in zip(requests, records + records, strict=True))
        if uncached:
            with report.step('paired_prompt_budget_preflight'):
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config['generator']['model'], revision=config['generator']['revision'])
                counts = [{'case_id': r['case_id'], 'condition': r['condition'],
                           'input_tokens': len(tokenizer.apply_chat_template(r['messages'], tokenize=True, add_generation_prompt=True))}
                          for r in requests]
                report.write_csv('paired_prompt_token_counts.csv', counts)
                if any(r['input_tokens'] > config['generator']['max_input_tokens'] for r in counts):
                    raise ValueError('Paired prompts exceed the common input allowance. No model inference started and no evidence truncated.')
        all_results = []
        for condition, reqs in [('C', cs), ('D', ds)]:
            cache = root / 'Outputs/diagnosis' / condition / 'responses'
            with report.step(f'Qwen_{condition}_paired_inference'):
                _infer(reqs, records, cache, config, report, False,
                       inspect_fn=inspect_d if condition == 'D' else inspect_c_paired,
                       validate_fn=(lambda raw, record: validate_d(raw, record, lookup[record['case_id']]['literature'])) if condition == 'D' else validate_paired,
                       capture_fn=capture)
            all_results.extend(capture(report, None, cache, reqs, records))
        # Scoring labels are opened only after both inference conditions finish.
        label_path = root / 'Datasets/PHM15-diagnosis/validation_pilot_v1/private_validation_labels.jsonl'
        if sha(label_path) != contract['files'][label_path.name]:
            raise ValueError('Scoring label hash mismatch')
        labels = {r['case_id']: r['reference_label'] for r in rows(label_path)}
        for row in all_results:
            row['reference_label'] = labels[row['case_id']]
            row['label_correct'] = row['predicted_class'] == row['reference_label']
        metrics = {condition: {'queries': maximum,
                   'classification_accuracy': sum(r['label_correct'] for r in all_results if r['condition'] == condition) / maximum,
                   'valid_outputs': sum(r['status'] == 'valid' for r in all_results if r['condition'] == condition)} for condition in ['C', 'D']}
        metrics.update(scope='paired validation pilot, not final test', prose_support='requires review',
                       corpus='22 assistant-reviewed passages; no independent human approval')
        atomic_json(report.folder / 'paired_results.json', all_results)
        atomic_json(report.folder / 'paired_metrics.json', metrics)
        report.data['metrics']['paired'] = metrics
        report.data['diagnostic_inference_observed'] = any(r['status'] in {'valid', 'invalid_response'} for r in all_results)
        report.data['remaining_pipeline'] = ['Review paired validation results and explanation support', 'Freeze settings before held-out evaluation']
        return metrics
    except BaseException as exc:
        report.fail(exc)
        raise
    finally:
        report.finish()
        print('Saved report:', report.folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--max-cases', type=int, default=10)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.max_cases, args.prepare_only), indent=2))
