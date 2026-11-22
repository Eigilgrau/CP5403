"""Audit saved paired runs without a GPU, model loading, or private label files.

Reference labels are read only from the already-scored run. This tool does not
change prompts, caches, corpus approvals, or inference results.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PASSAGE_REVIEW = {
    'lit_5a70b4debe132b2ebac5': {
        'role': 'dataset limitation',
        'assessment': 'Supports incomplete logging; absence of a log is not proof of healthy operation. Does not identify this query fault.'},
    'lit_fb985d1bcc1e4faf4e3b': {
        'role': 'general diagnostic limitation',
        'assessment': 'Supports threshold uncertainty and false alarms. Supplies no threshold applicable to anonymous PHM features.'},
    'lit_caffef7727b4f696804a': {
        'role': 'conditional AHU background',
        'assessment': 'Describes AHU operating modes. Does not diagnose excessive runtime or establish that this PHM plant has this configuration.'},
    'lit_a0fe91e2678aeeaec042': {
        'role': 'external counterexample',
        'assessment': 'Shows an alternative cause for low supply temperature in another installation. Cannot transfer its measurements, causes or rule thresholds to the query.'},
    'lit_0a83217290f5e46ed05c': {
        'role': 'fault-injection procedure',
        'assessment': 'Describes disconnecting sensor leads, without diagnostic symptoms or a usable inference rule. Poor diagnostic retrieval candidate.'},
}


def audit(run):
    names = ['diagnostic_requests.jsonl', 'paired_results.json', 'literature_retrieval.json',
             'C_responses.json', 'D_responses.json']
    content = {name: (run / name).read_bytes() for name in names}
    requests = [json.loads(line) for line in content[names[0]].decode('utf-8').splitlines() if line.strip()]
    results = json.loads(content['paired_results.json'])
    records = {(r['case_id'], r['condition']): r for r in results}
    queries = {}
    usage = Counter()
    passages = {}
    for req in requests:
        payload = json.loads(req['messages'][1]['content'])
        case_id = req['case_id']
        if req['condition'] == 'C':
            queries[case_id] = {
                'score_top_class': payload['score_order_descending'][0],
                'historical_outcomes_in_distance_order': [h['recorded_outcome'] for h in payload['historical_cases']],
                'reference_label': records[case_id, 'C']['reference_label'],
            }
        for passage in req.get('literature', []):
            pid = passage['chunk_id']
            usage[pid] += 1
            passages[pid] = {'title': passage['title'], 'text': passage['text'],
                            **PASSAGE_REVIEW.get(pid, {'assessment': 'Not reviewed'})}
    for condition in ('C', 'D'):
        for output in json.loads(content[f'{condition}_responses.json']):
            response, check = output['response'], output['checks']
            parsed = check.get('parsed') or {}
            queries[response['case_id']][condition] = {
                'status': check['status'], 'errors': check.get('errors', []),
                'predicted_class': parsed.get('predicted_class'),
                'hit_output_limit': response['generation_hit_limit'],
                'citations': parsed.get('literature_citations'),
                'explanation': parsed.get('explanation'), 'limitation': parsed.get('limitation'),
            }
    return {
        'run_id': run.name,
        'review_scope': 'Assistant offline review, not independent human assessment or automatic entailment scoring',
        'source_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in content.items()},
        'retrieved_distinct_chunks': len(usage), 'chunk_use_counts': dict(usage),
        'passage_assessments': passages, 'queries': queries,
        'score_top_class_correct': sum(q['score_top_class'] == q['reference_label'] for q in queries.values()),
        'score_comparison_scope': 'Descriptive existing-score check, not an added ablation condition or a final-test result',
        'prompt_findings': [
            'Existing V5 prompt already forbids physical channel mappings and requests short prose; failures persist despite those instructions.',
            'Retrieval uses only the top regression class template; cases sharing that class receive identical literature, regardless of their individual features.',
            'Empty citation lists may be appropriate when passages do not apply. Forcing citations would not establish support.',
            'Do not force predictions to match regression rankings: that changes the role of the LLM and requires an explicit methodological decision.',
        ],
        'candidate_revision_not_applied': {
            'retrieval': 'Review passage roles before retrieval. Exclude procedure-only snippets from diagnostic selection; retain externally sourced examples only as conditional context.',
            'candidate_passages': {
                'lit_21942c3d291b69d3e4d3': 'Dataset anonymity and incomplete logs: suitable limitation evidence across queries.',
                'lit_867a9fc28449e5289819': 'Scheduling inconsistencies: more relevant runtime context than generic AHU modes, but not a PHM fault mapping.',
            },
            'generation': 'Keep classification and evidence selections identical in design between C/D. Consider a separate short explanation call only if its extra inference cost is budgeted and both conditions use it.',
            'evaluation': 'Freeze any revision before held-out testing; count incomplete outputs as failures and assess citation entailment separately from structural validity.',
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run)
    destination = args.run / 'offline_retrieval_prompt_audit.json'
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Saved: {destination}')
    print(f"Distinct retrieved passages: {report['retrieved_distinct_chunks']}")
    print(f"Existing score top-class agreement: {report['score_top_class_correct']}/{len(report['queries'])}")
