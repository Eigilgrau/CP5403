"""Local class-stratified development sampling; no labels exported to inference."""
import json
from pathlib import Path
import phm_diagnosis as base
import phm_evidence_design as design
import phm_evidence_menu_smoke as smoke
from phm_new_case_review import BUNDLE
from phm_run_reporting import atomic_json,atomic_text


def export(root):
    root=Path(root);source=root/smoke.BUNDLE;out=root/BUNDLE
    if (out/'manifest.json').exists():
        from phm_new_case_review import prepare
        prepare(root);print('Existing fixed sample verified');return
    contract=json.loads((source/'export_contract.json').read_text(encoding='utf-8'))
    for name in ('private_validation_labels.jsonl','private_export_audit.json','validation_evidence.jsonl'):
        if design.sha(source/name)!=contract['files'][name]:raise ValueError('Sampling source hash mismatch')
    labels={r['case_id']:r['reference_label'] for r in base.read_rows(source/'private_validation_labels.jsonl')}
    audit={r['case_id']:r for r in json.loads((source/'private_export_audit.json').read_text(encoding='utf-8'))}
    previous,_=smoke.prepare(root);excluded={q['case_id'] for q in previous}
    selected=[];sites=set()
    for repeat in range(2):
        for label in base.LABELS:
            choices=[i for i,c in labels.items() if c==label and i not in excluded and i not in selected]
            if not choices:raise ValueError('Insufficient new validation cases')
            choice=min(choices,key=lambda i:(audit[i]['site_id'] in sites,base.fingerprint([5403,'new-case-review-v1',i])))
            if audit[choice]['split']!='validation':raise ValueError('Non-validation sample')
            selected.append(choice);sites.add(audit[choice]['site_id'])
    folder=root/design.OUTPUT
    dc=json.loads((folder/'design_contract.json').read_text(encoding='utf-8'))
    if design.sha(folder/'reviewed_candidate_requests.jsonl')!=dc['requests_sha256'] or design.sha(root/'Files/phm_evidence_design.py')!=dc['code_sha256']:
        raise ValueError('Reviewed design changed')
    candidates=base.read_rows(folder/'reviewed_candidate_requests.jsonl')
    lookup={(q['case_id'],q['condition']):q for q in candidates}
    requests=[lookup[i,c] for i in selected for c in ('C','D')]
    evidence_lookup={r['case_id']:r for r in base.read_rows(source/'validation_evidence.jsonl')}
    atomic_text(out/'requests.jsonl',''.join(base.canonical(q)+'\n' for q in requests))
    atomic_text(out/'evidence.jsonl',''.join(base.canonical(evidence_lookup[i])+'\n' for i in selected))
    # This separate audit is only for local sampling provenance, never read by runner.
    review=root/'Outputs/reviews/new_case_review_v1'
    atomic_json(review/'private_selection_audit.json',[{'case_id':i,'reference_label':labels[i],
        'validation_site':audit[i]['site_id'],'reason':'deterministic class quota, prefer distinct sites'} for i in selected])
    names=list(contract['code_sha256'])+['phm_evidence_menu_smoke.py','phm_evidence_design.py',
        'phm_new_case_review.py','phm_new_case_review_export.py','test_phm_new_case_review.py']
    atomic_json(out/'manifest.json',{'scope':'Development evidence review; not independent validation or accuracy evaluation',
        'case_ids':selected,'excluded_previous_smoke_ids':sorted(excluded),'cases':6,'requests':12,
        'selection':'Two per recorded class; seed 5403 SHA256 ranking; prefer distinct validation sites. Labels used locally for sampling only.',
        'distinct_validation_sites':len(sites),'not_used_in_previous_repeated_smoke_checks':True,
        'may_have_appeared_in_earlier_development_comparisons':True,
        'automatic_expansion':False,'accuracy_scoring':False,
        'source_design_contract_sha256':design.sha(folder/'design_contract.json'),
        'public_files':{n:design.sha(out/n) for n in ('requests.jsonl','evidence.jsonl')},
        'code_sha256':{n:design.sha(root/'Files'/n) for n in names}})
    print('Exported six cases from',len(sites),'validation sites; 12 unchanged C/D requests.')


if __name__=='__main__':export(Path(__file__).resolve().parent.parent)
