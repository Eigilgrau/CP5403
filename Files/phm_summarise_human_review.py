"""Aggregate actual completed human ratings; never generate or fill ratings."""
import argparse
from collections import Counter
from pathlib import Path
from phm_final_evaluation_pack import read, OUT, sha
from phm_run_reporting import atomic_json, atomic_text


def summarise(root, ratings_path):
    folder=Path(root)/OUT/'human_review'
    packet=read(folder/'review_packet.json');key=read(folder/'analysis_only_sample_key.json')
    data=read(ratings_path);dims=packet['protocol']['dimensions']
    if data.get('packet_id')!=packet['packet_id']:raise ValueError('Ratings belong to a different packet.')
    if data.get('status')!='completed' or not isinstance(data.get('reviewer'),str) or not data['reviewer'].strip():raise ValueError('Complete the review and supply a reviewer identifier first.')
    ratings=data.get('ratings',{})
    if set(ratings)!={x['item_id'] for x in packet['items']}:raise ValueError('Expected exactly the 20 selected items.')
    for item,r in ratings.items():
        if not isinstance(r.get('note'),str) or not r['note'].strip():raise ValueError(f'{item}: missing evidence note.')
        if any(r.get(d) not in ('0','1','2','U','NA') for d in dims):raise ValueError(f'{item}: incomplete/invalid ratings.')
    result={'packet_id':packet['packet_id'],'reviewer':data['reviewer'],'status':'completed',
        'scope':'Single developer-reviewer; post-hoc descriptive audit of ten paired cases; not independent expert validation.',
        'ratings_file_sha256':sha(ratings_path),'conditions':{}}
    for c in ('C','D'):
        selected=[ratings[k['item_id']] for k in key if k['condition']==c]
        result['conditions'][c]={}
        for d in dims:
            counts=Counter(r[d] for r in selected);numeric=[int(r[d]) for r in selected if r[d] in ('0','1','2')]
            result['conditions'][c][d]={'counts':{v:counts[v] for v in ('0','1','2','U','NA')},'assessable_n':len(numeric),'total_n':len(selected),'mean_ordinal_score':sum(numeric)/len(numeric) if numeric else None}
    atomic_json(folder/'human_review_summary.json',result)
    atomic_text(folder/'human_review_summary.txt','HUMAN REVIEW — DESCRIPTIVE RESULTS\n'+result['scope']+'\nCounts are primary; ordinal means exclude U/NA. No inter-rater reliability or overall grounding pass rate is inferred.\n\n'+__import__('json').dumps(result['conditions'],indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('ratings',type=Path);a=p.parse_args()
    summarise(Path(__file__).resolve().parent.parent,a.ratings)
    print('Saved human_review_summary.json and human_review_summary.txt alongside the review form.')
