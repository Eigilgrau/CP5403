"""Build a source-verifiable, passage-approved MiniLM literature index on CPU.

This index is a preparation artifact, not a completed Condition D diagnosis run.
Only spans explicitly listed in literature_passage_review.json are eligible.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import urllib.request

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from phm_literature import EMBEDDING, EMBEDDING_REVISION, EmbeddingTokenizer, make_units, baseline_groups, joined


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


class Encoder:
    def __init__(self, assets):
        assets = Path(assets)
        self.counter = EmbeddingTokenizer(assets)
        model = assets / 'minilm.onnx'
        if not model.exists():
            temporary = model.with_suffix('.download')
            urllib.request.urlretrieve(
                f'https://huggingface.co/{EMBEDDING}/resolve/{EMBEDDING_REVISION}/onnx/model.onnx', temporary)
            temporary.replace(model)
        self.model_hash = sha(model)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(model), sess_options=options, providers=['CPUExecutionProvider'])
        self.tokenizer = Tokenizer.from_file(str(assets / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')

    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), 16):
            batch = self.tokenizer.encode_batch(texts[start:start + 16])
            if any(len(x.ids) > 256 for x in batch):
                raise ValueError('Embedding input exceeds 256 tokens; truncation prohibited.')
            inputs = {'input_ids': np.array([x.ids for x in batch], dtype=np.int64),
                      'attention_mask': np.array([x.attention_mask for x in batch], dtype=np.int64),
                      'token_type_ids': np.array([x.type_ids for x in batch], dtype=np.int64)}
            hidden = self.session.run(None, {x.name: inputs[x.name] for x in self.session.get_inputs()})[0]
            mask = inputs['attention_mask'][..., None]
            pooled = (hidden * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1)
            pooled /= np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
            vectors.append(pooled.astype(np.float32))
        return np.concatenate(vectors)


def build(prepared, sources, review_path, output):
    prepared, sources, output = map(Path, (prepared, sources, output))
    output.mkdir(parents=True, exist_ok=True)
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    inventory = {r['document_id']: r for r in rows(prepared / 'inventory.jsonl')}
    paragraphs = {}
    for doc in inventory.values():
        if sha(sources / doc['filename']) != doc['source_sha256']:
            raise ValueError(f"Source changed: {doc['filename']}")
        paragraphs.update({r['paragraph_id']: r for r in rows(prepared / doc['document_id'] / 'paragraphs.jsonl')})
    encoder = Encoder(output / 'model_assets')
    chunks, decisions, seen = [], [], set()
    for selection in review['selections']:
        parts, spans = [], []
        for item in selection['spans']:
            source = paragraphs[item['paragraph_id']]
            text = source['text']
            begin = text.index(item['start']) if item.get('start') else 0
            end = text.index(item['end'], begin) if item.get('end') else len(text)
            if begin >= end:
                raise ValueError('Invalid source span')
            parts.append(text[begin:end])
            spans.append({'paragraph_id': item['paragraph_id'], 'page': source['page'],
                          'start': begin, 'end': end, 'bbox': source['bbox']})
        text = ' '.join(parts)
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ufffd]', text):
            raise ValueError(f"Unreviewed extraction characters: {selection['id']}")
        doc_ids = {paragraphs[s['paragraph_id']]['document_id'] for s in spans}
        if len(doc_ids) != 1:
            raise ValueError('A passage may not combine documents')
        document = inventory[next(iter(doc_ids))]
        record = {'paragraph_id': selection['id'], 'raw_text': text, 'text': text,
                  'document_id': document['document_id'], 'page': spans[0]['page'],
                  'section': selection['topic'], 'bbox': spans[0]['bbox'], 'flags': []}
        units = make_units([record], encoder.counter)
        groups = baseline_groups(units, encoder.counter)
        assert joined([unit for group in groups for unit in group]) == text
        for group in groups:
            body = joined(group)
            key = hashlib.sha256(body.casefold().encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            chunks.append({'chunk_id': 'lit_' + key[:20], 'text': body,
                           'document_id': document['document_id'], 'filename': document['filename'],
                           'title': document['policy']['title'], 'year': document['policy']['year'],
                           'source_sha256': document['source_sha256'], 'passage_id': selection['id'],
                           'topic': selection['topic'], 'scope': selection['scope'],
                           'applicability': selection['applicability'],
                           'passage_source_spans': spans,
                           'passage_start': group[0]['start'], 'passage_end': group[-1]['end'],
                           'page_start': min(s['page'] for s in spans),
                           'page_end': max(s['page'] for s in spans),
                           'token_count': encoder.counter.count(body),
                           'diagnostic_approved': True, 'reviewer': 'assistant',
                           'preparation_method': 'reviewed_source_spans_sentence_boundaries'})
        decisions.append({'passage_id': selection['id'], 'source_spans': spans,
                          'topic': selection['topic'], 'characters': len(text)})
    if not chunks:
        raise ValueError('No approved chunks')
    (output / 'approved_chunks.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in chunks), encoding='utf-8')
    embeddings = encoder.encode([c['text'] for c in chunks])
    np.save(output / 'embeddings.npy', embeddings, allow_pickle=False)
    manifest = {'version': 'literature-index-v1', 'embedding_model': EMBEDDING,
                'embedding_revision': EMBEDDING_REVISION, 'onnx_sha256': encoder.model_hash,
                'pooling': 'attention-mask mean pooling then L2 normalization',
                'retrieval': 'cosine similarity, at most one passage per document in top-k',
                'review_sha256': sha(review_path), 'chunks_sha256': sha(output / 'approved_chunks.jsonl'),
                'embeddings_sha256': sha(output / 'embeddings.npy'), 'chunk_count': len(chunks),
                'passage_count': len(decisions), 'document_count': len({c['document_id'] for c in chunks}),
                'dimensions': int(embeddings.shape[1]), 'reviewer': 'assistant, not independent human review',
                'scope': 'Conservative initial corpus; original candidate corpus remains unapproved.',
                'diagnosis_integration_complete': True, 'GPU_diagnosis_executed': False}
    dump(output / 'index_manifest.json', manifest)
    dump(output / 'passage_audit.json', decisions)
    return manifest


def retrieve(index, query, k=2):
    index = Path(index)
    contract = json.loads((index / 'index_manifest.json').read_text(encoding='utf-8'))
    for name, field in [('approved_chunks.jsonl', 'chunks_sha256'), ('embeddings.npy', 'embeddings_sha256')]:
        if sha(index / name) != contract[field]:
            raise ValueError('Index integrity failed')
    encoder = Encoder(index / 'model_assets')
    if encoder.model_hash != contract['onnx_sha256']:
        raise ValueError('Embedding model integrity failed')
    chunks = rows(index / 'approved_chunks.jsonl')
    scores = np.load(index / 'embeddings.npy', allow_pickle=False) @ encoder.encode([query])[0]
    results, documents = [], set()
    for i in np.argsort(-scores, kind='stable'):
        chunk = chunks[int(i)]
        if chunk['document_id'] in documents:
            continue
        results.append({**chunk, 'retrieval_score': float(scores[i])})
        documents.add(chunk['document_id'])
        if len(results) == k:
            break
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, default=Path('Datasets/PHM15-literature-D-v1'))
    parser.add_argument('--sources', type=Path, default=Path('Datasets/Literature-RAG'))
    parser.add_argument('--review', type=Path, default=Path('Files/literature_passage_review.json'))
    parser.add_argument('--output', type=Path, default=Path('Datasets/PHM15-literature-D-approved-v1'))
    args = parser.parse_args()
    print(json.dumps(build(args.prepared, args.sources, args.review, args.output), indent=2))
