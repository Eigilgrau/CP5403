"""Generation-time JSON constraints; semantic support still needs validation."""
from phm_diagnosis import LABELS, ABSTAIN, OUTPUT_KEYS, comparison_claims, display_number

FORMAT_ENFORCER_VERSION = '0.11.3'


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def exact(value):
    if value is None:
        return {'type': 'null'}
    return {'enum': [value]}


def output_schema(record, literature=()):
    names = list(record['feature_values'])
    comparisons, claims = [], []
    for case in record['historical_cases']:
        fields = {x['comparison']: x['feature'] for x in case['comparison']}
        comparisons.append(object_schema({
            'case_id': exact(case['case_id']),
            'similarity_feature': exact(fields['closest_numeric_feature']),
            'difference_feature': exact(fields['largest_standardised_difference'])}))
        claims.extend(object_schema({k: exact(v) for k, v in claim.items()})
                      for claim in comparison_claims(record, case))
    features = [object_schema({'feature': exact(name), 'value': exact(display_number(value))})
                for name, value in record['feature_values'].items() if value is not None]
    citation = {'type': 'array', 'maxItems': len(literature),
                'items': {'type': 'string', **({'enum': [p['chunk_id'] for p in literature]} if literature else {})}}
    result = object_schema({
        'predicted_class': {'type': 'string', 'enum': [*LABELS, ABSTAIN]},
        'feature_evidence': {'type': 'array', 'maxItems': 2 if features else 0,
                             'items': {'anyOf': features} if features else {'type': 'null'}},
        'case_comparison': {'anyOf': [*comparisons, {'type': 'null'}]},
        'comparison_claims': {'type': 'array', 'maxItems': 2, 'items': {'anyOf': claims}},
        'literature_citations': citation,
        'explanation': {'type': 'string', 'minLength': 1},
        'limitation': {'type': 'string', 'minLength': 1}})
    assert set(result['required']) == OUTPUT_KEYS
    return result


def json_parser(schema):
    import importlib.metadata
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.jsonschemaparser import ListParsingState, ForceStopParser
    if importlib.metadata.version('lm-format-enforcer') != FORMAT_ENFORCER_VERSION:
        raise ValueError('Install the pinned lm-format-enforcer version before inference.')
    # 0.11.3 permits the first member even for maxItems=0. Restrict the
    # newly pushed member parser to immediate termination for empty arrays.
    if not getattr(ListParsingState, '_phm_empty_array_fix', False):
        original = ListParsingState.add_character
        def add_character(self, character):
            result = original(self, character)
            if character == '[' and self.max_items == 0:
                self.root.context.active_parser.object_stack[-1] = ForceStopParser(allow_whitespace=True)
            return result
        ListParsingState.add_character = add_character
        ListParsingState._phm_empty_array_fix = True
    return JsonSchemaParser(schema)


def prefix_constraint(tokenizer, schema):
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    return build_transformers_prefix_allowed_tokens_fn(tokenizer, json_parser(schema))
