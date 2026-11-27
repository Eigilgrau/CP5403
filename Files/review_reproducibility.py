"""Review saved results without inference or modifying experimental artifacts.

Default: verify the frozen results using the documentation audit. Optional
--run-tests also runs the two existing final-workflow unittest modules, which
use fixtures/local embeddings and do not generate Qwen responses.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-tests', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    audit_path = root / 'Report-Documentation/recheck_saved_results.py'
    required = [audit_path, root / 'Files/phm_two_stage.py',
                root / 'Datasets/PHM15-diagnosis/heldout_test_v1/export_contract.json',
                root / 'Outputs/runs/20261122T103151_paired_CD_two_stage_heldout_13ba4e1a/paired_results.json']
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print(json.dumps({'status': 'missing_artifacts', 'root': str(root), 'missing': missing,
                          'help': 'Restore the CP5403 root layout described in Files/REPRODUCIBILITY.md. A Files-only checkout is insufficient.'}, indent=2))
        return 2
    spec = importlib.util.spec_from_file_location('phm_documentation_audit', audit_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        result = module.audit(root)
    except (AssertionError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({'status': 'audit_failed', 'error': str(exc), 'error_type': type(exc).__name__}, indent=2))
        return 1
    summary = {'status': 'saved_result_audit_passed', 'root': str(root),
               'classification': {c: {k: m[k] for k in ('n', 'accuracy', 'macro_f1')} for c, m in result['classification'].items()},
               'frozen_code_hashes': len(result['frozen_code_hash_matches']),
               'export_file_hashes': len(result['bundle_file_hash_matches']),
               'raw_response_hashes': result['raw_response_hashes_verified'],
               'scope': result['scope'], 'tests': []}
    if args.run_tests:
        for pattern in ('test_phm_two_stage.py', 'test_phm_heldout.py'):
            proc = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                                   '-s', str(root / 'Files'), '-p', pattern, '-v'],
                                  cwd=root, capture_output=True, text=True, errors='replace')
            summary['tests'].append({'module': pattern, 'exit_code': proc.returncode,
                                     'output': proc.stdout + proc.stderr})
        if any(t['exit_code'] for t in summary['tests']):
            summary['status'] = 'saved_result_audit_passed_but_tests_failed'
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0 if all(t['exit_code'] == 0 for t in summary['tests']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
