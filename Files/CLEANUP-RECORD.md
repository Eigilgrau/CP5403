# Reproducibility Cleanup Record

Date: November 27, 2026. Prepared with Codex assistance.

## Changes

- Archived 26 files: older notebook launchers and four non-identical sync-conflict copies. Each relocation preserves bytes and is recorded in `archive/relocation_manifest.json`.
- Left four primary notebooks in `Files`: preprocessing, EDA/case construction, numerical representation/retrieval and frozen held-out inference.
- Preserved all original Python modules and tests at their runtime paths. Static analysis identifies 16 modules in the final pipeline's transitive dependency closure; several retain historical names such as `pilot`.
- Added reviewer READMEs, a reproduction guide, a code map and a complete inventory of 70 run directories.
- Replaced two obsolete C-first setup guides with current navigation; their original text is saved under `archive/guides`. Marked the original project-design guide as historical.
- Corrected a live notebook link in `version_history/v0.4.0_literature_ingestion.md`. Historical archive contents were not rewritten.
- Added `review_reproducibility.py`, a no-GPU entry point to the existing saved-result audit and optional existing tests.
- Added local environment/IDE/OS ignore patterns. These do not unstage already staged IDE files.

## Verification Performed

`cleanup_validation.json` contains the executed command's result. The saved-result audit passed: C accuracy .52 / macro-F1 .31794871794871793; D accuracy .51 / macro-F1 .3134070230844424. All 13 frozen code/configuration hashes, six export-file hashes and 400 raw-response hashes matched.

The existing final-workflow suites passed: seven tests in `test_phm_two_stage.py` and five in `test_phm_heldout.py`. These cover request pairing, fixed decisions, schema constraints, evidence lookup, cache/resume behaviour, label separation, incomplete-sample rejection and scoring/bootstrap conventions. Their synthetic run directories were created in temporary test folders; no Qwen responses were generated.

Every original top-level Python, JSON, requirements-text and notebook file matched `archive/pre_cleanup_hashes.json` at its original or recorded archive location. The original named DOCX also matched. No runtime module was refactored, no frozen score was replaced, and no run directory or inference cache was moved or deleted.

## Limits and Packaging

This cleanup validates the saved-result route and targeted tests; it does not establish a fresh full-data rebuild, new GPU replication or independent semantic validation. Historical notebooks can reference old fingerprints not matched by the present frozen source. Preserve that history rather than bypassing its checks.

The local Git repository root remains `Files`. Existing staged changes and Git metadata were not relocated, committed or pushed. A checkout containing only that directory cannot by itself supply the parent report, audit script, datasets and results. Follow `REPRODUCIBILITY.md` when assembling the full reviewer bundle. Remote repository and Drive accessibility were not tested.

## Reversal

The manifest records each original and archived path plus SHA-256. A relocation can be reversed if its original destination is absent and the archived bytes match the recorded hash. The original root documentation is preserved under `archive/guides`. Do not overwrite later edits when reversing this organisation.
