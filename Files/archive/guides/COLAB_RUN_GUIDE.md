# Colab: run condition C first

For the complete setup, all five executable code cells, result inspection, resume instructions and troubleshooting, use [RAG_LLM_SETUP_GUIDE.md](RAG_LLM_SETUP_GUIDE.md). This file remains the shorter checklist.

C uses the PHM query features, saved logistic-regression scores and three retrieved historical training cases. D will add approved literature to exactly that evidence later. Charan is excluded from active processing, including old cached literature jobs. Keep the older files as history; do not run the literature notebook now.

## Your next run

1. Let Google Drive for desktop finish syncing the project.
2. Open `Files/condition_C_diagnosis_colab.ipynb` in Google Colab. Use a fresh runtime if the literature model is still loaded.
3. Select **Runtime > Change runtime type > T4 GPU**.
4. Run the notebook cells in order. Its project path is already set to your working mount:

   `/content/drive/Othercomputers/My Laptop (1)/CP5403`

   That longer path belongs only in `PROJECT_ROOT`. Keep the Drive mount call fixed at `drive.mount('/content/drive')`; the mountpoint must not be the project folder. Spaces in `PROJECT_ROOT` are allowed.

5. Leave `MAX_CASES = 3`, `RUN_C = True` and `RETRY_ERRORS = False` for this first pilot.
6. After the last cell saves the report and Drive sync finishes, tell Codex: **C pilot finished—check the latest report.** If a cell fails, stop there and request a report review; the failure is saved after setup has established the project path.

The generator remains **Qwen/Qwen3-4B-Instruct-2507**, pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554`, using NF4 4-bit loading, float16 compute and greedy generation. No Hugging Face approval or token is required for this public model. The first inference run downloads its weights. Existing Colab CUDA-enabled PyTorch is retained. If installation asks for a runtime restart, restart and rerun from setup.

## Files that must have synced

```text
CP5403/
  Files/
    condition_C_diagnosis_colab.ipynb
    phm_diagnosis.py
    phm_diagnosis_qwen.py
    phm_run_reporting.py
    experiment_config.json
    requirements-qwen-colab.txt
    test_phm_diagnosis.py
  Datasets/
    PHM15-diagnosis/
      validation_pilot_v1/
        export_contract.json
        feature_dictionary.json
        validation_evidence.jsonl
        private_validation_labels.jsonl
        private_export_audit.json
  Outputs/                     # created/updated by the reporting code
```

The fitted numerical model was loaded and the evidence exported locally using its original environment. Colab does not unpickle it, refit logistic regression or rerun preprocessing. Keep the project's existing numerical artifacts for provenance. The export contains ten validation queries, sampled across classes with preference for different sites; the first three contain one case per class. Query labels are in a separate scoring file. Historical training-case labels are legitimate input evidence.

Every prompt preserves the 31 selected feature descriptors and the three historical cases in their original order. A shared feature-name list avoids repeating long names in every record. Displayed values have six significant digits; the full-precision JSON export remains unchanged. Prompt length is checked with the pinned Qwen tokenizer, and over-budget prompts fail instead of silently losing evidence. The budgets remain provisional until validation, including the later D context, is reviewed.

## What gets saved for review

| Output | Purpose |
|---|---|
| `Outputs/LATEST_RUN.json` and `.md` | Locate the most recently updated report. Keep its run ID if reviewing a particular run. |
| `Outputs/runs/<run-id>/diagnostic_requests.jsonl` | Exact model requests, generation settings and evidence fingerprints; no query reference labels. |
| `diagnostic_responses.json` | Raw cached responses and freshly checked JSON/citation/value validity. |
| `diagnostic_status.csv`, `diagnostic_metrics.json` | Per-query outcomes and separate-label development scoring after inference. |
| `prompt_token_counts.csv`, `qwen_runtime.json` | Input lengths, hardware, package versions and model-loading cost. Each response also records output tokens and generation time. |
| `run_report.json`, `console.log`, `events.jsonl`, `traceback.txt` when present | Configuration, progress, errors and artifact fingerprints. |
| `Outputs/diagnosis/C/responses/` | Resumable response cache. Cache identity includes the exact prompt, evidence and generation contract. |

Automatic checks verify output structure, allowed classes, historical case IDs, comparison features and quoted values. They do not establish whether all explanatory prose is supported, and are neither human review nor an independent LLM judge. Invalid outputs and abstentions remain visible. Pilot accuracy is only a development check on a small, deliberately balanced sample, not a generalisation result.

## Resume and interpret status

Rerun from setup to create a new report. With the same settings, completed responses are reused. After we review the first three cases, changing `MAX_CASES` to `10` extends the same fixed pilot. Do not change settings or retry failures merely to improve scores. If a corrective retry is needed, set `RETRY_ERRORS=True` explicitly; previous attempts are archived.

- **running** means last known state, not proof the runtime is still alive. A hard disconnect cannot write a final error.
- **failed/interrupted** means an observed error or interruption. Inspect the saved traceback.
- **completed** means the named stage completed; it does not mean the paired final experiment is complete.
- **completed_with_issues** means invalid responses, errors or selected queries without responses remain.

If Drive cannot mount or the project folder cannot be found, no synced report can be written yet; provide that Colab error. After setup, saved reports are the primary debugging input. Allow sync to finish before local review and avoid concurrent edits of the same files.

## After C works

Review the C outputs, then extend to the ten validation cases. Next, select permitted literature, verify diagnostic applicability, build the literature index and add D to the same evidence and generation framework. Use paired C/D validation to freeze the prompt, context budget, retrieval and final sample size. Only then run the reserved test evaluation. D is deliberately blocked in the current runner so an empty-literature run cannot be misreported as D.
