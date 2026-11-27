# PHM 2015 RAG + Qwen: complete setup and run instructions

Updated: **21 November 2026**. Written for the files currently in your CP5403 project.

**Start with section 1. You can run the prepared case-based RAG pipeline without waiting for another instruction from Codex.** The remaining sections explain the setup, provide the exact code cells, and show how to recover from errors.

Jump to: [code cells](#6-complete-code-cells-for-a-blank-colab-notebook) · [results](#7-inspect-the-results) · [resume](#8-expand-resume-or-rerun) · [troubleshooting](#10-troubleshooting) · [remaining D work](#11-what-remains-for-literature-based-condition-d).

## 1. The shortest route to a working run

1. Let Google Drive for desktop finish syncing the CP5403 folder.
2. Open **`Files/condition_C_diagnosis_colab.ipynb`** in Google Colab.
3. Select **Runtime > Change runtime type > T4 GPU**. If a previous notebook still has Qwen loaded, use a fresh runtime.
4. In the first code cell, choose the size of this run:

   ```python
   MAX_CASES = 3       # Short initial check.
   RUN_C = True
   RETRY_ERRORS = False
   ```

   **To run all ten prepared validation cases in one session, set `MAX_CASES = 10` instead.** This is still a validation pilot. You do not need a second notebook, another export, or permission to run the ten cases.

5. Run the cells from top to bottom, or use **Runtime > Run all**. Respond to the Google Drive authorization prompt if it appears. If a cell fails, follow section 10 before continuing.
6. The final cell prints the report location and status. Let Drive sync the saved outputs back to your computer.
7. Inspect the output using section 7. If you want Codex to review it, send the short message in section 12.

The existing notebook already contains the setup and inference code. **You do not need to paste the alternative cells in section 6 into it.** Those cells are provided only if you prefer to start a blank Colab notebook.

### What this runs today

| Part | Current state | Your action |
|---|---|---|
| PHM preprocessing, fitted logistic regression and historical case bank | Existing local artifacts | Reuse them. |
| Portable validation evidence | Exported: ten queries across ten validation sites; 31 descriptors and three historical cases per query | Let the files sync. |
| **Condition C: case-based RAG + Qwen** | Runner and notebook implemented; local checks passed; actual diagnostic inference still needs the Colab run | Run the notebook above. |
| Literature preparation utilities | Present, but the previously used Charan source is excluded | Leave the literature notebook deferred. |
| **Condition D: C plus approved literature retrieval** | Literature index, D prompt integration and paired evaluation are not implemented yet | See section 11 for the remaining work. |
| Final held-out C/D evaluation | Not run; settings are not frozen | Complete paired validation before using the test set. |

**A setup instruction cannot enable code that has not yet been implemented.** This guide gets the existing C pipeline running end to end. It does not claim that changing a setting will activate the unfinished D pipeline.

## 2. What “RAG + LLM” means in this setup

Condition C uses **retrieval of historical cases**. The local case system searches the existing training bank, retrieves three comparable cases from distinct training sites, and includes their recorded outcomes and measurements in each exported query's evidence. Colab sends that evidence to Qwen to generate a classification and a short justification.

Retrieval for these ten queries has already happened during the local export. The Colab notebook consumes the saved results; it does not rerun the numerical nearest-neighbour search each time. This keeps the retrieved evidence fixed and avoids transferring a version-sensitive fitted Python model into Colab.

```text
Prepared PHM measurements
        |
        v
Query feature summary + saved logistic-regression scores
        |
        +--- numerical retrieval from the training case bank
        |                  |
        |                  v
        |          three historical cases
        |                  |
        +------------------+
                 |
          portable JSON evidence
                 |
          Qwen in Google Colab
                 |
       classification + cited features
       + historical-case comparison
       + explanation + limitation
                 |
          saved response and checks
                 |
       separate query label used for scoring
```

This is a case-based retrieval-augmented generation workflow. A literature vector index is an additional retrieval source for D. It is not required to run C.

You do not need to fine-tune Qwen, set up a paid inference API, deploy a server, create a vector database account, or add LangChain for this C pilot. Qwen's weights remain fixed; the evidence enters through the prompt.

## 3. Your two project locations

These paths refer to the same project through different computers and filesystems.

| Location                                      | Project root                                         |
| --------------------------------------------- | ---------------------------------------------------- |
| Windows computer, where Codex edits the files | `C:\Users\ges2j\OneDrive\Desktop\Assessments\CP5403` |
| Mounted Google Drive in your Colab session    | `/content/drive/Othercomputers/My Laptop (1)/CP5403` |

The notebook already uses your verified Colab path. Keep the spaces, parentheses and `Othercomputers` spelling exactly as shown. `PROJECT_ROOT` must point to the folder containing `Files`, `Datasets` and `Outputs`, not to the `Files` subfolder.

**These two code paths serve different purposes:** `drive.mount('/content/drive')` attaches Google Drive at a fixed mountpoint. `PROJECT_ROOT = Path('/content/drive/Othercomputers/My Laptop (1)/CP5403')` selects your project inside that mounted Drive. Put the longer path only in `PROJECT_ROOT`. Colab rejects spaces in the mountpoint, while spaces in the project-folder path are fine. [Colab's Drive mount implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)

Google Drive synchronization transfers files. It does not transfer the running Python session or GPU state. Save changes locally, allow sync, and then run them in Colab. After inference, allow sync in the other direction before asking Codex to inspect the outputs.

### Required files

```text
CP5403/
├── RAG_LLM_SETUP_GUIDE.md
├── Files/
│   ├── condition_C_diagnosis_colab.ipynb
│   ├── phm_diagnosis.py
│   ├── phm_diagnosis_qwen.py
│   ├── phm_run_reporting.py
│   ├── experiment_config.json
│   ├── requirements-qwen-colab.txt
│   └── test_phm_diagnosis.py
├── Datasets/
│   └── PHM15-diagnosis/
│       └── validation_pilot_v1/
│           ├── export_contract.json
│           ├── feature_dictionary.json
│           ├── validation_evidence.jsonl
│           ├── private_validation_labels.jsonl
│           └── private_export_audit.json
└── Outputs/
    ├── runs/                         # report folder for each run
    ├── diagnosis/C/responses/         # persistent response cache
    ├── LATEST_RUN.json
    └── LATEST_RUN.md
```

The output folders are created as needed. The export files already exist locally. The label file is required for the scoring step, but its contents are not sent to Qwen. The private audit records provenance and local evidence-construction timings.

Keep your original datasets and numerical artifacts in the project for reproducibility. The C notebook does not need to load them. It also does not need the Windows `.venv`, literature PDFs, or old literature chunks.

## 4. Google Colab, Hugging Face and package setup

### Open the notebook

Open [Google Colab](https://colab.research.google.com/) with the Google account that has access to your synced project. You can open the notebook from Drive or use **File > Upload notebook** to select the local `.ipynb`. Uploading the notebook does not upload the whole project; the code still reads the project through the mounted Drive path. Colab supports importing existing notebooks and running them in its hosted environment. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)

Use a GPU runtime. T4 is the intended starting hardware and was available in your earlier run. Free Colab GPU allocation and session duration are variable; a GPU is not guaranteed on every connection. The response cache is therefore stored on Drive. [Colab resource limits](https://research.google.com/colaboratory/faq.html#resource-limits)

### Which Qwen model?

| Setting | Current value |
|---|---|
| Repository | `Qwen/Qwen3-4B-Instruct-2507` |
| Pinned revision | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| Loading | NF4 4-bit, double quantization, float16 computation |
| Generation | Greedy: `do_sample=False` |
| Seed | `5403` |
| Input budget | `4096` tokens |
| Output limit | `512` new tokens |
| Historical cases per query | `3` |

These are the settings in `Files/experiment_config.json`, not new settings to type manually. The selected Qwen release is an instruction model with non-thinking output. [Official Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

**No Hugging Face approval request or access token is needed for this model.** Its repository metadata was checked on 21 November 2026 and reports `private=false` and `gated=false`. The runner downloads it using `from_pretrained` without a login cell. A missing-token notice alone is not a model-loading failure. [Model repository metadata](https://huggingface.co/api/models/Qwen/Qwen3-4B-Instruct-2507)

There is no mandatory “connect Hugging Face” step. Google authorization mounts Drive; downloading a public model from Hugging Face is a separate operation. If an actual authentication or download error occurs, use section 10 rather than adding credentials to the notebook.

### Packages installed by the notebook

`Files/requirements-qwen-colab.txt` contains:

```text
transformers==4.57.1
accelerate==1.11.0
bitsandbytes==0.48.1
tokenizers==0.22.2
huggingface-hub==0.36.0
```

The installation cell runs this requirements file using Colab's Python interpreter. It keeps Colab's existing CUDA-enabled PyTorch. You do not need to install the local numerical environment or copy its `site-packages` into Colab.

Your earlier Qwen chunking run confirmed that the model-loading approach worked on a T4. The new diagnostic prompts still require this C inference run; the earlier chunking result is not a diagnostic evaluation result.

## 5. What the notebook does automatically

| Code cell | What it does | What you should see |
|---|---|---|
| 1. Setup | Mounts Drive, selects the project, loads the C/D configuration, creates a report and sets the run size | `Condition: C only`, your project path and a new run-report path |
| 2. Dependencies | Installs the pinned inference libraries and records package versions | Successful installation or “requirement already satisfied” messages |
| 3. Evidence checks | Verifies the portable export and runs the diagnosis software checks | Selected query count, `Native features: 31`, and passing tests |
| 4. Inference | Downloads/loads Qwen as needed, generates responses sequentially, saves each response, checks it and scores against separate labels | A status for each case and pilot metrics |
| 5. Finish | Finalizes the report and prints saved locations | `completed` or `completed_with_issues`, with the report path |

The first run can spend substantial time downloading and loading the model. Look for download progress and the saved events before treating waiting as an error. A later run in the same runtime can reuse downloaded model files; completed response caches are also reusable across sessions when the request is unchanged.

The current local verification passed **29 software tests** across the diagnosis, reporting and literature-exclusion modules. The C notebook runs the smaller diagnosis-only suite: **11 tests** at this revision. These are software checks, not measurements of Qwen's diagnostic accuracy.

All ten current prompts were measured with the pinned Qwen tokenizer: **2,092–2,130 input tokens**, below the 4,096-token limit. All 31 selected descriptors and all three retrieved cases are retained. Values are displayed to six significant digits; full-precision evidence remains in the export. The runner rejects an over-budget prompt instead of truncating its evidence.

## 6. Complete code cells for a blank Colab notebook

**Use this section only as an alternative to the provided notebook.** Create five code cells and run them in this order. The code below matches the five code cells in `Files/condition_C_diagnosis_colab.ipynb` at this revision.

### Cell 1 — Mount Drive and start the report

Change only `MAX_CASES` if you want all ten cases now. The project path is already your working path.

```python
from pathlib import Path
import os
import sys
from google.colab import drive

# Project folder: spaces are allowed here. Change only this path if the folder moves.
PROJECT_ROOT = Path('/content/drive/Othercomputers/My Laptop (1)/CP5403')

# Fixed Drive mountpoint: never replace this with the project-folder path.
# Reuse an already accessible project instead of mounting Drive again.
if not (PROJECT_ROOT / 'Files' / 'phm_diagnosis_qwen.py').is_file():
    drive.mount('/content/drive')
if not (PROJECT_ROOT / 'Files' / 'phm_diagnosis_qwen.py').is_file():
    raise FileNotFoundError('The new C files have not synced, or PROJECT_ROOT differs from your Drive folder.')
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'Files'))
from phm_run_reporting import RunReport
report = RunReport(PROJECT_ROOT, 'condition_C_validation_pilot')
report.install_notebook_hooks(get_ipython())
config = report.configure(PROJECT_ROOT / 'Files' / 'experiment_config.json')
BUNDLE = PROJECT_ROOT / 'Datasets/PHM15-diagnosis/validation_pilot_v1'
MAX_CASES = 3
RUN_C = True
RETRY_ERRORS = False
print('Condition: C only')
print('Project:', PROJECT_ROOT)
print('Run report:', report.folder)
```

If Drive prints `Drive already mounted at /content/drive`, that is normally an informational message. Continue if the project-path check succeeds; do not force-remount a working mount.

### Cell 2 — Install inference dependencies

```python
report.run_command([sys.executable, '-m', 'pip', 'install', '-r',
                    str(PROJECT_ROOT / 'Files' / 'requirements-qwen-colab.txt')],
                   name='install_Qwen_dependencies')
report.refresh_packages()
report.save()
```

If Colab explicitly asks you to restart the session after installation, restart and run again from Cell 1. Dependencies loaded into memory do not necessarily change when their files are updated.

### Cell 3 — Verify the case evidence and software

```python
from phm_diagnosis import prepare_requests
with report.step('check_portable_C_inputs'):
    contract, records, requests = prepare_requests(BUNDLE, config, MAX_CASES)
    report.check('validation_only', contract['scope'] == 'validation_pilot')
    report.check('three_training_cases_each', all(len(r['historical_cases']) == 3 for r in records))
    print('Selected queries:', len(records), 'of', len(contract['case_ids']))
    print('Native features:', contract['native_features'])
    print('Query IDs:', [r['case_id'] for r in records])
report.run_command([sys.executable, '-m', 'unittest', 'discover', '-s', 'Files',
                    '-p', 'test_phm_diagnosis.py', '-v'], name='diagnostic_software_checks')
```

The first three selected cases contain one example per class. The full ten-case pilot contains four `NoRecordedTarget`, three `RuntimeExcessive` and three `SupplyTempTooCold` cases. This sampling is for development checks; it is not an estimate of the dataset's natural class distribution.

### Cell 4 — Run case-based RAG inference

```python
from phm_diagnosis_qwen import run_condition_C
if RUN_C:
    result = run_condition_C(PROJECT_ROOT, max_cases=MAX_CASES,
                             retry_errors=RETRY_ERRORS, reporter=report)
    print(result['pilot_metrics'])
else:
    report.note('CPU setup only: RUN_C=False; no diagnostic generation requested.')
    print('Setup checked. Rerun from the setup cell with RUN_C=True when ready.')
```

This is the command that invokes the actual model. You do not need to create another model-loading cell. Loading a second copy would consume additional GPU memory.

The `run_condition_C` function assembles the prompts from the verified evidence, checks response caches, loads Qwen only if generation is needed, and saves each raw response before moving to the next case. Query reference labels are opened for scoring after inference, outside prompt preparation.

### Cell 5 — Finalize and locate the outputs

```python
folder = report.finish()
print('Status:', report.data['status'])
print('Saved report:', folder)
print('Latest pointer:', PROJECT_ROOT / 'Outputs' / 'LATEST_RUN.json')
print('Response cache:', PROJECT_ROOT / 'Outputs' / 'diagnosis' / 'C' / 'responses')
```

If an earlier cell raises an error, its report records the failure once setup has succeeded. Inspect that error before moving on. Finishing a report does not repair a failed run.

## 7. Inspect the results

### Where to look

| File | What it tells you |
|---|---|
| `Outputs/LATEST_RUN.md` | Human-readable summary of the most recently updated run |
| `Outputs/LATEST_RUN.json` | Exact directory of that run |
| `Outputs/runs/<run-id>/run_report.json` | Stage, status, environment, configuration, checks and artifact fingerprints |
| `diagnostic_requests.jsonl` inside the run folder | Exact prompts, query IDs, generation settings and evidence fingerprints |
| `diagnostic_responses.json` | Raw responses and current automatic validation results |
| `diagnostic_status.csv` | One row per selected query, including prediction, status, cost fields and scoring fields after scoring completes |
| `diagnostic_metrics.json` | Counts of valid/invalid/error outputs and development accuracy when scoring completed |
| `prompt_token_counts.csv` | Token lengths for requests that needed inference in this run |
| `qwen_runtime.json` | Model revision, GPU, package versions and loading time when this run loaded Qwen |
| `console.log`, `events.jsonl`, `traceback.txt` when present | Progress and observed errors |
| `Outputs/diagnosis/C/responses/` | Individual cached responses used for resume |

A cache-only run can complete without creating new model-loading or token-count files. The individual cached responses retain their original runtime information. Use the exact run folder printed by the notebook if another notebook has since updated `LATEST_RUN`.

### Optional inspection cell

Run this after Cell 5 if you want to read the results directly in Colab. It only reads the saved outputs.

```python
import csv
import json

RUN_FOLDER = report.folder
status_file = RUN_FOLDER / "diagnostic_status.csv"
responses_file = RUN_FOLDER / "diagnostic_responses.json"

if not status_file.is_file() or not responses_file.is_file():
    print("Diagnostic outputs are not available for this run. Check run_report.json and traceback.txt.")
else:
    with status_file.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            print({key: row.get(key) for key in
                   ("case_id", "status", "predicted_class", "reference_label", "correct", "issue")})

    details = json.loads(responses_file.read_text(encoding="utf-8"))
    for item in details:
        print("\nCASE:", item["case_id"])
        print("CHECKS:", item["checks"])
        response = item.get("response") or {}
        print("RAW OUTPUT:", response.get("raw_output"))
```

### What the response should contain

| Field | Expected content |
|---|---|
| `predicted_class` | `NoRecordedTarget`, `RuntimeExcessive`, `SupplyTempTooCold`, or `InsufficientEvidence` |
| `feature_evidence` | One to three measured query features and their displayed values for a classification; may be empty when abstaining |
| `case_comparison` | A supplied historical case ID and the supplied similarity/difference feature names; may be null when abstaining |
| `literature_citations` | Empty in C |
| `explanation` | Brief justification based on the provided evidence |
| `limitation` | A material uncertainty or gap |

`NoRecordedTarget` means neither chosen target event is recorded as active under the dataset rules. It does not prove the equipment is healthy. The regression scores are uncalibrated model estimates.

The automatic validator checks the allowed class, JSON structure, supplied case IDs, comparison features and quoted query values. **It does not judge every claim in the prose.** Read whether the explanation follows from the evidence. These software metrics are neither human review nor a separate LLM judge.

Small-pilot accuracy is a development check. Invalid responses, runtime errors and abstentions count as incorrect in accuracy over attempted queries; queries with no response yet are reported separately. Do not use this balanced ten-case pilot as the final performance claim.

## 8. Expand, resume or rerun

### Expand from three cases to ten

Set the first cell to:

```python
MAX_CASES = 10
RUN_C = True
RETRY_ERRORS = False
```

Then rerun from Cell 1. You can do this after inspecting the first three outputs; another chat turn is not required. With unchanged prompts, evidence and generation settings, the completed three responses are reused and the remaining seven are generated. Existing failed or invalid responses remain visible unless an explicit retry is requested.

### Resume after a disconnect

Reconnect, mount Drive and rerun from Cell 1 with the same settings. The runner checks the cache before loading the model. Saved responses survive because they are in Drive, while an interrupted request without a saved response must run again.

Colab session state is temporary and its managed virtual machines can be deleted after inactivity or their maximum lifetime. Store the results in the mounted project, as this notebook does. [Google Colab session behavior](https://research.google.com/colaboratory/faq.html)

### Retry an identified error

If you have addressed the cause of a failure, set `RETRY_ERRORS = True` and rerun from Cell 1. The runner can then retry cached runtime errors, invalid responses or invalid caches within the selected cases. It archives the previous attempt under `responses/previous_attempts/` before writing the replacement.

Keep the failed attempt as part of the audit. Do not repeatedly regenerate until a desired answer appears. If the prompt or generation settings change, the request fingerprint changes and the outputs belong to a new request; compare experiments only after the final settings are frozen.

### A report is already finished

If you see `Start a new report by rerunning the notebook setup cell before continuing`, rerun Cell 1 and continue in order. Each execution session needs an active report. Reusing a completed report object is deliberately blocked.

## 9. What you can change for this pilot

| Setting | Meaning | Recommended use now |
|---|---|---|
| `PROJECT_ROOT` | Mounted project location | Keep the verified path unless the actual Drive location changes. |
| `MAX_CASES` | Number of cases from the fixed ten-case export | `3` for a short check; `10` for the entire prepared pilot. |
| `RUN_C` | Whether to perform diagnostic inference | `True` to run the model; `False` for setup and software checks only. |
| `RETRY_ERRORS` | Whether to retry selected failed/invalid cached responses | Keep `False` initially; enable for a deliberate, documented retry. |

Keep model revision, seed, features, retrieved case order and generation settings fixed while comparing outputs. The current runner supports only the exported validation pilot and checks its hashes; it is not an interface for arbitrary new queries or final-test inference yet.

In `experiment_config.json`, `diagnostic_generations_per_query = 2` describes the eventual paired study: one C response and one D response per query. It does not make the current runner generate C twice.

## 10. Troubleshooting

| Symptom | Action |
|---|---|
| `Drive already mounted at /content/drive` | Continue if the project files are found. This message alone is not a failure. |
| `Mountpoint must not contain a space` | Restore the call to `drive.mount('/content/drive')`. Keep the longer project path only in `PROJECT_ROOT`. Do not rename the project folder to address this error. |
| `mount failed` even with `/content/drive` | This is a separate Drive-mount failure; it is not diagnosed by the space error. Try a fresh Colab runtime and complete the Drive authorization prompt, then rerun setup with the fixed mountpoint. If it persists, preserve the exact error for diagnosis rather than changing `PROJECT_ROOT` to a guessed location. |
| Project folder or `phm_diagnosis_qwen.py` not found | Confirm sync completed and inspect the mounted Drive folder. `PROJECT_ROOT` must be the directory containing `Files`; use the Colab path, not a Windows path. |
| `ModuleNotFoundError: phm_diagnosis` or `phm_run_reporting` | Rerun Cell 1. Check that the relevant `.py` files synced into `Files`. |
| A package such as `transformers`, `accelerate` or `bitsandbytes` is missing | Run Cell 2. If installation fails, inspect its saved output instead of starting inference. |
| Colab requests a restart after installation | Restart the session, then rerun from Cell 1. |
| `CUDA unavailable` | Select a GPU runtime and reconnect. Verify CUDA using the check below. |
| GPU unavailable because of a Colab allocation limit | Resume when a GPU is available. The saved response cache remains usable. This is separate from Hugging Face access. |
| Notice that no Hugging Face token was found | For this public model, continue if downloads work. A notice is not the same as a raised authentication error. |
| Actual `401`, `403`, `429`, connection or download error | Check the logged error, model ID/revision and network access. `429` is a rate-limit response. Preserve the report; retry after resolving the specific problem. Do not paste tokens into code or chat. |
| `Export integrity failed` | Confirm all export files finished syncing together. Do not edit the expected hashes to bypass the check. Restore the verified export or use the local export command in section 13. |
| `CUDA out of memory` | Start a fresh GPU runtime so an earlier model is no longer resident. Run only this notebook. If the same configuration still fails, save the report for adjustment; reducing `MAX_CASES` alone does not shorten a single prompt. |
| Prompt exceeds the token budget | Preserve the full prompt and report. Review the context budget before inference; the runner intentionally does not truncate the evidence. |
| `invalid_response` or `generation_hit_limit: true` | Read the raw output. The model may have broken the JSON contract or reached its output limit. Keep the failed attempt; any prompt/budget change is a validation change to document. |
| All selected responses are cached | This is expected when resuming unchanged requests. The runner can review and score them without loading Qwen again. |
| A report still says `running` after a disconnect | Treat it as the last saved state. Check Colab and the last event; abrupt termination cannot write a final status. |
| “All available literature sources are excluded” | You opened the deferred literature workflow. Use the C notebook for the current run. Charan must remain excluded. |
| D is rejected by the runner | Expected at this stage. D requires additional implementation and permitted diagnostic literature; see section 11. |

### Optional GPU check

Run this after dependency installation if the GPU is uncertain:

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

If the error happens before Drive mounts or before the project root is found, no synced report can be written yet. Copy that error text when requesting help. Errors after report setup are normally saved in the run folder.

## 11. What remains for literature-based condition D

**C first remains the development order.** Charan is excluded from fresh ingestion and from previously prepared GPU jobs. Its historical files do not count as active or approved evidence.

Once C is operational, the remaining implementation is:

| Stage | Work required | Completion evidence |
|---|---|---|
| Select permitted literature | Choose relevant papers/passages and check their diagnostic applicability; methodological relevance alone is insufficient | Source registry and reviewed passage provenance |
| Prepare passages | Extract source text, preserve page/section identifiers, validate chunks and review useful passages | Approved chunks with original source locations |
| Build literature retrieval | Encode approved passages and search using information available at query time; exclude query reference labels | Versioned index and saved retrieved passage IDs/text per query |
| Add D to the generator | Reuse the identical query features, scores and historical cases; add retrieved literature and allow verified passage citations | A real D implementation with strict output checks |
| Run paired validation | Use the same validation queries and generation settings for C and D; verify context sizes and costs | Paired outputs and a frozen configuration |
| Run held-out evaluation | Export the reserved test queries under the frozen protocol and score both conditions | Paired effectiveness, grounding and efficiency results |

The current configuration already records an intended literature embedding model and retrieval count. Those settings do not create an index or implement D. There is no working `RUN_D` cell in the present notebook.

Qwen-assisted literature chunking, if retained, is offline preparation. It is not fine-tuning and does not itself provide retrieval at diagnosis time. The diagnostic runner still needs an approved index, retrieved passages and D-specific validation.

This guide deliberately gives executable code for the implemented workflow and an explicit completion checklist for the unfinished workflow. It does not provide placeholder D commands that would silently run C again.

## 12. What to send Codex for review

After sync, send:

> C pilot finished. I ran MAX_CASES = 3 [or 10]. Please inspect Outputs/LATEST_RUN.json and the associated diagnostic requests, responses, metrics and errors. Check whether the classifications and historical-case comparisons are supported before we extend the pipeline to D.

If there was an error:

> C setup/run failed at cell [number]. The error is [short error text]. Please inspect the saved run folder [path printed by the notebook]. Keep the existing cases, split and C/D experiment scope, and fix the immediate failure.

Providing the exact run folder is useful if another notebook has updated the latest-run pointer. You do not need to copy all model outputs into chat when the files have synced locally.

## 13. Code map and optional local export recovery

### Which file does what?

| File | Responsibility |
|---|---|
| `Files/condition_C_diagnosis_colab.ipynb` | User-facing Colab workflow |
| `Files/phm_diagnosis.py` | Portable evidence checks, compact prompts, response validation, cache checks and pilot scoring |
| `Files/phm_diagnosis_qwen.py` | Qwen loading, inference, per-response saving and run coordination |
| `Files/phm_run_reporting.py` | Logs, exceptions, reports, artifact fingerprints and latest-run pointer |
| `Files/experiment_config.json` | C/D design, model pin, decoding settings, retrieval settings and paired controls |
| `Files/requirements-qwen-colab.txt` | Colab inference dependencies |
| `Files/test_phm_diagnosis.py` | Synthetic software tests for leakage controls, output checks and resume behavior |
| `Files/phm_case_export.py` | Local export from the existing numerical model and case bank into portable JSON |
| `Files/phm_case_system.py` | Existing numerical classifier and historical-case retrieval |
| `Files/phm_literature.py` and `Files/phm_literature_qwen.py` | Deferred literature preparation utilities, with source-exclusion checks |
| `Files/literature_sources.json` | Literature source policy, including Charan's exclusion |

For the current C run, you do not need to execute `preprocessing_experimental_data.ipynb`, `eda_case_design.ipynb`, `case_representation_and_retrieval.ipynb`, or `literature_ingestion_and_chunking.ipynb`. Their preparation work is separate from running the exported C evidence.

### Recreate or verify the portable export locally

Use this only if the export is missing or needs a deliberate provenance check. Run it in **Windows PowerShell on your computer**, using the original project environment:

```powershell
Set-Location -LiteralPath 'C:\Users\ges2j\OneDrive\Desktop\Assessments\CP5403'
& '.\Files\.venv\Scripts\python.exe' '.\Files\phm_case_export.py'
```

This reads the existing fitted numerical artifact; it does not refit the classifier or inspect final-test outcomes. It verifies the saved numerical source hashes and refuses to silently replace a different existing export. If that guard fires, preserve the message and have the difference reviewed rather than deleting artifacts or changing hashes.

Do not run this local recovery command inside Colab. The exported JSON is the portability boundary between the local numerical environment and the Colab inference environment.

## 14. Completion checklist for this setup

- [ ] The required Python files and validation export are present in the mounted project.
- [ ] Colab has an available GPU and the pinned dependencies install successfully.
- [ ] The evidence checks and diagnosis software tests pass.
- [ ] The selected C requests either produce saved outputs or have an explicit saved error; none silently disappear.
- [ ] `diagnostic_requests.jsonl` records the exact prompts without query reference labels.
- [ ] Raw responses, validation results and per-query cost measurements are saved.
- [ ] The final run status and run directory are visible and synced locally.
- [ ] Explanations have been read for support, beyond merely passing JSON checks.
- [ ] The run is described as a C validation pilot; D and final held-out evaluation remain separately identified.

At that point, the current **case-based RAG + Qwen setup has been exercised end to end**. The saved report provides the concrete evidence needed to decide the next implementation step.
