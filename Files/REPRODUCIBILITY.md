# Reproducing and Inspecting the Assessed Experiment

This is the current reviewer guide, dated November 27, 2026. The final experiment compares the same 100 cases under C (features, numerical scores and historical cases) and D (the same inputs plus selected literature). Qwen's frozen decisions achieved 52% / 51% accuracy and .318 / .313 macro-F1. The original failures are part of the evidence.

## 1. Start with the Existing Results: No GPU

Open a terminal in the **CP5403 project root**, whose immediate children include `Files`, `Datasets`, `Outputs` and `Report-Documentation`.

Windows, using the existing local environment:

```powershell
& .\Files\.venv\Scripts\python.exe -B .\Files\review_reproducibility.py --root .
```

With another Python installation, the saved-result audit itself uses only the standard library:

```text
python Files/review_reproducibility.py --root .
```

This recalculates metrics, verifies paired prompts, checks 13 frozen code/configuration fingerprints, six export-file fingerprints and 400 raw-response fingerprints, and reads the preserved diagnostic comparisons. It does not refit the classifier, download a model, execute Qwen or write to the experiment. Detailed audit logic is in `Report-Documentation/recheck_saved_results.py`; the short command above provides an entry point.

To include the existing final-workflow tests in the configured local numerical environment:

```powershell
& .\Files\.venv\Scripts\python.exe -B .\Files\review_reproducibility.py --root . --run-tests
```

These tests need the project's numerical/ONNX/schema dependencies and validation artifacts. They do not generate Qwen answers. Missing dependencies are test-environment failures, not changes to the saved scores. Their scope is bounded; they do not replace a complete raw-data rebuild or a semantic review.

## 2. Authoritative Records

| Purpose | Path relative to CP5403 |
|---|---|
| Assessed report and detailed appendices | `CP5403-Assessment_2-Sabaysabay-Report.md` |
| Completed held-out run | `Outputs/runs/20261122T103151_paired_CD_two_stage_heldout_13ba4e1a/` |
| Exact effective generator settings | `two_stage_config.json` in that run |
| Actual requests and responses | `decision_requests.jsonl`, `explanation_requests.jsonl`, `decision_responses.json`, `explanation_responses.json` in that run |
| Scored predictions and metrics | `paired_results.json`, `final_evaluation_metrics.json` in that run |
| Frozen test protocol and exports | `Datasets/PHM15-diagnosis/heldout_test_v1/` |
| Approved literature index | `Datasets/PHM15-literature-D-approved-v1/` |
| Numerical model and historical bank | `Datasets/PHM15-prepared/case_system.duckdb` |
| Full run inventory | `Outputs/RUN_INDEX.md` |
| Code roles and import relationships | `Files/CODE_MAP.md` |

`Outputs/LATEST_RUN.*` is a mutable activity pointer, not the identifier of the assessed experiment. The inherited generic configuration still contains pilot-stage descriptions; the dedicated held-out protocol and actual saved requests define the frozen evaluation. Those files have not been rewritten to make history appear cleaner.

## 3. Four Primary Notebooks

1. `preprocessing_experimental_data.ipynb`: ingest raw component/event sources, align observations, construct eligibility and site partitions.
2. `eda_case_design.ipynb`: training exploration, sampling, coverage and baseline diagnostics.
3. `case_representation_and_retrieval.ipynb`: select the representation, fit the numerical system and assemble historical retrieval.
4. `condition_CD_heldout_100_colab.ipynb`: the preserved final Colab orchestration.

The first three publish databases when rerun. Rebuild in a separate working copy if the original artifacts must remain byte-identical. Use the saved source manifests and contracts to compare the reconstructed outputs. A successful saved-result audit alone is not a claim that these stages have been rebuilt independently.

The approved literature workflow uses `phm_literature.py`, `phm_literature_index.py`, `literature_passage_review.json` and `literature_retrieval_policy_v1.json`. The old literature notebooks are prototypes and are archived; do not use them as the default path to recreate the final approved corpus. Preserve the original source PDFs, extraction records, approved passages, embeddings and encoder assets.

## 4. Colab Inference Replication

Use a separate copy of the complete project and open `condition_CD_heldout_100_colab.ipynb`. The notebook contains a machine-specific `ROOT` value. In the replication copy, set `ROOT` to the actual mounted folder that contains `Files/phm_heldout_export.py`; do not assume a desktop `Computers` path is exposed under the same name in Colab. The mount point remains `/content/drive`.

The setup installs the checked-in literature/Colab requirements plus `onnxruntime==1.31.1` and `lm-format-enforcer==0.11.3`, while using Colab's CUDA-enabled PyTorch. The notebook checks the frozen module hashes, runs its two test modules, prepares requests and then executes/resumes inference. Its last code cell can generate up to 400 responses: use the no-GPU audit above when only checking the report.

The model is `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; effective limits are 6,144 input and 384 output tokens per stage. The recorded device was NVIDIA L4. The literature encoder revision is `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

Existing responses are cached under `Outputs/diagnosis/two_stage_heldout_v1`. Running against those caches checks/resumes existing attempts; it does not constitute fresh inference. For a genuinely fresh replication, retain the original project and use a separately named copy with a fresh cache location/state. Do not blend replacement responses with the published run or relabel repeated test tuning as independent validation. Exact cross-hardware outputs are not guaranteed by greedy decoding alone.

## 5. Files Needed When Sharing

Do not upload only loose Python files. Preserve the CP5403 directory layout:

```text
CP5403/
  README.md
  CP5403-Assessment_2-Sabaysabay-Report.md
  Files/                       # modules, four primary notebooks, tests, configs
    archive/                   # preserved older launchers and conflict copies
  Report-Documentation/        # audit script and documentation
  Datasets/
    PHM15-diagnosis/heldout_test_v1/
    PHM15-diagnosis/validation_pilot_v1/
    PHM15-literature-D-approved-v1/
    PHM15-prepared/             # needed for numerical rebuild, not basic metric audit
  Outputs/
    runs/20261122T103151_paired_CD_two_stage_heldout_13ba4e1a/
    diagnosis/two_stage_heldout_v1/
```

Raw reconstruction additionally needs `Datasets/PHM15-master`, the original RAG PDFs/extraction artifacts, and the corresponding development databases. The project's dataset-sharing link is in the root README. Its availability to a reviewer has not been independently tested here.

The local `.git` directory currently resides inside `Files`. Therefore, a commit made there does **not** automatically include the parent report, `Report-Documentation`, `Datasets` or `Outputs`. The cleanup did not relocate Git metadata or alter the existing staged changes. Check the remote layout and provide the complementary artifact bundle before claiming the GitHub checkout alone reproduces the experiment.

The pinned dependency files and runtime snapshots describe separate numerical and Colab environments; they are not a tested all-platform lockfile. Use the recorded environment when loading serialized numerical models.

## 6. Provenance and GenAI

Earlier notebooks and sync-conflict copies are preserved with their hashes in `archive/relocation_manifest.json`. A notebook may refer to an older code fingerprint that is no longer the active version; archiving it does not make that historical environment reproducible automatically. Failed and interrupted runs remain in `Outputs/runs` and are catalogued, not removed.

The cleanup and reviewer command were prepared with Codex assistance. The original frozen Python/configuration files and notebook content were not refactored. The report's GenAI declaration and Appendix J describe the broader assistance and verification limits.
