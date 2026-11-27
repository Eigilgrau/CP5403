# Source-Code Map

Static import inventory, November 27, 2026. Includes imports inside functions; membership does not mean execution during the saved-result audit. Source contents remain unchanged. Start with review_reproducibility.py, phm_two_stage.py and phm_final_metrics.py.

## Final pipeline dependency closure

| Module | Declared responsibility | Local imports |
|---|---|---|
| [phm_case_design.py](phm_case_design.py) | Scout training sites, construct case summaries, and diagnose a logistic pilot. | phm_case_samples, phm_preprocessing |
| [phm_case_samples.py](phm_case_samples.py) | Build measurement-only examples and independently check the private scores. | phm_preprocessing |
| [phm_case_system.py](phm_case_system.py) | Represent PHM cases, validate a classifier, and retrieve historical evidence. | phm_case_design, phm_case_samples, phm_preprocessing |
| [phm_condition_d.py](phm_condition_d.py) | Paired C/D validation pilot with a separately reviewed literature index. | phm_diagnosis, phm_diagnosis_qwen, phm_literature_index, phm_output_schema, phm_run_reporting |
| [phm_diagnosis.py](phm_diagnosis.py) | Portable condition-C prompts, evidence checks, response checks and pilot scoring. | phm_run_reporting |
| [phm_diagnosis_qwen.py](phm_diagnosis_qwen.py) | Run the case-only Qwen validation pilot on Colab, saving every response to Drive. | phm_diagnosis, phm_output_schema, phm_run_reporting |
| [phm_evidence_selection.py](phm_evidence_selection.py) | Select evidence IDs with Qwen; assemble numerical evidence in Python. | phm_condition_d, phm_diagnosis, phm_output_schema |
| [phm_filtered_selection_pilot.py](phm_filtered_selection_pilot.py) | Paired C/D validation pilot with a separately reviewed literature index. | phm_diagnosis, phm_diagnosis_qwen, phm_evidence_selection, phm_literature_index, phm_output_schema, phm_run_reporting |
| [phm_final_metrics.py](phm_final_metrics.py) | Fixed held-out classification metrics and paired site-cluster intervals. | phm_diagnosis, phm_heldout_export, phm_run_reporting |
| [phm_heldout_export.py](phm_heldout_export.py) | Freeze and export a site-balanced, label-blind held-out evaluation sample. | phm_case_system, phm_diagnosis, phm_run_reporting |
| [phm_literature.py](phm_literature.py) | Extract attributable prose and construct validated literature chunks. | phm_run_reporting |
| [phm_literature_index.py](phm_literature_index.py) | Build a source-verifiable, passage-approved MiniLM literature index on CPU. | phm_literature |
| [phm_output_schema.py](phm_output_schema.py) | Generation-time JSON constraints; semantic support still needs validation. | phm_diagnosis |
| [phm_preprocessing.py](phm_preprocessing.py) | Prepare the PHM 2015 component and event streams for case construction. | None |
| [phm_run_reporting.py](phm_run_reporting.py) | Persist notebook progress and reviewable results to the synced project directory. | None |
| [phm_two_stage.py](phm_two_stage.py) | Versioned paired C/D classification then explanation; validation data only. | phm_diagnosis, phm_diagnosis_qwen, phm_evidence_selection, phm_filtered_selection_pilot, phm_heldout_export, phm_output_schema, phm_run_reporting |

## Additional export, review and development modules

| Module | Declared responsibility | Local imports |
|---|---|---|
| [phm_case_export.py](phm_case_export.py) | Export validation evidence locally from the existing fitted PHM case system. | phm_case_system, phm_diagnosis, phm_run_reporting |
| [phm_design_smoke.py](phm_design_smoke.py) | Four generation checks only. No outcome labels, scoring, or automatic expansion. | phm_diagnosis, phm_diagnosis_qwen, phm_run_reporting, phm_two_stage |
| [phm_evidence_design.py](phm_evidence_design.py) | Local design-review preparation and structural validation; no inference or scoring. | phm_diagnosis, phm_evidence_menu_smoke, phm_run_reporting |
| [phm_evidence_menu_smoke.py](phm_evidence_menu_smoke.py) | Four generation checks only. No outcome labels, scoring, or automatic expansion. | phm_diagnosis, phm_diagnosis_qwen, phm_run_reporting, phm_two_stage |
| [phm_evidence_review_check.py](phm_evidence_review_check.py) | Four generation checks only. No outcome labels, scoring, or automatic expansion. | phm_diagnosis, phm_diagnosis_qwen, phm_evidence_design, phm_evidence_menu_smoke, phm_run_reporting, phm_two_stage |
| [phm_explanation_judge.py](phm_explanation_judge.py) | Exploratory Qwen14B review of saved PHM explanations, not diagnosis generation. | phm_diagnosis, phm_diagnosis_qwen, phm_grounding_review, phm_judge_v2, phm_run_reporting |
| [phm_final_evaluation_pack.py](phm_final_evaluation_pack.py) | Consolidate saved results and prepare unscored human review. No inference. | phm_diagnosis, phm_final_metrics, phm_run_reporting |
| [phm_foundation_audit.py](phm_foundation_audit.py) | Read-only train/validation foundation audit. No refit, test outcomes or LLM calls. | phm_run_reporting |
| [phm_grounding_review.py](phm_grounding_review.py) | Prepare a blinded evidence-grounding review pack from saved two-stage outputs. | None |
| [phm_judge_calibration.py](phm_judge_calibration.py) | Run eight synthetic grounding checks with a separately pinned Llama judge. | phm_diagnosis, phm_diagnosis_qwen, phm_grounding_review, phm_output_schema, phm_run_reporting |
| [phm_judge_v2.py](phm_judge_v2.py) | Consolidated judge experiment: compact separate grounding/citation tasks. | phm_diagnosis, phm_diagnosis_qwen, phm_output_schema, phm_run_reporting |
| [phm_literature_qwen.py](phm_literature_qwen.py) | Run resumable Qwen boundary selection on a CUDA Colab runtime. | phm_literature, phm_run_reporting |
| [phm_new_case_review.py](phm_new_case_review.py) | Twelve generation checks only. No query outcome labels, scoring, or automatic expansion. | phm_diagnosis, phm_diagnosis_qwen, phm_evidence_design, phm_evidence_menu_smoke, phm_run_reporting, phm_two_stage |
| [phm_new_case_review_export.py](phm_new_case_review_export.py) | Local class-stratified development sampling; no labels exported to inference. | phm_diagnosis, phm_evidence_design, phm_evidence_menu_smoke, phm_new_case_review, phm_run_reporting |
| [phm_offline_pilot_audit.py](phm_offline_pilot_audit.py) | Audit saved paired runs without a GPU, model loading, or private label files. | None |
| [phm_recovery_export.py](phm_recovery_export.py) | Local export of a fixed validation cohort; no held-out rows or model inference. | phm_case_system, phm_diagnosis, phm_filtered_selection_pilot, phm_recovery_validation, phm_run_reporting, phm_two_stage |
| [phm_recovery_validation.py](phm_recovery_validation.py) | One bounded, post-test development comparison. Never loads held-out data. | phm_diagnosis, phm_diagnosis_qwen, phm_final_metrics, phm_run_reporting, phm_two_stage |
| [phm_selection_pilot.py](phm_selection_pilot.py) | Paired C/D validation pilot with a separately reviewed literature index. | phm_diagnosis, phm_diagnosis_qwen, phm_evidence_selection, phm_literature_index, phm_output_schema, phm_run_reporting |
| [phm_structured_smoke.py](phm_structured_smoke.py) | Four generation checks only. No outcome labels, scoring, or automatic expansion. | phm_diagnosis, phm_diagnosis_qwen, phm_run_reporting, phm_two_stage |
| [phm_summarise_human_review.py](phm_summarise_human_review.py) | Aggregate actual completed human ratings; never generate or fill ratings. | phm_final_evaluation_pack, phm_run_reporting |
| [phm_temporal_retrieval_revision.py](phm_temporal_retrieval_revision.py) | One fixed training-only retrieval revision; no LLM or held-out evaluation. | phm_case_system, phm_run_reporting |
| [phm_training_evidence.py](phm_training_evidence.py) | Training-only descriptive evidence export. No generation, scoring or model fitting. | phm_case_system, phm_evidence_menu_smoke, phm_run_reporting |

## Existing Tests

The final notebook and reviewer command run test_phm_two_stage.py and test_phm_heldout.py. Other tests cover development tools and can need additional artifacts/dependencies.

- [test_phm_condition_d.py](test_phm_condition_d.py)
- [test_phm_design_smoke.py](test_phm_design_smoke.py)
- [test_phm_diagnosis.py](test_phm_diagnosis.py)
- [test_phm_evidence_design.py](test_phm_evidence_design.py)
- [test_phm_evidence_menu_smoke.py](test_phm_evidence_menu_smoke.py)
- [test_phm_explanation_judge.py](test_phm_explanation_judge.py)
- [test_phm_filtered_retrieval.py](test_phm_filtered_retrieval.py)
- [test_phm_grounding_review.py](test_phm_grounding_review.py)
- [test_phm_heldout.py](test_phm_heldout.py)
- [test_phm_judge_calibration.py](test_phm_judge_calibration.py)
- [test_phm_judge_v2.py](test_phm_judge_v2.py)
- [test_phm_literature.py](test_phm_literature.py)
- [test_phm_new_case_review.py](test_phm_new_case_review.py)
- [test_phm_recovery_validation.py](test_phm_recovery_validation.py)
- [test_phm_run_reporting.py](test_phm_run_reporting.py)
- [test_phm_selection.py](test_phm_selection.py)
- [test_phm_structured_smoke.py](test_phm_structured_smoke.py)
- [test_phm_temporal_retrieval_revision.py](test_phm_temporal_retrieval_revision.py)
- [test_phm_training_evidence.py](test_phm_training_evidence.py)
- [test_phm_two_stage.py](test_phm_two_stage.py)

## Why shared modules remain in place

For example, phm_filtered_selection_pilot.py remains a dependency of the final two-stage path despite its name. Moving or rewriting these modules would invalidate imports and the frozen contract. Historical notebook launchers are physically archived.
