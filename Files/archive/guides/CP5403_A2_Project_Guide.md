# CP5403 project guide for evidence-grounded fault diagnosis

**Working title** — *Case-based fault diagnosis at unseen industrial sites using an open-weights LLM and retrieval-augmented generation*

**Research question** — Given the same three-hour measurement summary, logistic-regression scores and retrieved historical cases, what does adding technical literature contribute to Qwen's classification of an active PHM event at an unseen site, the support for its explanations, and its inference cost?

**Active scope** — C and D only, following the user's time-limited scope revision. `Files/experiment_config.json` is the active configuration. Earlier version-history notes and the existing DOCX draft may describe the earlier A–D design; they are historical documents. Use the staged checkpoints below rather than the earlier November schedule.

**Development order: C first, then D.** First establish case-only inference, saved outputs and review on validation cases. Then add approved literature to the same query evidence and retrieved historical cases. Charan is excluded from active ingestion, prompts and pilots by user instruction; its older files and reports are historical records only.

## 1. The application you will demonstrate

A maintenance analyst selects a site and decision time. The application summarises what the instruments have recorded up to that time. Logistic regression scores the candidate event classes. A retriever finds comparable historical windows with known outcomes and a small number of relevant passages from the technical literature. Qwen returns a candidate class, a short justification with traceable evidence, and a limitation. The analyst can open the cited case or passage to check the explanation.

Choose **active-event diagnosis** for this assessment. If an event runs from 10:00 to 11:00, a decision at 10:15 can use readings from 07:15 to 10:15. This preserves your idea of examining conditions before and during a fault. An early-warning system would need a prediction horizon, a lead-time metric and a separate experiment. Record it as future work.

The brief permits RAG with an existing open-weights model. Its HD criteria reward a well-justified application and dataset, sophisticated implementation informed by independent research, and evaluation code covering effectiveness and efficiency. The report and presentation also assess how well you explain your choices. A finding that regression classifies better while retrieval yields better-supported explanations is worth reporting. [Assessment brief and rubric, pp. 1–5](Problem%20Brief/CP5403_A2.pdf) · [Your starting draft](CP5403-Assessment_2-Sabaysabay.docx)

## 2. Define exactly what counts as a case

**Use all 70 sites in the data audit.** Of the 210 local CSVs, 66 component files, 58 zone files and 69 event files are non-empty. Sites 50, 52, 62 and 64 lack component readings. Site 62 also has an empty event log. This leaves **66 candidate sites** for the component-based experiment before further quality checks. Make component measurements the core input so that missing zone files alone do not disqualify a site. Record each exclusion in the site inventory. [Local dataset](Datasets/PHM15-master/)

**Start with two target event names and a comparison class.** The local logs contain 49 distinct event names, including administrative-looking entries. In a read-only count limited to sites with component data, `RuntimeExcessive` has 17,458 logged intervals lasting at least 15 minutes across 65 sites. `SupplyTempTooCold` has 13,811 across 53 sites. These are raw interval counts before deduplication, overlap checks and alignment with measurements. Their breadth makes them sensible pilot targets. The original challenge describes numeric codes and a different scoring task. Describe yours as an adaptation using the event names in this release. [Local challenge rules](Datasets/PHM15-master/PHM_Data_Challenge_Rules_2015_Repository.pdf) · [PHM Society challenge description](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-phm-society-2015/phm-data-challenge-3/)

| At decision time t | Reference label for scoring |
|---|---|
| One of the two target codes is active, using start ≤ t < end | The active target code |
| Neither target is logged as active and both the measurement window and log coverage meet the eligibility rules | `NoRecordedTarget` |
| Both targets are active or an unresolved target interval makes the label ambiguous | Exclude from the primary comparison and count the reason |

`NoRecordedTarget` can include other events. It means the log does not record either selected target at that time. The challenge warns that some labels can be missing. As a conservative starting rule, draw controls only when valid log entries bracket the decision time within seven days on either side. Exclude known incomplete periods and censor a target with a missing end from its recorded start until a verified clearance or the end of the site record. Inspect how much data this costs on training sites before freezing the rule. Bracketing demonstrates logging activity, but faults can still go unrecorded. An equipment failure, its physical cause and the event name recorded by the monitoring system are distinct claims. This experiment can verify the last one.

**Separate sites before fitting anything.** Keep Site 1 in development because it supplied the worked example below. With 66 eligible sites, an initial allocation is 46 training, 10 validation and 10 test sites. Adjust for actual eligibility and class coverage, then save the assignments and seed. Training sites supply the regression fit, feature reference ranges and historical cases. Validation sites are for choosing the prompt and retrieval settings. Reserve the test sites for the frozen experiment.

**Construct windows from the measurement clock.** Use a fixed 15-minute decision grid and a three-hour trailing window, open at the left and closed at the right. Sort by component and actual timestamp. Within each 15-minute bin keep the latest observation available by the bin end. This handles logging jitter without looking ahead. Attach the reference label to a separate scoring table afterward. Merge duplicate or overlapping intervals of the same target. Sample at most one scored window per merged positive episode and space sampled background windows apart. Avoid overlapping test windows where practical. Log any discarded episodes, including short events missed by the decision grid.

## 3. Turn a sensor window into evidence the model can use

1. **Summarise each component first.** Begin with the median, interquartile range and change from first to last valid reading for each selected numeric channel. Then summarise across components, retaining both a typical value and an extreme so one affected unit remains visible. Include observed-component counts and missingness. Use channel IDs and documented units. Verify the column mapping before giving any channel a physical name. Add an operating-state proportion only if its encoding is consistent across the selected sites.
2. **Use a reproducible coverage rule.** As a pilot default, require at least 8 of the 12 bins for a component’s channel summary. Record the number of usable components as well as the number observed. On training sites, inspect the windows this removes and freeze a site-window threshold before evaluation. Fit imputation, scaling and reference percentiles using training sites alone.
3. **Fit one regularised multiclass logistic regression across training cases.** The coefficients belong to the fitted model. The three class scores belong to an individual query. Attach those scores to the query JSON and show the same scores to Qwen in C and D. This holds the numerical classifier evidence fixed while measuring the additional contribution of literature. Regression is retained as an evidence source, without a separate diagnostic ablation arm. Treat the scores as model estimates. A held-out calibration analysis would be needed to claim reliable real-world probabilities. Xiao’s PHM 2015 work gives you a specific, citable reason to use penalised logistic regression. [Xiao’s study](https://papers.phmsociety.org/index.php/ijphm/article/view/2360)

### A worked case from the local files

At one decision time in Site 1, the preceding three hours contained 65 populated component bins out of 72 possible for the six observed components on the first numeric channel. Five components met the 8-bin rule. Their individual medians were 709, 712, 736, 722.5 and 724. This produces the following genuine excerpt from a query. The fitted regression scores would be added when the baseline exists.

```json
{
  "case_id": "demo-01",
  "lookback_minutes": 180,
  "observed_components": 6,
  "usable_components_ch1": 5,
  "observed_component_bins_ch1": 65,
  "possible_component_bins_ch1": 72,
  "ch1_median_of_component_medians": 722.5,
  "ch1_lowest_component_median": 709,
  "ch1_highest_component_median": 736
}
```

The private scoring record gives `RuntimeExcessive` at 18:30 on 7 June 2010. The model can accurately describe the spread and the missing coverage. Calling 722.5 “unusually high” requires a reference distribution. Attributing the event to a particular component requires evidence beyond this site-level event record. That distinction gives you a concrete example for the report and the oral presentation. [Component readings](Datasets/PHM15-master/site1a.csv) · [Event log](Datasets/PHM15-master/site1c.csv)

## 4. Build the two retrieval collections

| Collection | Contents and retrieval | What a citation establishes |
|---|---|---|
| Historical cases | Index feature vectors from training sites. Use nearest neighbours on training-standardised features. Return up to three with their observed outcomes, raw feature values and opaque case IDs. Limit repeated examples from the same site and episode. | A comparable historical window had these measurements and this recorded outcome. |
| Research literature | Start with about five to eight verified sources. Keep short, self-contained passages with author, title, page or section, and a stable passage ID. Encode passages once on CPU. Retrieve up to two passages using a short description of the observed pattern and candidate classes available at inference. | A particular source supports a domain interpretation, method or limitation. |

Select permitted literature by the claim it can support, after the C pilot works. Xiao and Wu et al. are relevant for use of PHM 2015 and its classification challenges. A fault-detection field study can support a narrower observation about how operating conditions affect interpretation. Keep general RAG design papers in the report bibliography unless a passage also helps the analyst interpret a case. A paper's relevance to methodology does not automatically make it suitable diagnostic evidence. Charan must not be included. [Xiao](https://papers.phmsociety.org/index.php/ijphm/article/view/2360) · [Wu et al.](https://doi.org/10.3390/s18041096) · [Air-handling fault-detection study](https://www.nist.gov/publications/expert-rule-set-fault-detection-air-handling-units)

For a small corpus, a paragraph encoder and cosine similarity are enough to implement and inspect retrieval on Colab. Record the exact passage text from the source and its location so a citation can be checked. Screen PHM-specific papers for site tables or worked examples that reveal held-out answers. Keep original site identifiers and absolute timestamps in the private provenance table. Use opaque IDs and relative times in prompts and public examples. Credit PHM 2015 in the report and check redistribution terms before including raw logs or article text in a public repository. [Paragraph encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

The case-based reasoning step is the **comparison**. Ask Qwen to name one measurable similarity to a retrieved case and one important difference or gap. Require it to explain how these affect the candidate classification. Freeze the case collection during the test. The report can connect this design to retrieval and reuse in case-based reasoning and to the provenance of evidence in RAG. [Aamodt and Plaza](https://www.iiia.csic.es/~enric/papers/AICom.pdf) · [Lewis et al.](https://arxiv.org/abs/2005.11401)

## 5. Run one controlled experiment

Use **Qwen3-4B-Instruct-2507** in a Colab notebook, starting with 4-bit loading, short contexts and a fixed output limit. Pin the model revision, library versions and decoding settings. Perform the data preparation, regression and document indexing on CPU. Pilot ten validation queries on the assigned GPU and save each response immediately. Record the actual hardware and time per query before choosing the final sample size. [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) · [Colab resource limits](https://research.google.com/colaboratory/faq.html)

| Run | Evidence available | Purpose |
|---|---|---|
| C. Qwen with historical cases | Query features, regression scores and the nearest labelled training cases | Reference condition for the literature comparison |
| D. Qwen with cases and papers | Exactly the same evidence as C, plus retrieved approved literature passages | Additional contribution of literature to diagnosis, grounding and runtime |

Use identical held-out case IDs, features, regression scores, historical-case IDs/order and Qwen settings for C and D. Only the literature evidence should differ. This experiment estimates the additional value of literature in a case-based system; it does not isolate the benefit of case retrieval or the LLM over regression alone. Have Qwen return a class or `InsufficientEvidence`, observed feature IDs and values, historical case IDs, passage IDs, a brief justification, and one limitation. Permit empty citation fields where a source is unavailable, and record how often D retrieves no suitable passage. Keep the true label, event boundaries, split membership and phase names such as “during failure” outside every query and retrieval key. The known outcome of a retrieved training case is legitimate evidence.

**Prompt instruction to pilot**

> Choose the best-supported event class for the query at its decision time. Cite the observed feature and value underlying your decision. If historical cases are supplied, identify one relevant similarity and any consequential difference. Cite a passage only for a claim the passage supports. State one limitation. If the evidence is insufficient to select a class, return InsufficientEvidence. Return the agreed JSON fields.

**Colab budget.** Pilot ten validation queries in both conditions: 20 diagnostic responses. Choose the final sample size from the observed runtime and freeze the sample and seed before inspecting test results. There are two diagnostic responses per held-out case: 90 cases would require 180 responses, or 60 cases would require 120. Offline literature chunking is an additional one-time preparation cost. Report class and site counts because a balanced sampled set has a different class mix from continuous plant operation.

## 6. Evaluate what each part contributed

| Question | Measure and interpretation |
|---|---|
| Did it identify the logged event? | Report macro-F1 for the three target classes, per-class precision and recall, confusion counts and test support by site. Retain invalid responses and abstentions as unsuccessful classifications and report their rates. |
| Did literature change decisions helpfully? | On matched queries, count C-to-D class changes, wrong-to-right changes, right-to-wrong changes and changes involving abstention or invalid output. Report the paired macro-F1 difference alongside the class-level results. |
| Did retrieval find usable evidence? | For cases, report how often a top-three neighbour has the query label as a simple proxy. For passages, manually judge whether at least one top-two passage addresses the query. Give an example where similarity retrieved the wrong analogue. |
| Are the explanations supported? | Preselect 12 held-out cases spanning the classes. Review both outputs for each: 24 explanations. Check numerical claims against the JSON, historical comparisons against the cited cases, and domain claims against the exact passages. Mark each checkable claim supported, unsupported or contradicted. Report counts and the fraction supported. Separately report claims needing a citation that received a supporting one. Apply source-specific checks only when that source was available. |
| What did the extra evidence cost? | Record input and output tokens, retrieval time and median end-to-end seconds per case after model warm-up. Compare C and D on the same GPU where possible. Log hardware changes and separate those timings. |

Automate checks for allowed labels, valid JSON, existing citation IDs and numeric agreement to the displayed precision. Your manual audit answers the harder question of whether the cited evidence supports what Qwen said. Report the audit size and that you rated it yourself. Select one correct example and one failure for detailed discussion after computing the overall results. A real citation paired with a false interpretation makes an informative failure case. [Citation evaluation in ALCE](https://aclanthology.org/2023.emnlp-main.398/)

## 7. Finish the assessment around your other commitments

| Checkpoint | Work to finish | Your focused contribution |
|---|---|---|
| Existing numerical foundation | Reuse the prepared case bank, classifier, split and validation artifacts. | Confirm the existing label definitions and one example. |
| C pilot first | Use the portable evidence export and `condition_C_diagnosis_colab.ipynb`; run three validation queries, then review before expanding to ten. | Inspect classifications, cited measurements, case comparisons, limitations and saved costs. |
| Extend C into D | Select permitted diagnostic literature, verify passages, build retrieval and add it to the identical C evidence. | Review source applicability and traceability. Charan remains excluded. |
| Paired validation pilot | Run C and D with matching query IDs, historical cases, generation settings and output schema. | Inspect both outputs for the same validation cases, then freeze settings. |
| Paired test | Run the same held-out queries under C and D with resumable output files. | Review the 12-case grounding sample and a failure example. |
| Report and presentation | Interpret the paired results, cost and limitations; assemble visuals and run instructions. | Explain the findings and actual peer feedback in your own words. |
| Final checks | Verify citations, GenAI acknowledgement, reproducibility and submission files. | Rehearse the talk and confirm the deliverables open. |

**Budget your attention.** Protect time for checking the results, writing and rehearsing as well as coding. Move longer sessions around CC5901 and PLC practice. Save every run so exam preparation can interrupt the project without losing completed work. Use the validation pilot to estimate the remaining machine runtime.

**If time slips**, leave zone features, model fine-tuning, forecasting, a web app, PCA, LDA and a second LLM for future work. If the validation pilot exceeds the compute budget, reduce the fixed test sample before scoring it. Protect the unseen-site split, fixed regression evidence, paired C/D conditions, grounding audit and presentation reflection. The notebook and one traceable case are enough for the demonstration.

**Use coding assistance where it saves you time.** Delegate the parser, the checks for future readings and site overlap, the regression pipeline, retrieval, resumable inference, plots and scoring summaries. Your contributions are to verify the label definition against the logs, read the source passages, judge whether the explanations are supported, and interpret what the results mean. Save a short record of GenAI contributions for the required declaration.

### Report and presentation outline

- **Report.** Target around 3,400 words within the 3,000–4,000 limit. Allocate approximately 300 to the introduction, 300 to the target problem, 450 to data and literature selection, 900 to methodology, 800 to evaluation, 500 to discussion and 150 to the conclusion. Include the project repository link if used, APA 7 references and the required GenAI declaration. Provide the accompanying code and models, including the fitted regression pipeline, Qwen identifier and revision, loader, configuration and run instructions.
- **Presentation.** Plan a 14-minute rehearsal for the 15-minute slot. Spend 2 minutes on the analyst’s decision, 2 on labels and site separation, 3 walking through one case, 3 on results and efficiency, 2 on a failure, and 2 on reflection and conclusion. In the final slides, record actual peer feedback, what you changed, why and its effect. If the seminar occurs after your personal finish date, reserve the reflection slide and finalise it after the session.
- **Evidence to have ready.** One site inclusion table, one diagram showing the separate paths for historical cases and articles, one labelled case timeline, one paired results table, one confusion matrix, and one annotated example of an unsupported claim. These can serve both the report and the slides. [Submission and presentation requirements](Problem%20Brief/CP5403_A2.pdf)

**Next checkpoint.** Follow [the Colab run guide](COLAB_RUN_GUIDE.md) and run the first three cases in `Files/condition_C_diagnosis_colab.ipynb`. The local portable export and C runner now exist. Qwen inference for C still needs validation on Colab. Literature approval, D retrieval and the final paired evaluation remain later stages. No final test outcomes are used by this pilot. Do not report completed preparation or a small validation pilot as a completed C/D experiment.
