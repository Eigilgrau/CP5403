# README File for CP5403 Assessment 2



### File Cloning for Evaluation



&#x09;A shared link of a Google Drive is provided to store the Datasets because these files cannot be uploaded in the repository due to the limit imposed by GitHub. Do note that the file is shared to anyone with as a Viewer which is attached as follows. Should you want to run everything, download the repository files from GitHub and the others files unincluded in the repository for validation. Contain all of the files in the same folder just as how Google Drive does. OR you can clone the whole Google Drive shared directory at your own convenience to run using Google Collab.



* **Files "Un-Uploadable" to the Repository:** https://drive.google.com/drive/folders/1JjiE95bKFnal7M-HCymiZl0BlgwVk2Dt?usp=sharing
* **GitHub Repository of Original Dataset (PHM 2015 Dataset):** https://github.com/robot007/PHM15/tree/master

### Current experiment and Colab review outputs

The active diagnostic ablation is **C (Qwen with historical cases)** versus **D (the same evidence plus literature)**. See [the active project guide](CP5403_A2_Project_Guide.md) and [experiment configuration](Files/experiment_config.json).

Use [the complete RAG + Qwen setup guide](RAG_LLM_SETUP_GUIDE.md) for the full setup, executable Colab cells, output inspection and troubleshooting. Start with [condition_C_diagnosis_colab.ipynb](Files/condition_C_diagnosis_colab.ipynb); [COLAB_RUN_GUIDE.md](COLAB_RUN_GUIDE.md) remains the short checklist. The case-only pilot uses the portable validation export and does not load literature. Charan is excluded from active processing, including previously prepared GPU jobs. D will add approved literature after C is reviewed.

Each notebook writes a separate run folder under `Outputs/runs`; open [Outputs/LATEST_RUN.md](Outputs/LATEST_RUN.md) after syncing to inspect progress and results. The C pilot saves exact prompts, raw responses, automatic checks, separate-label scoring and runtime costs. A completed validation pilot is not the completed paired test experiment. D retrieval and the final C/D evaluation remain later stages.

