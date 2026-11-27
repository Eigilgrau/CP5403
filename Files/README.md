# Code and Notebook Entry Points

Start with [REPRODUCIBILITY.md](REPRODUCIBILITY.md), then [CODE_MAP.md](CODE_MAP.md). The project root is the parent of this directory and must also contain `Datasets`, `Outputs` and `Report-Documentation`.

| Order | Primary notebook | Purpose |
|---|---|---|
| 1 | [preprocessing_experimental_data.ipynb](preprocessing_experimental_data.ipynb) | Ingestion, alignment, eligibility and site split |
| 2 | [eda_case_design.ipynb](eda_case_design.ipynb) | Training exploration and case construction |
| 3 | [case_representation_and_retrieval.ipynb](case_representation_and_retrieval.ipynb) | Numerical model selection and historical retrieval |
| 4 | [condition_CD_heldout_100_colab.ipynb](condition_CD_heldout_100_colab.ipynb) | Frozen held-out Colab workflow |

For inspection without regeneration, run `python Files/review_reproducibility.py --root .` from CP5403. The root README includes the local-environment command.

Older notebook attempts are grouped under [archive](archive/README.md). Python modules and tests retain their paths because notebooks, imports and stored hashes depend on them. The code map distinguishes final dependencies from development tools. No failed result was removed.
