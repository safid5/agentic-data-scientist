# Offline Agentic Data Scientist

An autonomous, fully offline agent that performs end-to-end classification on a CSV dataset it has never seen. It profiles the data, plans a workflow, trains and evaluates models, reflects on its own results, re-plans if they fall short, and remembers what worked for next time. No LLMs or internet access are needed: all reasoning is rule-based, using heuristics and meta-learning.

Built in Python with scikit-learn and pandas for the CE888 module (University of Essex). The starter skeleton came from Sagi Haider (MIT licence, see [Credits](#credits)). The planning logic, reflection, memory, error handling and test suite are my own work.

## How it works

```
CSV + target (or "auto")
        |
        v
  Profile dataset ──> Check memory for a matching dataset fingerprint
        |                          |
        v                          v
   Plan (template + extras, guided by memory hints)
        |
        v
  Preprocess ──> Select models ──> Train ──> Evaluate vs dummy baseline
        |
        v
     Reflect ──> issues found? ──yes──> Re-plan (up to max_replans) ──┐
        |                                                             |
        no <──────────────────────────────────────────────────────────┘
        |
        v
  Save report, metrics, plan, reflection + update memory
```

| Component | File | What it does |
|---|---|---|
| Executor | `agentic_data_scientist.py` | Runs the loop, handles retries (max 3 per step) and logging, writes outputs |
| Profiler | `tools/data_profiler.py` | Extracts shape, missing values, feature types, class imbalance, high-cardinality columns and a dataset fingerprint; infers the target when none is given |
| Planner | `agents/planner.py` | Chooses a plan template from the profile and adds steps such as SMOTE, class weights, regularisation, feature selection and PCA |
| Reflector | `agents/reflector.py` | Compares the best model with a dummy baseline, checks for overfitting, imbalance and weak per-class performance, computes a trigger score and decides whether to re-plan |
| Memory | `agents/memory.py` | Stores each run as JSON, finds similar past datasets with a similarity score, and feeds the best-performing steps back into new plans |
| Modelling / evaluation | `tools/modelling.py`, `tools/evaluation.py` | Pipelines for Logistic Regression, Random Forest, Gradient Boosting and SVM, ranked by balanced accuracy and macro F1 |

**Planning rules.** Fewer than 800 rows uses the small-data template and more than 20,000 uses the large-data template. Class imbalance of 3 or more selects the imbalanced template, or the high-dimensional template (feature selection and PCA) if there are also more than 50 columns. Anything else uses the standard template.

**Reflection and re-planning.** The reflector adds up issues into a trigger score, and a score of 3 or more triggers `apply_replan_strategy()`. Underfitting adds ensemble methods, imbalance adds SMOTE, overfitting adds regularisation and a weak gap to the baseline adds feature engineering. Each change is written into the plan notes so the run stays explainable.

**Metrics.** Models are ranked on balanced accuracy and macro F1 rather than plain accuracy, so minority classes count as much as majority ones.

## Quick start

```bash
git clone https://github.com/safid5/agentic-data-scientist.git
cd agentic-data-scientist

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run_agent.py --data data/example_dataset.csv --target auto
```

Options: `--target <column>` or `auto`, `--output_root outputs`, `--seed 42`, `--test_size 0.2`, `--max_replans 1`, `--quiet`.

Each run creates a timestamped folder in `outputs/` containing a confusion matrix, a human-readable Markdown report, and JSON files for the EDA, metrics, plan and reflection.

Run the tests with:

```bash
pytest -q
```

Tests cover the memory, planner, profiler, reflector, a full smoke run and the executor. CI runs them on every push (Python 3.11).

## Results

I tested the agent on seven datasets. The full write-up is in [`report/REPORT.md`](report/REPORT.md). Only the small demo datasets are included here; the others are linked below.

| Dataset | Rows | Target | Best model | Balanced acc. | Macro F1 | Re-plan recommended |
|---|---|---|---|---|---|---|
| Example (module demo) | 20 | `label` (auto) | Logistic Regression | 1.00 | 1.00 | No |
| [Smartphone usage](https://www.kaggle.com/datasets/jimarahman/smartphone-usage-and-addiction-analysis-dataset) | 7,500 | Addicted label (auto) | Gradient Boosting | 1.00 | 1.00 | No |
| [Songs](https://www.kaggle.com/datasets/moonknightmarvel/dataset-of-songs-with-genreartistmovielanguage) | 101 | `language` | Random Forest | 0.667 | 0.708 | Yes |
| Songs | 101 | `emotion` | Logistic Regression | 0.196 | 0.187 | Yes |
| [Most viewed videos](https://www.kaggle.com/datasets/mubashirsidiki/most-watched-yt-videos-rankings-2026) | 1,000 | `content_type` | SVC (RBF) | 0.336 | 0.332 | Yes |

**What the results show**

- **The pipeline works end to end.** On the demo data it profiles, plans, trains, reflects and writes a report with no manual steps.
- **Target quality drives everything.** Predicting song `language` (few, distinct classes) worked far better than `emotion`. In both cases the reflector correctly flagged weak results and recommended re-planning.
- **Perfect scores are a warning, not a win.** The 1.00 on the smartphone dataset most likely reflects a target almost perfectly determined by other columns (possible leakage), not a strong model.
- **Failures are documented.** With `rank` (videos) as the target the best model was just the dummy baseline, because there was no learnable signal. With country (world happiness) as the target, the very high number of unique values made the agent abort. The sales and wine-quality datasets (provided by the module) produced empty outputs because target inference failed on them.

## Limitations and next steps

- **No validation of the target.** The agent cannot tell whether a chosen column makes sense as a classification label, which was the main weakness found in testing.
- **Target inference is narrow.** `infer_target_column()` recognises only a handful of column-name patterns and otherwise falls back to the last column.
- **Default hyperparameters.** Models are not tuned, and re-planning does not tune them either, so the system is not competitive with AutoML tools.
- **Fixed thresholds.** Template and trigger thresholds are hard-coded, so a borderline dataset (for example 802 rows) gets the standard template rather than the small one. There is no statistical testing or uncertainty estimation.
- **Simple memory similarity.** The dataset similarity score uses four dimensions only.
- **Planned improvements:** significance tests (t-test or Wilcoxon) when comparing models, richer similarity using feature distributions and PCA embeddings, hyperparameter search inside re-planning, and a check for suspiciously perfect scores.

## Project structure

```
agentic-data-scientist/
├── agentic_data_scientist.py   # Executor
├── run_agent.py                # CLI entry point
├── agents/                     # planner, reflector, memory
├── tools/                      # data_profiler, modelling, evaluation
├── data/                       # small demo datasets
├── report/REPORT.md            # Technical report
├── tests/                      # Unit, smoke and sanity tests
└── .github/workflows/ci.yml    # CI: runs pytest
```

## Ethics

Results are decision support, not ground truth: the system relies on heuristics, and its preprocessing choices should be reviewed before use in sensitive domains. Memory stores only dataset fingerprints and performance summaries, but users should still make sure columns and metadata contain no sensitive information.

## Credits

Starter skeleton and assignment brief by the CE888 professor Sagi Haider, released under the MIT licence (see `LICENSE`). Datasets as linked above. All other code, tests and the report are my own.

Safiya Dikko · [LinkedIn](https://linkedin.com/in/safiya-dikko)
