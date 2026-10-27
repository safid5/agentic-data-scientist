**CE888 REPORT**

**BY**

**SAFIYA D**

Introduction

My Offline Agentic AI (Artificial Intelligence) system works by taking a dataset in the form of a csv file as an input, and performing data science tasks on it. It autonomously performs the planning and reflection to produce the best output possible. The output is presented as a file containing a confusion matrix, human-readable report in a Markdown file and JSON files of exploratory data analysis, metrics, plan and reflection.

I have done an agentic AI system for several reasons. First, agentic AI works to correct itself when it makes mistakes rather than just spitting out false information like modern LLMs (Large Language Models). They are more reliable this way as well, as reflection reduces the risk of hallucination or incomplete information processing. It also works end-to-end to complete tasks so it can be given a lot of tasks to do and can get them done correctly and accurately. In its use cases, Agentic AI also allows for tool integration. This means that it can access the web to look for the best flights and hotels based on a budget given, for example, and then pay for it so the user does not have to.

The most interesting feature of the agentic AI data scientist is the fact that it is offline. Removing the dependency of LLM makes the system reproducible and deployable in environments without internet access.

I used seven datasets in total to test my project. They were:

- the example dataset provided in the skeleton published in GitHub.
- sales and wine quality \[1\] from an email from Haider.
- smartphone use from Kaggle (<https://www.kaggle.com/datasets/jimarahman/smartphone-usage-and-addiction-analysis-dataset?resource=download>).
- most viewed videos from Kaggle (https://[www.kaggle.com/datasets/mubashirsidiki/most-watched-yt-videos-rankings-2026](http://www.kaggle.com/datasets/mubashirsidiki/most-watched-yt-videos-rankings-2026)).
- songs from Kaggle (<https://www.kaggle.com/datasets/moonknightmarvel/dataset-of-songs-with-genreartistmovielanguage>).
- world happiness from Kaggle (<https://www.kaggle.com/datasets/obaidhere/world-happiness-report>). All the dataset files are CSVs (comma separated values).

In the report, I cover the main files that make up the agentic data scientist along with the most important features I coded. I will discuss what those features do and how the agent learns from itself. Lastly, I will also publish some results from my tests using the datasets I listed above and cover the ethics and limitations of the agentic system.

System Architecture

The original skeleton \[2\] in the GitHub is as follows:

ce888-agentic-data-scientist/

│

├── README.md # This file

├── requirements.txt # Python dependencies

├── .gitignore # Git ignore rules

│

├── agentic_data_scientist.py # Core agent (Executor) - extend this

├── run_agent.py # Entry point - students run this

│

├── agents/ # Agent components

│ ├── \__init_\_.py

│ ├── planner.py # TODO: EXTEND THIS

│ ├── reflector.py # TODO: EXTEND THIS

│ └── memory.py # Basic implementation - can extend

│

├── tools/ # Data science tools

│ ├── \__init_\_.py

│ ├── data_profiler.py # Provided - can extend

│ ├── modelling.py # Provided - can extend

│ └── evaluation.py # Provided - can extend

│

├── data/ # Datasets

│ ├── README.md # Dataset documentation template

│ └── example_dataset.csv # Small demo dataset

│

├── outputs/ # Generated outputs (gitignore)

│ └── .gitkeep

│

├── report/ # Your technical report

│ ├── README.md # Report guidelines

│ └── REPORT.md # TODO: WRITE YOUR REPORT HERE

│

└── tests/ # Test suite

├── \__init_\_.py

└── sanity_check.py # Basic sanity test

Mine is slightly different. It is:

ce888-agentic-data-scientist/

│

├── README.md # This file

├── requirements.txt # Python dependencies

├── .gitignore # Git ignore rules

│

├── agentic_data_scientist.py # Core agent (Executor)

├── run_agent.py # Entry point

│

├── agents/ # Agent components

│ ├── \__init_\_.py

│ ├── planner.py

│ ├── reflector.py

│ └── memory.py

│

├── tools/ # Data science tools

│ ├── \__init_\_.py

│ ├── data_profiler.py

│ ├── modelling.py

│ └── evaluation.py

│

├── data/ # Datasets

│ ├── README.md # Dataset documentation template

│ └── example_dataset.csv # Small demo dataset

│

├── outputs/ # Generated outputs (gitignore)

│ └── .gitkeep

│

├── report/ # technical report

│ ├── README.md # Report guidelines

│ └── REPORT.md

│

└── tests/ # Test suite

├── \__init_\_.py

└── sanity_check.py # Basic sanity test

├── test_agentic_ds.py

├── test_memory.py

└── test_planner.py

├── test_profiler.py

├── test_reflector.py

└── test_smoke_run.py

The agents file is the most important, as it holds all the rules the agent will use to process the data.

**Planner**

The Planner file tells the agent what to do depending on the dataset characteristics, type, and size. The function create_plan() is the main one. It reads the dataset profile and applies the template depending on the number of rows in the dataset. If they are less than 800, we use the small dataset template. For rows over 20,000 then we use the large dataset template and depending on the level of imbalance, we use the high dimensional plan, imbalanced dataset template and if nothing fits those conditions, we use the standard dataset template.

Once we have our template, we add SMOTE (Synthetic Minor Oversampling Technique), class weight steps, target encoding, regularisation or missing data handlers. These features are applied depending on the imbalance, missing values and/or high cardinality categoricals in the dataset given.

**Memory**

The memory file saves records every time the agent runs so it can learn from its previous runs. This is one of the main features of an Agentic AI - its ability to learn from itself. In this case, we are saving the JSON files that are produced in the output folders.

The main class, named JSONMemory, is where we have most of the functions. The functions initialise the memory system and allow us to read and save JSON files. Other functions can let us lookup specific datasets by their unique fingerprint. We can add notes to the dataset and save them structurally by including a summary of the dataset. There are also functions that allow us to find similar datasets and others that count the models that have been the best and how many times each step has failed across datasets.

Outside the main class, there is a function that returns the current time to timestamp memory records. Another function calculates the similarity score between two dataset dictionaries.

**Reflector**

This file evaluates the agents execution results, identifies issues and suggests improvements if they are necessary.

The main function takes the metrics of the best model identified by the agent and produces a structured assessment. It compares the best model's accuracy against a dummy baseline and flags weak improvement. If weak improvement is detected, an adaptive F1 threshold is applied that gives us imbalance specific suggestions by calling the helper functions and computing a trigger score to decide if replanning is necessary. The helper functions detect data quality issues, compares models statistically, detects overfitting, analyses class performance, analyses precision-recall tradeoff and prioritises certain suggestions with high priority keywords. Once all that is done, it returns a dictionary with status, best model, issues, suggestions and a boolean of true or false for replan recommended.

The function apply_replan_strategy() modifies the plan and profile based on the issues identified in the main function. For underfitting issues, we use ensemble methods, for imbalance issues, we use SMOTE oversampling, for overfitting issues, we use regularisation and for weak baseline gap issues we use feature engineering. After that, we append a human-readable note to the dataset profile notes list of the changes made.

The detect_data_quality_issues() function checks for structural problems such as a small row count, more columns than rows or columns that have a high cardinality. It returns a list of issues and suggestions.

**Agentic Data Scientist**

This is the executor file. It handles the main features of the agentic AI: dataset loading, planning training, evaluation and reflection.

This file is where we set up the agent with a memory instance and print timestamped messages in the terminal. We load our csv file and validate it before logging the shape of the dataset. We set the agent to execute a max of 3 times if steps fail.

The main function creates a unique run ID and output directory. It loads and profiles the dataset, checks the memory file for an exact fingerprint match and if one isn't found, a plan is made via the planner file. After that, we enter a while true loop that builds the preprocessor, selects models, trains them, evaluates, executes, reflects and saves the artefacts to disk. After that, it updates the memory and exits if no replan is needed or max replans are reached. If a replan is needed, the apply replan strategy is used and loops again. Once that is done, the output directory path is returned.

Dataset Understanding

The profiler file in the tools folder extracts signals from the datasets input by the user. The signals extracted help the agent analyse the data and determine the best model to use for data analysis.

The function infer_target_column() looks at which column is likely the target column if the agent is asked to choose a target column and isn't given one.

Other functions also look at the shape of the dataset which is the number of rows and columns. That information determines which dataset template the planner file should use.

Missing data is also extracted. This helps the planner determine if the missing data is severe or needs to be imputed.

Several feature signals are extracted as well. These include which features are numeric, categorical and how many there are of each. This helps the planner with model selection and preprocessing choices.

High cardinality categoricals are also extracted which affects the preprocessor build and the level of imbalance affects the features applied to the dataset analysis in the planner file.

The profiler also extracts the dataset fingerprint which is used by the memory file to store it in the system and look it up if needed.

Planning Logic

The main feature of the planner file is the conditional statement and how the features of the dataset affect which template is used.

If a dataset has less than 800 rows, the small dataset template is called. No additional preprocessing steps are embedded in this template.

If a dataset has over 20,000 rows, the large dataset template is used. There are no additional preprocessing steps embedded in this template as well.

If there is an imbalance of 3 and up along with column number of over 50, the high dimensional template is called. The template has feature selection and PCA (Principal Component Analysis) reduction included as high dimensional datasets risk having severe imbalance and too many features.

If there is an imbalance of 3 and up, the imbalanced dataset template is called. The class weights, SMOTE oversampling and F1 evaluation are called as severe class imbalance is the most likely risk with imbalanced datasets.

The default dataset template is the standard one.

Depending on the dataset signals, SMOTE, class weights, regularisation, etc will be applied which can be necessary as the small and large dataset template do not have any feature calling.

Tool Use: Modelling and Evaluation

For evaluation, the best model is selected. It is the model with the highest metric the agent uses to rank the models. I used balanced accuracy and macro F1 over plain accuracy as they treat all the classes equally. Balanced accuracy averages recall across classes while macro f1 averages f1 across classes. Balanced accuracy is also robust to imbalanced data, while macro F1 helps maintain a balanced performance and stable precision across the datasets.

The dummy baseline is used to compare the best model against an intentionally simple one. Doing this helps us detect when the model might have uninformative features or a noisy target. It also helps with preventing false confidence as if the model and dummy have the same accuracy then we know the model is bad. Comparing the two also helps us know if we need to trigger a replan with the computed trigger score.

Reflection and Re-Planning

The reflector file calculates a trigger score to know if we need to trigger a replan. If the trigger score is equal to 3 and up then the agent knows to apply the replan strategy function. A replan is triggered when multiple issues are detected, there is overfitting or if the best model doesn't/barely beats the dummy baseline.

The replan strategy function adjusts the execution plan based on the issues detected. If underfitting is detected, ensemble methods such as RandomForest or GradientBoosting will be applied. If imbalance is detected, SMOTE will be applied. If overfitting is detected, regularisation will be applied. If the baseline gap is weak, feature engineering will be applied. Finally, the notes will be updated to add the replan.

We can present this in an example. Let's say we have a dataset with a profile of 10,000 rows, 80 columns and imbalance ratio of 4.0. This profile will have the planner choose the high dimensional template which is set to add SMOTE, regularisation and class weights to the data analysis.

The results of this will present the best model with metrics of a balanced accuracy of 0.58, macro f1 of 0.49, train f1 of 0.82, precision of 0.70 and recall of 0.40. The dummy baseline produces a result of 0.52. The reflector file finds 5 issues with the overfitting adding 2 points and the baseline adding 1 more point. This makes the trigger score 8. It is more than 3 so the replan function is called. The overfitting, weak baseline gap, underfitting and imbalance issues makes us add regularisation, feature engineering and SMOTE before the train models step along with ensemble methods.

This turns our execution plan from:

profile_dataset, build_preprocessor, apply_feature_selection, apply_pca_reduction, select_models, train_models, evaluate, reflect, write_report, fallback_strategy

To

profile_dataset, build_preprocessor, apply_feature_selection, apply_pca_reduction, select_models, apply_feature_engineering, apply_regularisation, apply_smote_oversampling, train_models, evaluate, reflect, write_report, apply_ensemble_methods, fallback_strategy.

Memory and Learning

In the memory file, the function memory.store_experience() function writes a dataset record into the JSON class. The dataset fingerprint, timestamp of the run, summary, full execution plan, best model, evaluation metrics, reflection status, failed steps or whether replanning was done gets saved in the dataset record.

The similarity score function compares two dataset summaries and returns a number between 0 and 1. We use this function to know if we need to use a memory hint. The memory hint identifies which dataset performed best by loading the most similar profile and prioritising the step that performed best in the similar dataset in the new plan. This allows the agent to learn from performance.

Results

| Dataset            | Rows | Best Model         | Target                | Balanced Accuracy | F1 Macro | Replan Recommended |
| ------------------ | ---- | ------------------ | --------------------- | ----------------- | -------- | ------------------ |
| Example_Dataset    | 20   | LogisticRegression | Auto - Label          | 1.00              | 1.00     | false              |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| Sales              | N/A  | N/A                | N/A                   | N/A               | N/A      | N/A                |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| WineQuality        | N/A  | N/A                | N/A                   | N/A               | N/A      | N/A                |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| Smartphone_use     | 7500 | GradientBoosting   | Auto - Addicted Label | 1.00              | 1.00     | false              |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| most_viewed_videos | 1000 | DummyMostFrequent  | Manual - rank         | 0.00              | 0.00     | true               |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| most_viewed_videos | 1000 | SVC-RBF            | Manual - content_type | 0.336             | 0.332    | true               |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| songs              | 101  | LogisticRegression | Manual - emotion      | 0.196             | 0.187    | true               |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| songs              | 101  | RandomForest       | Manual - language     | 0.667             | 0.708    | true               |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |
| word_happiness     | 152  | DummyMostFrequent  | Manual - Country      | 0.00              | 0.00     | true               |
| ---                | ---  | ---                | ---                   | ---               | ---      | ---                |

Above is my results table. The example dataset has a balanced accuracy and F1 macro of 1.00 which shows that the agent works end-to-end because the dataset is small and simple.

The sales and wine quality datasets failed when they were run and generated empty output directories, so I put "N/A" in the table. This is due to the infer_target_column() function not working well as it seems to only recognise five column name patterns and falls back to the last column if cardinality is sufficiently low.

For the smartphone_use dataset, the agent gives a perfect score for balanced accuracy and F1 macro. This is bad because the target column Addicted Label is too easy because it has a near-perfect correlation with other columns in the dataset, or there is data leakage.

With the most_viewed_videos dataset, the agent at first doesn't work well with rank chosen as the target column. This is due to there being no learnable signal between the features and the target. The agent, however, works better with content_type chosen as the target column. It produces slightly better results than before and the model correctly identifies that replanning needs to be done.

For the songs dataset, emotion as a target column gives us a low performance as seen in the low values for balanced accuracy and F1 macro. This is because using emotion for classification is difficult. Using language, however, is much better. This is because there are fewer classes and clearer patterns. The agent still recommends replanning because the value for F1 macro is less than 0.75.

For the world happiness dataset, the agent produces values 0. This is because using country as a target column has too many unique values, causing the agent to abort the run.

Ethics and Limitations

**Ethics**

The agent uses balance accuracy and macro f1 to avoid favouring majority classes which is ethically important.

Automated decisions like model selection and preprocessing, which provide explanations and suggestions but the system still relies on heuristics rather than fully interpretable reasoning. This means that agents' decisions are assistance and not absolute truth which should be understood by the user and advised to validate results.

The risk of privacy is reduced as the system stores dataset fingerprint and performance summaries. However, users should still ensure columns and metadata do not include sensitive information.

The agent is also transparent. In the output file, there is a Markdown file with a human-readable report so the user can see the decisions the agent chose to make. Although this is good, automated preprocessing decisions should always be reviewed before deployment in sensitive domains, especially if the dataset being analysed has sensitive information.

**Limitations**

A limitation the system possesses is that the system operates offline. If an LLM was allowed to be integrated, the AI system would work more efficiently and quickly as it would have access to the internet. Most importantly, the agent will have access to even more information and datasets that it could learn from.

Another limitation is that techniques like SMOTE can introduce synthetic bias if misused, so human oversight is still needed to validate results.

The models used by the agent are trained with default parameters which can limit performance and makes the system less competitive against modern AutoML frameworks.

Replanning also does not tune hyperparameters, run multiple adaptive replan cycles or modify preprocessing pipeline in a principled way. These features would make the system work more efficiently.

The planner and reflector files rely on fixed thresholds and simple rules but there is no statistical testing, uncertainty estimation or adaptive thresholding. Because the thresholds are fixed, a dataset that is borderline (e.g: 802 rows) will call the standard dataset template rather than the small one. This makes the system less optimal.

Another limitation is in the similarity score. It uses only four dimensions so it could miss important structural differences between datasets.

The main limitation I discovered while testing was with the target quality. It is what determines how well the agent will run. There is no way the agent can validate whether a chosen target makes sense as a classification label.

Conclusion

The Agentic Data Scientist works end-to-end by successfully performing: dataset profiling, plan generation, preprocessing, model selection and training, evaluation with robust metrics, reflection with issue diagnosis, replanning and memory-guided adaptation.

My agentic system works well by being consistent with its design. It chooses the right templates, uses robust metrics and applies replans when appropriate.

The agent learning from previous runs through the memory file is also a main feature of an agentic system along with a multi signal reflector. It analyses the dummy baseline improvement, macro f1, precision-recall tradeoff, overfitting, etc. this is more sophisticated than a plain accuracy check.

For future improvements, some features can be added to really improve how the agentic system works. Adding statistical significance testing such as t-tests or Wilcoxon ranked tests to compare models will help the agent determine if performance differences are meaningful or just noise. An enhanced memory similarity scoring system using feature distribution, PCA embeddings and model performance patterns. It will allow the agent to learn which preprocessing steps consistently work, which strategies fail and the best models for particular dataset types.

Bibliography

\[1\] Raza, H. "CE888 - First Submission Approaching: Important Guidance" Personal email (01 February 2026)

\[2\] Raza, H. "ce888-agentic-data-scientist" [GitHub.com](http://github.com). \[Online\]. Available: <https://github.com/sagihaider/ce888-agentic-data-scientist> \[Accessed 09 February 2026\]