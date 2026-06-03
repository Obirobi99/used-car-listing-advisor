# AI Applications Project Documentation Template

Use this template to document your project concisely and completely.
Fill in all required fields. Keep answers short and precise.

## Documentation Hint

Important:
When possible, reference the corresponding code location directly in your description.

### Example: Reference to a notebook section
Reference to the header `## Data Preprocessing` in the notebook `analysis.ipynb`:

> See *Data Preprocessing* in
> [`analysis.ipynb`](analysis.ipynb#data-preprocessing)

### Example: Reference to Python code

Reference to a single line in `model.py`, line 42:
> [`model.py`, line 42](model.py#L42)

Reference to multiple lines in `train.py`, lines 15-38:
> [`train.py`, lines 15-38](train.py#L15-L38)

## Project Metadata

- Project title: Used-Car Fair Price and Listing Risk Advisor
- Student: Robert Obertol
- GitHub repository URL: https://github.com/Obirobi99/used-car-listing-advisor
- Deployment URL: https://robertobertol.com
- Submission date: 07 June 2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [ ] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [ ] Computer Vision

Primary blocks used for core solution (choose 2):
- Primary block 1: ML Numeric Data
- Primary block 2: NLP

If a third block is selected, it is documented and graded separately as extra work.

Guidance hint: Keep the project idea short and consistent. Focus most details on the selected blocks.
Evidence hint: Show where each selected block contributes to the final system.

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition
- Problem statement: Used-car listings contain structured market signals and unstructured seller descriptions, but buyers need one combined risk-aware price recommendation.
- Goal: Predict a fair listing price from vehicle fields and combine it with NLP text-risk analysis to recommend a negotiation range and buyer action.
- Success criteria: Runnable local pipeline, trained price model, NLP risk analysis, integrated Gradio app, saved metrics, error analysis, reproducible instructions, and deployment URL.

### 1.2 Integration Logic
- How the selected blocks interact: Numeric ML predicts fair price. NLP extracts risk and positive text features. The final system combines the price prediction, seller price difference, NLP risk label, and risk score to generate offer guidance.
- Data and output flow between blocks: Raw CSVs -> unified schema in [`src/data_loading.py`](src/data_loading.py) -> numeric features in [`src/preprocessing_numeric.py`](src/preprocessing_numeric.py) -> NLP features in [`src/nlp_features.py`](src/nlp_features.py) -> integrated model in [`src/train_integrated.py`](src/train_integrated.py) -> recommendation logic in [`src/inference.py`](src/inference.py).

Guidance hint: This section should be short. The detailed work belongs in block sections.
Evidence hint: Include one clear pipeline overview.

---

## 2. Block Documentation

Complete only selected blocks. Mark non-selected block sections as N/A.

### 2A. ML Numeric Data (If selected)

#### 2A.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | `austinreese/craigslist-carstrucks-data` | Kaggle CSV, structured vehicle listings | 365,248 processed rows in `models/models/metadata.json`; 75,532 rows after numeric cleaning/evaluation in `reports/reports/metrics_numeric.json` | Main structured price dataset; provides price, year, mileage, brand/model, condition, title, drive, location, and other vehicle fields |
| 2 | `andreinovikov/used-cars-dataset` | Kaggle CSV, structured Cars.com listings | 759,356 processed rows in `models/models/metadata.json`; 164,569 rows after numeric cleaning/evaluation in `reports/reports/metrics_numeric.json` | Secondary structured market dataset; improves coverage across brands, models, and listing conditions |
| 3 | N/A | N/A | N/A | N/A |

#### 2A.2 Preprocessing and Features
- Cleaning steps: [`src/data_loading.py`](src/data_loading.py) maps raw Craigslist and Cars.com columns into one schema, converts Craigslist odometer miles to kilometers, removes invalid price/year/mileage rows, and saves processed metadata. [`src/data_validation.py`](src/data_validation.py) validates required columns, price values, year range, mileage, and text availability.
- Preprocessing steps: [`src/preprocessing_numeric.py`](src/preprocessing_numeric.py) imputes numeric values, scales numeric features with `StandardScaler`, imputes categorical values, and one-hot encodes categorical features with unknown-category handling.
- Feature engineering and selection: Numeric features include `year`, `mileage_km`, `car_age`, `mileage_per_year`, `log_mileage`, and `engine_size_l`. Categorical features include brand, model, fuel type, transmission, body type, condition, title status, drive, paint color, location region, and source name.

#### 2A.3 Model Selection
- Models tested: Ridge Regression, RandomForestRegressor, and ExtraTreesRegressor in [`src/train_numeric.py`](src/train_numeric.py).
- Why these models were chosen: Ridge Regression gives a simple linear baseline. Random Forest and Extra Trees capture nonlinear relationships between price, age, mileage, brand, model, condition, and location while remaining reproducible with `random_state=42`.

#### 2A.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Structured baseline | Unified schema, engineered numeric fields, categorical one-hot encoding | Ridge Regression | RMSE 7,849.56 | Baseline |
| 2 | Nonlinear model comparison | Same features with ensemble model | RandomForestRegressor | RMSE 5,031.49 | Improved RMSE by 2,818.07 vs Ridge |
| 3 | Best structured model | Same features with randomized tree splits | ExtraTreesRegressor | RMSE 4,848.90 | Improved RMSE by 182.59 vs Random Forest |

#### 2A.5 Evaluation and Error Analysis
- Metrics used: MAE, RMSE, and R2 on a train/test split with `random_state=42`.
- Final results: Real-data numeric report in `reports/reports/metrics_numeric.json` used 240,101 cleaned/evaluated rows. The best model was ExtraTreesRegressor with MAE 2,969.22, RMSE 4,848.90, and R2 0.9101.
- Error patterns and likely causes: Largest errors occur for rare brands/models, unrealistic seller prices, incomplete condition/title information, mileage outliers, luxury or collector vehicles, and cross-source differences between Craigslist and Cars.com rows. The report stores the top prediction errors in `reports/reports/metrics_numeric.json`.

#### 2A.6 Integration with Other Block(s)
- Inputs received from other block(s): The integrated model receives NLP-derived features from [`src/nlp_features.py`](src/nlp_features.py): description length, risk keyword count, positive keyword count, text risk score, and NLP risk label.
- Outputs provided to other block(s): The numeric model provides predicted fair price, price difference, and price status to the final recommendation logic in [`src/inference.py`](src/inference.py), which combines these values with NLP risk signals.

Guidance hint: Keep entries practical and evidence-based.
Evidence hint: Add values, not only claims.

### 2B. NLP (If selected)

#### 2B.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | `austinreese/craigslist-carstrucks-data` | Seller description text | 365,246 non-empty text rows in `reports/reports/metrics_nlp.json` | Main NLP training/evaluation source for seller-description risk analysis |
| 2 | Manual app input in [`app.py`](app.py) | User-provided listing description | One text description per app prediction | Runtime NLP risk analysis and buyer explanation |
| 3 | N/A | N/A | N/A | N/A |

#### 2B.2 Preprocessing and Prompt Design
- Text preprocessing: [`src/nlp_features.py`](src/nlp_features.py) lowercases text, removes URLs and punctuation, normalizes spaces, extracts risk/positive keyword features, and handles negated risk phrases such as "no accident", "accident free", and "no damage".
- Prompt design or retrieval setup: N/A. This project uses classical NLP and weak-label classification rather than prompts, LLM calls, or retrieval-augmented generation.

#### 2B.3 Approach Selection
- Approach used (classical NLP, transformer, RAG, prompt engineering): Classical NLP with transparent rules, negation handling, TF-IDF unigrams/bigrams, and LogisticRegression in [`src/train_nlp.py`](src/train_nlp.py).
- Alternatives considered: A naive keyword-only baseline was used first. Transformer/LLM approaches were avoided because the project should remain lightweight, reproducible, local, and free of external API dependencies.

#### 2B.4 Comparison and Iterations
| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline risk signal | Count risk words directly | Rule-based keyword counts | Qualitative false-positive checks | Baseline |
| 2 | Reduce false positives | Add negation and positive phrase handling | Rules in [`src/nlp_features.py`](src/nlp_features.py) | Known edge cases such as "no accident" and "no damage" | Improved handling of negated risk words |
| 3 | Train NLP classifier | Train on weak labels from rule-based NLP | TF-IDF bigrams + LogisticRegression | Accuracy 0.9902, macro F1 0.7404 | Adds model-based risk classification over text features |

#### 2B.5 Evaluation and Error Analysis
- Evaluation strategy: Generate weak labels from transparent NLP rules, split text rows into train/test, report accuracy, macro F1, classification report, confusion matrix, and qualitative examples.
- Results: `reports/reports/metrics_nlp.json` records 365,246 text rows with class distribution Low 299,066, Medium 65,971, and High 209. TF-IDF + LogisticRegression achieved accuracy 0.9902 and macro F1 0.7404.
- Error patterns and likely causes: High-risk examples are rare, weak labels are not human-verified ground truth, keywords can be misleading in complex wording, and the classifier inherits bias from the rule-based labels. Qualitative examples and negation checks are stored in `reports/reports/metrics_nlp.json`.

#### 2B.6 Integration with Other Block(s)
- Inputs received from other block(s): The NLP component receives seller descriptions from the unified processed listing schema created by [`src/data_loading.py`](src/data_loading.py) and manual app descriptions from [`app.py`](app.py).
- Outputs provided to other block(s): NLP outputs include description length, risk keyword count, positive keyword count, text risk score, NLP risk label, risk terms, and positive terms. These are used by [`src/train_integrated.py`](src/train_integrated.py) and [`src/inference.py`](src/inference.py) for integrated price prediction, offer range adjustment, and buyer-facing explanation.

Guidance hint: Show concrete prompt or retrieval decisions.
Evidence hint: Include representative outputs or failure cases.

### 2C. Computer Vision (If selected)

#### 2C.1 Data Source(s)
List every usage of a data source as a separate entry. If the same source is used twice for different roles, add it twice.

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | N/A | N/A | N/A | Computer Vision was not selected |
| 2 | N/A | N/A | N/A | N/A |
| 3 | N/A | N/A | N/A | N/A |

#### 2C.2 Preprocessing and Augmentation
- Image preprocessing: N/A.
- Augmentation strategy: N/A.

#### 2C.3 Model Selection
- Vision model(s) used: N/A.
- Why these model(s) were chosen: N/A.

#### 2C.4 Model Comparison and Iterations
| Iteration | Objective | Key changes | Model(s) used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | N/A | N/A | N/A | N/A | N/A |
| 2 | N/A | N/A | N/A | N/A | N/A |
| 3 | N/A | N/A | N/A | N/A | N/A |

#### 2C.5 Evaluation and Error Analysis
- Metrics and/or visual checks: N/A.
- Final results: N/A.
- Error patterns and limitations: N/A.

#### 2C.6 Integration with Other Block(s)
- Inputs received from other block(s): N/A.
- Outputs provided to other block(s): N/A.

Guidance hint: Use concise examples from real predictions.
Evidence hint: Include sample outputs and observed failure cases.

---

## 3. Deployment

- Deployment URL: https://robertobertol.com
- Main user flow: The user enters vehicle fields and a seller description in the Gradio app. The app predicts fair price, extracts NLP risk signals, calculates a recommended offer range, and returns a buyer-friendly recommendation.
- Screenshot or short demo: Screenshots will be added in the `screenshots/` folder before final submission. Current placeholder files are `screenshots/placeholder_input.md` and `screenshots/placeholder_output.md`.

Guidance hint: Deployment must be usable.
Evidence hint: Add screenshots or short demo references.

---

## 4. Execution Instructions

- Environment setup: Create a Python virtual environment and install dependencies from `requirements.txt`.
- Data setup: Manually download the two Kaggle CSV files and place them at `data/raw/kaggle_craigslist_vehicles.csv` and `data/raw/kaggle_carscom_used_cars.csv`. The raw CSV files are intentionally excluded from GitHub because they are large.
- Training command(s): Run `python -m src.data_loading`, `python -m src.train_numeric`, `python -m src.train_nlp`, `python -m src.train_integrated`, and `python -m src.evaluate` from the project root.
- Inference/run command(s): Run `python app.py` from the project root.
- Reproducibility notes: All train/test splits use `random_state=42`. Scripts use project-relative paths through `pathlib`. The app prefers the real-data artifact set under `models/models/` when `models/models/metadata.json` has `using_demo_data: false`. Large `.joblib` model binaries are excluded from GitHub and must be restored from deployment storage or regenerated by running the training commands.

Exact commands:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python -m src.data_loading
python -m src.train_numeric
python -m src.train_nlp
python -m src.train_integrated
python -m src.evaluate
python app.py
```

Guidance hint: Another person should be able to run your project from this section.
Evidence hint: Include exact commands and versions.

---

## 5. Optional Bonus Evidence

Use this section for exceptional work beyond the core requirements.

- [ ] Third selected block implemented with strong quality
- [ ] More than two data sources used with clear added value
- [x] A core section is done exceptionally well
- [x] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [x] Creative or exceptional use case

Evidence for selected bonus items:

- A core section is done exceptionally well: [`src/inference.py`](src/inference.py) combines numeric ML output, NLP risk scoring, price-difference logic, and negotiation guidance into one buyer workflow.
- Extended evaluation: Reports include numeric metrics, NLP weak-label metrics, integrated structured-plus-NLP comparison, plots, sample predictions, confusion matrix, and qualitative NLP examples under `reports/reports/`.
- Ethics, bias, or fairness analysis: `README.md` documents limitations around historical listing bias, regional coverage, weak labels, missing condition/trim information, and responsible use.
- Creative or exceptional use case: The project turns used-car listings into a practical risk-aware buyer advisor instead of only returning a raw price prediction.
