# AI Applications Project Documentation

## Project Metadata

* Project title: Used-Car Fair Price and Listing Risk Advisor
* Student: Robert Obertol
* GitHub repository URL: https://github.com/Obirobi99/used-car-listing-advisor
* Deployment URL: robertobertol.com
* Submission date: Not provided in this local copy

### Mandatory Setup Checks

* At least 2 blocks selected: Yes, ML Numeric Data and NLP
* Multiple and different data sources used: Yes, Craigslist Kaggle and Cars.com Kaggle raw CSV files are present locally
* Deployment URL provided: Yes, robertobertol.com
* Required GitHub users added to repository (`jasminh`, `bkuehnis`): Pending; add both users in the GitHub repository settings before final submission

### Current Artifact Status

This local copy has two different artifact sets:

* Legacy top-level artifact paths: `models/`, `reports/`, and `data/processed/project_listings.csv`
* Legacy top-level artifact status: demo fallback artifacts
* Legacy top-level metadata: `models/metadata.json` has `using_demo_data: true` and `processed_rows: 18`
* Real-data artifact evidence: nested artifacts exist under `models/models/` and `reports/reports/`
* Real-data metadata: `models/models/metadata.json` has `using_demo_data: false`, `processed_rows: 1124604`, `kaggle_carscom: 759356`, and `kaggle_craigslist: 365248`
* Current app runtime status: `src/inference.py` prefers the nested real-data model set when its metadata confirms `using_demo_data: false`
* Screenshot status: `screenshots/` contains placeholder notes only, not real app screenshots

## Selected AI Blocks

* ML Numeric Data
* NLP

Computer Vision was not selected for this project. The project intentionally focuses on ML Numeric Data and NLP. No image data, image preprocessing, or vision model is used.

Primary blocks used for core solution:

* Primary block 1: ML Numeric Data
* Primary block 2: NLP

---

## 1. Project Foundation

### 1.1 Problem Definition

* Problem statement: Used-car listings contain structured market signals and unstructured seller descriptions, but buyers need one combined risk-aware price recommendation.
* Goal: Predict a fair listing price and combine it with NLP text-risk analysis to recommend a negotiation range and buyer action.
* Success criteria: Runnable local pipeline, trained price model, NLP risk analysis, integrated Gradio app, saved metrics, error analysis, and reproducible instructions.

### 1.2 Integration Logic

* How the selected blocks interact: Numeric ML predicts fair price; NLP extracts risk and positive text features; final inference combines price difference and text risk.
* Data and output flow between blocks: Raw CSVs -> unified schema in `src/data_loading.py` -> numeric features in `src/preprocessing_numeric.py` -> NLP features in `src/nlp_features.py` -> integrated decision logic in `src/inference.py`.
* Current runtime note: `src/inference.py` now prefers the nested real-data artifacts in `models/models/` and falls back to top-level models only if the real-data set is unavailable.

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

Entry | Source name or link | Type | Current evidence | Role in this block
--- | --- | --- | --- | ---
1 | `austinreese/craigslist-carstrucks-data` | Kaggle CSV | Raw file present at `data/raw/kaggle_craigslist_vehicles.csv`; real-data nested metadata records 365248 processed Craigslist rows | Main structured price dataset and seller-description source
2 | `andreinovikov/used-cars-dataset` | Kaggle CSV | Raw file present at `data/raw/kaggle_carscom_used_cars.csv`; real-data nested metadata records 759356 processed Cars.com rows | Secondary structured market dataset

#### 2A.2 Preprocessing and Features

* Cleaning steps: Map raw columns to internal schema, drop invalid critical rows, validate price/year/mileage, normalize categorical text. See `src/data_loading.py` and `src/data_validation.py`.
* Preprocessing steps: Impute numeric/categorical values, scale numeric features, one-hot encode categorical features. See `src/preprocessing_numeric.py`, function `build_preprocessor()`.
* Feature engineering and selection: Create car age, mileage per year, log mileage, optional engine size, and categorical vehicle/location fields. See `src/preprocessing_numeric.py`, function `engineer_numeric_features()`.

#### 2A.3 Model Selection

* Models implemented in current code: Ridge Regression, RandomForestRegressor, ExtraTreesRegressor.
* Why these models were chosen: Ridge provides a simple baseline, while Random Forest and Extra Trees handle nonlinear interactions and can use all available CPU cores.
* Current-code evidence: `src/train_numeric.py`, function `_candidate_models()`.

#### 2A.4 Model Comparison and Iterations

Real-data nested report evidence from `reports/reports/metrics_numeric.json`:

Iteration | Objective | Key changes | Models used | Main metric | Result
--- | --- | --- | --- | --- | ---
1 | baseline structured features | mapped numeric/categorical fields plus engineered numeric features | Ridge Regression | RMSE | 7849.56
2 | nonlinear structured models | same feature set with tree ensembles | RandomForestRegressor | RMSE | 5031.49
3 | best structured model | same feature set with randomized tree splits | ExtraTreesRegressor | RMSE | 4848.90

Best structured numeric model in nested real-data report: ExtraTreesRegressor with MAE 2969.22, RMSE 4848.90, R2 0.9101.

Top-level runtime report note: `reports/metrics_numeric.json` is a demo/stale runtime artifact with 18 rows and should not be treated as final real-data evidence.

#### 2A.5 Evaluation and Error Analysis

* Metrics used: MAE, RMSE, R2.
* Legacy top-level result: demo fallback report, 18 rows, Ridge Regression best model, RMSE 3789.83, R2 0.0922.
* Real-data nested result: 240101 cleaned/evaluated rows in `reports/reports/metrics_numeric.json`; ExtraTreesRegressor best model, MAE 2969.22, RMSE 4848.90, R2 0.9101.
* Error patterns and likely causes: Rare brands/models, unrealistic prices, incomplete condition data, cross-source differences, mileage outliers, missing descriptions, older vehicles, and luxury vehicles. See `src/train_numeric.py`, saved error analysis.

#### 2A.6 Integration with Other Block(s)

* Inputs received from other block(s): NLP features can be included in the integrated price model: description length, risk keyword count, positive keyword count, text risk score, and NLP risk label.
* Outputs provided to other block(s): Predicted fair price, price difference, and price status are combined with NLP risk in `src/inference.py`, function `predict_listing()`.

### 2B. NLP

#### 2B.1 Data Source(s)

Entry | Source name or link | Type | Current evidence | Role in this block
--- | --- | --- | --- | ---
1 | `austinreese/craigslist-carstrucks-data` | Seller description text | Real-data nested NLP report records 365246 text rows | Main NLP text source
2 | manual app input | User-provided text | Runtime form field in `app.py` | Runtime risk analysis

#### 2B.2 Preprocessing and Prompt Design

* Text preprocessing: Lowercase, remove URLs and punctuation, normalize spaces. See `src/nlp_features.py`, function `clean_text()`.
* Prompt design or retrieval setup: N/A. This project uses classical NLP rather than paid LLM prompts or retrieval.
* Keyword extraction: Risk and positive keywords are extracted in `extract_keyword_features()`.
* Negation handling: Phrases such as "no accident", "accident free", "no damage", and "no repair needed" avoid false high-risk counts.
* Weak label generation: `generate_weak_risk_label()` converts text risk score into Low, Medium, or High labels.
* TF-IDF vectorization: `src/train_nlp.py` uses `TfidfVectorizer` with unigrams and bigrams.
* Generated explanation template: `generate_buyer_explanation()` explains price difference, risk terms, positive terms, and buyer caution.

#### 2B.3 Approach Selection

* Approach used: Classical NLP with negation-aware rules and TF-IDF + LogisticRegression.
* Alternatives considered: Naive keyword counts were considered as a baseline; transformer or LLM approaches were avoided to keep the project lightweight, reproducible, and free of external APIs.

#### 2B.4 Comparison and Iterations

Iteration | Objective | Key changes | Model or setup | Main metric or qualitative check | Result
--- | --- | --- | --- | --- | ---
1 | naive keyword baseline | count risk words directly | rule-based keywords | qualitative false positive checks | baseline implemented
2 | negation-aware rule-based approach | add negation and positive phrases | rules in `src/nlp_features.py` | weak risk label sanity checks | reduces false positives for "no accident"
3 | TF-IDF + LogisticRegression classifier | train classifier on weak labels | TF-IDF bigrams + LogisticRegression | accuracy and macro F1 | nested real-data report: accuracy 0.9902, macro F1 0.7404

#### 2B.5 Evaluation and Error Analysis

* Evaluation strategy: Use weak labels generated from rule-based NLP; split text rows into train/test; report accuracy, macro F1, confusion matrix, classification report, and qualitative examples.
* Legacy top-level result: demo fallback report with 12 text rows and 3 test examples.
* Real-data nested result: `reports/reports/metrics_nlp.json` records 365246 text rows, class distribution Low 299066, Medium 65971, High 209, accuracy 0.9902, macro F1 0.7404.
* Error patterns and likely causes: Weak labels are not human labels; high-risk examples are rare; keywords can mislead when wording is complex; negation phrases require special handling.

#### 2B.6 Integration with Other Block(s)

* Inputs received from other block(s): Seller description from the unified processed data schema.
* Outputs provided to other block(s): Description length, risk keyword count, positive keyword count, text risk score, NLP risk label, risk terms, and positive terms. These are used by `src/train_integrated.py` and `src/inference.py`.

### 2C. Computer Vision

Computer Vision was not selected for this project. The project intentionally focuses on ML Numeric Data and NLP. No image data, image preprocessing, or vision model is used.

#### 2C.1 Data Source(s)
N/A

#### 2C.2 Preprocessing and Augmentation
N/A

#### 2C.3 Model Selection
N/A

#### 2C.4 Model Comparison and Iterations
N/A

#### 2C.5 Evaluation and Error Analysis
N/A

#### 2C.6 Integration with Other Block(s)
N/A

---

## 3. Deployment

* Deployment URL: robertobertol.com
* Main user flow: User enters a listing, the Gradio app predicts fair price, extracts NLP risk signals, recommends an offer range, and explains the result.
* Screenshot or short demo: Not complete in this local copy; `screenshots/` contains placeholder notes only.
* Gradio app evidence: `app.py`
* Deployment packaging note: upload `app.py`, `requirements.txt`, `src/`, `models/`, `documentation.md`, and `README.md` to a Gradio Hugging Face Space after final artifacts are aligned.

---

## 4. Execution Instructions

* Environment setup: Create a Python virtual environment and install `requirements.txt`.
* Data setup: Manually download the two Kaggle CSV files and place them in `data/raw/` with the exact expected filenames.
* Training commands: Run data loading, numeric, NLP, and integrated training from the project root.
* Inference/run command: Run `python app.py`.
* Reproducibility notes: All splits use `random_state=42`; data is read only from local CSV files; scripts use project-relative paths through pathlib. Training defaults use all available CPU cores and full feature/data capacity unless environment variables intentionally cap them.
* Important current-state note: real-data artifacts currently live in nested folders such as `models/models/` and `reports/reports/`. The app inference code now prefers the nested real-data model folder when its metadata confirms that it is not demo data.

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

---

## 5. Optional Bonus Evidence

* Third selected block implemented with strong quality: N/A
* More than two data sources used with clear added value: Two different Kaggle sources are integrated with safe schema mapping.
* A core section is done exceptionally well: Integrated inference combines numeric ML, NLP risk scoring, and negotiation logic.
* Extended evaluation: Numeric metrics, NLP weak-label metrics, integrated comparison, plots, sample predictions, and qualitative examples are generated by the scripts. Real-data examples currently exist under `reports/reports/`.
* Ethics, bias, or fairness analysis: Included in `README.md` limitations and ethics notes.
* Creative or exceptional use case: Risk-aware listing advisor for a real buyer workflow.

Evidence for selected bonus items:

* More than one Kaggle source: Craigslist and Cars.com.
* Transparent weak-label limitation: documented in `src/train_nlp.py`, report files, and this file.
* Ethics and responsible-use discussion: see `README.md`, Ethics and Responsible Use.
* Extended error analysis: see `reports/reports/metrics_numeric.json`, `reports/reports/metrics_nlp.json`, and `reports/reports/metrics_integrated.json` for the nested real-data artifact set.

---

## Current Completion Checklist

* [x] Student name added
* [x] GitHub repository created and pushed
* [x] Deployment URL added
* [ ] Required GitHub collaborators added (`jasminh`, `bkuehnis`)
* [ ] Real screenshots added
* [x] ML Numeric Data code implemented
* [x] NLP code implemented
* [x] Computer Vision kept out of scope
* [x] Raw Kaggle CSV files present locally
* [x] Real-data artifacts generated at least once
* [x] Runtime inference prefers the real-data model artifact set
* [x] Final selected metrics copied into documentation
