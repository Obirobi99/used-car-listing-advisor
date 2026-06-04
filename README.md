# Used-Car Fair Price and Listing Risk Advisor

## Project Overview

This project builds a local AI application that helps a buyer evaluate a used-car listing. It predicts a fair market price from structured vehicle data and analyzes the seller description for text-based risk and positive signals. The final advisor combines both outputs into a price status, NLP risk level, negotiation range, and buyer-friendly explanation.

## Selected AI Blocks

- ML Numeric Data: regression models predict used-car seller price from structured listing fields.
- NLP: rule-based keyword extraction with negation handling and a TF-IDF + LogisticRegression classifier analyze seller descriptions.
- Computer Vision: not selected and not used.

## Current Local Status

The implementation for data loading, numeric ML, NLP, integrated inference, evaluation scripts, a Gradio app, and a Colab workflow is present.

Important artifact note:

- The app now prefers the real-data model files under `models/models/` when `models/models/metadata.json` says `using_demo_data: false`.
- The top-level `models/` and `reports/` files still describe an older demo fallback run: `models/metadata.json` has `using_demo_data: true` and `processed_rows: 18`.
- The real-data artifact set exists under `models/models/` and `reports/reports/`: `models/models/metadata.json` has `using_demo_data: false`, `processed_rows: 1124604`, with 759356 Cars.com rows and 365248 Craigslist rows.
- The top-level `data/processed/project_listings.csv` currently contains the 18-row demo dataset.
- The `screenshots/` folder should contain final app screenshots before submission.
- Student name is Robert Obertol.
- GitHub repository: `https://github.com/Obirobi99/used-car-listing-advisor`.
- Required GitHub collaborators (`jasminh`, `bkuehnis`) still need to be added in the GitHub repository settings before final submission.
- Deployment URL: `robertobertol.com`.

## Why the Use Case Is Realistic

Used-car buyers often compare incomplete listings across platforms. Price alone is not enough: a cheap car with "oil leak" and "sold as is" can be riskier than a slightly higher-priced car with service history and a clean title. This project integrates structured market data and NLP risk signals into one decision workflow.

## Data Sources

Main source:

- Kaggle slug: `austinreese/craigslist-carstrucks-data`
- Original file: `vehicles.csv`
- Expected local path: `data/raw/kaggle_craigslist_vehicles.csv`
- Role: main structured numeric ML source and seller-description NLP source

Secondary source:

- Kaggle slug: `andreinovikov/used-cars-dataset`
- Expected local path: `data/raw/kaggle_carscom_used_cars.csv`
- Role: secondary structured numeric source for cross-source comparison and additional market rows

The current local copy includes both raw CSV files. The project does not download data automatically, does not use the Kaggle API, does not use Hugging Face datasets, does not scrape websites, and does not call external APIs.

## Data Preparation

Run from the project root:

```bash
python -m src.data_loading
```

The script reads local CSV files from `data/raw/`, maps both datasets into a unified schema, converts Craigslist odometer miles to kilometers, validates core fields, saves `data/processed/project_listings.csv`, and writes metadata.

If both raw CSV files are missing, a tiny demo fallback dataset is created only so the app and scripts can launch. This fallback is clearly marked in the console and in metadata and is not valid for final submission.

## Repository Structure

```text
used-car-listing-advisor/
  app.py
  requirements.txt
  README.md
  documentation.md
  colab_used_car_advisor.ipynb
  data/
    raw/
    processed/
  models/
  reports/
    figures/
  screenshots/
  src/
```

## Local Setup

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

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Train Models

Place the manually downloaded CSV files in `data/raw/`, then run these commands from the project root:

```bash
python -m src.data_loading
python -m src.train_numeric
python -m src.train_nlp
python -m src.train_integrated
```

Training defaults are set for a high-performance VM: parallel estimators use all CPU cores, numeric training uses the full processed dataset, one-hot category caps are disabled, and TF-IDF uses the full vocabulary. To cap resource use, set environment variables such as `TRAIN_N_JOBS`, `MAX_TRAIN_ROWS`, `RANDOM_FOREST_N_ESTIMATORS`, `LOGISTIC_REGRESSION_MAX_ITER`, `ONE_HOT_MAX_CATEGORIES`, `HIGH_CARDINALITY_MAX_CATEGORIES`, or `TFIDF_MAX_FEATURES`.

## How to Run Evaluation

```bash
python -m src.evaluate
```

This checks expected model/report files, evaluates the available price model, and updates `reports/sample_predictions.csv`.

## Current Metrics Evidence

Top-level artifacts still reflect an older demo fallback run:

- Top-level metadata: `using_demo_data: true`, `processed_rows: 18`.
- Top-level numeric best model: Ridge Regression, RMSE 3789.83, R2 0.0922.
- Top-level integrated best model: RandomForestRegressor, RMSE 6176.31, R2 -1.4110.
- Top-level NLP report uses 12 text rows.

Real-data artifacts exist in nested output folders:

- `models/models/metadata.json`: `using_demo_data: false`, `processed_rows: 1124604`.
- `reports/reports/metrics_numeric.json`: best numeric model ExtraTreesRegressor, MAE 2969.22, RMSE 4848.90, R2 0.9101.
- `reports/reports/metrics_nlp.json`: 365246 text rows, accuracy 0.9902, macro F1 0.7404.
- `reports/reports/metrics_integrated.json`: best integrated model ExtraTreesRegressor, MAE 2516.59, RMSE 4349.06, R2 0.9288.

The app runtime now prefers this nested real-data model set before falling back to the top-level demo models.

## How to Run the App

```bash
python app.py
```

If models are missing, the app returns: `Models not found. Please run the training scripts locally first.`

With the current code, the app loads the real-data model set from `models/models/` and should not report the demo fallback metadata warning unless those real-data files are removed.

## Google Colab Notebook

The repository includes `colab_used_car_advisor.ipynb`. Open it in Colab after uploading or cloning the full project folder. The notebook installs requirements, lets you manually upload the two Kaggle CSV files into `data/raw/`, runs the same training/evaluation scripts, and can optionally launch the Gradio app.

The notebook follows the same data rule as the scripts: it does not download Kaggle data automatically.

## Server Deployment

Deployment URL: `robertobertol.com`.

To deploy after training, upload or pull these files/folders to the server:

- `app.py`
- `requirements.txt`
- `src/`
- `models/`
- `documentation.md`
- `README.md`

The server should use saved model files. It should not train or download datasets during normal web inference.

## Example User Flow

1. User enters listing fields such as brand, model, year, mileage, condition, seller price, and description.
2. The app predicts a fair market price with the saved price model.
3. The NLP component extracts risk terms, positive terms, weak risk score, and risk label from the seller description.
4. Integrated decision logic compares seller price to predicted price and adjusts the recommended offer range based on NLP risk.
5. The app returns a buyer-friendly explanation.

## Limitations

- The app is not a professional vehicle inspection.
- Price prediction depends on historical listing data.
- Seller descriptions may be incomplete, misleading, or strategically written.
- Weak NLP labels are approximate and not human-labeled ground truth.
- Cross-country and cross-currency comparison is limited.
- Craigslist and Cars.com data may not represent European or Swiss used-car markets.
- Vehicle trim, maintenance records, accident history, and local market seasonality are only partially represented.
- The current local artifact layout contains both old top-level demo artifacts and nested real-data artifacts. Runtime inference now prefers the nested real-data artifacts.

## Ethics and Responsible Use

The advisor should support, not replace, human judgment. It may reflect biases in historical listings, platform coverage, regional pricing, and seller language. Buyers should verify documents, arrange an independent inspection, and avoid treating model output as financial or legal advice.

## Current Submission Checklist

- [x] Add student name.
- [x] GitHub repository created and pushed.
- [x] Add deployment URL.
- [ ] Add required GitHub collaborators (`jasminh`, `bkuehnis`).
- [ ] Add real screenshots.
- [x] Implement ML Numeric Data block.
- [x] Implement NLP block.
- [x] Keep Computer Vision out of scope.
- [x] Include manually provided local Kaggle CSV files.
- [x] Generate at least one real-data model/report artifact set.
- [x] Make runtime inference prefer the real-data model artifact set.
- [x] Paste final selected metrics into the documentation.
