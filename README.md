# Build an ML Pipeline for Short-Term Rental Prices in NYC

## Completed Project

This project implements a reusable machine learning pipeline for estimating short-term rental prices in New York City. The pipeline downloads and versions the source data, cleans and validates the data, creates training and test datasets, trains and compares Random Forest regression models, and evaluates the selected production model against a held-out test dataset.

The workflow uses MLflow for pipeline execution, Weights & Biases for experiment and artifact tracking, Hydra for configuration management, pytest for automated data validation, and GitHub for source control and releases.

## Project Links

- [GitHub repository](https://github.com/timwillimon/Project-Build-an-ML-Pipeline-Starter)
- [Public Weights & Biases project](https://forge.coreweave.com/wandb/timwillimon-western-governors-university/nyc_airbnb)

## Pipeline Steps

The pipeline contains the following steps:

1. `download`
   - Downloads the requested weekly data sample.
   - Stores the raw data in W&B as `sample.csv`.

2. `basic_cleaning`
   - Removes records outside the configured price range.
   - Stores the cleaned data as `clean_sample.csv`.

3. `data_check`
   - Confirms the expected columns are present.
   - Checks neighborhood values and geographic boundaries.
   - Compares the neighborhood-group distribution with the reference dataset.
   - Verifies row count and price boundaries.
   - Runs six automated pytest tests.

4. `data_split`
   - Splits the cleaned data into training/validation and test datasets.
   - Uses `neighbourhood_group` for stratification.
   - Stores `trainval_data.csv` and `test_data.csv` in W&B.

5. `train_random_forest`
   - Builds a preprocessing and inference pipeline.
   - Imputes missing numeric and categorical values.
   - Applies ordinal and one-hot encoding.
   - Creates a date-based feature from `last_review`.
   - Applies TF-IDF processing to the listing name.
   - Trains and exports a Random Forest regression model.

6. `test_regression_model`
   - Loads the model artifact carrying the `prod` alias.
   - Evaluates the production model against `test_data.csv:latest`.

## Data Validation

The completed data-check component runs six tests:

- `test_column_names`
- `test_neighborhood_names`
- `test_proper_boundaries`
- `test_similar_neigh_distrib`
- `test_row_count`
- `test_price_range`

All six tests pass for the initial data sample.

## Model Optimization

Four Random Forest parameter combinations were evaluated:

| Maximum Depth | Estimators | Validation MAE | Validation R² |
|---:|---:|---:|---:|
| 10 | 100 | 34.431949 | 0.545613 |
| 10 | 200 | 34.404576 | 0.546276 |
| 50 | 100 | 34.196322 | 0.550076 |
| 50 | 200 | 34.184256 | 0.550616 |

The selected model uses:

```yaml
n_estimators: 200
max_depth: 50
```

The selected model artifact is:

```text
model_export:prod
```

## Production Model Test

The selected production model was evaluated against the held-out test dataset.

```text
Test MAE: 33.846924076517
Test R²: 0.5619290498292977
```

## Running the Project

Create and activate the development environment:

```bash
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```

Authenticate with W&B:

```bash
wandb login
```

Run the complete default pipeline:

```bash
mlflow run .
```

Run an individual pipeline step:

```bash
mlflow run . -P steps=download
```

Run the production model test explicitly:

```bash
mlflow run . -P steps=test_regression_model
```

The production test is intentionally excluded from the default step list because a model artifact must first be reviewed and assigned the `prod` alias.

## Configuration

Pipeline parameters are maintained in `config.yaml`. This includes:

- source data sample
- accepted price boundaries
- KL-divergence threshold
- train/test and train/validation split sizes
- random seed
- stratification column
- TF-IDF feature limit
- Random Forest parameters

Configuration values are passed into the pipeline through Hydra instead of being hardcoded into the component scripts.

## Artifacts

The pipeline creates and tracks the following primary W&B artifacts:

- `sample.csv`
- `clean_sample.csv`
- `trainval_data.csv`
- `test_data.csv`
- `model_export`

The cleaned reference dataset uses the `reference` alias. The reviewed production model uses the `prod` alias.

## Release History

### Version 1.0.0

Initial completed pipeline for `sample1.csv`, including data preparation, validation, splitting, model training, hyperparameter comparison, production model selection, and held-out test evaluation.

### Version 1.0.1

Adds a geographic boundary filter that removes property records outside the expected New York City coordinate range. The released pipeline was run against sample2.csv and completed successfully, including data cleaning, validation, splitting, model training, and creation of a new trained model artifact.
