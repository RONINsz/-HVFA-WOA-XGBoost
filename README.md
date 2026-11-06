# HVFA Backfill Strength Prediction

This repository contains the Python code for predicting slump and uniaxial compressive strength using XGBoost, support vector regression (SVR), Random Forest (RF), and WOA-XGBoost.

## Input data

Place `HVFA_data.xls` in the same directory as the script. The script also accepts `HVFA_data.xls - Sheet1.csv` as a fallback.

Required columns:

- `MC`
- `FAC`
- `SSC`
- `SSM`
- `Slump`
- `UCS_3d`
- `UCS_7d`
- `UCS_28d`

## Analysis

The dataset is divided into training and test sets using an 80:20 split. Five-fold cross-validation is performed on the training set. WOA optimizes `n_estimators`, `learning_rate`, and `max_depth` using the mean fold RMSE as the fitness value.

SVR is standardized using training-fold statistics. The other models use the original feature scales.

WOA-XGBoost feature importance is calculated from 15 model runs for each target. The mean importance and standard deviation are saved to `WOA_XGBoost_Feature_Importance.csv`.

## Installation

```bash
pip install -r requirements.txt
```

For legacy `.xls` files, `xlrd` is required.

## Run

```bash
python HVFA_WOA_XGBoost.py
```

The script saves figures in four model-specific folders and writes the WOA-XGBoost feature-importance values to a CSV file.
