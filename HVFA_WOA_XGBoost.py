# -*- coding: utf-8 -*-

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


warnings.filterwarnings("ignore")
np.random.seed(42)

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False

FEATURE_COLUMNS = ["MC", "FAC", "SSC", "SSM"]
TARGET_COLUMNS = ["Slump", "UCS_3d", "UCS_7d", "UCS_28d"]
REQUIRED_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS

N_FOLDS = 5
IMPORTANCE_REPEATS = 15
OUTPUT_DIRECTORIES = {
    "XGBoost": "XGBoost_HVFA",
    "SVR": "SVR_HVFA",
    "RF": "RF_HVFA",
    "WOA-XGBoost": "HVFA_WOA_XGBoost",
}


def load_dataset():
    if os.path.exists("HVFA_data.xls"):
        data = pd.read_excel("HVFA_data.xls")
    elif os.path.exists("HVFA_data.xls - Sheet1.csv"):
        data = pd.read_csv("HVFA_data.xls - Sheet1.csv")
    else:
        raise FileNotFoundError(
            "The dataset file 'HVFA_data.xls' or its CSV equivalent was not found."
        )

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = data[REQUIRED_COLUMNS].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=REQUIRED_COLUMNS)
    data = data.drop_duplicates(subset=REQUIRED_COLUMNS).reset_index(drop=True)

    if len(data) < N_FOLDS:
        raise ValueError(
            f"At least {N_FOLDS} complete records are required for cross-validation."
        )

    return data


def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return [
        r2_score(y_true, y_pred),
        np.sqrt(mse),
        mse,
        mean_absolute_error(y_true, y_pred),
    ]


def report_cross_validation(model_name, target, y_train, oof_predictions, fold_rmses):
    metrics = calculate_metrics(y_train, oof_predictions)
    fold_mean = np.mean(fold_rmses)
    fold_std = np.std(fold_rmses, ddof=1)

    print(
        f"{model_name} | {target} | "
        f"5-fold CV: R2={metrics[0]:.4f}, "
        f"RMSE={metrics[1]:.6f}, "
        f"MSE={metrics[2]:.8f}, "
        f"MAE={metrics[3]:.6f}, "
        f"fold RMSE={fold_mean:.6f} ± {fold_std:.6f}"
    )

    return metrics


def make_standard_xgb():
    return xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.035,
        max_depth=2,
        min_child_weight=3,
        reg_lambda=5.0,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
    )


def make_svr():
    return SVR(
        kernel="rbf",
        C=0.3,
        epsilon=0.45,
        gamma=0.05,
    )


def make_random_forest():
    return RandomForestRegressor(
        n_estimators=400,
        max_depth=3,
        min_samples_leaf=4,
        min_samples_split=10,
        max_features=0.7,
        random_state=42,
        n_jobs=-1,
    )


def optimize_woa(X_train, y_train, cv_splits, max_iter=25, population_size=12):
    lower_bounds = np.array([30, 0.01, 2])
    upper_bounds = np.array([100, 0.10, 3])

    positions = np.random.uniform(
        lower_bounds,
        upper_bounds,
        (population_size, 3),
    )
    leader_position = np.zeros(3)
    leader_score = float("inf")
    convergence_curve = []

    for iteration in range(max_iter):
        coefficient = 2 - iteration * (2 / max_iter)

        for index in range(population_size):
            positions[index] = np.clip(
                positions[index],
                lower_bounds,
                upper_bounds,
            )

            n_estimators = int(np.round(positions[index][0]))
            learning_rate = float(positions[index][1])
            max_depth = int(np.round(positions[index][2]))
            fold_rmses = []

            for fold_train_idx, fold_valid_idx in cv_splits:
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_weight=3,
                    reg_lambda=5.0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    n_jobs=-1,
                )
                model.fit(
                    X_train[fold_train_idx],
                    y_train[fold_train_idx],
                )
                predictions = model.predict(X_train[fold_valid_idx])
                fold_rmses.append(
                    np.sqrt(
                        mean_squared_error(
                            y_train[fold_valid_idx],
                            predictions,
                        )
                    )
                )

            fitness = float(np.mean(fold_rmses))
            if fitness < leader_score:
                leader_score = fitness
                leader_position = positions[index].copy()

        for index in range(population_size):
            r1, r2, probability = np.random.rand(), np.random.rand(), np.random.rand()
            A = 2 * coefficient * r1 - coefficient
            C = 2 * r2

            if probability < 0.5:
                if abs(A) < 1:
                    distance = abs(C * leader_position - positions[index])
                    positions[index] = leader_position - A * distance
                else:
                    random_leader = positions[np.random.randint(0, population_size)]
                    distance = abs(C * random_leader - positions[index])
                    positions[index] = random_leader - A * distance
            else:
                distance = abs(leader_position - positions[index])
                positions[index] = (
                    distance
                    * np.exp(np.random.uniform(-1, 1))
                    * np.cos(2 * np.pi * np.random.uniform(-1, 1))
                    + leader_position
                )

        convergence_curve.append(leader_score)

    best_parameters = [
        int(np.round(leader_position[0])),
        float(leader_position[1]),
        int(np.round(leader_position[2])),
    ]

    return best_parameters, convergence_curve, leader_score


def calculate_woa_feature_importance(
    X_train,
    y_train,
    best_parameters,
    repeats=IMPORTANCE_REPEATS,
):
    importance_runs = []

    for repeat in range(repeats):
        model = xgb.XGBRegressor(
            n_estimators=best_parameters[0],
            learning_rate=best_parameters[1],
            max_depth=best_parameters[2],
            min_child_weight=3,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42 + repeat,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        importance_runs.append(model.feature_importances_)

    importance_runs = np.asarray(importance_runs)

    return (
        np.mean(importance_runs, axis=0),
        np.std(importance_runs, axis=0, ddof=1),
    )


def save_figure(filename, output_directory):
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_directory, f"{filename}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_model_outputs(
    model_name,
    results,
    all_importances,
    feature_columns,
    target_columns,
    output_directory,
):
    os.makedirs(output_directory, exist_ok=True)

    def plot_importance(values, title, filename):
        plt.figure(figsize=(8, 4.5))
        importance_series = pd.Series(
            values,
            index=feature_columns,
        ).sort_values()

        colors = sns.color_palette("viridis", len(importance_series))
        axis = importance_series.plot(
            kind="barh",
            color=colors,
            width=0.6,
        )

        maximum = float(importance_series.max())
        offset = maximum * 0.015 if maximum > 0 else 0.005

        for index, value in enumerate(importance_series):
            axis.text(
                value + offset,
                index,
                f"{value:.3f}",
                va="center",
                fontsize=10,
            )

        plt.title(title, fontsize=12)
        plt.xlabel("Importance score")
        save_figure(filename, output_directory)

    for target in target_columns:
        plot_importance(
            results[target]["importance"],
            f"{model_name} Feature Importance ({target})",
            f"1_Importance_{target}",
        )

    plot_importance(
        np.mean(all_importances, axis=0),
        f"{model_name} Overall Average Feature Importance",
        "1_Importance_Overall",
    )

    metric_names = ["R2", "RMSE", "MSE", "MAE"]

    for target in target_columns:
        train_metrics = results[target]["train_metrics"]
        test_metrics = results[target]["test_metrics"]

        metric_data = pd.DataFrame(
            {
                "Training set": train_metrics,
                "Test set": test_metrics,
            },
            index=metric_names,
        )

        axis = metric_data.plot(
            kind="bar",
            figsize=(6, 5),
            color=["#3498db", "#e67e22"],
            alpha=0.9,
        )

        for patch_index, patch in enumerate(axis.patches):
            value = patch.get_height()
            metric_index = patch_index % len(metric_names)
            metric_name = metric_names[metric_index]

            if metric_name == "MSE" and abs(value) < 0.001:
                label = f"{value:.2e}"
            else:
                label = f"{value:.3f}"

            axis.annotate(
                label,
                (
                    patch.get_x() + patch.get_width() / 2,
                    value,
                ),
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.title(f"{model_name} Performance Metrics ({target})", fontsize=12)
        plt.ylabel("Metric value")
        save_figure(f"2_Metrics_{target}", output_directory)

    for target in target_columns:
        result = results[target]
        plt.figure(figsize=(5, 5))

        all_values = np.concatenate(
            [
                np.asarray(result["y_train"]).ravel(),
                np.asarray(result["train_predictions"]).ravel(),
                np.asarray(result["y_test"]).ravel(),
                np.asarray(result["test_predictions"]).ravel(),
            ]
        )

        value_min = float(np.min(all_values))
        value_max = float(np.max(all_values))
        value_range = value_max - value_min
        padding = 0.1 * value_range if value_range > 0 else 1.0
        limits = [value_min - padding, value_max + padding]

        plt.plot(
            limits,
            limits,
            color="#555555",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            zorder=1,
        )

        plt.scatter(
            result["y_train"],
            result["train_predictions"],
            color="#1f77b4",
            alpha=0.5,
            s=55,
            marker="o",
            label=f'Training set ($R^2$ = {result["train_metrics"][0]:.3f})',
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

        plt.scatter(
            result["y_test"],
            result["test_predictions"],
            color="#d62728",
            alpha=0.8,
            s=65,
            marker="o",
            label=f'Test set ($R^2$ = {result["test_metrics"][0]:.3f})',
            edgecolors="#5a0000",
            linewidths=0.6,
            zorder=3,
        )

        axis = plt.gca()
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor("black")

        plt.grid(True, linestyle="--", alpha=0.4, zorder=0)
        plt.title(f"Predicted vs. Actual: {target}", fontsize=12, pad=15)
        plt.xlabel("Actual values", fontsize=11)
        plt.ylabel("Predicted values", fontsize=11)
        plt.xlim(limits)
        plt.ylim(limits)
        plt.legend(loc="upper left", frameon=True, fontsize=10)

        save_figure(f"3_Comparison_{target}", output_directory)


def run_xgboost(X_train, X_test, y_train, y_test, cv_splits, target):
    out_of_fold_predictions = np.empty(len(y_train), dtype=float)
    fold_rmses = []

    for fold_train_idx, fold_valid_idx in cv_splits:
        model = make_standard_xgb()
        model.fit(X_train[fold_train_idx], y_train[fold_train_idx])
        predictions = model.predict(X_train[fold_valid_idx])
        out_of_fold_predictions[fold_valid_idx] = predictions
        fold_rmses.append(
            np.sqrt(
                mean_squared_error(
                    y_train[fold_valid_idx],
                    predictions,
                )
            )
        )

    report_cross_validation(
        "XGBoost",
        target,
        y_train,
        out_of_fold_predictions,
        fold_rmses,
    )

    model = make_standard_xgb()
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    return {
        "train_metrics": calculate_metrics(y_train, train_predictions),
        "test_metrics": calculate_metrics(y_test, test_predictions),
        "y_train": y_train,
        "train_predictions": train_predictions,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "importance": model.feature_importances_,
    }


def run_svr(X_train, X_test, y_train, y_test, cv_splits, target):
    out_of_fold_predictions = np.empty(len(y_train), dtype=float)
    fold_rmses = []

    for fold_train_idx, fold_valid_idx in cv_splits:
        X_fold_train_raw = X_train[fold_train_idx]
        X_fold_valid_raw = X_train[fold_valid_idx]
        y_fold_train_raw = y_train[fold_train_idx]
        y_fold_valid_raw = y_train[fold_valid_idx]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_fold_train = x_scaler.fit_transform(X_fold_train_raw)
        X_fold_valid = x_scaler.transform(X_fold_valid_raw)
        y_fold_train = y_scaler.fit_transform(
            y_fold_train_raw.reshape(-1, 1)
        ).ravel()

        model = make_svr()
        model.fit(X_fold_train, y_fold_train)

        scaled_predictions = model.predict(X_fold_valid).reshape(-1, 1)
        predictions = y_scaler.inverse_transform(
            scaled_predictions
        ).ravel()

        out_of_fold_predictions[fold_valid_idx] = predictions
        fold_rmses.append(
            np.sqrt(
                mean_squared_error(
                    y_fold_valid_raw,
                    predictions,
                )
            )
        )

    report_cross_validation(
        "SVR",
        target,
        y_train,
        out_of_fold_predictions,
        fold_rmses,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(
        y_train.reshape(-1, 1)
    ).ravel()

    model = make_svr()
    model.fit(X_train_scaled, y_train_scaled)

    permutation = permutation_importance(
        model,
        X_train_scaled,
        y_train_scaled,
        n_repeats=10,
        random_state=42,
    )
    importance = permutation.importances_mean

    train_predictions = y_scaler.inverse_transform(
        model.predict(X_train_scaled).reshape(-1, 1)
    ).ravel()
    test_predictions = y_scaler.inverse_transform(
        model.predict(X_test_scaled).reshape(-1, 1)
    ).ravel()

    return {
        "train_metrics": calculate_metrics(y_train, train_predictions),
        "test_metrics": calculate_metrics(y_test, test_predictions),
        "y_train": y_train,
        "train_predictions": train_predictions,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "importance": importance,
    }


def run_random_forest(
    X_train,
    X_test,
    y_train,
    y_test,
    cv_splits,
    target,
):
    out_of_fold_predictions = np.empty(len(y_train), dtype=float)
    fold_rmses = []

    for fold_train_idx, fold_valid_idx in cv_splits:
        model = make_random_forest()
        model.fit(X_train[fold_train_idx], y_train[fold_train_idx])
        predictions = model.predict(X_train[fold_valid_idx])
        out_of_fold_predictions[fold_valid_idx] = predictions
        fold_rmses.append(
            np.sqrt(
                mean_squared_error(
                    y_train[fold_valid_idx],
                    predictions,
                )
            )
        )

    report_cross_validation(
        "RF",
        target,
        y_train,
        out_of_fold_predictions,
        fold_rmses,
    )

    model = make_random_forest()
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    return {
        "train_metrics": calculate_metrics(y_train, train_predictions),
        "test_metrics": calculate_metrics(y_test, test_predictions),
        "y_train": y_train,
        "train_predictions": train_predictions,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "importance": model.feature_importances_,
    }


def run_woa_xgboost(
    X_train,
    X_test,
    y_train,
    y_test,
    cv_splits,
    target,
):
    best_parameters, convergence_curve, best_cv_rmse = optimize_woa(
        X_train,
        y_train,
        cv_splits,
        max_iter=25,
        population_size=12,
    )

    print(
        f"WOA-XGBoost | {target} | "
        f"Best parameters: n_estimators={best_parameters[0]}, "
        f"learning_rate={best_parameters[1]:.6f}, "
        f"max_depth={best_parameters[2]}, "
        f"5-fold CV RMSE={best_cv_rmse:.6f}"
    )

    out_of_fold_predictions = np.empty(len(y_train), dtype=float)
    fold_rmses = []

    for fold_train_idx, fold_valid_idx in cv_splits:
        model = xgb.XGBRegressor(
            n_estimators=best_parameters[0],
            learning_rate=best_parameters[1],
            max_depth=best_parameters[2],
            min_child_weight=3,
            reg_lambda=5.0,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train[fold_train_idx], y_train[fold_train_idx])
        predictions = model.predict(X_train[fold_valid_idx])
        out_of_fold_predictions[fold_valid_idx] = predictions
        fold_rmses.append(
            np.sqrt(
                mean_squared_error(
                    y_train[fold_valid_idx],
                    predictions,
                )
            )
        )

    report_cross_validation(
        "WOA-XGBoost",
        target,
        y_train,
        out_of_fold_predictions,
        fold_rmses,
    )

    model = xgb.XGBRegressor(
        n_estimators=best_parameters[0],
        learning_rate=best_parameters[1],
        max_depth=best_parameters[2],
        min_child_weight=3,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    importance_mean, importance_std = calculate_woa_feature_importance(
        X_train,
        y_train,
        best_parameters,
        repeats=IMPORTANCE_REPEATS,
    )

    return {
        "train_metrics": calculate_metrics(y_train, train_predictions),
        "test_metrics": calculate_metrics(y_test, test_predictions),
        "y_train": y_train,
        "train_predictions": train_predictions,
        "y_test": y_test,
        "test_predictions": test_predictions,
        "importance": importance_mean,
        "importance_std": importance_std,
        "best_parameters": best_parameters,
        "convergence_curve": convergence_curve,
    }


def main():
    data = load_dataset()

    all_indices = np.arange(len(data))
    train_indices, test_indices = train_test_split(
        all_indices,
        test_size=0.2,
        random_state=100,
    )

    X_all = data[FEATURE_COLUMNS].to_numpy(dtype=float)
    X_train = X_all[train_indices]
    X_test = X_all[test_indices]

    kfold = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=42,
    )
    cv_splits = list(kfold.split(X_train))

    print(
        f"Dataset: {len(data)} records; "
        f"training set: {len(train_indices)}; "
        f"test set: {len(test_indices)}."
    )

    results = {
        "XGBoost": {},
        "SVR": {},
        "RF": {},
        "WOA-XGBoost": {},
    }

    all_importances = {
        "XGBoost": [],
        "SVR": [],
        "RF": [],
        "WOA-XGBoost": [],
    }

    for target in TARGET_COLUMNS:
        y_all = data[target].to_numpy(dtype=float)
        y_train = y_all[train_indices]
        y_test = y_all[test_indices]

        results["XGBoost"][target] = run_xgboost(
            X_train,
            X_test,
            y_train,
            y_test,
            cv_splits,
            target,
        )
        all_importances["XGBoost"].append(
            results["XGBoost"][target]["importance"]
        )

        results["SVR"][target] = run_svr(
            X_train,
            X_test,
            y_train,
            y_test,
            cv_splits,
            target,
        )
        all_importances["SVR"].append(
            results["SVR"][target]["importance"]
        )

        results["RF"][target] = run_random_forest(
            X_train,
            X_test,
            y_train,
            y_test,
            cv_splits,
            target,
        )
        all_importances["RF"].append(
            results["RF"][target]["importance"]
        )

        results["WOA-XGBoost"][target] = run_woa_xgboost(
            X_train,
            X_test,
            y_train,
            y_test,
            cv_splits,
            target,
        )
        all_importances["WOA-XGBoost"].append(
            results["WOA-XGBoost"][target]["importance"]
        )

    for model_name, output_directory in OUTPUT_DIRECTORIES.items():
        plot_model_outputs(
            model_name,
            results[model_name],
            all_importances[model_name],
            FEATURE_COLUMNS,
            TARGET_COLUMNS,
            output_directory,
        )

    importance_records = []
    for target in TARGET_COLUMNS:
        target_result = results["WOA-XGBoost"][target]
        for index, feature in enumerate(FEATURE_COLUMNS):
            importance_records.append(
                {
                    "Target": target,
                    "Feature": feature,
                    "Mean_Importance": target_result["importance"][index],
                    "SD_Importance": target_result["importance_std"][index],
                }
            )

    pd.DataFrame(importance_records).to_csv(
        "WOA_XGBoost_Feature_Importance.csv",
        index=False,
    )

    print("Model evaluation and figure generation completed.")


if __name__ == "__main__":
    main()
