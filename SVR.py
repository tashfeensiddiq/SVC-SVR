
from pathlib import Path
import itertools
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_STATE = 42
CSV_FILE = Path.cwd() / "car_prices.csv"
OUTPUT_DIR = Path.cwd() / "SVR_Results"
OUTPUT_DIR.mkdir(exist_ok=True)

# The lab requires 70% training, 15% validation, 15% testing.
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Run all required experiments.
RUN_RANDOM_SPLIT = True
RUN_KFOLD = True
RUN_REPEATED_KFOLD = True
RUN_FINAL_TEST = True

# The assignment requires a final test evaluation for selected models.
# Cross-validation uses the 85% development set, never the test set.


# ============================================================
# 2. LOAD AND PREPARE DATA
# ============================================================

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Cannot find {CSV_FILE.name}.\n"
        "Place car_prices.csv in the same folder as SVR.py."
    )

df = pd.read_csv(CSV_FILE)

# Clean column names while preserving readable names.
df.columns = [str(c).strip() for c in df.columns]


def normalize_name(name):
    """Normalize column names for matching."""
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


# Find the target column.
target_candidates = [
    col for col in df.columns
    if "selling price" in normalize_name(col)
]

if target_candidates:
    TARGET_COLUMN = target_candidates[0]
else:
    # The lab describes selling price as the 11th column.
    if len(df.columns) < 11:
        raise ValueError(
            "Expected at least 11 columns, but the CSV has "
            f"{len(df.columns)} columns."
        )
    TARGET_COLUMN = df.columns[10]

print("=" * 70)
print("USED CAR PRICE PREDICTION - SVR")
print("=" * 70)
print(f"Dataset: {CSV_FILE}")
print(f"Rows before cleaning: {len(df)}")
print(f"Target column: {TARGET_COLUMN}")
print(f"Columns: {list(df.columns)}")

# Convert the target to numeric and remove rows without a target.
df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN], errors="coerce"
)
df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

# The lab describes the other 10 columns as input features.
X = df.drop(columns=[TARGET_COLUMN]).copy()
y = df[TARGET_COLUMN].astype(float).copy()

# Expected categorical fields from the lab.
expected_categorical_names = {
    "fuel type",
    "seller type",
    "transmission",
}

categorical_columns = [
    col for col in X.columns
    if normalize_name(col) in expected_categorical_names
]

# If column names vary, use object/string columns as categorical too.
for col in X.columns:
    if (
        X[col].dtype == "object"
        or pd.api.types.is_string_dtype(X[col])
    ):
        if col not in categorical_columns:
            categorical_columns.append(col)

# Remaining features are numeric.
numeric_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

# Convert numeric columns; unparseable values become missing and
# will be imputed using training-fold medians.
for col in numeric_columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Normalize categorical values for reliable encoding.
for col in categorical_columns:
    X[col] = X[col].astype("object")
    X[col] = X[col].where(X[col].notna(), np.nan)

if len(X.columns) != 10:
    print(
        "\nWARNING: The dataset has "
        f"{len(X.columns)} input columns after selecting the target. "
        "Check that these match the 10 features listed in your lab."
    )

print(f"Rows after removing missing target values: {len(df)}")
print(f"Input features: {len(X.columns)}")
print(f"Numeric features ({len(numeric_columns)}): {numeric_columns}")
print(
    f"Categorical features ({len(categorical_columns)}): "
    f"{categorical_columns}"
)

if len(df) < 20:
    raise ValueError("Too few usable rows for the requested splits.")


# ============================================================
# 3. DEFINE THE 50 SVR CONFIGURATIONS FROM TABLE 2
# ============================================================

CONFIGS = []


def add_config(kernel, C, epsilon, tol, max_iter,
               coef0=0.0, degree=3, gamma="scale"):
    """Add one SVR configuration."""
    CONFIGS.append({
        "kernel": kernel,
        "C": C,
        "epsilon": epsilon,
        "tol": tol,
        "max_iter": max_iter,
        "coef0": coef0,
        "degree": degree,
        "gamma": gamma,
    })


# ---- Linear kernel: 12 configurations ----

linear_rows = [
    (0.1, 0.01, 1e-3, -1),
    (0.1, 0.05, 1e-3, -1),
    (0.1, 0.10, 1e-4, -1),
    (0.5, 0.01, 1e-3, -1),
    (0.5, 0.05, 1e-4, -1),
    (0.5, 0.10, 1e-4, 1000),
    (1.0, 0.01, 1e-3, -1),
    (1.0, 0.05, 1e-4, -1),
    (1.0, 0.10, 1e-5, 5000),
    (2.0, 0.01, 1e-3, -1),
    (2.0, 0.05, 1e-4, -1),
    (2.0, 0.10, 1e-5, 5000),
]

for C, epsilon, tol, max_iter in linear_rows:
    add_config(
        "linear", C, epsilon, tol, max_iter
    )


# ---- Sigmoid kernel: 12 configurations ----

sigmoid_rows = [
    (0.1, 0.01, 1e-3, -1),
    (0.1, 0.05, 1e-3, -1),
    (0.1, 0.10, 1e-4, -1),
    (0.5, 0.01, 1e-3, -1),
    (0.5, 0.05, 1e-4, -1),
    (0.5, 0.10, 1e-4, 1000),
    (1.0, 0.01, 1e-3, -1),
    (1.0, 0.05, 1e-4, -1),
    (1.0, 0.10, 1e-5, 5000),
    (2.0, 0.01, 1e-3, -1),
    (2.0, 0.05, 1e-4, -1),
    (2.0, 0.10, 1e-5, 5000),
]

for C, epsilon, tol, max_iter in sigmoid_rows:
    add_config(
        "sigmoid", C, epsilon, tol, max_iter
    )


# ---- Polynomial kernel: 13 configurations ----

poly_rows = [
    (0.1, 0.01, 1e-3, -1),
    (0.1, 0.05, 1e-3, -1),
    (0.1, 0.10, 1e-4, -1),
    (0.5, 0.01, 1e-3, -1),
    (0.5, 0.05, 1e-4, -1),
    (0.5, 0.10, 1e-4, 1000),
    (1.0, 0.01, 1e-3, -1),
    (1.0, 0.05, 1e-4, -1),
    (1.0, 0.10, 1e-5, 5000),
    (2.0, 0.01, 1e-3, -1),
    (2.0, 0.05, 1e-4, -1),
    (2.0, 0.10, 1e-5, 5000),
    (5.0, 0.01, 1e-4, -1),
]

for C, epsilon, tol, max_iter in poly_rows:
    add_config(
        "poly", C, epsilon, tol, max_iter
    )


# ---- RBF kernel: 13 configurations ----
# coef0 is recorded to match Table 2. It does not affect the RBF
# kernel in scikit-learn.

rbf_rows = [
    (0.1, 0.00, 1e-3, -1),
    (0.1, 0.10, 1e-3, -1),
    (0.1, 0.00, 1e-4, 1000),
    (0.5, 0.00, 1e-3, -1),
    (0.5, 0.10, 1e-4, -1),
    (0.5, 0.00, 1e-5, 5000),
    (1.0, 0.00, 1e-3, -1),
    (1.0, 0.10, 1e-4, -1),
    (1.0, 0.00, 1e-5, 10000),
    (2.0, 0.00, 1e-3, -1),
    (2.0, 0.10, 1e-4, -1),
    (5.0, 0.00, 1e-5, 5000),
    (5.0, 0.10, 1e-5, 10000),
]

for C, coef0, tol, max_iter in rbf_rows:
    # The RBF rows in Table 2 do not list epsilon.
    # Use epsilon=0.01, consistent with the first setting in the
    # other kernel groups. Confirm this against your lab table.
    add_config(
        "rbf", C, 0.01, tol, max_iter, coef0=coef0
    )


if len(CONFIGS) != 50:
    raise RuntimeError(
        f"Expected 50 configurations, found {len(CONFIGS)}."
    )

for i, config in enumerate(CONFIGS, start=1):
    config["config_id"] = i

print(f"\nTotal SVR configurations: {len(CONFIGS)}")
print(
    "Note: scikit-learn defaults gamma='scale' and polynomial "
    "degree=3 are used where Table 2 does not specify them."
)


# ============================================================
# 4. PREPROCESSING AND MODEL HELPERS
# ============================================================

def make_one_hot_encoder():
    """
    Support both newer and older scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False
        )


def make_pipeline(config):
    """
    Create a pipeline. All X preprocessing is fitted only on
    the training data passed to model.fit().
    """
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    svr_model = SVR(
        kernel=config["kernel"],
        C=config["C"],
        epsilon=config["epsilon"],
        tol=config["tol"],
        max_iter=config["max_iter"],
        coef0=config["coef0"],
        degree=config["degree"],
        gamma=config["gamma"],
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("svr", svr_model),
    ])


def calculate_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def fit_and_evaluate(config, X_fit, y_fit, X_eval, y_eval):
    """
    Fit X preprocessing on X_fit and fit target scaling on y_fit.
    Return metrics in the original selling-price units.
    """
    model = make_pipeline(config)

    y_fit_array = np.asarray(y_fit, dtype=float).reshape(-1, 1)
    y_eval_array = np.asarray(y_eval, dtype=float).reshape(-1, 1)

    target_scaler = StandardScaler()
    y_fit_scaled = target_scaler.fit_transform(
        y_fit_array
    ).ravel()

    model.fit(X_fit, y_fit_scaled)

    train_pred_scaled = model.predict(X_fit).reshape(-1, 1)
    eval_pred_scaled = model.predict(X_eval).reshape(-1, 1)

    train_pred = target_scaler.inverse_transform(
        train_pred_scaled
    ).ravel()

    eval_pred = target_scaler.inverse_transform(
        eval_pred_scaled
    ).ravel()

    train_rmse = calculate_rmse(
        y_fit_array.ravel(), train_pred
    )
    eval_rmse = calculate_rmse(
        y_eval_array.ravel(), eval_pred
    )

    train_r2 = float(
        r2_score(y_fit_array.ravel(), train_pred)
    )
    eval_r2 = float(
        r2_score(y_eval_array.ravel(), eval_pred)
    )

    return {
        "model": model,
        "target_scaler": target_scaler,
        "train_rmse": train_rmse,
        "validation_rmse": eval_rmse,
        "train_r2": train_r2,
        "validation_r2": eval_r2,
        "predictions": eval_pred,
    }


def config_columns(config):
    """Return the hyperparameters to save in result tables."""
    return {
        "config_id": config["config_id"],
        "kernel": config["kernel"],
        "C": config["C"],
        "epsilon": config["epsilon"],
        "tol": config["tol"],
        "max_iter": config["max_iter"],
        "coef0": config["coef0"],
        "degree": config["degree"],
        "gamma": config["gamma"],
    }


def print_best_result(title, results_df):
    if results_df.empty:
        return

    best = results_df.sort_values(
        "validation_rmse", ascending=True
    ).iloc[0]

    print(f"\n{title}")
    print("-" * 60)
    print(f"Best configuration: {int(best['config_id'])}")
    print(f"Kernel: {best['kernel']}")
    print(f"C: {best['C']}")
    print(f"Epsilon: {best['epsilon']}")
    print(f"Training RMSE: {best['train_rmse']:.4f}")
    print(f"Validation RMSE: {best['validation_rmse']:.4f}")
    print(f"Validation R²: {best['validation_r2']:.4f}")


# ============================================================
# 5. CREATE 70% TRAIN / 15% VALIDATION / 15% TEST SPLIT
# ============================================================

# First, reserve 15% as the test set.
X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

# Of the remaining 85%, allocate 70% of the full dataset to
# training and 15% of the full dataset to validation.
validation_fraction_of_development = (
    VALIDATION_SIZE / (TRAIN_SIZE + VALIDATION_SIZE)
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=validation_fraction_of_development,
    random_state=RANDOM_STATE,
)

# Combine training and validation for cross-validation and final
# model fitting. The held-out test set remains untouched.
X_development = pd.concat(
    [X_train, X_validation], axis=0
).reset_index(drop=True)

y_development = pd.concat(
    [y_train, y_validation], axis=0
).reset_index(drop=True)

X_train = X_train.reset_index(drop=True)
X_validation = X_validation.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)
y_validation = y_validation.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

print("\nDataset split")
print("-" * 60)
print(f"Training rows:   {len(X_train)} ({len(X_train)/len(X)*100:.2f}%)")
print(
    f"Validation rows: {len(X_validation)} "
    f"({len(X_validation)/len(X)*100:.2f}%)"
)
print(f"Test rows:       {len(X_test)} ({len(X_test)/len(X)*100:.2f}%)")


# ============================================================
# 6. RANDOM SPLIT EXPERIMENT: ALL 50 CONFIGURATIONS
# ============================================================

random_results = []

if RUN_RANDOM_SPLIT:
    print("\n" + "=" * 70)
    print("RANDOM SPLIT EXPERIMENT")
    print("=" * 70)

    for i, config in enumerate(CONFIGS, start=1):
        start = time.time()

        metrics = fit_and_evaluate(
            config,
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

        row = config_columns(config)
        row.update({
            "train_rmse": metrics["train_rmse"],
            "validation_rmse": metrics["validation_rmse"],
            "train_r2": metrics["train_r2"],
            "validation_r2": metrics["validation_r2"],
            "elapsed_seconds": time.time() - start,
        })
        random_results.append(row)

        print(
            f"[Random {i:02d}/{len(CONFIGS)}] "
            f"{config['kernel']:7s} C={config['C']:<4} "
            f"epsilon={config['epsilon']:<5} "
            f"validation RMSE={metrics['validation_rmse']:.4f}"
        )

    random_df = pd.DataFrame(random_results).sort_values(
        "validation_rmse", ascending=True
    )
    random_df.to_csv(
        OUTPUT_DIR / "partB_random_split_results.csv",
        index=False
    )

    print_best_result("Best random-split configuration", random_df)


# ============================================================
# 7. K-FOLD CROSS-VALIDATION: 5, 10, AND 15 FOLDS
# ============================================================

kfold_results_by_setting = {}

if RUN_KFOLD:
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION")
    print("=" * 70)

    for n_splits in [5, 10, 15]:
        print(f"\nRunning {n_splits}-Fold CV...")

        splitter = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

        setting_results = []

        for config_number, config in enumerate(CONFIGS, start=1):
            fold_train_rmse = []
            fold_validation_rmse = []
            fold_train_r2 = []
            fold_validation_r2 = []

            for fold_number, (train_idx, val_idx) in enumerate(
                splitter.split(X_development), start=1
            ):
                X_fold_train = X_development.iloc[train_idx]
                X_fold_val = X_development.iloc[val_idx]
                y_fold_train = y_development.iloc[train_idx]
                y_fold_val = y_development.iloc[val_idx]

                metrics = fit_and_evaluate(
                    config,
                    X_fold_train,
                    y_fold_train,
                    X_fold_val,
                    y_fold_val,
                )

                fold_train_rmse.append(metrics["train_rmse"])
                fold_validation_rmse.append(
                    metrics["validation_rmse"]
                )
                fold_train_r2.append(metrics["train_r2"])
                fold_validation_r2.append(
                    metrics["validation_r2"]
                )

            row = config_columns(config)
            row.update({
                "n_splits": n_splits,
                "train_rmse": float(np.mean(fold_train_rmse)),
                "validation_rmse": float(
                    np.mean(fold_validation_rmse)
                ),
                "train_rmse_std": float(np.std(fold_train_rmse)),
                "validation_rmse_std": float(
                    np.std(fold_validation_rmse)
                ),
                "train_r2": float(np.mean(fold_train_r2)),
                "validation_r2": float(
                    np.mean(fold_validation_r2)
                ),
            })
            setting_results.append(row)

            print(
                f"[KFold {n_splits}] "
                f"Config {config_number:02d}/{len(CONFIGS)} "
                f"({config['kernel']}, C={config['C']}): "
                f"mean validation RMSE="
                f"{row['validation_rmse']:.4f}"
            )

        setting_df = pd.DataFrame(setting_results).sort_values(
            "validation_rmse", ascending=True
        )
        kfold_results_by_setting[n_splits] = setting_df

        setting_df.to_csv(
            OUTPUT_DIR / f"partB_kfold_{n_splits}_fold_results.csv",
            index=False,
        )

        print_best_result(
            f"Best {n_splits}-Fold configuration",
            setting_df
        )


# ============================================================
# 8. REPEATED K-FOLD: ALL SIX REQUIRED SETTINGS
# ============================================================

repeated_results_by_setting = {}

if RUN_REPEATED_KFOLD:
    print("\n" + "=" * 70)
    print("REPEATED K-FOLD CROSS-VALIDATION")
    print("=" * 70)

    repeated_settings = [
        (5, 5),
        (10, 5),
        (15, 5),
        (5, 10),
        (10, 10),
        (15, 10),
    ]

    for n_splits, n_repeats in repeated_settings:
        setting_name = f"{n_splits}_fold_{n_repeats}_repeats"

        print(
            f"\nRunning Repeated K-Fold: "
            f"{n_splits} folds, {n_repeats} repetitions..."
        )

        splitter = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=RANDOM_STATE,
        )

        # Store the same splits for every configuration, ensuring
        # each configuration is evaluated on the same folds.
        splits = list(splitter.split(X_development))
        setting_results = []

        for config_number, config in enumerate(CONFIGS, start=1):
            fold_train_rmse = []
            fold_validation_rmse = []
            fold_train_r2 = []
            fold_validation_r2 = []

            for train_idx, val_idx in splits:
                X_fold_train = X_development.iloc[train_idx]
                X_fold_val = X_development.iloc[val_idx]
                y_fold_train = y_development.iloc[train_idx]
                y_fold_val = y_development.iloc[val_idx]

                metrics = fit_and_evaluate(
                    config,
                    X_fold_train,
                    y_fold_train,
                    X_fold_val,
                    y_fold_val,
                )

                fold_train_rmse.append(metrics["train_rmse"])
                fold_validation_rmse.append(
                    metrics["validation_rmse"]
                )
                fold_train_r2.append(metrics["train_r2"])
                fold_validation_r2.append(
                    metrics["validation_r2"]
                )

            row = config_columns(config)
            row.update({
                "n_splits": n_splits,
                "n_repeats": n_repeats,
                "number_of_folds_evaluated": len(splits),
                "train_rmse": float(np.mean(fold_train_rmse)),
                "validation_rmse": float(
                    np.mean(fold_validation_rmse)
                ),
                "train_rmse_std": float(np.std(fold_train_rmse)),
                "validation_rmse_std": float(
                    np.std(fold_validation_rmse)
                ),
                "train_r2": float(np.mean(fold_train_r2)),
                "validation_r2": float(
                    np.mean(fold_validation_r2)
                ),
            })
            setting_results.append(row)

            print(
                f"[Repeated {n_splits}x{n_repeats}] "
                f"Config {config_number:02d}/{len(CONFIGS)} "
                f"({config['kernel']}, C={config['C']}): "
                f"mean validation RMSE="
                f"{row['validation_rmse']:.4f}"
            )

        setting_df = pd.DataFrame(setting_results).sort_values(
            "validation_rmse", ascending=True
        )
        repeated_results_by_setting[setting_name] = setting_df

        setting_df.to_csv(
            OUTPUT_DIR / f"partB_repeated_kfold_{setting_name}_results.csv",
            index=False,
        )

        print_best_result(
            f"Best Repeated K-Fold ({n_splits} folds, "
            f"{n_repeats} repeats)",
            setting_df
        )


# ============================================================
# 9. SELECT CONFIGURATIONS AND EVALUATE ON THE TEST SET
# ============================================================

final_test_results = []
selected_config_ids = set()


def get_config_by_id(config_id):
    for config in CONFIGS:
        if int(config["config_id"]) == int(config_id):
            return config
    raise ValueError(f"Configuration {config_id} was not found.")


def select_best_config(results_df, method_name):
    """Choose the lowest validation RMSE, not the test RMSE."""
    if results_df is None or results_df.empty:
        return

    best_row = results_df.sort_values(
        "validation_rmse", ascending=True
    ).iloc[0]

    config_id = int(best_row["config_id"])
    selected_config_ids.add(config_id)

    return {
        "method": method_name,
        "config_id": config_id,
        "validation_rmse": float(best_row["validation_rmse"]),
        "config": get_config_by_id(config_id),
    }


if RUN_FINAL_TEST:
    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    selected_models = []

    if RUN_RANDOM_SPLIT and random_results:
        selected = select_best_config(
            pd.DataFrame(random_results),
            "Random split",
        )
        if selected:
            selected_models.append(selected)

    if RUN_KFOLD:
        for n_splits, results_df in kfold_results_by_setting.items():
            selected = select_best_config(
                results_df,
                f"KFold {n_splits}",
            )
            if selected:
                selected_models.append(selected)

    if RUN_REPEATED_KFOLD:
        for setting_name, results_df in repeated_results_by_setting.items():
            selected = select_best_config(
                results_df,
                f"Repeated KFold {setting_name}",
            )
            if selected:
                selected_models.append(selected)

    # Avoid evaluating the same hyperparameter configuration more
    # than once. Keep the selection method with the lowest CV RMSE.
    unique_selected = {}
    for item in selected_models:
        config_id = item["config_id"]
        if (
            config_id not in unique_selected
            or item["validation_rmse"]
            < unique_selected[config_id]["validation_rmse"]
        ):
            unique_selected[config_id] = item

    for counter, (config_id, selection) in enumerate(
        unique_selected.items(), start=1
    ):
        config = selection["config"]

        print(
            f"\nTest model {counter}/{len(unique_selected)}: "
            f"Config {config_id}, selected by {selection['method']}"
        )

        # Fit using the full 85% development data. The test set
        # is used only for this final evaluation.
        metrics = fit_and_evaluate(
            config,
            X_development,
            y_development,
            X_test,
            y_test,
        )

        row = config_columns(config)
        row.update({
            "selected_by": selection["method"],
            "cross_validation_rmse": selection["validation_rmse"],
            "test_rmse": metrics["validation_rmse"],
            "test_r2": metrics["validation_r2"],
            "development_train_rmse": metrics["train_rmse"],
            "development_train_r2": metrics["train_r2"],
            "test_rows": len(X_test),
        })
        final_test_results.append(row)

        print(f"Test RMSE: {metrics['validation_rmse']:.4f}")
        print(f"Test R²:   {metrics['validation_r2']:.4f}")

    final_df = pd.DataFrame(final_test_results)

    if not final_df.empty:
        final_df = final_df.sort_values(
            "test_rmse", ascending=True
        )
        final_df.to_csv(
            OUTPUT_DIR / "partB_final_test_results.csv",
            index=False,
        )

        print("\n" + "=" * 70)
        print("FINAL TEST RESULTS (sorted by test RMSE)")
        print("=" * 70)
        print(
            final_df[
                [
                    "config_id",
                    "selected_by",
                    "kernel",
                    "C",
                    "epsilon",
                    "cross_validation_rmse",
                    "test_rmse",
                    "test_r2",
                ]
            ].to_string(index=False)
        )

        best_test_row = final_df.iloc[0]
        print("\nBest test RMSE among selected models:")
        print(f"Configuration: {int(best_test_row['config_id'])}")
        print(f"Kernel: {best_test_row['kernel']}")
        print(f"Test RMSE: {best_test_row['test_rmse']:.4f}")
        print(f"Test R²: {best_test_row['test_r2']:.4f}")
        print("Lab RMSE target: 80,000 or less")
        print(
            "Meets RMSE target:",
            "YES" if best_test_row["test_rmse"] <= 80000 else "NO"
        )


# ============================================================
# 10. SAVE A COMBINED SUMMARY AND FINISH
# ============================================================

summary_rows = []

if RUN_RANDOM_SPLIT and random_results:
    best = pd.DataFrame(random_results).sort_values(
        "validation_rmse"
    ).iloc[0]
    summary_rows.append({
        "experiment": "Random split",
        "setting": "70/15/15",
        "best_config_id": int(best["config_id"]),
        "kernel": best["kernel"],
        "C": best["C"],
        "epsilon": best["epsilon"],
        "train_rmse": best["train_rmse"],
        "validation_rmse": best["validation_rmse"],
        "validation_r2": best["validation_r2"],
    })

for n_splits, results_df in kfold_results_by_setting.items():
    best = results_df.iloc[0]
    summary_rows.append({
        "experiment": "KFold",
        "setting": f"{n_splits} folds",
        "best_config_id": int(best["config_id"]),
        "kernel": best["kernel"],
        "C": best["C"],
        "epsilon": best["epsilon"],
        "train_rmse": best["train_rmse"],
        "validation_rmse": best["validation_rmse"],
        "validation_r2": best["validation_r2"],
    })

for setting_name, results_df in repeated_results_by_setting.items():
    best = results_df.iloc[0]
    summary_rows.append({
        "experiment": "Repeated KFold",
        "setting": setting_name,
        "best_config_id": int(best["config_id"]),
        "kernel": best["kernel"],
        "C": best["C"],
        "epsilon": best["epsilon"],
        "train_rmse": best["train_rmse"],
        "validation_rmse": best["validation_rmse"],
        "validation_r2": best["validation_r2"],
    })

if summary_rows:
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT_DIR / "partB_best_validation_summary.csv",
        index=False,
    )

print("\n" + "=" * 70)
print("ALL EXPERIMENTS FINISHED")
print("=" * 70)
print(f"Results folder: {OUTPUT_DIR.resolve()}")
print("\nGenerated CSV files include:")
print("- partB_random_split_results.csv")
print("- partB_kfold_5_fold_results.csv")
print("- partB_kfold_10_fold_results.csv")
print("- partB_kfold_15_fold_results.csv")
print("- Six partB_repeated_kfold_*_results.csv files")
print("- partB_final_test_results.csv")
print("- partB_best_validation_summary.csv")
