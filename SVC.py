
import glob
from pathlib import Path
import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from skimage.feature import hog

from sklearn import svm
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    ParameterGrid
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# 1. LOAD AND PREPROCESS THE DATASET
# ============================================================

dataset_path = Path(__file__).parent / "Utensil images"

classes = ["FORK", "KNIFE", "SPOON"]

X, y = [], []

for label, cls in enumerate(classes):
    files = glob.glob(str(dataset_path / cls / "*.png"))
    print(f"{cls}: found {len(files)} PNG files")

    for file in files:
        img = imread(file)

        # Convert RGBA or RGB images to grayscale
        if img.ndim == 3 and img.shape[2] == 4:
            img = rgba2rgb(img)

        if img.ndim == 3:
            img = rgb2gray(img)

        # Resize images to 64 x 64

        # Resize images to 64 x 64
        img_resized = resize(
            img.astype(np.float32),
            (64, 64),
            anti_aliasing=True
        )


        # Extract Histogram of Oriented Gradients features
        features = hog(
            img_resized,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=False
        )

        X.append(features)
        y.append(label)

X = np.asarray(X)
y = np.asarray(y)

if len(X) == 0:
    raise ValueError(
        "No images loaded. Check the dataset path, folder names "
        "and PNG file extensions."
    )

if len(np.unique(y)) != len(classes):
    raise ValueError(
        "One or more classes have no images. Check the dataset folders."
    )

print("\nTotal samples:", len(X))
print("Feature matrix shape:", X.shape)

for label, cls in enumerate(classes):
    print(f"{cls} samples:", np.sum(y == label))

# ============================================================
# 2. RANDOM TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

# Preserve the original held-out test data
X_test_holdout = X_test.copy()
y_test_holdout = y_test.copy()

# Combine training and validation data for cross-validation
X_dev = np.vstack((X_train, X_val))
y_dev = np.concatenate((y_train, y_val))

print("\n--- Dataset Split ---")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Testing samples:", len(X_test_holdout))

# ============================================================
# 3. HYPERPARAMETER CONFIGURATIONS FROM LAB TABLE 1
# ============================================================

# Each dictionary corresponds to a configuration in the lab.
# Parameters irrelevant to a particular kernel are omitted.

configurations = [
    # Linear kernel: 12 configurations
    {"kernel": "linear", "C": 0.1, "tol": 1e-3, "max_iter": -1},
    {"kernel": "linear", "C": 1.0, "tol": 1e-4, "max_iter": -1},
    {"kernel": "linear", "C": 10.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "linear", "C": 0.5, "tol": 1e-4, "max_iter": -1},
    {"kernel": "linear", "C": 5.0, "tol": 1e-2, "max_iter": -1},
    {"kernel": "linear", "C": 1.0, "tol": 1e-3, "max_iter": 5000},
    {"kernel": "linear", "C": 0.1, "tol": 1e-3, "max_iter": 10000},
    {"kernel": "linear", "C": 2.0, "tol": 1e-4, "max_iter": 2000},
    {"kernel": "linear", "C": 0.5, "tol": 1e-2, "max_iter": -1},
    {"kernel": "linear", "C": 3.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "linear", "C": 1.0, "tol": 1e-2, "max_iter": 1000},
    {"kernel": "linear", "C": 0.05, "tol": 1e-4, "max_iter": -1},

    # Sigmoid kernel: 12 configurations
    {"kernel": "sigmoid", "C": 0.1, "gamma": 0.01,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "sigmoid", "C": 1.0, "gamma": 0.01,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "sigmoid", "C": 10.0, "gamma": 0.01,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "sigmoid", "C": 0.5, "gamma": 0.05,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "sigmoid", "C": 1.0, "gamma": 0.05,
     "coef0": 0.0, "tol": 1e-4, "max_iter": -1},
    {"kernel": "sigmoid", "C": 5.0, "gamma": 0.05,
     "coef0": 0.1, "tol": 1e-3, "max_iter": -1},
    {"kernel": "sigmoid", "C": 1.0, "gamma": 0.1,
     "coef0": 0.0, "tol": 1e-3, "max_iter": 5000},
    {"kernel": "sigmoid", "C": 0.1, "gamma": 0.1,
     "coef0": 0.1, "tol": 1e-4, "max_iter": 10000},
    {"kernel": "sigmoid", "C": 2.0, "gamma": 0.1,
     "coef0": 0.0, "tol": 1e-3, "max_iter": 2000},
    {"kernel": "sigmoid", "C": 0.5, "gamma": 0.2,
     "coef0": 0.1, "tol": 1e-2, "max_iter": -1},
    {"kernel": "sigmoid", "C": 3.0, "gamma": 0.2,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "sigmoid", "C": 1.0, "gamma": 0.05,
     "coef0": 0.05, "tol": 1e-4, "max_iter": 1000},

    # Polynomial kernel: 12 configurations
    {"kernel": "poly", "C": 0.1, "degree": 2, "gamma": 0.01,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "poly", "C": 0.5, "degree": 2, "gamma": 0.05,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "poly", "C": 1.0, "degree": 2, "gamma": 0.1,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},
    {"kernel": "poly", "C": 5.0, "degree": 3, "gamma": 0.01,
     "coef0": 0.1, "tol": 1e-4, "max_iter": -1},
    {"kernel": "poly", "C": 10.0, "degree": 3, "gamma": 0.05,
     "coef0": 0.1, "tol": 1e-3, "max_iter": -1},
    {"kernel": "poly", "C": 1.0, "degree": 3, "gamma": 0.1,
     "coef0": 0.0, "tol": 1e-4, "max_iter": 5000},
    {"kernel": "poly", "C": 0.1, "degree": 3, "gamma": 0.1,
     "coef0": 0.05, "tol": 1e-3, "max_iter": 10000},
    {"kernel": "poly", "C": 2.0, "degree": 4, "gamma": 0.01,
     "coef0": 0.0, "tol": 1e-3, "max_iter": 2000},
    {"kernel": "poly", "C": 0.5, "degree": 4, "gamma": 0.05,
     "coef0": 0.1, "tol": 1e-2, "max_iter": -1},
    {"kernel": "poly", "C": 3.0, "degree": 4, "gamma": 0.1,
     "coef0": 0.05, "tol": 1e-3, "max_iter": -1},
    {"kernel": "poly", "C": 1.0, "degree": 5, "gamma": 0.05,
     "coef0": 0.0, "tol": 1e-4, "max_iter": 1000},
    {"kernel": "poly", "C": 5.0, "degree": 5, "gamma": 0.1,
     "coef0": 0.1, "tol": 1e-3, "max_iter": -1},
    {"kernel": "poly", "C": 10.0, "degree": 5, "gamma": 0.2,
     "coef0": 0.0, "tol": 1e-3, "max_iter": -1},

    # RBF kernel: 12 configurations
    {"kernel": "rbf", "C": 0.1, "gamma": 0.01,
     "tol": 1e-3, "max_iter": -1},
    {"kernel": "rbf", "C": 0.5, "gamma": 0.01,
     "tol": 1e-4, "max_iter": -1},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.01,
     "tol": 1e-3, "max_iter": -1},
    {"kernel": "rbf", "C": 5.0, "gamma": 0.01,
     "tol": 1e-3, "max_iter": -1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.01,
     "tol": 1e-4, "max_iter": -1},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.05,
     "tol": 1e-3, "max_iter": 5000},
    {"kernel": "rbf", "C": 0.1, "gamma": 0.05,
     "tol": 1e-4, "max_iter": 10000},
    {"kernel": "rbf", "C": 2.0, "gamma": 0.05,
     "tol": 1e-3, "max_iter": 2000},
    {"kernel": "rbf", "C": 0.5, "gamma": 0.1,
     "tol": 1e-2, "max_iter": -1},
    {"kernel": "rbf", "C": 3.0, "gamma": 0.1,
     "tol": 1e-3, "max_iter": -1},
    {"kernel": "rbf", "C": 1.0, "gamma": 0.2,
     "tol": 1e-4, "max_iter": 1000},
    {"kernel": "rbf", "C": 5.0, "gamma": 0.2,
     "tol": 1e-3, "max_iter": -1},
    {"kernel": "rbf", "C": 10.0, "gamma": 0.2,
     "tol": 1e-3, "max_iter": -1}
]

# Add degree=3 to configurations where the kernel does not use degree.
# Add coef0=0.0 to kernels where it is ignored.
for params in configurations:
    params.setdefault("degree", 3)
    params.setdefault("coef0", 0.0)

# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def create_model(params):
    return svm.SVC(**params)


def classify_behavior(train_acc, val_acc):
    gap = train_acc - val_acc

    if val_acc >= 0.90 and gap <= 0.05:
        return "Good generalization"
    elif gap > 0.10:
        return "Possible overfitting"
    elif train_acc < 0.80 and val_acc < 0.80:
        return "Possible underfitting"
    else:
        return "Moderate performance; inspect results"


def evaluate_random_split(params):
    model = create_model(params)
    model.fit(X_train, y_train)

    train_acc = accuracy_score(
        y_train, model.predict(X_train)
    )
    val_acc = accuracy_score(
        y_val, model.predict(X_val)
    )

    return {
        "parameters": params.copy(),
        "training_accuracy": train_acc,
        "validation_accuracy": val_acc,
        "comments": classify_behavior(train_acc, val_acc)
    }


# ============================================================
# 5. RANDOM SPLIT HYPERPARAMETER EXPERIMENTS
# ============================================================

print("\n========== RANDOM SPLIT EXPERIMENTS ==========")

random_results = []

for i, params in enumerate(configurations, start=1):
    try:
        result = evaluate_random_split(params)
        random_results.append(result)

        print(
            f"{i}/{len(configurations)} | "
            f"{params} | "
            f"Train={result['training_accuracy']:.4f} | "
            f"Validation={result['validation_accuracy']:.4f}"
        )

    except Exception as exc:
        print(f"Configuration {i} failed: {exc}")

if not random_results:
    raise RuntimeError("All random split experiments failed.")

random_df = pd.DataFrame([
    {
        "hyperparameters": str(r["parameters"]),
        "training_accuracy": r["training_accuracy"],
        "validation_accuracy": r["validation_accuracy"],
        "comments": r["comments"]
    }
    for r in random_results
])

random_df.to_csv("partA_random_split_results.csv", index=False)

# Select by validation accuracy, not training accuracy
best_random = max(
    random_results,
    key=lambda r: r["validation_accuracy"]
)

print("\nBest random split configuration:")
print(best_random["parameters"])
print("Training accuracy:", best_random["training_accuracy"])
print("Validation accuracy:", best_random["validation_accuracy"])

# ============================================================
# 6. K-FOLD CROSS VALIDATION
#    5, 10 and 15 folds
# ============================================================

print("\n========== K-FOLD CROSS VALIDATION ==========")

kfold_results = []
best_kfold_models = {}

for n_splits in [5, 10, 15]:
    print(f"\n--- {n_splits}-Fold ---")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    config_results = []

    for config_no, params in enumerate(configurations, start=1):
        train_scores = []
        val_scores = []
        true_labels = []
        predicted_labels = []

        try:
            for train_idx, val_idx in cv.split(X_dev, y_dev):
                X_fold_train = X_dev[train_idx]
                X_fold_val = X_dev[val_idx]
                y_fold_train = y_dev[train_idx]
                y_fold_val = y_dev[val_idx]

                model = create_model(params)
                model.fit(X_fold_train, y_fold_train)

                train_scores.append(
                    model.score(X_fold_train, y_fold_train)
                )

                predictions = model.predict(X_fold_val)
                val_scores.append(
                    accuracy_score(y_fold_val, predictions)
                )

                true_labels.extend(y_fold_val)
                predicted_labels.extend(predictions)

            result = {
                "folds": n_splits,
                "parameters": params.copy(),
                "training_accuracy": float(np.mean(train_scores)),
                "validation_accuracy": float(np.mean(val_scores)),
                "comments": classify_behavior(
                    float(np.mean(train_scores)),
                    float(np.mean(val_scores))
                )
            }

            config_results.append(result)
            kfold_results.append(result)

            print(
                f"Config {config_no}/{len(configurations)} | "
                f"Train={result['training_accuracy']:.4f} | "
                f"Validation={result['validation_accuracy']:.4f}"
            )

        except Exception as exc:
            print(f"Config {config_no} failed: {exc}")

    if config_results:
        best_fold = max(
            config_results,
            key=lambda r: r["validation_accuracy"]
        )
        best_kfold_models[n_splits] = best_fold

        print(f"\nBest {n_splits}-fold parameters:")
        print(best_fold["parameters"])
        print("Mean validation accuracy:",
              best_fold["validation_accuracy"])

kfold_df = pd.DataFrame([
    {
        "folds": r["folds"],
        "hyperparameters": str(r["parameters"]),
        "training_accuracy": r["training_accuracy"],
        "validation_accuracy": r["validation_accuracy"],
        "comments": r["comments"]
    }
    for r in kfold_results
])

kfold_df.to_csv("partA_kfold_results.csv", index=False)

# ============================================================
# 7. REPEATED K-FOLD CROSS VALIDATION
#    5, 10 and 15 folds, with 5 and 10 repetitions
# ============================================================

print("\n========== REPEATED K-FOLD CROSS VALIDATION ==========")

repeated_results = []
best_repeated_models = {}

for n_splits in [5, 10, 15]:
    for n_repeats in [5, 10]:

        print(
            f"\n--- Repeated K-Fold: "
            f"{n_splits} folds, {n_repeats} repetitions ---"
        )

        cv = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=42
        )

        config_results = []

        for config_no, params in enumerate(configurations, start=1):
            train_scores = []
            val_scores = []

            try:
                for train_idx, val_idx in cv.split(X_dev, y_dev):
                    model = create_model(params)
                    model.fit(X_dev[train_idx], y_dev[train_idx])

                    train_scores.append(
                        model.score(X_dev[train_idx], y_dev[train_idx])
                    )
                    val_scores.append(
                        model.score(X_dev[val_idx], y_dev[val_idx])
                    )

                result = {
                    "folds": n_splits,
                    "repetitions": n_repeats,
                    "parameters": params.copy(),
                    "training_accuracy": float(np.mean(train_scores)),
                    "validation_accuracy": float(np.mean(val_scores)),
                    "comments": classify_behavior(
                        float(np.mean(train_scores)),
                        float(np.mean(val_scores))
                    )
                }

                config_results.append(result)
                repeated_results.append(result)

                print(
                    f"Config {config_no}/{len(configurations)} | "
                    f"Train={result['training_accuracy']:.4f} | "
                    f"Validation={result['validation_accuracy']:.4f}"
                )

            except Exception as exc:
                print(f"Config {config_no} failed: {exc}")

        if config_results:
            best_repeat = max(
                config_results,
                key=lambda r: r["validation_accuracy"]
            )

            best_repeated_models[(n_splits, n_repeats)] = best_repeat

            print("\nBest parameters:", best_repeat["parameters"])
            print("Mean validation accuracy:",
                  best_repeat["validation_accuracy"])

repeated_df = pd.DataFrame([
    {
        "folds": r["folds"],
        "repetitions": r["repetitions"],
        "hyperparameters": str(r["parameters"]),
        "training_accuracy": r["training_accuracy"],
        "validation_accuracy": r["validation_accuracy"],
        "comments": r["comments"]
    }
    for r in repeated_results
])

repeated_df.to_csv(
    "partA_repeated_kfold_results.csv", index=False
)

# ============================================================
# 8. FINAL TESTING OF SELECTED MODELS
#    Train + validation data; test data remains untouched
# ============================================================

print("\n========== FINAL TEST EVALUATION ==========")

selected_models = {
    "Random split": best_random["parameters"]
}

if best_kfold_models:
    # Select the best K-fold result across fold counts
    best_kfold_overall = max(
        best_kfold_models.values(),
        key=lambda r: r["validation_accuracy"]
    )
    selected_models["K-fold"] = best_kfold_overall["parameters"]

if best_repeated_models:
    # Select the best repeated K-fold result across settings
    best_repeated_overall = max(
        best_repeated_models.values(),
        key=lambda r: r["validation_accuracy"]
    )
    selected_models["Repeated K-fold"] = (
        best_repeated_overall["parameters"]
    )

final_results = []

for name, params in selected_models.items():
    final_model = create_model(params)
    final_model.fit(X_dev, y_dev)

    y_pred_test = final_model.predict(X_test_holdout)
    test_acc = accuracy_score(y_test_holdout, y_pred_test)

    print(f"\n--- {name} ---")
    print("Selected hyperparameters:", params)
    print(f"Test accuracy: {test_acc:.4f}")

    print("\nTest confusion matrix:")
    print(confusion_matrix(
        y_test_holdout,
        y_pred_test,
        labels=[0, 1, 2]
    ))

    print("\nTest classification report:")
    print(classification_report(
        y_test_holdout,
        y_pred_test,
        labels=[0, 1, 2],
        target_names=classes,
        zero_division=0
    ))

    final_results.append({
        "method": name,
        "hyperparameters": str(params),
        "test_accuracy": test_acc
    })

pd.DataFrame(final_results).to_csv(
    "partA_final_test_results.csv", index=False
)

print("\nAll Part A experiments completed.")
print("Results saved to CSV files in the working directory.")
