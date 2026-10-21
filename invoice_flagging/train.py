from pathlib import Path
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from data_preprocessing import prepare_data
from model_evaluation import evaluate_classifier

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # 1. Load and prepare data.
    X, y, df = prepare_data()

    print("Dataset shape:", X.shape)
    print("\nTarget distribution:")
    print(y.value_counts())

    # 2. Split the data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=11,
        stratify=y,
    )

    # 3. Define models using preprocessing pipelines.
    models = {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=11,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "Decision Tree Classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        max_depth=10,
                        random_state=11,
                    ),
                ),
            ]
        ),
        "Random Forest Classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        max_depth=7,
                        random_state=11,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    # 4. Train and evaluate baseline models.
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        evaluate_classifier(
            model,
            X_test,
            y_test,
            name,
        )

    # 5. Tune Random Forest using F1 score.
    rf_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=11,
                    n_jobs=1,
                ),
            ),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 4, 5, 6],
        "classifier__min_samples_split": [2, 3, 5],
        "classifier__min_samples_leaf": [1, 2, 5],
        "classifier__criterion": ["gini", "entropy"],
    }

    scorer = make_scorer(f1_score, pos_label=1)

    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=2,
        n_jobs=-1,
    )

    print("\nStarting Random Forest Grid Search...")
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest Cross-Validation F1 Score:",
        round(grid_search.best_score_, 3),
    )

    # 6. Evaluate the tuned model on the held-out test set.
    evaluate_classifier(
        best_model,
        X_test,
        y_test,
        "Tuned Random Forest Classifier",
    )

    # 7. Save the trained model and metadata as .pkl files.
    model_path = MODEL_DIR / "invoice_risk_model.pkl"
    metadata_path = MODEL_DIR / "model_metadata.pkl"

    with open(model_path, "wb") as file:
        pickle.dump(best_model, file)

    metadata = {
        "features": list(X.columns),
        "best_params": grid_search.best_params_,
        "cv_f1_score": grid_search.best_score_,
        "target_name": "flag_invoice",
        "target_labels": {
            0: "Normal invoice",
            1: "Flag for manual review",
        },
    }

    with open(metadata_path, "wb") as file:
        pickle.dump(metadata, file)

    print("\nModel training completed.")
    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
