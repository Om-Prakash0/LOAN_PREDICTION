import joblib
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "loan_data.csv"
MODEL_PATH = "loan_model.pkl"

# ✅ Updated feature columns based on YOUR dataset
FEATURE_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]


def load_and_prepare_data(path: str):
    df = pd.read_csv(path)

    # -------------------------------
    # Clean column names
    # -------------------------------
    df.columns = df.columns.str.strip()

    # -------------------------------
    # Clean values (remove spaces)
    # -------------------------------
    df["education"] = df["education"].str.strip()
    df["self_employed"] = df["self_employed"].str.strip()
    df["loan_status"] = df["loan_status"].str.strip()

    # -------------------------------
    # Encoding
    # -------------------------------
    df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
    df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
    df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    # -------------------------------
    # Features & Target
    # -------------------------------
    X = df[FEATURE_COLUMNS]
    y = df["loan_status"]

    return X, y


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(32, 16),  # improved ANN
                    activation="relu",
                    solver="adam",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def main():
    X, y = load_and_prepare_data(DATA_PATH)

    # -------------------------------
    # Train-Test Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # -------------------------------
    # Train Model
    # -------------------------------
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # -------------------------------
    # Evaluation
    # -------------------------------
    y_pred = pipeline.predict(X_test)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\nConfusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    # -------------------------------
    # K-Fold Validation
    # -------------------------------
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=kfold)

    print("\nK-Fold Accuracy: %.2f%%" % (scores.mean() * 100))

    # -------------------------------
    # Save Model
    # -------------------------------
    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": FEATURE_COLUMNS,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved successfully to {MODEL_PATH}")


if __name__ == "__main__":
    main()