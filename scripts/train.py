# scripts/train.py
import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature

# --------------------------------------------------------------------------------------
# 0) Configure MLflow Tracking
# --------------------------------------------------------------------------------------
# Either keep this hardcoded for local dev or set MLFLOW_TRACKING_URI in your env.
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

EXPERIMENT_NAME = "resume-screener-exp"
REGISTERED_MODEL_NAME = "resume-screener"


def load_dummy_data() -> pd.DataFrame:
    """
    Replace this with your real labeled data later.
    We simulate resume + job pairs with binary labels (1=match, 0=not-match).
    """
    data = pd.DataFrame(
        {
            "resume": [
                "Python developer with FastAPI and ML experience, scikit-learn, Docker",
                "Sales associate with POS systems and customer service",
                "Data scientist, NLP, TF-IDF, cosine similarity, sklearn",
                "Frontend developer React, Next.js, Tailwind, UI/UX",
                "Machine learning engineer, pipelines, MLflow, model registry",
            ],
            "job_description": [
                "Looking for Python developer experienced with FastAPI and ML",
                "Role: retail sales clerk, POS experience and customer handling",
                "Hiring data scientist with NLP skills and scikit-learn",
                "We need React/Next.js frontend developer with Tailwind",
                "Seeking ML engineer familiar with MLflow and deployment",
            ],
            "label": [1, 1, 1, 1, 1],
        }
    )

    # Create mismatches as negatives
    neg = data.copy()
    neg["job_description"] = (
        neg["job_description"].sample(frac=1, random_state=42).values
    )
    neg["label"] = 0

    full = pd.concat([data, neg], ignore_index=True)
    full = full.sample(frac=1, random_state=42).reset_index(drop=True)
    return full


def make_text_pairs(df: pd.DataFrame):
    """
    Create a single text feature by concatenating resume + job description.
    """
    X = (df["resume"] + " || " + df["job_description"]).values
    y = df["label"].values
    return X, y


def main():
    # ----------------------------------------------------------------------------------
    # 1) Data
    # ----------------------------------------------------------------------------------
    df = load_dummy_data()
    X_text, y = make_text_pairs(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.3, random_state=42, stratify=y
    )

    # ----------------------------------------------------------------------------------
    # 2) Model pipeline (baseline)
    # ----------------------------------------------------------------------------------
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    params = {
        "vectorizer": "tfidf",
        "ngram_range": "(1,2)",
        "classifier": "logistic_regression",
        "max_iter": 1000,
    }

    # ----------------------------------------------------------------------------------
    # 3) MLflow: experiment + run
    # ----------------------------------------------------------------------------------
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="baseline-tfidf-logreg") as run:
        # Log params
        mlflow.log_params(params)

        # Train
        pipe.fit(X_train, y_train)

        # Evaluate
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        # Artifacts (sample test rows)
        sample_test = pd.DataFrame({"text": X_test[:5], "label": y_test[:5]})
        sample_path = "sample_test.csv"
        sample_test.to_csv(sample_path, index=False)
        mlflow.log_artifact(sample_path)

        # Tags (nice for filtering)
        mlflow.set_tags(
            {
                "purpose": "baseline",
                "component": "matching",
                "owner": "Omer",
                "env": "local",
            }
        )

        # Signature + input example (removes MLflow warnings)
        train_df_for_sig = pd.DataFrame({"text": X_train})
        input_example = pd.DataFrame({"text": [X_train[0]]})
        # Predict a small slice to infer output schema
        pred_slice = pipe.predict(X_train[:5])
        signature = infer_signature(train_df_for_sig, pred_slice)

        # Log & Register the model (auto-creates/bumps versions)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",  # replaces deprecated artifact_path
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=input_example,
            signature=signature,
        )

        # Optional: tag the latest registered version with a note
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if versions:
            latest_version = max(int(v.version) for v in versions)
            client.set_model_version_tag(
                name=REGISTERED_MODEL_NAME,
                version=str(latest_version),
                key="notes",
                value="baseline tfidf+logreg with signature & input_example",
            )

        print(f"Run ID: {run.info.run_id}")
        print(f"Logged metrics -> accuracy: {acc:.3f}, f1: {f1:.3f}")
        print(f"Model registered as '{REGISTERED_MODEL_NAME}' (check MLflow UI)")


if __name__ == "__main__":
    main()
