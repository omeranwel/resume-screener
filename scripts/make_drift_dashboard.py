# scripts/make_drift_dashboard.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Evidently 0.4.x API
from evidently.report import Report
from evidently.metrics import DataDriftTable


def load_dummy_data() -> pd.DataFrame:
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

    neg = data.copy()
    neg["job_description"] = (
        neg["job_description"].sample(frac=1, random_state=42).values
    )
    neg["label"] = 0
    full = pd.concat([data, neg], ignore_index=True)
    return full.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    df = load_dummy_data()
    ref, cur = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["label"]
    )

    # Tabular frames Evidently can analyze
    reference = pd.DataFrame(
        {"text": ref["resume"] + " || " + ref["job_description"], "label": ref["label"]}
    )
    current = pd.DataFrame(
        {"text": cur["resume"] + " || " + cur["job_description"], "label": cur["label"]}
    )

    # Build a data drift report
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs("dashboards", exist_ok=True)
    out_html = os.path.join("dashboards", "data_drift.html")
    report.save_html(out_html)
    print(f"✅ Data drift report saved at: {out_html}")
    print("Serve it via: cd dashboards && python -m http.server 7000")
    print("Open http://127.0.0.1:7000/data_drift.html")


if __name__ == "__main__":
    main()
