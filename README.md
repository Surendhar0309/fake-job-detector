# AI-Driven Fake Job Post Detection using Machine Learning and NLP


## Overview
Detects fraudulent online job advertisements. Text fields (title, company profile, description, requirements,
benefits, employment details) are cleaned (tokenization, stop-word removal, lemmatization), converted with
**TF-IDF**, and classified by **Logistic Regression, Random Forest and XGBoost** (Accuracy / Precision / Recall).
**SHAP** explains predictions and a **Fraud Risk Score (0-100%)** maps each posting to Low / Moderate / High risk.

## Structure
```
fake_job_detector/
├── app.py                    # Streamlit UI (layout from your video template)
├── generate_sample_data.py   # synthetic demo data (only for testing the pipeline)
├── requirements.txt
├── src/
│   ├── preprocess.py         # cleaning, lemmatization
│   ├── train.py              # TF-IDF + 3 models + metrics
│   ├── explain.py            # SHAP / feature contributions
│   └── risk.py               # Fraud Risk Score + categories
├── data/                     # put fake_job_postings.csv here
└── models/                   # saved models + metrics (created by train.py)
```

## Run
```bash
pip install -r requirements.txt
# 1. Dataset: download EMSCAD "fake_job_postings.csv" (Kaggle) into data/
#    (or for a quick test: python generate_sample_data.py)
python -m src.train --data data/fake_job_postings.csv
streamlit run app.py
```

## Risk levels
| Score | Category |
|---|---|
| 0-39% | Low Risk |
| 40-69% | Moderate Risk |
| 70-100% | High Risk |

> The included sample data is synthetic and gives near-perfect scores. Report results from the real EMSCAD dataset.
