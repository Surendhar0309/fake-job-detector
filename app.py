"""Streamlit UI - layout modelled on the Hybrid Healthcare Disease Prediction demo:
sidebar with About + Risk Categories, main form for job details, prediction + risk gauge + SHAP factors.

Run:  streamlit run app.py
"""
import json
import os
import joblib
import pandas as pd
import streamlit as st

from src.explain import top_factors
from src.preprocess import build_text
from src.risk import fraud_risk_score, risk_category

st.set_page_config(page_title="Fake Job Post Detector", page_icon="🕵️", layout="wide")
MODEL_FILES = {"Logistic Regression": "logistic_regression", "Random Forest": "random_forest", "XGBoost": "xgboost"}


@st.cache_resource
def load_artifacts():
    if not os.path.exists("models/tfidf.joblib"):
        return None
    tfidf = joblib.load("models/tfidf.joblib")
    models = {n: joblib.load(f"models/{f}.joblib") for n, f in MODEL_FILES.items()}
    best = open("models/best_model.txt").read().strip()
    metrics = json.load(open("models/metrics.json"))
    return tfidf, models, best, metrics


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("ℹ️ About This System")
    st.write("This tool estimates whether an online job advertisement is **genuine or fake** using NLP (TF-IDF) and machine learning:")
    st.markdown("- Logistic Regression\n- Random Forest\n- XGBoost")
    st.header("📊 Risk Categories")
    st.markdown("🟢 **Low Risk** — 0% to 39%\n\n🟠 **Moderate Risk** — 40% to 69%\n\n🔴 **High Risk** — 70% to 100%")

# ---------------- Main ----------------
st.markdown("<h2 style=\"text-align:center\">🕵️ AI-Driven Fake Job Post Detection</h2>", unsafe_allow_html=True)
st.caption("Machine Learning + NLP with SHAP interpretability and a Fraud Risk Score")

art = load_artifacts()
if art is None:
    st.error("No trained models found. Run `python -m src.train --data data/fake_job_postings.csv` first.")
    st.stop()
tfidf, models, best, metrics = art
st.success("Trained models loaded successfully. Enter the job posting details below to get a prediction.")

st.subheader("Enter Job Details")
c1, c2 = st.columns(2)
with c1:
    title = st.text_input("Job Title", "Work From Home Data Entry")
    company_profile = st.text_area("Company Profile", "", height=100)
    description = st.text_area("Job Description", "Earn 5000 dollars weekly from home. Send your bank details to start immediately.", height=140)
with c2:
    requirements = st.text_area("Requirements", "No qualification required.", height=100)
    benefits = st.text_area("Benefits", "Unlimited earning, instant weekly payout.", height=100)
    e1, e2 = st.columns(2)
    employment_type = e1.selectbox("Employment Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other"])
    required_experience = e2.selectbox("Experience", ["Not Applicable", "Entry level", "Associate", "Mid-Senior level", "Director", "Executive"])
    required_education = e1.text_input("Education", "")
    industry = e2.text_input("Industry", "")

model_name = st.selectbox("Model", list(models), index=list(models).index(best))

if st.button("🔍 Analyse Job Post", type="primary"):
    row = dict(title=title, company_profile=company_profile, description=description, requirements=requirements,
               benefits=benefits, employment_type=employment_type, required_experience=required_experience,
               required_education=required_education, industry=industry, function="")
    text = build_text(row)
    model = models[model_name]
    p_fake = float(model.predict_proba(tfidf.transform([text]))[0][1])
    score = fraud_risk_score(p_fake)
    label, colour = risk_category(score)

    st.markdown("---")
    m1, m2 = st.columns(2)
    m1.metric("Fraud Risk Score", f"{score}%")
    m2.markdown(f"### :{colour}[{label}]")
    st.progress(min(int(score), 100))
    st.write("**Verdict:** " + ("⚠️ Likely FAKE job posting" if p_fake >= 0.5 else "✅ Likely GENUINE job posting"))

    st.subheader("Why this prediction? (SHAP / feature contributions)")
    factors = top_factors(model, tfidf, text, k=10)
    if factors:
        df = pd.DataFrame(factors, columns=["Word / phrase", "Impact"]).set_index("Word / phrase")
        st.bar_chart(df)
        st.caption("Positive impact pushes towards FAKE, negative towards GENUINE.")
    else:
        st.info("No known vocabulary terms found in this posting.")

with st.expander("📈 Model comparison (test set)"):
    st.dataframe(pd.DataFrame(metrics).T)
