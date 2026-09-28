"""Fraud Risk Score and risk categories (same banding idea as the Streamlit template)."""


def risk_category(score: float):
    """score in [0, 100] -> (label, colour)."""
    if score < 40:
        return "Low Risk", "green"
    if score < 70:
        return "Moderate Risk", "orange"
    return "High Risk", "red"


def fraud_risk_score(proba_fake: float) -> float:
    return round(float(proba_fake) * 100, 1)
