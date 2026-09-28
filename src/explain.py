"""SHAP interpretability: which words pushed a posting towards fake."""
import numpy as np
import shap


def top_factors(model, tfidf, cleaned_text: str, k: int = 10):
    """Return [(word, contribution)] sorted by absolute impact for one posting."""
    X = tfidf.transform([cleaned_text])
    names = np.array(tfidf.get_feature_names_out())
    if type(model).__name__ == "LogisticRegression":
        vals = X.toarray()[0] * model.coef_[0]  # exact linear contribution
    else:
        sv = shap.TreeExplainer(model).shap_values(X.toarray())
        sv = sv[1] if isinstance(sv, list) else np.asarray(sv)
        vals = sv[0]
        if vals.ndim > 1:  # (features, classes) layout
            vals = vals[:, 1]
    present = X.toarray()[0] != 0          # only words that appear in this posting
    vals = np.where(present, vals, 0.0)
    idx = np.argsort(-np.abs(vals))[:k]
    return [(str(names[i]), float(vals[i])) for i in idx if vals[i] != 0]
