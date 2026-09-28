"""Text cleaning, tokenization, stop-word removal and lemmatization."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
META_COLUMNS = ["employment_type", "required_experience", "required_education", "industry", "function"]


def _ensure_nltk():
    for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


_ensure_nltk()
_STOP = set(stopwords.words("english"))
_LEMMA = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/digits/punctuation, tokenize, remove stop-words, lemmatize."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)               # HTML
    text = re.sub(r"http\S+|www\.\S+", " url ", text)  # URLs
    text = re.sub(r"\S+@\S+", " email ", text)         # emails
    text = re.sub(r"[^a-z\s]", " ", text)              # punctuation / digits
    tokens = [t for t in text.split() if len(t) > 2 and t not in _STOP]
    return " ".join(_LEMMA.lemmatize(t) for t in tokens)


def build_text(row) -> str:
    """Merge all job-related fields into one document."""
    parts = [str(row.get(c, "") or "") for c in TEXT_COLUMNS + META_COLUMNS]
    return clean_text(" ".join(parts))
