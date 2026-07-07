"""
ensemble_model.py  -  Fake-News Detection using Ensemble ML
============================================================
Ensemble of 5 classifiers:
- Logistic Regression
- Random Forest
- Multinomial Naive Bayes
- Gradient Boosting
- Linear SVC

Dataset : WELFake_Dataset.csv  (72,134 articles)
Download: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
Path    : D:\DEEPu\AI Engineer Roadmap\Python Projects\pythonProject\Credible\

Label fix: WELFake uses 0=Fake, 1=Real — corrected at source in load_data()
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import nltk
import gdown

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings("ignore")


# ── NLTK bootstrap ────────────────────────────────────────────────────────────
def _ensure_nltk():
    for pkg in ("stopwords", "punkt"):
        try:
            nltk.data.find(f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()


# ── Text Pre-processing ───────────────────────────────────────────────────────
_stemmer   = PorterStemmer()
_stopwords = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [_stemmer.stem(t) for t in tokens if t not in _stopwords and len(t) > 2]
    return " ".join(tokens)


# ── Load WELFake Dataset ──────────────────────────────────────────────────────
# NOTE: WELFake_Dataset.csv is too large for a plain `docs.google.com/uc?...`
# link — Google Drive serves an HTML "can't scan for viruses" confirmation
# page instead of the raw file for anything over ~25-100MB, which is what
# pandas was choking on (it was parsing HTML as CSV). gdown handles that
# confirmation flow correctly, and we cache the file locally so it's only
# downloaded once per container instance instead of on every rerun.

WELFAKE_FILE_ID = "1bjKkh-EMPQ8lvlHjeuyeOjtt8iDMIe3V"
WELFAKE_LOCAL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "welfake_dataset.csv")


def _download_welfake(local_path: str) -> None:
    url = f"https://drive.google.com/uc?id={WELFAKE_FILE_ID}"
    print(f"\nDownloading WELFake dataset via gdown -> {local_path}")
    gdown.download(url, local_path, quiet=False)

    # Sanity check: a failed/HTML download will be tiny. The real dataset
    # is tens of MB, so anything under ~1MB almost certainly isn't the CSV.
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1_000_000:
        raise RuntimeError(
            "WELFake download failed or returned an unexpected (too small) file. "
            "Check that the Google Drive file is shared as 'Anyone with the link' "
            "and that the file ID is correct."
        )


def load_data():
    # Download once, then reuse the cached local copy on subsequent runs.
    if not os.path.exists(WELFAKE_LOCAL_PATH) or os.path.getsize(WELFAKE_LOCAL_PATH) < 1_000_000:
        _download_welfake(WELFAKE_LOCAL_PATH)
    else:
        print(f"\nUsing cached WELFake dataset -> {WELFAKE_LOCAL_PATH}")

    print("-" * 55)
    df = pd.read_csv(WELFAKE_LOCAL_PATH)
    print(f"Raw rows loaded   : {len(df)}")

    # Rename label column if needed
    if "Label" in df.columns and "label" not in df.columns:
        df.rename(columns={"Label": "label"}, inplace=True)

    # Drop rows where text or label is missing
    df = df.dropna(subset=["text", "label"])

    # Keep only valid labels
    df = df[df["label"].isin([0, 1])]

    # ── FIX LABEL FLIP ────────────────────────────────────────────
    # WELFake: 0=Fake, 1=Real  →  flip so model learns correctly
    df["label"] = df["label"].map({0: 1, 1: 0})

    # ── FIX: WELFake uses 0=Fake, 1=Real ─────────────────────────
    # We keep this convention:  0 = Fake,  1 = Real
    # Verify by printing class distribution
    print(f"  Label 0 (Fake)    : {(df['label'] == 0).sum()}")
    print(f"  Label 1 (Real)    : {(df['label'] == 1).sum()}")

    # Combine title + text for richer features
    if "title" in df.columns:
        df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    else:
        df["content"] = df["text"].fillna("")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["content"])
    after  = len(df)
    if before != after:
        print(f"  Duplicates removed: {before - after}")

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  Total samples     : {len(df)}")
    print("-" * 55)

    return df["content"].tolist(), df["label"].tolist()


# ── Model Save Path ───────────────────────────────────────────────────────────
# Use a path relative to this file so it works both locally and on Streamlit
# Cloud (the old Windows "D:\..." path only existed on your own machine).
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_news_ensemble.pkl")

# Hosted copy of the already-trained model. If MODEL_PATH doesn't exist locally
# (e.g. on a fresh Streamlit Cloud deploy), we download it from here instead of
# training from scratch, which is what was blowing past Streamlit's resource/time
# limits.
MODEL_URL = "https://huggingface.co/datasets/DC-1005/credible-fakenews-model/resolve/main/fake_news_ensemble.pkl"


def _download_model(local_path: str) -> None:
    import requests

    print(f"\nDownloading trained ensemble model -> {local_path}")
    with requests.get(MODEL_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    # Sanity check: a failed download (e.g. an HTML error page) will be tiny.
    if not os.path.exists(local_path) or os.path.getsize(local_path) < 1_000_000:
        raise RuntimeError(
            "Model download failed or returned an unexpected (too small) file. "
            "Check that the Hugging Face URL is correct and publicly accessible."
        )


# ── Build Pipelines ───────────────────────────────────────────────────────────
def _make_pipelines():

    # ── 1. Logistic Regression ────────────────────────────────────
    lr = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=10.0,
            solver="saga",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])

    # ── 2. Random Forest ──────────────────────────────────────────
    rf = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
        )),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # ── 3. Multinomial Naive Bayes ────────────────────────────────
    mnb = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 3),
            sublinear_tf=False,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
        )),
        ("clf", MultinomialNB(alpha=0.01)),
    ])

    # ── 4. Gradient Boosting ──────────────────────────────────────
    gbc = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=3,
            max_df=0.90,
            strip_accents="unicode",
        )),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            max_features="sqrt",
            min_samples_leaf=5,
            random_state=42,
        )),
    ])

    # ── 5. Linear SVC ─────────────────────────────────────────────
    svc = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=100_000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            analyzer="word",
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(
                max_iter=5000,
                C=1.0,
                class_weight="balanced",
                dual=True,
            ),
            cv=3,
        )),
    ])

    return {
        "Logistic Regression": lr,
        "Random Forest":       rf,
        "Naive Bayes":         mnb,
        "Gradient Boosting":   gbc,
        "Linear SVC":          svc,
    }


# ── Train ─────────────────────────────────────────────────────────────────────
def train(save: bool = True):
    print("\n" + "=" * 55)
    print("   Credible -- Training Weighted Ensemble Model")
    print("=" * 55)

    texts, labels = load_data()

    print("\nPreprocessing text (may take a few minutes)...")
    processed = [preprocess(t) for t in texts]
    print("Preprocessing complete.")

    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels,
        test_size=0.20,
        random_state=42,
        stratify=labels
    )
    print(f"\nTrain size : {len(X_train)}")
    print(f"Test size  : {len(X_test)}")

    pipelines = _make_pipelines()

    # Train each model
    print("\nTraining individual models...")
    print("-" * 55)

    model_accuracies = {}
    for name, model in pipelines.items():
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        model_accuracies[name] = acc
        print(f"Accuracy: {acc:.4f}  ({acc * 100:.2f}%)")

    print("-" * 55)

    # Compute weights
    print("\nComputing voting weights based on accuracy...")
    weights = []
    for name, acc in model_accuracies.items():
        w = round(acc ** 2 * 10, 2)
        weights.append(w)
        print(f"  {name:<25} acc={acc:.4f}  weight={w:.2f}")

    # Build weighted ensemble
    ensemble = VotingClassifier(
        estimators=list(pipelines.items()),
        voting="soft",
        weights=weights,
    )

    print("\n  Training Weighted Voting Ensemble...", end=" ", flush=True)
    ensemble.fit(X_train, y_train)
    ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
    print(f"Accuracy: {ens_acc:.4f}  ({ens_acc * 100:.2f}%)")

    report = classification_report(
        y_test,
        ensemble.predict(X_test),
        target_names=["Fake", "Real"],
        output_dict=True
    )

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        ensemble.predict(X_test),
        target_names=["Fake", "Real"]
    ))

    # ── Sanity check: verify labels are correct ───────────────────
    print("\nSanity Check (should show Fake for fake, Real for real):")
    checks = [
        ("SHOCKING: Government putting microchips in vaccines to control population minds", "Fake"),
        ("Federal Reserve raises interest rates citing persistent inflation concerns",       "Real"),
    ]
    for text, expected in checks:
        prob  = ensemble.predict_proba([preprocess(text)])[0]
        # prob[0] = P(Fake),  prob[1] = P(Real)
        label = "Fake" if prob[0] >= 0.5 else "Real"
        status = "OK" if label == expected else "WRONG - labels may be flipped!"
        print(f"  [{status}] Expected={expected} Got={label}  text={text[:55]}...")

    payload = {
        "ensemble":         ensemble,
        "individual":       pipelines,
        "accuracy":         ens_acc,
        "model_accuracies": model_accuracies,
        "weights":          dict(zip(pipelines.keys(), weights)),
        "report":           report,
    }

    if save:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(payload, f)
        print(f"\nModel saved -> {MODEL_PATH}")

    print("\n" + "=" * 55)
    print("   Training Complete!")
    print("=" * 55 + "\n")

    return payload


# ── Load or Train ─────────────────────────────────────────────────────────────
def load_or_train() -> dict:
    # 1. Use the local cached copy if we already have it.
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) >= 1_000_000:
        print(f"Loading saved model from: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # 2. Otherwise, fetch the pre-trained model from Hugging Face instead of
    #    training in-app (Streamlit Cloud doesn't have the resources/time to
    #    train this ensemble on every cold start).
    try:
        _download_model(MODEL_PATH)
        print(f"Loading downloaded model from: {MODEL_PATH}")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Could not download pre-trained model ({e}). Falling back to training from scratch...")

    # 3. Last resort: train locally.
    return train(save=True)


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(text: str, payload: dict) -> dict:
    cleaned = preprocess(text)
    individual_results = {}
    fake_votes, real_votes = 0, 0

    for name, model in payload["individual"].items():
        prob = model.predict_proba([cleaned])[0]
        # prob[0] = P(Fake=0),  prob[1] = P(Real=1)
        fake_p = float(prob[0])
        real_p = float(prob[1])
        label  = "Fake" if fake_p >= 0.5 else "Real"
        conf   = float(max(prob))

        individual_results[name] = {
            "label":      label,
            "confidence": conf,
            "proba":      prob.tolist(),
        }
        if label == "Fake":
            fake_votes += 1
        else:
            real_votes += 1

    ens_prob  = payload["ensemble"].predict_proba([cleaned])[0]
    # ens_prob[0] = P(Fake),  ens_prob[1] = P(Real)
    fake_p    = float(ens_prob[0])
    real_p    = float(ens_prob[1])
    ens_label = "Fake" if fake_p >= 0.5 else "Real"
    ens_conf  = float(max(ens_prob))

    return {
        "label":      ens_label,
        "confidence": ens_conf,
        "fake_prob":  fake_p,
        "real_prob":  real_p,
        "individual": individual_results,
        "votes":      {"Fake": fake_votes, "Real": real_votes},
    }


# ── Run directly to train ─────────────────────────────────────────────────────
if __name__ == "__main__":
    payload = train(save=True)

    print("\nFinal Prediction Test:")
    print("-" * 60)
    samples = [
        ("SHOCKING: Government secretly putting microchips in vaccines to track population", "Fake"),
        ("Federal Reserve raises interest rates citing persistent inflation concerns",        "Real"),
        ("Scientists publish peer-reviewed study on climate change in Nature journal",        "Real"),
        ("BREAKING: Celebrity found dead mainstream media covering up the real cause",        "Fake"),
        ("Drinking bleach cures all diseases according to rogue scientist exposed",           "Fake"),
        ("The unemployment rate fell to 3.8 percent says Bureau of Labor Statistics",        "Real"),
    ]
    print(f"  {'Expected':<8} {'Got':<8} {'Conf':<8} Text")
    print("-" * 60)
    for text, expected in samples:
        r      = predict(text, payload)
        status = "OK" if r["label"] == expected else "WRONG"
        print(f"  [{status:<5}] {expected:<8} {r['label']:<8} {r['confidence']:.0%}  {text[:50]}...")