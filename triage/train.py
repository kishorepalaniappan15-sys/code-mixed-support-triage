"""Train the triage model on Bitext + a small code-mixed seed set.

Usage:
    python -m triage.train            # Bitext (needs internet) + seed set
    python -m triage.train --offline  # seed set only (quick smoke test)
"""
import argparse
import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from triage.normalize import normalize

BITEXT = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
MODEL_PATH = "models/triage.joblib"
SEED_WEIGHT = 15  # code-mixed seed rows count more than English Bitext rows


def load_data(offline: bool) -> pd.DataFrame:
    seed = pd.read_csv("data/seed_train.csv")
    seed["weight"] = SEED_WEIGHT
    frames = [seed]
    if not offline:
        from datasets import load_dataset
        ds = load_dataset(BITEXT, split="train").to_pandas()
        bit = ds.rename(columns={"instruction": "text"})[["text", "category"]]
        bit["weight"] = 1
        frames.append(bit)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df["text"].map(normalize)
    return df


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
    ])
    return Pipeline([("tfidf", features),
                     ("clf", LogisticRegression(max_iter=2000, C=5))])


def train_and_save(offline: bool = False):
    df = load_data(offline)
    print(f"Training on {len(df)} rows, {df['category'].nunique()} categories")
    pipe = build_pipeline()
    pipe.fit(df["text"], df["category"], clf__sample_weight=df["weight"])
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")
    return pipe


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    train_and_save(ap.parse_args().offline)
