# src/check_clean.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

from collections import Counter

from data_loader import load_all
from preprocess import preprocess, clean_dataframe

df, bad = load_all()
print("loaded:", df.shape, "| dropped by loader:", len(bad))

df = clean_dataframe(df)
print("after cleaning:", df.shape)
print(df["label"].value_counts())

# top words per class on the CLEANED tokens
for lab in (0, 1):
    counts = Counter(tok for text in df[df.label == lab]["news_context"]
                     for tok in preprocess(text))
    print(f"label {lab}:", counts.most_common(25))