import sys
sys.stdout.reconfigure(encoding="utf-8")

import math
from collections import Counter

from data_loader import load_all
from preprocess import clean, preprocess, clean_dataframe, MODES

df, _ = load_all()
df = clean_dataframe(df)

# 1. duplicates AFTER cleaning (templates that differ only in numbers/spelling)
cleaned = df["news_context"].apply(clean)
print("\n=== DUPLICATES AFTER CLEANING ===")
print("duplicates after cleaning:", cleaned.duplicated().sum())
groups = df.assign(c=cleaned).groupby("c")["label"].nunique()
print("cleaned texts with both labels:", (groups > 1).sum())
print("distinct cleaned texts:", cleaned.nunique(), "of", len(df))

real = df[df.label == 0]["news_context"]
fake = df[df.label == 1]["news_context"]
n_real, n_fake = len(real), len(fake)


def doc_freq(texts, mode):
    c = Counter()
    for t in texts:
        c.update(set(preprocess(t, mode=mode)))
    return c


for mode in MODES:
    df_real, df_fake = doc_freq(real, mode), doc_freq(fake, mode)

    rows = []
    for w in set(df_real) | set(df_fake):
        f, r = df_fake[w], df_real[w]
        if f + r < 30:                                   # ignore very rare words
            continue
        score = math.log(((f + 1) / (n_fake + 2)) / ((r + 1) / (n_real + 2)))
        rows.append((score, w, f, r))
    rows.sort()

    print(f"\n=== MODE: {mode} | vocabulary (>=30 docs): {len(rows)} ===")

    print("most REAL-leaning (top 15)")
    for s, w, f, r in rows[:15]:
        print(f"{s:6.2f}  {w:<16} fake={f:<6} real={r}")

    print("most FAKE-leaning (top 15)")
    for s, w, f, r in rows[::-1][:15]:
        print(f"{s:6.2f}  {w:<16} fake={f:<6} real={r}")

    cues = {w for _, w, _, _ in rows[::-1][:20]}
    has_cue = df["news_context"].apply(lambda t: bool(cues & set(preprocess(t, mode=mode))))
    print(f"share of fake docs with a top-20 fake cue: {has_cue[df.label == 1].mean():.3f}")
    print(f"share of real docs with a top-20 fake cue: {has_cue[df.label == 0].mean():.3f}")