# src/analyze_cues.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

import math
from collections import Counter

from data_loader import load_all
from preprocess import clean, preprocess, clean_dataframe

df, _ = load_all()
df = clean_dataframe(df)

# 1. duplicates AFTER cleaning (templates that differ only in numbers/spelling)
cleaned = df["news_context"].apply(clean)
print("\nduplicates after cleaning:", cleaned.duplicated().sum())
groups = df.assign(c=cleaned).groupby("c")["label"].nunique()
print("cleaned texts with both labels:", (groups > 1).sum())
print("distinct cleaned texts:", cleaned.nunique(), "of", len(df))

# 2. document frequency per class
def doc_freq(texts):
    c = Counter()
    for t in texts:
        c.update(set(preprocess(t)))
    return c

real = df[df.label == 0]["news_context"]
fake = df[df.label == 1]["news_context"]
df_real, df_fake = doc_freq(real), doc_freq(fake)
n_real, n_fake = len(real), len(fake)

# 3. log ratio of "share of fake docs with the word" vs "share of real docs with the word"
rows = []
for w in set(df_real) | set(df_fake):
    f, r = df_fake[w], df_real[w]
    if f + r < 30:                       # ignore very rare words
        continue
    score = math.log(((f + 1) / (n_fake + 2)) / ((r + 1) / (n_real + 2)))
    rows.append((score, w, f, r))
rows.sort()

print("\nmost REAL-leaning words (score, word, fake docs, real docs)")
for s, w, f, r in rows[:30]:
    print(f"{s:6.2f}  {w:<14} fake={f:<6} real={r}")

print("\nmost FAKE-leaning words")
for s, w, f, r in rows[::-1][:30]:
    print(f"{s:6.2f}  {w:<14} fake={f:<6} real={r}")

# 4. how many documents contain at least one of the strongest fake cues
cues = {w for _, w, _, _ in rows[::-1][:20]}
has_cue = df["news_context"].apply(lambda t: bool(cues & set(preprocess(t))))
print("\nshare of fake docs with a top-20 fake cue:", round(has_cue[df.label == 1].mean(), 3))
print("share of real docs with a top-20 fake cue:", round(has_cue[df.label == 0].mean(), 3))