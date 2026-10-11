import sys
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
from data_loader import load_all, summarize_bad, show_bad

df, bad = load_all()
print(df.shape)
summarize_bad(bad, len(df))

show_bad(bad, "too few fields")
show_bad(bad, "bad label/id")

print("Duplicate IDs:", df["news_id"].duplicated().sum())
print(df["label"].value_counts())
print(df["file"].value_counts())

# are the recovered rows (no metadata) balanced?
missing = df["source_type"].isna()
print("Rows without metadata:", missing.sum())
print(df[missing]["label"].value_counts(normalize=True))
print(df.groupby("file")["source_type"].apply(lambda s: s.isna().mean()))

# which source types exist, and do they give away the label?
print(pd.crosstab(df["source_type"], df["label"]))

# do files differ by label? (file = generation batch)
print(pd.crosstab(df["file"], df["label"], normalize="index"))

# does the text prefix give away the label?
prefix = df["news_context"].str.extract(r"^(Title:|शीर्षक:|शिर्षक:|काठमाण्डौ|काठमाडौँ)")[0]
print(pd.crosstab(prefix.fillna("other"), df["label"]))

# length by label
print(df.groupby("label")["news_context"].apply(lambda s: s.str.len().describe()))

print(df["source_type"].value_counts().head(15))

# exact duplicate texts, and texts that appear with both labels
dup = df["news_context"].duplicated(keep=False)
print("rows in duplicated texts:", dup.sum())
conflict = df.groupby("news_context")["label"].nunique()
print("texts with conflicting labels:", (conflict > 1).sum())

# the 2505-character outlier
print(df.loc[df["news_context"].str.len().idxmax(), "news_context"][:500])

# top words per class (rough tokenizer is fine here)
from collections import Counter
import re
tok = lambda s: re.findall(r"[\u0900-\u097F]+|[a-z]+", s.lower())
for lab in (0, 1):
    c = Counter(w for t in df[df.label == lab]["news_context"] for w in tok(t))
    print(lab, c.most_common(25))