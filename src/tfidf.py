import json
import math
from collections import Counter

import numpy as np

from preprocess import preprocess, MODES


class TfidfVectorizer:
    def __init__(self, max_features=3000, min_df=5, mode="raw", strip_dateline=False):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.max_features = max_features
        self.min_df = min_df
        self.mode = mode
        self.strip_dateline = strip_dateline
        self.vocab = {}
        self.idf = None

    # the ONE place where text becomes tokens, so training and the API always agree
    def tokenizer(self, doc):
        return preprocess(doc, mode=self.mode, strip_dateline=self.strip_dateline)

    # learn vocabulary and idf from TRAINING texts only
    def fit(self, docs):
        n = len(docs)
        df = Counter()
        for doc in docs:
            df.update(set(self.tokenizer(doc)))               # count each word once per document

        kept = [(w, c) for w, c in df.items() if c >= self.min_df]
        kept.sort(key=lambda x: (-x[1], x[0]))                 # most common first, ties alphabetical
        kept = kept[:self.max_features]

        self.vocab = {w: i for i, (w, _) in enumerate(kept)}
        self.idf = np.array([math.log((1 + n) / (1 + c)) + 1 for _, c in kept],
                            dtype=np.float32)                  # smoothed idf, always > 0
        return self

    # one document -> (indices, values), L2-normalized
    def _vectorize(self, doc):
        tokens = [t for t in self.tokenizer(doc) if t in self.vocab]
        if not tokens:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
        counts = Counter(tokens)
        total = len(tokens)
        idx = np.fromiter((self.vocab[t] for t in counts), dtype=np.int32, count=len(counts))
        tf = np.fromiter((c / total for c in counts.values()), dtype=np.float32, count=len(counts))
        vals = tf * self.idf[idx]
        norm = np.linalg.norm(vals)
        return idx, (vals / norm if norm > 0 else vals)

    # list of (indices, values) pairs: memory-light, good for big training sets
    def transform_sparse(self, docs):
        return [self._vectorize(d) for d in docs]

    # dense matrix: fine for small sets, one document at the API, or toy tests
    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for r, d in enumerate(docs):
            idx, vals = self._vectorize(d)
            X[r, idx] = vals
        return X

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)

    # the tokenizer settings are saved with the vocabulary, so the API cannot use different ones
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"mode": self.mode,
                       "strip_dateline": self.strip_dateline,
                       "vocab": self.vocab,
                       "idf": self.idf.tolist()}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        v = cls(mode=data["mode"], strip_dateline=data["strip_dateline"])
        v.vocab = data["vocab"]
        v.idf = np.array(data["idf"], dtype=np.float32)
        return v


if __name__ == "__main__":
    docs = ["सरकारले नयाँ कार्यक्रम घोषणा गर्यो", "सबै नागरिकलाई १०० रुपैयाँ", "सरकारले सबै बन्द गर्यो"]
    for mode in MODES:
        v = TfidfVectorizer(max_features=50, min_df=1, mode=mode).fit(docs)
        print(mode, len(v.vocab), "features")
    v = TfidfVectorizer(max_features=50, min_df=1).fit(docs)
    X = v.transform(docs)
    print(v.vocab)
    print(X.round(3))
    print("row norms (should be 1):", np.linalg.norm(X, axis=1).round(3))