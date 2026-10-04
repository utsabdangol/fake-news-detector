import re, math
import numpy as np
from collections import Counter

def tokenize(text):
    return re.findall(r"[\u0900-\u097F]+|[a-z]+", text.lower())

class TfidfVectorizer:
    def __init__(self, max_features=5000, min_df=5):
        self.max_features, self.min_df = max_features, min_df

    def fit(self, docs):
        df = Counter()
        for d in docs:
            df.update(set(tokenize(d)))
        vocab = [w for w, c in df.most_common(self.max_features) if c >= self.min_df]
        self.vocab = {w: i for i, w in enumerate(vocab)}
        n = len(docs)
        self.idf = np.zeros(len(vocab))
        for w, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[w])) + 1   # smoothed idf
        return self

    def transform(self, docs):
        X = np.zeros((len(docs), len(self.vocab)), dtype=np.float32)
        for r, d in enumerate(docs):
            counts = Counter(t for t in tokenize(d) if t in self.vocab)
            total = sum(counts.values()) or 1
            for t, c in counts.items():
                X[r, self.vocab[t]] = (c / total) * self.idf[self.vocab[t]]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.where(norms == 0, 1, norms)    # L2 normalize