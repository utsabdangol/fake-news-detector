import re

#  top words per class again
STOPWORDS = {
    "र", "को", "मा", "छ", "यो", "कि", "ले", "लाई", "पनि", "हो", "थियो",
    "गरेको", "गर्ने", "गरे", "गरियो", "भएको", "भने", "तर", "यस", "यसको",
    "उनले", "आज", "आजै", "जसले", "सँग", "बाट", "का", "की", "एक",
}

# variant spellings of the same word (batches of the dataset spell them differently)
SPELLING = {
    "काठमाण्डौ": "काठमाडौं",
    "काठमाण्डौं": "काठमाडौं",
    "काठमाडौँ": "काठमाडौं",
    "निपाल": "नेपाल",
}

DATELINE = re.compile(r"(काठमाण्डौं?|काठमाडौँ|काठमाडौं)\s*[–\-]\s*")


# ---------- text level (used in training AND in the API) ----------

def clean(text, strip_dateline=False, normalize_spelling=True):
    text = text.replace("\u200d", "").replace("\u200c", "")          # zero-width joiners
    text = re.sub(r"(शीर्षक|शिर्षक|Title|Body)\s*:", " ", text)       # format markers
    text = re.sub(r"https?://\S+", " ", text)                        # urls
    if strip_dateline:
        text = DATELINE.sub(" ", text)
    text = re.sub(r"[०-९0-9]+(?:[.,][०-९0-9]+)*", " <num> ", text)   # numbers, incl. 1.5 and 10,000
    text = re.sub(r"[।॥,.!?;:%\"'()\[\]‘’“”\-–]", " ", text)         # punctuation
    text = re.sub(r"\s+", " ", text).strip().lower()
    if normalize_spelling:
        text = " ".join(SPELLING.get(w, w) for w in text.split())
    return text


def tokenize(text, remove_stopwords=True):
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1 or t == "<num>"]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def preprocess(text, **kwargs):
    strip_dateline = kwargs.pop("strip_dateline", False)
    return tokenize(clean(text, strip_dateline=strip_dateline), **kwargs)


# ---------- dataset level (training only, never at prediction time) ----------

def is_degenerate(text, min_tokens=20, max_unique_ratio=0.3):
    toks = text.split()
    return len(toks) >= min_tokens and len(set(toks)) / len(toks) < max_unique_ratio


def clean_dataframe(df):
    n0 = len(df)
    df = df.dropna(subset=["news_context", "label"])
    df = df[df["news_context"].str.len() >= 20]
    n1 = len(df)
    df = df.drop_duplicates(subset="news_context", keep="first")
    n2 = len(df)
    df = df[~df["news_context"].apply(is_degenerate)]
    n3 = len(df)
    print(f"start {n0} | empty/short removed {n0 - n1} | duplicates removed {n1 - n2} "
          f"| degenerate removed {n2 - n3} | final {n3}")
    return df.reset_index(drop=True)
