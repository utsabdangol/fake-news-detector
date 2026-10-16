# src/split.py
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from data_loader import load_all
from preprocess import clean, clean_dataframe

BLOCK = 50          # consecutive news_ids that stay together
SEED = 42
FRACTIONS = (0.70, 0.15, 0.15)     # train, validation, test


def load_clean_table():
    df, _ = load_all()
    df = clean_dataframe(df)
    df["cleaned"] = df["news_context"].apply(clean)

    # 1. drop every row whose cleaned text appears with both labels
    n_labels = df.groupby("cleaned")["label"].transform("nunique")
    print("rows dropped (cleaned text has both labels):", int((n_labels > 1).sum()))
    df = df[n_labels == 1]

    # 2. keep one row per cleaned text
    before = len(df)
    df = df.drop_duplicates(subset="cleaned", keep="first")
    print("rows dropped (duplicate cleaned text):", before - len(df))

    # 3. numeric id and block
    df = df.assign(id_num=df["news_id"].str.extract(r"(\d+)")[0].astype(int))
    df["block"] = df["id_num"] // BLOCK
    return df.reset_index(drop=True)


def make_split(df, seed=SEED, fractions=FRACTIONS):
    rng = np.random.default_rng(seed)
    blocks = df["block"].unique()
    rng.shuffle(blocks)

    n = len(blocks)
    n_train = int(n * fractions[0])
    n_val = int(n * fractions[1])
    train_b = set(blocks[:n_train])
    val_b = set(blocks[n_train:n_train + n_val])

    def which(b):
        return "train" if b in train_b else "val" if b in val_b else "test"

    df = df.assign(split=df["block"].map(which))
    return (df[df.split == "train"].reset_index(drop=True),
            df[df.split == "val"].reset_index(drop=True),
            df[df.split == "test"].reset_index(drop=True))


def get_splits():
    return make_split(load_clean_table())


if __name__ == "__main__":
    train, val, test = get_splits()

    print()
    for name, part in (("train", train), ("val", val), ("test", test)):
        share_fake = part["label"].mean()
        print(f"{name:<5} {len(part):>6} rows | fake {share_fake:.3f} | blocks {part['block'].nunique()}")

    # leakage checks: all three must print 0
    print("\ncleaned texts shared train/test:", len(set(train.cleaned) & set(test.cleaned)))
    print("cleaned texts shared train/val: ", len(set(train.cleaned) & set(val.cleaned)))
    print("blocks shared train/test:       ", len(set(train.block) & set(test.block)))