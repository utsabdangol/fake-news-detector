# src/data_loader.py
import csv
import re
from collections import Counter
from pathlib import Path

import pandas as pd

COLS = ["news_context", "label", "category", "source_type",
        "news_id", "generated_at", "meta_intent", "meta_style"]

ID_PATTERN = re.compile(r"NP_\d+")

# project_root/data  (works no matter which folder you run the script from)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# STEP 1: find the files
def find_files(pattern="nepali_news_part_*.csv"):
    files = sorted(str(p) for p in DATA_DIR.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} under {DATA_DIR}")
    return files


# STEP 2 (diagnostic only): which records does the plain csv module see as not 8 fields?
def report_broken_rows(path):
    broken = []
    with open(path, mode="r", encoding="utf-8", errors="ignore", newline="") as f:
        for idx, row in enumerate(csv.reader(f), start=1):
            if len(row) != 8:
                broken.append((idx, len(row)))
    return broken


# STEP 3: parse ONE record (a list of fields from csv.reader)
# Returns (row, None) on success or (None, reason) on failure.
def parse_record(rec):
    rec = list(rec)
    n = len(rec)

    def is_id(s):
        return bool(ID_PATTERN.fullmatch(s.strip()))

    # layout A: text, label, category, id, timestamp
    if n == 5 and is_id(rec[3]):
        text, label, cat, nid, ts = rec
        src = intent = style = None

    # layout B: text, label, category, source_type, id, timestamp [, intent [, style [, junk...]]]
    elif n >= 6 and is_id(rec[4]):
        text, label, cat, src, nid, ts = rec[:6]
        intent = rec[6] if n > 6 else None
        style = rec[7] if n > 7 else None

    # layout C: unquoted commas inside the text; the last 7 columns are fixed
    elif n > 8 and is_id(rec[-6]):
        text = ",".join(rec[:n - 7])
        label, cat, src, nid, ts, intent, style = rec[-7:]

    else:
        return None, "too few fields" if n < 5 else "unrecognized layout"

    if label not in ("0", "1"):
        return None, "bad label/id"
    if ID_PATTERN.search(text):          # a second ID inside the text means glued records
        return None, "glued records"

    # store empty strings as None so missing values are consistent
    return [text.strip(), int(label), cat or None, src or None,
            nid, ts, intent or None, style or None], None


# STEP 4: parse ONE file
def load_file(path):
    rows, bad = [], []
    with open(path, encoding="utf-8", newline="") as f:    # newline="" is required for multi-line fields
        reader = csv.reader(f)
        next(reader)                                       # skip header
        for rec_no, rec in enumerate(reader, start=2):
            row, reason = parse_record(rec)
            if row is None:
                bad.append((rec_no, reason, rec))
            else:
                rows.append(row)
    return pd.DataFrame(rows, columns=COLS), bad


# STEP 5: parse ALL files and combine
def load_all(pattern="nepali_news_part_*.csv"):
    frames, all_bad = [], []
    for path in find_files(pattern):
        df, bad = load_file(path)
        df["file"] = path
        frames.append(df)
        all_bad += [(path, n, reason, rec) for n, reason, rec in bad]
    return pd.concat(frames, ignore_index=True), all_bad


# STEP 6: summarize what was dropped
def summarize_bad(all_bad, total_kept):
    total = total_kept + len(all_bad)
    print(f"Dropped {len(all_bad)} records out of {total} ({len(all_bad) / total:.1%})")
    print("By reason:", Counter(reason for _, _, reason, _ in all_bad))
    print("By file:  ", Counter(path for path, _, _, _ in all_bad))
    print("By field count:", Counter(len(rec) for _, _, _, rec in all_bad))


# STEP 7: look at a few dropped records of one kind
def show_bad(all_bad, reason, n=5):
    for path, rec_no, r, rec in [b for b in all_bad if b[2] == reason][:n]:
        print(path, rec_no, repr(rec)[:300])


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    df, bad = load_all()
    print(df.shape)
    summarize_bad(bad, len(df))