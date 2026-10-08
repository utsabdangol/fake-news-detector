# data/data_loader.py
import csv
import glob
import re
from collections import Counter

import pandas as pd

COLS = ["news_context", "label", "category", "source_type",
        "news_id", "generated_at", "meta_intent", "meta_style"]


# STEP 1: find the files
def find_files(pattern="datasets/nepali_news_part_*.csv"):
    return sorted(glob.glob(pattern))


# STEP 2 (diagnostic only): which rows does the csv module consider broken?
def report_broken_rows(path):
    broken = []
    with open(path, mode="r", encoding="utf-8", errors="ignore") as f:
        for idx, row in enumerate(csv.reader(f), start=1):
            if len(row) != 8:
                broken.append((idx, len(row)))
    return broken


# STEP 3: parse ONE line. Returns (row, None) on success or (None, reason) on failure.
def parse_line(line):
    parts = line.rsplit(",", 7)           # split from the right; the last 7 columns are fixed
    if len(parts) != 8:
        return None, "too few fields"

    text = parts[0].strip().strip('"').replace('""', '"')
    label, cat, src, nid, ts, intent, style = parts[1:]

    if label not in ("0", "1") or not re.fullmatch(r"NP_\d+", nid):
        return None, "bad label/id"
    if re.search(r"NP_\d+", text):        # an ID inside the text means two records were glued together
        return None, "glued records"

    return [text, int(label), cat, src, nid, ts, intent, style], None


# STEP 4: parse ONE file using parse_line
def load_file(path):
    rows, bad = [], []
    with open(path, encoding="utf-8", newline="") as f:
        f.readline()                      # skip header
        for n, line in enumerate(f, start=2):
            row, reason = parse_line(line.rstrip("\r\n"))
            if row is None:
                bad.append((n, reason, line))
            else:
                rows.append(row)
    return pd.DataFrame(rows, columns=COLS), bad


# STEP 5: parse ALL files and combine
def load_all(pattern="datasets/nepali_news_part_*.csv"):
    frames, all_bad = [], []
    for path in find_files(pattern):
        df, bad = load_file(path)
        df["file"] = path
        frames.append(df)
        all_bad += [(path, n, reason, line) for n, reason, line in bad]
    return pd.concat(frames, ignore_index=True), all_bad


# STEP 6: summarize what was dropped
def summarize_bad(all_bad, total_rows):
    print(f"Dropped {len(all_bad)} lines out of {total_rows + len(all_bad)}")
    print(Counter(reason for _, _, reason, _ in all_bad))