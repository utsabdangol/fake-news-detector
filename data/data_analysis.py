# data/data_analysis.py
from data_loader import *

# 1. check the files are found
print(find_files())

# 2. see what the plain csv module complains about (optional)
for f in find_files():
    print(f, report_broken_rows(f)[:5])

# 3. test the parser on a single line before trusting it
sample = "काठमाण्डौं – ...,0,agriculture,online_portal,NP_0001326,2026-01-15T16:47:27,informative,formal"
print(parse_line(sample))

# 4. load everything
df, bad = load_all()
print(df.shape)

# 5. look at what was dropped
summarize_bad(bad, len(df))
for path, n, reason, line in bad[:5]:
    print(path, n, reason, repr(line[:150]))