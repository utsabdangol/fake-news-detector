import sys
sys.stdout.reconfigure(encoding="utf-8")
from preprocess import preprocess, MODES

s = "मन्त्रालयले विद्यार्थीहरूलाई दाबीको प्रमाण दिएको छैन"
for m in MODES:
    print(m, preprocess(s, mode=m))