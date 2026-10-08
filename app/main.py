@app.post("/predict")
def predict(req: UrlRequest):
    text = scrape(req.url)
    x = vec.transform([clean(text)])
    score = float(svm.decision(x)[0])
    return {"label": "real" if score >= 0 else "fake", "score": score}