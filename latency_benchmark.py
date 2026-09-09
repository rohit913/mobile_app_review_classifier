"""Measures per-review dashboard latency against the proposal's 10-second target.

Times the same inference path app.py runs. Model loading is excluded because
Streamlit caches it for the session, so it is paid once rather than per review.

Writes reports/dashboard_latency.json. Nothing else in this project produces that file.

Run from the repository root, since paths are resolved relative to the working
directory:

    python latency_benchmark.py

Running it overwrites reports/dashboard_latency.json. Given the run-to-run
variation this project documents, a fresh run will not reproduce the recorded
figure exactly.
"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

warnings.filterwarnings("ignore")

CATEGORY_NAMES = ["Bug Report", "Feature Request", "UX Feedback", "Positive Praise"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print("Device:", DEVICE, flush=True)


def load_model(path):
    tok = BertTokenizer.from_pretrained(path)
    mdl = BertForSequenceClassification.from_pretrained(path).to(DEVICE)
    mdl.eval()
    return tok, mdl


def predict_proba(texts, tokenizer, model):
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]
    enc = tokenizer(texts, max_length=128, padding=True, truncation=True,
                    return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits
        return torch.softmax(logits, dim=1).cpu().numpy()


cat_tok, cat_mdl = load_model("models/bert_category")
sen_tok, sen_mdl = load_model("models/bert_sentiment")

test = pd.read_csv("data/processed/test.csv")
# stratified: 5 reviews per category, seed-fixed
sample = (test.groupby("category_name", group_keys=False)
              .apply(lambda g: g.sample(min(5, len(g)), random_state=42))
              .reset_index(drop=True))
texts = sample["review_text"].astype(str).tolist()
print(f"Benchmarking {len(texts)} reviews\n", flush=True)

from lime.lime_text import LimeTextExplainer
import shap

# warm-up (excluded from timings: first call pays model/kernel init cost)
predict_proba([texts[0]], cat_tok, cat_mdl)
predict_proba([texts[0]], sen_tok, sen_mdl)

classify_t, lime_t, shap_t = [], [], []

for i, txt in enumerate(texts, 1):
    t0 = time.perf_counter()
    cat_probs = predict_proba([txt], cat_tok, cat_mdl)[0]
    predict_proba([txt], sen_tok, sen_mdl)[0]
    t1 = time.perf_counter()
    classify_t.append(t1 - t0)
    cat_idx = int(cat_probs.argmax())

    t0 = time.perf_counter()
    explainer = LimeTextExplainer(class_names=CATEGORY_NAMES)
    explainer.explain_instance(
        txt, lambda ts: predict_proba(ts, cat_tok, cat_mdl),
        labels=(cat_idx,), num_features=10, num_samples=500)
    t1 = time.perf_counter()
    lime_t.append(t1 - t0)

    t0 = time.perf_counter()
    masker = shap.maskers.Text(cat_tok)
    sh = shap.Explainer(lambda ts: predict_proba(ts, cat_tok, cat_mdl), masker)
    sh([txt])
    t1 = time.perf_counter()
    shap_t.append(t1 - t0)

    print(f"[{i}/{len(texts)}] words={len(txt.split()):3d} "
          f"classify={classify_t[-1]:.3f}s lime={lime_t[-1]:.2f}s shap={shap_t[-1]:.2f}s",
          flush=True)


def stats(xs):
    a = np.array(xs)
    return {"mean": round(float(a.mean()), 3),
            "median": round(float(np.median(a)), 3),
            "min": round(float(a.min()), 3),
            "max": round(float(a.max()), 3)}


default_path = np.array(classify_t) + np.array(lime_t)
full_path = default_path + np.array(shap_t)

out = {
    "device": str(DEVICE),
    "n_reviews": len(texts),
    "sampling": "5 reviews per category, stratified from test.csv, random_state=42",
    "lime_num_samples": 500,
    "note": "Model load time excluded (Streamlit caches models via @st.cache_resource, "
            "so it is paid once per session, not per review). One untimed warm-up "
            "inference runs before the loop, so no timed review pays one-off "
            "initialisation cost; all reviews reported here are timed.",
    "classify_only_seconds": stats(classify_t),
    "lime_only_seconds": stats(lime_t),
    "shap_only_seconds": stats(shap_t),
    "default_path_classify_plus_lime_seconds": stats(default_path),
    "full_path_classify_plus_lime_plus_shap_seconds": stats(full_path),
    "criterion_seconds": 10.0,
    "default_path_meets_criterion": bool(default_path.max() < 10.0),
    "full_path_meets_criterion": bool(full_path.max() < 10.0),
}

print("\n=== SUMMARY ===")
print(json.dumps(out, indent=2))
with open("reports/dashboard_latency.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved reports/dashboard_latency.json")
