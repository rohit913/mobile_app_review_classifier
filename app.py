"""Streamlit dashboard for the app review classifier.

Classifies a review by category and sentiment, and shows which words drove the
category decision using LIME, with SHAP available as a second opinion.

Run with:  streamlit run app.py
"""
import html
import re
from pathlib import Path

import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer

CATEGORY_NAMES = ["Bug Report", "Feature Request", "UX Feedback", "Positive Praise"]
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]
MAX_TOKENS = 128

# resolve model paths from this file, so the app runs from any working directory
ROOT = Path(__file__).resolve().parent
CATEGORY_DIR = ROOT / "models" / "bert_category"
SENTIMENT_DIR = ROOT / "models" / "bert_sentiment"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="App Review Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Colours chosen to stay legible against both the light and dark Streamlit themes,
# which is why the surfaces are semi-transparent rather than fixed greys.
st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1150px; }
      .hero { padding: 0 0 .4rem 0; }
      .hero h1 { font-size: 2.05rem; font-weight: 700; margin: 0 0 .25rem 0; letter-spacing: -.02em; }
      .hero p  { opacity: .68; margin: 0; font-size: .95rem; }

      .card {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px;
        padding: 1.05rem 1.15rem;
        background: rgba(128,128,128,.055);
        height: 100%;
      }
      .card .label {
        font-size: .72rem; letter-spacing: .09em; text-transform: uppercase;
        opacity: .6; margin-bottom: .35rem;
      }
      .card .value { font-size: 1.5rem; font-weight: 700; line-height: 1.2; margin-bottom: .6rem; }
      .card .conf  { font-size: .8rem; opacity: .7; margin-top: .4rem; }

      .track { height: 7px; border-radius: 4px; background: rgba(128,128,128,.22); overflow: hidden; }
      .fill  { height: 100%; border-radius: 4px; }

      .wordrow { display: flex; align-items: center; gap: .6rem; margin: .3rem 0; }
      .wordrow .w {
        min-width: 132px; text-align: right; font-weight: 600;
        font-size: .88rem; word-break: break-word;
      }
      .wordrow .barwrap { flex: 1; display: flex; align-items: center; }
      .wordrow .bar { height: 17px; border-radius: 4px; }
      .wordrow .sc { font-size: .74rem; opacity: .62; min-width: 56px; font-variant-numeric: tabular-nums; }

      .legend { font-size: .78rem; opacity: .62; margin: .1rem 0 .7rem 0; }
      .pill {
        display: inline-block; padding: .12rem .5rem; border-radius: 99px;
        font-size: .72rem; border: 1px solid rgba(128,128,128,.3);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(path: str):
    tokenizer = BertTokenizer.from_pretrained(path)
    model = BertForSequenceClassification.from_pretrained(path).to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_proba(texts, tokenizer, model):
    if isinstance(texts, str):
        texts = [texts]
    texts = [str(t) for t in texts]
    encodings = tokenizer(
        texts, max_length=MAX_TOKENS, padding=True, truncation=True, return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        logits = model(**encodings).logits
        return torch.softmax(logits, dim=1).cpu().numpy()


def normalise(word):
    """Strip WordPiece markers and punctuation so SHAP and LIME words compare fairly."""
    return re.sub(r"[^a-z0-9]", "", word.lower().replace("##", ""))


def merge_subwords(tokens, values):
    """Recombine tokenizer fragments into whole words, summing their attributions.

    SHAP's Text masker splits on the model's own WordPiece vocabulary, so "faultless"
    arrives as "fault" + "less " and, unmerged, is ranked and displayed as two words.
    That is not just untidy: "fault" shown alone reverses the sense of the word the
    reviewer actually wrote.

    The masker strips the "##" continuation markers but preserves whitespace, so the
    trailing space is what marks a word boundary. Grouping on it rebuilds exactly the
    whitespace-delimited words of the review, punctuation included ("doesn't", not
    "doesn" + "'" + "t"). Attributions are summed across the fragments of a word, which
    is the additive combination SHAP values are defined to support.

    Only display and ranking change; no reported figure depends on this. The 0.6350
    SHAP/LIME agreement cited below comes from notebook 06, not from this path.
    """
    words, scores = [], []
    continuing = False
    for token, value in zip(tokens, values):
        if not token.strip():
            continue
        if continuing and words:
            words[-1] += token.strip()
            scores[-1] += float(value)
        else:
            words.append(token.strip())
            scores.append(float(value))
        continuing = not token[-1:].isspace()
    return list(zip(words, scores))


def bar(pct, colour):
    return f"<div class='track'><div class='fill' style='width:{pct:.1f}%;background:{colour}'></div></div>"


def prediction_card(label, value, confidence, colour):
    return (
        f"<div class='card'><div class='label'>{html.escape(label)}</div>"
        f"<div class='value'>{html.escape(value)}</div>"
        f"{bar(confidence * 100, colour)}"
        f"<div class='conf'>{confidence:.1%} confidence</div></div>"
    )


def explanation_rows(pairs, predicted):
    """Render word attributions as diverging bars, widest contribution first."""
    if not pairs:
        return "<p style='opacity:.6'>No attributions returned for this review.</p>"
    peak = max(abs(s) for _, s in pairs) or 1.0
    out = []
    for word, score in pairs:
        width = abs(score) / peak * 100
        colour = "#2e9e5b" if score > 0 else "#c0392b"
        out.append(
            f"<div class='wordrow'><div class='w'>{html.escape(str(word))}</div>"
            f"<div class='barwrap' style='width:100%'>"
            f"<div class='bar' style='width:{width:.1f}%;background:{colour}'></div></div>"
            f"<div class='sc'>{score:+.3f}</div></div>"
        )
    return (
        f"<div class='legend'>Green pushes the prediction toward "
        f"<b>{html.escape(predicted)}</b>, red pushes away from it. "
        f"Bar length is relative to the strongest word.</div>" + "".join(out)
    )


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### About")
    st.write(
        "MSc dissertation project. A BERT model fine-tuned on 1,540 training reviews "
        "from a 2,200-review manually labelled Google Play dataset, with word-level "
        "explanations."
    )

    st.markdown("### Test-set performance")
    a, b = st.columns(2)
    a.metric("Category macro F1", "0.6754")
    b.metric("Accuracy", "0.7212")
    st.caption("Single held-out split; five-fold cross-validation gives 0.6872 ± 0.0302. Best of six models: SVM 0.6596, LLaMA-3 zero-shot 0.6210.")

    st.divider()
    st.markdown("### Settings")
    lime_samples = st.slider(
        "LIME perturbation samples", 200, 1000, 500, 100,
        help="More samples give a steadier explanation but take longer.",
    )
    show_shap = st.checkbox("Also compute SHAP", value=False,
                            help="Roughly doubles the response time.")

    st.divider()
    st.caption(f"Compute device: `{DEVICE.type}`  ·  max {MAX_TOKENS} tokens per review")

# ----------------------------------------------------------------------------- header
st.markdown(
    "<div class='hero'><h1>App Review Classifier</h1>"
    "<p>Sorts a Google Play review into Bug Report, Feature Request, UX Feedback or "
    "Positive Praise, predicts its sentiment, and shows which words drove the "
    "<b>category</b> decision.</p></div>",
    unsafe_allow_html=True,
)
st.write("")

if not CATEGORY_DIR.exists() or not SENTIMENT_DIR.exists():
    st.error(
        f"Trained models not found.\n\n"
        f"Expected `{CATEGORY_DIR}` and `{SENTIMENT_DIR}`.\n\n"
        f"Run notebook `04_bert_training.ipynb` to produce them."
    )
    st.stop()

with st.spinner("Loading models…"):
    category_tokenizer, category_model = load_model(str(CATEGORY_DIR))
    sentiment_tokenizer, sentiment_model = load_model(str(SENTIMENT_DIR))

EXAMPLES = {
    "Bug report": "The app keeps crashing every time I try to upload a photo. Please fix this.",
    "Feature request": "Would be great if you could add a dark mode and let me export my data to CSV.",
    "Ambiguous UX": "The menu is so confusing, I can never find the settings I need.",
}

if "review_text" not in st.session_state:
    st.session_state.review_text = ""

st.markdown("**Try an example**")
cols = st.columns(len(EXAMPLES) + 1)
for col, (name, text) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.review_text = text
if cols[-1].button("Clear", use_container_width=True):
    st.session_state.review_text = ""

review_text = st.text_area(
    "Review text", key="review_text", height=130,
    placeholder="Paste a Google Play review here, or pick an example above…",
)

submitted = st.button("Classify review", type="primary", use_container_width=True)

# ----------------------------------------------------------------------------- results
if submitted:
    if not review_text.strip():
        st.warning("Enter a review first.")
        st.stop()

    n_tokens = len(category_tokenizer.tokenize(review_text))
    if n_tokens > MAX_TOKENS:
        st.warning(
            f"This review is {n_tokens} tokens and the model reads the first "
            f"{MAX_TOKENS}. The explanation below covers only that portion."
        )

    with st.spinner("Classifying…"):
        cat_probs = predict_proba([review_text], category_tokenizer, category_model)[0]
        sent_probs = predict_proba([review_text], sentiment_tokenizer, sentiment_model)[0]

    cat_idx, sent_idx = int(cat_probs.argmax()), int(sent_probs.argmax())
    predicted = CATEGORY_NAMES[cat_idx]

    st.write("")
    left, right = st.columns(2)
    left.markdown(
        prediction_card("Predicted category", predicted, float(cat_probs[cat_idx]), "#3b7dd8"),
        unsafe_allow_html=True,
    )
    right.markdown(
        prediction_card("Predicted sentiment", SENTIMENT_NAMES[sent_idx],
                        float(sent_probs[sent_idx]), "#8e5bd0"),
        unsafe_allow_html=True,
    )

    if cat_probs[cat_idx] < 0.5:
        st.info(
            "Confidence is below 50%, so the model is genuinely unsure. Reviews that "
            "sit between UX Feedback and Bug Report are the hardest case in this "
            "dataset, and are the known-difficult boundary."
        )

    with st.expander("Full probability distribution"):
        d, e = st.columns(2)
        with d:
            st.caption("Category")
            for name, p in sorted(zip(CATEGORY_NAMES, cat_probs), key=lambda x: -x[1]):
                st.markdown(
                    f"<div style='display:flex;gap:.6rem;align-items:center;margin:.28rem 0'>"
                    f"<div style='min-width:118px;font-size:.85rem'>{name}</div>"
                    f"<div style='flex:1'>{bar(p * 100, '#3b7dd8')}</div>"
                    f"<div style='font-size:.76rem;opacity:.65;min-width:44px'>{p:.1%}</div></div>",
                    unsafe_allow_html=True,
                )
        with e:
            st.caption("Sentiment")
            for name, p in sorted(zip(SENTIMENT_NAMES, sent_probs), key=lambda x: -x[1]):
                st.markdown(
                    f"<div style='display:flex;gap:.6rem;align-items:center;margin:.28rem 0'>"
                    f"<div style='min-width:118px;font-size:.85rem'>{name}</div>"
                    f"<div style='flex:1'>{bar(p * 100, '#8e5bd0')}</div>"
                    f"<div style='font-size:.76rem;opacity:.65;min-width:44px'>{p:.1%}</div></div>",
                    unsafe_allow_html=True,
                )

    st.write("")
    tabs = st.tabs(["LIME explanation", "SHAP explanation"] if show_shap else ["LIME explanation"])

    with tabs[0]:
        with st.spinner("Computing LIME explanation…"):
            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=CATEGORY_NAMES)
            exp = explainer.explain_instance(
                review_text,
                lambda texts: predict_proba(texts, category_tokenizer, category_model),
                labels=(cat_idx,), num_features=10, num_samples=lime_samples,
            )
        lime_pairs = exp.as_list(label=cat_idx)
        st.markdown(explanation_rows(lime_pairs, predicted), unsafe_allow_html=True)
        st.caption(
            f"LIME perturbs the review {lime_samples} times and fits a simple local model "
            f"to approximate the classifier's behaviour around this example."
        )

    if show_shap:
        with tabs[1]:
            with st.spinner("Computing SHAP explanation…"):
                import shap

                masker = shap.maskers.Text(category_tokenizer)
                shap_explainer = shap.Explainer(
                    lambda texts: predict_proba(texts, category_tokenizer, category_model),
                    masker,
                )
                shap_values = shap_explainer([review_text])
                values = shap_values[0].values[:, cat_idx]
                merged = merge_subwords(shap_values[0].data, values)
                pairs = sorted(merged, key=lambda x: abs(x[1]), reverse=True)
                top_pairs = pairs[:5]

            st.markdown(explanation_rows(top_pairs, predicted), unsafe_allow_html=True)

            shap_set = {normalise(w) for w, _ in top_pairs} - {""}
            lime_set = {normalise(w) for w, _ in lime_pairs[:5]} - {""}
            shared = shap_set & lime_set
            overlap = len(shared) / (min(len(shap_set), len(lime_set)) or 1)

            st.divider()
            m1, m2 = st.columns([1, 2])
            m1.metric("SHAP / LIME overlap", f"{overlap:.0%}")
            m2.markdown(
                "<div style='padding-top:.55rem'>"
                + (" ".join(f"<span class='pill'>{html.escape(w)}</span>" for w in sorted(shared))
                   if shared else "<span style='opacity:.6'>No words in common.</span>")
                + "</div>",
                unsafe_allow_html=True,
            )
            st.caption(
                "Across 20 test reviews the two methods averaged 0.6350 overlap on their top five words. "
                "They agree moderately, and agreement alone does not make an "
                "explanation correct; see reports/coherence_assessment_completed.csv."
            )
