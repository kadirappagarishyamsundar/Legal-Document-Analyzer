import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import docx
import io
from deep_translator import GoogleTranslator
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import re
import shutil

# -----------------------------
# STREAMLIT CONFIG
# -----------------------------
st.set_page_config(
    page_title="Legal Document Analyzer",
    layout="wide",
    page_icon="⚖️"
)

# -----------------------------
# TESSERACT AUTO DETECT
# -----------------------------
pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# -----------------------------
# LOAD AI MODELS
# -----------------------------
@st.cache_resource
def load_all_engines():

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-base-mnli"
    )

    semantic_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

    ner_model = pipeline(
        "ner",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple"
    )

    return classifier, semantic_model, summarizer, ner_model


classifier, semantic_model, summarizer, ner_model = load_all_engines()

# -----------------------------
# HELPERS
# -----------------------------
def merge_fragmented_tokens(entities):

    merged = []

    for ent in entities:

        text = ent["word"]
        label = ent["entity_group"]

        if text.startswith("##") and merged:
            merged[-1]["word"] += text.replace("##", "")

        else:
            merged.append({
                "word": text,
                "entity_group": label
            })

    for e in merged:
        e["word"] = e["word"].strip()

    return merged


def highlight_xai(text):

    highlights = {

        r"\b(sale|deed|agreement|vendor|purchaser|mortgage|lease|agent|attorney)\b": "#ffd700",

        r"\b(consideration|amount|rupees|paid|taxes|fees|receipt)\b": "#90ee90",

        r"\b(shall|agrees|hereby|title|possession|transfers)\b": "#add8e6",

        r"\b(property|plot|survey|schedule|boundaries|land)\b": "#ffa07a"
    }

    for pattern, color in highlights.items():

        text = re.sub(
            pattern,
            f'<span style="background-color:{color};padding:3px;border-radius:4px;">\\1</span>',
            text,
            flags=re.IGNORECASE
        )

    return text


# -----------------------------
# UI
# -----------------------------
st.title("⚖️ Legal Document Analyzer")
st.write("---")

tab_photo, tab_text = st.tabs(
    ["📷 Upload Document", "📝 Paste Text"]
)

clean_text = ""

# -----------------------------
# FILE UPLOAD
# -----------------------------
with tab_photo:

    uploaded_file = st.file_uploader(
        "Upload Document",
        type=["jpg", "png", "jpeg", "pdf", "docx"]
    )

    if uploaded_file:

        ext = uploaded_file.name.split(".")[-1].lower()

        # DOCX
        if ext == "docx":

            with st.spinner("Reading Word document..."):

                doc = docx.Document(uploaded_file)

                clean_text = "\n".join(
                    [p.text for p in doc.paragraphs]
                )

                st.success("Word file loaded.")

        # PDF
        elif ext == "pdf":

            with st.spinner("Processing PDF..."):

                images = convert_from_bytes(
                    uploaded_file.read()
                )

                image = images[0]

                st.image(
                    image,
                    caption="PDF Preview",
                    use_container_width=True
                )

                img_array = np.array(image)

                gray = cv2.cvtColor(
                    img_array,
                    cv2.COLOR_RGB2GRAY
                )

                _, thresh = cv2.threshold(
                    gray,
                    0,
                    255,
                    cv2.THRESH_BINARY | cv2.THRESH_OTSU
                )

                clean_text = pytesseract.image_to_string(
                    thresh
                )

        # IMAGE
        else:

            image = Image.open(uploaded_file)

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

            img_array = np.array(image)

            gray = cv2.cvtColor(
                img_array,
                cv2.COLOR_RGB2GRAY
            )

            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                cv2.THRESH_BINARY | cv2.THRESH_OTSU
            )

            clean_text = pytesseract.image_to_string(
                thresh
            )


# -----------------------------
# TEXT INPUT
# -----------------------------
with tab_text:

    user_text = st.text_area(
        "Paste Legal Text",
        height=300
    )

    if user_text:
        clean_text = user_text


# =====================================
# ANALYSIS ENGINE
# =====================================
if clean_text:

    with st.spinner("Analyzing document..."):

        try:

            # -----------------------------
            # TRANSLATION
            # -----------------------------
            max_chars = 4000

            chunks = [
                clean_text[i:i + max_chars]
                for i in range(0, len(clean_text), max_chars)
            ]

            translator = GoogleTranslator(
                source="auto",
                target="en"
            )

            english_text = " ".join(
                [translator.translate(c) for c in chunks]
            )

            # -----------------------------
            # CLASSIFICATION
            # -----------------------------
            labels = [
                "Sale Deed",
                "Lease Deed",
                "Mortgage Deed",
                "Power of Attorney",
                "Affidavit",
                "Will and Testament",
                "Loan Agreement",
                "Employment Agreement",
                "Service Agreement",
                "Non-Disclosure Agreement"
            ]

            res = classifier(
                english_text[:1500],
                labels
            )

            doc_type = res["labels"][0]
            conf = res["scores"][0]

            # -----------------------------
            # SEMANTIC SIMILARITY SCORE
            # -----------------------------
            doc_vec = semantic_model.encode(
                english_text,
                convert_to_tensor=True
            )

            essence = "formal legal contract establishing obligations"

            essence_vec = semantic_model.encode(
                essence,
                convert_to_tensor=True
            )

            semantic_score = util.pytorch_cos_sim(
                doc_vec,
                essence_vec
            ).item()

            lri_score = (0.4 * semantic_score) + (0.6 * conf)

            # -----------------------------
            # DASHBOARD
            # -----------------------------
            col1, col2 = st.columns([1, 2])

            with col1:

                st.metric(
                    "Legal Relevance Index",
                    f"{lri_score:.3f}"
                )

                st.info(f"Document Type: **{doc_type}**")

            # -----------------------------
            # SUMMARY
            # -----------------------------
            with col2:

                st.subheader("Executive Summary")

                summary = summarizer(
                    english_text[:2000],
                    max_length=120,
                    min_length=40,
                    do_sample=False
                )[0]["summary_text"]

                st.markdown(
                    highlight_xai(summary),
                    unsafe_allow_html=True
                )

            # -----------------------------
            # NAMED ENTITIES
            # -----------------------------
            st.write("---")

            st.subheader("🔎 Named Entities")

            raw_entities = ner_model(
                english_text[:1500]
            )

            entities = merge_fragmented_tokens(
                raw_entities
            )

            for ent in entities:

                if len(ent["word"]) > 2:

                    st.write(
                        f"🔹 **{ent['word']}** ({ent['entity_group']})"
                    )

        except Exception as e:

            st.error(f"Analysis failed: {e}")
