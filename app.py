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
import os

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Legal Document Analyzer", layout="wide", page_icon="⚖️")

# FIXED: No hardcoded path
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD", "tesseract")

# --- LOAD LIGHTWEIGHT AI MODELS ---
@st.cache_resource
def load_all_engines():
    classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    ner_model = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return classifier, semantic_model, summarizer, ner_model

classifier, semantic_model, summarizer, ner_model = load_all_engines()

# --- HELPERS ---
def merge_fragmented_tokens(entities):
    merged_entities = []
    for ent in entities:
        text = ent['word']
        label = ent['entity_group']
        if text.startswith("##") and merged_entities:
            merged_entities[-1]['word'] += text.replace("##", "")
        else:
            merged_entities.append({'word': text, 'entity_group': label})
    for ent in merged_entities:
        ent['word'] = ent['word'].strip()
    return merged_entities

def highlight_xai(text):
    highlights = {
        r"\b(sale|deed|agreement|vendor|purchaser|mortgage|lease|agent|attorney)\b": "#ffd700",
        r"\b(consideration|amount|rupees|paid|taxes|fees)\b": "#90ee90",
        r"\b(shall|agrees|hereby|title|possession)\b": "#add8e6",
        r"\b(property|plot|survey|schedule|land)\b": "#ffa07a"
    }
    for pattern, color in highlights.items():
        text = re.sub(pattern, f'<span style="background-color: {color}; padding: 2px;">\\1</span>', text, flags=re.IGNORECASE)
    return text

# --- UI ---
st.title("⚖️ Legal Document Analyzer")
st.write("---")

tab_photo, tab_text = st.tabs(["📷 Upload Document", "📝 Paste Text"])
clean_text = ""

# --- FILE UPLOAD ---
with tab_photo:
    uploaded_file = st.file_uploader("Upload Document", type=["jpg", "png", "jpeg", "pdf", "docx"])

    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'docx':
            doc = docx.Document(uploaded_file)
            clean_text = "\n".join([para.text for para in doc.paragraphs])
            st.success("Word Document loaded.")

        elif file_extension == 'pdf':
            images = convert_from_bytes(uploaded_file.read())
            image = images[0]
            st.image(image, caption="PDF Preview", use_container_width=True)
            img_array = np.array(image)
            process_ocr = True

        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image Preview", use_container_width=True)
            img_array = np.array(image)
            process_ocr = True

        if 'process_ocr' in locals():
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            clean_text = pytesseract.image_to_string(thresh)

# --- TEXT INPUT ---
with tab_text:
    user_text = st.text_area("Paste Legal Content:", height=300)
    if user_text:
        clean_text = user_text

# --- ANALYSIS ---
if clean_text:
    try:
        st.subheader("🔍 Processing...")

        # Translation
        translator = GoogleTranslator(source='auto', target='en')
        english_text = translator.translate(clean_text[:4000])

        # Classification
        labels = ["Sale Deed", "Lease Deed", "Loan Agreement", "Employment Agreement"]
        result = classifier(english_text, labels)
        doc_type = result['labels'][0]

        st.success(f"Document Type: {doc_type}")

        # Summary
        summary = summarizer(english_text, max_length=120, min_length=40, do_sample=False)[0]['summary_text']
        st.subheader("📝 Summary")
        st.markdown(highlight_xai(summary), unsafe_allow_html=True)

        # Named Entities
        st.subheader("🔍 Named Entities")
        entities = merge_fragmented_tokens(ner_model(english_text))
        for ent in entities:
            if len(ent['word']) > 2:
                st.write(f"• {ent['word']} ({ent['entity_group']})")

    except Exception as e:
        st.error(f"Error: {e}")
