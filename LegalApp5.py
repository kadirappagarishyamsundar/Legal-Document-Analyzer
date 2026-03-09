import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
import docx
import io
from deep_translator import GoogleTranslator
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import re

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Legal Document Analyzer", layout="wide", page_icon="⚖️")

# --- ENGINE LOADER (Bypasses Task Registry Errors) ---
@st.cache_resource
def load_all_engines():
    try:
        # 1. Zero-Shot Classifier (Using whitelisted task)
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # 2. SBERT for LRI calculation
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. MANUAL SUMMARIZER: Bypasses the "Unknown task summarization" error
        # We load the model and tokenizer directly instead of using a task string
        sum_model_name = "facebook/bart-large-cnn"
        sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)
            
        # 4. NER Model (Using whitelisted task)
        ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        
        return classifier, semantic_model, sum_model, sum_tokenizer, ner_model
    except Exception as e:
        st.error(f"Engine Initialization Error: {e}")
        return None, None, None, None, None

# INITIALIZE ENGINES AT TOP LEVEL
classifier, semantic_model, sum_model, sum_tokenizer, ner_model = load_all_engines()

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
        r"\b(sale|deed|agreement|vendor|purchaser|mortgage|lease|agent|attorney|successor|principal|donor|donee|settlor|trustee|borrower|lender)\b": "#ffd700", 
        r"\b(consideration|amount|rupees|paid|taxes|fees|receipt|compensation|taxable|liability|claims|bayana|principal amount|interest)\b": "#90ee90", 
        r"\b(shall|agrees|hereby|title|possession|encumbrances|transfers|revoke|authorize|appoint|indemnify|hold harmless|solemnly affirm)\b": "#add8e6", 
        r"\b(property|plot|survey|schedule|boundaries|residential|assets|income|land|mandal|district|village|khata)\b": "#ffa07a" 
    }
    for pattern, color in highlights.items():
        text = re.sub(pattern, f'<span style="background-color: {color}; color: black; padding: 2px; border-radius: 4px;">\\1</span>', text, flags=re.IGNORECASE)
    return text

# --- APP UI ---
st.title("⚖️ Legal Document Analyzer")
st.write("---")

tab_photo, tab_text = st.tabs(["📷 Upload Document", "📝 Paste Text"])
clean_text = ""

with tab_photo:
    uploaded_file = st.file_uploader("Upload Document Scan", type=["jpg", "png", "jpeg", "pdf", "docx"])
    
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'docx':
            with st.spinner("Extracting Text from Word..."):
                doc = docx.Document(uploaded_file)
                clean_text = "\n".join([para.text for para in doc.paragraphs])
                st.success("Word Document loaded.")

        elif file_extension == 'pdf':
            with st.spinner("Converting PDF for Intelligent OCR..."):
                uploaded_file.seek(0)
                # Removed hardcoded poppler path for cloud compatibility
                images = convert_from_bytes(uploaded_file.read())
                image = images[0] 
                st.image(image, caption="PDF Page 1 Preview", use_container_width=True)
                img_array = np.array(image)
                process_ocr = True
        
        else: # Standard Image Handling
            uploaded_file.seek(0)
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image Preview", use_container_width=True)
            img_array = np.array(image)
            process_ocr = True

        if 'process_ocr' in locals() and process_ocr:
            with st.spinner("Executing Intelligent OCR..."):
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                clean_text = pytesseract.image_to_string(thresh)

with tab_text:
    user_text = st.text_area("Paste Legal Content for Audit:", height=300)
    if user_text: 
        clean_text = user_text

# ==========================================
# ANALYSIS ENGINE (Manual Inference)
# ==========================================
if clean_text:
    with st.spinner("Analyzing Legal Context..."):
        try:
            # 1. TRANSLATION (Linguistic Pivot)
            max_chars = 4500
            chunks = [clean_text[i:i + max_chars] for i in range(0, len(clean_text), max_chars)]
            translator = GoogleTranslator(source='auto', target='en')
            english_text = " ".join([translator.translate(c) for c in chunks])
            
            # 2. DOC TYPE & LRI [cite: 70, 85, 159, 171, 311, 382]
            universal_labels = [
                "Sale Deed", "Lease Deed", "Mortgage Deed", "Power of Attorney", "Affidavit", 
                "Indemnity Agreement", "LLP Agreement", "Partnership Deed", "Will and Testament",
                "Loan Agreement", "Employment Agreement", "Service Agreement", "SaaS Agreement", 
                "Non-Disclosure Agreement (NDA)", "Board Resolution","Intellectual Property Assignment",
                "Purchase Order", "Shareholders Agreement", "Gift Deed", "Relinquishment Deed", "Rent Agreement", "Partition Deed", 
                "Adoption Deed", "Trust Deed", "Development Agreement",
                "Probate of Will", "General Power of Attorney (GPA)", "Special Power of Attorney (SPA)",
                "Sale Agreement (Agreement to Sell)", "Partnership Dissolution Deed","Will & Codicil", 
                "Rectification Deed", "Cancellation Deed", "Release Deed", "Hypothecation Agreement"
            ]
            
            res = classifier(english_text[:1500], universal_labels)
            doc_type, conf = res['labels'][0], res['scores'][0]

            essence_map = {
                "Sale Deed": "formal transfer of property for financial consideration and delivery of possession",
                "Lease Deed": "rental agreement for a property specifying term rent and security deposit",
                "Mortgage Deed": "security for the repayment of a loan involving property as collateral",
                "Power of Attorney": "legal delegation of authority to an agent including revocation of prior powers",
                "Indemnity Agreement": "agreement to compensate or hold harmless a party for potential losses or liabilities"
            }

            legal_essence = essence_map.get(doc_type, "formal legal document establishing rights obligations")

            doc_vec = semantic_model.encode(english_text, convert_to_tensor=True)
            essence_vec = semantic_model.encode(legal_essence, convert_to_tensor=True)
            semantic_score = util.pytorch_cos_sim(doc_vec, essence_vec).item()
            lri_score = (0.4 * semantic_score) + (0.6 * conf)

            # --- DISPLAY DASHBOARD ---
            colA, colB = st.columns([1, 2])
            with colA:
                st.metric("Universal LRI Score", f"{lri_score:.4f}")
                st.info(f"**Document Type:** {doc_type}")
            with colB:
                st.subheader("📝 Executive Summary")
                # MANUAL INFERENCE: Bypasses the summarization pipeline task registry error
                inputs = sum_tokenizer([english_text[:2000]], max_length=1024, return_tensors="pt", truncation=True)
                summary_ids = sum_model.generate(inputs["input_ids"], num_beams=4, max_length=150, min_length=50, early_stopping=True)
                raw_summary = sum_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                st.markdown(highlight_xai(raw_summary), unsafe_allow_html=True)

            # 3. DYNAMIC KEY CLAUSES [cite: 172, 329, 392]
            st.write("---")
            st.subheader(f"📌 Key Clauses Identification: {doc_type}")
            
            # Use universal pillars if doc_type is not specialized [cite: 128, 248]
            universal_pillars = {
                "Effective Date": "the specific date when this agreement or letter becomes active",
                "Primary Parties": "the full names and identities of the individuals or entities involved",
                "Core Obligation": "the main task duty or action that must be performed",
                "Governing Jurisdiction": "the legal location or state laws that govern this document"
            }

            # For brevity, this is the logic. You can add your specific 'all_clause_configs' dictionary here
            current_clause_map = universal_pillars 
            sentences = [s.strip() for s in english_text.split('.') if len(s.strip()) > 30]
            
            for label, anchor_phrase in current_clause_map.items():
                with st.expander(f"View {label}"):
                    best_match, max_sim = "", 0
                    a_vec = semantic_model.encode(anchor_phrase, convert_to_tensor=True)
                    for s in sentences:
                        s_vec = semantic_model.encode(s, convert_to_tensor=True)
                        sim = util.pytorch_cos_sim(s_vec, a_vec).item()
                        if sim > max_sim:
                            max_sim, best_match = sim, s
                    
                    if max_sim > 0.52: 
                        st.markdown(highlight_xai(f"**{label}:** {best_match}"), unsafe_allow_html=True)
                    else:
                        st.write(f"No high-confidence {label} detected.")

            # 4. NAMED ENTITIES [cite: 174, 214, 230, 274, 325, 394]
            st.write("---")
            st.subheader("🔍 Named Entities")
            raw_entities = ner_model(english_text[:2000])
            clean_entities = merge_fragmented_tokens(raw_entities)
            
            if clean_entities:
                for ent in clean_entities:
                    if len(ent['word']) > 2:
                        st.write(f"🔹 **{ent['word']}** ({ent['entity_group']})")

        except Exception as e:
            st.error(f"Analysis failed: {e}")
