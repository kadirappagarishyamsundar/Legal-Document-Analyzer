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

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Legal Document Analyzer", layout="wide", page_icon="⚖️")
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@st.cache_resource
def load_all_engines():
    try:
        # Zero-shot classification engine [cite: 170, 212]
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
        # Semantic SBERT model for LRI math [cite: 171, 213]
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # FIXED: Try-except fallback for the summarization task 
        try:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        except:
            # Fallback to text-generation for newer library versions
            summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")
            
        # BERT-Large for Named Entity Recognition [cite: 174, 214]
        ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")
        
        return classifier, semantic_model, summarizer, ner_model
    except Exception as e:
        st.error(f"Engine Initialization Error: {e}")
        return None, None, None, None

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
    uploaded_file = st.file_uploader("Upload Document", type=["jpg", "png", "jpeg", "pdf", "docx"])
    
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'docx':
            with st.spinner("Extracting Text from Word..."):
                doc = docx.Document(uploaded_file)
                clean_text = "\n".join([para.text for para in doc.paragraphs])
                st.success("Word Document loaded.")

        elif file_extension == 'pdf':
            with st.spinner("Converting PDF for Intelligent OCR..."):
                # Path to poppler must be correct
                poppler_path = r'C:\training data\poppler-25.12.0\Library\bin'
                uploaded_file.seek(0)
                images = convert_from_bytes(uploaded_file.read(), poppler_path=poppler_path)
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
            del process_ocr # Cleanup

with tab_text:
    user_text = st.text_area("Paste Legal Content:", height=300)
    if user_text: 
        clean_text = user_text

# ==========================================
# ANALYSIS ENGINE
# ==========================================
if clean_text:
    with st.spinner("Analyzing Legal Context..."):
        try:
            # 1. TRANSLATION (Linguistic Pivot)
            max_chars = 4500
            chunks = [clean_text[i:i + max_chars] for i in range(0, len(clean_text), max_chars)]
            translator = GoogleTranslator(source='auto', target='en')
            english_text = " ".join([translator.translate(c) for c in chunks])
            
            # 2. DOC TYPE & LRI
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
                "Power of Attorney": "legal delegation of authority to an agent including revocation of prior powers, successor appointment, and indemnification of third parties",
                "Affidavit": "a written statement of facts confirmed by oath or affirmation under legal authority",
                "Indemnity Agreement": "agreement to compensate or hold harmless a party for potential losses or liabilities",
                "LLP Agreement": "governance document for limited liability partnership defining partner roles and capital",
                "Partnership Deed": "agreement between partners defining business terms capital and profit sharing",
                "Will and Testament": "legal declaration of a person's wishes regarding their property after death",
                "Loan Agreement": "contractual terms for a sum of money borrowed and repayment obligations",
                "Employment Agreement": "terms of service including roles compensation and termination for an employee",
                "Service Agreement": "contract defining the scope of work and payment for professional services",
                "SaaS Agreement": "software service delivery terms including uptime, data processing, and subscriptions",
                "Non-Disclosure Agreement (NDA)": "protection of confidential information, trade secrets, and non-compete duties",
                "Board Resolution": "formal corporate decision-making record with authorization and voting results",
                "Intellectual Property Assignment": "transfer of ownership for patents, trademarks, or copyrights to an employer",
                "Purchase Order": "commercial document for goods and services including prices and delivery terms",
                "Shareholders Agreement": "governance of company ownership including drag-along, tag-along, and ROFR rights",
                "Gift Deed": "voluntary transfer of property out of love and affection without any monetary consideration",
                "Relinquishment Deed": "legal renouncing of rights and interest in an ancestral property by a legal heir",
                "Rent Agreement": "short term rental contract typically for eleven months including security deposit and maintenance",
                "Partition Deed": "legal division of co-owned property into separate shares among various owners",
                "Adoption Deed": "formal act of giving and taking a child into a new family under personal laws",
                "Trust Deed": "creation of a legal entity for charitable or religious purposes with dedicated property",
                "Development Agreement": "contract between landowner and builder for construction and sharing of built-up area",
                "Will & Codicil": "testamentary document or supplementary addition defining distribution of assets after death",
                "Rectification Deed": "supplementary document executed to correct clerical or factual errors in a previously registered deed",
                "Cancellation Deed": "legal document executed to revoke and nullify a previously registered agreement or contract",
                "Release Deed": "formal instrument used to free a person from an obligation or to surrender a legal claim",
                "Hypothecation Agreement": "security for a loan where movable assets remain in possession of the borrower but are charged to the lender",
                "Probate of Will": "judicial process by which a will is proved in a court of law and accepted as a valid public document",
                "General Power of Attorney (GPA)": "broad legal authority granted to an agent to handle multiple affairs on behalf of the principal",
                "Special Power of Attorney (SPA)": "restricted legal authority granted for a specific single transaction or act",
                "Sale Agreement (Agreement to Sell)": "preliminary contract outlining terms of a future property sale and advance payment",
                "Partnership Dissolution Deed": "legal document recording the formal closure of a partnership firm and settlement of accounts"
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
                raw_summary = summarizer(english_text[:2000], max_length=150, min_length=50, do_sample=False)[0]['summary_text']
                st.markdown(highlight_xai(raw_summary), unsafe_allow_html=True)

            # 3. DYNAMIC KEY CLAUSES
            st.write("---")
            st.subheader(f"📌 Key Clauses Identification: {doc_type}")
            
            all_clause_configs = {
                "Sale Deed": {
                    "Sale Consideration": "total sale consideration paid by bank transfer receipt",
                    "Title & Encumbrance": "property is free from all encumbrances charges mortgages and liens",
                    "Possession Clause": "delivered vacant and physical possession for peaceful enjoyment",
                    "Statutory Duties": "pay all taxes charges duties and registration formalities"
                },
                "Lease Deed": {
                    "Rent & Deposit": "monthly rent amount and security deposit paid at execution",
                    "Lease Term": "the duration of the lease period and renewal conditions",
                    "Maintenance": "responsibility for repairs maintenance and utility payments"
                },
                "Power of Attorney": {
                    "Revocation & Successor": "revoke all previous powers and designate a successor agent",
                    "Powers & Authority": "full power and authority to act manage conduct all affairs",
                    "Indemnification": "indemnify and hold harmless any third party who accepts this document"
                },
                "Affidavit": {
                    "Statement of Facts": "solemnly affirm and state the following facts as true and correct",
                    "Verification": "verified at the place and date that contents are true to my knowledge",
                    "Deponent Identity": "name age and residence of the person making the sworn statement"
                },
                "Will and Testament": {
                    "Executor Appointment": "appoint a specific person to manage and distribute the estate",
                    "Bequest Details": "give devise and bequeath specific property or assets to beneficiaries",
                    "Revocation Clause": "revoke all former wills and codicils previously made by me"
                },
                "Loan Agreement": {
                    "Principal Amount": "total sum of money borrowed and disbursed to the borrower",
                    "Repayment Schedule": "monthly installment amount and tenure of the loan",
                    "Interest Rate": "annual percentage rate of interest charged on the principal"
                },
                "Employment Agreement": {
                    "Job Role": "designation responsibilities and specific duties of the employee",
                    "Remuneration": "annual CTC salary structure and monthly compensation details",
                    "Notice Period": "period of notice required for termination by either party"
                },
                "Service Agreement": {
                    "Scope of Work": "detailed description of services to be provided by the consultant",
                    "Payment Terms": "invoicing schedule and fees for services rendered",
                    "Confidentiality": "protection of proprietary information and non-disclosure obligations"
                },
                "Indemnity Agreement": {
                    "Primary Parties": "identities of the Indemnifier and Indemnitee entering the agreement",
                    "Indemnity Scope": "losses damages liabilities and claims to be compensated",
                    "Limitation": "total liability amount not to exceed specified limit"
                },
                "LLP Agreement": {
                    "Capital Contribution": "monetary or asset contribution made by each designated partner",
                    "Profit Sharing": "ratio for distribution of profits and losses among members",
                    "Management Powers": "decision making authority and voting rights of the partners"
                },
                "Partnership Deed": {
                    "Firm Name": "the name and style under which the partnership business is conducted",
                    "Capital Investment": "initial capital brought in by each partner for the business",
                    "Dissolution": "terms and conditions for closing the partnership or partner exit"
                },
                "SaaS Agreement": {
                    "Service Level Agreement (SLA)": "guaranteed uptime performance metrics and service credits",
                    "Data Processing": "handling of personal data compliance with GDPR and security standards",
                    "Subscription Terms": "recurring fees billing cycles and renewal conditions"
                },
                "Non-Disclosure Agreement (NDA)": {
                    "Confidential Information": "definition of protected proprietary data and trade secrets",
                    "Non-Compete": "restriction on working for competitors for a specific period",
                    "Duration": "how long the confidentiality duty lasts after termination"
                },
                "Board Resolution": {
                    "Authorization": "formal power granted to an officer to sign documents or open accounts",
                    "Voting Results": "record of directors in favor or against the proposed action"
                },
                "Intellectual Property Assignment": {
                    "Assignment of Rights": "total transfer of patents trademarks or copyrights from creator",
                    "Work for Hire": "confirmation that work was created during employment for employer"
                },
                "Purchase Order": {
                    "Itemized List": "description quantity and unit price of goods or services ordered",
                    "Payment Terms": "credit period and terms for invoice payment such as net 30"
                },
                "Shareholders Agreement": {
                    "Exit Rights": "drag-along and tag-along rights during a company sale",
                    "ROFR": "right of first refusal for existing partners to buy shares before an outside sale"
                },
                "Mortgage Deed": {
                    "Loan Principal": "the specific principal sum of money borrowed and secured by the property",
                    "Interest Rate": "the annual percentage rate of interest charged on the loan balance",
                    "Security/Collateral": "the legal description of the property granted as security for the debt",
                    "Repayment Terms": "the conditions and schedule for the repayment of the principal and interest"
                },
                "Gift Deed": {
                    "Donor & Donee": "the person giving the property out of natural love and affection and the receiver",
                    "Acceptance Clause": "confirmation that the gift was accepted during the lifetime of the donor",
                    "Property Value": "the market value of the property for the purpose of stamp duty calculation",
                    "No Consideration": "declaration that the transfer is made without any monetary exchange"
                },
                "Relinquishment Deed": {
                "Legal Heirs": "identities of the legal heirs giving up their share in an ancestral property",
                "Releasors": "the specific parties who are renouncing their rights title and interest",
                "Releasee": "the party in whose favor the rights are being surrendered",
                "Ancestral Description": "details of the deceased person's property being redistributed"
                },
                "General Power of Attorney (GPA)": {
                "Authorized Acts": "specific powers to sell mortgage lease or manage property in India",
                "Sub-Delegation": "power granted to the agent to appoint another agent if necessary",
                "Stamp Duty Compliance": "mention of the registration and stamp duty paid to the sub-registrar",
                "Identification of Agent": "Aadhar or PAN details of the appointed attorney-in-fact"
                },
                "Special Power of Attorney (SPA)": {
                "Limited Purpose": "the specific single task or transaction for which power is granted",
                "Automatic Termination": "clause stating power ends once the specific task is completed",
                "No General Authority": "express restriction that the agent cannot act outside the single task"
                },
                "Rent Agreement": {
                "Lock-in Period": "minimum duration during which neither party can terminate the agreement",
                "Maintenance Charges": "responsibility for society maintenance water and electricity bills",
                "Security Deposit": "interest-free refundable deposit paid to the owner at commencement",
                "Notice Period": "the time required for either party to vacate the premises"
                },
                "Sale Agreement (Agreement to Sell)": {
                "Advance Payment": "the token money or bayana paid at the time of signing",
                "Timeline for Completion": "the deadline for the final sale deed and balance payment",
                "Default Penalty": "consequences if the buyer or seller backs out of the transaction",
                "Vendor Liabilities": "clearance of all existing loans and dues before final transfer"
                },
                "Partnership Dissolution Deed": {
                "Settlement of Accounts": "final calculation and distribution of assets and liabilities",
                "Cessation of Business": "the specific date from which the firm ceases to exist",
                "Public Notice": "agreement to publish the dissolution in the official gazette",
                "Indemnity for Liability": "partner's responsibility for past debts of the firm"
                },
                "Probate of Will": {
                "Executor Appointment": "the person named to carry out the instructions of the deceased",
                "Beneficiary List": "specific people or entities receiving assets or property",
                "Signature & Witnesses": "legal verification by two witnesses who saw the testator sign",
                "Last Will Clause": "declaration that this document revokes all previous testaments"
                },
                "Trust Deed": {
                "Settlor & Trustees": "the creator of the trust and the persons managing the assets",
                "Objects of the Trust": "the specific charitable or religious purposes of the trust",
                "Trust Property": "details of the initial corpus or assets dedicated to the trust",
                "Dissolution of Trust": "what happens to the assets if the trust is wound up"
               },
               "Adoption Deed": {
               "Giving and Taking": "the formal ceremony or act of transferring the child",
               "Biological Parents": "the consent and identity of the natural parents",
               "Adoptive Parents": "the eligibility and identity of the parents receiving the child",
               "Hindu Adoptions Act": "compliance with the Hindu Adoptions and Maintenance Act 1956"
              },
              "Partition Deed": {
              "Schedule of Shares": "the specific portions of property divided among co-owners",
              "Mutual Consent": "declaration that the division is voluntary and agreed upon",
              "Boundary Demarcation": "the physical boundaries of each newly created plot",
              "Easement Rights": "rights to common areas like staircases or pathways"
             },
             "Development Agreement": {
             "FSI/FAR Details": "the floor space index or built-up area allowed for construction",
             "Owner's Share": "the specific percentage of flat or money the landowner receives",
             "Completion Timeline": "deadline for the builder to hand over the finished project",
             "Force Majeure": "unforeseeable circumstances that delay construction progress"
             },
             "Rectification Deed": {
             "Original Deed Details": "reference to the previous registered document being corrected",
             "Nature of Error": "description of the clerical or factual mistake being rectified",
             "Corrected Version": "the new and accurate wording intended to replace the error"
             },
             "Cancellation Deed": {
             "Revocation Clause": "explicit statement that the earlier agreement is now null and void",
             "Reason for Cancellation": "circumstances or mutual consent leading to the termination",
             "No Further Claims": "declaration that neither party has future rights under the cancelled deed"
             },
             "Release Deed": {
             "Consideration for Release": "any amount paid to the party surrendering their claim",
             "Description of Claim": "the specific legal right or obligation being released"
             },
             "Hypothecation Agreement": {
             "Asset Description": "details of the movable property or vehicle used as security",
             "Lender Rights": "power of the bank to seize the asset in case of loan default",
             "Insurance Clause": "requirement for the borrower to keep the asset insured"
             },
              "Will & Codicil": {
              "Supplementary Changes": "specific amendments or additions made to the original will",
              "Reference to Original Will": "details and date of the primary testament being modified",
              "Confirmation of Terms": "statement that all other portions of the original will remain in effect",
              "Witness Attestation": "signatures of witnesses confirming the supplemental changes"
             }
            }

            universal_pillars = {
                "Effective Date": "the specific date when this agreement or letter becomes active",
                "Primary Parties": "the full names and identities of the individuals or entities involved",
                "Core Obligation": "the main task duty or action that must be performed",
                "Governing Jurisdiction": "the legal location or state laws that govern this document"
            }

            current_clause_map = all_clause_configs.get(doc_type, universal_pillars)
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

            # 4. NAMED ENTITIES (Textual Fidelity)
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




